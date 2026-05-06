import os, random, time, copy
import numpy as np
from collections import defaultdict
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import StratifiedKFold
from PIL import Image

import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms, models

scalo_dir = "/content/drive/MyDrive/AACE/Morse_Scalo_Processed_RawSignals"
signal_dir = "/content/drive/MyDrive/AACE/Processed_RawSignals_new"
gdrive_dir = "/content/drive/MyDrive/Bimodal_Models_final"

batch_size = 8
num_epochs = 40
lr = 5e-5
patience = 8
n_splits = 5
seed = 42
mixup_alpha = 0.2
label_smoothing = 0.1
tta_enabled = True
use_amp = True
use_swa = False
backbone_name = "resnet50"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class_names = ["interictal", "preictal"]
num_classes = len(class_names)
os.makedirs(gdrive_dir, exist_ok=True)

random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

train_transform = transforms.Compose([
    transforms.Resize((320,320)),
    transforms.RandomResizedCrop(308, scale=(0.86,1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(0.12,0.12,0.03),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

test_transform = transforms.Compose([
    transforms.Resize((320,320)),
    transforms.CenterCrop(308),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

class BimodalECGDataset(Dataset):
    def __init__(self, scalo_dir, signal_dir, classes, transform=None):
        self.samples = []
        self.transform = transform
        self.class_to_idx = {c:i for i,c in enumerate(classes)}
        for c in classes:
            sc_dir = os.path.join(scalo_dir, c)
            si_dir = os.path.join(signal_dir, c)
            if not os.path.isdir(sc_dir) or not os.path.isdir(si_dir): continue
            sc_files = sorted(os.listdir(sc_dir))
            si_files = sorted(os.listdir(si_dir))
            map_si = {f.lower(): f for f in si_files}
            for f in sc_files:
                k = f.lower()
                if k in map_si:
                    self.samples.append((os.path.join(sc_dir,f), os.path.join(si_dir,map_si[k]), self.class_to_idx[c]))
    def __len__(self): return len(self.samples)
    def __getitem__(self, idx):
        sc, si, lbl = self.samples[idx]
        sc_img = Image.open(sc).convert("RGB")
        si_img = Image.open(si).convert("RGB")
        if self.transform:
            sc_img = self.transform(sc_img)
            si_img = self.transform(si_img)
        return sc_img, si_img, lbl

class SEBlock(nn.Module):
    def __init__(self, dim, red=16):
        super().__init__()
        self.net = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(dim, dim//red, bias=False), nn.ReLU(inplace=True),
            nn.Linear(dim//red, dim, bias=False), nn.Sigmoid()
        )
    def forward(self, x):
        b,c,_,_ = x.size()
        w = self.net(x).view(b,c,1,1)
        return x * w

class CrossAttentionFusion(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.out = nn.Sequential(nn.Linear(dim, dim), nn.ReLU(inplace=True))
    def forward(self, f1, f2):
        q1,k2,v2 = self.q(f1), self.k(f2), self.v(f2)
        a2 = v2 * torch.sigmoid((q1 * k2).sum(dim=1, keepdim=True))
        q2,k1,v1 = self.q(f2), self.k(f1), self.v(f1)
        a1 = v1 * torch.sigmoid((q2 * k1).sum(dim=1, keepdim=True))
        fused = torch.cat([f1 + self.out(a2), f2 + self.out(a1)], dim=1)
        return fused

class BimodalModel(nn.Module):
    def __init__(self, num_classes=2, unfreeze_layer3=True):
        super().__init__()
        self.scalo = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        self.signal = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        for p in self.scalo.parameters(): p.requires_grad = False
        for p in self.signal.parameters(): p.requires_grad = False
        for p in self.scalo.layer4.parameters(): p.requires_grad = True
        for p in self.signal.layer4.parameters(): p.requires_grad = True
        if unfreeze_layer3:
            for p in self.scalo.layer3.parameters(): p.requires_grad = True
            for p in self.signal.layer3.parameters(): p.requires_grad = True
        def enable_bn(m):
            for mm in m.modules():
                if isinstance(mm, nn.BatchNorm2d):
                    for p in mm.parameters(): p.requires_grad = True
                    mm.train()
        enable_bn(self.scalo); enable_bn(self.signal)
        self.scalo.fc = nn.Identity(); self.signal.fc = nn.Identity()
        self.se1 = SEBlock(2048); self.se2 = SEBlock(2048)
        self.cross = CrossAttentionFusion(2048)
        self.gate = nn.Sequential(nn.Linear(2048*2,512), nn.ReLU(inplace=True), nn.Linear(512,2), nn.Softmax(dim=1))
        self.classifier = nn.Sequential(nn.Linear(2048*2,512), nn.ReLU(), nn.Dropout(0.5), nn.Linear(512,num_classes))
    def forward(self, x1, x2):
        def feat_backbone(backbone, x):
            x = backbone.conv1(x); x = backbone.bn1(x); x = backbone.relu(x); x = backbone.maxpool(x)
            x = backbone.layer1(x); x = backbone.layer2(x); x = backbone.layer3(x); x = backbone.layer4(x)
            return x
        f1 = feat_backbone(self.scalo, x1); f2 = feat_backbone(self.signal, x2)
        f1 = self.se1(f1); f2 = self.se2(f2)
        v1 = torch.flatten(nn.functional.adaptive_avg_pool2d(f1, (1,1)),1)
        v2 = torch.flatten(nn.functional.adaptive_avg_pool2d(f2, (1,1)),1)
        gates = self.gate(torch.cat([v1,v2], dim=1))
        v1 = v1 * gates[:,0].unsqueeze(1); v2 = v2 * gates[:,1].unsqueeze(1)
        fused = self.cross(v1, v2)
        return self.classifier(fused)

def mixup(x1,x2,y,alpha=0.2):
    if alpha<=0: return x1,x2,y,y,1.0
    lam = np.random.beta(alpha,alpha); lam = max(lam, 1-lam)
    idx = torch.randperm(x1.size(0)).to(x1.device)
    mixed1 = lam*x1 + (1-lam)*x1[idx]
    mixed2 = lam*x2 + (1-lam)*x2[idx]
    return mixed1, mixed2, y, y[idx], lam

def mixup_loss(crit, pred, ya, yb, lam):
    return lam*crit(pred, ya) + (1-lam)*crit(pred, yb)

def class_weights_from_labels(labels):
    u,c = np.unique(labels, return_counts=True)
    freq = c.astype(float)/c.sum(); inv = 1.0/(freq+1e-12)
    norm = inv/np.sum(inv)*len(u)
    w = np.zeros(len(u), dtype=np.float32)
    for i,uu in enumerate(u): w[int(uu)] = norm[i]
    return torch.tensor(w, device=device)

def tta_predict(model, x1, x2):
    with torch.no_grad():
        p1 = nn.functional.softmax(model(x1,x2), dim=1)
        if not tta_enabled: return p1
        p2 = nn.functional.softmax(model(torch.flip(x1,[-1]), torch.flip(x2,[-1])), dim=1)
        return (p1 + p2) / 2.0

dataset = BimodalECGDataset(scalo_dir, signal_dir, class_names, transform=None)
labels = [s[2] for s in dataset.samples]
if len(labels)==0: raise RuntimeError("No samples found")
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
cw = class_weights_from_labels(labels)
criterion = nn.CrossEntropyLoss(weight=cw, label_smoothing=label_smoothing)

fold_results=[]
saved_paths=[]
oof_probs_store = []

for fold, (train_idx, test_idx) in enumerate(skf.split(np.zeros(len(labels)), labels), 1):
    train_ds = Subset(dataset, train_idx); test_ds = Subset(dataset, test_idx)
    train_ds.dataset.transform = train_transform; test_ds.dataset.transform = test_transform
    nw = 4 if torch.cuda.is_available() else 0
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=nw, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=nw, pin_memory=True)

    model = BimodalModel(num_classes=num_classes, unfreeze_layer3=True).to(device)
    opt = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=5, T_mult=2)
    if use_swa:
        from torch.optim.swa_utils import AveragedModel, SWALR
        swa_model = AveragedModel(model); swa_start = int(num_epochs*0.75)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    best_acc=0.0; best_w=None; counter=0
    for epoch in range(1, num_epochs+1):
        model.train()
        run_loss=0.0; c=0; tot=0
        t0=time.time()
        for x1,x2,y in train_loader:
            x1,x2,y = x1.to(device), x2.to(device), y.to(device)
            opt.zero_grad()
            mx1,mx2,ya,yb,lam = mixup(x1,x2,y,alpha=mixup_alpha)
            with torch.amp.autocast(device_type='cuda', enabled=use_amp):
                out = model(mx1, mx2)
                loss = mixup_loss(criterion, out, ya, yb, lam)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            scaler.step(opt); scaler.update()
            run_loss += loss.item() * y.size(0)
            _,preds = out.max(1)
            c += (preds==ya).sum().item(); tot += y.size(0)
        train_acc = c/(tot+1e-12)
        scheduler.step(epoch)
        model.eval()
        correct=0; total=0; probs_list=[]; labs_list=[]
        with torch.no_grad():
            for x1,x2,y in test_loader:
                x1,x2,y = x1.to(device), x2.to(device), y.to(device)
                probs = tta_predict(model, x1, x2)
                preds = probs.argmax(1)
                correct += (preds==y).sum().item(); total += y.size(0)
                probs_list.append(probs.cpu().numpy()); labs_list.append(y.cpu().numpy())
        val_acc = correct/total
        if val_acc>best_acc:
            best_acc=val_acc; best_w=copy.deepcopy(model.state_dict()); counter=0
        else:
            counter+=1
            if counter>=patience: break
    print(f"Fold {fold} best val acc: {best_acc:.4f}")
    model.load_state_dict(best_w); model.eval()
    saved_path = os.path.join(gdrive_dir, f"bimodal_fold{fold}_acc{best_acc:.4f}.pth")
    torch.save(best_w, saved_path); saved_paths.append(saved_path)
    fold_probs=[]; fold_labels=[]
    with torch.no_grad():
        for x1,x2,y in test_loader:
            x1,x2,y = x1.to(device), x2.to(device), y.to(device)
            p = tta_predict(model, x1, x2)
            fold_probs.append(p.cpu().numpy()); fold_labels.append(y.cpu().numpy())
    fold_probs = np.vstack(fold_probs); fold_labels = np.concatenate(fold_labels)
    oof_probs_store.append((test_idx, fold_probs, fold_labels))
    fold_results.append(best_acc)
print("\nPer-fold accuracies:")
for i,a in enumerate(fold_results,1): print(f"Fold {i}: {a:.4f}")
print(f"Mean: {np.mean(fold_results):.4f} ± {np.std(fold_results):.4f}")

sample_probs = {}
sample_labels = {}
for test_idx, probs, labs in oof_probs_store:
    for i_local, sidx in enumerate(test_idx):
        sample_probs.setdefault(sidx, []).append(probs[i_local])
        sample_labels[sidx] = int(labs[i_local])
y_true=[]; y_pred=[]
for s in sorted(sample_probs.keys()):
    avg = np.mean(np.vstack(sample_probs[s]), axis=0)
    y_pred.append(int(avg.argmax()))
    y_true.append(int(sample_labels[s]))
ens_acc = accuracy_score(y_true, y_pred)
print(f"Ensembled OOF Accuracy: {ens_acc:.4f}")
print(classification_report(y_true, y_pred, target_names=class_names))
print(confusion_matrix(y_true, y_pred))

with open(os.path.join(gdrive_dir, "saved_models_list.txt"), "w") as f:
    for p in saved_paths: f.write(p+"\n")
print("Models saved to:", gdrive_dir)
