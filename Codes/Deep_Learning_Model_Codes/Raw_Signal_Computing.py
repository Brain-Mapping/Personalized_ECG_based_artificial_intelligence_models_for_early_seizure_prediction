import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

input_dir = "/content/drive/MyDrive/PT_Data)"
output_dir = "/content/drive/MyDrive/Processed_RawSignals_PT"
fs = 250
img_size = (224, 224)
dpi = 120

os.makedirs(os.path.join(output_dir, "preictal"), exist_ok=True)
os.makedirs(os.path.join(output_dir, "interictal"), exist_ok=True)

def save_signal_plot(sig, label, file_prefix, row_idx):
    t = np.arange(len(sig)) / fs

    plt.figure(figsize=(4, 2))
    plt.plot(t, sig, color="black", linewidth=1)
    plt.axis("off")
    plt.tight_layout()

    out_path = os.path.join(output_dir, label, f"{file_prefix}_row{row_idx}.png")
    plt.savefig(out_path, dpi=dpi, bbox_inches="tight", pad_inches=0)
    plt.close()

    img = Image.open(out_path).convert("RGB")
    img = img.resize(img_size, resample=Image.BICUBIC)
    img.save(out_path)

    sig_dir = os.path.join(output_dir, "raw_signals", label)
    os.makedirs(sig_dir, exist_ok=True)
    np.save(os.path.join(sig_dir, f"{file_prefix}_row{row_idx}.npy"), sig)

def process_xlsx_file(filepath):
    fname = os.path.basename(filepath).lower()
    if "preictal" in fname:
        label = "preictal"
    elif "interictal" in fname:
        label = "interictal"
    else:
        print(f"Skipping {fname}, label not found in name")
        return

    df = pd.read_excel(filepath, header=None)

    for row_idx, row in df.iterrows():
        sig = np.nan_to_num(row.values.astype(float))
        if np.all(sig == 0) or np.all(np.isnan(sig)):
            continue
        save_signal_plot(sig, label, os.path.splitext(fname)[0], row_idx)

def process_all(input_dir):
    for file in os.listdir(input_dir):
        if file.endswith(".xlsx"):
            process_xlsx_file(os.path.join(input_dir, file))
            print(f"Processed {file}")

process_all(input_dir)
print("Done! Raw ECG plots saved in:", output_dir)