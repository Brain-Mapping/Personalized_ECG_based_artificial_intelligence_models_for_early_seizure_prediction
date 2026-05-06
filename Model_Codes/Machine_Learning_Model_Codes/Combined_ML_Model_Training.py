import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import (
    RandomForestClassifier, BaggingClassifier, ExtraTreesClassifier, AdaBoostClassifier
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score, roc_curve, accuracy_score
)


file_path = r"Path"
df = pd.read_excel(file_path)

print("Shape of dataset:", df.shape)
print("Missing values:\n", df.isnull().sum())

df.dropna(subset=['Label'], inplace=True)

X = df.drop(columns=['Label'])
y = df['Label']


models = {
    'Logistic Regression': Pipeline([
        ('scaler', StandardScaler()),
        ('clf', LogisticRegression(
            max_iter=2000, solver='liblinear',
            C=1.0, penalty='l2',
            class_weight='balanced', random_state=42
        ))
    ]),

    'SVM': Pipeline([
        ('scaler', StandardScaler()),
        ('clf', SVC(
            probability=True, kernel='rbf', C=0.5,
            gamma='scale', class_weight='balanced',
            random_state=42
        ))
    ]),

    'Random Forest': RandomForestClassifier(
        n_estimators=500, max_depth=12,
        min_samples_split=10, min_samples_leaf=4,
        max_features='sqrt',
        class_weight='balanced_subsample',
        random_state=42, n_jobs=-1
    ),

    'Decision Tree': DecisionTreeClassifier(
        max_depth=8, min_samples_split=10, min_samples_leaf=5,
        class_weight='balanced', random_state=42
    ),

    'Extra Trees': ExtraTreesClassifier(
        n_estimators=400, max_depth=10,
        min_samples_split=10, min_samples_leaf=4,
        max_features='sqrt',
        class_weight='balanced', random_state=42, n_jobs=-1
    ),

    'AdaBoost': AdaBoostClassifier(
        n_estimators=300, learning_rate=0.5, random_state=42
    ),

    'KNN (uniform)': Pipeline([
        ('scaler', StandardScaler()),
        ('clf', KNeighborsClassifier(
            n_neighbors=5, weights="uniform"
        ))
    ]),

    'XGBoost': XGBClassifier(
        n_estimators=800, max_depth=10, learning_rate=0.01,
        subsample=0.9, colsample_bytree=0.8,
        gamma=3, reg_alpha=0.6, reg_lambda=0.8,
        min_child_weight=1,
        use_label_encoder=False, eval_metric='logloss',
        random_state=42
    )
}


roc_output_file = "roc_values_per_model.csv"   # <-- change this name as needed

# ==============================
# STRATIFIED K-FOLD CROSS VALIDATION + MEAN ROC 
# ==============================
print("\n====================== K-FOLD CROSS-VALIDATION RESULTS ======================\n")

kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_results = []
roc_results = {}   # raw fold-wise values

plt.figure(figsize=(8, 6))

mean_fpr = np.linspace(0, 1, 100)  # common FPR grid for interpolation

for name, model in models.items():
    print(f"\n{name}:")
    train_accuracies, test_accuracies = [], []
    fold_rocs = []
    interp_tprs, aucs = [], []

    for fold, (train_index, test_index) in enumerate(kf.split(X, y), 1):
        X_train_fold, X_test_fold = X.iloc[train_index], X.iloc[test_index]
        y_train_fold, y_test_fold = y.iloc[train_index], y.iloc[test_index]

        model.fit(X_train_fold, y_train_fold)
        train_pred = model.predict(X_train_fold)
        test_pred = model.predict(X_test_fold)

        train_accuracies.append(accuracy_score(y_train_fold, train_pred))
        test_accuracies.append(accuracy_score(y_test_fold, test_pred))

        # --- ROC computation ---
        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_test_fold)[:, 1]
        elif hasattr(model, "decision_function"):
            scores = model.decision_function(X_test_fold).reshape(-1, 1)
            y_prob = MinMaxScaler().fit_transform(scores).ravel()
        else:
            y_prob = test_pred

        try:
            auc = roc_auc_score(y_test_fold, y_prob)
            fpr, tpr, thresholds = roc_curve(y_test_fold, y_prob)

            # Interpolate to common grid
            interp_tpr = np.interp(mean_fpr, fpr, tpr)
            interp_tpr[0] = 0.0
            interp_tprs.append(interp_tpr)
            aucs.append(auc)

            fold_rocs.append({
                "fold": fold, "fpr": fpr, "tpr": tpr,
                "thresholds": thresholds, "auc": auc
            })
        except Exception as e:
            print("ROC failed:", e)

    # Store raw fold values
    roc_results[name] = fold_rocs

    # Mean ROC across folds
    if interp_tprs:
        mean_tpr = np.mean(interp_tprs, axis=0)
        mean_tpr[-1] = 1.0
        mean_auc = np.mean(aucs)
        std_auc = np.std(aucs)

        plt.plot(
            mean_fpr, mean_tpr,
            label=f'{name} (AUC = {mean_auc:.2f} ± {std_auc:.2f})'
        )

    # Accuracy summary
    mean_train, mean_test, std_test = np.mean(train_accuracies), np.mean(test_accuracies), np.std(test_accuracies)
    print("Train Accuracies:", np.round(train_accuracies, 4))
    print("Mean Train Accuracy:", mean_train.round(4))
    print("Test Accuracies :", np.round(test_accuracies, 4))
    print("Mean Test Accuracy :", mean_test.round(4))
    print("Std Dev (Test)     :", std_test.round(4))

    cv_results.append({
        "Model": name,
        "Mean Train Accuracy": mean_train,
        "Mean Test Accuracy": mean_test,
        "Std Test Accuracy": std_test
    })

# Final ROC summary plot
plt.plot([0, 1], [0, 1], 'k--', label='Random Guess')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Mean ROC Curves (5-Fold CV)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Save ROC values to CSV with chosen file name
all_roc_rows = []
for model_name, folds in roc_results.items():
    for entry in folds:
        for i in range(len(entry["fpr"])):
            all_roc_rows.append({
                "Model": model_name,
                "Fold": entry["fold"],
                "FPR": entry["fpr"][i],
                "TPR": entry["tpr"][i],
                "Threshold": entry["thresholds"][i],
                "AUC": entry["auc"]
            })

roc_df = pd.DataFrame(all_roc_rows)
roc_df.to_csv(roc_output_file, index=False)
print(f"\nROC values saved to: {roc_output_file}")

print("\n====================== SUMMARY (sorted by mean test accuracy) ======================\n")
summary_df = pd.DataFrame(cv_results).sort_values(by="Mean Test Accuracy", ascending=False)
print(summary_df.to_string(index=False))