import sys
sys.path.append(".")
from features.feature_engineering import build_features

import numpy as np
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix, precision_recall_curve
import joblib, os

def train():
    X_train, X_test, y_train, y_test, FEATURES = build_features()

    print("Applying SMOTE to balance classes...")
    smote = SMOTE(sampling_strategy=0.3, random_state=42)
    X_res, y_res = smote.fit_resample(X_train, y_train)
    print(f"After SMOTE — total: {len(X_res)}, abuse: {y_res.sum()}")

    model = XGBClassifier(
        n_estimators=400,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=10,
        eval_metric="auc",
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_res, y_res, eval_set=[(X_test, y_test)], verbose=100)

    y_pred_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_pred_proba >= 0.5).astype(int)

    print("\n=== Classification Report ===")
    print(classification_report(y_test, y_pred, target_names=["Legit", "Abuse"]))
    print(f"AUC-ROC: {roc_auc_score(y_test, y_pred_proba):.4f}")

    precision, recall, thresholds = precision_recall_curve(y_test, y_pred_proba)
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
    best_thresh = thresholds[np.argmax(f1_scores)]
    print(f"Best threshold by F1: {best_thresh:.3f}")

    os.makedirs("models", exist_ok=True)
    joblib.dump(model, "models/xgboost_model.pkl")
    joblib.dump(best_thresh, "models/threshold.pkl")
    print("Model saved to models/xgboost_model.pkl")

if __name__ == "__main__":
    train()