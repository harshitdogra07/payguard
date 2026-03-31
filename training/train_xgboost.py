import sys
sys.path.append(".")
from features.feature_engineering import build_features

import numpy as np
from xgboost import XGBClassifier
import shap
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, roc_auc_score, average_precision_score, precision_recall_curve, precision_score, recall_score
import joblib, os

def train():
    X_train, X_test, y_train, y_test, FEATURES = build_features()

    print("\nCalculating scale_pos_weight...")
    scale_pos_weight = len(y_train[y_train==0]) / len(y_train[y_train==1])
    print(f"Scale Pos Weight: {scale_pos_weight:.2f}")

    model = XGBClassifier(
        n_estimators=600,
        max_depth=7,
        learning_rate=0.03,
        subsample=0.9,
        colsample_bytree=0.9,
        scale_pos_weight=scale_pos_weight,
        eval_metric="aucpr",
        early_stopping_rounds=50,
        random_state=42,
        n_jobs=-1
    )

    print("\nTraining XGBoost with Early Stopping...")
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=100
    )

    print("\nEvaluating...")
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    precision, recall, thresholds = precision_recall_curve(y_test, y_pred_proba)
    target_recall = 0.85
    
    # We find the *highest* threshold that maintains the target recall
    valid_indices = np.where(recall[:-1] >= target_recall)[0]
    best_idx = valid_indices[-1] if len(valid_indices) > 0 else 0
    best_thresh = thresholds[best_idx]
    
    y_pred = (y_pred_proba >= best_thresh).astype(int)

    print(f"Target Recall: {target_recall} -> Best Threshold: {best_thresh:.4f}")
    
    precision_val = precision_score(y_test, y_pred)
    recall_val = recall_score(y_test, y_pred)
    auc_pr_val = average_precision_score(y_test, y_pred_proba)

    print("\n=== KEY METRICS ===")
    print(f"Precision (Fraud): {precision_val*100:.2f}%")
    print(f"Recall (Fraud):    {recall_val*100:.2f}%")
    print(f"AUC-PR:            {auc_pr_val*100:.2f}%")

    print("\nGenerating SHAP summary plot...")
    explainer = shap.TreeExplainer(model)
    X_test_sample = X_test[:1000] if hasattr(X_test, "iloc") else X_test[:1000]
    shap_values = explainer.shap_values(X_test_sample)

    plt.figure()
    shap.summary_plot(shap_values, X_test_sample, show=False)
    plt.savefig("models/shap_summary.png", bbox_inches='tight')
    plt.close()
    
    print("SHAP plot saved to models/shap_summary.png")

    os.makedirs("models", exist_ok=True)
    joblib.dump(model, "models/xgboost_v1.pkl")
    joblib.dump(best_thresh, "models/threshold_v1.pkl")
    print("Model saved to models/xgboost_v1.pkl and models/threshold_v1.pkl")

if __name__ == "__main__":
    train()