"""
FocusWebCam — Model Training Script
=====================================
Input  : features.csv (output from extract_features.py)
Output : focus_model.pkl  (ready-to-use model)
         training_report.txt

How to use:
  1. pip install scikit-learn pandas matplotlib
  2. python train_model.py --input features.csv
  3. Result: focus_model.pkl in the same folder
"""

import pandas as pd
import numpy as np
import pickle
import argparse
from pathlib import Path

from sklearn.linear_model  import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline      import Pipeline
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    accuracy_score
)


# ─────────────────────────────────────────────
# Preprocessing
# ─────────────────────────────────────────────

def load_and_clean(csv_path: str) -> tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(csv_path)

    print(f"Total rows loaded : {len(df)}")

    # Remove rows with NaN or Inf values
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    print(f"After cleaning    : {len(df)}")

    # Remove extreme outliers (EAR > 1 is physically impossible)
    df = df[df["ear"] <= 1.0]
    df = df[df["ear"] >= 0.0]
    df = df[df["head_pose"] <= 1.0]
    df = df[df["mouth_ratio"] <= 2.0]
    print(f"After outlier filter: {len(df)}")

    X = df[["ear", "head_pose", "mouth_ratio"]]
    y = df["label"]

    return X, y


# ─────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────

def train(X: pd.DataFrame, y: pd.Series) -> Pipeline:
    """
    Pipeline: StandardScaler → Logistic Regression
    
    Why Logistic Regression?
    - Outputs probabilities (we know "how confident the model is about focus")
    - Interpretable — we can see each feature's weight
    - Suitable for small-to-medium datasets
    - Matches Chapter 4 materials
    """
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("clf",    LogisticRegression(
            max_iter=1000,
            class_weight="balanced",  # in case of imbalanced classes
            random_state=42
        ))
    ])

    # 5-fold cross-validation for honest performance estimation
    print("\nCross-validation (5-fold)...")
    cv_scores = cross_val_score(pipeline, X, y, cv=5, scoring="f1")
    print(f"  F1 per fold : {cv_scores.round(3)}")
    print(f"  Avg F1      : {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

    # Train final model with all training data
    pipeline.fit(X, y)
    return pipeline


# ─────────────────────────────────────────────
# Evaluation
# ─────────────────────────────────────────────

def evaluate(pipeline: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> str:
    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    f1  = f1_score(y_test, y_pred)
    cm  = confusion_matrix(y_test, y_pred)
    report = classification_report(
        y_test, y_pred,
        target_names=["NOT_FOCUSED", "FOCUSED"]
    )

    # Model coefficients (interpretation of feature weights)
    scaler = pipeline.named_steps["scaler"]
    clf    = pipeline.named_steps["clf"]
    feature_names = ["ear", "head_pose", "mouth_ratio"]
    coefs = clf.coef_[0]

    output = []
    output.append("=" * 50)
    output.append("MODEL EVALUATION RESULTS")
    output.append("=" * 50)
    output.append(f"\nAccuracy : {acc:.4f} ({acc*100:.2f}%)")
    output.append(f"F1 Score : {f1:.4f}")
    output.append(f"\nConfusion Matrix:")
    output.append(f"  [[TN={cm[0,0]}  FP={cm[0,1]}]")
    output.append(f"   [FN={cm[1,0]}  TP={cm[1,1]}]]")
    output.append(f"\nClassification Report:")
    output.append(report)
    output.append(f"\nFeature Weights (Logistic Regression coefficients):")
    for name, coef in zip(feature_names, coefs):
        direction = "↑ increases FOCUS" if coef > 0 else "↓ decreases FOCUS"
        output.append(f"  {name:<15}: {coef:+.4f}  {direction}")

    output.append("\n" + "=" * 50)
    if f1 >= 0.80:
        output.append("✅ F1 ≥ 0.80 — Model ready for integration into FocusWebCam")
    else:
        output.append("⚠️  F1 < 0.80 — Consider adding more data or trying other algorithms")
    output.append("=" * 50)

    return "\n".join(output)


# ─────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Train FocusWebCam model from features.csv"
    )
    parser.add_argument(
        "--input",
        type=str,
        default="features.csv",
        help="Path to CSV file from extract_features.py"
    )
    parser.add_argument(
        "--output_model",
        type=str,
        default="focus_model.pkl",
        help="Output model file name (default: focus_model.pkl)"
    )
    parser.add_argument(
        "--test_size",
        type=float,
        default=0.2,
        help="Test set proportion (default: 0.2 = 20%%)"
    )
    args = parser.parse_args()

    print("=" * 50)
    print("FocusWebCam — Model Training")
    print("=" * 50)

    # 1. Load data
    X, y = load_and_clean(args.input)
    print(f"\nLabel distribution:")
    print(f"  FOCUS        : {(y==1).sum()} samples")
    print(f"  NOT_FOCUSED  : {(y==0).sum()} samples")

    # 2. Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=args.test_size,
        random_state=42,
        stratify=y  # ensure class proportions are same in train and test
    )
    print(f"\nTraining set : {len(X_train)} samples")
    print(f"Test set     : {len(X_test)} samples")

    # 3. Train
    pipeline = train(X_train, y_train)

    # 4. Evaluate
    report = evaluate(pipeline, X_test, y_test)
    print("\n" + report)

    # 5. Save model
    with open(args.output_model, "wb") as f:
        pickle.dump(pipeline, f)
    print(f"\nModel saved: {args.output_model}")

    # 6. Save report
    report_path = "training_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"Report saved: {report_path}")


if __name__ == "__main__":
    main()