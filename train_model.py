"""
FocusWebCam — Script Training Model
=====================================
Input  : features.csv (hasil dari extract_features.py)
Output : focus_model.pkl  (model siap pakai)
         label_encoder.pkl
         training_report.txt

Cara pakai:
  1. pip install scikit-learn pandas matplotlib
  2. python train_model.py --input features.csv
  3. Hasil: focus_model.pkl di folder yang sama
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

    print(f"Total baris dimuat : {len(df)}")

    # Hapus baris dengan nilai NaN atau Inf
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    print(f"Setelah cleaning   : {len(df)}")

    # Hapus outlier ekstrem (EAR > 1 tidak mungkin secara fisik)
    df = df[df["ear"] <= 1.0]
    df = df[df["ear"] >= 0.0]
    df = df[df["head_pose"] <= 1.0]
    df = df[df["mouth_ratio"] <= 2.0]
    print(f"Setelah filter outlier: {len(df)}")

    X = df[["ear", "head_pose", "mouth_ratio"]]
    y = df["label"]

    return X, y


# ─────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────

def train(X: pd.DataFrame, y: pd.Series) -> Pipeline:
    """
    Pipeline: StandardScaler → Logistic Regression
    
    Kenapa Logistic Regression?
    - Output probabilitas (kita tahu "seberapa yakin model ini fokus")
    - Interpretable — bisa lihat bobot tiap fitur
    - Cocok untuk dataset kecil-menengah
    - Sesuai materi Bab 4
    """
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("clf",    LogisticRegression(
            max_iter=1000,
            class_weight="balanced",  # antisipasi kalau kelas tidak seimbang
            random_state=42
        ))
    ])

    # Cross-validation 5-fold untuk estimasi performa yang lebih jujur
    print("\nCross-validation (5-fold)...")
    cv_scores = cross_val_score(pipeline, X, y, cv=5, scoring="f1")
    print(f"  F1 per fold : {cv_scores.round(3)}")
    print(f"  F1 rata-rata: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

    # Train final model dengan semua data training
    pipeline.fit(X, y)
    return pipeline


# ─────────────────────────────────────────────
# Evaluasi
# ─────────────────────────────────────────────

def evaluate(pipeline: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> str:
    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    f1  = f1_score(y_test, y_pred)
    cm  = confusion_matrix(y_test, y_pred)
    report = classification_report(
        y_test, y_pred,
        target_names=["TIDAK_FOKUS", "FOKUS"]
    )

    # Koefisien model (interpretasi bobot fitur)
    scaler = pipeline.named_steps["scaler"]
    clf    = pipeline.named_steps["clf"]
    feature_names = ["ear", "head_pose", "mouth_ratio"]
    coefs = clf.coef_[0]

    output = []
    output.append("=" * 50)
    output.append("HASIL EVALUASI MODEL")
    output.append("=" * 50)
    output.append(f"\nAccuracy : {acc:.4f} ({acc*100:.2f}%)")
    output.append(f"F1 Score : {f1:.4f}")
    output.append(f"\nConfusion Matrix:")
    output.append(f"  [[TN={cm[0,0]}  FP={cm[0,1]}]")
    output.append(f"   [FN={cm[1,0]}  TP={cm[1,1]}]]")
    output.append(f"\nClassification Report:")
    output.append(report)
    output.append(f"\nBobot Fitur (koefisien Logistic Regression):")
    for name, coef in zip(feature_names, coefs):
        arah = "↑ meningkatkan FOKUS" if coef > 0 else "↓ menurunkan FOKUS"
        output.append(f"  {name:<15}: {coef:+.4f}  {arah}")

    output.append("\n" + "=" * 50)
    if f1 >= 0.80:
        output.append("✅ F1 ≥ 0.80 — Model siap diintegrasikan ke FocusWebCam")
    else:
        output.append("⚠️  F1 < 0.80 — Pertimbangkan tambah data atau coba algoritma lain")
    output.append("=" * 50)

    return "\n".join(output)


# ─────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Training model FocusWebCam dari features.csv"
    )
    parser.add_argument(
        "--input",
        type=str,
        default="features.csv",
        help="Path ke file CSV hasil extract_features.py"
    )
    parser.add_argument(
        "--output_model",
        type=str,
        default="focus_model.pkl",
        help="Nama file model output (default: focus_model.pkl)"
    )
    parser.add_argument(
        "--test_size",
        type=float,
        default=0.2,
        help="Proporsi test set (default: 0.2 = 20%%)"
    )
    args = parser.parse_args()

    print("=" * 50)
    print("FocusWebCam — Training Model")
    print("=" * 50)

    # 1. Load data
    X, y = load_and_clean(args.input)
    print(f"\nDistribusi label:")
    print(f"  FOKUS       : {(y==1).sum()} sampel")
    print(f"  TIDAK_FOKUS : {(y==0).sum()} sampel")

    # 2. Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=args.test_size,
        random_state=42,
        stratify=y  # pastikan proporsi kelas sama di train dan test
    )
    print(f"\nTraining set : {len(X_train)} sampel")
    print(f"Test set     : {len(X_test)} sampel")

    # 3. Train
    pipeline = train(X_train, y_train)

    # 4. Evaluasi
    report = evaluate(pipeline, X_test, y_test)
    print("\n" + report)

    # 5. Simpan model
    with open(args.output_model, "wb") as f:
        pickle.dump(pipeline, f)
    print(f"\nModel disimpan: {args.output_model}")

    # 6. Simpan report
    report_path = "training_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"Report disimpan: {report_path}")


if __name__ == "__main__":
    main()