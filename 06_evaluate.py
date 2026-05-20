"""
모델 평가: Accuracy, Precision, Recall, F1, ROC-AUC, PR-AUC

Transformer 개선 반영:
- transformer_proba가 있으면 threshold를 적용해 label 재계산
- Improved Transformer 최종 threshold 기본값 = 0.51
- ROC-AUC / PR-AUC / predicted positive rate 추가
- results/model_metrics.csv 저장

실행법:
    python 06_evaluate.py
    python 06_evaluate.py --transformer_threshold 0.51
"""

import os
import argparse
import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    average_precision_score,
    classification_report,
)

from config import DATA_DIR, BASE_DIR


RESULT_DIR = os.path.join(BASE_DIR, "results")
os.makedirs(RESULT_DIR, exist_ok=True)


def safe_roc_auc(y_true, y_score):
    if y_score is None:
        return np.nan
    try:
        return roc_auc_score(y_true, y_score)
    except Exception:
        return np.nan


def safe_pr_auc(y_true, y_score):
    if y_score is None:
        return np.nan
    try:
        return average_precision_score(y_true, y_score)
    except Exception:
        return np.nan


def evaluate_model(name, y_true, y_pred, y_proba=None, threshold=None):
    """
    단일 모델 평가.

    y_proba가 있고 threshold가 주어지면,
    y_pred를 probability 기준으로 다시 계산합니다.
    """
    if y_proba is not None:
        y_proba = np.asarray(y_proba).reshape(-1)
        if threshold is not None:
            y_pred = (y_proba > threshold).astype(int)
    else:
        y_proba = None

    y_pred = np.asarray(y_pred).astype(int).reshape(-1)
    y_true = np.asarray(y_true).astype(int).reshape(-1)

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    rec = recall_score(y_true, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    f1_binary = f1_score(y_true, y_pred, zero_division=0)

    roc_auc = safe_roc_auc(y_true, y_proba)
    pr_auc = safe_pr_auc(y_true, y_proba)

    true_positive_rate = float(np.mean(y_true == 1))
    pred_positive_rate = float(np.mean(y_pred == 1))

    print(f"\n{'=' * 60}")
    print(f"  {name}")
    print(f"{'=' * 60}")
    print(f"  Threshold:           {threshold if threshold is not None else 'N/A'}")
    print(f"  Accuracy:            {acc:.6f}")
    print(f"  Precision(weighted): {prec:.6f}")
    print(f"  Recall(weighted):    {rec:.6f}")
    print(f"  F1(weighted):        {f1:.6f}")
    print(f"  F1(binary Up):       {f1_binary:.6f}")

    if not np.isnan(roc_auc):
        print(f"  ROC-AUC:             {roc_auc:.6f}")
    else:
        print("  ROC-AUC:             NaN")

    if not np.isnan(pr_auc):
        print(f"  PR-AUC/AP:           {pr_auc:.6f}")
    else:
        print("  PR-AUC/AP:           NaN")

    print(f"  True positive rate:  {true_positive_rate:.6f}")
    print(f"  Pred positive rate:  {pred_positive_rate:.6f}")

    print(
        "\n"
        + classification_report(
            y_true,
            y_pred,
            target_names=["Down", "Up"],
            zero_division=0,
        )
    )

    return {
        "model": name,
        "threshold": threshold,
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "f1_binary_up": f1_binary,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "true_positive_rate": true_positive_rate,
        "pred_positive_rate": pred_positive_rate,
        "has_proba": y_proba is not None,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Default threshold for probability-based models such as LSTM",
    )
    parser.add_argument(
        "--transformer_threshold",
        type=float,
        default=0.51,
        help="Threshold for improved Transformer. Final selected value is 0.51",
    )
    args = parser.parse_args()

    pred_path = os.path.join(DATA_DIR, "predictions.npz")

    if not os.path.exists(pred_path):
        raise FileNotFoundError(
            f"predictions.npz 파일이 없습니다: {pred_path}\n"
            "먼저 05_train_models.py를 실행하세요."
        )

    pred_data = np.load(pred_path, allow_pickle=True)
    y_test = pred_data["y_test"]

    print(f"predictions.npz keys: {pred_data.files}")
    print(f"\n테스트 샘플 수: {len(y_test)}")
    print(f"  Down(0): {(y_test == 0).sum()} ({(y_test == 0).mean() * 100:.2f}%)")
    print(f"  Up(1):   {(y_test == 1).sum()} ({(y_test == 1).mean() * 100:.2f}%)")

    model_names = ["arima", "lstm", "transformer"]
    results = []

    for name in model_names:
        if name not in pred_data.files:
            print(f"\n[Skip] {name}: predictions.npz에 없음")
            continue

        y_pred = pred_data[name]
        proba_key = f"{name}_proba"
        y_proba = pred_data[proba_key] if proba_key in pred_data.files else None

        if name == "transformer" and y_proba is not None:
            threshold = args.transformer_threshold
            display_name = "TRANSFORMER_IMPROVED"
        elif y_proba is not None:
            threshold = args.threshold
            display_name = name.upper()
        else:
            threshold = None
            display_name = name.upper()

        result = evaluate_model(
            name=display_name,
            y_true=y_test,
            y_pred=y_pred,
            y_proba=y_proba,
            threshold=threshold,
        )

        results.append(result)

    if results:
        df = pd.DataFrame(results)

        print(f"\n{'=' * 60}")
        print("  모델 비교")
        print(f"{'=' * 60}")
        print(df.to_string(index=False))

        save_path = os.path.join(RESULT_DIR, "model_metrics.csv")
        df.to_csv(save_path, index=False, encoding="utf-8-sig")
        print(f"\n결과 저장 완료: {save_path}")

        # 보고서용 간단 표도 저장
        report_cols = ["model", "accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"]
        report_df = df[report_cols].copy()
        report_path = os.path.join(RESULT_DIR, "model_metrics_report_table.csv")
        report_df.to_csv(report_path, index=False, encoding="utf-8-sig")
        print(f"보고서용 표 저장 완료: {report_path}")


if __name__ == "__main__":
    main()
