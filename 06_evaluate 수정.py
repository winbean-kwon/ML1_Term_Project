"""
모델 평가: Accuracy, Precision, Recall, F1, ROC-AUC, PR-AUC

수정 내용:
- predictions.npz에서 *_proba가 있으면 ROC-AUC / PR-AUC 계산
- pred_positive_rate 추가
- true_positive_rate 추가
- ARIMA처럼 proba가 없는 모델은 ROC-AUC / PR-AUC를 NaN 처리
- 결과를 results/model_metrics.csv로 저장

실행법:
    python 06_evaluate.py
"""

import os
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
    """
    ROC-AUC 안전 계산.
    y_true가 한 클래스만 있거나 y_score가 없으면 NaN 반환.
    """
    if y_score is None:
        return np.nan

    try:
        return roc_auc_score(y_true, y_score)
    except Exception:
        return np.nan


def safe_pr_auc(y_true, y_score):
    """
    PR-AUC, Average Precision 안전 계산.
    y_true가 한 클래스만 있거나 y_score가 없으면 NaN 반환.
    """
    if y_score is None:
        return np.nan

    try:
        return average_precision_score(y_true, y_score)
    except Exception:
        return np.nan


def evaluate_model(name, y_true, y_pred, y_proba=None):
    """
    단일 모델 평가 결과를 출력합니다.

    Parameters
    ----------
    name : str
        모델 이름
    y_true : np.ndarray
        실제 정답
    y_pred : np.ndarray
        0/1 예측값
    y_proba : np.ndarray or None
        상승 class에 대한 예측 확률
    """
    y_pred = np.asarray(y_pred).astype(int)

    if y_proba is not None:
        y_proba = np.asarray(y_proba).reshape(-1)

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    rec = recall_score(y_true, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)

    roc_auc = safe_roc_auc(y_true, y_proba)
    pr_auc = safe_pr_auc(y_true, y_proba)

    true_positive_rate = float(np.mean(y_true == 1))
    pred_positive_rate = float(np.mean(y_pred == 1))

    print(f"\n{'=' * 50}")
    print(f"  {name}")
    print(f"{'=' * 50}")
    print(f"  Accuracy:            {acc:.4f}")
    print(f"  Precision(weighted): {prec:.4f}")
    print(f"  Recall(weighted):    {rec:.4f}")
    print(f"  F1(weighted):        {f1:.4f}")

    if not np.isnan(roc_auc):
        print(f"  ROC-AUC:             {roc_auc:.4f}")
    else:
        print("  ROC-AUC:             NaN (proba 없음)")

    if not np.isnan(pr_auc):
        print(f"  PR-AUC/AP:           {pr_auc:.4f}")
    else:
        print("  PR-AUC/AP:           NaN (proba 없음)")

    print(f"  True positive rate:  {true_positive_rate:.4f}")
    print(f"  Pred positive rate:  {pred_positive_rate:.4f}")

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
        "accuracy": acc,
        "precision_weighted": prec,
        "recall_weighted": rec,
        "f1_weighted": f1,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "true_positive_rate": true_positive_rate,
        "pred_positive_rate": pred_positive_rate,
        "has_proba": y_proba is not None,
    }


def main():
    pred_path = os.path.join(DATA_DIR, "predictions.npz")

    if not os.path.exists(pred_path):
        raise FileNotFoundError(
            f"predictions.npz 파일이 없습니다: {pred_path}\n"
            "먼저 05_train_models.py를 실행하세요."
        )

    pred_data = np.load(pred_path)
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

        result = evaluate_model(
            name=name.upper(),
            y_true=y_test,
            y_pred=y_pred,
            y_proba=y_proba,
        )

        results.append(result)

    if results:
        df = pd.DataFrame(results)

        # ROC-AUC가 있는 모델이 위로 오도록 정렬
        df_sorted = df.sort_values(
            by=["roc_auc", "accuracy"],
            ascending=[False, False],
            na_position="last",
        )

        print(f"\n{'=' * 50}")
        print("  모델 비교")
        print(f"{'=' * 50}")
        print(df_sorted.to_string(index=False))

        save_path = os.path.join(RESULT_DIR, "model_metrics.csv")
        df_sorted.to_csv(save_path, index=False, encoding="utf-8-sig")

        print(f"\n결과 저장 완료: {save_path}")


if __name__ == "__main__":
    main()