"""
모델 평가 결과 시각화

사용법:
    python 07_visualization.py
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.family'] = 'sans-serif'

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

from config import DATA_DIR, BASE_DIR


RESULT_DIR = os.path.join(BASE_DIR, "results")
os.makedirs(RESULT_DIR, exist_ok=True)


def load_predictions():
    """predictions.npz 로드 (data/ 또는 프로젝트 루트에서 탐색)"""
    candidates = [
        os.path.join(DATA_DIR, "predictions.npz"),
        os.path.join(BASE_DIR, "predictions.npz"),
    ]
    pred_path = next((p for p in candidates if os.path.exists(p)), None)

    if pred_path is None:
        raise FileNotFoundError(
            "predictions.npz 파일이 없습니다.\n"
            "먼저 05_train_models.py를 실행하세요."
        )

    pred_data = np.load(pred_path)
    y_test = pred_data["y_test"]

    return pred_data, y_test


def get_model_names(pred_data):
    """predictions.npz 안에 실제 존재하는 모델 이름만 반환"""
    candidate_models = ["arima", "lstm", "transformer"]

    return [name for name in candidate_models if name in pred_data.files]


def calculate_metrics(pred_data, y_test):
    """모델별 분류 성능 계산"""
    rows = []

    for name in get_model_names(pred_data):
        y_pred = pred_data[name]

        rows.append({
            "model": name.upper(),
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(
                y_test,
                y_pred,
                average="weighted",
                zero_division=0,
            ),
            "recall": recall_score(
                y_test,
                y_pred,
                average="weighted",
                zero_division=0,
            ),
            "f1_score": f1_score(
                y_test,
                y_pred,
                average="weighted",
                zero_division=0,
            ),
        })

    return pd.DataFrame(rows)


def plot_metric_bar(df_metrics):
    """Accuracy / Precision / Recall / F1-score 비교 막대그래프 (랜덤 베이스라인 포함)"""
    metric_cols = ["accuracy", "precision", "recall", "f1_score"]

    x = np.arange(len(df_metrics))
    width = 0.18

    plt.figure(figsize=(10, 6))

    for i, col in enumerate(metric_cols):
        plt.bar(
            x + (i - 1.5) * width,
            df_metrics[col],
            width,
            label=col
        )

    plt.axhline(y=0.5, color="red", linestyle="--", linewidth=1.2, label="Random Baseline (0.5)")
    plt.xticks(x, df_metrics["model"])
    plt.ylim(0.4, 0.75)
    plt.ylabel("Score")
    plt.title("Model Classification Performance")
    plt.legend()
    plt.tight_layout()

    save_path = os.path.join(RESULT_DIR, "model_classification_metrics.png")
    plt.savefig(save_path, dpi=150)
    plt.show()

    print(f"저장 완료: {save_path}")


def plot_confusion_matrices(pred_data, y_test):
    """모델별 confusion matrix 저장"""
    for name in get_model_names(pred_data):
        y_pred = pred_data[name]
        cm = confusion_matrix(y_test, y_pred)

        disp = ConfusionMatrixDisplay(
            confusion_matrix=cm,
            display_labels=["Down/Flat", "Up"]
        )

        fig, ax = plt.subplots(figsize=(5, 5))
        disp.plot(values_format="d", ax=ax, colorbar=False)

        ax.set_title(f"{name.upper()} Confusion Matrix")
        plt.tight_layout()

        save_path = os.path.join(RESULT_DIR, f"{name}_confusion_matrix.png")
        plt.savefig(save_path, dpi=150)
        plt.show()
        plt.close(fig)

        print(f"저장 완료: {save_path}")


def plot_per_class_f1(pred_data, y_test):
    """모델별 Down / Up 클래스 F1-score 분리 비교"""
    from sklearn.metrics import f1_score as _f1

    models, down_f1s, up_f1s = [], [], []

    for name in get_model_names(pred_data):
        y_pred = pred_data[name]
        f1s = _f1(y_test, y_pred, average=None, zero_division=0)
        models.append(name.upper())
        down_f1s.append(f1s[0])
        up_f1s.append(f1s[1] if len(f1s) > 1 else 0.0)

    x = np.arange(len(models))
    width = 0.35

    plt.figure(figsize=(8, 5))
    bars_down = plt.bar(x - width / 2, down_f1s, width, label="Down", color="#4C72B0")
    bars_up   = plt.bar(x + width / 2, up_f1s,   width, label="Up",   color="#DD8452")

    for bar in bars_down + bars_up:
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.005,
            f"{bar.get_height():.3f}",
            ha="center", va="bottom", fontsize=9,
        )

    plt.axhline(y=0.5, color="red", linestyle="--", linewidth=1.2, label="Random Baseline")
    plt.xticks(x, models)
    plt.ylim(0, 0.85)
    plt.ylabel("F1-score")
    plt.title("Per-Class F1-score by Model (Down vs Up)")
    plt.legend()
    plt.tight_layout()

    save_path = os.path.join(RESULT_DIR, "per_class_f1.png")
    plt.savefig(save_path, dpi=150)
    plt.show()
    print(f"저장 완료: {save_path}")


def plot_model_agreement(pred_data, y_test):
    """2개 이상 / 3개 모두 Up 동의 시 정밀도 분석"""
    model_names = get_model_names(pred_data)
    preds = {n: pred_data[n] for n in model_names}

    agree_counts = sum(preds[n] for n in model_names)  # 0~3 사이 값

    labels  = ["1개 이상 Up", "2개 이상 Up", "3개 모두 Up"]
    threshs = [1, 2, 3]
    precisions, coverages = [], []

    for t in threshs:
        mask = agree_counts >= t
        if mask.sum() == 0:
            precisions.append(0.0)
            coverages.append(0.0)
        else:
            precisions.append(float(y_test[mask].mean()))
            coverages.append(float(mask.mean()))

    x = np.arange(len(labels))
    width = 0.35

    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax2 = ax1.twinx()

    bars = ax1.bar(x - width / 2, precisions, width, label="Up Precision", color="#55A868")
    ax2.bar(x + width / 2, coverages, width, label="Coverage (샘플 비율)", color="#C44E52", alpha=0.7)

    for bar, val in zip(bars, precisions):
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.005,
            f"{val:.3f}",
            ha="center", va="bottom", fontsize=9,
        )

    ax1.axhline(y=0.5, color="red", linestyle="--", linewidth=1.2, label="Random Baseline")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)
    ax1.set_ylim(0, 0.85)
    ax1.set_ylabel("Up Precision")
    ax2.set_ylim(0, 1)
    ax2.set_ylabel("Coverage")
    plt.title("Model Agreement Analysis: Up Precision vs Coverage")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right")

    plt.tight_layout()

    save_path = os.path.join(RESULT_DIR, "model_agreement.png")
    plt.savefig(save_path, dpi=150)
    plt.show()
    print(f"저장 완료: {save_path}")


def plot_prediction_distribution(pred_data):
    """모델별 상승/하락 예측 비율 시각화"""
    rows = []

    for name in get_model_names(pred_data):
        y_pred = pred_data[name]

        down_count = int((y_pred == 0).sum())
        up_count = int((y_pred == 1).sum())
        total = len(y_pred)

        rows.append({
            "model": name.upper(),
            "Down/Flat": down_count / total,
            "Up": up_count / total,
        })

    df_dist = pd.DataFrame(rows)

    x = np.arange(len(df_dist))
    width = 0.35

    plt.figure(figsize=(8, 5))
    plt.bar(x - width / 2, df_dist["Down/Flat"], width, label="Down/Flat")
    plt.bar(x + width / 2, df_dist["Up"], width, label="Up")

    plt.xticks(x, df_dist["model"])
    plt.ylim(0, 1)
    plt.ylabel("Prediction Ratio")
    plt.title("Prediction Distribution by Model")
    plt.legend()
    plt.tight_layout()

    save_path = os.path.join(RESULT_DIR, "prediction_distribution.png")
    plt.savefig(save_path, dpi=150)
    plt.show()

    print(f"저장 완료: {save_path}")


def save_prediction_summary(pred_data, y_test):
    """모델별 예측값 분포와 실제 y 분포 저장"""
    rows = []

    true_down = int((y_test == 0).sum())
    true_up = int((y_test == 1).sum())
    true_total = len(y_test)

    rows.append({
        "model": "TRUE_LABEL",
        "down_flat_count": true_down,
        "up_count": true_up,
        "down_flat_ratio": true_down / true_total,
        "up_ratio": true_up / true_total,
    })

    for name in get_model_names(pred_data):
        y_pred = pred_data[name]

        down = int((y_pred == 0).sum())
        up = int((y_pred == 1).sum())
        total = len(y_pred)

        rows.append({
            "model": name.upper(),
            "down_flat_count": down,
            "up_count": up,
            "down_flat_ratio": down / total,
            "up_ratio": up / total,
        })

    df_summary = pd.DataFrame(rows)

    save_path = os.path.join(RESULT_DIR, "prediction_distribution_summary.csv")
    df_summary.to_csv(save_path, index=False, encoding="utf-8-sig")

    print(f"예측 분포 요약 저장 완료: {save_path}")

    return df_summary


def main():
    pred_data, y_test = load_predictions()

    df_metrics = calculate_metrics(pred_data, y_test)

    print("\n모델 성능 요약")
    print(df_metrics.to_string(index=False))

    metric_csv_path = os.path.join(RESULT_DIR, "model_classification_metrics.csv")
    df_metrics.to_csv(metric_csv_path, index=False, encoding="utf-8-sig")
    print(f"\n성능표 저장 완료: {metric_csv_path}")

    print("\n예측 분포 요약")
    df_summary = save_prediction_summary(pred_data, y_test)
    print(df_summary.to_string(index=False))

    plot_metric_bar(df_metrics)
    plot_per_class_f1(pred_data, y_test)
    plot_model_agreement(pred_data, y_test)
    plot_confusion_matrices(pred_data, y_test)
    plot_prediction_distribution(pred_data)


if __name__ == "__main__":
    main()
