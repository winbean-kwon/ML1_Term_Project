"""
모델 평가: Accuracy, F1-score, 누적 수익률, 샤프 비율
"""
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, classification_report
import os
from config import DATA_DIR


def cumulative_return(y_true, y_pred, log_returns):
    """
    예측 방향에 따라 매수/매도 시뮬레이션.
    y_pred=1이면 매수(수익률 그대로), y_pred=0이면 관망(수익률 0).
    """
    strategy_returns = log_returns * y_pred
    cum_return = np.exp(strategy_returns.cumsum())[-1] - 1
    return cum_return, strategy_returns


def sharpe_ratio(strategy_returns, trading_days=252):
    """연간화 샤프 비율 (무위험 수익률 = 0 가정)"""
    if strategy_returns.std() == 0:
        return 0.0
    return (strategy_returns.mean() / strategy_returns.std()) * np.sqrt(trading_days)


def evaluate_model(name, y_true, y_pred, log_returns=None):
    """단일 모델 평가 결과를 출력합니다."""
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="weighted")

    print(f"\n{'='*40}")
    print(f"  {name}")
    print(f"{'='*40}")
    print(f"  Accuracy:  {acc:.4f}")
    print(f"  F1-score:  {f1:.4f}")

    if log_returns is not None:
        cum_ret, strat_ret = cumulative_return(y_true, y_pred, log_returns)
        sr = sharpe_ratio(strat_ret)
        print(f"  Cumulative Return: {cum_ret:.4f} ({cum_ret*100:.2f}%)")
        print(f"  Sharpe Ratio:      {sr:.4f}")

    print(f"\n{classification_report(y_true, y_pred, target_names=['Down', 'Up'])}")
    return {"model": name, "accuracy": acc, "f1": f1}


def main():
    # 예측 결과 로드
    pred_data = np.load(os.path.join(DATA_DIR, "predictions.npz"))
    y_test = pred_data["y_test"]

    # 로그수익률 로드 (시뮬레이션용, 없으면 None)
    log_returns = None
    dataset_path = os.path.join(DATA_DIR, "dataset.npz")
    try:
        ds = np.load(dataset_path)
        X_test = ds["X_test"]
        # log_return은 첫 번째 피처, 마지막 시점
        log_returns = X_test[:, -1, 0]
    except Exception:
        print("로그수익률 로드 실패 → 금융 지표 생략")

    model_names = ["arima", "lstm", "transformer"]
    results = []

    for name in model_names:
        if name in pred_data:
            r = evaluate_model(name.upper(), y_test, pred_data[name], log_returns)
            results.append(r)

    # 비교 테이블
    if results:
        df = pd.DataFrame(results)
        print(f"\n{'='*40}")
        print("  모델 비교")
        print(f"{'='*40}")
        print(df.to_string(index=False))


if __name__ == "__main__":
    main()
