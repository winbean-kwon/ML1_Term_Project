"""
주가 피처 + 감성 점수를 합쳐 시계열 입력 데이터셋 생성

각 샘플: (WINDOW_SIZE일치 피처 벡터) → target (다음 날 상승/하락)
"""
import pandas as pd
import numpy as np
import os
from config import DATA_DIR, WINDOW_SIZE, TEST_RATIO, SEED


FEATURE_COLS = [
    "log_return", "volume",
    "sma_5", "sma_20", "sma_60",
    "rsi", "macd", "macd_signal", "macd_hist",
    "bb_upper", "bb_lower", "bb_width",
    "sentiment_mean", "sentiment_std", "news_count",
]


def create_sequences(df: pd.DataFrame, window: int, feature_cols: list):
    """시계열 윈도우 데이터 생성"""
    X, y = [], []
    data = df[feature_cols].values
    target = df["target"].values

    for i in range(window, len(data)):
        X.append(data[i - window:i])
        y.append(target[i])

    return np.array(X), np.array(y)


def main():
    # 피처 데이터 로드
    features_path = os.path.join(DATA_DIR, "features.csv")
    sentiment_path = os.path.join(DATA_DIR, "sentiment.csv")

    df_feat = pd.read_csv(features_path, parse_dates=["date"])
    df_feat["code"] = df_feat["code"].astype(str).str.zfill(6)

    df_sent = pd.read_csv(sentiment_path, parse_dates=["date"])
    df_sent.rename(columns={"종목코드": "code"}, inplace=True)
    df_sent["code"] = df_sent["code"].astype(str).str.zfill(6)

    # 주가 + 감성 병합 (날짜·종목 기준, 뉴스 없는 날은 0으로 채움)
    df = df_feat.merge(df_sent, on=["code", "date"], how="left")
    df["sentiment_mean"] = df["sentiment_mean"].fillna(0)
    df["sentiment_std"] = df["sentiment_std"].fillna(0)
    df["news_count"] = df["news_count"].fillna(0)

    print(f"병합 데이터: {len(df)}행, 종목 수: {df['code'].nunique()}")

    # 종목별 시퀀스 생성
    all_X, all_y = [], []
    for code, group in df.groupby("code"):
        group = group.sort_values("date").reset_index(drop=True)
        if len(group) < WINDOW_SIZE + 1:
            continue
        X, y = create_sequences(group, WINDOW_SIZE, FEATURE_COLS)
        all_X.append(X)
        all_y.append(y)

    X = np.concatenate(all_X)
    y = np.concatenate(all_y)
    print(f"시퀀스 생성 완료: X={X.shape}, y={y.shape}")

    # 학습/테스트 분할 (시계열이므로 시간순 분할)
    split_idx = int(len(X) * (1 - TEST_RATIO))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    print(f"Train: {X_train.shape}, Test: {X_test.shape}")

    # 저장
    np.savez(
        os.path.join(DATA_DIR, "dataset.npz"),
        X_train=X_train, X_test=X_test,
        y_train=y_train, y_test=y_test,
    )
    print("저장 완료: dataset.npz")


if __name__ == "__main__":
    main()
