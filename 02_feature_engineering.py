"""
기술적 지표 생성 + 로그수익률 + Min-Max 정규화

생성 지표: SMA(5,20,60), RSI(14), MACD, Bollinger Bands
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import os
from config import DATA_DIR, TARGET_THRESHOLD


def add_log_return(df: pd.DataFrame) -> pd.DataFrame:
    """로그 수익률 계산"""
    df["log_return"] = np.log(df["close"] / df["close"].shift(1))
    return df


def add_sma(df: pd.DataFrame, windows=[5, 20, 60]) -> pd.DataFrame:
    """단순 이동평균"""
    for w in windows:
        df[f"sma_{w}"] = df["close"].rolling(window=w).mean()
    return df


def add_rsi(df: pd.DataFrame, period=14) -> pd.DataFrame:
    """RSI"""
    delta = df["close"].diff()
    gain = delta.where(delta > 0, 0.0).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(window=period).mean()
    rs = gain / loss
    df["rsi"] = 100 - (100 / (1 + rs))
    return df


def add_macd(df: pd.DataFrame, fast=12, slow=26, signal=9) -> pd.DataFrame:
    """MACD"""
    ema_fast = df["close"].ewm(span=fast).mean()
    ema_slow = df["close"].ewm(span=slow).mean()
    df["macd"] = ema_fast - ema_slow
    df["macd_signal"] = df["macd"].ewm(span=signal).mean()
    df["macd_hist"] = df["macd"] - df["macd_signal"]
    return df


def add_bollinger(df: pd.DataFrame, window=20, num_std=2) -> pd.DataFrame:
    """볼린저 밴드"""
    sma = df["close"].rolling(window=window).mean()
    std = df["close"].rolling(window=window).std()
    df["bb_upper"] = sma + num_std * std
    df["bb_lower"] = sma - num_std * std
    df["bb_width"] = df["bb_upper"] - df["bb_lower"]
    return df


def add_target(df: pd.DataFrame) -> pd.DataFrame:
    """
    Target threshold 기반 다음 날 주가 방향 생성

    1: 다음 날 수익률이 TARGET_THRESHOLD보다 큰 경우
    0: 다음 날 수익률이 -TARGET_THRESHOLD보다 작은 경우
    NaN: 변동폭이 작아 노이즈성 횡보로 보는 경우
    """
    # 다음 날 수익률 계산
    df["future_return"] = df["close"].shift(-1) / df["close"] - 1

    # target 초기화
    df["target"] = np.nan

    # 기준값 이상 상승한 경우 상승 class
    df.loc[df["future_return"] > TARGET_THRESHOLD, "target"] = 1

    # 기준값 이상 하락한 경우 하락 class
    df.loc[df["future_return"] < -TARGET_THRESHOLD, "target"] = 0

    return df


def normalize(df: pd.DataFrame, feature_cols: list) -> pd.DataFrame:
    """Min-Max 정규화"""
    scaler = MinMaxScaler()
    df[feature_cols] = scaler.fit_transform(df[feature_cols])
    return df


def process_stock(df: pd.DataFrame) -> pd.DataFrame:
    """한 종목에 대해 전체 피처 엔지니어링 수행"""
    df = df.sort_values("date").reset_index(drop=True)
    df = add_log_return(df)
    df = add_sma(df)
    df = add_rsi(df)
    df = add_macd(df)
    df = add_bollinger(df)
    df = add_target(df)
    df = df.dropna().reset_index(drop=True)
    return df


def main():
    input_path = os.path.join(DATA_DIR, "ohlcv.csv")
    df = pd.read_csv(input_path, parse_dates=["date"])

    results = []
    for code, group in df.groupby("code"):
        print(f"{code} 피처 생성 중... ({len(group)}행)", end=" ")
        processed = process_stock(group.copy())
        results.append(processed)
        print(f"→ {len(processed)}행")

    result = pd.concat(results, ignore_index=True)
    output_path = os.path.join(DATA_DIR, "features.csv")
    result.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"\n저장 완료: {output_path} ({len(result)}행)")


if __name__ == "__main__":
    main()
