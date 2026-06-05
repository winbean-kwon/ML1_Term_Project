"""
기술적 지표 생성 + 로그수익률 + Min-Max 정규화

생성 지표: SMA(5,20,60), RSI(14), MACD, Bollinger Bands,
          모멘텀(5d/20d), 거래량 비율, 52주 위치, 변동성, ADX, OBV
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import os
from config import DATA_DIR, TARGET_THRESHOLD


def add_log_return(df: pd.DataFrame) -> pd.DataFrame:
    df["log_return"] = np.log(df["close"] / df["close"].shift(1))
    return df


def add_sma(df: pd.DataFrame, windows=[5, 20, 60]) -> pd.DataFrame:
    for w in windows:
        df[f"sma_{w}"] = df["close"].rolling(window=w).mean()
    return df


def add_rsi(df: pd.DataFrame, period=14) -> pd.DataFrame:
    delta = df["close"].diff()
    gain = delta.where(delta > 0, 0.0).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(window=period).mean()
    df["rsi"] = 100 - (100 / (1 + gain / loss))
    return df


def add_macd(df: pd.DataFrame, fast=12, slow=26, signal=9) -> pd.DataFrame:
    ema_fast = df["close"].ewm(span=fast).mean()
    ema_slow = df["close"].ewm(span=slow).mean()
    df["macd"] = ema_fast - ema_slow
    df["macd_signal"] = df["macd"].ewm(span=signal).mean()
    df["macd_hist"] = df["macd"] - df["macd_signal"]
    return df


def add_bollinger(df: pd.DataFrame, window=20, num_std=2) -> pd.DataFrame:
    sma = df["close"].rolling(window=window).mean()
    std = df["close"].rolling(window=window).std()
    df["bb_upper"] = sma + num_std * std
    df["bb_lower"] = sma - num_std * std
    df["bb_width"] = df["bb_upper"] - df["bb_lower"]
    return df


def add_momentum(df: pd.DataFrame) -> pd.DataFrame:
    df["return_5d"]  = df["close"] / df["close"].shift(5)  - 1
    df["return_20d"] = df["close"] / df["close"].shift(20) - 1
    return df


def add_volume_ratio(df: pd.DataFrame, window=20) -> pd.DataFrame:
    df["volume_ratio_20"] = df["volume"] / df["volume"].rolling(window=window).mean()
    return df


def add_52w_position(df: pd.DataFrame) -> pd.DataFrame:
    df["high_52w_ratio"] = df["close"] / df["close"].rolling(window=252).max()
    return df


def add_volatility(df: pd.DataFrame, window=20) -> pd.DataFrame:
    df["volatility_20"] = df["log_return"].rolling(window=window).std()
    return df


def add_adx(df: pd.DataFrame, period=14) -> pd.DataFrame:
    """ADX (Average Directional Index): 추세 강도 지표"""
    high, low, close = df["high"], df["low"], df["close"]

    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low  - close.shift(1)).abs(),
    ], axis=1).max(axis=1)

    up   = high - high.shift(1)
    down = low.shift(1) - low
    plus_dm  = np.where((up > down) & (up > 0),   up,   0.0)
    minus_dm = np.where((down > up) & (down > 0), down, 0.0)

    tr_roll  = tr.rolling(period).mean()
    plus_di  = 100 * pd.Series(plus_dm,  index=df.index).rolling(period).mean() / (tr_roll + 1e-9)
    minus_di = 100 * pd.Series(minus_dm, index=df.index).rolling(period).mean() / (tr_roll + 1e-9)

    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di + 1e-9)
    df["adx"] = dx.rolling(period).mean()
    return df


def add_obv(df: pd.DataFrame) -> pd.DataFrame:
    """OBV (On-Balance Volume): 거래량+가격 방향 결합 지표"""
    direction = np.sign(df["close"].diff()).fillna(0)
    df["obv"] = (df["volume"] * direction).cumsum()
    return df


def add_target(df: pd.DataFrame) -> pd.DataFrame:
    """종목별 분위수 기반 target (상위 30% = Up, 하위 30% = Down, 중간 40% = NaN)"""
    df["future_return"] = df["close"].shift(-3) / df["close"] - 1
    q30 = df["future_return"].quantile(0.30)
    q70 = df["future_return"].quantile(0.70)
    df["target"] = np.nan
    df.loc[df["future_return"] >= q70, "target"] = 1
    df.loc[df["future_return"] <= q30, "target"] = 0
    return df


def normalize(df: pd.DataFrame, feature_cols: list) -> pd.DataFrame:
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
    df = add_momentum(df)
    df = add_volume_ratio(df)
    df = add_52w_position(df)
    df = add_volatility(df)
    df = add_adx(df)
    df = add_obv(df)
    df = add_target(df)
    df = df.dropna().reset_index(drop=True)
    return df


def main():
    input_path = os.path.join(DATA_DIR, "ohlcv.csv")
    df = pd.read_csv(input_path, parse_dates=["date"])

    # KOSPI proxy: 전종목 동일가중 평균 수익률을 시장 수익률로 사용
    df["_log_ret"] = np.log(
        df.groupby("code")["close"].transform(lambda x: x / x.shift(1))
    )
    market_return = df.groupby("date")["_log_ret"].mean()

    results = []
    for code, group in df.groupby("code"):
        print(f"{code} 피처 생성 중... ({len(group)}행)", end=" ")
        processed = process_stock(group.copy())
        # 시장 대비 초과 수익률
        processed = processed.join(market_return.rename("_mkt"), on="date")
        processed["relative_return"] = processed["log_return"] - processed["_mkt"]
        processed = processed.drop(columns=["_mkt"])
        results.append(processed)
        print(f"→ {len(processed)}행")

    result = pd.concat(results, ignore_index=True)
    output_path = os.path.join(DATA_DIR, "features.csv")
    result.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"\n저장 완료: {output_path} ({len(result)}행)")


if __name__ == "__main__":
    main()
