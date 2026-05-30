"""
주가 피처 + 감성 점수를 합쳐 시계열 입력 데이터셋 생성

수정 내용:
1. news_count를 직접 피처로 사용하지 않음
2. news_count를 sentiment_mean, sentiment_std에 가중치로 반영
3. sentiment_lag/change/news_count_zscore_20 감성 파생 피처 사용
4. 종목별 train/test split 이후 train 기준으로 Min-Max 정규화
5. 종목별 시간 순서를 유지한 채 시퀀스 생성
6. 뉴스 데이터 기간(NEWS_START_DATE ~ NEWS_END_DATE)에만 학습/테스트 구간 한정
   - 앞 TRAIN_MONTHS개월 → train, 나머지 → test (날짜 기준 고정 split)
   - sentiment feature가 실제로 존재하는 구간만 사용하므로 0-패딩 문제 해소
"""

import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import MinMaxScaler
from config import DATA_DIR, WINDOW_SIZE, SEED, NEWS_START_DATE, NEWS_END_DATE, TRAIN_MONTHS


FEATURE_COLS = [
    "log_return", "volume",
    "sma_5", "sma_20", "sma_60",
    "rsi", "macd", "macd_signal", "macd_hist",
    "bb_upper", "bb_lower", "bb_width",

    # 모멘텀
    "return_5d", "return_20d",

    # 거래량 이상
    "volume_ratio_20",

    # 52주 고가 대비 위치
    "high_52w_ratio",

    # 변동성
    "volatility_20",

    # 시장 대비 초과 수익률
    "relative_return",

    # 추세 강도 (ADX)
    "adx",

    # 거래량+가격 방향 결합 (OBV)
    "obv",

    # news_count 기반 weighted sentiment
    "sentiment_mean_weighted",
    "sentiment_std_weighted",

    # 감성 시계열 파생 피처
    "sentiment_lag1",
    "sentiment_lag2",
    "sentiment_change",
    "news_count_zscore_20",

    # 감성 방향성 정교화: 연속형 신호 (binary 플래그 대체)
    # sentiment_up_signal   = max(sentiment_mean, 0) * log1p(news_count)
    # sentiment_down_signal = max(-sentiment_mean, 0) * log1p(news_count)
    "sentiment_up_signal",
    "sentiment_down_signal",
]


def create_sequences(df: pd.DataFrame, window: int, feature_cols: list,
                     with_meta: bool = False):
    """시계열 윈도우 데이터 생성.
    with_meta=True 이면 각 샘플의 (code, date) 메타데이터도 반환.
    """
    X, y, meta = [], [], []

    data   = df[feature_cols].values
    target = df["target"].values

    for i in range(window, len(data)):
        X.append(data[i - window:i])
        y.append(target[i])
        if with_meta:
            meta.append((df["code"].iloc[i], str(df["date"].iloc[i])[:10]))

    if with_meta:
        return np.array(X), np.array(y), meta
    return np.array(X), np.array(y)


def add_weighted_sentiment(df: pd.DataFrame) -> pd.DataFrame:
    """
    news_count를 감성 피처에 가중치로 반영합니다.

    [기존] sentiment_mean_weighted, sentiment_std_weighted
    [추가] sentiment_up_signal, sentiment_down_signal
        - binary 플래그(sentiment_positive/negative) 대신 연속형으로 교체
        - sentiment_up_signal   = max(sentiment_mean, 0) * log1p(news_count)
          → 긍정 감성이 강하고 뉴스가 많을수록 큰 값
        - sentiment_down_signal = max(-sentiment_mean, 0) * log1p(news_count)
          → 부정 감성이 강하고 뉴스가 많을수록 큰 값
        - binary(0/1)보다 뉴스 양·감성 강도를 모두 반영하므로 모델 변별력 향상
    """
    df = df.copy()

    weight = np.log1p(df["news_count"])

    df["sentiment_mean_weighted"] = df["sentiment_mean"] * weight
    df["sentiment_std_weighted"]  = df["sentiment_std"] * weight

    # 연속형 방향성 감성 피처 (binary 플래그 대체)
    df["sentiment_up_signal"]   = np.clip(df["sentiment_mean"],  0, None) * weight
    df["sentiment_down_signal"] = np.clip(-df["sentiment_mean"], 0, None) * weight

    return df


def main():
    np.random.seed(SEED)

    features_path = os.path.join(DATA_DIR, "features.csv")
    sentiment_path = os.path.join(DATA_DIR, "sentiment.csv")

    df_feat = pd.read_csv(features_path, parse_dates=["date"])
    df_feat["code"] = df_feat["code"].astype(str).str.zfill(6)

    df_sent = pd.read_csv(sentiment_path, parse_dates=["date"])
    df_sent.rename(columns={"종목코드": "code"}, inplace=True)
    df_sent["code"] = df_sent["code"].astype(str).str.zfill(6)

    # 주가 피처와 감성 점수를 종목코드·날짜 기준으로 병합
    df = df_feat.merge(
        df_sent,
        on=["code", "date"],
        how="left",
        validate="one_to_one"
    )

    sentiment_cols = [
        "sentiment_mean",
        "sentiment_std",
        "news_count",
        "sentiment_lag1",
        "sentiment_lag2",
        "sentiment_change",
        "news_count_zscore_20",
    ]

    for col in sentiment_cols:
        if col not in df.columns:
            df[col] = 0

    df[sentiment_cols] = df[sentiment_cols].fillna(0)

    df = add_weighted_sentiment(df)

    # 감성 방향성 정교화: has_news 대체
    df["sentiment_positive"] = (df["sentiment_mean"] > 0.1).astype(float)
    df["sentiment_negative"] = (df["sentiment_mean"] < -0.1).astype(float)

    # 모델 입력에 필요한 컬럼 결측 제거
    df = df.dropna(subset=FEATURE_COLS + ["target"])

    print(f"병합 데이터: {len(df)}행, 종목 수: {df['code'].nunique()}")

    # 뉴스 기간 기준 날짜 경계 계산
    NEWS_START   = pd.Timestamp(NEWS_START_DATE)
    NEWS_END     = pd.Timestamp(NEWS_END_DATE)
    TRAIN_CUTOFF = NEWS_START + pd.DateOffset(months=TRAIN_MONTHS)

    print(f"뉴스 기간: {NEWS_START.date()} ~ {NEWS_END.date()}")
    print(f"Train 구간: {NEWS_START.date()} ~ {TRAIN_CUTOFF.date()}")
    print(f"Test  구간: {TRAIN_CUTOFF.date()} ~ {NEWS_END.date()}")

    all_X_train, all_y_train = [], []
    all_X_test, all_y_test = [], []
    all_train_meta, all_test_meta = [], []

    used_codes = 0
    skipped_codes = 0

    for code, group in df.groupby("code"):
        # 각 종목 내부에서 날짜 기준으로 정렬
        group = group.sort_values("date").reset_index(drop=True)

        # 뉴스 기간 전체 + context 버퍼(시퀀스 생성용 과거 WINDOW_SIZE일)만 남김
        context_start = NEWS_START - pd.Timedelta(days=40)
        group = group[(group["date"] >= context_start) & (group["date"] <= NEWS_END)].copy()

        if len(group) < WINDOW_SIZE + 2:
            skipped_codes += 1
            continue

        # 날짜 기준 고정 split: NEWS_START ~ TRAIN_CUTOFF → train
        train_group = group[group["date"] < TRAIN_CUTOFF].copy()

        # test context: train 마지막 WINDOW_SIZE 행 + TRAIN_CUTOFF 이후 행
        test_context = train_group.iloc[-WINDOW_SIZE:].copy()
        test_only    = group[group["date"] >= TRAIN_CUTOFF].copy()
        test_group   = pd.concat([test_context, test_only]).reset_index(drop=True)

        if len(train_group) < WINDOW_SIZE + 1 or len(test_group) < WINDOW_SIZE + 1:
            skipped_codes += 1
            continue

        # 정규화는 train 기준 fit, test는 transform만
        scaler = MinMaxScaler()
        train_group[FEATURE_COLS] = scaler.fit_transform(train_group[FEATURE_COLS])
        test_group[FEATURE_COLS] = scaler.transform(test_group[FEATURE_COLS])

        X_train_part, y_train_part, train_meta = create_sequences(
            train_group, WINDOW_SIZE, FEATURE_COLS, with_meta=True
        )
        X_test_part, y_test_part, test_meta = create_sequences(
            test_group, WINDOW_SIZE, FEATURE_COLS, with_meta=True
        )

        all_X_train.append(X_train_part)
        all_y_train.append(y_train_part)
        all_X_test.append(X_test_part)
        all_y_test.append(y_test_part)
        all_train_meta.extend(train_meta)
        all_test_meta.extend(test_meta)

        used_codes += 1

    if not all_X_train or not all_X_test:
        raise ValueError(
            "No train/test sequences were generated. "
            "Check data size or WINDOW_SIZE."
        )

    X_train = np.concatenate(all_X_train)
    y_train = np.concatenate(all_y_train)
    X_test = np.concatenate(all_X_test)
    y_test = np.concatenate(all_y_test)

    # 테스트 샘플별 (종목코드, 날짜) 메타데이터
    train_codes = np.array([m[0] for m in all_train_meta])
    train_dates = np.array([m[1] for m in all_train_meta])
    test_codes  = np.array([m[0] for m in all_test_meta])
    test_dates  = np.array([m[1] for m in all_test_meta])

    print(f"사용 종목 수: {used_codes}, 제외 종목 수: {skipped_codes}")
    print(f"Train: {X_train.shape}, Test: {X_test.shape}")

    output_path = os.path.join(DATA_DIR, "dataset.npz")

    np.savez(
        output_path,
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        feature_cols=np.array(FEATURE_COLS),
        train_codes=train_codes,
        train_dates=train_dates,
        test_codes=test_codes,
        test_dates=test_dates,
    )

    print(f"저장 완료: {output_path}")
    print(f"Train: {X_train.shape}  |  train_dates: {train_dates.shape}")
    print(f"Test : {X_test.shape}   |  test_dates : {test_dates.shape}")


if __name__ == "__main__":
    main()