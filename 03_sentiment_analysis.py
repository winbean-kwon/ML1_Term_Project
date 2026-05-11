"""
FinBERT를 사용한 뉴스 감성 분석

뉴스 제목(또는 본문)을 FinBERT에 넣어 감성 점수(positive/negative/neutral)를 산출하고,
날짜별·종목별로 집계합니다.
"""
import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import os
from config import DATA_DIR, NEWS_PATH


MODEL_NAME = "ProsusAI/finbert"


def load_finbert():
    """FinBERT 모델과 토크나이저 로드"""
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    model.eval()
    return tokenizer, model


def predict_sentiment(texts: list, tokenizer, model, batch_size=32) -> list:
    """텍스트 리스트에 대해 감성 점수를 반환합니다.
    Returns: list of dict with keys: positive, negative, neutral
    """
    results = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        inputs = tokenizer(batch, padding=True, truncation=True,
                           max_length=512, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        # FinBERT labels: positive, negative, neutral
        for p in probs:
            results.append({
                "positive": p[0].item(),
                "negative": p[1].item(),
                "neutral": p[2].item(),
            })
    return results


def compute_sentiment_score(row: dict) -> float:
    """감성 점수를 단일 스칼라로 변환 (positive - negative)"""
    return row["positive"] - row["negative"]

def add_sentiment_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    종목별 일별 감성 데이터에 시계열 파생 피처를 추가합니다.

    추가 피처:
    - sentiment_lag1: 1일 전 감성 평균
    - sentiment_lag2: 2일 전 감성 평균
    - sentiment_change: 전날 대비 감성 변화량
    - news_count_zscore_20: 최근 20일 기준 뉴스 집중도 z-score
    """
    df = df.copy()

    # 종목코드와 날짜 형식 정리
    df["종목코드"] = df["종목코드"].astype(str).str.zfill(6)
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()

    # 종목별 날짜순 정렬
    df = df.sort_values(["종목코드", "date"]).reset_index(drop=True)

    # 기본 감성 컬럼 결측 처리
    base_cols = ["sentiment_mean", "sentiment_std", "news_count"]
    for col in base_cols:
        df[col] = df[col].fillna(0)

    group = df.groupby("종목코드", group_keys=False)

    # 1일 전, 2일 전 감성값
    df["sentiment_lag1"] = group["sentiment_mean"].shift(1)
    df["sentiment_lag2"] = group["sentiment_mean"].shift(2)

    # 전날 대비 감성 변화량
    df["sentiment_change"] = group["sentiment_mean"].diff()

    # 최근 20일 기준 뉴스 개수 rolling 평균/표준편차
    # 현재 날짜의 news_count가 기준 계산에 들어가지 않도록 shift(1) 사용
    rolling_mean_20 = group["news_count"].transform(
        lambda x: x.shift(1).rolling(window=20, min_periods=5).mean()
    )

    rolling_std_20 = group["news_count"].transform(
        lambda x: x.shift(1).rolling(window=20, min_periods=5).std()
    )

    # 평소 뉴스량 대비 오늘 뉴스가 얼마나 몰렸는지 계산
    df["news_count_zscore_20"] = (
        (df["news_count"] - rolling_mean_20) / (rolling_std_20 + 1e-6)
    )

    # 초반 구간 NaN, 표준편차 0 등으로 생기는 결측/무한대 처리
    new_cols = [
        "sentiment_lag1",
        "sentiment_lag2",
        "sentiment_change",
        "news_count_zscore_20",
    ]

    df[new_cols] = (
        df[new_cols]
        .replace([np.inf, -np.inf], 0)
        .fillna(0)
    )

    return df    

def main():
    # 뉴스 데이터 로드
    df_news = pd.read_csv(NEWS_PATH)
    df_news = df_news.dropna(subset=["제목"])
    print(f"뉴스 {len(df_news)}건 로드")

    # FinBERT 로드
    print("FinBERT 로드 중...")
    tokenizer, model = load_finbert()

    # 감성 분석 수행
    texts = df_news["제목"].tolist()
    print(f"감성 분석 수행 중... ({len(texts)}건)")
    sentiments = predict_sentiment(texts, tokenizer, model)

    df_news["sentiment_pos"] = [s["positive"] for s in sentiments]
    df_news["sentiment_neg"] = [s["negative"] for s in sentiments]
    df_news["sentiment_neu"] = [s["neutral"] for s in sentiments]
    df_news["sentiment_score"] = [compute_sentiment_score(s) for s in sentiments]

    # 날짜 파싱 (YYYY.MM.DD HH:MM → date)
    df_news["date"] = pd.to_datetime(
        df_news["날짜"].str.strip().str[:10], format="%Y.%m.%d", errors="coerce"
    )

    # 날짜 파싱에 실패한 뉴스는 집계에서 제외
    df_news = df_news.dropna(subset=["date"])

    # 종목코드·날짜별 감성 점수 집계
    daily_sentiment = df_news.groupby(["종목코드", "date"]).agg(
        sentiment_mean=("sentiment_score", "mean"),
        sentiment_std=("sentiment_score", "std"),
        news_count=("sentiment_score", "count"),
    ).reset_index()

    daily_sentiment["종목코드"] = daily_sentiment["종목코드"].astype(str).str.zfill(6)
    daily_sentiment["sentiment_std"] = daily_sentiment["sentiment_std"].fillna(0)

    # 감성 시계열 파생 피처 추가
    daily_sentiment = add_sentiment_time_features(daily_sentiment)

    output_path = os.path.join(DATA_DIR, "sentiment.csv")
    daily_sentiment.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"저장 완료: {output_path} ({len(daily_sentiment)}행)")
    
    
if __name__ == "__main__":
    main()
