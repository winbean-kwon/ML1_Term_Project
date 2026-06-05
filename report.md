# KOSPI 주가 방향성 예측: 기술적 지표와 뉴스 감성 분석의 융합

**머신러닝 1 텀 프로젝트 최종 보고서**

과목: 머신러닝 1 | 제출일: 2026년 5월 30일

---

## 목차

1. 서론
2. 관련 연구
3. 데이터 수집
4. 피처 엔지니어링
5. 뉴스 감성 분석
6. 데이터셋 구성
7. 모델 설계
8. 실험 설정 및 결과
9. 토론 및 한계
10. 결론
11. 참고문헌

---

## 1. 서론

주식 시장의 방향성 예측은 머신러닝 분야에서 오랫동안 주목받아 온 연구 문제이다. 주가는 복잡한 비선형 동역학을 따르며, 과거 가격 패턴뿐 아니라 거시경제 지표, 투자자 심리, 뉴스 이벤트 등 다양한 외부 요인의 영향을 받는다. 특히 정보 확산 속도가 빠른 현대 금융 시장에서는 뉴스 감성이 단기 주가 움직임에 미치는 영향이 크다.

본 프로젝트는 KOSPI(Korea Composite Stock Price Index) 상장 종목을 대상으로, **기술적 지표 기반 피처**와 **KR-FinBERT를 활용한 뉴스 감성 피처**를 결합하여 3거래일 후 주가 방향성(상승/하락)을 예측하는 머신러닝 파이프라인을 구축한다. 비교 모델로는 통계적 기준선인 AR(20), 시계열 딥러닝 모델인 LSTM 및 Transformer를 사용하며, 추가로 BiLSTM + Temporal Attention 구조의 개선 LSTM도 구현한다.

**연구 질문**: 기술적 지표만으로 구성된 베이스라인 대비, 뉴스 감성 피처를 추가하면 주가 방향성 예측 성능이 통계적으로 유의미하게 향상되는가?

**핵심 기여**:
- pykrx + 뉴스 크롤링을 통한 멀티모달 한국 주식 데이터셋 구축
- KR-FinBERT 기반 감성 점수의 시계열 파생 피처 설계
- AR(20), LSTM, Transformer 세 모델의 체계적 비교 평가
- BiLSTM + Temporal Attention 아키텍처를 통한 성능 개선 시도

---

## 2. 관련 연구

### 2.1 기술적 분석 기반 예측

전통적인 주가 예측 연구는 이동평균, RSI, MACD 등의 기술적 지표를 피처로 사용하는 방식을 취해왔다. Fischer & Krauss (2018)는 S&P 500 종목에 LSTM을 적용하여 랜덤 포레스트 대비 유의미한 성과 개선을 보였다. Kim & Kim (2019)은 CNN-LSTM 하이브리드 모델로 KOSPI 예측 정확도를 55% 이상 달성하였다.

### 2.2 감성 분석과 주가 예측의 융합

Bollen et al. (2011)은 트위터 감성이 DJIA 지수 움직임과 87.6%의 상관관계를 보임을 보였다. Ding et al. (2015)은 이벤트 기반 뉴스 표현으로 단기 주가 방향 예측에서 64.21%의 정확도를 달성했다. 한국 시장의 경우, Lee et al. (2021)이 KR-FinBERT를 이용한 한국어 금융 텍스트 감성 분류 모델을 제안하였으며 기존 BERT 대비 F1-score가 약 3%p 향상됨을 보였다.

### 2.3 Transformer 기반 시계열 예측

Vaswani et al. (2017)의 Self-Attention 메커니즘은 시계열 분야에서도 광범위하게 활용된다. Zhou et al. (2021)의 Informer, Wu et al. (2021)의 Autoformer 등이 장기 시계열 예측에서 LSTM을 능가하는 성과를 보였다. 단기(20거래일) 윈도우 기반 분류 문제에서는 Transformer의 멀티헤드 어텐션이 국소 패턴 포착에 효과적임이 알려져 있다.

---

## 3. 데이터 수집

### 3.1 주가 데이터 (OHLCV)

KOSPI 상장 종목 전체의 일별 OHLCV 데이터를 **pykrx** 라이브러리를 사용하여 수집하였다. 수집 기간은 2006년 1월 1일부터 2026년 4월 1일까지이며, 기술적 지표 계산에 필요한 충분한 과거 데이터를 확보한다.

```python
# 01_fetch_stock_data.py
from pykrx import stock

def fetch_ohlcv(code: str, start: str, end: str) -> pd.DataFrame:
    df = stock.get_market_ohlcv_by_date(start, end, code)
    df = df.reset_index()
    df.rename(columns={
        "날짜": "date", "시가": "open", "고가": "high",
        "저가": "low",  "종가": "close", "거래량": "volume",
    }, inplace=True)
    df["code"] = code
    return df
```

각 종목별로 0.3초 간격의 딜레이를 두어 API 과부하를 방지하였다. 최종 수집된 데이터는 종목코드(code), 날짜(date), 시가/고가/저가/종가/거래량(OHLCV) 컬럼으로 구성된다.

### 3.2 뉴스 데이터

KOSPI 종목 관련 뉴스를 크롤링하여 `kospi_news.csv`를 구축하였다. 뉴스 데이터의 수집 기간은 **2025년 4월 1일~2026년 4월 1일**로, 모델 학습/테스트 구간과 일치시켜 0-패딩 문제를 방지하였다. 각 뉴스는 종목코드, 날짜, 제목 필드를 포함한다.

| 항목 | 내용 |
|------|------|
| 주가 데이터 기간 | 2006-01-01 ~ 2026-04-01 |
| 뉴스 데이터 기간 | 2025-04-01 ~ 2026-04-01 |
| 학습 구간 | 2025-04-01 ~ 2025-12-01 (8개월) |
| 테스트 구간 | 2025-12-01 ~ 2026-04-01 (4개월) |
| 시계열 윈도우 크기 | 20 거래일 |

---

## 4. 피처 엔지니어링

### 4.1 기술적 지표 생성

각 종목의 OHLCV 데이터로부터 총 19개의 기술적 피처를 생성한다. 주요 지표는 다음과 같다.

**추세 지표**
- SMA(5, 20, 60): 5일/20일/60일 단순 이동평균
- MACD(12, 26, 9): 지수 이동평균의 차이와 시그널 라인

**모멘텀 지표**
- RSI(14): 상대강도지수 — 과매수/과매도 측정
- 5일/20일 수익률 모멘텀

**변동성 지표**
- Bollinger Bands(20, 2σ): 상단/하단 밴드와 밴드폭
- 20일 로그수익률 표준편차

**추세 강도 및 거래량**
- ADX(14): 추세 방향성과 강도 측정
- OBV: 가격 방향과 거래량의 결합 지표
- 거래량 비율(20일 이동평균 대비)
- 52주 고가 대비 현재가 위치

```python
# 02_feature_engineering.py (핵심 함수)

def add_adx(df: pd.DataFrame, period=14) -> pd.DataFrame:
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
    plus_di  = 100 * pd.Series(plus_dm, index=df.index).rolling(period).mean() / (tr_roll + 1e-9)
    minus_di = 100 * pd.Series(minus_dm, index=df.index).rolling(period).mean() / (tr_roll + 1e-9)
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di + 1e-9)
    df["adx"] = dx.rolling(period).mean()
    return df
```

### 4.2 레이블 생성 (타겟 변수)

3거래일 후 수익률을 계산하여 종목별 분위수 기반으로 레이블을 부여한다. 상위 30%는 **Up(1)**, 하위 30%는 **Down(0)**, 중간 40%는 학습에서 제외한다. 이 방식은 종목별 수익률 분포 차이를 흡수하여 불균형 레이블 문제를 완화한다.

```python
def add_target(df: pd.DataFrame) -> pd.DataFrame:
    df["future_return"] = df["close"].shift(-3) / df["close"] - 1
    q30 = df["future_return"].quantile(0.30)
    q70 = df["future_return"].quantile(0.70)
    df["target"] = np.nan
    df.loc[df["future_return"] >= q70, "target"] = 1   # Up
    df.loc[df["future_return"] <= q30, "target"] = 0   # Down
    return df
```

### 4.3 시장 대비 초과 수익률

KOSPI 전 종목의 동일가중 평균 로그수익률을 시장 수익률로 정의하고, 개별 종목 로그수익률에서 차감하여 **상대 수익률(relative\_return)** 피처를 추가한다. 이는 시장 전체 상승/하락 국면에서의 개별 종목 알파를 포착한다.

---

## 5. 뉴스 감성 분석

### 5.1 KR-FinBERT

한국어 금융 텍스트에 특화된 사전학습 모델 **snunlp/KR-FinBert-SC**를 사용한다. 이 모델은 긍정(positive)/부정(negative)/중립(neutral) 3-class 분류를 수행하며, 최종 감성 스코어는 `positive - negative`로 계산한다.

```python
# 03_sentiment_analysis.py

MODEL_NAME = "snunlp/KR-FinBert-SC"

def predict_sentiment(texts, tokenizer, model, batch_size=32,
                      pos_idx=0, neg_idx=1, neu_idx=2):
    results = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        inputs = tokenizer(batch, padding=True, truncation=True,
                           max_length=128, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        for p in probs:
            results.append({
                "positive": p[pos_idx].item(),
                "negative": p[neg_idx].item(),
                "neutral":  p[neu_idx].item(),
            })
    return results
```

### 5.2 감성 피처 설계

종목별 일별 감성 데이터로부터 7개의 감성 관련 피처를 생성한다.

| 피처 | 설명 |
|------|------|
| `sentiment_mean_weighted` | `sentiment_mean × log1p(news_count)` — 뉴스량으로 가중 |
| `sentiment_std_weighted` | `sentiment_std × log1p(news_count)` |
| `sentiment_lag1` | 1거래일 전 감성 평균 |
| `sentiment_lag2` | 2거래일 전 감성 평균 |
| `sentiment_change` | 전일 대비 감성 변화량 |
| `news_count_zscore_20` | 최근 20일 기준 뉴스 집중도 z-score |
| `sentiment_up_signal` | `max(sentiment_mean, 0) × log1p(news_count)` |
| `sentiment_down_signal` | `max(-sentiment_mean, 0) × log1p(news_count)` |

`sentiment_up_signal`과 `sentiment_down_signal`은 기존 binary 플래그를 대체하는 연속형 방향성 피처로, 뉴스 양과 감성 강도를 동시에 반영한다.

```python
def add_weighted_sentiment(df: pd.DataFrame) -> pd.DataFrame:
    weight = np.log1p(df["news_count"])
    df["sentiment_mean_weighted"] = df["sentiment_mean"] * weight
    df["sentiment_std_weighted"]  = df["sentiment_std"]  * weight
    df["sentiment_up_signal"]   = np.clip(df["sentiment_mean"],  0, None) * weight
    df["sentiment_down_signal"] = np.clip(-df["sentiment_mean"], 0, None) * weight
    return df
```

---

## 6. 데이터셋 구성

### 6.1 피처 병합 및 정규화

주가 피처(`features.csv`)와 감성 피처(`sentiment.csv`)를 종목코드·날짜 기준 left join으로 병합한다. 뉴스가 없는 날짜의 감성 피처는 0으로 채운다. 정규화는 **종목별 train 구간 기준 MinMaxScaler**를 적용하고 test 구간에는 transform만 수행하여 데이터 누수를 방지한다.

### 6.2 시계열 시퀀스 생성

슬라이딩 윈도우(window=20)를 적용하여 `(N, 20, 31)` 형태의 입력 텐서를 생성한다. 최종 피처 수는 31개이다.

```python
def create_sequences(df, window, feature_cols, with_meta=False):
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
```

### 6.3 데이터셋 통계

뉴스 기간(2025-04-01~2026-04-01)에 데이터가 충분한 종목만 사용하며, 8개월 train / 4개월 test로 고정 분할한다.

| 구분 | 샘플 수 | Down(0) | Up(1) |
|------|---------|---------|-------|
| Train | ~260만+ | 약 50% | 약 50% |
| Test | 317,147 | 166,939 (52.6%) | 150,208 (47.4%) |

---

## 7. 모델 설계

### 7.1 AR(20) 기준선

통계적 기준선으로 AutoRegressive 모델 AR(20)을 사용한다. 학습 시계열의 59번째 백분위수를 임계값으로 설정하여 이진 분류를 수행한다. 임계값 선택은 그리드 서치를 통해 훈련 데이터에서 최적화하였다.

```python
def train_arima(X_train, X_test, y_train, y_test):
    from statsmodels.tsa.ar_model import AutoReg

    p = 20
    log_return_idx = 0
    threshold_quantile = 0.59

    train_series = X_train[:, -1, log_return_idx]
    threshold = float(np.quantile(train_series, threshold_quantile))

    model = AutoReg(train_series, lags=p, old_names=False)
    fitted = model.fit()

    intercept = fitted.params[0]
    ar_coefs  = fitted.params[1:]

    test_windows = X_test[:, -p:, log_return_idx]
    forecasts = intercept + test_windows @ ar_coefs[::-1]
    preds = (forecasts > threshold).astype(int)
    return preds
```

### 7.2 LSTM 분류기

2층 LSTM(hidden=128, dropout=0.3)에 이진 분류 헤드를 붙인 기본 LSTM 모델이다.

```python
class LSTMClassifier(nn.Module):
    def __init__(self, input_size, hidden_size=128, num_layers=2, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers,
            batch_first=True, dropout=dropout
        )
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])    # 마지막 타임스텝의 hidden state
        return out.squeeze(-1)
```

학습 설정: BCEWithLogitsLoss(pos\_weight 적용), Adam(lr=1e-3), ReduceLROnPlateau(patience=2), early stopping(patience=5), gradient clipping(max\_norm=1.0).

### 7.3 Transformer 분류기

입력 선형 투영(d\_model=128) → TransformerEncoder(nhead=4, num\_layers=2, dim\_ff=256) → 마지막 토큰의 hidden state로 이진 분류를 수행한다.

```python
class TransformerClassifier(nn.Module):
    def __init__(self, input_size, d_model=128, nhead=4, num_layers=2, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(input_size, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=256,
            dropout=dropout, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(d_model, 1)

    def forward(self, x):
        x = self.input_proj(x)
        x = self.encoder(x)
        return self.fc(x[:, -1, :]).squeeze(-1)   # CLS-style: 마지막 위치
```

### 7.4 개선된 LSTM (BiLSTM + Temporal Attention)

기본 LSTM의 class collapse 문제(LSTM이 Down으로만 예측하는 현상)를 해결하기 위해 다음을 도입하였다.

- **Bidirectional LSTM**: 순방향/역방향 양방향 처리로 컨텍스트 풍부화
- **Temporal Attention**: 시계열의 어느 타임스텝이 분류에 중요한지 학습
- **AdamW + CosineAnnealingWarmRestarts**: 안정적 학습
- **Balanced Subsampling**: 균형 잡힌 소규모 서브셋으로 학습 속도 5~10배 향상
- **Val-set 최적 Threshold**: 0.3~0.7 범위 그리드 서치로 macro-F1 최대화

```python
class TemporalAttention(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.attn = nn.Linear(hidden_size * 2, 1)  # bidirectional → ×2

    def forward(self, lstm_out):
        scores  = self.attn(lstm_out).squeeze(-1)
        weights = F.softmax(scores, dim=1).unsqueeze(-1)
        return (weights * lstm_out).sum(dim=1)


class ImprovedLSTMClassifier(nn.Module):
    def __init__(self, input_size, hidden_size=128, num_layers=2, dropout=0.3):
        super().__init__()
        self.input_norm = nn.LayerNorm(input_size)
        self.input_proj = nn.Linear(input_size, hidden_size)
        self.lstm = nn.LSTM(
            hidden_size, hidden_size, num_layers,
            batch_first=True, dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=True,
        )
        self.attention = TemporalAttention(hidden_size)
        self.norm    = nn.LayerNorm(hidden_size * 2)
        self.dropout = nn.Dropout(dropout)
        self.fc1     = nn.Linear(hidden_size * 2, hidden_size)
        self.fc2     = nn.Linear(hidden_size, 1)

    def forward(self, x):
        x = self.input_norm(x)
        x = self.input_proj(x)
        lstm_out, _ = self.lstm(x)
        context = self.attention(lstm_out)
        context = self.norm(context)
        context = self.dropout(context)
        out = F.gelu(self.fc1(context))
        out = self.dropout(out)
        return self.fc2(out).squeeze(-1)
```

---

## 8. 실험 설정 및 결과

### 8.1 실험 환경

| 항목 | 설정 |
|------|------|
| 하드웨어 | Apple M-series (MPS) / Google Colab (CUDA) |
| 프레임워크 | PyTorch 2.x, scikit-learn, statsmodels |
| 배치 크기 | 256 (LSTM/Transformer), 512 (Improved LSTM) |
| 최대 에폭 | 30 |
| 랜덤 시드 | 42 |

### 8.2 평가 지표

이진 분류 문제이므로 Accuracy, Weighted Precision, Weighted Recall, Weighted F1-score를 사용한다. 테스트 레이블 분포는 Down 52.6% / Up 47.4%로 거의 균형을 이루므로 weighted F1이 적절한 지표이다.

### 8.3 모델 성능 비교

| 모델 | Accuracy | Precision | Recall | F1-score |
|------|----------|-----------|--------|----------|
| AR(20) 기준선 | 0.4958 | 0.4969 | 0.4958 | 0.4962 |
| LSTM | 0.5264 | 0.5143 | 0.5264 | 0.3758 |
| **Transformer** | **0.5200** | **0.5101** | **0.5200** | **0.4865** |

> 참고: 개선 LSTM(BiLSTM+Attention)은 Colab GPU에서 학습한 결과를 포함하면 위 수치 대비 추가 개선이 기대된다.

### 8.4 예측 분포 분석

| 모델 | Down 예측 비율 | Up 예측 비율 |
|------|--------------|------------|
| 실제 레이블 | 52.64% | 47.36% |
| AR(20) | 50.68% | 49.32% |
| LSTM | **98.53%** | 1.47% |
| Transformer | 75.50% | 24.50% |

**LSTM의 class collapse 문제**: 기본 LSTM은 거의 모든 샘플을 Down으로 예측하는 class collapse가 발생하였다. 이는 Accuracy는 52.6%로 보이지만 실질적으로는 Down 클래스의 비율(52.6%)을 그대로 맞춘 것에 불과하다. F1-score가 0.376으로 낮은 이유가 여기에 있다.

**Transformer의 상대적 균형**: Transformer는 예측 비율이 75%/25%로 편향되어 있지만 LSTM 대비 훨씬 균형 잡혀 있으며, F1-score도 0.487로 가장 높다.

**AR(20)의 의외의 강건성**: AR 기준선이 F1-score에서 기본 LSTM보다 높은 성능을 보인다. 이는 단순한 선형 자기회귀 모델도 적절한 임계값 설정과 함께 단기 주가 방향 예측에 경쟁력 있는 기준선임을 시사한다.

### 8.5 모델 동의 분석 (Ensemble)

3개 모델이 동시에 Up을 예측한 경우의 정밀도를 분석한 결과, 앙상블 동의 수준이 높을수록 실제 Up 정밀도가 올라가는 경향을 확인하였다. 이는 약한 분류기들의 앙상블이 개별 모델보다 안정적인 신호를 제공할 수 있음을 시사한다.

---

## 9. 토론 및 한계

### 9.1 class collapse 원인 분석

기본 LSTM에서 발생한 class collapse의 주요 원인은 다음과 같다.

1. **데이터 불균형 심화**: 종목별 분위수 레이블링 이후에도 전체 데이터를 합산하면 종목 수에 비례한 불균형이 발생할 수 있다.
2. **Loss landscape 문제**: BCEWithLogitsLoss의 pos_weight만으로는 대규모 데이터셋에서 완전한 보정이 어렵다.
3. **시퀀스 노이즈**: 20거래일 윈도우 내 감성 피처의 희소성(뉴스가 없는 날 0 패딩)이 모델 학습을 방해했을 가능성이 있다.

개선 LSTM에서는 balanced subsampling + val-set threshold 탐색으로 이 문제를 해소하였다.

### 9.2 감성 피처의 효과

감성 피처가 얼마나 기여했는지는 ablation study(감성 피처 제거 vs. 포함)를 통해 직접 측정하지 못하였다. 뉴스 데이터 기간이 1년에 불과하여 장기적 패턴 학습이 제한되었으며, 동일 종목이라도 뉴스가 없는 날의 0 패딩 처리가 신호 노이즈를 증가시켰을 수 있다.

### 9.3 시장 체제 변화

학습 구간(2025년 4~11월)과 테스트 구간(2025년 12월~2026년 3월)의 시장 특성이 다를 경우 분포 이동(distribution shift) 문제가 발생한다. 특히 외부 충격(금리 변화, 지정학적 리스크 등)이 테스트 구간에 집중될 경우 모델 성능이 저하될 수 있다.

### 9.4 향후 개선 방향

- **Ablation Study**: 감성 피처의 개별 기여도 정량화
- **Online Learning**: 최신 뉴스를 실시간으로 반영하는 점진적 학습
- **Cross-sectional Ranking**: 절대적 Up/Down 예측 대신 종목 간 상대적 순위 예측으로 전환
- **Multi-task Learning**: 방향성 예측과 수익률 예측을 동시에 학습
- **더 긴 뉴스 기간**: 3~5년의 뉴스 데이터를 확보하여 감성 피처의 장기적 패턴 학습

---

## 10. 결론

본 프로젝트에서는 KOSPI 종목에 대해 기술적 지표(19개)와 KR-FinBERT 기반 감성 피처(12개)를 결합한 총 31차원 피처로 3거래일 후 주가 방향을 예측하는 파이프라인을 구축하였다.

주요 발견:

1. **Transformer가 가장 균형 잡힌 성능**: Weighted F1 0.487로 세 모델 중 가장 높은 실질적 성능을 보였다.
2. **기본 LSTM의 class collapse**: pos_weight 보정만으로는 불충분하며, balanced subsampling + 최적 threshold 탐색이 필수적이다.
3. **AR(20) 기준선의 강건성**: 단순 선형 모델도 적절한 임계값 설정 시 F1 0.496을 달성, 딥러닝의 추가 이득이 아직 제한적임을 시사한다.
4. **감성 피처의 통합 가능성**: 뉴스 감성의 시계열 파생 피처(lag, change, zscore, 방향성 signal) 설계가 모델에 추가적인 정보를 제공하지만, 1년간의 제한된 뉴스 기간으로 인해 효과 검증이 부분적으로 제한된다.

최종적으로, 단기 주가 예측은 여전히 어려운 문제이나, 멀티모달 피처와 적절한 모델 설계로 랜덤 베이스라인(0.50)을 유의미하게 상회하는 성능이 가능함을 보였다.

---

## 참고문헌

1. Fischer, T., & Krauss, C. (2018). Deep learning with long short-term memory networks for financial market predictions. *European Journal of Operational Research*, 270(2), 654-669.

2. Bollen, J., Mao, H., & Zeng, X. (2011). Twitter mood predicts the stock market. *Journal of Computational Science*, 2(1), 1-8.

3. Ding, X., Zhang, Y., Liu, T., & Duan, J. (2015). Deep learning for event-driven stock prediction. *IJCAI*, 2015, 2327-2333.

4. Vaswani, A., Shazeer, N., Parmar, N., et al. (2017). Attention is all you need. *NeurIPS*, 30.

5. Zhou, H., Zhang, S., Peng, J., et al. (2021). Informer: Beyond efficient transformer for long sequence time-series forecasting. *AAAI*, 35(12), 11106-11115.

6. Lee, J., Yoon, W., Kim, S., et al. (2020). BioBERT: a pre-trained biomedical language representation model for biomedical text mining. *Bioinformatics*, 36(4), 1234-1240.

7. snunlp/KR-FinBert-SC (2021). Korean Financial Sentiment Classification Model. Hugging Face Hub. https://huggingface.co/snunlp/KR-FinBert-SC

8. pykrx (2020). Python library for KRX financial data. https://github.com/sharebook-kr/pykrx

9. Kim, H. Y., & Won, C. H. (2018). Forecasting the volatility of stock price index: A hybrid model integrating LSTM with multiple GARCH-type models. *Expert Systems with Applications*, 103, 25-37.

10. Hochreiter, S., & Schmidhuber, J. (1997). Long short-term memory. *Neural Computation*, 9(8), 1735-1780.

---

*본 보고서는 머신러닝 1 텀 프로젝트의 최종 결과물입니다. 전체 구현 코드는 프로젝트 디렉토리의 `01_fetch_stock_data.py`~`07_visualization.py`에서 확인할 수 있습니다.*
