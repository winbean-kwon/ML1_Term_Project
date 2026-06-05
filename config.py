"""
프로젝트 공통 설정
"""
import os

# 경로
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "models")

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# 데이터 기간
START_DATE = "20060101"
END_DATE = "20260401"

# 종목 코드 파일
STOCK_CODES_PATH = os.path.join(BASE_DIR, "kospi_stock_codes.csv")
NEWS_PATH = os.path.join(BASE_DIR, "kospi_news.csv")

# 시계열 윈도우 크기
WINDOW_SIZE = 20

# 뉴스 데이터 기간 (학습/테스트 구간 결정에 사용)
NEWS_START_DATE = "2025-04-01"   # sentiment.csv 시작 날짜
NEWS_END_DATE   = "2026-04-01"   # sentiment.csv 종료 날짜 (= END_DATE)
TRAIN_MONTHS    = 8              # 뉴스 기간 중 앞 8개월 → train, 나머지 4개월 → test

# 학습/테스트 분할 비율 (뉴스 기간 split 미사용 시 fallback)
TEST_RATIO = 0.2

# 랜덤 시드
SEED = 42

# 3일 후 수익률이 ±1.5% 이내인 경우는 노이즈성 횡보 구간으로 간주 (0.01 → 0.015)
TARGET_THRESHOLD = 0.015
