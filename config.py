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

# 학습/테스트 분할 비율
TEST_RATIO = 0.2

# 랜덤 시드
SEED = 42

# 다음 날 수익률이 ±0.5% 이내인 경우는 노이즈성 횡보 구간으로 간주
TARGET_THRESHOLD = 0.005
