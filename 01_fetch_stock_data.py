"""
KOSPI 종목별 OHLCV 데이터를 pykrx로 수집하여 저장

사용법:
    python 01_fetch_stock_data.py
    python 01_fetch_stock_data.py --code 005930
"""
import argparse
import pandas as pd
from pykrx import stock
import time
import os
from config import BASE_DIR, DATA_DIR, START_DATE, END_DATE, STOCK_CODES_PATH


def fetch_ohlcv(code: str, start: str, end: str) -> pd.DataFrame:
    """종목 하나의 OHLCV 데이터를 가져옵니다."""
    df = stock.get_market_ohlcv_by_date(start, end, code)
    df = df.reset_index()
    df.rename(columns={
        "날짜": "date",
        "시가": "open",
        "고가": "high",
        "저가": "low",
        "종가": "close",
        "거래량": "volume",
    }, inplace=True)
    df["code"] = code
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--code", type=str, default=None)
    args = parser.parse_args()

    if args.code:
        codes = [args.code.zfill(6)]
    else:
        df_codes = pd.read_csv(STOCK_CODES_PATH, dtype={"종목코드": str})
        codes = df_codes["종목코드"].str.zfill(6).tolist()

    all_data = []
    total = len(codes)

    for i, code in enumerate(codes, 1):
        print(f"[{i}/{total}] {code} OHLCV 수집 중...", end=" ", flush=True)
        try:
            df = fetch_ohlcv(code, START_DATE, END_DATE)
            all_data.append(df)
            print(f"{len(df)}행")
        except Exception as e:
            print(f"실패: {e}")
        time.sleep(0.3)

    if all_data:
        result = pd.concat(all_data, ignore_index=True)
        output_path = os.path.join(DATA_DIR, "ohlcv.csv")
        result.to_csv(output_path, index=False, encoding="utf-8-sig")
        print(f"\n저장 완료: {output_path} ({len(result)}행)")


if __name__ == "__main__":
    main()
