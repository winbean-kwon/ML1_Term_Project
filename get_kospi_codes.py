"""
KOSPI 전체 종목 코드를 한국거래소(KIND)에서 가져와 CSV로 저장하는 스크립트
"""

import requests
import pandas as pd
from io import BytesIO


def get_kospi_stock_codes():
    """한국거래소 KIND에서 KOSPI 상장 종목 목록을 가져옵니다."""
    url = (
        "https://kind.krx.co.kr/corpgeneral/corpList.do"
        "?method=download&searchType=13&marketType=stockMkt"
    )
    headers = {"User-Agent": "Mozilla/5.0"}
    response = requests.get(url, headers=headers)
    response.raise_for_status()

    df = pd.read_html(BytesIO(response.content), encoding="euc-kr")[0]

    # 종목코드를 6자리 문자열로 변환
    df["종목코드"] = df["종목코드"].astype(str).str.zfill(6)

    return df


if __name__ == "__main__":
    print("KOSPI 종목 코드를 가져오는 중...")
    df = get_kospi_stock_codes()
    import os
    output_path = os.path.join(os.path.dirname(__file__), "kospi_stock_codes.csv")
    df.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"총 {len(df)}개 종목 저장 완료: {output_path}")
    print(df.head(10).to_string(index=False))
