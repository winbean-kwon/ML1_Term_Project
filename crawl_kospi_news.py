"""
KOSPI 전 종목의 네이버 금융 뉴스를 크롤링하여 CSV로 저장하는 스크립트

사용법:
    python crawl_kospi_news.py                  # 전체 종목, 최근 1년 뉴스
    python crawl_kospi_news.py --code 005930    # 특정 종목만
    python crawl_kospi_news.py --days 180       # 최근 180일
    python crawl_kospi_news.py --max-pages 5    # 종목당 최대 5페이지
"""

import requests
from lxml import html
import pandas as pd
import time
import os
import argparse
from datetime import datetime, timedelta


BASE_URL = "https://finance.naver.com/item/news_news.naver"
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def get_headers(code):
    return {
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        ),
        "Referer": f"https://finance.naver.com/item/news.naver?code={code}",
    }


def get_last_page(tree):
    """페이지네이션에서 마지막 페이지 번호를 추출합니다."""
    last_link = tree.xpath('//td[@class="pgRR"]/a/@href')
    if last_link:
        # /item/news_news.naver?code=005930&page=200&clusterId=
        for part in last_link[0].split("&"):
            if part.startswith("page="):
                return int(part.split("=")[1])
    # pgRR이 없으면 현재 보이는 페이지 번호 중 최대값
    page_links = tree.xpath('//table[@class="Nnavi"]//td/a/text()')
    nums = [int(p) for p in page_links if p.strip().isdigit()]
    return max(nums) if nums else 1


def parse_news_date(date_str):
    """날짜 문자열을 datetime으로 변환합니다."""
    date_str = date_str.strip()
    for fmt in ("%Y.%m.%d %H:%M", "%Y.%m.%d"):
        try:
            return datetime.strptime(date_str, fmt)
        except ValueError:
            continue
    return None


def parse_news_page(tree, code):
    """한 페이지의 뉴스 목록을 파싱합니다."""
    results = []
    tbody = tree.xpath("/html/body/div/table[1]/tbody")
    if not tbody:
        return results

    rows = tbody[0].xpath("./tr")
    for row in rows:
        row_class = (row.get("class") or "").strip()
        # 관련뉴스 목록(relation_lst)은 건너뛰기
        if "relation_lst" in row_class:
            continue

        a_tag = row.xpath('.//td[@class="title"]/a')
        if not a_tag:
            continue

        title = a_tag[0].text_content().strip()
        href = a_tag[0].get("href", "")

        info = ""
        date = ""
        for td in row.xpath("./td"):
            td_class = (td.get("class") or "").strip()
            if td_class == "info":
                info = td.text_content().strip()
            elif td_class == "date":
                date = td.text_content().strip()

        # article_id, office_id 추출
        article_id = ""
        office_id = ""
        if href:
            for part in href.split("&"):
                if "article_id=" in part:
                    article_id = part.split("=")[-1]
                elif "office_id=" in part:
                    office_id = part.split("=")[-1]

        news_url = ""
        if office_id and article_id:
            news_url = f"https://n.news.naver.com/mnews/article/{office_id}/{article_id}"

        is_related = "relation_tit" in row_class

        results.append({
            "종목코드": code,
            "제목": title,
            "정보제공": info,
            "날짜": date,
            "뉴스URL": news_url,
            "관련뉴스여부": is_related,
        })

    return results


def crawl_stock_news(code, max_pages=None, cutoff_date=None, session=None):
    """특정 종목의 전체 뉴스를 크롤링합니다.
    cutoff_date가 지정되면 해당 날짜 이전 뉴스가 나오면 중단합니다.
    """
    if session is None:
        session = requests.Session()

    headers = get_headers(code)
    all_news = []

    # 첫 페이지에서 마지막 페이지 확인
    url = f"{BASE_URL}?code={code}&page=1&clusterId="
    try:
        resp = session.get(url, headers=headers, timeout=10)
        resp.encoding = "euc-kr"
    except requests.RequestException as e:
        print(f"  [오류] {code} 첫 페이지 요청 실패: {e}")
        return all_news

    tree = html.fromstring(resp.text)
    total_pages = get_last_page(tree)

    if max_pages:
        total_pages = min(total_pages, max_pages)

    # 첫 페이지 파싱
    news = parse_news_page(tree, code)
    filtered, hit_cutoff = _filter_by_date(news, cutoff_date)
    all_news.extend(filtered)

    if not news or hit_cutoff:
        return all_news

    # 나머지 페이지 크롤링
    for page in range(2, total_pages + 1):
        url = f"{BASE_URL}?code={code}&page={page}&clusterId="
        try:
            resp = session.get(url, headers=headers, timeout=10)
            resp.encoding = "euc-kr"
            tree = html.fromstring(resp.text)
            news = parse_news_page(tree, code)
            if not news:
                break
            filtered, hit_cutoff = _filter_by_date(news, cutoff_date)
            all_news.extend(filtered)
            if hit_cutoff:
                break
        except requests.RequestException as e:
            print(f"  [오류] {code} page {page} 요청 실패: {e}")
            continue

        time.sleep(0.3)  # 서버 부하 방지

    return all_news


def _filter_by_date(news_list, cutoff_date):
    """cutoff_date 이전 뉴스를 제거하고, 기준일에 도달했는지 여부를 반환합니다.
    뉴스는 최신순 정렬이므로, cutoff 이전 뉴스가 나오면 이후는 전부 오래된 것입니다.
    """
    if cutoff_date is None:
        return news_list, False

    filtered = []
    for item in news_list:
        dt = parse_news_date(item["날짜"])
        if dt and dt < cutoff_date:
            return filtered, True
        filtered.append(item)

    return filtered, False



def main():
    parser = argparse.ArgumentParser(description="KOSPI 종목 네이버 금융 뉴스 크롤러")
    parser.add_argument("--code", type=str, help="특정 종목코드만 크롤링 (예: 005930)")
    parser.add_argument("--max-pages", type=int, default=None, help="종목당 최대 크롤링 페이지 수")
    parser.add_argument("--days", type=int, default=365, help="최근 N일 이내 뉴스만 크롤링 (기본: 365)")
    parser.add_argument("--output", type=str, default="kospi_news.csv", help="출력 CSV 파일명")
    args = parser.parse_args()

    output_path = os.path.join(SCRIPT_DIR, args.output)
    session = requests.Session()
    cutoff_date = datetime.now() - timedelta(days=args.days)

    if args.code:
        # 특정 종목만
        code = args.code.zfill(6)
        codes = [code]
        names = {code: code}
    else:
        # 전체 KOSPI 종목
        codes_path = os.path.join(SCRIPT_DIR, "kospi_stock_codes.csv")
        if not os.path.exists(codes_path):
            print("kospi_stock_codes.csv가 없습니다. get_kospi_codes.py를 먼저 실행하세요.")
            return
        df_codes = pd.read_csv(codes_path, dtype={"종목코드": str})
        df_codes["종목코드"] = df_codes["종목코드"].str.zfill(6)
        codes = df_codes["종목코드"].tolist()
        names = dict(zip(df_codes["종목코드"], df_codes["회사명"]))

    total = len(codes)
    all_news = []
    start_time = datetime.now()

    print(f"크롤링 시작: {total}개 종목, 시작 시간: {start_time.strftime('%H:%M:%S')}")
    print(f"기간: 최근 {args.days}일 ({cutoff_date.strftime('%Y-%m-%d')} ~ 오늘)")
    print(f"출력 파일: {output_path}")
    print("-" * 60)

    for i, code in enumerate(codes, 1):
        name = names.get(code, code)
        print(f"[{i}/{total}] {name}({code}) 크롤링 중...", end=" ", flush=True)

        news = crawl_stock_news(code, max_pages=args.max_pages, cutoff_date=cutoff_date, session=session)
        # 종목명 추가
        for n in news:
            n["종목명"] = name
        all_news.extend(news)

        print(f"뉴스 {len(news)}건")

        # 50종목마다 중간 저장
        if i % 50 == 0 and all_news:
            df_temp = pd.DataFrame(all_news)
            df_temp.to_csv(output_path, index=False, encoding="utf-8-sig")
            print(f"  >> 중간 저장 완료 (누적 {len(all_news)}건)")

        time.sleep(0.5)  # 종목 간 딜레이

    # 최종 저장
    if all_news:
        df = pd.DataFrame(all_news)
        # 컬럼 순서 정리
        col_order = ["종목코드", "종목명", "제목", "정보제공", "날짜", "뉴스URL", "관련뉴스여부"]
        df = df[[c for c in col_order if c in df.columns]]
        df.to_csv(output_path, index=False, encoding="utf-8-sig")

    end_time = datetime.now()
    elapsed = end_time - start_time
    print("-" * 60)
    print(f"크롤링 완료: 총 {len(all_news)}건, 소요 시간: {elapsed}")
    print(f"저장 완료: {output_path}")


if __name__ == "__main__":
    main()
