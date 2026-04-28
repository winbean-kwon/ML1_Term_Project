"""
KOSPI 종목별 네이버 금융 뉴스 목록 + 본문을 크롤링하여 CSV로 저장하는 스크립트

kospi_stock_codes.csv의 종목코드를 순회하며:
1) 종목별 뉴스 목록을 크롤링 (날짜 기준 필터)
2) 각 뉴스 본문을 크롤링
3) 결과를 kospi_news_content.csv에 저장

사용법:
    python crawl_kospi_news_content.py                # 전체 종목, 최근 1년
    python crawl_kospi_news_content.py --days 180     # 최근 180일
    python crawl_kospi_news_content.py --code 005930  # 특정 종목만
"""

import re
import requests
from lxml import html
import pandas as pd
import time
import os
import argparse
from datetime import datetime, timedelta


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
NEWS_LIST_URL = "https://finance.naver.com/item/news_news.naver"


def get_list_headers(code):
    return {
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        ),
        "Referer": f"https://finance.naver.com/item/news.naver?code={code}",
    }


ARTICLE_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    ),
}


# ── 날짜 파싱 ──

def parse_news_date(date_str):
    date_str = date_str.strip()
    for fmt in ("%Y.%m.%d %H:%M", "%Y.%m.%d"):
        try:
            return datetime.strptime(date_str, fmt)
        except ValueError:
            continue
    return None


# ── 뉴스 목록 크롤링 ──

def get_last_page(tree):
    last_link = tree.xpath('//td[@class="pgRR"]/a/@href')
    if last_link:
        for part in last_link[0].split("&"):
            if part.startswith("page="):
                return int(part.split("=")[1])
    page_links = tree.xpath('//table[@class="Nnavi"]//td/a/text()')
    nums = [int(p) for p in page_links if p.strip().isdigit()]
    return max(nums) if nums else 1


def parse_news_page(tree, code):
    results = []
    tbody = tree.xpath("/html/body/div/table[1]/tbody")
    if not tbody:
        return results

    for row in tbody[0].xpath("./tr"):
        row_class = (row.get("class") or "").strip()
        if "relation_lst" in row_class:
            continue

        a_tag = row.xpath('.//td[@class="title"]/a')
        if not a_tag:
            continue

        title = a_tag[0].text_content().strip()
        href = a_tag[0].get("href", "")

        info, date = "", ""
        for td in row.xpath("./td"):
            tc = (td.get("class") or "").strip()
            if tc == "info":
                info = td.text_content().strip()
            elif tc == "date":
                date = td.text_content().strip()

        article_id, office_id = "", ""
        for part in href.split("&"):
            if "article_id=" in part:
                article_id = part.split("=")[-1]
            elif "office_id=" in part:
                office_id = part.split("=")[-1]

        news_url = ""
        if office_id and article_id:
            news_url = f"https://n.news.naver.com/mnews/article/{office_id}/{article_id}"

        results.append({
            "종목코드": code,
            "제목": title,
            "정보제공": info,
            "날짜": date,
            "뉴스URL": news_url,
        })

    return results


def crawl_news_list(code, cutoff_date, max_pages, session):
    """종목의 뉴스 목록을 크롤링합니다. cutoff_date 이전이면 중단."""
    headers = get_list_headers(code)
    all_news = []

    try:
        resp = session.get(
            f"{NEWS_LIST_URL}?code={code}&page=1&clusterId=",
            headers=headers, timeout=10,
        )
        resp.encoding = "euc-kr"
    except requests.RequestException:
        return all_news

    tree = html.fromstring(resp.text)
    total_pages = get_last_page(tree)
    if max_pages:
        total_pages = min(total_pages, max_pages)

    news = parse_news_page(tree, code)
    filtered, stop = _filter_by_date(news, cutoff_date)
    all_news.extend(filtered)
    if not news or stop:
        return all_news

    for page in range(2, total_pages + 1):
        try:
            resp = session.get(
                f"{NEWS_LIST_URL}?code={code}&page={page}&clusterId=",
                headers=headers, timeout=10,
            )
            resp.encoding = "euc-kr"
            tree = html.fromstring(resp.text)
            news = parse_news_page(tree, code)
            if not news:
                break
            filtered, stop = _filter_by_date(news, cutoff_date)
            all_news.extend(filtered)
            if stop:
                break
        except requests.RequestException:
            continue
        time.sleep(0.3)

    return all_news


def _filter_by_date(news_list, cutoff_date):
    if cutoff_date is None:
        return news_list, False
    filtered = []
    for item in news_list:
        dt = parse_news_date(item["날짜"])
        if dt and dt < cutoff_date:
            return filtered, True
        filtered.append(item)
    return filtered, False


# ── 뉴스 본문 크롤링 ──

def fetch_article_body(url, session):
    """네이버 뉴스 본문을 가져옵니다."""
    if not url:
        return ""
    try:
        resp = session.get(url, headers=ARTICLE_HEADERS, timeout=10)
        resp.raise_for_status()
        tree = html.fromstring(resp.text)
        article = tree.xpath('//article[@id="dic_area"]')
        if article:
            text = article[0].text_content().strip()
            # 연속 줄바꿈(사진/이미지 자리 등)을 단일 줄바꿈으로 정리
            text = re.sub(r"\n{2,}", "\n", text)
            return text
    except Exception:
        pass
    return ""


# ── 메인 ──

def main():
    parser = argparse.ArgumentParser(description="KOSPI 종목 뉴스 본문 크롤러")
    parser.add_argument("--code", type=str, help="특정 종목코드 (예: 005930)")
    parser.add_argument("--missing", type=str, default=None, help="누락 종목 CSV 파일 경로 (예: missing_news_codes.csv)")
    parser.add_argument("--days", type=int, default=365, help="최근 N일 (기본: 365)")
    parser.add_argument("--max-pages", type=int, default=None, help="종목당 최대 페이지 수")
    parser.add_argument("--max-rows", type=int, default=500000, help="파일당 최대 행 수 (기본: 500000)")
    parser.add_argument("--output", type=str, default="kospi_news_content.csv", help="출력 파일명")
    args = parser.parse_args()

    output_path = os.path.join(SCRIPT_DIR, args.output)
    session = requests.Session()
    cutoff_date = datetime.now() - timedelta(days=args.days)

    # 종목 목록 로드
    if args.code:
        code = args.code.zfill(6)
        codes = [code]
        names = {code: code}
    elif args.missing:
        missing_path = os.path.join(SCRIPT_DIR, args.missing) if not os.path.isabs(args.missing) else args.missing
        if not os.path.exists(missing_path):
            print(f"{args.missing} 파일이 없습니다.")
            return
        df_missing = pd.read_csv(missing_path, dtype={"종목코드": str})
        df_missing["종목코드"] = df_missing["종목코드"].str.zfill(6)
        codes = df_missing["종목코드"].tolist()
        if "종목명" in df_missing.columns:
            names = dict(zip(df_missing["종목코드"], df_missing["종목명"]))
        else:
            names = {c: c for c in codes}
        # missing 모드에서는 progress 무시 (강제 재크롤링)
        print(f"누락 종목 재크롤링 모드: {len(codes)}개 종목")
    else:
        codes_path = os.path.join(SCRIPT_DIR, "kospi_stock_codes.csv")
        if not os.path.exists(codes_path):
            print("kospi_stock_codes.csv가 없습니다. get_kospi_codes.py를 먼저 실행하세요.")
            return
        df_codes = pd.read_csv(codes_path, dtype={"종목코드": str})
        df_codes["종목코드"] = df_codes["종목코드"].str.zfill(6)
        codes = df_codes["종목코드"].tolist()
        names = dict(zip(df_codes["종목코드"], df_codes["회사명"]))

    total = len(codes)
    start_time = datetime.now()

    print(f"크롤링 시작: {total}개 종목")
    print(f"기간: 최근 {args.days}일 ({cutoff_date.strftime('%Y-%m-%d')} ~ 오늘)")
    print(f"파일당 최대 {args.max_rows:,}행")
    print("-" * 60)

    # 출력 파일명에서 확장자 분리 (분할용)
    base_name, ext = os.path.splitext(args.output)

    # 완료된 종목코드 기록 파일 (CSV 삭제해도 진행 상황 유지)
    progress_path = os.path.join(SCRIPT_DIR, ".crawl_progress.txt")
    done_codes = set()
    if not args.missing and os.path.exists(progress_path):
        with open(progress_path, "r") as f:
            done_codes = set(line.strip() for line in f if line.strip())

    # 기존 분할 파일들에서 현재 행 수 파악
    file_index = 1
    current_row_count = 0

    while True:
        if file_index == 1:
            check_path = os.path.join(SCRIPT_DIR, f"{base_name}{ext}")
        else:
            check_path = os.path.join(SCRIPT_DIR, f"{base_name}_{file_index}{ext}")

        if not os.path.exists(check_path):
            break

        try:
            # 행 수만 빠르게 카운트
            with open(check_path, "r", encoding="utf-8-sig") as f:
                row_count = sum(1 for _ in f) - 1  # 헤더 제외
            current_row_count = max(0, row_count)
            file_index += 1
        except Exception:
            break

    # 마지막 파일이 꽉 찼으면 다음 파일로
    if file_index > 1 and current_row_count >= args.max_rows:
        current_row_count = 0
    else:
        file_index = max(1, file_index - 1)

    if done_codes:
        print(f"완료된 종목 {len(done_codes)}개 발견 → 이어서 크롤링")

    def get_output_path(idx):
        if idx == 1:
            return os.path.join(SCRIPT_DIR, f"{base_name}{ext}")
        return os.path.join(SCRIPT_DIR, f"{base_name}_{idx}{ext}")

    output_path = get_output_path(file_index)
    write_header = not os.path.exists(output_path) or current_row_count == 0
    col_order = ["종목코드", "종목명", "제목", "정보제공", "날짜", "뉴스URL", "본문"]

    print(f"현재 파일: {os.path.basename(output_path)} ({current_row_count:,}행)")
    print("-" * 60)

    for i, code in enumerate(codes, 1):
        name = names.get(code, code)

        if code in done_codes:
            continue

        print(f"[{i}/{total}] {name}({code})", end=" ", flush=True)

        # 1) 뉴스 목록 크롤링
        news_list = crawl_news_list(code, cutoff_date, args.max_pages, session)

        if not news_list:
            print("→ 뉴스 0건 (스킵)")
            # 뉴스 없어도 완료 기록
            with open(progress_path, "a") as f:
                f.write(code + "\n")
            time.sleep(0.3)
            continue

        print(f"→ 뉴스 {len(news_list)}건, 본문 수집 중...", end=" ", flush=True)

        # 2) 본문 크롤링
        rows = []
        for j, item in enumerate(news_list):
            body = fetch_article_body(item["뉴스URL"], session)
            rows.append({
                "종목코드": code,
                "종목명": name,
                "제목": item["제목"],
                "정보제공": item["정보제공"],
                "날짜": item["날짜"],
                "뉴스URL": item["뉴스URL"],
                "본문": body,
            })
            if (j + 1) % 5 == 0:
                time.sleep(0.3)

        # 3) 파일 분할 저장
        remaining = rows
        while remaining:
            space = args.max_rows - current_row_count
            to_write = remaining[:space]
            remaining = remaining[space:]

            df_chunk = pd.DataFrame(to_write)[col_order]
            df_chunk.to_csv(
                output_path,
                mode="a",
                header=write_header,
                index=False,
                encoding="utf-8-sig",
            )
            write_header = False
            current_row_count += len(to_write)

            # 파일이 꽉 찼으면 다음 파일로
            if current_row_count >= args.max_rows and remaining:
                file_index += 1
                output_path = get_output_path(file_index)
                current_row_count = 0
                write_header = True
                print(f"\n  >> 새 파일: {os.path.basename(output_path)}", end=" ", flush=True)

        # 종목 완료 기록
        with open(progress_path, "a") as f:
            f.write(code + "\n")

        body_ok = sum(1 for r in rows if r["본문"])
        print(f"완료 (본문 {body_ok}/{len(rows)}건)")

        time.sleep(0.5)

    elapsed = datetime.now() - start_time
    print("-" * 60)
    print(f"크롤링 완료, 소요 시간: {elapsed}")
    print(f"저장: {output_path}")


if __name__ == "__main__":
    main()
