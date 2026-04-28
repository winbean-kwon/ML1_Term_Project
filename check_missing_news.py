"""
Google Colab에서 실행: .crawl_progress.txt에 완료로 기록됐지만
실제 CSV에 뉴스 데이터가 없는 종목을 찾아내는 스크립트

사용법 (Colab):
    1) 드라이브 마운트 후 파일 경로를 아래 설정에 맞게 수정
    2) 셀에서 실행
"""

import pandas as pd

# ── 경로 설정 (Colab 드라이브 경로에 맞게 수정) ──
DRIVE_BASE = "/content/drive/MyDrive"  # 드라이브 기본 경로
PROGRESS_PATH = f"{DRIVE_BASE}/.crawl_progress.txt"
CODES_CSV_PATH = f"{DRIVE_BASE}/kospi_stock_codes.csv"

# ── 1) 종목코드 목록 로드 ──
df_codes = pd.read_csv(CODES_CSV_PATH, dtype={"종목코드": str})
df_codes["종목코드"] = df_codes["종목코드"].str.zfill(6)
code_to_name = dict(zip(df_codes["종목코드"], df_codes["회사명"]))

# ── 2) progress에 기록된 완료 종목 ──
with open(PROGRESS_PATH, "r") as f:
    done_codes = [line.strip() for line in f if line.strip()]
done_set = set(done_codes)

# ── 3) CSV에 실제 데이터가 있는 종목 (분할 파일 포함) ──
import glob

csv_patterns = [
    f"{DRIVE_BASE}/kospi_news_content.csv",
    f"{DRIVE_BASE}/kospi_news_content (*).csv",  # (1)~(5)
]
csv_files = []
for pat in csv_patterns:
    csv_files.extend(glob.glob(pat))

saved_codes = set()
total_rows = 0
for f in sorted(csv_files):
    df_tmp = pd.read_csv(f, dtype={"종목코드": str}, usecols=["종목코드"])
    df_tmp["종목코드"] = df_tmp["종목코드"].str.zfill(6)
    saved_codes.update(df_tmp["종목코드"].unique())
    total_rows += len(df_tmp)
    print(f"  {f.split('/')[-1]}: {len(df_tmp):,}행, {len(df_tmp['종목코드'].unique())}종목")

print(f"  → 총 {len(csv_files)}개 파일, {total_rows:,}행")

# ── 4) 비교 ──
all_codes = df_codes["종목코드"].tolist()
total_codes = len(all_codes)

# CSV에 데이터가 없는 모든 종목 = 누락 종목
missing = [c for c in all_codes if c not in saved_codes]
# 그 중 progress에 잘못 기록된 종목
false_done = [c for c in done_codes if c not in saved_codes]

print(f"\n전체 종목 수: {total_codes}")
print(f"progress 완료 종목: {len(done_set)}")
print(f"CSV에 데이터 있는 종목: {len(saved_codes)}")
print(f"CSV에 데이터 없는 종목 (재크롤링 필요): {len(missing)}")
print(f"  - progress에 잘못 기록된 종목: {len(false_done)}")
print(f"  - progress에도 없는 종목: {len(missing) - len(false_done)}")
print("-" * 60)

if missing:
    print(f"\n재크롤링 필요 종목 ({len(missing)}개):")
    for c in missing:
        name = code_to_name.get(c, "?")
        in_progress = "← progress에 잘못 기록됨" if c in done_set else ""
        print(f"  {name}({c}) {in_progress}")

    # ── 5) 누락 종목 CSV 저장 ──
    missing_df = pd.DataFrame([
        {"종목코드": c, "종목명": code_to_name.get(c, "?")} for c in missing
    ])
    missing_csv_path = f"{DRIVE_BASE}/missing_news_codes.csv"
    missing_df.to_csv(missing_csv_path, index=False, encoding="utf-8-sig")
    print(f"\n누락 종목 CSV 저장: {missing_csv_path} ({len(missing)}개)")

    # ── 6) 수정된 progress 파일 저장 (잘못 기록된 종목 제거) ──
    if false_done:
        cleaned = [c for c in done_codes if c in saved_codes]
        cleaned_path = PROGRESS_PATH.replace(".crawl_progress.txt", ".crawl_progress_cleaned.txt")
        with open(cleaned_path, "w") as f:
            f.write("\n".join(cleaned) + "\n")
        print(f"\n정리된 progress 파일 저장: {cleaned_path}")
        print(f"  기존 {len(done_codes)}개 → {len(cleaned)}개 (제거: {len(false_done)}개)")
        print(f"\n이 파일을 .crawl_progress.txt로 교체하면 누락 종목2만 재크롤링됩니다.")
else:
    print("\n모든 종목에 데이터가 있습니다. 문제 없음!")
