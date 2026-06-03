# model/collect_gdelt_event_features.py

from __future__ import annotations

import time
from pathlib import Path

import pandas as pd
import requests

# ==============================
# Path settings
# ==============================

PROJECT_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_DIR / "data" / "raw"

OUTPUT_PATH = RAW_DIR / "gdelt_event_features_daily.csv"


# ==============================
# Collection settings
# ==============================

START_DATE = "2018-01-01"
END_DATE = "2026-05-31"

GDELT_DOC_API = "https://api.gdeltproject.org/api/v2/doc/doc"

REQUEST_SLEEP_SEC = 7.0
REQUEST_TIMEOUT_SEC = 60
MAX_RETRY = 3


# ==============================
# Keyword groups
# ==============================

KEYWORD_GROUPS = {"hormuz_risk": ['"Strait of Hormuz"']}


# ==============================
# Date utils
# ==============================


def month_ranges(start_date: str, end_date: str):
    start = pd.to_datetime(start_date)
    end = pd.to_datetime(end_date)

    cur = pd.Timestamp(start.year, start.month, 1)

    while cur <= end:
        month_start = max(cur, start)

        next_month = cur + pd.offsets.MonthBegin(1)
        month_end = min(
            next_month - pd.Timedelta(seconds=1),
            end + pd.Timedelta(hours=23, minutes=59, seconds=59),
        )

        yield month_start, month_end

        cur = next_month


def gdelt_datetime(ts: pd.Timestamp) -> str:
    return ts.strftime("%Y%m%d%H%M%S")


def clean_date(series: pd.Series) -> pd.Series:
    return (
        pd.to_datetime(series, errors="coerce", utc=True)
        .dt.tz_convert(None)
        .dt.floor("D")
    )


# ==============================
# Query utils
# ==============================


def parse_gdelt_timeline_json(data: dict) -> pd.DataFrame:
    """
    GDELT TimelineVolRaw 응답 구조가 조금씩 다를 수 있어서 여러 형태를 처리한다.

    가능한 형태:
    1. {"timeline": [{"date": "...", "value": ...}, ...]}
    2. {"timeline": [{"series": "...", "data": [{"date": "...", "value": ...}, ...]}]}
    3. {"timeline": {"data": [...]}}
    """
    rows = []

    timeline = data.get("timeline", [])

    if isinstance(timeline, dict):
        timeline = [timeline]

    if not isinstance(timeline, list):
        return pd.DataFrame(columns=["date", "count"])

    for item in timeline:
        if not isinstance(item, dict):
            continue

        # case 1: item 자체가 date/value를 가진 경우
        if any(k in item for k in ["date", "datetime", "time"]):
            date_value = item.get("date") or item.get("datetime") or item.get("time")
            count_value = (
                item.get("value")
                or item.get("count")
                or item.get("volume")
                or item.get("norm")
                or 0
            )

            rows.append(
                {
                    "date": date_value,
                    "count": count_value,
                }
            )

        # case 2: item 안에 data list가 있는 경우
        nested_data = item.get("data", [])

        if isinstance(nested_data, list):
            for sub in nested_data:
                if not isinstance(sub, dict):
                    continue

                date_value = sub.get("date") or sub.get("datetime") or sub.get("time")
                count_value = (
                    sub.get("value")
                    or sub.get("count")
                    or sub.get("volume")
                    or sub.get("norm")
                    or 0
                )

                rows.append(
                    {
                        "date": date_value,
                        "count": count_value,
                    }
                )

    if not rows:
        return pd.DataFrame(columns=["date", "count"])

    df = pd.DataFrame(rows)

    df["date"] = clean_date(df["date"])
    df["count"] = pd.to_numeric(df["count"], errors="coerce").fillna(0)

    df = df.dropna(subset=["date"])

    df = df.groupby("date", as_index=False)["count"].sum()

    return df


def fetch_timeline_raw(
    query: str, start: pd.Timestamp, end: pd.Timestamp
) -> pd.DataFrame:
    params = {
        "query": query,
        "mode": "TimelineVolRaw",
        "format": "json",
        "startdatetime": gdelt_datetime(start),
        "enddatetime": gdelt_datetime(end),
    }

    last_error = None

    for attempt in range(1, MAX_RETRY + 1):
        try:
            response = requests.get(
                GDELT_DOC_API,
                params=params,
                timeout=REQUEST_TIMEOUT_SEC,
            )

            if response.status_code != 200:
                raise RuntimeError(
                    f"HTTP {response.status_code}: {response.text[:300]}"
                )

            data = response.json()
            return parse_gdelt_timeline_json(data)

        except Exception as e:
            last_error = e
            print(f"[재시도 {attempt}/{MAX_RETRY}] 실패:", e)
            time.sleep(10 * attempt)

    raise RuntimeError(f"GDELT 요청 최종 실패: {last_error}")


def fetch_group_daily_count(group_name: str, keywords: list[str]) -> pd.DataFrame:
    print("\n" + "=" * 80)
    print("수집 그룹:", group_name)
    print("keywords:", keywords)
    print("=" * 80)

    group_parts = []

    for keyword in keywords:
        print("\n--- keyword:", keyword, "---")

        keyword_parts = []

        for start, end in month_ranges(START_DATE, END_DATE):
            print("기간:", start.date(), "~", end.date())

            try:
                part = fetch_timeline_raw(keyword, start, end)
                keyword_parts.append(part)
                print("  수집 행:", len(part))

            except Exception as e:
                print("  [실패]", start.date(), "~", end.date(), e)

            time.sleep(REQUEST_SLEEP_SEC)

        if keyword_parts:
            keyword_df = pd.concat(keyword_parts, axis=0, ignore_index=True)

            if not keyword_df.empty:
                keyword_df = keyword_df.groupby("date", as_index=False)["count"].sum()
                group_parts.append(keyword_df)

    if not group_parts:
        return pd.DataFrame(columns=["date", f"{group_name}_count"])

    df = pd.concat(group_parts, axis=0, ignore_index=True)
    df = df.groupby("date", as_index=False)["count"].sum()
    df = df.rename(columns={"count": f"{group_name}_count"})

    return df


# ==============================
# Feature engineering
# ==============================


def add_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    count_cols = [col for col in df.columns if col.endswith("_count")]

    new_features = {}

    for col in count_cols:
        new_features[f"{col}_rolling_3_sum"] = df[col].rolling(3).sum()
        new_features[f"{col}_rolling_5_sum"] = df[col].rolling(5).sum()
        new_features[f"{col}_rolling_10_sum"] = df[col].rolling(10).sum()
        new_features[f"{col}_rolling_20_sum"] = df[col].rolling(20).sum()

        new_features[f"{col}_diff_3"] = df[col] - df[col].shift(3)
        new_features[f"{col}_diff_5"] = df[col] - df[col].shift(5)
        new_features[f"{col}_diff_10"] = df[col] - df[col].shift(10)

        rolling_5 = df[col].rolling(5).mean()
        rolling_20 = df[col].rolling(20).mean()

        new_features[f"{col}_rolling_5_20_gap"] = rolling_5 - rolling_20
        new_features[f"{col}_spike_ratio_5_20"] = rolling_5 / (rolling_20 + 1e-9)

    feature_df = pd.DataFrame(new_features, index=df.index)
    result = pd.concat([df, feature_df], axis=1)

    result = result.replace([float("inf"), float("-inf")], pd.NA)
    result = result.fillna(0)

    return result


# ==============================
# Main
# ==============================


def main():
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("GDELT 이벤트 feature 수집 시작")
    print("=" * 80)
    print("기간:", START_DATE, "~", END_DATE)
    print("출력:", OUTPUT_PATH)

    base = pd.DataFrame({"date": pd.date_range(START_DATE, END_DATE, freq="D")})
    base["date"] = clean_date(base["date"])

    result = base.copy()

    for group_name, keywords in KEYWORD_GROUPS.items():
        group_df = fetch_group_daily_count(group_name, keywords)

        result["date"] = clean_date(result["date"])
        group_df["date"] = clean_date(group_df["date"])

        result = result.merge(group_df, on="date", how="left")

    count_cols = [col for col in result.columns if col.endswith("_count")]
    result[count_cols] = result[count_cols].fillna(0)

    result = result.sort_values("date").reset_index(drop=True)

    result = add_rolling_features(result)

    result.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")

    print("\n" + "=" * 80)
    print("GDELT 이벤트 feature 저장 완료")
    print("=" * 80)
    print("경로:", OUTPUT_PATH)
    print("크기:", result.shape)
    print("날짜 범위:", result["date"].min(), "~", result["date"].max())

    print("\n앞 5행:")
    print(result.head().to_string(index=False))

    print("\n뒤 5행:")
    print(result.tail().to_string(index=False))

    print("\ncount 통계:")
    print(result[count_cols].describe())

    print("\n2026년 3월 주변:")
    start_check = pd.Timestamp("2026-02-15")
    end_check = pd.Timestamp("2026-04-15")

    check = result[(result["date"] >= start_check) & (result["date"] <= end_check)]

    print(check[["date"] + count_cols].to_string(index=False))

    print("\n수집 완료")


if __name__ == "__main__":
    main()
