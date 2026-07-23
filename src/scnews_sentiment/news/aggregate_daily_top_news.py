#!/usr/bin/env python3
"""按新闻聚合 GDELT 事件 AvgTone，并筛选每日影响强度最大的新闻。

脚本处理的数据结构：
1. data/News/*_Raws：无表头的 GDELT 原始事件 CSV，文件名中的 YYYYMMDD 表示事件日期。
2. data/News/*_News：已抓取的新闻正文 CSV，其中 GlobalEventID 字段保存该新闻绑定的事件 ID 列表。

固定计算口径：
1. AvgTone_news = sum(valid AvgTone_i) / len(valid Events)。
2. len(valid Events) 只统计该新闻 GlobalEventID 绑定、成功匹配到 _Raws、且 AvgTone 有效的事件。
3. AvgTone 为空、无法转成数字、NaN 或无穷大时跳过；分母同步减少。
4. 每日筛选按 abs(AvgTone_news) 取最大影响强度。
5. Automaker 和 Nexperia 按 Topic + Date 各自每天保留 1 条新闻。

输出：
1. data/News/DailyTop_News/all_article_scores.csv：全部新闻评分明细。
2. data/News/DailyTop_News/daily_top_news.csv：每日最高影响强度新闻。
"""

from __future__ import annotations

import argparse
import ast
import csv
import math
import re
import sys
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


# GDELT Events 原始文件没有表头，这里按官方 Events Export 58 列顺序补齐字段名。
# 读取 _Raws 文件时会用这些字段名把每一行事件转换成字典。
GDELT_EVENT_COLUMNS = [
    "GlobalEventID",
    "SQLDATE",
    "MonthYear",
    "Year",
    "FractionDate",
    "Actor1Code",
    "Actor1Name",
    "Actor1CountryCode",
    "Actor1KnownGroupCode",
    "Actor1EthnicCode",
    "Actor1Religion1Code",
    "Actor1Religion2Code",
    "Actor1Type1Code",
    "Actor1Type2Code",
    "Actor1Type3Code",
    "Actor2Code",
    "Actor2Name",
    "Actor2CountryCode",
    "Actor2KnownGroupCode",
    "Actor2EthnicCode",
    "Actor2Religion1Code",
    "Actor2Religion2Code",
    "Actor2Type1Code",
    "Actor2Type2Code",
    "Actor2Type3Code",
    "IsRootEvent",
    "EventCode",
    "EventBaseCode",
    "EventRootCode",
    "QuadClass",
    "GoldsteinScale",
    "NumMentions",
    "NumSources",
    "NumArticles",
    "AvgTone",
    "Actor1Geo_Type",
    "Actor1Geo_FullName",
    "Actor1Geo_CountryCode",
    "Actor1Geo_ADM1Code",
    "Actor1Geo_Lat",
    "Actor1Geo_Long",
    "Actor1Geo_FeatureID",
    "Actor2Geo_Type",
    "Actor2Geo_FullName",
    "Actor2Geo_CountryCode",
    "Actor2Geo_ADM1Code",
    "Actor2Geo_Lat",
    "Actor2Geo_Long",
    "Actor2Geo_FeatureID",
    "ActionGeo_Type",
    "ActionGeo_FullName",
    "ActionGeo_CountryCode",
    "ActionGeo_ADM1Code",
    "ActionGeo_Lat",
    "ActionGeo_Long",
    "ActionGeo_FeatureID",
    "DATEADDED",
    "SOURCEURL",
]

# 这些字段在 GDELT 原始事件中应按数值处理。
# 无法转换成有限数字时保留为 None，后续均值计算会跳过。
NUMERIC_COLUMNS = {
    "GlobalEventID",
    "SQLDATE",
    "MonthYear",
    "Year",
    "FractionDate",
    "IsRootEvent",
    "EventCode",
    "EventBaseCode",
    "EventRootCode",
    "QuadClass",
    "GoldsteinScale",
    "NumMentions",
    "NumSources",
    "NumArticles",
    "AvgTone",
    "Actor1Geo_Type",
    "Actor1Geo_Lat",
    "Actor1Geo_Long",
    "Actor2Geo_Type",
    "Actor2Geo_Lat",
    "Actor2Geo_Long",
    "ActionGeo_Type",
    "ActionGeo_Lat",
    "ActionGeo_Long",
    "DATEADDED",
}

# 输出文件中只保留需求指定的事件指标派生列：
# 1. GoldsteinScale 和 AvgTone：按绝对值最大选择，但输出保留原始正负号。
# 2. NumMentions、NumSources、NumArticles：对新闻绑定的全部有效子事件求和。
# 3. QuadClass 不再输出任何派生字段。
ABS_MAX_COLUMNS = ["GoldsteinScale", "AvgTone"]
SUM_COLUMNS = ["NumMentions", "NumSources", "NumArticles"]
EVENT_METRIC_OUTPUT_COLUMNS = [
    "GoldsteinScale_AbsMax",
    "NumMentions_Sum",
    "NumSources_Sum",
    "NumArticles_Sum",
    "AvgTone_AbsMax",
]

DEFAULT_NEWS_ROOT = Path("data") / "News"
OUTPUT_DIR_NAME = "DailyTop_News"
ALL_SCORES_FILENAME = "all_article_scores.csv"
DAILY_TOP_FILENAME = "daily_top_news.csv"
SCORE_FORMULA_TEXT = "AvgTone_news = sum(valid AvgTone_i) / len(valid Events)"
EXCLUDED_NEWS_OUTPUT_COLUMNS = {"Article_Date", "Scrape_Status", "Risk_Relevance_Label"}


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """定义命令行参数。

    脚本现在只保留正式流程所需参数：
    1. --news-root：指定 News 根目录；
    2. --encoding：指定 CSV 读写编码。

    输出目录固定为 <news-root>/DailyTop_News，不再提供自定义输出目录参数，
    这样下次换新数据运行时不会再产生 *_check 等临时目录。
    """
    parser = argparse.ArgumentParser(
        description="Aggregate valid AvgTone by article and select daily strongest-impact news."
    )
    parser.add_argument(
        "--news-root",
        type=Path,
        default=DEFAULT_NEWS_ROOT,
        help="Root directory containing *_Raws and *_News directories.",
    )
    parser.add_argument(
        "--encoding",
        default="utf-8-sig",
        help="Input and output CSV encoding.",
    )
    return parser.parse_args(argv)


def parse_float(value: Any) -> Optional[float]:
    """把输入值转换为 float；无效数字返回 None。

    这里特意不把无效 AvgTone 转成 0，因为需求要求无效 AvgTone 跳过，
    并且均值分母也同步减少。
    """
    if value is None or value == "":
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(result) or math.isinf(result):
        return None
    return result


def to_float(value: Any) -> float:
    """把输入值转换为 float；无效数字按 0 处理。

    该函数只用于排名字段等必须有数字的场景，不用于 AvgTone_news 的分母计算。
    """
    result = parse_float(value)
    return 0.0 if result is None else result


def format_number(value: float) -> str:
    """统一输出数字格式，避免 CSV 中出现过长的小数尾巴。"""
    return f"{value:.12g}"


def read_raw_events(raw_file: Path, encoding: str) -> Dict[str, Dict[str, Any]]:
    """读取单个 _Raws 文件，返回 GlobalEventID 到事件记录的映射。

    _Raws 文件没有表头，所以按 GDELT_EVENT_COLUMNS 手动补字段名。
    GlobalEventID 是新闻文件和原始事件文件之间的绑定键。
    """
    events: Dict[str, Dict[str, Any]] = {}
    with raw_file.open("r", encoding=encoding, newline="") as handle:
        reader = csv.reader(handle, delimiter="\t")
        for line_number, row in enumerate(reader, start=1):
            if not row:
                continue
            if len(row) < len(GDELT_EVENT_COLUMNS):
                row = row + [""] * (len(GDELT_EVENT_COLUMNS) - len(row))
            elif len(row) > len(GDELT_EVENT_COLUMNS):
                row = row[: len(GDELT_EVENT_COLUMNS)]

            event: Dict[str, Any] = OrderedDict(zip(GDELT_EVENT_COLUMNS, row))

            # 所有数值字段先转成 float 或 None；None 会在聚合时被跳过。
            for column in NUMERIC_COLUMNS:
                event[column] = parse_float(event.get(column))

            event_id_value = event.get("GlobalEventID")
            event_id = str(int(event_id_value)) if event_id_value is not None else ""
            if not event_id:
                print(f"Warning: missing GlobalEventID in {raw_file}:{line_number}", file=sys.stderr)
                continue
            events[event_id] = event
    return events


def read_news_rows(news_file: Path, encoding: str) -> Tuple[List[str], List[Dict[str, str]]]:
    """读取对应的 *_News 文件，保留原始新闻字段和值。"""
    with news_file.open("r", encoding=encoding, newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        fieldnames = list(reader.fieldnames or [])
        rows = [dict(row) for row in reader]
    return fieldnames, rows


def parse_event_ids(value: str) -> List[str]:
    """解析新闻表 GlobalEventID 字段中的事件 ID 列表。

    当前数据通常形如 ['1268259447', '1268259772']。
    这里先尝试按 Python 字面量解析；如果格式不标准，再退回到正则提取数字。
    """
    if not value:
        return []
    try:
        parsed = ast.literal_eval(value)
    except (SyntaxError, ValueError):
        parsed = None

    ids: List[str] = []
    if isinstance(parsed, (list, tuple, set)):
        ids = [str(item).strip() for item in parsed if str(item).strip()]
    elif parsed is not None:
        ids = [str(parsed).strip()]
    else:
        ids = re.findall(r"\d+", value)
    return [str(int(item)) for item in ids if item.isdigit()]


def extract_yyyymmdd(path: Path) -> Optional[str]:
    """从文件名中提取 YYYYMMDD 日期。"""
    match = re.search(r"(\d{8})", path.stem)
    return match.group(1) if match else None


def find_news_file(news_root: Path, topic: str, date_text: str) -> Optional[Path]:
    """根据 topic 和日期找到与 _Raws 文件对应的 *_News 文件。"""
    directory = news_root / f"{topic}_News"
    if not directory.is_dir():
        return None

    wanted_stems = {f"{topic}_News_{date_text}".lower(), f"{topic}_{date_text}".lower()}
    for candidate in directory.iterdir():
        if candidate.is_file() and candidate.suffix.lower() == ".csv" and candidate.stem.lower() in wanted_stems:
            return candidate
    return None


def valid_numeric_values(events: Sequence[Dict[str, Any]], column: str) -> List[float]:
    """提取某个事件指标的有效数值列表。

    None、空字符串、非数字、NaN 和无穷大会被排除。
    """
    values: List[float] = []
    for event in events:
        value = parse_float(event.get(column))
        if value is not None:
            values.append(value)
    return values


def mean_or_zero(values: Sequence[float]) -> float:
    """计算均值；没有有效值时返回 0。

    “全部 AvgTone 无效时返回 0”是当前脚本的默认兜底策略。
    如需改为跳过该新闻或输出空值，需要再明确需求。
    """
    return sum(values) / len(values) if values else 0.0


def abs_max_keep_sign(values: Sequence[float]) -> float:
    """取绝对值最大的数，但输出保留原始正负号。

    如果绝对值并列，例如 -5 和 5，按需求优先保留负数，所以返回 -5。
    没有任何有效值时返回 0。
    """
    if not values:
        return 0.0

    best = values[0]
    for value in values[1:]:
        if abs(value) > abs(best):
            best = value
        elif abs(value) == abs(best) and value < best:
            best = value
    return best


def summarize_event_metrics(events: Sequence[Dict[str, Any]]) -> Dict[str, str]:
    """生成输出文件中的事件指标列。

    这不是原始 _Raws 文件中的直接字段，而是基于新闻绑定的子事件计算得到：
    1. GoldsteinScale_AbsMax：GoldsteinScale 绝对值最大值，保留原始符号。
    2. NumMentions_Sum：NumMentions 求和。
    3. NumSources_Sum：NumSources 求和。
    4. NumArticles_Sum：NumArticles 求和。
    5. AvgTone_AbsMax：AvgTone 绝对值最大值，保留原始符号。

    QuadClass 不再输出派生字段。
    """
    summary: Dict[str, str] = {}
    for column in ABS_MAX_COLUMNS:
        values = valid_numeric_values(events, column)
        summary[f"{column}_AbsMax"] = format_number(abs_max_keep_sign(values))
    for column in SUM_COLUMNS:
        values = valid_numeric_values(events, column)
        summary[f"{column}_Sum"] = format_number(sum(values))
    return summary


def build_scored_row(
    *,
    topic: str,
    date_text: str,
    news_row: Dict[str, str],
    news_fieldnames: Sequence[str],
    matched_events: Sequence[Dict[str, Any]],
    missing_event_ids: Sequence[str],
) -> Dict[str, str]:
    """把一条新闻及其绑定事件转换为一行评分结果。

    ArticleScore 是 AvgTone_news，保留正负方向。
    SelectionScore 是 abs(ArticleScore)，用于每日最大影响强度筛选。
    """
    avg_tone_values = valid_numeric_values(matched_events, "AvgTone")
    article_score = mean_or_zero(avg_tone_values)
    selection_score = abs(article_score)

    output: Dict[str, str] = OrderedDict()
    output["Topic"] = topic
    output["Date"] = date_text
    output["ArticleScore"] = format_number(article_score)
    output["SelectionScore"] = format_number(selection_score)

    # 以下字段保留在内存中供校验或并列排序使用，但不写入 CSV：
    # ScoreFormula、EventCount、ValidAvgToneEventCount、MissingEventIDCount、MissingEventIDs。
    output["EventCount"] = str(len(matched_events))
    output["ValidAvgToneEventCount"] = str(len(avg_tone_values))
    output["MissingEventIDCount"] = str(len(missing_event_ids))
    output["MatchedEventIDs"] = ";".join(str(int(event["GlobalEventID"])) for event in matched_events)
    output["MissingEventIDs"] = ";".join(missing_event_ids)
    output.update(summarize_event_metrics(matched_events))

    # 追加原始新闻字段，例如 SourceURL、GlobalEventID、Article_Title、Article_Text 等。
    for fieldname in news_fieldnames:
        output[fieldname] = news_row.get(fieldname, "")
    return output


def iter_raw_files(news_root: Path) -> Iterable[Tuple[str, Path]]:
    """遍历所有 *_Raws 目录中的 CSV 文件，并从目录名提取 topic。"""
    for raw_dir in sorted(news_root.glob("*_Raws")):
        if not raw_dir.is_dir():
            continue
        topic = raw_dir.name[: -len("_Raws")]
        for raw_file in sorted(raw_dir.iterdir()):
            if raw_file.is_file() and raw_file.suffix.lower() == ".csv":
                yield topic, raw_file


def score_articles(news_root: Path, encoding: str) -> Tuple[List[Dict[str, str]], List[str]]:
    """完成全量新闻评分。

    处理流程：
    1. 遍历 data/News 下所有 *_Raws 目录；
    2. 从 _Raws 文件名提取日期；
    3. 找到同 topic、同日期的 *_News 文件；
    4. 根据新闻 GlobalEventID 找回所有成功匹配的事件；
    5. 计算 AvgTone_news 和 abs(AvgTone_news)；
    6. 返回所有新闻评分行，供后续每日筛选。
    """
    scored_rows: List[Dict[str, str]] = []
    all_news_fields: List[str] = []
    seen_news_fields = set()

    for topic, raw_file in iter_raw_files(news_root):
        date_text = extract_yyyymmdd(raw_file)
        if not date_text:
            print(f"Warning: cannot parse date from {raw_file}", file=sys.stderr)
            continue

        news_file = find_news_file(news_root, topic, date_text)
        if news_file is None:
            print(f"Warning: missing news file for {raw_file}", file=sys.stderr)
            continue

        events = read_raw_events(raw_file, encoding)
        news_fieldnames, news_rows = read_news_rows(news_file, encoding)
        for fieldname in news_fieldnames:
            if fieldname not in seen_news_fields:
                all_news_fields.append(fieldname)
                seen_news_fields.add(fieldname)

        for news_row in news_rows:
            event_ids = parse_event_ids(news_row.get("GlobalEventID", ""))

            # len(Events) 的口径：只统计新闻绑定且在对应 _Raws 文件中成功匹配到的事件。
            # AvgTone_news 的分母进一步只统计 AvgTone 有效的事件数量。
            matched_events = [events[event_id] for event_id in event_ids if event_id in events]
            missing_event_ids = [event_id for event_id in event_ids if event_id not in events]
            scored_rows.append(
                build_scored_row(
                    topic=topic,
                    date_text=date_text,
                    news_row=news_row,
                    news_fieldnames=news_fieldnames,
                    matched_events=matched_events,
                    missing_event_ids=missing_event_ids,
                )
            )

    return scored_rows, all_news_fields


def select_daily_maxima(rows: Sequence[Dict[str, str]]) -> List[Dict[str, str]]:
    """按 Topic + Date 筛选最高影响强度新闻。

    Automaker 和 Nexperia 各自每天选 1 条。
    比较时使用 SelectionScore，即 abs(AvgTone_news)。
    若分数完全相同，则优先保留绑定事件数更多的新闻，再用 SourceURL 做稳定排序。
    """
    winners: Dict[Tuple[str, str], Dict[str, str]] = {}
    for row in rows:
        key = (row["Topic"], row["Date"])
        score = to_float(row.get("SelectionScore"))
        current = winners.get(key)
        if current is None:
            winners[key] = row
            continue

        current_score = to_float(current.get("SelectionScore"))
        if score > current_score:
            winners[key] = row
        elif score == current_score:
            row_event_count = int(row.get("EventCount", "0"))
            current_event_count = int(current.get("EventCount", "0"))
            if row_event_count > current_event_count:
                winners[key] = row
            elif row_event_count == current_event_count and row.get("SourceURL", "") < current.get("SourceURL", ""):
                winners[key] = row

    return sorted(winners.values(), key=lambda item: (item.get("Topic", ""), item.get("Date", "")))


def output_fieldnames(news_fieldnames: Sequence[str]) -> List[str]:
    """定义输出 CSV 字段顺序。

    先输出脚本生成的评分与汇总字段，再追加原始新闻文件中的字段，
    这样结果既方便排序分析，也能直接看到新闻标题、正文、URL 等信息。
    """
    base = [
        "Topic",
        "Date",
        "ArticleScore",
        "SelectionScore",
        "MatchedEventIDs",
    ]
    base.extend(EVENT_METRIC_OUTPUT_COLUMNS)
    return base + [
        fieldname
        for fieldname in news_fieldnames
        if fieldname not in base and fieldname not in EXCLUDED_NEWS_OUTPUT_COLUMNS
    ]


def write_csv(path: Path, rows: Sequence[Dict[str, str]], fieldnames: Sequence[str], encoding: str) -> None:
    """写出 CSV 文件，父目录不存在时自动创建。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding=encoding, newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main(argv: Optional[Sequence[str]] = None) -> int:
    """脚本入口：评分、筛选、写出正式结果文件。"""
    args = parse_args(argv)
    news_root = args.news_root
    if not news_root.is_dir():
        print(f"Error: news root does not exist: {news_root}", file=sys.stderr)
        return 2

    scored_rows, news_fieldnames = score_articles(news_root, args.encoding)
    fields = output_fieldnames(news_fieldnames)
    daily_top_rows = select_daily_maxima(scored_rows)

    output_dir = news_root / OUTPUT_DIR_NAME
    all_scores_file = output_dir / ALL_SCORES_FILENAME
    daily_top_file = output_dir / DAILY_TOP_FILENAME
    write_csv(all_scores_file, scored_rows, fields, args.encoding)
    write_csv(daily_top_file, daily_top_rows, fields, args.encoding)

    print(f"Wrote {len(scored_rows)} scored article rows to {all_scores_file}")
    print(f"Wrote {len(daily_top_rows)} daily top news rows to {daily_top_file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
