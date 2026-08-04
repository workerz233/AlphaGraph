from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha1
from pathlib import Path
import re
from typing import Any

import pandas as pd


SECTION_HEADER_RE = re.compile(r"^##\s+(.+?)\s*$", re.MULTILINE)
METADATA_RE = re.compile(r"^-\s*([^:：]+)\s*[:：]\s*(.+?)\s*$", re.MULTILINE)
RELATED_STOCK_RE = re.compile(
    r"^-\s*股票\s*\d+\s*[:：]\s*([^\(（]+?)\s*[\(（]\s*([0-9]{6}\.(?:SH|SZ))\s*[\)）]\s*$",
    re.MULTILINE,
)
CHINESE_DATE_RE = re.compile(r"(\d{4})年(\d{2})月(\d{2})日")


@dataclass(frozen=True)
class ParsedSections:
    title: str
    sections: dict[str, str]


def scan_report_markdown(
    root: str | Path,
    markdown_dir: str | Path,
    start_month: str | None = None,
    end_month: str | None = None,
) -> list[Path]:
    base_dir = Path(root) / markdown_dir
    if not base_dir.exists():
        return []
    if base_dir.is_file():
        return [base_dir] if base_dir.suffix == ".md" and not base_dir.name.startswith(".") else []
    resolved_start = _normalize_report_month(start_month)
    resolved_end = _normalize_report_month(end_month)
    if _is_single_month_markdown_dir(base_dir.name):
        return sorted(
            path
            for path in base_dir.glob("*.md")
            if path.is_file() and not path.name.startswith(".")
        )
    if resolved_start and resolved_end:
        monthly_paths: list[Path] = []
        for month_dir in sorted(path for path in base_dir.iterdir() if path.is_dir()):
            month = _month_from_directory_name(month_dir.name)
            if not month or month < resolved_start or month > resolved_end:
                continue
            monthly_paths.extend(
                sorted(
                    path
                    for path in month_dir.glob("*.md")
                    if path.is_file() and not path.name.startswith(".")
                )
            )
        return monthly_paths
    return sorted(
        path
        for path in base_dir.rglob("*.md")
        if path.is_file() and not path.name.startswith(".")
    )


def parse_report_file(path: str | Path) -> dict[str, Any]:
    report_path = Path(path)
    raw_text = report_path.read_text(encoding="utf-8")
    parsed = _split_sections(raw_text)
    metadata = _parse_metadata(parsed.sections.get("基本信息", ""))
    publish_date = _normalize_publish_date(metadata.get("发布日期", ""))
    related_stocks = _parse_related_stocks(parsed.sections.get("关联标的", ""))
    conclusion_text = "\n".join(
        filter(
            None,
            [
                parsed.sections.get("结论", "").strip(),
                parsed.sections.get("总结", "").strip(),
            ],
        )
    ).strip()

    return {
        "source_path": str(report_path),
        "report_title": parsed.title,
        "industry": metadata.get("行业", "").strip(),
        "publish_date": publish_date,
        "report_month": publish_date[:7] if publish_date else "",
        "broker": metadata.get("发布机构", "").strip(),
        "raw_text": raw_text,
        "title_text": parsed.title,
        "conclusion_text": conclusion_text,
        "related_stocks": related_stocks,
    }


def compute_incidence_weight(report_text: str, stock_name: str, stock_code: str) -> dict[str, float | int]:
    mention_count = _count_mentions(report_text, stock_name, stock_code)
    title_hits = _count_mentions(_extract_title(report_text), stock_name, stock_code)
    conclusion_hits = _count_mentions(_extract_conclusion(report_text), stock_name, stock_code)
    weight = 1.0 + title_hits * 2.0 + conclusion_hits * 1.5 + mention_count * 0.2
    return {
        "mention_count": mention_count,
        "title_hits": title_hits,
        "conclusion_hits": conclusion_hits,
        "incidence_weight": round(weight, 6),
    }


def build_hyperedge_tables(reports_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    normalized = reports_df.copy()
    if "report_id" not in normalized.columns:
        normalized["report_id"] = normalized.apply(_build_report_id, axis=1)

    incidence_rows: list[dict[str, Any]] = []
    for row in normalized.to_dict(orient="records"):
        for stock in row.get("related_stocks", []) or []:
            weight_parts = compute_incidence_weight(
                row.get("raw_text", ""),
                stock["stock_name"],
                stock["stock_code"],
            )
            incidence_rows.append(
                {
                    "report_id": row["report_id"],
                    "stock_code": stock["stock_code"],
                    "stock_name": stock["stock_name"],
                    **weight_parts,
                }
            )

    incidence_df = pd.DataFrame(
        incidence_rows,
        columns=[
            "report_id",
            "stock_code",
            "stock_name",
            "mention_count",
            "title_hits",
            "conclusion_hits",
            "incidence_weight",
        ],
    )

    if not incidence_df.empty:
        incidence_df = (
            incidence_df.groupby(["report_id", "stock_code", "stock_name"], as_index=False)
            .agg(
                {
                    "mention_count": "sum",
                    "title_hits": "sum",
                    "conclusion_hits": "sum",
                    "incidence_weight": "sum",
                }
            )
            .sort_values(["report_id", "stock_code"], ignore_index=True)
        )

    return normalized.reset_index(drop=True), incidence_df


def _split_sections(raw_text: str) -> ParsedSections:
    lines = raw_text.splitlines()
    title = ""
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("# "):
            title = stripped[2:].strip()
            break

    matches = list(SECTION_HEADER_RE.finditer(raw_text))
    sections: dict[str, str] = {}
    for index, match in enumerate(matches):
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(raw_text)
        sections[match.group(1).strip()] = raw_text[start:end].strip()
    return ParsedSections(title=title, sections=sections)


def _parse_metadata(section_text: str) -> dict[str, str]:
    return {key.strip(): value.strip() for key, value in METADATA_RE.findall(section_text)}


def _parse_related_stocks(section_text: str) -> list[dict[str, str]]:
    return [
        {"stock_name": stock_name.strip(), "stock_code": stock_code.strip()}
        for stock_name, stock_code in RELATED_STOCK_RE.findall(section_text)
    ]


def _normalize_publish_date(raw_value: str) -> str:
    match = CHINESE_DATE_RE.search(raw_value)
    if not match:
        return raw_value.strip()
    year, month, day = match.groups()
    return f"{year}-{month}-{day}"


def _build_report_id(row: pd.Series) -> str:
    source = "|".join(
        [
            str(row.get("publish_date", "")),
            str(row.get("report_title", "")),
            str(row.get("source_path", "")),
        ]
    )
    return f"report_{sha1(source.encode('utf-8')).hexdigest()[:16]}"


def _extract_title(report_text: str) -> str:
    for line in report_text.splitlines():
        stripped = line.strip()
        if stripped.startswith("# "):
            return stripped[2:].strip()
    return ""


def _extract_conclusion(report_text: str) -> str:
    parsed = _split_sections(report_text)
    return "\n".join(
        filter(
            None,
            [
                parsed.sections.get("结论", "").strip(),
                parsed.sections.get("总结", "").strip(),
            ],
        )
    )


def _count_mentions(text: str, stock_name: str, stock_code: str) -> int:
    total = 0
    if stock_name:
        total += text.count(stock_name)
    if stock_code:
        total += text.upper().count(stock_code.upper())
        total += text.upper().count(stock_code.split(".")[0].upper())
    return total


def _normalize_report_month(value: Any) -> str:
    if pd.isna(value):
        return ""

    text = str(value).strip()
    if not text:
        return ""

    try:
        return str(pd.Period(text, freq="M"))
    except Exception:
        match = re.search(r"(?P<year>\d{4})\D*(?P<month>\d{1,2})", text)
        if not match:
            return ""
        month = int(match.group("month"))
        if month < 1 or month > 12:
            return ""
        return f"{int(match.group('year')):04d}-{month:02d}"


def _is_single_month_markdown_dir(name: str) -> bool:
    return bool(re.match(r"^\d{6}_.+", str(name).strip()))


def _month_from_directory_name(name: str) -> str:
    text = str(name).strip()
    match = re.match(r"^(?P<year>\d{4})(?P<month>\d{2})(?:\D|$)", text)
    if not match:
        return ""
    month = int(match.group("month"))
    if month < 1 or month > 12:
        return ""
    return f"{int(match.group('year')):04d}-{month:02d}"
