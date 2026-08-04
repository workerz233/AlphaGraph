from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Callable
from urllib import error, request
import warnings

import pandas as pd

from alphagraph.data.market_data import normalize_baostock_code


LLM_KG_RELATION_TYPES = (
    "core_subject",
    "peer",
    "positive_driver",
    "risk_pressure",
)
LLM_KG_SENTIMENTS = ("positive", "neutral", "risk")
LLM_KG_COLUMNS = [
    "report_id",
    "report_month",
    "relation_type",
    "head_type",
    "head_code",
    "head_name",
    "tail_type",
    "tail_code",
    "tail_name",
    "topic",
    "driver_type",
    "risk_type",
    "sentiment",
    "confidence",
    "evidence_text",
    "source_section",
]


def empty_llm_relations() -> pd.DataFrame:
    return pd.DataFrame(columns=LLM_KG_COLUMNS)


def parse_llm_relation_response(
    response_text: str,
    report_id: str,
    report_month: str,
    min_confidence: float = 0.0,
    allowed_stock_codes: set[str] | None = None,
) -> pd.DataFrame:
    payload = _load_json_object(response_text)
    raw_relations = payload.get("relations", payload if isinstance(payload, list) else [])
    if not isinstance(raw_relations, list):
        return empty_llm_relations()

    allowed_codes = {_normalize_stock_code(code) for code in (allowed_stock_codes or set()) if str(code or "").strip()}
    rows: list[dict[str, Any]] = []
    for relation in raw_relations:
        if not isinstance(relation, dict):
            continue
        relation_type = str(relation.get("relation_type", "")).strip()
        if relation_type not in LLM_KG_RELATION_TYPES:
            continue
        confidence = _safe_float(relation.get("confidence", 0.0), default=0.0)
        if confidence < float(min_confidence):
            continue
        sentiment = str(relation.get("sentiment", "neutral") or "neutral").strip().lower()
        if sentiment not in LLM_KG_SENTIMENTS:
            sentiment = _default_sentiment(relation_type)
        head_code = _normalize_stock_code(relation.get("head_code", ""))
        tail_code = _normalize_stock_code(relation.get("tail_code", ""))
        if allowed_codes and not _relation_within_allowed_stocks(relation_type, head_code, tail_code, allowed_codes):
            continue
        rows.append(
            {
                "report_id": str(report_id),
                "report_month": str(report_month or ""),
                "relation_type": relation_type,
                "head_type": _clean_text(relation.get("head_type", "")),
                "head_code": head_code,
                "head_name": _clean_text(relation.get("head_name", "")),
                "tail_type": _clean_text(relation.get("tail_type", "")),
                "tail_code": tail_code,
                "tail_name": _clean_text(relation.get("tail_name", "")),
                "topic": _clean_text(relation.get("topic", "")),
                "driver_type": _clean_text(relation.get("driver_type", "")),
                "risk_type": _clean_text(relation.get("risk_type", "")),
                "sentiment": sentiment,
                "confidence": float(confidence),
                "evidence_text": _clean_text(relation.get("evidence_text", "")),
                "source_section": _clean_text(relation.get("source_section", "")),
            }
        )
    if not rows:
        return empty_llm_relations()
    return pd.DataFrame(rows, columns=LLM_KG_COLUMNS)


def extract_report_llm_relations(
    reports_df: pd.DataFrame,
    base_url: str,
    api_key: str,
    model: str,
    max_chars: int = 12000,
    min_confidence: float = 0.0,
    cache_dir: str | Path | None = None,
    enabled: bool = True,
    timeout: int = 120,
    raw_log_dir: str | Path | None = None,
    progress_callback: Callable[[str], None] | None = None,
) -> pd.DataFrame:
    if not enabled or reports_df.empty:
        _emit_progress(
            progress_callback,
            f"[kg][llm] 跳过 LLM 关系抽取：enabled={enabled}, reports={len(reports_df)}",
        )
        return empty_llm_relations()
    if not str(base_url or "").strip():
        raise ValueError("llm_kg_base_url is required when LLM KG extraction is enabled")
    if not str(api_key or "").strip():
        raise ValueError("llm_kg_api_key or LLM_KG_API_KEY is required when LLM KG extraction is enabled")
    if not str(model or "").strip():
        raise ValueError("llm_kg_model is required when LLM KG extraction is enabled")

    cache_path = Path(cache_dir).expanduser() if cache_dir else None
    if cache_path is not None:
        cache_path.mkdir(parents=True, exist_ok=True)
    raw_log_path = Path(raw_log_dir).expanduser() if raw_log_dir else None
    if raw_log_path is not None:
        raw_log_path.mkdir(parents=True, exist_ok=True)

    frames: list[pd.DataFrame] = []
    records = reports_df.to_dict(orient="records")
    total = len(records)
    for index, report in enumerate(records, start=1):
        report_id = str(report.get("report_id", "") or "")
        if not report_id:
            _emit_progress(progress_callback, f"[kg][llm] {index}/{total} 跳过：report_id 为空")
            continue
        report_month = str(report.get("report_month", "") or "")
        report_text = _report_text(report, max_chars=max_chars)
        if not report_text:
            _emit_progress(progress_callback, f"[kg][llm] {index}/{total} 跳过：report_id={report_id} 文本为空")
            continue
        candidate_stocks = report_candidate_stocks(report)
        candidate_codes = {stock["stock_code"] for stock in candidate_stocks}
        if len(candidate_codes) < 2:
            _emit_progress(
                progress_callback,
                f"[kg][llm] {index}/{total} 跳过：report_id={report_id} 候选股票少于2只",
            )
            _write_raw_llm_log(
                raw_log_path,
                report_id=report_id,
                report_month=report_month,
                model=model,
                base_url=base_url,
                status="skipped_single_stock",
                prompt="",
                report_text=report_text,
                candidate_stocks=candidate_stocks,
            )
            frames.append(empty_llm_relations())
            continue
        prompt = _build_prompt(report_text, candidate_stocks=candidate_stocks)
        cache_file = _cache_file(cache_path, report_id, report_text, model, prompt) if cache_path else None
        response_text = ""
        response_source = "api"
        if cache_file is not None and cache_file.exists():
            _emit_progress(
                progress_callback,
                f"[kg][llm] {index}/{total} 命中缓存：report_id={report_id} candidates={len(candidate_codes)}",
            )
            response_text = cache_file.read_text(encoding="utf-8")
            response_source = "cache"
        else:
            try:
                _emit_progress(
                    progress_callback,
                    f"[kg][llm] {index}/{total} 请求API：report_id={report_id} "
                    f"month={report_month} chars={len(report_text)} candidates={len(candidate_codes)}",
                )
                response_text = call_openai_compatible_chat(
                    base_url=base_url,
                    api_key=api_key,
                    model=model,
                    prompt=prompt,
                    timeout=timeout,
                )
            except Exception as exc:
                _write_raw_llm_log(
                    raw_log_path,
                    report_id=report_id,
                    report_month=report_month,
                    model=model,
                    base_url=base_url,
                    status="request_failed",
                    prompt=prompt,
                    report_text=report_text,
                    candidate_stocks=candidate_stocks,
                    error=exc,
                )
                _emit_progress(
                    progress_callback,
                    f"[kg][llm] {index}/{total} 请求失败：report_id={report_id} error={type(exc).__name__}: {exc}",
                )
                warnings.warn(f"LLM KG extraction failed for report_id={report_id}: {exc}")
                frames.append(empty_llm_relations())
                continue
            if cache_file is not None:
                cache_file.write_text(response_text, encoding="utf-8")
        try:
            parsed = parse_llm_relation_response(
                response_text,
                report_id=report_id,
                report_month=report_month,
                min_confidence=min_confidence,
                allowed_stock_codes=candidate_codes,
            )
            _write_raw_llm_log(
                raw_log_path,
                report_id=report_id,
                report_month=report_month,
                model=model,
                base_url=base_url,
                status="parsed",
                prompt=prompt,
                report_text=report_text,
                candidate_stocks=candidate_stocks,
                response_text=response_text,
                response_source=response_source,
                parsed_relation_count=int(len(parsed)),
            )
            _emit_progress(
                progress_callback,
                f"[kg][llm] {index}/{total} 解析完成：report_id={report_id} "
                f"relations={len(parsed)} source={response_source}",
            )
            frames.append(parsed)
        except Exception as exc:
            _write_raw_llm_log(
                raw_log_path,
                report_id=report_id,
                report_month=report_month,
                model=model,
                base_url=base_url,
                status="parse_failed",
                prompt=prompt,
                report_text=report_text,
                candidate_stocks=candidate_stocks,
                response_text=response_text,
                response_source=response_source,
                error=exc,
            )
            _emit_progress(
                progress_callback,
                f"[kg][llm] {index}/{total} 解析失败：report_id={report_id} error={type(exc).__name__}: {exc}",
            )
            warnings.warn(f"LLM KG response parse failed for report_id={report_id}: {exc}")
            frames.append(empty_llm_relations())
    if not frames:
        return empty_llm_relations()
    return pd.concat(frames, ignore_index=True).reindex(columns=LLM_KG_COLUMNS)


def call_openai_compatible_chat(
    base_url: str,
    api_key: str,
    model: str,
    prompt: str,
    timeout: int = 120,
) -> str:
    endpoint = _chat_completions_endpoint(base_url)
    payload: dict[str, Any] = {
        "model": model,
        "temperature": 0,
        "messages": [
            {"role": "system", "content": "你是金融研报关系抽取器，只返回严格 JSON。"},
            {"role": "user", "content": prompt},
        ],
    }
    payload["response_format"] = {"type": "json_object"}
    try:
        return _post_openai_compatible_chat(endpoint, api_key, payload, timeout)
    except error.HTTPError as exc:
        if exc.code not in {400, 422}:
            raise RuntimeError(_format_http_error(exc)) from exc
        payload.pop("response_format", None)
        try:
            return _post_openai_compatible_chat(endpoint, api_key, payload, timeout)
        except error.HTTPError as retry_exc:
            raise RuntimeError(_format_http_error(retry_exc)) from retry_exc


def _chat_completions_endpoint(base_url: str) -> str:
    normalized = str(base_url or "").strip().rstrip("/")
    if normalized.endswith("/v1"):
        return normalized + "/chat/completions"
    return normalized + "/v1/chat/completions"


def _format_http_error(exc: error.HTTPError) -> str:
    try:
        body = exc.read().decode("utf-8", errors="replace").strip()
    except Exception:
        body = ""
    if body:
        return f"HTTP {exc.code} {exc.reason}: {body}"
    return f"HTTP {exc.code} {exc.reason}"


def _post_openai_compatible_chat(
    endpoint: str,
    api_key: str,
    payload: dict[str, Any],
    timeout: int,
) -> str:
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = request.Request(
        endpoint,
        data=data,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    with request.urlopen(req, timeout=timeout) as response:
        response_payload = json.loads(response.read().decode("utf-8"))
    return str(response_payload["choices"][0]["message"]["content"])


def report_candidate_stocks(report: dict[str, Any]) -> list[dict[str, str]]:
    raw_text = str(report.get("raw_text", "") or "")
    candidates: list[dict[str, str]] = []
    candidates.extend(_candidate_stocks_from_source_path(str(report.get("source_path", "") or "")))
    candidates.extend(_candidate_stocks_from_related_section(raw_text))
    return _dedupe_candidate_stocks(candidates)


def _candidate_stocks_from_source_path(source_path: str) -> list[dict[str, str]]:
    stem = Path(str(source_path or "")).stem
    candidates: list[dict[str, str]] = []
    for match in re.finditer(r"(?<!\d)(?P<code>\d{6})(?P<name>[\u4e00-\u9fffA-Za-z]{1,24})", stem):
        name = _clean_stock_name(match.group("name"))
        if name:
            candidates.append({"stock_code": _normalize_stock_code(match.group("code")), "stock_name": name})
    return candidates


def _candidate_stocks_from_related_section(raw_text: str) -> list[dict[str, str]]:
    section_text = _related_stock_section_text(raw_text)
    candidates: list[dict[str, str]] = []
    patterns = [
        re.compile(r"(?P<name>[\u4e00-\u9fffA-Za-z]{2,24})\s*[（(]\s*(?P<code>\d{6})(?:\.(?:SZ|SH|BJ))?\s*[）)]", re.IGNORECASE),
        re.compile(r"(?P<name>[\u4e00-\u9fffA-Za-z]{2,24})\s*[（(]\s*(?P<code>\d{6}\.(?:SZ|SH|BJ))\s*[）)]", re.IGNORECASE),
        re.compile(r"(?P<name>[\u4e00-\u9fffA-Za-z]{2,24})\s*[：:]\s*(?P<code>\d{6}\.(?:SZ|SH|BJ)|\d{6})", re.IGNORECASE),
    ]
    for pattern in patterns:
        for match in pattern.finditer(section_text):
            name = _clean_stock_name(match.group("name"))
            if name:
                candidates.append({"stock_code": _normalize_stock_code(match.group("code")), "stock_name": name})
    return candidates


def _related_stock_section_text(raw_text: str) -> str:
    lines = str(raw_text or "").splitlines()
    for index, line in enumerate(lines):
        if "关联标的" not in line:
            continue
        section_lines: list[str] = []
        for section_line in lines[index + 1 :]:
            stripped = section_line.strip()
            if stripped.startswith("##") and "关联标的" not in stripped:
                break
            section_lines.append(section_line)
        return "\n".join(section_lines)
    return str(raw_text or "")


def _dedupe_candidate_stocks(candidates: list[dict[str, str]]) -> list[dict[str, str]]:
    deduped: dict[str, str] = {}
    for candidate in candidates:
        code = _normalize_stock_code(candidate.get("stock_code", ""))
        name = _clean_stock_name(candidate.get("stock_name", ""))
        if not code:
            continue
        if code not in deduped or (not deduped[code] and name):
            deduped[code] = name
    return [{"stock_code": code, "stock_name": name} for code, name in deduped.items()]


def _clean_stock_name(value: Any) -> str:
    text = str(value or "").strip()
    text = re.sub(r"^[\s\-_：:、，,0-9]+", "", text)
    text = re.sub(r"[_\-\s]+$", "", text)
    text = re.sub(r"(_\d+)$", "", text)
    return text.strip()


def _relation_within_allowed_stocks(
    relation_type: str,
    head_code: str,
    tail_code: str,
    allowed_codes: set[str],
) -> bool:
    del relation_type
    return head_code in allowed_codes and tail_code in allowed_codes and head_code != tail_code


def _build_prompt(report_text: str, candidate_stocks: list[dict[str, str]] | None = None) -> str:
    relation_types = ", ".join(LLM_KG_RELATION_TYPES)
    stock_lines = "\n".join(
        f"- {stock['stock_code']} {stock['stock_name']}".rstrip()
        for stock in (candidate_stocks or [])
    )
    stock_scope = (
        "只抽取这些股票之间的关系；head_code 和 tail_code 必须来自下列股票白名单。\n"
        f"{stock_lines}\n"
        "每条关系必须是两只不同白名单股票之间的股票-股票边。\n"
        "不要抽取白名单之外的公司、行业、政策、需求、风险等实体作为 head 或 tail。\n"
        "core_subject 表示两只股票同为研报核心标的或核心推荐组合。\n"
        "peer 表示两只股票同属可比公司、同赛道、同行业或同一投资篮子。\n"
        "positive_driver 表示两只股票共享同一个正向驱动因素，请把共享因素写入 topic 或 driver_type。\n"
        "risk_pressure 表示两只股票共享同一个风险压力，请把共享风险写入 topic 或 risk_type。\n"
        if stock_lines
        else ""
    )
    return (
        "请从下面中文金融研报中抽取关系。只允许 relation_type 为："
        f"{relation_types}。\n"
        f"{stock_scope}"
        "不要抽取上下游、客户、供应商关系，不要输出 competition。\n"
        "返回 JSON 对象，格式为 {\"relations\": [ ... ]}。每条关系包含："
        "relation_type, head_type, head_code, head_name, tail_type, tail_code, tail_name, "
        "topic, driver_type, risk_type, sentiment, confidence, evidence_text, source_section。\n"
        "股票代码尽量返回 000001.SZ 或 600000.SH 格式；confidence 为 0 到 1。\n\n"
        f"研报文本：\n{report_text}"
    )


def _load_json_object(text: str) -> Any:
    cleaned = _strip_code_fence(str(text or "").strip())
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start >= 0 and end > start:
            return json.loads(cleaned[start : end + 1])
        start = cleaned.find("[")
        end = cleaned.rfind("]")
        if start >= 0 and end > start:
            return json.loads(cleaned[start : end + 1])
        raise


def _strip_code_fence(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        return "\n".join(lines).strip()
    return stripped


def _report_text(report: dict[str, Any], max_chars: int) -> str:
    text = "\n".join(
        str(report.get(column, "") or "")
        for column in ("report_title", "title_text", "conclusion_text", "summary_text", "raw_text")
        if str(report.get(column, "") or "").strip()
    )
    return text[: max(int(max_chars), 1)]


def _cache_file(cache_dir: Path | None, report_id: str, report_text: str, model: str, prompt: str = "") -> Path | None:
    if cache_dir is None:
        return None
    digest = hashlib.sha256(f"stock_pair_v2\n{model}\n{report_id}\n{report_text}\n{prompt}".encode("utf-8")).hexdigest()[:24]
    return cache_dir / f"{report_id}-{digest}.json"


def _write_raw_llm_log(
    raw_log_dir: Path | None,
    report_id: str,
    report_month: str,
    model: str,
    base_url: str,
    status: str,
    prompt: str,
    report_text: str,
    response_text: str | None = None,
    response_source: str = "api",
    parsed_relation_count: int | None = None,
    candidate_stocks: list[dict[str, str]] | None = None,
    error: Exception | None = None,
) -> None:
    if raw_log_dir is None:
        return
    digest = hashlib.sha256(f"{model}\n{report_id}\n{report_text}".encode("utf-8")).hexdigest()[:24]
    payload: dict[str, Any] = {
        "report_id": report_id,
        "report_month": report_month,
        "model": model,
        "base_url": base_url,
        "status": status,
        "response_source": response_source,
        "report_text_chars": len(report_text),
        "prompt_chars": len(prompt),
        "prompt": prompt,
        "report_text": report_text,
    }
    if candidate_stocks is not None:
        payload["candidate_stocks"] = candidate_stocks
    if response_text is not None:
        payload["response_text"] = response_text
        payload["response_text_chars"] = len(response_text)
    if parsed_relation_count is not None:
        payload["parsed_relation_count"] = int(parsed_relation_count)
    if error is not None:
        payload["error_type"] = type(error).__name__
        payload["error"] = str(error)
    raw_log_dir.mkdir(parents=True, exist_ok=True)
    (raw_log_dir / f"{report_id}-{digest}.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _normalize_stock_code(value: Any) -> str:
    raw = str(value or "").strip()
    if not raw:
        return ""
    try:
        return normalize_baostock_code(raw)
    except Exception:
        return raw.lower()


def _clean_text(value: Any) -> str:
    return str(value or "").strip()


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        parsed = float(value)
    except Exception:
        return default
    if pd.isna(parsed):
        return default
    return parsed


def _default_sentiment(relation_type: str) -> str:
    if relation_type == "positive_driver":
        return "positive"
    if relation_type == "risk_pressure":
        return "risk"
    return "neutral"


def _emit_progress(callback: Callable[[str], None] | None, message: str) -> None:
    if callback is not None:
        callback(message)
