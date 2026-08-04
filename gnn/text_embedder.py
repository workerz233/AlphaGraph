from __future__ import annotations

import argparse
import importlib
import inspect
import os
import re
from pathlib import Path
from typing import Callable

import pandas as pd


DEFAULT_EMBEDDING_MODEL_NAME = "Qwen/Qwen3-Embedding-0.6B"
DEFAULT_VLLM_GPU_MEMORY_UTILIZATION = 0.08
DEFAULT_VLLM_MAX_MODEL_LEN = 4096
Vector = list[float]
EmbedderFn = Callable[[list[str], str], list[Vector]]


def embed_report_texts(
    reports_df: pd.DataFrame,
    model_name: str,
    output_path: str | Path,
    embedder: EmbedderFn | None = None,
) -> pd.DataFrame:
    records = reports_df.to_dict(orient="records")
    if not records:
        embedding_df = pd.DataFrame(columns=["report_id"])
        _write_frame(embedding_df, Path(output_path))
        return embedding_df

    texts = [str(row.get("raw_text", "")) for row in records]
    embed_fn = embedder or _embed_with_vllm
    vectors = embed_fn(texts, model_name)

    if len(vectors) != len(records):
        raise ValueError("Embedding count does not match report count.")

    dimensions = {len(vector) for vector in vectors}
    if len(dimensions) != 1:
        raise ValueError("Embeddings must all have the same dimension.")

    embeddings = [
        {
            "report_id": row["report_id"],
            **{f"emb_{index}": value for index, value in enumerate(vector)},
        }
        for row, vector in zip(records, vectors, strict=True)
    ]

    embedding_df = pd.DataFrame(embeddings).sort_values("report_id", ignore_index=True)
    _write_frame(embedding_df, Path(output_path))
    return embedding_df


def _embed_with_vllm(texts: list[str], model_name: str) -> list[Vector]:
    try:
        from vllm import LLM
    except ImportError as exc:
        raise RuntimeError(
            "Missing dependency vllm>=0.8.5. Please install requirements first."
        ) from exc

    model_ref = _resolve_model_reference(model_name)
    llm = _build_vllm_embed_llm(LLM, model_ref)
    outputs = llm.embed(texts)
    return [_extract_embedding_vector(output) for output in outputs]


def _build_vllm_embed_llm(llm_cls: Callable[..., object], model_ref: str) -> object:
    llm_kwargs = _build_vllm_kwargs(llm_cls, model_ref)
    attempt_kwargs = dict(llm_kwargs)
    for _ in range(6):
        try:
            return llm_cls(**attempt_kwargs)
        except TypeError as exc:
            unsupported_kwarg = _extract_unexpected_kwarg(exc)
            if unsupported_kwarg in {"model", "trust_remote_code"}:
                break
            if unsupported_kwarg and unsupported_kwarg in attempt_kwargs:
                attempt_kwargs.pop(unsupported_kwarg, None)
                continue
            break

    # Fall back to the minimum constructor surface for unknown older builds.
    return llm_cls(model=model_ref, trust_remote_code=True)


def _build_vllm_kwargs(llm_cls: Callable[..., object], model_ref: str) -> dict[str, object]:
    kwargs: dict[str, object] = {
        "model": model_ref,
        "trust_remote_code": True,
    }
    signature = _safe_signature(llm_cls)
    if signature is None:
        kwargs["task"] = "embed"
        return kwargs

    parameters = signature.parameters
    accepts_var_kwargs = any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in parameters.values()
    )

    def _supports(name: str) -> bool:
        return accepts_var_kwargs or name in parameters

    if _supports("task"):
        kwargs["task"] = "embed"
    else:
        if _supports("runner"):
            kwargs["runner"] = "pooling"
        if _supports("convert"):
            kwargs["convert"] = "embed"

    if _supports("gpu_memory_utilization"):
        kwargs["gpu_memory_utilization"] = _read_env_float(
            name="EMBEDDING_VLLM_GPU_MEMORY_UTILIZATION",
            default=DEFAULT_VLLM_GPU_MEMORY_UTILIZATION,
            minimum=0.01,
            maximum=0.99,
        )
    if _supports("max_model_len"):
        kwargs["max_model_len"] = _read_env_int(
            name="EMBEDDING_VLLM_MAX_MODEL_LEN",
            default=DEFAULT_VLLM_MAX_MODEL_LEN,
            minimum=128,
        )
    if _supports("disable_log_stats"):
        kwargs["disable_log_stats"] = True
    return kwargs


def _safe_signature(callable_obj: Callable[..., object]) -> inspect.Signature | None:
    try:
        return inspect.signature(callable_obj)
    except (TypeError, ValueError):
        return None


def _extract_unexpected_kwarg(exc: TypeError) -> str | None:
    match = re.search(r"unexpected keyword argument ['\"]([^'\"]+)['\"]", str(exc))
    if not match:
        return None
    return match.group(1)


def _read_env_float(name: str, default: float, minimum: float, maximum: float) -> float:
    raw_value = os.getenv(name)
    if not raw_value:
        return default
    try:
        value = float(raw_value)
    except ValueError as exc:
        raise ValueError(f"{name} must be a float, got: {raw_value!r}") from exc
    if value < minimum or value > maximum:
        raise ValueError(f"{name} must be in [{minimum}, {maximum}], got: {value}")
    return value


def _read_env_int(name: str, default: int, minimum: int) -> int:
    raw_value = os.getenv(name)
    if not raw_value:
        return default
    try:
        value = int(raw_value)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got: {raw_value!r}") from exc
    if value < minimum:
        raise ValueError(f"{name} must be >= {minimum}, got: {value}")
    return value


def _extract_embedding_vector(output: object) -> Vector:
    outputs_obj = getattr(output, "outputs", None)
    if outputs_obj is not None and hasattr(outputs_obj, "embedding"):
        embedding = getattr(outputs_obj, "embedding")
        return [float(value) for value in embedding]

    if hasattr(output, "embedding"):
        embedding = getattr(output, "embedding")
        return [float(value) for value in embedding]

    raise ValueError("Unexpected vLLM embedding output format.")


def _resolve_model_reference(model_name: str) -> str:
    if not model_name:
        raise ValueError("model_name must not be empty.")

    local_path = Path(model_name)
    if local_path.exists():
        return str(local_path)

    explicit_modelscope = model_name.startswith("modelscope:")
    use_modelscope = explicit_modelscope or _is_truthy(os.getenv("EMBEDDING_USE_MODELSCOPE", ""))
    if not use_modelscope:
        return model_name

    repo_id = model_name.split(":", 1)[1] if explicit_modelscope else model_name
    return _download_from_modelscope(repo_id)


def _download_from_modelscope(repo_id: str) -> str:
    try:
        modelscope_module = importlib.import_module("modelscope")
    except ImportError as exc:
        raise RuntimeError(
            "ModelScope support requires modelscope. Please install requirements first."
        ) from exc

    snapshot_download = getattr(modelscope_module, "snapshot_download", None)
    if snapshot_download is None:
        snapshot_module = importlib.import_module("modelscope.hub.snapshot_download")
        snapshot_download = getattr(snapshot_module, "snapshot_download")

    cache_dir = os.getenv("MODELSCOPE_CACHE")
    if cache_dir:
        path = snapshot_download(model_id=repo_id, cache_dir=cache_dir)
    else:
        path = snapshot_download(model_id=repo_id)
    return str(path)


def _is_truthy(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def _write_frame(df: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        df.to_parquet(output_path, index=False)
    except Exception:
        df.to_pickle(output_path)


def _read_frame(path: Path) -> pd.DataFrame:
    try:
        return pd.read_parquet(path)
    except Exception:
        try:
            return pd.read_pickle(path)
        except Exception:
            return pd.read_csv(path)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Embed report texts with vLLM and Qwen3-Embedding-0.6B"
    )
    parser.add_argument("--reports-path", type=Path, required=True)
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument("--model-name", type=str, default=DEFAULT_EMBEDDING_MODEL_NAME)
    parser.add_argument("--id-column", type=str, default="report_id")
    parser.add_argument("--text-column", type=str, default="raw_text")
    parser.add_argument("--limit", type=int, default=0)
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    reports_df = _read_frame(args.reports_path)
    required_columns = [args.id_column, args.text_column]
    missing_columns = [column for column in required_columns if column not in reports_df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    use_df = reports_df[[args.id_column, args.text_column]].rename(
        columns={args.id_column: "report_id", args.text_column: "raw_text"}
    )
    if args.limit and args.limit > 0:
        use_df = use_df.head(args.limit).copy()

    embedding_df = embed_report_texts(
        reports_df=use_df,
        model_name=args.model_name,
        output_path=args.output_path,
    )

    embedding_dim = len([column for column in embedding_df.columns if column.startswith("emb_")])
    print(
        f"Embedded {len(embedding_df)} reports with dim={embedding_dim}. "
        f"Saved to {args.output_path}"
    )


if __name__ == "__main__":
    main()
