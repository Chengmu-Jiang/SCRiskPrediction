from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from src.scnews_sentiment.llm_tone.data import (
    build_samples,
    load_daily_news,
    write_daily_compressions_real,
    write_sample_metadata,
)
from src.scnews_sentiment.llm_tone.eval import select_high_loss_predictions
from src.scnews_sentiment.llm_tone.llm_clients import BaseLLM, get_llm
from src.scnews_sentiment.llm_tone.prompts import (
    build_compression_prompt,
    build_forecast_prompt,
    build_prompt_feedback_agent_prompt,
    get_forecast_prompt_template,
    merge_guidance,
    parse_agent_suggestions,
    save_prompt_versions,
)
from src.scnews_sentiment.llm_tone.scorers import evaluate_fixed_scorers
from src.scnews_sentiment.llm_tone.train import train_mlp
from src.scnews_sentiment.llm_tone.utils import (
    clean_generated_text,
    clone_args,
    ensure_dir,
    load_json_file,
    read_csv_or_empty,
    save_csv,
    stable_hash,
    truncate_text,
)


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return str(value)


def _write_empty_generation_log(
    args: argparse.Namespace,
    llm: BaseLLM,
    task: str,
    item_id: str,
    attempt: int,
    max_attempts: int,
    prompt_hash: str,
    cache_key: str,
) -> Path:
    log_path = args.output_dir / "llm_empty_generations.jsonl"
    ensure_dir(log_path.parent)
    record = {
        "task": task,
        "item_id": item_id,
        "attempt": attempt,
        "max_attempts": max_attempts,
        "prompt_hash": prompt_hash,
        "cache_key": cache_key,
        "response_info": _json_safe(getattr(llm, "last_response_info", {})),
    }
    with log_path.open("a", encoding="utf-8") as handle:
        json.dump(record, handle, ensure_ascii=False)
        handle.write("\n")
    return log_path


def _generate_non_empty(
    llm: BaseLLM,
    prompt: str,
    max_tokens: int,
    args: argparse.Namespace,
    task: str,
    item_id: str,
    prompt_hash: str,
    cache_key: str,
) -> str:
    max_attempts = max(1, int(getattr(args, "llm_empty_retries", 1)))
    retry_sleep = max(0.0, float(getattr(args, "llm_empty_retry_sleep", 0.0)))
    log_path: Optional[Path] = None

    for attempt in range(1, max_attempts + 1):
        generated = llm.generate(prompt, max_tokens=max_tokens)
        if generated:
            return generated

        log_path = _write_empty_generation_log(
            args=args,
            llm=llm,
            task=task,
            item_id=item_id,
            attempt=attempt,
            max_attempts=max_attempts,
            prompt_hash=prompt_hash,
            cache_key=cache_key,
        )
        if attempt < max_attempts:
            print(
                f"[{task}] empty LLM response for {item_id}; "
                f"retrying {attempt + 1}/{max_attempts}"
            )
            if retry_sleep > 0:
                time.sleep(retry_sleep)

    detail = f" See {log_path} for response metadata." if log_path else ""
    raise RuntimeError(
        f"LLM returned empty {task} for {item_id} after {max_attempts} attempt(s).{detail}"
    )


def ensure_daily_compressions(news: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    output_path = args.output_dir / "daily_compressions.csv"
    existing = read_csv_or_empty(output_path)
    existing_map: Dict[Tuple[str, str], Dict[str, Any]] = {}
    if not existing.empty:
        for _, row in existing.iterrows():
            existing_map[(str(row.get("daily_id", "")), str(row.get("cache_key", "")))] = row.to_dict()

    llm: Optional[BaseLLM] = None
    rows: List[Dict[str, Any]] = []
    for idx, row in news.iterrows():
        prompt = build_compression_prompt(row)
        prompt_hash = stable_hash(prompt, length=32)
        cache_key = stable_hash(row["daily_id"], row["raw_text_hash"], prompt_hash, length=32)
        cached = existing_map.get((row["daily_id"], cache_key))
        if cached and not args.force_llm:
            cached_text = clean_generated_text(str(cached.get("compressed_text", "")))
            if cached_text:
                cached = dict(cached)
                cached["compressed_text"] = cached_text
                rows.append(cached)
                continue

        if llm is None:
            llm = get_llm(args)
        print(f"[compress] {idx + 1}/{len(news)} {row['daily_id']}")
        compressed = _generate_non_empty(
            llm=llm,
            prompt=prompt,
            max_tokens=args.compress_max_tokens,
            args=args,
            task="compression",
            item_id=str(row["daily_id"]),
            prompt_hash=prompt_hash,
            cache_key=cache_key,
        )
        rows.append(
            {
                "daily_id": row["daily_id"],
                "topic": row["topic"],
                "hs_code_4d": row["hs_code_4d"],
                "hs_short_label": row["hs_short_label"],
                "date_iso": row["date_iso"],
                "date_yyyymmdd": row["date_yyyymmdd"],
                "article_score": row["article_score"],
                "raw_text_hash": row["raw_text_hash"],
                "prompt_hash": prompt_hash,
                "cache_key": cache_key,
                "compressed_text": compressed,
            }
        )
        if args.request_sleep > 0:
            time.sleep(args.request_sleep)
        save_csv(pd.DataFrame(rows), output_path)

    result = pd.DataFrame(rows).sort_values(["topic", "date_yyyymmdd"]).reset_index(drop=True)
    save_csv(result, output_path)
    return result


def ensure_forecasts(
    samples: pd.DataFrame,
    compressions: pd.DataFrame,
    args: argparse.Namespace,
) -> pd.DataFrame:
    output_path = args.output_dir / "forecast_events.csv"
    existing = read_csv_or_empty(output_path)
    existing_map: Dict[Tuple[str, str], Dict[str, Any]] = {}
    if not existing.empty:
        for _, row in existing.iterrows():
            existing_map[(str(row.get("sample_id", "")), str(row.get("cache_key", "")))] = row.to_dict()

    comp = compressions.copy()
    comp_by_id = comp.set_index("daily_id", drop=False)
    llm: Optional[BaseLLM] = None
    rows: List[Dict[str, Any]] = []

    for idx, sample in samples.iterrows():
        history_ids = [item for item in str(sample["history_daily_ids"]).split(";") if item]
        missing = [item for item in history_ids if item not in comp_by_id.index]
        if missing:
            raise ValueError(f"Missing compressed history rows for sample {sample['sample_id']}: {missing}")

        history_rows = comp_by_id.loc[history_ids].copy()
        if isinstance(history_rows, pd.Series):
            history_rows = history_rows.to_frame().T
        history_rows["date_sort"] = pd.to_datetime(history_rows["date_iso"])
        history_rows = history_rows.sort_values("date_sort")

        prompt = build_forecast_prompt(
            sample,
            history_rows,
            prompt_guidance=getattr(args, "forecast_prompt_guidance", ""),
            prompt_path=getattr(args, "forecast_prompt_path", None),
        )
        prompt_hash = stable_hash(prompt, length=32)
        history_cache = "|".join(history_rows["cache_key"].astype(str).tolist())
        cache_key = stable_hash(sample["sample_id"], history_cache, prompt_hash, length=32)
        cached = existing_map.get((sample["sample_id"], cache_key))
        if cached and not args.force_llm:
            cached_text = clean_generated_text(str(cached.get("predicted_event_text", "")))
            if cached_text:
                cached = dict(cached)
                cached["predicted_event_text"] = cached_text
                rows.append(cached)
                continue

        if llm is None:
            llm = get_llm(args)
        print(f"[forecast] {idx + 1}/{len(samples)} {sample['sample_id']} {sample['topic']}->{sample['target_date']}")
        predicted_event = _generate_non_empty(
            llm=llm,
            prompt=prompt,
            max_tokens=args.forecast_max_tokens,
            args=args,
            task="forecast",
            item_id=f"sample {sample['sample_id']}",
            prompt_hash=prompt_hash,
            cache_key=cache_key,
        )

        rows.append(
            {
                "sample_id": sample["sample_id"],
                "split": sample["split"],
                "topic": sample["topic"],
                "hs_code_4d": sample["hs_code_4d"],
                "hs_short_label": sample["hs_short_label"],
                "target_date": sample["target_date"],
                "target_article_score": sample["target_article_score"],
                "target_daily_id": sample["target_daily_id"],
                "topic_order": sample["topic_order"],
                "history_len": sample["history_len"],
                "history_start_date": sample["history_start_date"],
                "last_history_date": sample["last_history_date"],
                "history_daily_ids": sample["history_daily_ids"],
                "history_dates": sample["history_dates"],
                "history_compression_keys": history_cache,
                "prompt_hash": prompt_hash,
                "cache_key": cache_key,
                "predicted_event_text": predicted_event,
            }
        )
        if args.request_sleep > 0:
            time.sleep(args.request_sleep)
        save_csv(pd.DataFrame(rows), output_path)

    result = pd.DataFrame(rows).sort_values(["topic", "target_date"]).reset_index(drop=True)
    save_csv(result, output_path)
    return result


def build_prompt_feedback_examples(
    high_loss_rows: pd.DataFrame,
    compressions: pd.DataFrame,
    news: pd.DataFrame,
    args: argparse.Namespace,
) -> pd.DataFrame:
    comp_by_id = compressions.set_index("daily_id", drop=False)
    news_by_id = news.set_index("daily_id", drop=False)
    rows: List[Dict[str, Any]] = []

    for _, row in high_loss_rows.iterrows():
        history_ids = [item for item in str(row["history_daily_ids"]).split(";") if item]
        history_blocks: List[str] = []
        for pos, daily_id in enumerate(history_ids, start=1):
            if daily_id not in comp_by_id.index:
                continue
            comp_row = comp_by_id.loc[daily_id]
            history_blocks.append(
                f"[{pos}] {comp_row['date_iso']} {truncate_text(comp_row['compressed_text'], args.prompt_feedback_history_chars)}"
            )

        target_daily_id = str(row["target_daily_id"])
        if target_daily_id not in news_by_id.index:
            raise ValueError(f"Missing true target news for {target_daily_id}")
        true_row = news_by_id.loc[target_daily_id]

        rows.append(
            {
                "sample_id": row["sample_id"],
                "topic": row["topic"],
                "target_date": row["target_date"],
                "history_len": row["history_len"],
                "target_article_score": row["target_article_score"],
                "predicted_article_score": row["predicted_article_score"],
                "squared_error": row["squared_error"],
                "abs_error": row["abs_error"],
                "history_compressed_news": "\n\n".join(history_blocks),
                "generated_future_event": truncate_text(
                    row["predicted_event_text"],
                    args.prompt_feedback_text_chars,
                ),
                "true_next_news_title": true_row["title"],
                "true_next_news_text": truncate_text(
                    true_row["article_text"],
                    args.prompt_feedback_text_chars,
                ),
            }
        )
    return pd.DataFrame(rows)


def run_prompt_version_round(
    version: int,
    guidance: str,
    samples: pd.DataFrame,
    compressions: pd.DataFrame,
    args: argparse.Namespace,
    opt_dir: Path,
) -> Tuple[Path, pd.DataFrame, Dict[str, Any]]:
    round_dir = opt_dir / f"round_{version:02d}"
    round_args = clone_args(
        args,
        output_dir=round_dir,
        forecast_prompt_guidance=guidance,
    )
    ensure_dir(round_dir)
    write_sample_metadata(samples, round_args)
    forecasts = ensure_forecasts(samples, compressions, round_args)
    train_mlp(forecasts, round_args)
    predictions = pd.read_csv(round_dir / "tone_predictions.csv", encoding="utf-8-sig")
    metrics = load_json_file(round_dir / "metrics.json")
    return round_dir, predictions, metrics


def request_prompt_suggestions(
    round_index: int,
    guidance: str,
    examples: pd.DataFrame,
    args: argparse.Namespace,
) -> Tuple[str, str, List[str]]:
    llm = get_llm(args)
    prompt = build_prompt_feedback_agent_prompt(
        round_index,
        guidance,
        examples,
        prompt_path=getattr(args, "optimizer_prompt_path", None),
    )
    raw_text = llm.generate(prompt, max_tokens=args.prompt_agent_max_tokens)
    if not raw_text:
        raise RuntimeError(f"Prompt feedback agent returned empty text at round {round_index}.")
    diagnosis, suggestions = parse_agent_suggestions(raw_text)
    if not suggestions:
        raise RuntimeError(f"Prompt feedback agent returned no usable suggestions at round {round_index}.")
    return diagnosis, raw_text, suggestions


def optimize_prompt_with_language_feedback(
    news: pd.DataFrame,
    samples: pd.DataFrame,
    compressions: pd.DataFrame,
    args: argparse.Namespace,
) -> None:
    opt_dir = args.output_dir / "prompt_optimization"
    ensure_dir(opt_dir)

    train_val_samples = samples[samples["split"].isin(["train", "val"])].copy()
    test_samples = samples[samples["split"].astype(str) == "test"].copy()
    if train_val_samples.empty or test_samples.empty:
        raise ValueError("Prompt optimization requires non-empty train/val and test samples.")

    prompt_versions: List[Dict[str, Any]] = []
    round_records: List[Dict[str, Any]] = []
    guidance = str(getattr(args, "forecast_prompt_guidance", "") or "").strip()

    for version in range(args.prompt_optimization_rounds + 1):
        guidance_hash = stable_hash(guidance or "original", length=12)
        prompt_versions.append(
            {
                "version": version,
                "guidance": guidance,
                "guidance_hash": guidance_hash,
                "forecast_prompt_path": (
                    None
                    if getattr(args, "forecast_prompt_path", None) is None
                    else str(args.forecast_prompt_path)
                ),
                "optimizer_prompt_path": (
                    None
                    if getattr(args, "optimizer_prompt_path", None) is None
                    else str(args.optimizer_prompt_path)
                ),
                "prompt_template": get_forecast_prompt_template(
                    guidance,
                    prompt_path=getattr(args, "forecast_prompt_path", None),
                ),
            }
        )
        save_prompt_versions(prompt_versions, opt_dir)

        print(f"[prompt-opt] running prompt version {version} guidance={guidance_hash}")
        round_dir, predictions, metrics = run_prompt_version_round(
            version=version,
            guidance=guidance,
            samples=samples,
            compressions=compressions,
            args=args,
            opt_dir=opt_dir,
        )
        val_metrics = metrics["metrics"].get("val", {})
        test_metrics = metrics["metrics"].get("test", {})
        val_mse = val_metrics.get("mse")
        val_rmse = val_metrics.get("rmse")
        val_mae = val_metrics.get("mae")
        val_r2 = val_metrics.get("r2")
        test_mse = test_metrics.get("mse")
        test_rmse = test_metrics.get("rmse")
        test_mae = test_metrics.get("mae")
        test_r2 = test_metrics.get("r2")
        prompt_versions[-1].update(
            {
                "round_dir": str(round_dir),
                "val_mse": val_mse,
                "val_rmse": val_rmse,
                "val_mae": val_mae,
                "val_r2": val_r2,
                "test_mse": test_mse,
                "test_rmse": test_rmse,
                "test_mae": test_mae,
                "test_r2": test_r2,
            }
        )
        round_records.append(
            {
                "version": version,
                "guidance_hash": guidance_hash,
                "round_dir": str(round_dir),
                "val_mse": val_mse,
                "val_rmse": val_rmse,
                "val_mae": val_mae,
                "val_r2": val_r2,
                "test_mse": test_mse,
                "test_rmse": test_rmse,
                "test_mae": test_mae,
                "test_r2": test_r2,
            }
        )
        save_prompt_versions(prompt_versions, opt_dir)
        save_csv(pd.DataFrame(round_records), opt_dir / "round_metrics.csv")

        if version >= args.prompt_optimization_rounds:
            break

        high_loss = select_high_loss_predictions(
            predictions,
            top_fraction=args.prompt_loss_top_fraction,
        )
        examples = build_prompt_feedback_examples(high_loss, compressions, news, args)
        save_csv(examples, opt_dir / f"feedback_examples_for_version_{version:02d}.csv")

        diagnosis, raw_agent_text, suggestions = request_prompt_suggestions(
            round_index=version + 1,
            guidance=guidance,
            examples=examples,
            args=args,
        )
        feedback_record = {
            "from_version": version,
            "to_version": version + 1,
            "diagnosis": diagnosis,
            "raw_agent_text": raw_agent_text,
            "suggestions": suggestions,
        }
        with (opt_dir / f"agent_feedback_to_version_{version + 1:02d}.json").open(
            "w",
            encoding="utf-8",
        ) as handle:
            json.dump(feedback_record, handle, indent=2, ensure_ascii=False)

        guidance = merge_guidance(
            guidance,
            suggestions,
            max_bullets=args.prompt_guidance_max_bullets,
        )

    best_version = min(
        prompt_versions,
        key=lambda item: float("inf") if item.get("val_rmse") is None else float(item["val_rmse"]),
    )
    best_guidance = str(best_version["guidance"])
    best_round_dir = Path(str(best_version["round_dir"]))
    final_dir = opt_dir / "final"
    final_args = clone_args(
        args,
        output_dir=final_dir,
        forecast_prompt_guidance=best_guidance,
    )
    ensure_dir(final_dir)

    print(f"[prompt-opt] best prompt version {best_version['version']} by val RMSE")
    best_round_forecasts = pd.read_csv(
        best_round_dir / "forecast_events.csv",
        encoding="utf-8-sig",
    )
    best_round_ids = set(best_round_forecasts["sample_id"].astype(str))
    test_ids = set(test_samples["sample_id"].astype(str))
    if test_ids.issubset(best_round_ids):
        final_forecasts = best_round_forecasts.copy()
    else:
        missing_test_samples = test_samples[
            ~test_samples["sample_id"].astype(str).isin(best_round_ids)
        ].copy()
        test_forecasts = ensure_forecasts(missing_test_samples, compressions, final_args)
        final_forecasts = pd.concat([best_round_forecasts, test_forecasts], ignore_index=True)
    final_forecasts = final_forecasts.sort_values(["topic", "target_date"]).reset_index(drop=True)
    save_csv(final_forecasts, final_dir / "forecast_events.csv")
    write_sample_metadata(samples, final_args)
    train_mlp(final_forecasts, final_args)

    final_metrics = load_json_file(final_dir / "metrics.json")
    summary = {
        "best_version": best_version["version"],
        "best_guidance_hash": best_version["guidance_hash"],
        "selection_metric": "val_rmse",
        "prompt_versions": prompt_versions,
        "round_metrics": round_records,
        "final_metrics": final_metrics,
        "final_dir": str(final_dir),
    }
    with (opt_dir / "optimization_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)


def run_event_tone_pipeline(args: argparse.Namespace) -> None:
    ensure_dir(args.output_dir)

    news = load_daily_news(args)
    samples = build_samples(news, args)
    write_sample_metadata(samples, args)

    if args.stage == "build-samples":
        return

    if args.stage == "eval-fixed-scorers":
        evaluate_fixed_scorers(samples, args)
        return

    combined_prompt_and_fixed_scorer_stage = (
        args.stage == "optimize-prompt-and-eval-fixed-scorers"
    )
    compressions: Optional[pd.DataFrame] = None
    if args.stage in {"compress", "forecast", "all", "optimize-prompt"} or combined_prompt_and_fixed_scorer_stage:
        compressions = ensure_daily_compressions(news, args)
        write_daily_compressions_real(compressions, samples, args)
        if args.stage == "compress":
            return

    if args.stage == "optimize-prompt" or combined_prompt_and_fixed_scorer_stage:
        assert compressions is not None
        optimize_prompt_with_language_feedback(news, samples, compressions, args)
        if combined_prompt_and_fixed_scorer_stage:
            evaluate_fixed_scorers(samples, args)
        return

    if args.stage in {"forecast", "all"}:
        assert compressions is not None
        forecasts = ensure_forecasts(samples, compressions, args)
        if args.stage == "forecast":
            return
    else:
        forecast_path = args.output_dir / "forecast_events.csv"
        if not forecast_path.exists():
            raise FileNotFoundError(
                f"{forecast_path} does not exist. Run --stage forecast or --stage all first."
            )
        forecasts = pd.read_csv(forecast_path, encoding="utf-8-sig")

    if args.stage in {"train", "all"}:
        train_mlp(forecasts, args)
