from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd

from src.scnews_sentiment.llm_tone.utils import clean_generated_text, ensure_dir


def _parse_prompt_yaml_fallback(path: Path) -> Dict[str, Any]:
    data: Dict[str, Any] = {}
    lines = path.read_text(encoding="utf-8").splitlines()
    index = 0
    while index < len(lines):
        raw_line = lines[index]
        stripped = raw_line.strip()
        if not stripped or stripped.startswith("#"):
            index += 1
            continue
        if raw_line.startswith((" ", "\t")) or ":" not in stripped:
            raise ValueError(f"Unsupported prompt YAML line in {path}: {raw_line}")
        key, raw_value = stripped.split(":", 1)
        key = key.strip()
        value = raw_value.strip()
        if value in {"|", "|-", "|+"}:
            block_lines: List[str] = []
            index += 1
            block_indent: Optional[int] = None
            while index < len(lines):
                block_line = lines[index]
                if block_line.strip() and not block_line.startswith((" ", "\t")):
                    break
                if block_line.strip() and block_indent is None:
                    block_indent = len(block_line) - len(block_line.lstrip(" "))
                trim = block_indent or 0
                block_lines.append(block_line[trim:] if len(block_line) >= trim else "")
                index += 1
            data[key] = "\n".join(block_lines).rstrip("\n")
            continue
        if value.lower() in {"true", "false"}:
            data[key] = value.lower() == "true"
        elif value.lower() in {"null", "none", "~"}:
            data[key] = None
        else:
            data[key] = value.strip("'\"")
        index += 1
    return data


def load_prompt_config(path: Optional[Path]) -> Dict[str, Any]:
    if path is None:
        return {}
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Prompt config does not exist: {path}")
    suffix = path.suffix.lower()
    if suffix == ".json":
        with path.open("r", encoding="utf-8") as handle:
            loaded = json.load(handle)
    elif suffix in {".yaml", ".yml"}:
        try:
            import yaml

            with path.open("r", encoding="utf-8") as handle:
                loaded = yaml.safe_load(handle) or {}
        except ModuleNotFoundError:
            loaded = _parse_prompt_yaml_fallback(path)
    else:
        raise ValueError(f"Unsupported prompt config file type: {path}")
    if not isinstance(loaded, dict):
        raise ValueError(f"Prompt config must contain a mapping/object: {path}")
    return loaded


def prompt_template_from_config(path: Optional[Path]) -> Optional[str]:
    config = load_prompt_config(path)
    template = config.get("template")
    if template is None:
        return None
    template = str(template)
    if not template.strip():
        raise ValueError(f"Prompt template is empty: {path}")
    return template


def build_compression_prompt(row: pd.Series) -> str:
    return f"""You are compressing one daily news article for a supply-chain sentiment forecasting task.

Topic/Product: {row['topic']}
HS4 code: {row['hs_code_4d']}
HS4 label: {row['hs_short_label']}
Date: {row['date_iso']}
Title: {row['title']}

News article:
{row['article_text']}

Task:
Write a compact factual event summary in English.

Constraints:
- Preserve actors, actions, dates or time references, policies, production or supply-chain signals, causal links, and tone-relevant facts.
- Do not add facts that are not in the article.
- Do not output a sentiment score.
- Do not include hidden reasoning, chain-of-thought, analysis notes, or <think> tags.
- Use 80 to 160 words.
- Return only the summary text."""


def format_forecast_guidance(prompt_guidance: str) -> str:
    prompt_guidance = str(prompt_guidance or "").strip()
    if not prompt_guidance:
        return ""
    return f"""

Additional tone-calibration guidance learned from training feedback:
{prompt_guidance}"""


def get_forecast_prompt_template(
    prompt_guidance: str = "",
    prompt_path: Optional[Path] = None,
) -> str:
    guidance_block = format_forecast_guidance(prompt_guidance)
    guidance_block = guidance_block.replace("{", "{{").replace("}", "}}")
    configured_template = prompt_template_from_config(prompt_path)
    if configured_template is not None:
        return configured_template.replace("{guidance_block}", guidance_block)

    return f"""You are forecasting the next effective news event for a product/topic.

Topic/Product: {{topic}}
HS4 code: {{hs_code_4d}}
HS4 label: {{hs_short_label}}
Last observed effective news date: {{last_history_date}}
Prediction target: the next effective news date after the last observed effective news date, not the next calendar day.

Historical compressed daily news sequence:
{{history_text}}

Task:
Generate a plausible future event description for the next effective news date.

Constraints:
- Use only the historical sequence as evidence.
- Focus on the likely event narrative and tone drivers.
- Do not output ArticleScore, AvgTone, labels, bullet points, or explanations.
- Do not include hidden reasoning, chain-of-thought, analysis notes, or <think> tags.
- Use 80 to 180 words.
- Return only the generated future-event description.{guidance_block}"""


def build_forecast_prompt(
    sample: pd.Series,
    history_rows: pd.DataFrame,
    prompt_guidance: str = "",
    prompt_path: Optional[Path] = None,
) -> str:
    history_blocks: List[str] = []
    for pos, (_, row) in enumerate(history_rows.iterrows(), start=1):
        history_blocks.append(
            f"[{pos}] Date: {row['date_iso']}\nSummary: {row['compressed_text']}"
        )
    history_text = "\n\n".join(history_blocks)
    template = get_forecast_prompt_template(prompt_guidance, prompt_path=prompt_path)
    return template.format(
        topic=sample["topic"],
        hs_code_4d=sample["hs_code_4d"],
        hs_short_label=sample["hs_short_label"],
        last_history_date=sample["last_history_date"],
        history_text=history_text,
    )


def build_prompt_feedback_agent_prompt(
    round_index: int,
    current_guidance: str,
    examples: pd.DataFrame,
    prompt_path: Optional[Path] = None,
) -> str:
    example_blocks: List[str] = []
    for pos, (_, row) in enumerate(examples.iterrows(), start=1):
        example_blocks.append(
            f"""Example {pos}
Topic: {row['topic']}
Target date: {row['target_date']}
True ArticleScore: {row['target_article_score']}
MLP predicted ArticleScore: {row['predicted_article_score']}
Squared error: {row['squared_error']}

Historical compressed news:
{row['history_compressed_news']}

Generated future-event description:
{row['generated_future_event']}

True next effective-date news title:
{row['true_next_news_title']}

True next effective-date news text:
{row['true_next_news_text']}"""
        )

    examples_text = "\n\n".join(example_blocks)
    guidance_text = current_guidance.strip() or "(none)"
    configured_template = prompt_template_from_config(prompt_path)
    if configured_template is not None:
        return configured_template.format(
            round_index=round_index,
            guidance_text=guidance_text,
            examples_text=examples_text,
        )

    return f"""You are a prompt-improvement agent for a news tone forecasting pipeline.

The current global P_pre guidance is:
{guidance_text}

The following examples are training-set samples with the highest squared prediction errors.
Compare the generated future-event descriptions with the true next effective-date news.
Focus only on tone-relevant differences: risk severity, policy conflict, supply disruption, production impact, litigation/regulatory pressure, market optimism/pessimism, and missing or exaggerated causal drivers.

Training high-loss examples:
{examples_text}

Task:
Suggest concise modifications that should be added to the global P_pre guidance.

Rules:
- Use only the examples above.
- Do not write a full prompt.
- Do not mention validation or test data.
- Do not include hidden reasoning, chain-of-thought, analysis notes, or <think> tags.
- Prefer generalizable guidance, not sample-specific event facts.
- Return strict JSON only, with this schema:
{{
  "diagnosis": "one short paragraph summarizing recurring tone errors",
  "guidance_bullets": [
    "actionable guidance bullet 1",
    "actionable guidance bullet 2"
  ]
}}"""


def extract_json_object(text: str) -> Optional[Dict[str, Any]]:
    text = clean_generated_text(text)
    text = re.sub(r"^```(?:json)?", "", text.strip(), flags=re.IGNORECASE).strip()
    text = re.sub(r"```$", "", text.strip()).strip()
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        parsed = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def parse_agent_suggestions(agent_text: str) -> Tuple[str, List[str]]:
    cleaned = clean_generated_text(agent_text)
    parsed = extract_json_object(cleaned)
    if parsed is not None:
        diagnosis = str(parsed.get("diagnosis", "") or "").strip()
        raw_bullets = parsed.get("guidance_bullets", [])
        if isinstance(raw_bullets, str):
            raw_bullets = [raw_bullets]
        suggestions = [str(item).strip() for item in raw_bullets if str(item).strip()]
        return diagnosis, suggestions

    suggestions: List[str] = []
    for line in cleaned.splitlines():
        line = line.strip()
        if not line:
            continue
        line = re.sub(r"^[-*\d.)\s]+", "", line).strip()
        if line:
            suggestions.append(line)
    return "", suggestions


def merge_guidance(
    previous_guidance: str,
    suggestions: Sequence[str],
    max_bullets: int,
) -> str:
    bullets: List[str] = []
    for line in str(previous_guidance or "").splitlines():
        line = line.strip()
        if line.startswith("-"):
            bullets.append(line[1:].strip())
        elif line:
            bullets.append(line)

    seen = {re.sub(r"\s+", " ", item.lower()) for item in bullets}
    for suggestion in suggestions:
        suggestion = re.sub(r"\s+", " ", str(suggestion)).strip()
        if not suggestion:
            continue
        key = suggestion.lower()
        if key in seen:
            continue
        bullets.append(suggestion)
        seen.add(key)

    bullets = bullets[-max_bullets:]
    return "\n".join(f"- {item}" for item in bullets)


def save_prompt_versions(prompt_versions: List[Dict[str, Any]], opt_dir: Path) -> None:
    ensure_dir(opt_dir)
    with (opt_dir / "prompt_versions.json").open("w", encoding="utf-8") as handle:
        json.dump(prompt_versions, handle, indent=2, ensure_ascii=False)

    lines: List[str] = ["# Prompt Versions", ""]
    for item in prompt_versions:
        lines.append(f"## Version {item['version']}")
        lines.append("")
        lines.append(f"- Guidance hash: `{item['guidance_hash']}`")
        if item.get("forecast_prompt_path"):
            lines.append(f"- Forecast prompt path: `{item['forecast_prompt_path']}`")
        if item.get("optimizer_prompt_path"):
            lines.append(f"- Optimizer prompt path: `{item['optimizer_prompt_path']}`")
        if "val_rmse" in item:
            lines.append(f"- Val RMSE: `{item['val_rmse']}`")
            lines.append(f"- Val MAE: `{item.get('val_mae')}`")
        if "test_rmse" in item:
            lines.append(f"- Test RMSE: `{item['test_rmse']}`")
            lines.append(f"- Test MAE: `{item.get('test_mae')}`")
        lines.append("")
        lines.append("### Guidance")
        lines.append("")
        lines.append(item["guidance"] or "(original prompt; no extra guidance)")
        lines.append("")
        lines.append("### Prompt Template")
        lines.append("")
        lines.append("```text")
        lines.append(item["prompt_template"])
        lines.append("```")
        lines.append("")
    (opt_dir / "prompt_versions.md").write_text("\n".join(lines), encoding="utf-8")
