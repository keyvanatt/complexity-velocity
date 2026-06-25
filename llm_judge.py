"""LLM judge: score markers in each cluster on a 1-10 complexity scale.

The LLM assigns a numeric score (1=simplest, 10=most complex) to each marker.
We then bin scores into thirds (by global quantile) to get: simple / intermediate / complex.

Input:  clusters/cluster_*_all_markers.csv  (marker, complexity, velocity)
Output: clusters/cluster_*_llm_classification.csv  (adds llm_score, llm_category columns)
        clusters/llm_classification_summary.csv

Usage:
    python llm_judge.py [--model mistralai/Ministral-8B-Instruct-2410] [--cluster-ids 0 1 2]
"""

import argparse
import json
import logging
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# packages installed in /tmp/pypackages
sys.path.insert(0, "/tmp/pypackages")

import os
os.environ.setdefault("HF_HOME", "/tmp/hf_cache")
os.environ.setdefault("TRANSFORMERS_CACHE", "/tmp/hf_cache")

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline


logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_MODEL = "mistralai/Ministral-8B-Instruct-2410"

CATEGORY_COLORS = {
    "simple": "green",
    "intermediate": "orange",
    "complex": "red",
    None: "lightgray",
}


def load_model(model_name: str):
    logger.info("Loading model %s ...", model_name)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        device_map="auto",
    )
    logger.info("Model loaded on %s", device)
    return pipe


def build_prompt(markers_data: List[Dict]) -> str:
    marker_lines = "\n".join(f'  "{d["marker"]}"' for d in markers_data)
    return f"""The following concepts are thematic markers extracted from CausalityLink, a database of causal relationships mined from financial and economic news articles (press, central bank reports, analyst notes).

Rate the economic complexity of each concept with an integer score from 1 (simplest) to 10 (most complex), relative to the other concepts in this list.

Markers:
{marker_lines}

Reply with ONLY a valid JSON object, no surrounding text:
{{"marker_name": score, ...}}"""


def strip_markdown_fence(text: str) -> str:
    text = text.strip()
    text = re.sub(r"^```[a-zA-Z]*\n?", "", text)
    text = re.sub(r"\n?```$", "", text)
    return text.strip()


def extract_json(text: str) -> Optional[Dict]:
    text = strip_markdown_fence(text)
    match = re.search(r"\{[^{}]+\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


def run_batch(pipe, prompt: str, max_new_tokens: int = 1024) -> Dict[str, float]:
    messages = [{"role": "user", "content": prompt}]
    out = pipe(
        messages,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        temperature=None,
        top_p=None,
        pad_token_id=pipe.tokenizer.eos_token_id,
    )
    generated = out[0]["generated_text"]
    if isinstance(generated, list):
        text = generated[-1]["content"]
    else:
        text = generated

    cleaned_text = strip_markdown_fence(text)
    if not cleaned_text.endswith("}"):
        logger.error(
            "LLM output appears truncated (does not end with '}'). "
            "Increase max_new_tokens or reduce cluster size.\n"
            "Last 200 chars: ...%s",
            cleaned_text[-200:],
        )

    result = extract_json(text)
    if result is None:
        logger.error("JSON parsing failed — full output (%d chars):\n%s", len(text), text)
        return {}

    scores = {}
    invalid = []
    for marker, val in result.items():
        try:
            score = float(val)
            if 1.0 <= score <= 10.0:
                scores[marker.strip()] = score
            else:
                invalid.append((marker, val))
        except (TypeError, ValueError):
            invalid.append((marker, val))
    if invalid:
        logger.error(
            "%d marker(s) got an invalid score (expected 1-10): %s",
            len(invalid), invalid[:10],
        )
    return scores


def scores_to_categories(scores: pd.Series) -> pd.Series:
    """Bin llm_score into exact thirds by rank → simple / intermediate / complex.
    Uses rank(method='first') to break ties, guaranteeing balanced thirds even when scores cluster."""
    result = pd.Series(None, index=scores.index, dtype=object)
    valid_idx = scores.dropna().index
    if valid_idx.empty:
        return result
    ranks = scores[valid_idx].rank(method="first")
    n = len(ranks)
    labels = pd.cut(
        ranks,
        bins=[0, n / 3, 2 * n / 3, n],
        labels=["simple", "intermediate", "complex"],
        include_lowest=True,
    )
    result[valid_idx] = labels.values
    return result


def classify_cluster(pipe, cluster_id: int, path: Path) -> pd.DataFrame:
    df = pd.read_csv(path).dropna(subset=["complexity"])
    n_total = len(df)

    markers_data = df[["marker", "complexity"]].to_dict("records")
    logger.info("  Cluster %d — scoring %d markers in a single call", cluster_id, n_total)

    prompt = build_prompt(markers_data)
    max_new = max(1024, n_total * 15)
    logger.info("  Prompt ~%d tokens, max_new_tokens=%d", len(prompt) // 4, max_new)

    scores = run_batch(pipe, prompt, max_new_tokens=max_new)

    df["llm_score"] = df["marker"].map(scores)
    df["llm_category"] = scores_to_categories(df["llm_score"])

    n_missing = int(df["llm_score"].isna().sum())
    if n_missing > 0:
        missing_rate = n_missing / n_total
        level = logger.error if missing_rate > 0.1 else logger.warning
        level(
            "Cluster %d: %d/%d markers unscored (%.0f%%)%s",
            cluster_id, n_missing, n_total, missing_rate * 100,
            " — likely truncated output" if missing_rate > 0.1 else "",
        )

    dist = df["llm_category"].value_counts().to_dict()
    logger.info("  Cluster %d distribution: %s", cluster_id, dist)
    return df


def plot_cdfs(clusters_dir: Path, out_path: Path) -> None:
    """One CDF curve per cluster, drawn as colored line segments (green=simple, orange=intermediate, red=complex).
    Terciles are balanced per cluster (1/3 each) based on per-cluster LLM score quantiles."""
    from matplotlib.collections import LineCollection
    from matplotlib.lines import Line2D

    paths = sorted(clusters_dir.glob("cluster_*_llm_classification.csv"))
    if not paths:
        logger.warning("No llm_classification CSVs found, skipping CDF plot.")
        return

    cat_color = {"simple": "green", "intermediate": "orange", "complex": "red", None: "lightgray"}

    fig, ax = plt.subplots(figsize=(11, 7))

    for path in paths:
        df = pd.read_csv(path).dropna(subset=["complexity"])
        if df.empty or "llm_score" not in df.columns:
            continue

        df = df.copy()
        # Per-cluster terciles → exactly 1/3 per category
        df["llm_category"] = scores_to_categories(df["llm_score"])

        # Normalize log-complexity within this cluster to [0, 1]
        c = df["complexity"].to_numpy(dtype=float)
        log_c = np.log(c)
        lc_min, lc_max = log_c.min(), log_c.max()
        df["norm_complexity"] = (log_c - lc_min) / (lc_max - lc_min) if lc_max > lc_min else np.zeros(len(c))

        # Sort by norm_complexity → CDF y = rank / n
        df_sorted = df.sort_values("norm_complexity").reset_index(drop=True)
        n = len(df_sorted)
        xs = df_sorted["norm_complexity"].to_numpy()
        ys = np.arange(1, n + 1) / n
        colors = [cat_color.get(cat, "lightgray") for cat in df_sorted["llm_category"]]

        # Draw one colored segment per consecutive pair of markers
        points = np.column_stack([xs, ys]).reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        lc = LineCollection(segments, colors=colors[:-1], linewidth=1.0, alpha=0.55)
        ax.add_collection(lc)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    legend_handles = [
        Line2D([0], [0], color="green", linewidth=2.5, label="Simple (bottom LLM score tercile)"),
        Line2D([0], [0], color="orange", linewidth=2.5, label="Intermediate (middle LLM score tercile)"),
        Line2D([0], [0], color="red", linewidth=2.5, label="Complex (top LLM score tercile)"),
    ]
    ax.legend(handles=legend_handles, fontsize=10, loc="upper left")
    ax.set_xlabel("Normalized log-complexity", fontsize=11)
    ax.set_ylabel("CDF", fontsize=11)
    ax.set_title("Log-complexity CDF per cluster", fontsize=12)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    logger.info("CDF plot saved -> %s", out_path)
    plt.close(fig)

def main():
    parser = argparse.ArgumentParser(description="LLM judge for marker complexity scoring")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="HuggingFace model ID")
    parser.add_argument("--clusters-dir", default="clusters")
    parser.add_argument("--cluster-ids", nargs="*", type=int, help="Specific cluster IDs (default: all)")
    args = parser.parse_args()

    clusters_dir = Path(args.clusters_dir)

    if args.cluster_ids is not None:
        paths = [clusters_dir / f"cluster_{cid}_all_markers.csv" for cid in args.cluster_ids]
        paths = [p for p in paths if p.exists()]
    else:
        paths = sorted(clusters_dir.glob("cluster_*_all_markers.csv"))

    if not paths:
        logger.error(
            "No cluster_*_all_markers.csv found in %s.\n"
            "Run run_all(all_clusters=True) first to generate those files.",
            clusters_dir,
        )
        return

    logger.info("Processing %d cluster(s) with %s", len(paths), args.model)
    pipe = load_model(args.model)

    summary_rows = []
    for path in paths:
        cluster_id = int(path.stem.split("_")[1])
        logger.info("=== Cluster %d ===", cluster_id)

        df = classify_cluster(pipe, cluster_id, path)

        out_path = clusters_dir / f"cluster_{cluster_id}_llm_classification.csv"
        df.to_csv(out_path, index=False)

        counts = df["llm_category"].value_counts()
        summary_rows.append({
            "cluster_id": cluster_id,
            "n_total": len(df),
            "n_simple": int(counts.get("simple", 0)),
            "n_intermediate": int(counts.get("intermediate", 0)),
            "n_complex": int(counts.get("complex", 0)),
            "n_unscored": int(df["llm_score"].isna().sum()),
            "mean_llm_score": round(float(df["llm_score"].mean()), 2) if df["llm_score"].notna().any() else None,
        })

    summary_df = pd.DataFrame(summary_rows).set_index("cluster_id").sort_index()
    summary_path = clusters_dir / "llm_classification_summary.csv"
    summary_df.to_csv(summary_path)
    logger.info("Summary saved -> %s", summary_path)
    print(summary_df.to_string())

    cdf_path = clusters_dir / "llm_classification_cdfs.png"
    plot_cdfs(clusters_dir, cdf_path)


if __name__ == "__main__":
    main()
