"""LLM judge: score markers in each cluster on a 1-10 complexity scale.

The LLM assigns a numeric score (1=simplest, 10=most complex) to each marker.
We then split each cluster's scores into per-cluster terciles (three equal-sized groups):
simple / intermediate / complex.

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


def load_model(model_name: str, quant: str = "none"):
    """quant='4bit' loads the model in NF4 (bitsandbytes) — lets a ~32B model fit in ~18 GB VRAM,
    e.g. a 24 GB RTX A5000. quant='8bit' ~= half precision footprint. 'none' = bf16."""
    logger.info("Loading model %s (quant=%s) ...", model_name, quant)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    load_kwargs = {"device_map": "auto"}
    if quant in ("4bit", "8bit"):
        from transformers import BitsAndBytesConfig
        if quant == "4bit":
            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )
        else:
            load_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
    else:
        load_kwargs["torch_dtype"] = torch.bfloat16

    model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)
    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)
    logger.info("Model loaded on %s", device)
    return pipe


def build_prompt(markers_data: List[Dict]) -> str:
    marker_lines = "\n".join(f'  "{d["marker"]}"' for d in markers_data)
    return f"""The following concepts are thematic markers extracted from CausalityLink, a database of causal relationships mined from financial and economic news articles (press, central bank reports, analyst notes).

We define the COMPLEXITY of a concept as the degree to which it presupposes other concepts for its proper articulation in this kind of corpus:
- A concept is SIMPLE (low score) if it can appear in isolation and be discussed without any particular contextual scaffolding: generic, self-contained notions (e.g. "debt", "smartphone", "housing").
- A concept is COMPLEX (high score) if discussing it requires mobilizing many other, simpler concepts as prerequisites, so it only appears in specialized contexts alongside a rich constellation of related notions (e.g. "default" presupposes "debt" and "maturity"; "systemic_risk" presupposes "debt", "default" and "financial_interconnection"; a specific product line like "lenovo_thinkpad_x1_carbon" presupposes knowledge of brands, product categories and competitive positioning, unlike the generic "personal_computer").

Note that complexity in this sense is NOT cognitive difficulty or cultural familiarity: highly specific entities (niche subsidiaries, product lines, specialized financial metrics) are complex because they presuppose specialized context, even if the words themselves are easy to understand.

Rate the complexity of each concept, in the sense defined above, with an integer score from 1 (simplest) to 10 (most complex), relative to the other concepts in this list. Use the full range of scores. Do not use decimal points or fractions. Do not provide any explanation, only return a JSON object mapping each marker to its score.

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
    """Split one cluster's llm_scores into per-cluster terciles → three ~equal-sized groups
    (bottom / middle / top third), labelled simple / intermediate / complex. Called once per
    cluster, so the split is always relative to that cluster's own score distribution; the labels
    are NOT comparable across clusters (a 'complex' in one cluster may be lower-scored than a
    'simple' in another).

    We rank the scores (ties broken by original order) and cut the *ranks* into thirds rather than
    qcut-ing the raw scores: with heavy ties — the model often reuses the same note many times —
    raw qcut collapses the bin edges and raises 'Bin edges must be unique'. Ranking guarantees three
    balanced groups even when every marker shares a score; the cost is that equal-scored markers
    straddling a boundary are separated by arbitrary order. NaN (unscored) stays NaN."""
    labels = np.array(["simple", "intermediate", "complex"])
    out = pd.Series(pd.NA, index=scores.index, dtype=object)
    valid = scores.dropna()
    n = len(valid)
    if n == 0:
        return out
    ranks = valid.rank(method="first").to_numpy()  # 1..n, unique
    idx = np.minimum(2, ((ranks - 1) * 3 // n)).astype(int)  # 0/1/2, ~n/3 each
    out.loc[valid.index] = labels[idx]
    return out


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


def _gaussian_kde(values: np.ndarray, grid: np.ndarray, bw_scale: float = 1.0) -> np.ndarray:
    """Simple Gaussian KDE (Silverman bandwidth) evaluated on `grid`.
    `bw_scale` < 1 sharpens the peaks (a legitimate smoothing choice, not a data distortion).
    Avoids a hard scipy dependency."""
    values = np.asarray(values, dtype=float)
    n = len(values)
    if n == 0:
        return np.zeros_like(grid)
    std = values.std(ddof=1) if n > 1 else 0.0
    if std == 0:
        # Degenerate: all values identical → fall back to a small bandwidth
        std = 1.0
        bw = 0.05
    else:
        bw = 1.06 * std * n ** (-1 / 5)
    bw = max(bw * bw_scale, 1e-3)
    # (grid x values) matrix of standardized distances
    u = (grid[:, None] - values[None, :]) / bw
    kernel = np.exp(-0.5 * u ** 2) / np.sqrt(2 * np.pi)
    return kernel.sum(axis=1) / (n * bw)


def plot_cdfs(clusters_dir: Path, out_path: Path) -> None:
    """Single figure, two stacked panels sharing the x-axis (normalized log-complexity):
      - top:    one CDF curve per cluster, colored by tercile (green/amber/red)
      - bottom: pooled density ("3 bumps") — one Gaussian KDE per category color
    Terciles are balanced per cluster (1/3 each) based on per-cluster LLM score quantiles."""
    from matplotlib.collections import LineCollection
    from matplotlib.lines import Line2D
    import matplotlib.patheffects as pe

    paths = sorted(clusters_dir.glob("cluster_*_llm_classification.csv"))
    if not paths:
        logger.warning("No llm_classification CSVs found, skipping CDF plot.")
        return

    # Softer, ordered palette (green -> amber -> red). Fills are the light tint,
    # strokes the darker tone, so overlapping bumps never muddy into brown.
    FILL = {"simple": "green", "intermediate": "orange", "complex": "red"}
    STROKE = {"simple": "green", "intermediate": "orange", "complex": "red"}
    label_txt = {"simple": "Simple", "intermediate": "Intermediate", "complex": "Complex"}
    sub_txt = {"simple": "bottom tercile", "intermediate": "middle tercile", "complex": "top tercile"}

    fig, (ax, ax_d) = plt.subplots(
        2, 1, figsize=(11, 9), sharex=True,
        gridspec_kw={"height_ratios": [2, 1.15], "hspace": 0.25},
    )

    # Pool normalized complexities per category across all clusters for the density panel
    pooled = {"simple": [], "intermediate": [], "complex": []}

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

        for cat in pooled:
            pooled[cat].append(df.loc[df["llm_category"] == cat, "norm_complexity"].to_numpy())

        # Sort by norm_complexity → CDF y = rank / n
        df_sorted = df.sort_values("norm_complexity").reset_index(drop=True)
        n = len(df_sorted)
        xs = df_sorted["norm_complexity"].to_numpy()
        ys = np.arange(1, n + 1) / n
        colors = [STROKE.get(cat, "#cccccc") for cat in df_sorted["llm_category"]]

        # Wider, moderate-alpha segments so each tercile band stays legible
        points = np.column_stack([xs, ys]).reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        lc = LineCollection(segments, colors=colors[:-1], linewidth=1.7, alpha=0.4)
        ax.add_collection(lc)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_ylabel("CDF", fontsize=11)
    ax.set_title("Log-complexity CDF per cluster", fontsize=12)
    ax.grid(alpha=0.25)

    # Density panel: one KDE bump per category, pooled across clusters.
    # BW_SCALE < 1 sharpens the peaks so adjacent terciles read as distinct (honest smoothing choice).
    BW_SCALE = 0.7
    grid = np.linspace(0, 1, 400)
    med_handles = []
    for cat in ("simple", "intermediate", "complex"):
        vals = np.concatenate(pooled[cat]) if pooled[cat] else np.array([])
        if len(vals) == 0:
            continue
        dens = _gaussian_kde(vals, grid, bw_scale=BW_SCALE)
        # Light tint fill + crisp stroke → bumps stay readable where they overlap
        ax_d.fill_between(grid, dens, color=FILL[cat], alpha=0.12, zorder=1)
        ax_d.plot(grid, dens, color=STROKE[cat], linewidth=2.2, zorder=3)

        # Median: short vertical tick from baseline to the curve, with a value label above it.
        med = float(np.median(vals))
        y_at_med = float(np.interp(med, grid, dens))
        ax_d.plot([med, med], [0, y_at_med], color=STROKE[cat], linewidth=1.4,
                  linestyle=(0, (4, 3)), zorder=4)
        ax_d.annotate(
            f"{med:.2f}", xy=(med, y_at_med), xytext=(0, 5),
            textcoords="offset points", ha="center", va="bottom",
            fontsize=8.5, color=STROKE[cat], fontweight="bold",
            path_effects=[pe.withStroke(linewidth=2.5, foreground="white")], zorder=6,
        )
        med_handles.append(
            Line2D([0], [0], color=STROKE[cat], linewidth=2.5,
                   label=f"{label_txt[cat]} ({sub_txt[cat]}, median {med:.2f})")
        )

    ax_d.set_xlim(0, 1)
    ax_d.set_ylim(bottom=0)
    ax_d.margins(y=0.12)
    ax_d.set_xlabel("Normalized log-complexity", fontsize=11)
    ax_d.set_ylabel("Density", fontsize=11)
    ax_d.set_title("Log-complexity density per tercile (pooled over clusters)", fontsize=12)
    ax_d.grid(alpha=0.25)

    # Single legend shared by both panels (same colors mean the same terciles everywhere)
    if med_handles:
        ax_d.legend(handles=med_handles, fontsize=9, loc="upper right")

    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    logger.info("CDF + density plot saved -> %s", out_path)
    plt.close(fig)


def plot_density_small_multiples(clusters_dir: Path, out_path: Path, ncols: int = 4) -> None:
    """One small density panel per cluster (small-multiples). LLM scores are only comparable
    *within* a cluster (they are assigned 'relative to the other concepts in this list'), so we
    never pool across clusters: each panel shows the 3 per-cluster terciles over that cluster's
    own normalized log-complexity. Terciles are ordered by construction, so the bumps stay
    monotone green -> orange -> red within each panel."""
    from matplotlib.lines import Line2D

    FILL = {"simple": "green", "intermediate": "orange", "complex": "red"}
    label_txt = {"simple": "Simple", "intermediate": "Intermediate", "complex": "Complex"}

    paths = sorted(clusters_dir.glob("cluster_*_llm_classification.csv"),
                   key=lambda p: int(p.stem.split("_")[1]))
    dfs = []
    for path in paths:
        df = pd.read_csv(path).dropna(subset=["complexity"])
        if df.empty or "llm_score" not in df.columns or df["llm_score"].dropna().empty:
            continue
        dfs.append((int(path.stem.split("_")[1]), df))
    if not dfs:
        logger.warning("No usable classification CSVs found, skipping small-multiples plot.")
        return

    nrows = int(np.ceil(len(dfs) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.2 * ncols, 2.3 * nrows),
                             sharex=True, squeeze=False)
    grid = np.linspace(0, 1, 300)

    for ax_i, (cid, df) in zip(axes.ravel(), dfs):
        df = df.copy()
        df["llm_category"] = scores_to_categories(df["llm_score"])
        c = df["complexity"].to_numpy(dtype=float)
        log_c = np.log(c)
        lc_min, lc_max = log_c.min(), log_c.max()
        df["norm_complexity"] = (log_c - lc_min) / (lc_max - lc_min) if lc_max > lc_min else np.zeros(len(c))

        for cat in ("simple", "intermediate", "complex"):
            vals = df.loc[df["llm_category"] == cat, "norm_complexity"].to_numpy()
            if len(vals) == 0:
                continue
            dens = _gaussian_kde(vals, grid, bw_scale=0.8)
            ax_i.fill_between(grid, dens, color=FILL[cat], alpha=0.18, zorder=1)
            ax_i.plot(grid, dens, color=FILL[cat], linewidth=1.6, zorder=3)

        ax_i.set_title(f"Cluster {cid} (n={len(df)})", fontsize=9)
        ax_i.set_xlim(0, 1)
        ax_i.set_ylim(bottom=0)
        ax_i.tick_params(labelsize=7)
        ax_i.grid(alpha=0.2)

    # Blank out any unused axes in the grid
    for ax_i in axes.ravel()[len(dfs):]:
        ax_i.axis("off")

    handles = [Line2D([0], [0], color=FILL[c], linewidth=2.5, label=label_txt[c])
               for c in ("simple", "intermediate", "complex")]
    fig.suptitle(
        "Log-complexity density per tercile — one panel per cluster (no pooling)\n"
        "x: normalized log-complexity   ·   y: density",
        fontsize=13,
    )
    fig.supylabel("Density", fontsize=10)
    fig.tight_layout(rect=(0.01, 0.04, 1, 0.96))
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=11,
               bbox_to_anchor=(0.5, 0.005))
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    logger.info("Per-cluster density small-multiples saved -> %s", out_path)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="LLM judge for marker complexity scoring")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="HuggingFace model ID")
    parser.add_argument("--quant", default="none", choices=["none", "4bit", "8bit"],
                        help="Quantization (4bit lets a ~32B fit on a 24 GB GPU)")
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
    pipe = load_model(args.model, quant=args.quant)

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

    sm_path = clusters_dir / "llm_classification_density_by_cluster.png"
    plot_density_small_multiples(clusters_dir, sm_path)


if __name__ == "__main__":
    main()
