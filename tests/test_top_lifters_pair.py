"""Check that `top_lifters_pair` returns the same lifts as the full matrix pipeline.

`compute_pair_lifts` computes a single row of the lift matrix without materialising it.
Two checks:

* on a small synthetic dataset, against
  `compute_cocitation_probability_matrix` + `compute_lift_matrix`;
* on the real dataset, against the sub-lift matrix of cluster 5 — the very matrix whose
  complexity / velocity columns are stored in `clusters/cluster_5_all_markers.csv`.
"""

from pathlib import Path

import numpy as np
import polars as pl
import pytest

from complexity_clusters import (
    compute_cocitation_probability_matrix,
    compute_lift_matrix,
    get_complexity_fast,
)
from top_lifters_pair import compute_pair_lifts

ROOT = Path(__file__).resolve().parent.parent

DATASET = ROOT / "data" / "dataset.csv"
CLUSTER_5 = ROOT / "clusters" / "cluster_5_all_markers.csv"
REFS = ["daihatsu_brand", "automotive_chip"]


def lift_matrix_from_occurrences(df: pl.DataFrame, markers) -> tuple:
    """Reference implementation: build the whole matrix, as the pipeline does."""
    conv = {m: i for i, m in enumerate(markers)}
    cocitation = compute_cocitation_probability_matrix(np.array(markers), df, conv)
    return compute_lift_matrix(cocitation), conv


@pytest.fixture
def synthetic_dataset(tmp_path: Path) -> Path:
    """Small occurrence table, with repeated markers inside an article and articles
    citing none of the markers of interest (so N > number of relevant articles)."""
    rng = np.random.default_rng(0)
    markers = [f"m{k}" for k in range(8)]
    rows = []
    for article in range(200):
        # each article cites a random subset, some markers twice
        chosen = [m for m in markers if rng.random() < 0.25]
        for m in chosen:
            rows.append((f"a{article}", m, "publisher_1", "economie"))
            if rng.random() < 0.3:
                rows.append((f"a{article}", m, "publisher_1", "economie"))
        if not chosen:  # article citing nothing relevant still counts in N
            rows.append((f"a{article}", "other", "publisher_1", "sport"))

    path = tmp_path / "dataset.csv"
    pl.DataFrame(
        rows, schema=["id", "marker", "publisher_label", "journal_theme"], orient="row"
    ).write_csv(path)
    return path


def test_matches_full_lift_matrix(synthetic_dataset: Path) -> None:
    markers = sorted(
        pl.read_csv(synthetic_dataset)["marker"].unique().to_list()
    )
    refs = ["m0", "m3"]

    df = compute_pair_lifts(
        str(synthetic_dataset), refs, universe=markers, verbose=False
    ).sort("marker")

    occurrences = pl.read_csv(synthetic_dataset)
    lift, conv = lift_matrix_from_occurrences(occurrences, markers)

    for ref in refs:
        expected = np.array([lift[conv[m], conv[ref]] for m in df["marker"]])
        np.testing.assert_allclose(df[f"lift_{ref}"].to_numpy(), expected, rtol=1e-12)

    expected_mean = np.mean(
        [[lift[conv[m], conv[r]] for r in refs] for m in df["marker"]], axis=1
    )
    np.testing.assert_allclose(df["lift_mean"].to_numpy(), expected_mean, rtol=1e-12)


@pytest.mark.slow
@pytest.mark.skipif(not DATASET.exists(), reason="data/dataset.csv not available")
def test_matches_cluster_5_matrix() -> None:
    """Same lifts as the cluster-5 sub-lift matrix behind `clusters/` artifacts."""
    cluster = pl.read_csv(CLUSTER_5)
    markers = cluster["marker"].to_list()

    pairs = pl.scan_csv(DATASET).select("id", "marker").unique()
    sub = pairs.filter(pl.col("marker").is_in(markers)).collect(engine="streaming")
    # `compute_cocitation_probability_matrix` reads the article universe from the whole
    # table (N = df["id"].n_unique(), before the marker filter). The rest of the table
    # contributes nothing else to the counts, so it is stood in for by one filler row
    # per article id.
    all_ids = pairs.select("id").unique().collect(engine="streaming")
    occurrences = pl.concat(
        [sub, all_ids.with_columns(pl.lit("__filler__").alias("marker"))]
    )

    lift, conv = lift_matrix_from_occurrences(occurrences, markers)

    # the stored complexity / velocity columns must come out of that matrix
    complexity = np.array([get_complexity_fast(lift, conv, m) for m in markers])
    velocity = np.array([lift[conv[m], conv[m]] ** -1 for m in markers])
    np.testing.assert_allclose(complexity, cluster["complexity"].to_numpy(), rtol=1e-9)
    np.testing.assert_allclose(velocity, cluster["velocity"].to_numpy(), rtol=1e-9)

    # and compute_pair_lifts must reproduce the two corresponding rows
    df = compute_pair_lifts(str(DATASET), REFS, universe=markers, verbose=False)
    for ref in REFS:
        expected = np.array([lift[conv[m], conv[ref]] for m in df["marker"]])
        np.testing.assert_allclose(df[f"lift_{ref}"].to_numpy(), expected, rtol=1e-12)
