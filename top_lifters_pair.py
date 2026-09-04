"""Lift of every marker with respect to one or several reference markers.

Same lift as `complexity_clusters.compute_lift_matrix`:

    lift(i, j) = P(i, j) / (P(i) * P(j)) = n_ij * N / (n_i * n_j)

with N the number of distinct articles of the filtered table and n_i the number of
articles citing marker i. Building the full n x n matrix is unnecessary here: only the
marginals and the co-citation counts with the reference markers are needed, so this
streams over `data/dataset.csv` in a few passes.

The marker universe is the same as in the main pipeline: `data/dataset.csv` already
carries the `prepare_filtered_marker_table` filters (no-country markers, known
publishers, journal theme), and `select_markers` below replicates
`select_markers_by_theme` (markers cited in the 6 broad themes, top `fraction` by
occurrence count).

Usage:
    python top_lifters_pair.py daihatsu_brand automotive_chip --top-n 10
"""

import argparse
from typing import Dict, List, Optional, Sequence

import polars as pl


DEFAULT_THEMES = ["sante", "economie", "sport", "politique", "transport", "information"]


def select_markers(
    lf: pl.LazyFrame,
    themes: Optional[List[str]] = None,
    fraction: float = 1 / 3,
    top: bool = True,
    seed: int = 42,
) -> List[str]:
    """Same marker selection as `complexity_clusters.select_markers_by_theme`.

    Markers cited in `themes`, ranked by occurrence count (rows, not distinct articles),
    keeping the top `fraction` of them.
    """
    if themes is None:
        counts = lf.group_by("marker").len("marker_count")
    else:
        counts = (
            lf.filter(pl.col("journal_theme").is_in(themes))
            .group_by("marker")
            .len("marker_count")
        )
    counts = counts.collect(engine="streaming")

    n_total = len(counts)
    keep_n = max(1, int(n_total * fraction))
    if top:
        counts = counts.sort(["marker_count", "marker"], descending=[True, False]).head(keep_n)
    else:
        # group_by does not preserve row order, so sort first to make `seed` reproducible
        counts = counts.sort("marker").sample(keep_n, shuffle=True, seed=seed)
    print(f"marker universe: {keep_n} markers kept out of {n_total} (fraction={fraction:.2f})")
    return counts["marker"].to_list()


def compute_pair_lifts(
    dataset: str,
    refs: Sequence[str],
    themes: Optional[List[str]] = None,
    fraction: float = 1 / 3,
    top: bool = True,
    seed: int = 42,
    universe: Optional[Sequence[str]] = None,
    verbose: bool = True,
) -> pl.DataFrame:
    """Return, for every selected marker, its lift with each reference marker.

    Columns: `marker`, `n` (articles citing it), `n_co_<k>` (articles citing it together
    with reference k), `lift_<ref>` for each reference, and `lift_mean`, the mean lift
    over the references — the aggregation used by `get_complexity_fast`.

    Args:
        dataset: CSV with columns `id`, `marker`, `publisher_label`, `journal_theme`
            (output of `data/extract_dataset.py`, one row per occurrence).
        refs: reference markers to lift against.
        themes / fraction / top / seed: marker selection, see `select_markers`.
        universe: explicit marker list, bypassing the `select_markers` selection.
    """
    scan = pl.scan_csv(dataset)
    selected = list(universe) if universe is not None else select_markers(
        scan, themes, fraction, top, seed
    )

    missing = [m for m in refs if m not in set(selected)]
    if missing:
        raise ValueError(f"reference markers outside the selected universe: {missing}")

    # N counts every article of the filtered table, as in
    # `compute_cocitation_probability_matrix` (n_articles is read before the marker filter)
    pairs_all = scan.select("id", "marker").unique()
    n_articles = pairs_all.select(pl.col("id").n_unique()).collect(engine="streaming").item()

    # one row per (article, marker) — a marker can be repeated inside an article
    pairs = pairs_all.filter(pl.col("marker").is_in(selected))

    # articles citing each reference marker (small: a few hundred / thousand ids)
    ref_ids: Dict[str, List[str]] = {}
    for m in refs:
        ids = (
            pairs.filter(pl.col("marker") == m)
            .select("id")
            .collect(engine="streaming")["id"]
            .to_list()
        )
        if not ids:
            raise ValueError(f"marker {m!r} not found in {dataset}")
        ref_ids[m] = ids
        if verbose:
            print(f"{m}: cited in {len(ids)} articles")
    if verbose:
        print(f"total articles: {n_articles}")

    # single pass: marginal counts + co-citation counts with each reference
    aggs = [pl.len().alias("n")]
    for k, m in enumerate(refs):
        aggs.append(pl.col("id").is_in(ref_ids[m]).sum().alias(f"n_co_{k}"))

    df = pairs.group_by("marker").agg(aggs).collect(engine="streaming")

    lift_cols = []
    for k, m in enumerate(refs):
        n_ref = len(ref_ids[m])
        col = f"lift_{m}"
        df = df.with_columns(
            (pl.col(f"n_co_{k}") * n_articles / (pl.col("n") * n_ref)).alias(col)
        )
        lift_cols.append(col)

    df = df.with_columns(pl.mean_horizontal(lift_cols).alias("lift_mean"))
    return df.sort("lift_mean", descending=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("markers", nargs="+", help="reference markers")
    parser.add_argument("--dataset", default="data/dataset.csv")
    parser.add_argument("--top-n", type=int, default=10)
    parser.add_argument(
        "--themes",
        nargs="*",
        default=DEFAULT_THEMES,
        help="journal themes for the marker selection (empty = no theme filter)",
    )
    parser.add_argument("--fraction", type=float, default=1 / 3)
    parser.add_argument(
        "--random-markers",
        action="store_true",
        help="sample the marker universe instead of taking the most frequent ones",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default=None, help="optional CSV output path")
    args = parser.parse_args()

    df = compute_pair_lifts(
        args.dataset,
        args.markers,
        themes=args.themes or None,
        fraction=args.fraction,
        top=not args.random_markers,
        seed=args.seed,
    )
    df = df.filter(~pl.col("marker").is_in(args.markers))

    with pl.Config(tbl_rows=args.top_n + 5, tbl_cols=-1, fmt_str_lengths=40):
        print(df.head(args.top_n))

    if args.out:
        df.write_csv(args.out)
        print(f"saved {args.out}")


if __name__ == "__main__":
    main()
