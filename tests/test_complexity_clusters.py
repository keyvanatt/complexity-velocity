"""Tests for the co-citation / lift / complexity core of `complexity_clusters`."""

import numpy as np
import polars as pl
import pytest

from complexity_clusters import (
    compute_cocitation_probability_matrix,
    compute_complexity_df,
    compute_lift_matrix,
    compute_sub_lift_matrix,
    fit_loglog_regression,
    get_complexity_fast,
    markers_from_cluster,
    save_top_bottom_csv,
    select_markers_by_theme,
    top_lifters,
)


def test_cocitation_counts_articles_not_occurrences(occurrences, markers, conv):
    """A marker repeated inside an article counts once, and N covers every article."""
    cm = compute_cocitation_probability_matrix(np.array(markers), occurrences, conv)

    expected = (
        np.array(
            [
                [3, 2, 1],
                [2, 3, 0],
                [1, 0, 1],
            ]
        )
        / 5.0  # 5 articles, including a5 which cites none of m1/m2/m3
    )
    np.testing.assert_allclose(cm, expected)


def test_lift_matrix_values(occurrences, markers, conv):
    lift = compute_lift_matrix(
        compute_cocitation_probability_matrix(np.array(markers), occurrences, conv)
    )

    # lift(m1,m2) = (2/5) / ((3/5)(3/5)) = 10/9 ; lift(m1,m3) = (1/5) / ((3/5)(1/5)) = 5/3
    assert lift[0, 1] == pytest.approx(10 / 9)
    assert lift[0, 2] == pytest.approx(5 / 3)
    assert lift[1, 2] == 0.0  # never co-cited
    np.testing.assert_allclose(lift, lift.T)


def test_lift_diagonal_is_inverse_marginal(occurrences, markers, conv):
    """velocity = lift[i,i]**-1 must be P(marker), as used by the plots."""
    lift = compute_lift_matrix(
        compute_cocitation_probability_matrix(np.array(markers), occurrences, conv)
    )
    velocities = np.array([lift[i, i] ** -1 for i in range(len(markers))])
    np.testing.assert_allclose(velocities, [3 / 5, 3 / 5, 1 / 5])


def test_lift_of_independent_markers_is_one():
    """Independent markers have lift ~ 1: the metric is properly normalised."""
    rng = np.random.default_rng(0)
    n_docs, probs = 20000, [0.3, 0.5]
    draws = rng.random((n_docs, 2)) < probs
    rows = [
        (f"a{d}", f"m{k}") for d in range(n_docs) for k in range(2) if draws[d, k]
    ]
    df = pl.DataFrame(rows, schema=["id", "marker"], orient="row")
    # every article must appear in the table, even when it drew no marker
    df = pl.concat(
        [df, pl.DataFrame({"id": [f"a{d}" for d in range(n_docs)], "marker": "z"})]
    )

    lift = compute_lift_matrix(
        compute_cocitation_probability_matrix(
            np.array(["m0", "m1"]), df, {"m0": 0, "m1": 1}
        )
    )
    assert lift[0, 1] == pytest.approx(1.0, abs=0.05)


def test_get_complexity_fast_is_mean_off_diagonal_lift():
    lift = np.array(
        [
            [10.0, 2.0, 4.0],
            [2.0, 10.0, 0.0],
            [4.0, 0.0, 10.0],
        ]
    )
    conv = {"a": 0, "b": 1, "c": 2}
    assert get_complexity_fast(lift, conv, "a") == pytest.approx(3.0)
    assert get_complexity_fast(lift, conv, "b") == pytest.approx(1.0)


def test_get_complexity_fast_single_marker_is_zero():
    assert get_complexity_fast(np.array([[5.0]]), {"a": 0}, "a") == 0.0


def test_sub_lift_matrix_is_the_submatrix_of_the_full_one(occurrences, markers, conv):
    """Restricting the markers does not change the lifts: N stays the article count."""
    full = compute_lift_matrix(
        compute_cocitation_probability_matrix(np.array(markers), occurrences, conv)
    )
    sub, sub_conv = compute_sub_lift_matrix(np.array(["m1", "m3"]), occurrences)

    assert sub_conv == {"m1": 0, "m3": 1}
    idx = [conv["m1"], conv["m3"]]
    np.testing.assert_allclose(sub, full[np.ix_(idx, idx)])


def test_select_markers_by_theme_keeps_top_fraction_by_occurrence(occurrences):
    """Ranking is on occurrence rows (m1 has a duplicate), ties broken by name."""
    selected, conv, journals = select_markers_by_theme(
        occurrences, themes=["economie"], fraction=1.0
    )
    # in "economie": m1 x3 rows, m2 x2 rows, other x1 row
    assert list(selected) == ["m1", "m2", "other"]
    assert conv == {"m1": 0, "m2": 1, "other": 2}
    assert sorted(journals[0]) == ["pub_a"]

    selected_third, _, _ = select_markers_by_theme(
        occurrences, themes=["economie"], fraction=1 / 3
    )
    assert list(selected_third) == ["m1"]


def test_select_markers_by_theme_ignores_other_themes(occurrences):
    selected, _, _ = select_markers_by_theme(occurrences, themes=["sport"], fraction=1.0)
    assert sorted(selected) == ["m1", "m3"]


def test_select_markers_by_theme_random_is_reproducible(occurrences):
    kwargs = dict(themes=None, fraction=1 / 2, top=False)
    first, _, _ = select_markers_by_theme(occurrences, seed=7, **kwargs)
    second, _, _ = select_markers_by_theme(occurrences, seed=7, **kwargs)
    assert list(first) == list(second)


def test_fit_loglog_regression_recovers_the_power_law():
    c = np.linspace(1.0, 10.0, 50)
    v = 3.0 * c ** (-1.5)
    reg = fit_loglog_regression(c, v)

    assert reg["beta1"] == pytest.approx(-1.5)
    assert reg["beta0"] == pytest.approx(np.log(3.0))
    assert reg["r2"] == pytest.approx(1.0)
    assert reg["kendall_tau"] == pytest.approx(-1.0)
    assert reg["n"] == 50
    assert reg["beta1_ci"][0] <= reg["beta1"] <= reg["beta1_ci"][1]


def test_fit_loglog_regression_drops_invalid_points():
    c = np.array([1.0, 2.0, 4.0, 0.0, -1.0, np.nan])
    v = np.array([1.0, 0.5, 0.25, 1.0, 1.0, 1.0])
    reg = fit_loglog_regression(c, v)

    assert reg["n"] == 3
    assert reg["beta1"] == pytest.approx(-1.0)


def test_markers_from_cluster():
    labels = np.array([0, 1, -1, 1])
    selected = np.array(["a", "b", "c", "d"])

    np.testing.assert_array_equal(markers_from_cluster(labels, 1, selected), ["b", "d"])
    assert len(markers_from_cluster(labels, 2, selected)) == 0


def test_top_lifters_sorted_and_excludes_the_marker_itself():
    lift = np.array(
        [
            [100.0, 3.0, 5.0, 1.0],
            [3.0, 100.0, 2.0, 4.0],
            [5.0, 2.0, 100.0, 0.0],
            [1.0, 4.0, 0.0, 100.0],
        ]
    )
    conv = {"a": 0, "b": 1, "c": 2, "d": 3}
    top = top_lifters("a", lift, conv, complexity_df=None, top_n=2)

    assert [m for m, _ in top] == ["c", "b"]
    assert [v for _, v in top] == [5.0, 3.0]
    assert top_lifters("unknown", lift, conv, complexity_df=None) == []


def test_compute_complexity_df_is_sorted():
    df = compute_complexity_df({"a": 2.0, "b": 1.0, "c": 3.0})

    assert list(df.index) == ["b", "a", "c"]
    assert df.index.name == "marker"


def test_save_top_bottom_csv(tmp_path):
    complexities = {f"m{k}": float(k) for k in range(10)}
    out = tmp_path / "top_bottom.csv"
    df = save_top_bottom_csv(complexities, str(out), top_n=2)

    assert out.exists()
    assert list(df.index) == ["m0", "m1", "m8", "m9"]
    assert list(df["rank"]) == ["bottom", "bottom", "top", "top"]
