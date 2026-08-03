"""The lift-derived dissimilarity used for semantic clustering.

Single source of truth, so the synthetic benchmarks and the corpus pipeline
cannot drift apart. NumPy-only, so every caller can import it cheaply.

For markers i and j with pairwise lift l[i,j] and marginal probability
p_i = P(M_i = 1):

    D[i,j] = log(1 + 1/(l[i,j] + eps) - p_i),   symmetrised, shifted >= 0.

The transform resolves deviations around independence and saturates for
strongly associated pairs, so the magnitude of a large lift - the noisiest part
of the scale - barely moves the geometry. The ``- p_i`` term is a row-wise
offset that damps the contribution of very frequent markers; it also puts the
diagonal at zero, since l[i,i] = 1/p_i implies 1/(l[i,i] + eps) - p_i ~ 0.

D is not a metric: it need not satisfy the triangle inequality. It is fed to
UMAP as a precomputed dissimilarity, which does not require one.
"""

import numpy as np

__all__ = ["lift_dissimilarity", "marginals_from_lift"]

# Guards against degenerate lifts; eps also keeps 1/(l + eps) finite when l = 0.
DEFAULT_EPSILON = 1e-4
_LOG1P_FLOOR = -1.0 + 1e-12


def marginals_from_lift(lift: np.ndarray) -> np.ndarray:
    """Recover the marginal probabilities p_i from the diagonal of a lift matrix.

    ``lift[i,i] = p_ii / p_i^2 = 1 / p_i``, so ``p_i = 1 / lift[i,i]``. Markers
    that never occur have an undefined diagonal; they are reported as p_i = 0.
    Results are clipped to [0, 1], which a probability must satisfy anyway and
    which neutralises the epsilon-guarded zero diagonals produced upstream.
    """
    diag = np.diag(np.asarray(lift, dtype=float))
    with np.errstate(divide="ignore", invalid="ignore"):
        marginals = np.divide(1.0, diag, out=np.zeros_like(diag), where=diag > 0)
    marginals[~np.isfinite(marginals)] = 0.0
    return np.clip(marginals, 0.0, 1.0)


def lift_dissimilarity(lift: np.ndarray, epsilon: float = DEFAULT_EPSILON) -> np.ndarray:
    """Dissimilarity matrix derived from a lift matrix.

    Args:
        lift: square, symmetric matrix of pairwise lifts. The diagonal carries
            the marginals via ``lift[i,i] = 1 / p_i``.
        epsilon: regularisation added to the lift before inversion.

    Returns:
        A symmetric, non-negative matrix suitable for
        ``UMAP(metric="precomputed")``.
    """
    lift = np.asarray(lift, dtype=float)
    marginals = marginals_from_lift(lift)

    with np.errstate(divide="ignore", invalid="ignore"):
        argument = (lift + epsilon) ** (-1) - marginals[:, None]

    # log1p diverges at -1; only reachable from degenerate (never-observed)
    # markers, whose marginal is clipped to 1 above.
    argument = np.maximum(argument, _LOG1P_FLOOR)

    D = np.log1p(argument)
    D = 0.5 * (D + D.T)
    D -= D.min()
    return D
