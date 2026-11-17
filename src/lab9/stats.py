"""Statistics students implement (M2 judge calibration, M4 paired bootstrap).

Rules: a judge is a classifier -- measure it like one. A higher mean is not
"better" until the paired CI of the difference excludes 0.
Reference values to self-check are in tests/test_lab9_stats.py.
"""

from __future__ import annotations

from collections.abc import Sequence
from math import comb

POS = "FAIL"   # the judge's job is to FIND failures, so FAIL is the positive class


def confusion(human: Sequence[str], judge: Sequence[str]) -> dict[str, int]:
    """tp/fp/fn/tn with FAIL as positive."""
    if len(human) != len(judge):
        raise ValueError("human and judge sequences must have the same length")
    tp = fp = fn = tn = 0
    for h, j in zip(human, judge):
        h_pos = (h == POS)
        j_pos = (j == POS)
        if h_pos and j_pos:
            tp += 1
        elif not h_pos and j_pos:
            fp += 1
        elif h_pos and not j_pos:
            fn += 1
        else:
            tn += 1
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn}


def precision_recall(human: Sequence[str], judge: Sequence[str]) -> tuple[float, float]:
    # precision = tp/(tp+fp), recall = tp/(tp+fn); return 0.0 when the denominator is 0
    c = confusion(human, judge)
    tp, fp, fn = c["tp"], c["fp"], c["fn"]
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    return precision, recall


def cohen_kappa(human: Sequence[str], judge: Sequence[str]) -> float:
    """kappa = (p_o - p_e) / (1 - p_e). Raise ValueError when undefined (p_e == 1):
    a constant judge has no signal and must FAIL calibration, not pass on a technicality."""
    if len(human) != len(judge) or not human:
        raise ValueError("human and judge sequences must have the same non-zero length")
    n = len(human)
    labels = set(human) | set(judge)
    p_o = sum(h == j for h, j in zip(human, judge)) / n

    p_e = 0.0
    for lbl in labels:
        h_count = sum(h == lbl for h in human)
        j_count = sum(j == lbl for j in judge)
        p_e += (h_count / n) * (j_count / n)

    if abs(1.0 - p_e) < 1e-12:
        raise ValueError("kappa is undefined when p_e == 1 (constant judge)")

    return (p_o - p_e) / (1.0 - p_e)


def rogan_gladen(observed_rate: float, sensitivity: float, specificity: float) -> float:
    """Correct an observed failure rate for judge error; clip to [0, 1]."""
    denom = sensitivity + specificity - 1.0
    if abs(denom) < 1e-12:
        raise ValueError("sensitivity + specificity - 1 cannot be zero")
    val = (observed_rate + specificity - 1.0) / denom
    return max(0.0, min(1.0, float(val)))


def paired_bootstrap(a: Sequence[float], b: Sequence[float], *, n_resamples: int = 1000,
                     rng: int = 2027, confidence: float = 0.95) -> tuple[float, float, float]:
    """a[i], b[i] = per-item mean over seeds, SAME items, SAME order.
    Return (mean(a)-mean(b), lo, hi) with scipy.stats.bootstrap(paired=True, method="percentile",
    rng=numpy.random.default_rng(rng)). If a == b everywhere return (0.0, 0.0, 0.0)."""
    import numpy as np
    from scipy.stats import bootstrap

    if len(a) != len(b) or not a:
        raise ValueError("paired samples must have the same non-zero length")
    if n_resamples < 1:
        raise ValueError("n_resamples must be positive")
    if not 0 < confidence < 1:
        raise ValueError("confidence must be between 0 and 1")
    if all(x == y for x, y in zip(a, b)):
        return 0.0, 0.0, 0.0

    def mean_diff(x, y, axis=-1):
        return np.mean(x, axis=axis) - np.mean(y, axis=axis)

    res = bootstrap(
        (np.asarray(a, dtype=float), np.asarray(b, dtype=float)),
        statistic=mean_diff,
        paired=True,
        vectorized=False,
        n_resamples=n_resamples,
        confidence_level=confidence,
        method="percentile",
        random_state=np.random.default_rng(rng),
    )
    observed = float(np.mean(np.asarray(a, dtype=float)) - np.mean(np.asarray(b, dtype=float)))
    return observed, float(res.confidence_interval.low), float(res.confidence_interval.high)


def pass_at_k(n: int, c: int, k: int) -> float:
    """Probability that AT LEAST ONE of k trials drawn from n (c of them passing) passes.
    Unbiased estimator (Chen et al. 2021): 1 - C(n-c, k) / C(n, k). Raise ValueError if k > n."""
    if not all(isinstance(v, int) for v in (n, c, k)) or n < 0 or c < 0 or c > n or k < 1 or k > n:
        raise ValueError("require integers with 0 <= c <= n and 1 <= k <= n")
    failures = n - c
    return 1.0 - (comb(failures, k) / comb(n, k) if k <= failures else 0.0)


def pass_hat_k(n: int, c: int, k: int) -> float:
    """pass^k (tau-bench, Yao et al. 2024): probability that ALL k trials drawn from n pass.
    Unbiased estimator: C(c, k) / C(n, k). Raise ValueError if k > n.
    pass^k <= pass@1 <= pass@k; a reliable agent keeps pass^k close to pass@1."""
    if not all(isinstance(v, int) for v in (n, c, k)) or n < 0 or c < 0 or c > n or k < 1 or k > n:
        raise ValueError("require integers with 0 <= c <= n and 1 <= k <= n")
    return comb(c, k) / comb(n, k) if k <= c else 0.0
