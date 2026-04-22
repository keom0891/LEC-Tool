# -*- coding: utf-8 -*-
"""
lec_core.py — LEC Tool
=======================
Core LEC computations: empirical curve, bootstrap confidence intervals,
and hybrid tail blending.

All functions are pure (no file I/O, no plotting).

Functions
---------
compute_empirical_lec   Build empirical LEC with bootstrap CIs from an event catalogue.
build_hybrid_lec        Extend an empirical LEC with a probabilistic tail.

Dependencies
------------
hybrid_lec  (run_hybrid_lec)
utils       (compute_aal)
"""

import numpy as np
from hybrid_lec import run_hybrid_lec
from utils import compute_aal


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_empirical_lec(
    event_loss_df,
    loss_scale_factor=1.0,
    freq_scale_factor=1.0,
    B=1000,
    random_seed=42,
):
    """
    Build an empirical Loss Exceedance Curve with bootstrap confidence intervals.

    Each observed event is assigned an empirical exceedance rate
    rank / valid_timeframe, where rank runs from 1 (largest loss) to n
    (smallest loss) and valid_timeframe is the span of the catalogue in years.
    Bootstrap resampling provides uncertainty bounds around this estimate.

    Parameters
    ----------
    event_loss_df : pandas.DataFrame
        Must contain at minimum:
          - 'year'      (int)   Calendar year of each event.
          - 'econ_loss' (float) Economic loss of each event (any consistent unit).
    loss_scale_factor : float, default 1.0
        Multiplicative scale applied to every loss before fitting.
        Use to convert units or stress-test sensitivity.
    freq_scale_factor : float, default 1.0
        Multiplicative scale applied to every exceedance rate.
    B : int, default 1000
        Number of bootstrap resamples for confidence-interval estimation.
    random_seed : int, default 42
        RNG seed for reproducibility.

    Returns
    -------
    dict with the following keys:

    empirical          ndarray, shape (n,)
        Scaled losses sorted in *descending* order.
    lambda_empirical   ndarray, shape (n,)
        Empirical exceedance rates in *ascending* order.
        Paired element-wise with *empirical*: empirical[i] ↔ lambda_empirical[i].
    lec_mean           ndarray, shape (n,)
        Bootstrap mean losses (aligned with lambda_empirical).
    lec_p05            ndarray, shape (n,)
        5th-percentile bootstrap losses.
    lec_p50            ndarray, shape (n,)
        50th-percentile bootstrap losses (median).
    lec_p95            ndarray, shape (n,)
        95th-percentile bootstrap losses.
    aal                float
        Average Annual Loss computed from the empirical LEC.
    total_loss         float
        Sum of all scaled observed losses.
    min_loss           float
        Smallest scaled observed loss.
    max_loss           float
        Largest scaled observed loss.
    lec_curve          ndarray, shape (n, 2)
        Empirical LEC as [[loss, rate], …] in *ascending* loss order.
        Column 0 is loss; column 1 is the exceedance rate.
    valid_timeframe    int
        Number of calendar years spanned by the input catalogue
        (= year_max − year_min + 1).

    Notes
    -----
    The 90 % bootstrap confidence interval is defined by lec_p05 / lec_p95.
    Resampling is performed with replacement over the full loss vector,
    so the number of events per resample equals n.
    """
    valid_timeframe = int(
        event_loss_df['year'].max() - event_loss_df['year'].min() + 1
    )
    n = len(event_loss_df['econ_loss'])

    # Step 1 — empirical curve (descending losses, ascending rates)
    empirical = loss_scale_factor * np.sort(event_loss_df['econ_loss'].values)[::-1]
    lambda_empirical = freq_scale_factor * np.arange(1, n + 1) / valid_timeframe

    # Step 2 — bootstrap
    rng = np.random.default_rng(random_seed)
    bootstrap_losses = np.zeros((B, n))
    for b in range(B):
        resampled = rng.choice(empirical, size=n, replace=True)
        bootstrap_losses[b] = np.sort(resampled)[::-1]

    # Step 3 — LEC curve in ascending loss order (for downstream consumers)
    lec_curve = np.column_stack((empirical[::-1], lambda_empirical[::-1]))

    return {
        'empirical': empirical,
        'lambda_empirical': lambda_empirical,
        'lec_mean': np.mean(bootstrap_losses, axis=0),
        'lec_p05': np.percentile(bootstrap_losses, 5, axis=0),
        'lec_p50': np.percentile(bootstrap_losses, 50, axis=0),
        'lec_p95': np.percentile(bootstrap_losses, 95, axis=0),
        'aal': compute_aal(lec_curve[:, 0], lec_curve[:, 1]),
        'total_loss': float(np.sum(empirical)),
        'min_loss': float(np.min(empirical)),
        'max_loss': float(np.max(empirical)),
        'lec_curve': lec_curve,
        'valid_timeframe': valid_timeframe,
    }


def build_hybrid_lec(lec_curve, tail_loss, tail_aep):
    """
    Extend an empirical LEC with a probabilistic tail via smooth blending.

    Delegates to ``hybrid_lec.run_hybrid_lec``, which finds the optimal
    join point in log-log space and applies a cubic Hermite blend.

    Parameters
    ----------
    lec_curve : ndarray, shape (n, 2)
        Empirical curve as [[loss, rate], …] in ascending loss order,
        as returned by ``compute_empirical_lec`` in the 'lec_curve' key.
    tail_loss : array_like
        Loss values of the probabilistic tail (e.g. from a CAT model).
        Must extend beyond the empirical range in at least the upper tail.
    tail_aep : array_like
        Annual exceedance probabilities corresponding to *tail_loss*.

    Returns
    -------
    hybrid_curve : ndarray, shape (m, 2)
        Blended curve [[loss, rate], …] in ascending loss order.
        Replaces *lec_curve* for all downstream computations when a
        hybrid approach is selected.
    aal : float
        Average Annual Loss of the hybrid curve.

    Notes
    -----
    The blending algorithm is documented in ``hybrid_lec.run_hybrid_lec``.
    Default blending hyper-parameters (n_candidates, w_level, …) are used;
    pass kwargs through if custom tuning is needed.
    """
    emp_loss = lec_curve[:, 0]
    emp_aep = lec_curve[:, 1]

    hybrid_loss, hybrid_aep = run_hybrid_lec(emp_loss, emp_aep, tail_loss, tail_aep)
    hybrid_curve = np.column_stack((hybrid_loss, hybrid_aep))
    aal = compute_aal(hybrid_loss, hybrid_aep)

    return hybrid_curve, aal
