# -*- coding: utf-8 -*-
"""
risk_reduction.py — LEC Tool
==============================
Ex-ante risk reduction modelling.

The user-supplied per-year AAL reduction targets are represented as a
downward shift of the LEC curve. A bisection algorithm calibrates the
shift parameter (alpha) so that the shifted curve achieves the requested
AAL reduction. The modified LEC is then used to regenerate the event
catalogue using the exact same Compound Poisson draws (CRN streams) as
the base run, so that differences in outcomes are attributable solely to
the risk reduction and not to sampling noise.

All functions are pure (no file I/O, no plotting).

Functions
---------
calibrate_LEC_AAL           Calibrate LEC shift to match a target AAL reduction.
generate_reduced_catalogue  Replay CRN streams with per-year reduced LEC curves.
"""

import numpy as np
import pandas as pd

from simulation import build_inv_cdf
from utils import compute_aal


# ---------------------------------------------------------------------------
# Ex-ante calibration
# ---------------------------------------------------------------------------

def calibrate_LEC_AAL(Ct, L, lam, L_cut=None, tol=1e-6, max_iter=80):
    """
    Calibrate a parametric LEC downward shift to match a target AAL reduction.

    Finds alpha such that the shifted LEC achieves an AAL reduction of *Ct*
    relative to the original curve. The shift applies a non-linear
    transformation to losses below *L_cut*:

        Lr(L) = L · (1 − (1 − L/L_cut)^alpha)   for L ≤ L_cut
        Lr(L) = L                                  for L >  L_cut

    * alpha → 0  :  Lr → 0 for all L ≤ L_cut  (maximum reduction)
    * alpha → ∞  :  Lr → L                     (no reduction)

    A bisection loop over alpha is used to match *Ct*.  An AAL proxy
    based on the derivative dLr/dL drives the bisection; the final Lr
    and AAL are computed exactly at convergence.

    Parameters
    ----------
    Ct : float
        Target AAL reduction ($MM).  Must satisfy 0 ≤ Ct ≤ C_max, where
        C_max is the maximum feasible reduction for the chosen L_cut.
    L : array_like, shape (n,)
        Loss values in ascending order (as stored in lec_curve[:, 0]).
    lam : array_like, shape (n,)
        Annual exceedance rates in descending order (as stored in lec_curve[:, 1]).
    L_cut : float, optional
        Upper loss threshold above which no reduction is applied.
        Defaults to max(L), meaning all losses are eligible for reduction.
    tol : float, default 1e-6
        Convergence tolerance on the AAL proxy.
    max_iter : int, default 80
        Maximum bisection iterations.

    Returns
    -------
    Lr : ndarray, shape (n,)
        Reduced loss values (same shape as L; losses above L_cut unchanged).
    aal : float
        Average Annual Loss of the reduced curve (not the reduction itself).

    Raises
    ------
    ValueError
        If Ct is outside [0, C_max].
    """
    L = np.asarray(L, float)
    lam = np.asarray(lam, float)
    L_cut = float(L.max()) if L_cut is None else float(L_cut)

    def dLred(a):
        """Derivative d(Lr)/dL as a function of alpha."""
        mask = L <= L_cut
        t = np.clip(1.0 - L[mask] / L_cut, 1e-15, 1.0)
        d = np.ones_like(L)
        d[mask] = np.clip(
            1.0 - t ** a + (a * L[mask] / L_cut) * t ** (a - 1),
            0.0, None,
        )
        return d

    # Proxy for AAL reduction (faster than building Lr on every bisection step)
    def _aal_proxy(a):
        mod_lam = lam * (1.0 - dLred(a))
        return (abs(np.trapezoid(mod_lam, L)) + abs(np.trapezoid(L, mod_lam))) / 2

    C_max = (
        abs(np.trapezoid(lam * (L <= L_cut), L))
        + abs(np.trapezoid(L, lam * (L <= L_cut)))
    ) / 2

    if not 0.0 <= Ct <= C_max + 1e-12:
        raise ValueError(
            f"C_target={Ct:.6g} is outside the feasible range [0, {C_max:.6g}]. "
            "Reduce the target AAL reduction or increase L_cut."
        )

    if Ct == 0.0:
        alpha_star = np.inf
    else:
        lo, hi = 0.0, 1.0
        while _aal_proxy(hi) > Ct and hi < 1e12:
            hi *= 2
        for _ in range(max_iter):
            mid = 0.5 * (lo + hi)
            Cm = _aal_proxy(mid)
            if abs(Cm - Ct) <= tol * max(1.0, Ct):
                alpha_star = mid
                break
            lo, hi = (mid, hi) if Cm > Ct else (lo, mid)
        else:
            alpha_star = 0.5 * (lo + hi)

    # Build reduced losses exactly
    mask = L <= L_cut
    s = np.zeros_like(L)
    if np.isfinite(alpha_star):
        s[mask] = (1.0 - L[mask] / L_cut) ** alpha_star
    Lr = L.copy()
    Lr[mask] = L[mask] * (1.0 - s[mask])

    return Lr, compute_aal(Lr, lam)


# ---------------------------------------------------------------------------
# Ex-ante catalogue generation
# ---------------------------------------------------------------------------

def generate_reduced_catalogue(
    base_lec_curve,
    red,
    N_events,
    U_times,
    U_loss,
    catalogue_length,
    simulation_number,
):
    """
    Regenerate the synthetic catalogue under per-year reduced LEC curves.

    For each unique value in *red*, the base LEC is shifted downward using
    ``calibrate_LEC_AAL``.  The same Poisson counts (N_events) and uniform
    draws (U_times, U_loss) used to build the base catalogue are replayed
    through the modified inverse CDFs, so that the two catalogues differ
    only because of the changed loss distribution, not because of different
    random numbers.

    Parameters
    ----------
    base_lec_curve : ndarray, shape (n, 2)
        Base LEC as [[loss, rate], …] in ascending loss order, as returned
        by ``lec_core.compute_empirical_lec`` or ``lec_core.build_hybrid_lec``.
    red : list of float, length catalogue_length
        Per-year cumulative AAL reduction targets ($MM), as returned by
        ``compute_expost_reduction``.  red[y] is applied to year y.
    N_events : ndarray, shape (simulation_number, catalogue_length)
        Poisson event counts from the base simulation (CRN stream).
    U_times : list of lists of ndarray
        Uniform draws for event timing (CRN stream).
    U_loss : list of lists of ndarray
        Uniform draws for event loss sampling (CRN stream).
    catalogue_length : int
        Number of years per simulation.
    simulation_number : int
        Number of independent synthetic catalogues.

    Returns
    -------
    dict with the following keys:

    event_catalogue_red         list of dicts, length simulation_number
        Same structure as ``simulation.generate_synthetic_catalogue``
        ['event_catalogue']: each element has 'times' and 'losses' arrays.

    synthetic_annual_red_df     pandas.DataFrame, shape (simulation_number, catalogue_length)
        Aggregate annual losses under risk reduction. Columns: Year_1 … Year_N.

    synthetic_annual_max_red_df pandas.DataFrame, shape (simulation_number, catalogue_length)
        Annual maximum single-event loss under risk reduction.

    lec_curves_by_target        dict
        Maps each unique C_target value (from *red*) to:
          'Loss'   ndarray  Reduced loss values (same length as base_lec_curve).
          'Lambda' ndarray  Exceedance rates (unchanged from base curve).
          'aal'    float    AAL of the reduced curve.
        Use this to plot the family of reduced LEC curves.
    """
    L = base_lec_curve[:, 0]
    lam = base_lec_curve[:, 1]
    column_names = [f'Year_{i + 1}' for i in range(catalogue_length)]

    # --- Build one reduced LEC per unique reduction target ---
    lec_curves_by_target = {}
    for C_target in np.unique(red):
        Lr, aal = calibrate_LEC_AAL(Ct=C_target, L=L, lam=lam)
        lec_curves_by_target[C_target] = {'Loss': Lr, 'Lambda': lam, 'aal': aal}

    # --- Build inverse CDFs for each unique target ---
    inv_cdf_cache = {}
    for C_target, curve in lec_curves_by_target.items():
        inv_cdf_cache[C_target], _ = build_inv_cdf(curve['Lambda'], curve['Loss'])

    # Map each year to its inverse CDF
    inv_cdf_year = [inv_cdf_cache[float(c)] for c in red]

    # --- Replay CRN streams through reduced inverse CDFs ---
    event_catalogue_red = [None] * simulation_number

    for k in range(simulation_number):
        times_k, losses_k = [], []

        for y in range(catalogue_length):
            n = int(N_events[k, y])
            if n == 0:
                continue
            t = U_times[k][y] + y
            l = inv_cdf_year[y](U_loss[k][y])
            times_k.append(t)
            losses_k.append(l)

        if len(times_k) == 0:
            times_all = np.empty(0)
            losses_all = np.empty(0)
        else:
            times_all = np.concatenate(times_k)
            losses_all = np.concatenate(losses_k)

        event_catalogue_red[k] = {'times': times_all, 'losses': losses_all}

    # --- Aggregate to annual matrices ---
    synthetic_catalogue_annual_red = []
    synthetic_catalogue_max_red = []

    for i in range(simulation_number):
        partial_df = pd.DataFrame(event_catalogue_red[i])

        if partial_df.empty:
            synthetic_catalogue_annual_red.extend([0] * catalogue_length)
            synthetic_catalogue_max_red.extend([0] * catalogue_length)
            continue

        partial_df['times'] = np.floor(partial_df['times']).astype(int)

        agg = partial_df.groupby('times')['losses'].sum()
        mx = partial_df.groupby('times')['losses'].max()

        synthetic_catalogue_annual_red.extend(
            agg.reindex(range(catalogue_length), fill_value=0).values
        )
        synthetic_catalogue_max_red.extend(
            mx.reindex(range(catalogue_length), fill_value=0).values
        )

    synthetic_annual_red_df = pd.DataFrame(
        np.reshape(synthetic_catalogue_annual_red, (simulation_number, catalogue_length)),
        columns=column_names,
    )
    synthetic_annual_max_red_df = pd.DataFrame(
        np.reshape(synthetic_catalogue_max_red, (simulation_number, catalogue_length)),
        columns=column_names,
    )

    return {
        'event_catalogue_red': event_catalogue_red,
        'synthetic_annual_red_df': synthetic_annual_red_df,
        'synthetic_annual_max_red_df': synthetic_annual_max_red_df,
        'lec_curves_by_target': lec_curves_by_target,
    }