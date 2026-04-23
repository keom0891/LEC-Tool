# -*- coding: utf-8 -*-
"""
simulation.py — LEC Tool
=========================
Synthetic catalogue generation using the Compound Poisson process with
Common Random Numbers (CRN).

Events per year are drawn from a Poisson distribution calibrated to the
maximum exceedance rate of the LEC.  Event losses are sampled via the
inverse CDF of the LEC.  CRN streams (N_events, U_times, U_loss) are
returned alongside the catalogues so that downstream modules (e.g. ex-ante
risk reduction) can replay the same event sequence under a modified LEC.

All functions are pure (no file I/O, no plotting).

Functions
---------
build_inv_cdf               Monotone inverse CDF from a LEC curve.
make_random_streams         Pre-generate Poisson counts and uniform draws (CRN).
generate_synthetic_catalogue  Full catalogue generation pipeline.
"""

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def build_inv_cdf(lambda_loss, LS):
    """
    Build a monotone inverse CDF (loss = F⁻¹(u)) from a LEC curve.

    Converts the exceedance-rate curve into a proper CDF, enforces strict
    monotonicity (required by ``interp1d``), and returns an interpolator
    that maps uniform draws u ∈ [0, 1] to loss values.

    Parameters
    ----------
    lambda_loss : array_like, shape (n,)
        Annual exceedance rates of the LEC, in *ascending* order
        (i.e. rates go from λ_max at index 0 to λ_min at index n-1 when
        the curve is stored with ascending loss — pass them as stored in
        ``lec_curve[:, 1]``).
    LS : array_like, shape (n,)
        Loss values corresponding to *lambda_loss*, in *ascending* order
        (as stored in ``lec_curve[:, 0]``).

    Returns
    -------
    inv_cdf : callable
        Interpolator: inv_cdf(u) → loss.  Out-of-range u values are
        clamped to the min/max loss.
    lambda_min : float
        Maximum exceedance rate of the curve (= the rate of the smallest
        loss). Used as the Poisson intensity parameter λ when drawing the
        number of events per year.

    Notes
    -----
    The CDF is defined as F(L) = 1 − λ(L) / λ_min, where λ_min is
    max(lambda_loss). A small epsilon jitter is added to enforce strict
    monotonicity before interpolation.
    """
    lam = np.asarray(lambda_loss, float)
    L = np.asarray(LS, float)

    lambda_min = lam.max()
    cdf_vals = 1.0 - lam / lambda_min

    # Enforce strict monotonicity for interp1d
    eps = np.finfo(float).eps
    cdf_vals = np.maximum.accumulate(
        cdf_vals + np.arange(1, len(cdf_vals) + 1) * eps
    )

    inv = interp1d(
        cdf_vals, L,
        kind='linear',
        bounds_error=False,
        fill_value=(L[0], L[-1]),
    )
    return inv, lambda_min


def make_random_streams(simulation_number, catalogue_length, lam, seed):
    """
    Pre-generate Common Random Number (CRN) streams for the catalogue.

    Draws all Poisson event counts and uniform random numbers once and
    stores them.  Re-using these same streams across different LEC curves
    (base vs. reduced) ensures that differences in catalogue outcomes are
    driven purely by the modified curve, not by sampling noise.

    Parameters
    ----------
    simulation_number : int
        Number of independent synthetic catalogues to generate.
    catalogue_length : int
        Duration of each catalogue in years.
    lam : float
        Poisson intensity (expected events per year).  Should equal
        ``lambda_min`` returned by ``build_inv_cdf``.
    seed : int
        RNG seed for reproducibility.

    Returns
    -------
    N : ndarray, shape (simulation_number, catalogue_length)
        Poisson event counts for each simulation and year.
    U_times : list of lists of ndarray
        U_times[k][y] is a sorted 1-D array of uniform draws used to
        place event times within year y of simulation k.
    U_loss : list of lists of ndarray
        U_loss[k][y] is a 1-D array of uniform draws used to sample
        event losses for year y of simulation k via the inverse CDF.
        Lengths match U_times[k][y].
    """
    rng = np.random.default_rng(seed)
    N = rng.poisson(lam=lam, size=(simulation_number, catalogue_length))

    U_times = []
    U_loss = []

    for k in range(simulation_number):
        row_times, row_loss = [], []
        for y in range(catalogue_length):
            n = int(N[k, y])
            if n > 0:
                u_t = np.sort(rng.random(n))
                u_l = rng.random(n)
            else:
                u_t = np.empty(0)
                u_l = np.empty(0)
            row_times.append(u_t)
            row_loss.append(u_l)
        U_times.append(row_times)
        U_loss.append(row_loss)

    return N, U_times, U_loss


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_synthetic_catalogue(
    lec_curve,
    catalogue_length,
    simulation_number,
    random_seed=42,
):
    """
    Generate a stochastic event catalogue from a Loss Exceedance Curve.

    Uses a Compound Poisson process: the number of events per year follows
    Poisson(λ_min), and individual event losses are sampled from the inverse
    CDF of the LEC.  Annual aggregate and annual maximum loss matrices are
    produced for all simulations.

    Parameters
    ----------
    lec_curve : ndarray, shape (n, 2)
        LEC as [[loss, rate], …] in ascending loss order, as returned by
        ``lec_core.compute_empirical_lec`` or ``lec_core.build_hybrid_lec``.
        Column 0 is loss; column 1 is annual exceedance rate.
    catalogue_length : int
        Simulation horizon in years (e.g. 10).
    simulation_number : int
        Number of independent synthetic catalogues (e.g. 1000).
    random_seed : int, default 42
        RNG seed.  Changing this value produces a different but fully
        reproducible set of catalogues.

    Returns
    -------
    dict with the following keys:

    event_catalogue    list of dicts, length simulation_number
        event_catalogue[k] = {'times': ndarray, 'losses': ndarray}
        where *times* are fractional years (e.g. 2.73) and *losses* are
        the corresponding event-level economic losses.  Both arrays are
        sorted by time.  An empty catalogue has shape-(0,) arrays.

    synthetic_annual_df    pandas.DataFrame, shape (simulation_number, catalogue_length)
        Aggregate annual losses per simulation. Column names: Year_1 … Year_N.

    synthetic_annual_max_df  pandas.DataFrame, shape (simulation_number, catalogue_length)
        Maximum single-event loss per year per simulation.
        Same column names as *synthetic_annual_df*.

    N_events   ndarray, shape (simulation_number, catalogue_length)
        Poisson event counts used to generate the catalogue.
        Pass to risk_reduction.generate_reduced_catalogue to replay
        the same event occurrence sequence under a modified LEC.

    U_times    list of lists of ndarray
        Uniform draws used to place event times (CRN stream).
        Same indexing as *N_events*.

    U_loss     list of lists of ndarray
        Uniform draws used to sample event losses (CRN stream).
        Same indexing as *N_events*.

    Notes
    -----
    The CRN streams (N_events, U_times, U_loss) must be forwarded to
    ``risk_reduction.generate_reduced_catalogue`` when ex-ante risk
    reduction is applied, so that base and reduced catalogues share
    the same random draws.
    """
    lambda_loss = lec_curve[:, 1]
    LS = lec_curve[:, 0]

    inv_cdf, lambda_min = build_inv_cdf(lambda_loss, LS)
    N_events, U_times, U_loss = make_random_streams(
        simulation_number, catalogue_length, lambda_min, random_seed
    )

    # --- Build event-level catalogue ---
    event_catalogue = [None] * simulation_number

    for k in range(simulation_number):
        times_k, losses_k = [], []

        for y in range(catalogue_length):
            n = int(N_events[k, y])
            if n == 0:
                continue
            t = U_times[k][y] + y          # absolute fractional year
            l = inv_cdf(U_loss[k][y])      # loss via inverse CDF
            times_k.append(t)
            losses_k.append(l)

        if len(times_k) == 0:
            times_all = np.empty(0)
            losses_all = np.empty(0)
        else:
            times_all = np.concatenate(times_k)
            losses_all = np.concatenate(losses_k)

        event_catalogue[k] = {'times': times_all, 'losses': losses_all}

    # --- Aggregate to annual matrices ---
    column_names = [f'Year_{i + 1}' for i in range(catalogue_length)]
    synthetic_catalogue_annual = []
    synthetic_catalogue_max = []

    for i in range(simulation_number):
        partial_df = pd.DataFrame(event_catalogue[i])

        if partial_df.empty:
            synthetic_catalogue_annual.extend([0] * catalogue_length)
            synthetic_catalogue_max.extend([0] * catalogue_length)
            continue

        partial_df['times'] = np.floor(partial_df['times']).astype(int)

        agg = partial_df.groupby('times')['losses'].sum()
        mx = partial_df.groupby('times')['losses'].max()

        synthetic_catalogue_annual.extend(
            agg.reindex(range(catalogue_length), fill_value=0).values
        )
        synthetic_catalogue_max.extend(
            mx.reindex(range(catalogue_length), fill_value=0).values
        )

    synthetic_annual_df = pd.DataFrame(
        np.reshape(synthetic_catalogue_annual, (simulation_number, catalogue_length)),
        columns=column_names,
    )
    synthetic_annual_max_df = pd.DataFrame(
        np.reshape(synthetic_catalogue_max, (simulation_number, catalogue_length)),
        columns=column_names,
    )

    return {
        'event_catalogue': event_catalogue,
        'synthetic_annual_df': synthetic_annual_df,
        'synthetic_annual_max_df': synthetic_annual_max_df,
        'N_events': N_events,
        'U_times': U_times,
        'U_loss': U_loss,
    }