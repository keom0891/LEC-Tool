# -*- coding: utf-8 -*-
"""
utils.py — LEC Tool
====================
Shared helper utilities used across all modules.

Functions
---------
aggregate_event_values_by_year  Sum event-level values into annual buckets.
compute_aal                      Average Annual Loss from a LEC curve.
"""

import numpy as np


def aggregate_event_values_by_year(times, values, catalogue_length):
    """
    Sum event-level values into annual buckets.

    Maps each event to the year it falls in (floor of fractional time) and
    accumulates the corresponding value.  Events outside the range
    [0, catalogue_length) are silently ignored.

    Parameters
    ----------
    times : array_like
        Event times in fractional years (e.g. 2.73 → year index 2).
    values : array_like
        Per-event scalar values to aggregate (losses, payouts, etc.).
        Must be the same length as *times*.
    catalogue_length : int
        Number of years in the simulation horizon.

    Returns
    -------
    ndarray, shape (catalogue_length,)
        Sum of values falling in each year index 0 … catalogue_length-1.

    Examples
    --------
    >>> aggregate_event_values_by_year([0.3, 1.7, 1.9], [10, 5, 8], 3)
    array([10., 13.,  0.])
    """
    annual_values = np.zeros(catalogue_length)

    if len(times) == 0:
        return annual_values

    year_index = np.floor(np.asarray(times)).astype(int)

    for y, v in zip(year_index, values):
        if 0 <= y < catalogue_length:
            annual_values[y] += v

    return annual_values


def compute_aal(loss, rate):
    """
    Compute Average Annual Loss (AAL) from a loss exceedance curve.

    Uses the symmetric trapezoid identity:

        AAL ≈ ( |∫ rate d(loss)| + |∫ loss d(rate)| ) / 2

    which is equivalent to the area enclosed between the LEC curve and
    the two coordinate axes, and reduces to the standard ∫ λ(L) dL when
    boundary terms vanish.

    Parameters
    ----------
    loss : array_like
        Loss values in *ascending* order.
    rate : array_like
        Corresponding annual exceedance rates in *descending* order.
        Must be the same length as *loss*.

    Returns
    -------
    float
        Average Annual Loss in the same monetary units as *loss*.

    Notes
    -----
    The input curve should span a meaningful range.  Narrow curves (e.g.,
    a single point) will produce a near-zero AAL regardless of the rate.
    """
    loss = np.asarray(loss, dtype=float)
    rate = np.asarray(rate, dtype=float)
    return (abs(np.trapezoid(rate, loss)) + abs(np.trapezoid(loss, rate))) / 2
