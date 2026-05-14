"""
LEC-CBA Discounting Module
============================
Time value adjustment for costs and benefits.
"""

import numpy as np
from .config import DiscountConfig


def discount_factors(cfg: DiscountConfig) -> np.ndarray:
    """
    Generate vector of discount factors for each year.
    d_t = 1 / (1 + r)^t  for t = 0, 1, ..., T-1
    """
    t = np.arange(cfg.analysis_horizon)
    return (1.0 + cfg.social_discount_rate) ** (-t)


def present_value(annual_values: np.ndarray, cfg: DiscountConfig) -> float:
    """
    Compute present value of a stream of annual values.
    PV = sum_t (value_t × discount_factor_t)
    """
    df = discount_factors(cfg)
    n = min(len(annual_values), len(df))
    return float(np.sum(annual_values[:n] * df[:n]))


def present_value_matrix(
    matrix: np.ndarray,
    cfg: DiscountConfig
) -> np.ndarray:
    """
    Discount each simulation's annual values to present value.

    Args:
        matrix: shape (num_sims, horizon_years)
        cfg: discount configuration

    Returns:
        np.ndarray of shape (num_sims,) — PV per simulation
    """
    df = discount_factors(cfg)
    n = min(matrix.shape[1], len(df))
    return np.sum(matrix[:, :n] * df[:n], axis=1)
