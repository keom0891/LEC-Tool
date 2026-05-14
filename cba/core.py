"""
LEC-CBA Core Indicators
=========================
Primary evaluation metrics computed per simulation and aggregated
across the full distribution.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class CoreIndicators:
    """Container for core indicator results."""
    # Per-simulation arrays
    bc_ratios: np.ndarray = None            # shape: (num_sims,)
    unpaid_losses_pv: np.ndarray = None     # shape: (num_sims,)
    bul_ratios: np.ndarray = None           # shape: (num_sims,)

    # Per-instrument B/C ratios
    bc_ratios_by_instrument: Dict[str, np.ndarray] = field(default_factory=dict)

    # Aggregated metrics
    expected_bc: float = 0.0
    prob_bc_gt_1: float = 0.0
    expected_unpaid_loss: float = 0.0
    prob_gap_gt_threshold: Dict[float, float] = field(default_factory=dict)
    expected_bul: float = 0.0

    # Percentiles
    bc_percentiles: Dict[str, float] = field(default_factory=dict)
    ul_percentiles: Dict[str, float] = field(default_factory=dict)


def compute_core_indicators(
    benefits_pv: np.ndarray,
    costs_pv: np.ndarray,
    losses_pv: np.ndarray,
    payouts_pv: np.ndarray,
    benefits_by_instrument_pv: Dict[str, np.ndarray],
    costs_by_instrument_pv: Dict[str, np.ndarray],
    gap_thresholds: List[float] = None
) -> CoreIndicators:
    """
    Compute core indicators across all simulations.

    Args:
        benefits_pv: PV of total benefits per sim, shape (num_sims,)
        costs_pv: PV of total costs per sim, shape (num_sims,)
        losses_pv: PV of total losses per sim, shape (num_sims,)
        payouts_pv: PV of total payouts (pre indirect factor) per sim
        benefits_by_instrument_pv: dict of instrument -> PV benefits array
        costs_by_instrument_pv: dict of instrument -> PV costs array
        gap_thresholds: list of threshold values for P(gap > threshold)
    """
    if gap_thresholds is None:
        gap_thresholds = [500.0, 1000.0, 2000.0, 5000.0]

    indicators = CoreIndicators()
    num_sims = len(benefits_pv)

    # B/C ratio (aggregate)
    indicators.bc_ratios = np.where(
        costs_pv > 0, benefits_pv / costs_pv, np.inf
    )
    indicators.expected_bc = float(np.mean(indicators.bc_ratios[np.isfinite(indicators.bc_ratios)]))
    indicators.prob_bc_gt_1 = float(np.mean(indicators.bc_ratios > 1.0))

    # B/C ratio by instrument
    for name in benefits_by_instrument_pv:
        b = benefits_by_instrument_pv[name]
        c = costs_by_instrument_pv.get(name, np.zeros_like(b))
        ratio = np.where(c > 0, b / c, np.inf)
        indicators.bc_ratios_by_instrument[name] = ratio

    # Unpaid loss (financing gap)
    indicators.unpaid_losses_pv = np.maximum(0, losses_pv - payouts_pv)
    indicators.expected_unpaid_loss = float(np.mean(indicators.unpaid_losses_pv))

    # P(gap > threshold)
    for thr in gap_thresholds:
        p = float(np.mean(indicators.unpaid_losses_pv > thr))
        indicators.prob_gap_gt_threshold[thr] = p

    # B/UL ratio
    indicators.bul_ratios = np.where(
        losses_pv > 0, payouts_pv / losses_pv, 1.0
    )
    indicators.expected_bul = float(np.mean(indicators.bul_ratios))

    # Percentiles
    finite_bc = indicators.bc_ratios[np.isfinite(indicators.bc_ratios)]
    if len(finite_bc) > 0:
        for p_label, p_val in [("p5", 5), ("p25", 25), ("p50", 50),
                                ("p75", 75), ("p95", 95)]:
            indicators.bc_percentiles[p_label] = float(np.percentile(finite_bc, p_val))

    for p_label, p_val in [("p5", 5), ("p25", 25), ("p50", 50),
                            ("p75", 75), ("p95", 95)]:
        indicators.ul_percentiles[p_label] = float(
            np.percentile(indicators.unpaid_losses_pv, p_val)
        )

    return indicators
