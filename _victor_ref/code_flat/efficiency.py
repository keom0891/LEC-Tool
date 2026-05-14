"""
LEC-CBA Efficiency Indicators
===============================
Complementary metrics for cross-instrument comparison
and VfM-compatible communication.

- Cost Multiple (CM): units of cost per unit of expected payout
- Money Value (MV): coverage per unit of cost (CM inverse)
- Optimized Money Value (OMV): risk-adjusted coverage per unit of cost
  Following World Bank methodology for CCRIF SPC evaluation.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict


@dataclass
class EfficiencyIndicators:
    """Container for efficiency indicator results."""
    # Per instrument
    cost_multiples: Dict[str, float] = field(default_factory=dict)
    money_values: Dict[str, float] = field(default_factory=dict)
    omv_values: Dict[str, float] = field(default_factory=dict)

    # Aggregate
    aggregate_cost_multiple: float = 0.0
    aggregate_money_value: float = 0.0
    aggregate_omv: float = 0.0


def compute_efficiency_indicators(
    costs_by_instrument_pv: Dict[str, np.ndarray],
    payouts_by_instrument_pv: Dict[str, np.ndarray],
    total_costs_pv: np.ndarray,
    total_payouts_pv: np.ndarray,
    lambda_risk: float = 0.05
) -> EfficiencyIndicators:
    """
    Compute Cost Multiple, Money Value, and OMV indicators.

    Cost Multiple = E[Cost] / E[Payout]
        Lower is better. CM < 1 means instrument pays more than it costs.

    Money Value = E[Payout] / E[Cost]
        Higher is better. Inverse of CM.

    Optimized Money Value (OMV):
        OMV_k = (E[P_k] + λ × σ[P_k]) / NF_k

        Where NF_k is the normalization factor (annualized cost).
        The λ term rewards instruments with higher payout variability
        (more responsive to tail events). Following World Bank practice,
        λ = 0.05 by default.

        For OMV we use total PV costs as NF (mean annual cost × horizon).
    """
    indicators = EfficiencyIndicators()

    for name in costs_by_instrument_pv:
        costs = costs_by_instrument_pv[name]
        payouts = payouts_by_instrument_pv.get(name, np.zeros_like(costs))

        mean_cost = float(np.mean(costs))
        mean_payout = float(np.mean(payouts))

        # Cost Multiple
        if mean_payout > 0:
            cm = mean_cost / mean_payout
        else:
            cm = np.inf
        indicators.cost_multiples[name] = cm

        # Money Value
        if mean_cost > 0:
            mv = mean_payout / mean_cost
        else:
            mv = np.inf
        indicators.money_values[name] = mv

        # OMV: (E[P] + λ × σ[P]) / NF
        std_payout = float(np.std(payouts))
        nf = mean_cost  # normalization factor = mean PV cost
        if nf > 0:
            omv = (mean_payout + lambda_risk * std_payout) / nf
        else:
            omv = np.inf
        indicators.omv_values[name] = omv

    # Aggregate
    mean_total_cost = float(np.mean(total_costs_pv))
    mean_total_payout = float(np.mean(total_payouts_pv))
    std_total_payout = float(np.std(total_payouts_pv))

    if mean_total_payout > 0:
        indicators.aggregate_cost_multiple = mean_total_cost / mean_total_payout
    if mean_total_cost > 0:
        indicators.aggregate_money_value = mean_total_payout / mean_total_cost
        indicators.aggregate_omv = (
            (mean_total_payout + lambda_risk * std_total_payout) / mean_total_cost
        )

    return indicators
