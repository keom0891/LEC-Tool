"""
LEC-CBA Benefits Module
========================
Calculates benefits from instrument payouts.

Benefit is defined as resources effectively deployed during disasters,
not contracted capacity. The indirect benefit factor captures
second-order effects (fiscal stabilization, avoided coping costs).
"""

import numpy as np
from config import IndirectBenefitConfig


def compute_direct_benefit(payouts: dict) -> np.ndarray:
    """
    Sum payouts across all instruments for a given period.
    
    Args:
        payouts: dict mapping instrument_name -> payout amount (float)
    
    Returns:
        Total direct benefit (float)
    """
    return sum(payouts.values())


def apply_indirect_benefit(
    direct_benefit: float,
    cfg: IndirectBenefitConfig
) -> float:
    """
    Apply indirect benefit multiplier.
    Total benefit = direct × (1 + indirect_factor)
    """
    return direct_benefit * (1.0 + cfg.factor)


def compute_total_benefit(
    payouts: dict,
    indirect_cfg: IndirectBenefitConfig
) -> tuple:
    """
    Compute total benefit (direct + indirect) from all instrument payouts.
    
    Returns:
        (total_benefit, direct_benefit, indirect_benefit)
    """
    direct = compute_direct_benefit(payouts)
    total = apply_indirect_benefit(direct, indirect_cfg)
    indirect = total - direct
    return total, direct, indirect


def compute_unpaid_loss(loss: float, total_payout: float) -> float:
    """
    Financing gap: loss minus what instruments covered.
    """
    return max(0.0, loss - total_payout)
