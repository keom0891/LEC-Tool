"""
LEC-CBA Diagnostic Indicators
================================
- DRR Evaluation: cost-effectiveness of risk reduction investments
- Sensitivity Analysis: parameter variation and elasticity
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Tuple


# ============================================================
# DRR Cost-Effectiveness Evaluation
# ============================================================

@dataclass
class DRRAnalysis:
    """
    Results of the Disaster Risk Reduction cost-effectiveness analysis.

    Two types of benefit are computed:
      - Direct: reduction in total expected losses (independent of instruments)
      - Indirect: reduction in the financing gap (interaction with instruments)

    The direct benefit answers: "How much does prevention reduce losses?"
    The indirect benefit answers: "How much does prevention reduce the
    portion of losses that instruments cannot cover?"
    """
    # PV of DRR investments (deterministic)
    pv_drr_cost: float = 0.0

    # Direct benefit: E[L_original] - E[L_reduced]
    pv_direct_benefit: float = 0.0
    bc_direct: float = 0.0  # PV(direct benefit) / PV(cost)

    # Indirect benefit: E[UL_original] - E[UL_reduced]
    pv_indirect_benefit: float = 0.0
    bc_indirect: float = 0.0  # PV(indirect benefit) / PV(cost)

    # Per-simulation distributions
    direct_benefit_dist: np.ndarray = None   # shape: (num_sims,)
    indirect_benefit_dist: np.ndarray = None  # shape: (num_sims,)

    # Cumulative reduction vector
    cumulative_reduction: List[float] = field(default_factory=list)


def compute_drr_analysis(
    losses_pv_original: np.ndarray,
    losses_pv_reduced: np.ndarray,
    unpaid_pv_original: np.ndarray,
    unpaid_pv_reduced: np.ndarray,
    inv_vector: List[float],
    discount_rate: float,
    cumulative_reduction: List[float]
) -> DRRAnalysis:
    """
    Compute DRR cost-effectiveness indicators.

    Args:
        losses_pv_original: PV of losses per sim, original catalog (num_sims,)
        losses_pv_reduced: PV of losses per sim, reduced catalog (num_sims,)
        unpaid_pv_original: PV of unpaid losses per sim, original (num_sims,)
        unpaid_pv_reduced: PV of unpaid losses per sim, reduced (num_sims,)
        inv_vector: annual investment amounts (M USD)
        discount_rate: social discount rate
        cumulative_reduction: cumulative AAL reduction per year
    """
    drr = DRRAnalysis()
    drr.cumulative_reduction = cumulative_reduction

    # PV of DRR cost (deterministic)
    drr.pv_drr_cost = sum(
        inv / (1 + discount_rate)**t
        for t, inv in enumerate(inv_vector)
    )

    # Direct benefit: reduction in total losses
    drr.direct_benefit_dist = losses_pv_original - losses_pv_reduced
    drr.pv_direct_benefit = float(np.mean(drr.direct_benefit_dist))

    # Indirect benefit: reduction in financing gap
    drr.indirect_benefit_dist = unpaid_pv_original - unpaid_pv_reduced
    drr.pv_indirect_benefit = float(np.mean(drr.indirect_benefit_dist))

    # B/C ratios
    if drr.pv_drr_cost > 0:
        drr.bc_direct = drr.pv_direct_benefit / drr.pv_drr_cost
        drr.bc_indirect = drr.pv_indirect_benefit / drr.pv_drr_cost

    return drr


# ============================================================
# Sensitivity Analysis
# ============================================================

@dataclass
class SensitivityResult:
    """Result of a single sensitivity run."""
    parameter_name: str
    parameter_value: float
    base_value: float
    expected_bc: float
    prob_bc_gt_1: float
    expected_gap: float
    elasticity: float = 0.0


@dataclass
class SensitivityAnalysis:
    """Collection of sensitivity results."""
    results: List[SensitivityResult] = field(default_factory=list)
    tornado_data: Dict[str, Tuple[float, float]] = field(default_factory=dict)

    def add_result(self, result: SensitivityResult):
        self.results.append(result)

    def compute_tornado(self):
        """Organize results into tornado chart format."""
        params = set(r.parameter_name for r in self.results)
        for param in params:
            param_results = [r for r in self.results if r.parameter_name == param]
            if len(param_results) >= 2:
                bcs = [r.expected_bc for r in param_results]
                self.tornado_data[param] = (min(bcs), max(bcs))
