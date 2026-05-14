"""
cba/engine.py — LEC Tool CBA Extension
========================================
Orchestrates cost-benefit analysis on top of risk_management.apply_strategy outputs.

Entry point: run_cba()
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from cba.config import LECCBAConfig
from cba.costs import (
    insurance_annual_costs,
    ppo_annual_costs,
    ccf_annual_costs,
    ddo_annual_costs,
)
from cba.discounting import present_value_matrix
from cba.core import compute_core_indicators, CoreIndicators
from cba.efficiency import compute_efficiency_indicators, EfficiencyIndicators


@dataclass
class CBAResults:
    """Complete results from the LEC-CBA analysis."""
    config: LECCBAConfig = None

    # Raw matrices (num_sims × horizon)
    losses_matrix: np.ndarray = None
    total_costs_matrix: np.ndarray = None
    total_benefits_matrix: np.ndarray = None
    total_payouts_matrix: np.ndarray = None
    unpaid_losses_matrix: np.ndarray = None

    # Per-instrument matrices
    costs_by_instrument: Dict[str, np.ndarray] = field(default_factory=dict)
    payouts_by_instrument: Dict[str, np.ndarray] = field(default_factory=dict)
    benefits_by_instrument: Dict[str, np.ndarray] = field(default_factory=dict)

    # Indicators
    core: CoreIndicators = None
    efficiency: EfficiencyIndicators = None

    # Optional fields (not computed by run_cba, available for downstream use)
    drr: Optional[object] = None
    sensitivity: Optional[object] = None


def _get_cost_function(instrument_type: str):
    """Map instrument type string to its cost function."""
    mapping = {
        'insurance': insurance_annual_costs,
        'ppo': ppo_annual_costs,
        'ccf': ccf_annual_costs,
        'ddo': ddo_annual_costs,
    }
    if instrument_type not in mapping:
        raise ValueError(
            f"Unknown instrument type '{instrument_type}'. "
            f"Supported: {list(mapping.keys())}"
        )
    return mapping[instrument_type]


def _get_instrument_config(instrument_type: str, cba_config: LECCBAConfig):
    """Map instrument type string to its config object."""
    mapping = {
        'insurance': cba_config.insurance,
        'ppo': cba_config.ppo,
        'ccf': cba_config.ccf,
        'ddo': cba_config.ddo,
    }
    return mapping[instrument_type]


def run_cba(
    losses_df: pd.DataFrame,
    payout_dfs: list,
    drm_configs: list,
    cba_config: LECCBAConfig,
) -> CBAResults:
    """
    Run cost-benefit analysis on outputs from risk_management.apply_strategy.

    Parameters
    ----------
    losses_df : pd.DataFrame, shape (num_sims, catalogue_length)
        Aggregate annual losses per simulation.
        From simulation.generate_synthetic_catalogue ['synthetic_annual_df'].
    payout_dfs : list of pd.DataFrame
        Annual payouts per instrument, from apply_strategy ['payout_dfs'].
        payout_dfs[j] corresponds to drm_configs[j].
    drm_configs : list of dicts
        Same list passed to apply_strategy. Each dict has a 'type' key.
    cba_config : LECCBAConfig
        CBA configuration parameters.

    Returns
    -------
    CBAResults
    """
    losses_matrix = losses_df.values.astype(float)
    num_sims, horizon = losses_matrix.shape
    n_instruments = len(drm_configs)

    results = CBAResults(config=cba_config)
    results.losses_matrix = losses_matrix

    # Initialize output matrices
    results.total_costs_matrix = np.zeros((num_sims, horizon))
    results.total_benefits_matrix = np.zeros((num_sims, horizon))
    results.total_payouts_matrix = np.zeros((num_sims, horizon))
    results.unpaid_losses_matrix = np.zeros((num_sims, horizon))

    # Initialize per-instrument matrices using unique instrument types.
    # Multiple instruments of the same type (e.g. two DDOs) are accumulated
    # under the same key so their combined costs/payouts are reported together.
    instrument_types = list(dict.fromkeys(cfg['type'] for cfg in drm_configs))
    for itype in instrument_types:
        results.costs_by_instrument[itype] = np.zeros((num_sims, horizon))
        results.payouts_by_instrument[itype] = np.zeros((num_sims, horizon))
        results.benefits_by_instrument[itype] = np.zeros((num_sims, horizon))

    # --- Per-simulation loop ---
    for s in range(num_sims):
        total_payout_s = np.zeros(horizon)

        for j, drm_cfg in enumerate(drm_configs):
            itype = drm_cfg['type']
            annual_payouts = payout_dfs[j].iloc[s].values.astype(float)

            # Get cost function and instrument config
            cost_fn = _get_cost_function(itype)
            inst_cfg = _get_instrument_config(itype, cba_config)

            # Compute annual costs
            if itype == 'insurance':
                annual_costs = cost_fn(inst_cfg, horizon)
            else:
                annual_costs = cost_fn(inst_cfg, annual_payouts)

            # Benefits = payouts × (1 + indirect_factor)
            annual_benefits = annual_payouts * (1.0 + cba_config.indirect_benefit.factor)

            # Accumulate with += so multiple instruments of the same type combine
            results.costs_by_instrument[itype][s] += annual_costs
            results.payouts_by_instrument[itype][s] += annual_payouts
            results.benefits_by_instrument[itype][s] += annual_benefits

            results.total_costs_matrix[s] += annual_costs
            results.total_benefits_matrix[s] += annual_benefits
            total_payout_s += annual_payouts

        results.total_payouts_matrix[s] = total_payout_s
        results.unpaid_losses_matrix[s] = np.maximum(
            0.0, losses_matrix[s] - total_payout_s
        )

    # --- Discount to present value ---
    benefits_pv = present_value_matrix(results.total_benefits_matrix, cba_config.discount)
    costs_pv = present_value_matrix(results.total_costs_matrix, cba_config.discount)
    losses_pv = present_value_matrix(results.losses_matrix, cba_config.discount)
    payouts_pv = present_value_matrix(results.total_payouts_matrix, cba_config.discount)

    benefits_by_inst_pv = {}
    costs_by_inst_pv = {}
    payouts_by_inst_pv = {}
    for itype in instrument_types:
        benefits_by_inst_pv[itype] = present_value_matrix(
            results.benefits_by_instrument[itype], cba_config.discount
        )
        costs_by_inst_pv[itype] = present_value_matrix(
            results.costs_by_instrument[itype], cba_config.discount
        )
        payouts_by_inst_pv[itype] = present_value_matrix(
            results.payouts_by_instrument[itype], cba_config.discount
        )

    # --- Indicators ---
    results.core = compute_core_indicators(
        benefits_pv=benefits_pv,
        costs_pv=costs_pv,
        losses_pv=losses_pv,
        payouts_pv=payouts_pv,
        benefits_by_instrument_pv=benefits_by_inst_pv,
        costs_by_instrument_pv=costs_by_inst_pv,
    )

    results.efficiency = compute_efficiency_indicators(
        costs_by_instrument_pv=costs_by_inst_pv,
        payouts_by_instrument_pv=payouts_by_inst_pv,
        total_costs_pv=costs_pv,
        total_payouts_pv=payouts_pv,
        lambda_risk=cba_config.omv.lambda_risk_adjustment,
    )

    return results
