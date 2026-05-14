"""
LEC-CBA Engine Module
=======================
Main orchestrator aligned to BID's LEC tool instruments:
  1. CCRIF (standard_insurance_payout) — parametric insurance
  2. PPO   (apply_ppo_coverage)        — contingent credit, single trigger
  3. CCF   (ccf_coverage)              — contingent credit, recurrent

Pipeline:
  1. Load LEC data and generate synthetic catalog
  2. For each simulation and year: compute costs, benefits, payouts
  3. Discount to present value
  4. Compute indicators (core, efficiency, DRR)
  5. If DRR enabled: re-run with reduced catalog and build comparative table
  6. Run sensitivity analysis
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, Optional
import copy

from config import LECCBAConfig, get_default_config
from io_bridge import LECData, load_and_prepare
from costs import (
    PPOState, CCFState,
    insurance_cost_primary, insurance_cost_alternative,
    ppo_cost_primary,
    ccf_cost_primary,
)
from benefits import compute_total_benefit, compute_unpaid_loss
from discounting import present_value_matrix, discount_factors
from core import compute_core_indicators, CoreIndicators
from efficiency import compute_efficiency_indicators, EfficiencyIndicators
from diagnostics import (
    DRRAnalysis, compute_drr_analysis,
    SensitivityAnalysis, SensitivityResult,
)


INSTRUMENT_NAMES = ["insurance", "ppo", "ccf"]


@dataclass
class SimulationResults:
    """Complete results from the LEC-CBA analysis."""
    config: LECCBAConfig = None
    lec_data: LECData = None

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

    # Indicator results
    core: CoreIndicators = None
    efficiency: EfficiencyIndicators = None

    # DRR analysis (None if DRR not enabled)
    drr: Optional[DRRAnalysis] = None

    # Sensitivity
    sensitivity: Optional[SensitivityAnalysis] = None


def run_single_simulation(
    losses: np.ndarray,
    cfg: LECCBAConfig
) -> dict:
    """
    Run CBA analysis for a single simulation (one row of synthetic catalog).

    Each instrument evaluates INDEPENDENTLY against the TOTAL fiscal loss.
    This matches the BID's LEC tool architecture:
        drm1_payout = f(loss, drm1_params)   # CCRIF
        drm2_payout = f(loss, drm2_params)   # PPO
        drm3_payout = f(loss, drm3_params)   # CCF
        total_coverage = drm1 + drm2 + drm3
    """
    horizon = len(losses)

    costs = {name: np.zeros(horizon) for name in INSTRUMENT_NAMES}
    payouts = {name: np.zeros(horizon) for name in INSTRUMENT_NAMES}

    # Initialize stateful instruments
    ppo_state = PPOState()
    ccf_state = CCFState()
    insurance_claims = []

    for t in range(horizon):
        loss = losses[t]

        # --- CCRIF (Parametric Insurance) ---
        if cfg.insurance.cost_function == "primary":
            ins_cost, ins_payout = insurance_cost_primary(
                cfg.insurance, t, loss
            )
        else:
            ins_cost, ins_payout = insurance_cost_alternative(
                cfg.insurance, t, loss, insurance_claims
            )
        if ins_payout > 0:
            insurance_claims.append(ins_payout)
        costs["insurance"][t] = ins_cost
        payouts["insurance"][t] = ins_payout

        # --- PPO (Contingent Credit — single trigger) ---
        ppo_cost, ppo_payout, ppo_state = ppo_cost_primary(
            cfg.ppo, ppo_state, t, loss
        )
        costs["ppo"][t] = ppo_cost
        payouts["ppo"][t] = ppo_payout

        # --- CCF (Contingent Credit — recurrent) ---
        ccf_cost, ccf_payout_val, ccf_state = ccf_cost_primary(
            cfg.ccf, ccf_state, t, loss
        )
        costs["ccf"][t] = ccf_cost
        payouts["ccf"][t] = ccf_payout_val

    return {"costs": costs, "payouts": payouts}


def run_analysis(
    lec_data: LECData,
    cfg: LECCBAConfig,
    catalog_override: np.ndarray = None
) -> SimulationResults:
    """
    Run the complete LEC-CBA analysis across all simulations.

    Args:
        lec_data: LEC data container
        cfg: configuration parameters
        catalog_override: if provided, use this catalog instead of
                          lec_data.synthetic_annual_losses (used for DRR)
    """
    catalog = catalog_override if catalog_override is not None else lec_data.synthetic_annual_losses
    num_sims, horizon = catalog.shape

    results = SimulationResults()
    results.config = cfg
    results.lec_data = lec_data
    results.losses_matrix = catalog

    # Initialize matrices
    results.total_costs_matrix = np.zeros((num_sims, horizon))
    results.total_benefits_matrix = np.zeros((num_sims, horizon))
    results.total_payouts_matrix = np.zeros((num_sims, horizon))
    results.unpaid_losses_matrix = np.zeros((num_sims, horizon))

    for name in INSTRUMENT_NAMES:
        results.costs_by_instrument[name] = np.zeros((num_sims, horizon))
        results.payouts_by_instrument[name] = np.zeros((num_sims, horizon))
        results.benefits_by_instrument[name] = np.zeros((num_sims, horizon))

    # Run each simulation
    for s in range(num_sims):
        sim_result = run_single_simulation(catalog[s], cfg)

        for name in INSTRUMENT_NAMES:
            results.costs_by_instrument[name][s] = sim_result["costs"][name]
            results.payouts_by_instrument[name][s] = sim_result["payouts"][name]

            # Benefit = payout × (1 + indirect_factor)
            results.benefits_by_instrument[name][s] = (
                sim_result["payouts"][name] * (1.0 + cfg.indirect_benefit.factor)
            )

        # Period-level totals
        for t in range(horizon):
            total_cost = sum(sim_result["costs"][n][t] for n in INSTRUMENT_NAMES)
            total_payout = sum(sim_result["payouts"][n][t] for n in INSTRUMENT_NAMES)
            total_benefit = total_payout * (1.0 + cfg.indirect_benefit.factor)
            unpaid = compute_unpaid_loss(catalog[s, t], total_payout)

            results.total_costs_matrix[s, t] = total_cost
            results.total_benefits_matrix[s, t] = total_benefit
            results.total_payouts_matrix[s, t] = total_payout
            results.unpaid_losses_matrix[s, t] = unpaid

    # ---- Discount to present value ----
    benefits_pv = present_value_matrix(results.total_benefits_matrix, cfg.discount)
    costs_pv = present_value_matrix(results.total_costs_matrix, cfg.discount)
    losses_pv = present_value_matrix(results.losses_matrix, cfg.discount)
    payouts_pv = present_value_matrix(results.total_payouts_matrix, cfg.discount)

    benefits_by_inst_pv = {}
    costs_by_inst_pv = {}
    payouts_by_inst_pv = {}
    for name in INSTRUMENT_NAMES:
        benefits_by_inst_pv[name] = present_value_matrix(
            results.benefits_by_instrument[name], cfg.discount
        )
        costs_by_inst_pv[name] = present_value_matrix(
            results.costs_by_instrument[name], cfg.discount
        )
        payouts_by_inst_pv[name] = present_value_matrix(
            results.payouts_by_instrument[name], cfg.discount
        )

    # ---- Core Indicators ----
    results.core = compute_core_indicators(
        benefits_pv=benefits_pv,
        costs_pv=costs_pv,
        losses_pv=losses_pv,
        payouts_pv=payouts_pv,
        benefits_by_instrument_pv=benefits_by_inst_pv,
        costs_by_instrument_pv=costs_by_inst_pv,
    )

    # ---- Efficiency Indicators ----
    results.efficiency = compute_efficiency_indicators(
        costs_by_instrument_pv=costs_by_inst_pv,
        payouts_by_instrument_pv=payouts_by_inst_pv,
        total_costs_pv=costs_pv,
        total_payouts_pv=payouts_pv,
        lambda_risk=cfg.omv.lambda_risk_adjustment,
    )

    return results


def run_drr_analysis(
    lec_data: LECData,
    cfg: LECCBAConfig,
    results_original: SimulationResults,
    reduced_catalog: np.ndarray
) -> SimulationResults:
    """
    Compute DRR cost-effectiveness.

    Compares losses between original and reduced catalogs to determine
    direct benefit (loss reduction) and indirect benefit (gap reduction).
    Does NOT re-run instrument analysis on the reduced catalog — instrument
    indicators are evaluated only on the original catalog.

    Args:
        lec_data: original LEC data
        cfg: configuration
        results_original: results from original analysis
        reduced_catalog: synthetic catalog with DRR-reduced losses
    """
    # PV of original losses and gaps (already computed)
    losses_pv_orig = present_value_matrix(
        results_original.losses_matrix, cfg.discount
    )
    unpaid_pv_orig = present_value_matrix(
        results_original.unpaid_losses_matrix, cfg.discount
    )

    # PV of reduced losses (no instrument re-run needed)
    losses_pv_red = present_value_matrix(reduced_catalog, cfg.discount)

    # For indirect benefit, compute unpaid losses on reduced catalog
    # using the SAME instrument payouts from the original analysis
    unpaid_red = np.maximum(0, reduced_catalog - results_original.total_payouts_matrix)
    unpaid_pv_red = present_value_matrix(unpaid_red, cfg.discount)

    drr_analysis = compute_drr_analysis(
        losses_pv_original=losses_pv_orig,
        losses_pv_reduced=losses_pv_red,
        unpaid_pv_original=unpaid_pv_orig,
        unpaid_pv_reduced=unpaid_pv_red,
        inv_vector=cfg.drr.inv,
        discount_rate=cfg.discount.social_discount_rate,
        cumulative_reduction=cfg.drr.cumulative_reduction(),
    )

    results_original.drr = drr_analysis
    return results_original


def run_sensitivity(
    lec_data: LECData,
    base_cfg: LECCBAConfig,
    verbose: bool = False
) -> SensitivityAnalysis:
    """
    Run sensitivity analysis varying key parameters one at a time.
    Includes DRR parameters if DRR is enabled.
    """
    sensitivity = SensitivityAnalysis()

    variations = [
        ("discount_rate", [0.03, 0.05, 0.07, 0.10]),
        ("indirect_benefit", [0.0, 0.05, 0.10, 0.20]),
        ("insurance_ROL", [0.03, 0.05, 0.07, 0.10]),
        ("credit_interest_ppo", [0.02, 0.035, 0.05, 0.07]),
    ]

    # Add DRR sensitivity parameters if enabled
    if base_cfg.drr.enabled:
        variations.extend([
            ("drr_rbc", [2, 4, 6, 8]),
            ("drr_hor", [10, 15, 20, 30]),
        ])

    base_result = run_analysis(lec_data, base_cfg)
    base_bc = base_result.core.expected_bc

    for param_name, values in variations:
        for val in values:
            cfg_copy = copy.deepcopy(base_cfg)

            if param_name == "discount_rate":
                base_val = cfg_copy.discount.social_discount_rate
                cfg_copy.discount.social_discount_rate = val
            elif param_name == "indirect_benefit":
                base_val = cfg_copy.indirect_benefit.factor
                cfg_copy.indirect_benefit.factor = val
            elif param_name == "insurance_ROL":
                base_val = cfg_copy.insurance.rate_on_line
                cfg_copy.insurance.rate_on_line = val
                cfg_copy.insurance.premium = None
                cfg_copy.insurance.__post_init__()
            elif param_name == "credit_interest_ppo":
                base_val = cfg_copy.ppo.interest_rate
                cfg_copy.ppo.interest_rate = val
            elif param_name == "drr_rbc":
                base_val = cfg_copy.drr.rbc[0]  # assume uniform
                cfg_copy.drr.rbc = [val] * len(cfg_copy.drr.rbc)
            elif param_name == "drr_hor":
                base_val = cfg_copy.drr.hor[0]
                cfg_copy.drr.hor = [val] * len(cfg_copy.drr.hor)
            else:
                continue

            result = run_analysis(lec_data, cfg_copy)

            if base_bc > 0 and base_val > 0:
                d_bc = result.core.expected_bc - base_bc
                d_param = val - base_val
                elasticity = (d_bc / base_bc) / (d_param / base_val) if d_param != 0 else 0
            else:
                elasticity = 0

            sr = SensitivityResult(
                parameter_name=param_name,
                parameter_value=val,
                base_value=base_val,
                expected_bc=result.core.expected_bc,
                prob_bc_gt_1=result.core.prob_bc_gt_1,
                expected_gap=result.core.expected_unpaid_loss,
                elasticity=elasticity,
            )
            sensitivity.add_result(sr)

            if verbose:
                print(f"  {param_name}={val:.3f} -> E[B/C]={result.core.expected_bc:.3f}")

    sensitivity.compute_tornado()
    return sensitivity


# ============================================================
# Main entry point
# ============================================================

if __name__ == "__main__":
    print("Loading LEC curve and generating synthetic catalog...")
    cfg = get_default_config()
    print(cfg.summary())

    lec = load_and_prepare(
        "LEC_curve_PER.csv",
        num_simulations=cfg.discount.num_simulations,
        horizon_years=cfg.discount.analysis_horizon,
        government_exposure_factor=cfg.government_exposure.factor,
    )
    print(f"\nFiscal AAL: ${lec.aal:.1f}M")
    print(f"Catalog shape: {lec.synthetic_annual_losses.shape}")

    print("\nRunning LEC-CBA analysis...")
    results = run_analysis(lec, cfg)

    print("\n" + "=" * 60)
    print("CORE INDICATORS")
    print("=" * 60)
    print(f"Expected B/C Ratio: {results.core.expected_bc:.3f}")
    print(f"P(B/C > 1): {results.core.prob_bc_gt_1:.1%}")
    print(f"Expected Unpaid Loss (PV): ${results.core.expected_unpaid_loss:.1f}M")
    print(f"Expected B/UL Ratio: {results.core.expected_bul:.3f}")

    print(f"\nB/C by Instrument:")
    for name, ratios in results.core.bc_ratios_by_instrument.items():
        finite = ratios[np.isfinite(ratios)]
        if len(finite) > 0:
            print(f"  {name}: E[B/C]={np.mean(finite):.3f}, "
                  f"P(B/C>1)={np.mean(ratios > 1.0):.1%}")

    print("\n" + "=" * 60)
    print("EFFICIENCY INDICATORS")
    print("=" * 60)
    print(f"Aggregate CM: {results.efficiency.aggregate_cost_multiple:.3f}")
    print(f"Aggregate MV: {results.efficiency.aggregate_money_value:.3f}")
    print(f"Aggregate OMV: {results.efficiency.aggregate_omv:.3f}")
    print(f"\nPer Instrument:")
    for name in INSTRUMENT_NAMES:
        cm = results.efficiency.cost_multiples.get(name, float('inf'))
        mv = results.efficiency.money_values.get(name, float('inf'))
        omv = results.efficiency.omv_values.get(name, float('inf'))
        print(f"  {name}: CM={cm:.3f}, MV={mv:.3f}, OMV={omv:.3f}")

    print("\nRunning sensitivity analysis...")
    sensitivity = run_sensitivity(lec, cfg, verbose=True)
    results.sensitivity = sensitivity

    print("\nTornado (B/C range by parameter):")
    for param, (low, high) in sensitivity.tornado_data.items():
        print(f"  {param}: [{low:.3f}, {high:.3f}]")
