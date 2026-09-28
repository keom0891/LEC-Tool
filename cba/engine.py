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

    # Resolved per-instrument configs (built once from drm_configs + cba_config)
    resolved_instrument_configs: Dict[str, object] = field(default_factory=dict)
    # maps instrument name → type string ('insurance', 'ppo', 'ccf', 'ddo')
    instrument_types: Dict[str, str] = field(default_factory=dict)
    # fiscal responsibility share passed from main.py (for reporting only)
    resp_fiscal: float = 1.0
    # 'total' or 'fiscal': what losses_df contained (set by the caller)
    loss_basis: str = 'total'

    # Optional fields (not computed by run_cba, available for downstream use)
    drr: Optional[object] = None
    sensitivity: Optional[object] = None
    # CNC (Comparative Net Cost) — populated only if cba_config.cnc.enabled
    cnc: Optional[object] = None
    # v8 — one entry per insurance instrument: economic and fiscal B/C,
    # implied pricing multiple and protection indicators (see
    # _insurance_analysis). Populated by run_cba.
    insurance_analysis: Dict[str, dict] = field(default_factory=dict)


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


def _build_instrument_config(drm_cfg: dict, cba_config: LECCBAConfig):
    """
    Build a typed cost-config object for one instrument, merging
    per-instrument overrides from drm_cfg with cba_config defaults.

    Cost parameters recognised per instrument type:

    'insurance': rate_on_line, attachment_point, exhaustion_point,
                 ceding_percentage, premium
    'ppo':       commitment_fee_rate, loan_interest_rate, repayment_years,
                 front_end_fee_rate, credit_line (auto from ppo_schedule if absent)
    'ccf':       drawdown_fee_rate, loan_interest_rate, repayment_years,
                 grace_period_years
    'ddo':       loan_interest_rate, repayment_years

    NOTE (v6 breaking change): the loan interest rate key is 'loan_interest_rate',
    renamed from the former 'interest_rate' to make explicit that it is the
    contractual rate on the loan, distinct from the social discount rate
    (cba_config.discount.social_discount_rate). Configs still passing the old
    'interest_rate' key raise a ValueError so the mistake cannot go unnoticed.

    All unrecognised keys in drm_cfg are silently ignored.
    """
    from cba.config import InsuranceConfig, PPOConfig, CCFConfig, DDOConfig
    import copy

    itype = drm_cfg['type']

    # v6 renamed 'interest_rate' -> 'loan_interest_rate'. Stop here rather than
    # silently falling back to the default rate for a loan-type instrument.
    if itype in ('ppo', 'ccf', 'ddo') and 'interest_rate' in drm_cfg:
        raise ValueError(
            f"Instrument '{drm_cfg.get('name', itype)}': the key 'interest_rate' "
            f"is no longer supported. Use 'loan_interest_rate' (contractual loan "
            f"rate); the social discount rate is configured separately in [cba]."
        )

    if itype == 'insurance':
        base = copy.copy(cba_config.insurance)
        # Override payout params (must match what risk_management uses)
        for attr in ('attachment_point', 'exhaustion_point', 'ceding_percentage'):
            if attr in drm_cfg:
                setattr(base, attr, float(drm_cfg[attr]))
        # Override cost params
        for attr in ('rate_on_line', 'premium'):
            if attr in drm_cfg:
                setattr(base, attr, float(drm_cfg[attr]))
        # Recompute premium if not explicitly set
        if 'premium' not in drm_cfg:
            coverage = (base.exhaustion_point - base.attachment_point) * base.ceding_percentage
            base.premium = base.rate_on_line * coverage
        # v8: donor discount, pricing method and payout rule (set by
        # insurance_layer.resolve_instruments; informational for the costs).
        base.donor_discount = float(drm_cfg.get('donor_discount', 0.0))
        base.pricing_method = str(drm_cfg.get('pricing', 'fixed_rol'))
        if 'payout_mode' in drm_cfg:
            base.payout_mode = str(drm_cfg['payout_mode'])
        if 'one_payout_per_year' in drm_cfg:
            base.one_payout_per_year = bool(drm_cfg['one_payout_per_year'])
        if not 0.0 <= base.donor_discount <= base.premium:
            raise ValueError(
                f"Insurance '{drm_cfg.get('name', 'insurance')}': donor_discount "
                f"({base.donor_discount}) must be between 0 and the premium ({base.premium})."
            )
        return base

    elif itype == 'ppo':
        base = copy.copy(cba_config.ppo)
        for attr in ('commitment_fee_rate', 'loan_interest_rate', 'front_end_fee_rate'):
            if attr in drm_cfg:
                setattr(base, attr, float(drm_cfg[attr]))
        if 'repayment_years' in drm_cfg:
            base.repayment_years = int(drm_cfg['repayment_years'])
        if 'grace_period_years' in drm_cfg:
            base.grace_period_years = float(drm_cfg['grace_period_years'])
        if base.grace_period_years >= base.repayment_years:
            raise ValueError(
                f"PPO '{drm_cfg.get('name', 'ppo')}': grace_period_years "
                f"({base.grace_period_years}) must be strictly less than "
                f"repayment_years ({base.repayment_years})."
            )
        # credit_line: explicit override or derive from ppo_schedule
        if 'credit_line' in drm_cfg:
            base.credit_line = float(drm_cfg['credit_line'])
        elif 'ppo_schedule' in drm_cfg and drm_cfg['ppo_schedule'] is not None:
            base.credit_line = float(max(drm_cfg['ppo_schedule']))
        return base

    elif itype == 'ccf':
        base = copy.copy(cba_config.ccf)
        for attr in ('drawdown_fee_rate', 'loan_interest_rate'):
            if attr in drm_cfg:
                setattr(base, attr, float(drm_cfg[attr]))
        for attr in ('repayment_years', 'grace_period_years'):
            if attr in drm_cfg:
                setattr(base, attr, int(drm_cfg[attr]))
        # Re-validate grace vs repayment after overrides. copy.copy does not
        # re-run __post_init__, so an invalid combination passed via drm_cfg
        # would otherwise slip through. Same rule as CCFConfig.__post_init__.
        if base.grace_period_years >= base.repayment_years:
            raise ValueError(
                f"CCF '{drm_cfg.get('name', 'ccf')}': grace_period_years "
                f"({base.grace_period_years}) must be strictly less than "
                f"repayment_years ({base.repayment_years}). With grace >= "
                f"repayment the principal would never be repaid."
            )
        return base

    elif itype == 'ddo':
        base = copy.copy(cba_config.ddo)
        if 'loan_interest_rate' in drm_cfg:
            base.loan_interest_rate = float(drm_cfg['loan_interest_rate'])
        if 'repayment_years' in drm_cfg:
            base.repayment_years = int(drm_cfg['repayment_years'])
        if 'ddo_threshold' in drm_cfg:
            base.ddo_threshold = float(drm_cfg['ddo_threshold'])
        if 'ddo_available' in drm_cfg:
            base.ddo_available = float(drm_cfg['ddo_available'])
        if 'grace_period_years' in drm_cfg:
            base.grace_period_years = float(drm_cfg['grace_period_years'])
        if base.grace_period_years >= base.repayment_years:
            raise ValueError(
                f"DDO '{drm_cfg.get('name', 'ddo')}': grace_period_years "
                f"({base.grace_period_years}) must be strictly less than "
                f"repayment_years ({base.repayment_years})."
            )
        return base

    else:
        raise ValueError(f"Unknown instrument type '{itype}'")


def run_cba(
    losses_df: pd.DataFrame,
    payout_dfs: list,
    drm_configs: list,
    cba_config: LECCBAConfig,
    resp_fiscal: float = 1.0,
    gap_thresholds: Optional[List[float]] = None,
    loss_basis: str = 'total',
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
    resp_fiscal : float, default 1.0
        Fiscal responsibility share from main.py, stored for reporting only.
    gap_thresholds : list of float, optional
        Absolute thresholds ($MM) for P(unpaid loss PV > threshold). main.py
        passes the same thresholds used by the main report (fractions of the
        maximum LEC loss); the default falls back to cba.core.
    loss_basis : {'total', 'fiscal'}
        Whether losses_df holds total economic losses or losses already
        multiplied by resp_fiscal. Stored for reporting only; the caller
        applies the scaling.

    Returns
    -------
    CBAResults
    """
    # losses_df is used as-is. The choice between total and fiscal losses is
    # made by the caller through config.toml ([cba] loss_basis) and recorded
    # here for the report.
    losses_matrix = losses_df.values.astype(float)
    num_sims, horizon = losses_matrix.shape

    results = CBAResults(config=cba_config)
    results.losses_matrix = losses_matrix
    results.resp_fiscal = resp_fiscal
    results.loss_basis = loss_basis

    # Initialize output matrices
    results.total_costs_matrix = np.zeros((num_sims, horizon))
    results.total_benefits_matrix = np.zeros((num_sims, horizon))
    results.total_payouts_matrix = np.zeros((num_sims, horizon))
    results.unpaid_losses_matrix = np.zeros((num_sims, horizon))

    # Each instrument gets its own key (name), so two DDOs with different names
    # are tracked separately rather than collapsed into one type bucket.
    instrument_names = [
        cfg.get('name', f"{cfg['type']}_{j}") for j, cfg in enumerate(drm_configs)
    ]
    for name in instrument_names:
        results.costs_by_instrument[name] = np.zeros((num_sims, horizon))
        results.payouts_by_instrument[name] = np.zeros((num_sims, horizon))
        results.benefits_by_instrument[name] = np.zeros((num_sims, horizon))

    # Build per-instrument configs once (not per simulation)
    instrument_configs = {}
    for j, drm_cfg in enumerate(drm_configs):
        inst_name = instrument_names[j]
        instrument_configs[inst_name] = _build_instrument_config(drm_cfg, cba_config)

    results.resolved_instrument_configs = instrument_configs
    results.instrument_types = {
        instrument_names[j]: drm_cfg['type']
        for j, drm_cfg in enumerate(drm_configs)
    }

    # --- Per-simulation loop ---
    for s in range(num_sims):
        total_payout_s = np.zeros(horizon)

        for j, drm_cfg in enumerate(drm_configs):
            itype = drm_cfg['type']
            inst_name = instrument_names[j]
            annual_payouts = payout_dfs[j].iloc[s].values.astype(float)

            cost_fn = _get_cost_function(itype)
            inst_cfg = instrument_configs[inst_name]

            # Compute annual costs
            if itype == 'insurance':
                annual_costs = cost_fn(inst_cfg, horizon)
            else:
                # Debt-type instruments (PPO, CCF, DDO) book the full present
                # value of debt service at disbursement (v6). They need the
                # social discount rate and the legacy-truncation flag.
                annual_costs = cost_fn(
                    inst_cfg,
                    annual_payouts,
                    social_rate=cba_config.discount.social_discount_rate,
                    legacy_truncation=getattr(cba_config, 'legacy_truncation', False),
                )

            # Benefits = payouts × (1 + indirect_factor)
            annual_benefits = annual_payouts * (1.0 + cba_config.indirect_benefit.factor)

            # Each instrument has its own key — no += accumulation needed here
            results.costs_by_instrument[inst_name][s] = annual_costs
            results.payouts_by_instrument[inst_name][s] = annual_payouts
            results.benefits_by_instrument[inst_name][s] = annual_benefits

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
    for name in instrument_names:
        benefits_by_inst_pv[name] = present_value_matrix(
            results.benefits_by_instrument[name], cba_config.discount
        )
        costs_by_inst_pv[name] = present_value_matrix(
            results.costs_by_instrument[name], cba_config.discount
        )
        payouts_by_inst_pv[name] = present_value_matrix(
            results.payouts_by_instrument[name], cba_config.discount
        )

    # --- Indicators ---
    results.core = compute_core_indicators(
        benefits_pv=benefits_pv,
        costs_pv=costs_pv,
        losses_pv=losses_pv,
        payouts_pv=payouts_pv,
        benefits_by_instrument_pv=benefits_by_inst_pv,
        costs_by_instrument_pv=costs_by_inst_pv,
        gap_thresholds=gap_thresholds,
    )

    results.efficiency = compute_efficiency_indicators(
        costs_by_instrument_pv=costs_by_inst_pv,
        payouts_by_instrument_pv=payouts_by_inst_pv,
        total_costs_pv=costs_pv,
        total_payouts_pv=payouts_pv,
        lambda_risk=cba_config.omv.lambda_risk_adjustment,
    )

    # --- v8: insurance analysis (economic / fiscal B/C, pricing, protection) ---
    for name in instrument_names:
        if results.instrument_types.get(name) == 'insurance':
            results.insurance_analysis[name] = _insurance_analysis(
                name, results, cba_config, costs_by_inst_pv[name]
            )

    # --- CNC (Comparative Net Cost) — optional, modular ---
    # Runs only if the cnc config is present and enabled. Strategy A cost is the
    # total ex-ante instruments cost (costs_pv); Strategy B is ex-post commercial
    # debt on the full loss. Fully removable: if cba_config.cnc is None or the
    # module is gone, this block is skipped and results.cnc stays None.
    cnc_cfg = getattr(cba_config, 'cnc', None)
    if cnc_cfg is not None and getattr(cnc_cfg, 'enabled', False):
        try:
            from cba.cnc import compute_cnc
            results.cnc = compute_cnc(
                losses_matrix=losses_matrix,
                payouts_matrix=results.total_payouts_matrix,
                cost_a_pv=costs_pv,
                cnc_cfg=cnc_cfg,
                social_rate=cba_config.discount.social_discount_rate,
            )
        except ImportError:
            results.cnc = None

    return results


def _insurance_analysis(name, results, cba_config, costs_pv_inst) -> dict:
    """
    Indicators reported for an insurance instrument (v8).

    economic B/C   benefits over the GROSS premium (the true cost of the cover).
    fiscal B/C     benefits over the premium the government pays, net of the
                   donor discount. Differs from the economic ratio only when
                   a donor pays part of the premium.
    multiple       gross premium per unit of expected annual payout. A price
                   below 1.0 would sell cover below its expected loss; market
                   prices for catastrophe layers sit well above 1.
    protection     how much the policy lowers the unpaid loss in bad outcomes:
                   the 95th and 99th percentiles of the unpaid loss (present
                   value over the horizon) with and without this policy.

    For insurance, a benefit-cost ratio below 1 is the normal outcome of a
    fairly priced policy: the premium exceeds the expected payout by the
    insurer's cost of holding capital. The policy is bought for protection in
    the bad years, which the protection indicators measure.
    """
    from cba.discounting import present_value_matrix

    cfg = results.resolved_instrument_configs[name]
    payouts = results.payouts_by_instrument[name]
    gross = float(cfg.premium)
    net = float(cfg.premium - getattr(cfg, 'donor_discount', 0.0))

    econ = results.core.bc_ratios_by_instrument[name]
    fiscal = econ * (gross / net) if net > 0 else np.full_like(econ, np.inf)

    def _mean_finite(x):
        f = x[np.isfinite(x)]
        return float(np.mean(f)) if f.size else float('inf')

    expected_payout = float(np.mean(payouts))           # per year
    multiple = gross / expected_payout if expected_payout > 0 else float('inf')

    others = results.total_payouts_matrix - payouts
    unpaid_with = present_value_matrix(results.unpaid_losses_matrix, cba_config.discount)
    unpaid_without = present_value_matrix(
        np.maximum(0.0, results.losses_matrix - others), cba_config.discount
    )
    paying_years = payouts[payouts > 0]
    paid_mask = payouts > 0
    with np.errstate(divide='ignore', invalid='ignore'):
        share = payouts[paid_mask] / results.losses_matrix[paid_mask]
    share = share[np.isfinite(share)]

    return {
        'gross_premium': gross,
        'net_premium': net,
        'donor_discount': gross - net,
        'pricing_method': getattr(cfg, 'pricing_method', 'fixed_rol'),
        'payout_mode': getattr(cfg, 'payout_mode', 'proportional'),
        'one_payout_per_year': getattr(cfg, 'one_payout_per_year', False),
        'coverage_limit': (cfg.exhaustion_point - cfg.attachment_point) * cfg.ceding_percentage,
        'expected_annual_payout': expected_payout,
        'implied_multiple': multiple,
        'bc_economic': _mean_finite(econ),
        'bc_fiscal': _mean_finite(fiscal),
        'prob_bc_economic_gt_1': float(np.mean(econ > 1.0)),
        'prob_pays_in_horizon': float(np.mean((payouts > 0).any(axis=1))),
        'mean_payout_when_paid': float(np.mean(paying_years)) if paying_years.size else 0.0,
        'mean_share_of_loss_when_paid': float(np.mean(share)) if share.size else 0.0,
        'unpaid_p95_without': float(np.percentile(unpaid_without, 95)),
        'unpaid_p95_with': float(np.percentile(unpaid_with, 95)),
        'unpaid_p99_without': float(np.percentile(unpaid_without, 99)),
        'unpaid_p99_with': float(np.percentile(unpaid_with, 99)),
        'premium_pv_gross': float(np.mean(costs_pv_inst)),
    }
