"""
cba/costs.py — LEC Tool CBA Extension
========================================
Instrument cost functions operating on annual payout arrays already produced
by risk_management.apply_strategy.

All functions are pure: no file I/O, no matplotlib, no print statements.
Each function receives a 1-D numpy array of annual payouts (one value per year,
shape (horizon,)) and returns an annual cost array of the same shape.
"""

import numpy as np
from .config import InsuranceConfig, PPOConfig, CCFConfig, DDOConfig


def insurance_annual_costs(cfg: InsuranceConfig, horizon: int) -> np.ndarray:
    """
    Return flat annual premium vector, shape (horizon,).

    The premium is paid every year regardless of activation:
        premium = ROL × (exhaustion_point - attachment_point) × ceding_percentage

    If cfg.premium is already set (post __post_init__), use it directly.
    """
    return np.full(horizon, cfg.premium)


def ppo_annual_costs(cfg: PPOConfig, annual_payouts: np.ndarray) -> np.ndarray:
    """
    Compute PPO annual costs for one simulation.

    Parameters
    ----------
    cfg : PPOConfig
    annual_payouts : np.ndarray, shape (horizon,)
        Annual PPO payouts from apply_strategy. At most one non-zero entry
        (PPO activates at most once per catalogue).

    Returns
    -------
    np.ndarray, shape (horizon,)
        Annual cost vector.

    Cost structure (chronological within a simulation):
    - Year 0: front-end fee = credit_line × front_end_fee_rate  (one-time)
    - Every year: commitment fee = max(0, credit_line - outstanding_debt)
                                   × commitment_fee_rate
    - Year of activation and after: annuity debt service
          A = D × r(1+r)^n / ((1+r)^n - 1)
      where D = drawn amount, r = interest_rate, n = repayment_years.
      Debt reduces each year by (A - D×r). Stop when debt reaches 0.

    Sequencing within each year t:
      1. Compute annuity on outstanding debt (if any).
      2. Pay commitment fee on (credit_line - outstanding_debt).
      3. If annual_payouts[t] > 0: add payout to outstanding_debt,
         set remaining_years = repayment_years.
      4. Add front-end fee if t == 0.
    """
    horizon = len(annual_payouts)
    annual_costs = np.zeros(horizon)
    outstanding_debt = 0.0
    remaining_years = 0

    r = cfg.interest_rate

    for t in range(horizon):
        cost = 0.0

        # 1. Annuity on outstanding debt
        if outstanding_debt > 1e-10 and remaining_years > 0:
            n = remaining_years
            if r > 0:
                annuity = outstanding_debt * r * (1.0 + r)**n / ((1.0 + r)**n - 1.0)
            else:
                annuity = outstanding_debt / n
            cost += annuity
            principal_payment = annuity - outstanding_debt * r
            outstanding_debt -= principal_payment
            remaining_years -= 1
            if remaining_years <= 0 or outstanding_debt < 1e-10:
                outstanding_debt = 0.0
                remaining_years = 0

        # 2. Commitment fee on undrawn balance
        available = cfg.credit_line - outstanding_debt
        cost += max(0.0, available) * cfg.commitment_fee_rate

        # 3. Drawdown: payout adds to outstanding debt
        if annual_payouts[t] > 0:
            outstanding_debt += annual_payouts[t]
            remaining_years = cfg.repayment_years

        # 4. Front-end fee (year 0 only)
        if t == 0:
            cost += cfg.credit_line * cfg.front_end_fee_rate

        annual_costs[t] = cost

    return annual_costs


def ccf_annual_costs(cfg: CCFConfig, annual_payouts: np.ndarray) -> np.ndarray:
    """
    Compute CCF annual costs for one simulation.

    Parameters
    ----------
    cfg : CCFConfig
    annual_payouts : np.ndarray, shape (horizon,)
        Annual CCF payouts from apply_strategy. Multiple non-zero entries
        are possible (CCF is recurrent).

    Returns
    -------
    np.ndarray, shape (horizon,)
        Annual cost vector.

    Cost structure:
    - No commitment fee, no front-end fee.
    - On activation: drawdown_fee = payout × drawdown_fee_rate
    - After activation: debt service with grace period.
      During grace_period_years: pay interest only = D × interest_rate
      After grace period: pay full annuity
          A = D × r(1+r)^m / ((1+r)^m - 1)
      where m = repayment_years - grace_period_years.
    - Multiple draws accumulate independently. Each draw has its own
      grace period counter and repayment counter.
    - Track draws as a list of dicts:
      {'principal': float, 'grace_remaining': int, 'repay_remaining': int}

    Sequencing within each year t:
      1. For each existing draw: if grace_remaining > 0, charge interest only
         and decrement grace_remaining. Else charge annuity and reduce principal.
         Remove draws with repay_remaining == 0.
      2. If annual_payouts[t] > 0: add drawdown_fee to cost,
         append new draw to list.
    """
    horizon = len(annual_payouts)
    annual_costs = np.zeros(horizon)

    r = cfg.interest_rate
    # Repayment years after the grace period ends
    m = cfg.repayment_years - cfg.grace_period_years

    draws = []  # list of {'principal': float, 'grace_remaining': int, 'repay_remaining': int}

    for t in range(horizon):
        cost = 0.0

        # 1. Service existing draws
        active_draws = []
        for draw in draws:
            if draw['grace_remaining'] > 0:
                # Interest-only payment during grace period
                cost += draw['principal'] * r
                draw['grace_remaining'] -= 1
                active_draws.append(draw)
            elif draw['repay_remaining'] > 0:
                # Full annuity after grace period
                D = draw['principal']
                n = draw['repay_remaining']
                if r > 0 and n > 0:
                    annuity = D * r * (1.0 + r)**n / ((1.0 + r)**n - 1.0)
                else:
                    annuity = D / max(n, 1)
                cost += annuity
                principal_payment = annuity - D * r
                draw['principal'] -= principal_payment
                draw['repay_remaining'] -= 1
                if draw['repay_remaining'] > 0 and draw['principal'] > 1e-10:
                    active_draws.append(draw)
            # else: fully repaid, drop from list
        draws = active_draws

        # 2. New drawdown this year
        if annual_payouts[t] > 0:
            cost += annual_payouts[t] * cfg.drawdown_fee_rate
            draws.append({
                'principal': annual_payouts[t],
                'grace_remaining': cfg.grace_period_years,
                'repay_remaining': max(m, 1),
            })

        annual_costs[t] = cost

    return annual_costs


def ddo_annual_costs(cfg: DDOConfig, annual_payouts: np.ndarray) -> np.ndarray:
    """
    Compute DDO annual costs for one simulation.

    DDO triggers on every event above its threshold (recurrent, like CCF).
    Cost structure: interest on drawn amount, no commitment fee, no drawdown fee.
    Each activation creates an independent debt with its own repayment schedule.

    Same debt-tracking logic as ccf_annual_costs but without grace period
    and without drawdown fee. Use cfg.interest_rate and cfg.repayment_years.
    """
    horizon = len(annual_payouts)
    annual_costs = np.zeros(horizon)

    r = cfg.interest_rate
    draws = []  # list of {'principal': float, 'repay_remaining': int}

    for t in range(horizon):
        cost = 0.0

        # Service existing draws (no grace period)
        active_draws = []
        for draw in draws:
            D = draw['principal']
            n = draw['repay_remaining']
            if n > 0 and D > 1e-10:
                if r > 0:
                    annuity = D * r * (1.0 + r)**n / ((1.0 + r)**n - 1.0)
                else:
                    annuity = D / n
                cost += annuity
                principal_payment = annuity - D * r
                draw['principal'] -= principal_payment
                draw['repay_remaining'] -= 1
                if draw['repay_remaining'] > 0 and draw['principal'] > 1e-10:
                    active_draws.append(draw)
        draws = active_draws

        # New drawdown this year (no drawdown fee)
        if annual_payouts[t] > 0:
            draws.append({
                'principal': annual_payouts[t],
                'repay_remaining': cfg.repayment_years,
            })

        annual_costs[t] = cost

    return annual_costs
