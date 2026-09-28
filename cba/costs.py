"""
cba/costs.py — LEC Tool CBA Extension
========================================
Instrument cost functions operating on annual payout arrays already produced
by risk_management.apply_strategy.

All functions are pure: no file I/O, no matplotlib, no print statements.
Each function receives a 1-D numpy array of annual payouts (one value per year,
shape (horizon,)) and returns an annual cost array of the same shape.

DEBT SERVICE ACCOUNTING (v6 change)
-----------------------------------
Loan-type instruments (PPO, CCF, DDO) book the FULL present value of the debt
service stream at the moment of disbursement, instead of spreading the annuity
year by year across the analysis horizon.

Rationale: when a loan is disbursed late in the horizon and its repayment term
extends beyond it (e.g. CCF with a 25-year term inside a 10-year horizon), the
year-by-year approach silently drops every annuity payment that falls outside
the horizon, severely understating the loan's true cost. The full-cost-at-
disbursement approach is the standard treatment in the World Bank's framework
for evaluating sovereign disaster risk finance (Clarke, Mahul, Poulter & Teh,
2017, The Geneva Papers on Risk and Insurance).

The present value (at the social discount rate, measured at the disbursement
year t) of a loan of principal D, contractual rate e, term n, is:

    PV_at_t = D * (a_{n|i} / a_{n|e})

where a_{n|g} = (1 - (1+g)^(-n)) / g is the ordinary annuity factor.
engine.py then discounts this value from year t back to year 0 with the social
rate, completing the present-value calculation.

A `legacy_truncation` flag (default False) reproduces the old year-by-year
behaviour for cross-validation against the BID Colab.
"""

import numpy as np
from .config import InsuranceConfig, PPOConfig, CCFConfig, DDOConfig


# ============================================================
# Annuity helpers
# ============================================================

def annuity_factor(rate: float, n: int) -> float:
    """
    Ordinary annuity factor a_{n|rate} = (1 - (1+rate)^(-n)) / rate.

    Converts a periodic payment into a present value (immediate annuity,
    payments in arrears). For rate == 0 it degenerates to n.
    """
    if n <= 0:
        return 0.0
    if abs(rate) < 1e-12:
        return float(n)
    return (1.0 - (1.0 + rate) ** (-n)) / rate


def loan_pv_at_disbursement(
    principal: float,
    loan_rate: float,
    social_rate: float,
    term_years: int,
    grace_years: int = 0,
) -> float:
    """
    Present value of a loan's full debt-service stream, measured at the year
    of disbursement, discounted at the social rate.

    The annuity that the borrower pays is sized with the contractual loan rate;
    the resulting payment stream is then valued at the social discount rate.

    Without grace:
        annuity A = principal / a_{term|loan_rate}
        PV_at_t   = A * a_{term|social_rate}
                  = principal * a_{term|social} / a_{term|loan}

    With grace (interest-only during grace, then amortise over term-grace):
        - During grace (g years): interest-only payment = principal * loan_rate
          valued at social rate over g years -> principal*loan_rate*a_{g|social}
        - After grace: amortise principal over (term-g) years.
          Annuity A = principal / a_{(term-g)|loan_rate}
          Its PV at the start of amortisation = A * a_{(term-g)|social}
          Discounted back g years to disbursement: * (1+social)^(-g)

    Parameters
    ----------
    principal : float
        Disbursed amount D (M USD).
    loan_rate : float
        Contractual interest rate on the loan.
    social_rate : float
        Social discount rate used for present value.
    term_years : int
        Total repayment term n (including grace).
    grace_years : int
        Interest-only grace years g (0 for no grace).

    Returns
    -------
    float
        Present value of the debt-service stream at the disbursement year.

    Raises
    ------
    ValueError
        If any input is non-finite (NaN/inf), if principal or term is negative,
        or if the rates fall outside an economically sensible range. These are
        absurd inputs that must stop the run rather than produce a false number.
    """
    # --- Defensive validation: stop on absurd inputs, don't return garbage ---
    if not np.isfinite(principal):
        raise ValueError(
            f"loan_pv_at_disbursement: principal must be finite, got {principal}. "
            f"A NaN or infinite disbursement would silently contaminate the analysis."
        )
    if principal < 0:
        raise ValueError(
            f"loan_pv_at_disbursement: principal must be >= 0, got {principal}. "
            f"A negative disbursement has no meaning."
        )
    if not np.isfinite(loan_rate):
        raise ValueError(
            f"loan_pv_at_disbursement: loan_rate must be finite, got {loan_rate}."
        )
    if not (0.0 <= loan_rate <= 1.0):
        raise ValueError(
            f"loan_rate must be in [0, 1] (0%–100%), got {loan_rate} "
            f"({loan_rate:.0%}). Rates outside this range are not credible for a "
            f"sovereign loan; check the units (decimal, not percent)."
        )
    if not np.isfinite(social_rate):
        raise ValueError(
            f"loan_pv_at_disbursement: social_rate must be finite, got {social_rate}."
        )
    if not (0.0 <= social_rate <= 1.0):
        raise ValueError(
            f"social_rate must be in [0, 1] (0%–100%), got {social_rate} "
            f"({social_rate:.0%}). A social discount rate outside this range is "
            f"not sensible; check the units (decimal, not percent)."
        )
    if term_years < 0:
        raise ValueError(
            f"loan_pv_at_disbursement: term_years must be >= 0, got {term_years}. "
            f"A negative repayment term has no meaning."
        )
    if grace_years < 0:
        raise ValueError(
            f"loan_pv_at_disbursement: grace_years must be >= 0, got {grace_years}."
        )

    # Valid corner: no disbursement or no term => no debt service.
    if principal == 0 or term_years == 0:
        return 0.0

    g = max(0, int(grace_years))
    if g >= term_years:
        # Degenerate: all grace, no amortisation window. Treat as interest-only
        # over the full term (should not happen with sane configs).
        return principal * loan_rate * annuity_factor(social_rate, term_years)

    if g == 0:
        a_term_loan = annuity_factor(loan_rate, term_years)
        a_term_social = annuity_factor(social_rate, term_years)
        if a_term_loan <= 0:
            return 0.0
        return principal * (a_term_social / a_term_loan)

    # With grace period
    amort_years = term_years - g
    # Interest-only phase valued at social rate
    pv_grace = principal * loan_rate * annuity_factor(social_rate, g)
    # Amortisation phase
    a_amort_loan = annuity_factor(loan_rate, amort_years)
    a_amort_social = annuity_factor(social_rate, amort_years)
    if a_amort_loan <= 0:
        return pv_grace
    annuity = principal / a_amort_loan
    pv_amort_at_grace_end = annuity * a_amort_social
    pv_amort = pv_amort_at_grace_end * (1.0 + social_rate) ** (-g)
    return pv_grace + pv_amort


# ============================================================
# CCRIF — Parametric Insurance
# ============================================================

def insurance_annual_costs(cfg: InsuranceConfig, horizon: int) -> np.ndarray:
    """
    Return flat annual premium vector, shape (horizon,).

    The premium is paid every year regardless of activation:
        premium = ROL × (exhaustion_point - attachment_point) × ceding_percentage

    Insurance has no debt service, so the v6 change does not affect it.
    """
    return np.full(horizon, cfg.premium)


# ============================================================
# PPO — single-activation contingent credit
# ============================================================

def ppo_annual_costs(
    cfg: PPOConfig,
    annual_payouts: np.ndarray,
    social_rate: float = 0.05,
    legacy_truncation: bool = False,
) -> np.ndarray:
    """
    Compute PPO annual costs for one simulation.

    Parameters
    ----------
    cfg : PPOConfig
    annual_payouts : np.ndarray, shape (horizon,)
        Annual PPO payouts from apply_strategy. At most one non-zero entry.
    social_rate : float
        Social discount rate, needed for full-cost-at-disbursement.
    legacy_truncation : bool
        If True, reproduce the old year-by-year annuity (truncated at horizon).
        If False (default), book full debt-service PV at disbursement.

    Cost components:
    - Year 0: front-end fee = credit_line × front_end_fee_rate (one-time)
    - Every year: commitment fee = max(0, credit_line - outstanding) × rate
    - Debt service on the drawn amount.

    For commitment-fee purposes, the drawn principal is treated as outstanding
    from the activation year to the end of the horizon (the line is consumed).
    """
    horizon = len(annual_payouts)
    annual_costs = np.zeros(horizon)
    r = cfg.loan_interest_rate

    if legacy_truncation:
        return _ppo_annual_costs_legacy(cfg, annual_payouts)

    # Track drawn principal only to reduce the undrawn balance for commitment fee
    outstanding_for_fee = 0.0

    for t in range(horizon):
        cost = 0.0

        # Commitment fee on undrawn balance (before any new draw this year)
        available = cfg.credit_line - outstanding_for_fee
        cost += max(0.0, available) * cfg.commitment_fee_rate

        # Drawdown: book full debt-service PV at this disbursement year
        if annual_payouts[t] > 0:
            cost += loan_pv_at_disbursement(
                principal=annual_payouts[t],
                loan_rate=r,
                social_rate=social_rate,
                term_years=cfg.repayment_years,
                grace_years=getattr(cfg, "grace_period_years", 0.0),
            )
            outstanding_for_fee += annual_payouts[t]

        # Front-end fee (year 0 only)
        if t == 0:
            cost += cfg.credit_line * cfg.front_end_fee_rate

        annual_costs[t] = cost

    return annual_costs


def _ppo_annual_costs_legacy(cfg: PPOConfig, annual_payouts: np.ndarray) -> np.ndarray:
    """Legacy year-by-year PPO cost (truncated at horizon). For validation only."""
    horizon = len(annual_payouts)
    annual_costs = np.zeros(horizon)
    outstanding_debt = 0.0
    remaining_years = 0
    r = cfg.loan_interest_rate

    for t in range(horizon):
        cost = 0.0
        if outstanding_debt > 1e-10 and remaining_years > 0:
            n = remaining_years
            if r > 0:
                annuity = outstanding_debt * r * (1.0 + r) ** n / ((1.0 + r) ** n - 1.0)
            else:
                annuity = outstanding_debt / n
            cost += annuity
            principal_payment = annuity - outstanding_debt * r
            outstanding_debt -= principal_payment
            remaining_years -= 1
            if remaining_years <= 0 or outstanding_debt < 1e-10:
                outstanding_debt = 0.0
                remaining_years = 0
        available = cfg.credit_line - outstanding_debt
        cost += max(0.0, available) * cfg.commitment_fee_rate
        if annual_payouts[t] > 0:
            outstanding_debt += annual_payouts[t]
            remaining_years = cfg.repayment_years
        if t == 0:
            cost += cfg.credit_line * cfg.front_end_fee_rate
        annual_costs[t] = cost

    return annual_costs


# ============================================================
# CCF — recurrent contingent credit, with grace period
# ============================================================

def ccf_annual_costs(
    cfg: CCFConfig,
    annual_payouts: np.ndarray,
    social_rate: float = 0.05,
    legacy_truncation: bool = False,
) -> np.ndarray:
    """
    Compute CCF annual costs for one simulation.

    Parameters
    ----------
    cfg : CCFConfig
    annual_payouts : np.ndarray, shape (horizon,)
        Annual CCF payouts (multiple non-zero entries possible; recurrent).
    social_rate : float
        Social discount rate.
    legacy_truncation : bool
        If True, reproduce old year-by-year debt service.
        If False (default), book full debt-service PV at each disbursement,
        including the interest-only grace period.

    Cost components:
    - No commitment fee, no front-end fee.
    - On activation: drawdown_fee = payout × drawdown_fee_rate
    - Debt service: interest-only during grace_period_years, then amortise
      over (repayment_years - grace_period_years).
    """
    horizon = len(annual_payouts)
    annual_costs = np.zeros(horizon)

    if legacy_truncation:
        return _ccf_annual_costs_legacy(cfg, annual_payouts)

    for t in range(horizon):
        cost = 0.0
        if annual_payouts[t] > 0:
            # Drawdown fee at disbursement
            cost += annual_payouts[t] * cfg.drawdown_fee_rate
            # Full debt-service PV at disbursement (with grace)
            cost += loan_pv_at_disbursement(
                principal=annual_payouts[t],
                loan_rate=cfg.loan_interest_rate,
                social_rate=social_rate,
                term_years=cfg.repayment_years,
                grace_years=cfg.grace_period_years,
            )
        annual_costs[t] = cost

    return annual_costs


def _ccf_annual_costs_legacy(cfg: CCFConfig, annual_payouts: np.ndarray) -> np.ndarray:
    """Legacy year-by-year CCF cost (truncated at horizon). For validation only."""
    horizon = len(annual_payouts)
    annual_costs = np.zeros(horizon)
    r = cfg.loan_interest_rate
    m = cfg.repayment_years - cfg.grace_period_years
    draws = []

    for t in range(horizon):
        cost = 0.0
        active_draws = []
        for draw in draws:
            if draw['grace_remaining'] > 0:
                cost += draw['principal'] * r
                draw['grace_remaining'] -= 1
                active_draws.append(draw)
            elif draw['repay_remaining'] > 0:
                D = draw['principal']
                n = draw['repay_remaining']
                if r > 0 and n > 0:
                    annuity = D * r * (1.0 + r) ** n / ((1.0 + r) ** n - 1.0)
                else:
                    annuity = D / max(n, 1)
                cost += annuity
                principal_payment = annuity - D * r
                draw['principal'] -= principal_payment
                draw['repay_remaining'] -= 1
                if draw['repay_remaining'] > 0 and draw['principal'] > 1e-10:
                    active_draws.append(draw)
        draws = active_draws
        if annual_payouts[t] > 0:
            cost += annual_payouts[t] * cfg.drawdown_fee_rate
            draws.append({
                'principal': annual_payouts[t],
                'grace_remaining': cfg.grace_period_years,
                'repay_remaining': max(m, 1),
            })
        annual_costs[t] = cost

    return annual_costs


# ============================================================
# DDO — recurrent deferred drawdown option (no grace, no drawdown fee)
# ============================================================

def ddo_annual_costs(
    cfg: DDOConfig,
    annual_payouts: np.ndarray,
    social_rate: float = 0.05,
    legacy_truncation: bool = False,
) -> np.ndarray:
    """
    Compute DDO annual costs for one simulation.

    DDO triggers on every event above its threshold (recurrent, like CCF).
    Cost: interest on drawn amount, no commitment fee, no drawdown fee,
    no grace period. Each activation books full debt-service PV at disbursement.
    """
    horizon = len(annual_payouts)
    annual_costs = np.zeros(horizon)

    if legacy_truncation:
        return _ddo_annual_costs_legacy(cfg, annual_payouts)

    for t in range(horizon):
        cost = 0.0
        if annual_payouts[t] > 0:
            cost += loan_pv_at_disbursement(
                principal=annual_payouts[t],
                loan_rate=cfg.loan_interest_rate,
                social_rate=social_rate,
                term_years=cfg.repayment_years,
                grace_years=getattr(cfg, "grace_period_years", 0.0),
            )
        annual_costs[t] = cost

    return annual_costs


def _ddo_annual_costs_legacy(cfg: DDOConfig, annual_payouts: np.ndarray) -> np.ndarray:
    """Legacy year-by-year DDO cost (truncated at horizon). For validation only."""
    horizon = len(annual_payouts)
    annual_costs = np.zeros(horizon)
    r = cfg.loan_interest_rate
    draws = []

    for t in range(horizon):
        cost = 0.0
        active_draws = []
        for draw in draws:
            D = draw['principal']
            n = draw['repay_remaining']
            if n > 0 and D > 1e-10:
                if r > 0:
                    annuity = D * r * (1.0 + r) ** n / ((1.0 + r) ** n - 1.0)
                else:
                    annuity = D / n
                cost += annuity
                principal_payment = annuity - D * r
                draw['principal'] -= principal_payment
                draw['repay_remaining'] -= 1
                if draw['repay_remaining'] > 0 and draw['principal'] > 1e-10:
                    active_draws.append(draw)
        draws = active_draws
        if annual_payouts[t] > 0:
            draws.append({
                'principal': annual_payouts[t],
                'repay_remaining': cfg.repayment_years,
            })
        annual_costs[t] = cost

    return annual_costs


def theoretical_credit_bc(
    social_rate: float,
    loan_rate: float,
    term_years: int,
    grace_years: float = 0.0,
    indirect_benefit_factor: float = 0.10,
    extra_fee_rate: float = 0.0,
) -> float:
    """
    Closed-form benefit-cost ratio of a concessional loan.

    For a credit instrument the benefit is the disbursed amount (grossed up by
    the indirect benefit factor) and the cost is the present value of repaying
    that same amount. The disbursement cancels, so the ratio does not depend on
    the loss catalogue, on which events occurred, or on how large they were:

        B/C = (1 + f_B) / (phi + extra_fee_rate)

    where phi is the present value of full debt service per unit disbursed,
    as computed by ``loan_pv_at_disbursement``.

    Two consequences follow. First, the benefit-cost ratio of a credit
    instrument is bounded near unity: a loan is close to a zero-NPV operation
    and the only margin comes from the concessional spread between the social
    discount rate and the contractual rate, plus the grace period. It can
    never reach the values a risk-transfer instrument can reach, and comparing
    the two on this indicator alone is misleading. Second, the ratio is
    identical in every simulation, so its distribution is degenerate and any
    reported probability is an algebraic identity rather than a probability.

    The function is intended for two uses: as an internal consistency check
    against the simulated ratio, and as a sensitivity tool that lets the
    analyst see the effect of renegotiating the term, the rate or the grace
    period without re-running the simulation.

    Note that the ratio is exact only for instruments whose entire cost scales
    with the disbursement. The PPO also pays a commitment fee on its undrawn
    balance and a front-end fee, neither of which scales with the amount drawn,
    so its simulated ratio is lower than this closed form and does vary across
    simulations.

    Parameters
    ----------
    social_rate : float
        Social discount rate.
    loan_rate : float
        Contractual interest rate on the loan.
    term_years : int
        Total repayment term, including grace.
    grace_years : float
        Interest-only grace years.
    indirect_benefit_factor : float
        Indirect benefit factor f_B applied to payouts.
    extra_fee_rate : float
        Fees proportional to the disbursement, such as the CCF drawdown fee.

    Returns
    -------
    float
        Benefit-cost ratio, or ``inf`` when the present value of the cost is
        zero or negative.
    """
    phi = loan_pv_at_disbursement(
        principal=1.0,
        loan_rate=loan_rate,
        social_rate=social_rate,
        term_years=term_years,
        grace_years=grace_years,
    ) + extra_fee_rate

    if phi <= 0:
        return float("inf")
    return (1.0 + indirect_benefit_factor) / phi
