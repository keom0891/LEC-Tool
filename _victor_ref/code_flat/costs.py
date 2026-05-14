"""
LEC-CBA Cost Functions Module
==============================
Instrument-specific cost and payout calculations aligned to the BID's
LEC tool base code. Three instruments:

  1. CCRIF (standard_insurance_payout) — parametric insurance
  2. PPO   (apply_ppo_coverage)        — contingent credit, single trigger
  3. CCF   (ccf_coverage)              — contingent credit, recurrent

Each instrument evaluates INDEPENDENTLY against the total fiscal loss.
This is consistent with the BID's parallel evaluation approach where:
    total_coverage = drm1_payout_df + drm2_payout_df + drm3_payout_df
Instruments do NOT see each other's payouts.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List
from config import InsuranceConfig, PPOConfig, CCFConfig


# ============================================================
# CCRIF — Parametric Insurance
# Mirrors BID's standard_insurance_payout()
# ============================================================

def insurance_payout(cfg: InsuranceConfig, loss: float) -> float:
    """
    Standard layer payout — identical to BID's standard_insurance_payout.

    If loss <= AP: 0
    If AP < loss < EP: (loss - AP) × ceding_percentage
    If loss >= EP: (EP - AP) × ceding_percentage
    """
    if loss <= cfg.attachment_point:
        return 0.0
    elif loss >= cfg.exhaustion_point:
        return (cfg.exhaustion_point - cfg.attachment_point) * cfg.ceding_percentage
    else:
        return (loss - cfg.attachment_point) * cfg.ceding_percentage


def insurance_cost_primary(
    cfg: InsuranceConfig, year: int, loss: float
) -> tuple:
    """
    Primary cost: flat annual premium, independent of activation.
    Cost_t = ROL × (EP - AP) × ceding_percentage
    """
    cost = cfg.premium
    payout = insurance_payout(cfg, loss)
    return cost, payout


def insurance_cost_alternative(
    cfg: InsuranceConfig, year: int, loss: float, claim_history: list
) -> tuple:
    """
    Alternative cost: experience-rated premium.
    Premium adjusts based on claims history via loading factor.
    """
    base_premium = cfg.premium
    if len(claim_history) > 0 and sum(claim_history) > 0:
        loss_ratio = sum(claim_history) / (base_premium * len(claim_history))
        loading = min(1.5, max(1.0, 0.8 + 0.2 * loss_ratio))
    else:
        loading = 1.0
    cost = base_premium * loading
    payout = insurance_payout(cfg, loss)
    return cost, payout


# ============================================================
# PPO — Policy Payout Option (single-trigger contingent credit)
# Mirrors BID's apply_ppo_coverage()
# ============================================================

@dataclass
class PPOState:
    """Tracks PPO activation and debt across periods."""
    triggered: bool = False
    outstanding_debt: float = 0.0
    annual_debt_service: float = 0.0
    remaining_repayment_years: int = 0
    front_end_fee_paid: bool = False


def ppo_payout(cfg: PPOConfig, state: PPOState, year: int, loss: float) -> float:
    """
    PPO payout — identical logic to BID's apply_ppo_coverage.
    Triggers ONCE when loss > trigger and not yet triggered.
    Available amount depends on the year (ppo_available vector).

    IMPORTANT: Matches BID behavior where triggering consumes the PPO
    even if ppo_available[year] == 0. This means a large loss in year 0
    (when available = $0) permanently disables the PPO. This is a known
    limitation of the BID's code (603/1000 sims waste the trigger in
    testing with Peru data). Maintained for strict compatibility.
    """
    if loss > cfg.loss_trigger and not state.triggered:
        state.triggered = True  # consumed regardless of available amount
        available_idx = min(year, len(cfg.ppo_available) - 1)
        return cfg.ppo_available[available_idx]
    return 0.0


def ppo_cost_primary(
    cfg: PPOConfig, state: PPOState, year: int, loss: float
) -> tuple:
    """
    Primary cost function for PPO:
      - Front-end fee (year 0 only)
      - Commitment fee on undrawn balance (annual)
      - If drawn: annuity debt service over repayment period

    Annuity: A = D × r(1+r)^n / ((1+r)^n - 1)
    """
    cost = 0.0

    # Front-end fee (one-time)
    if year == 0 and not state.front_end_fee_paid:
        cost += cfg.credit_line * cfg.front_end_fee_rate
        state.front_end_fee_paid = True

    # Debt service on outstanding draws
    if state.outstanding_debt > 0 and state.remaining_repayment_years > 0:
        r = cfg.interest_rate
        n = state.remaining_repayment_years
        if r > 0:
            annuity = state.outstanding_debt * (r * (1 + r)**n) / ((1 + r)**n - 1)
        else:
            annuity = state.outstanding_debt / n
        cost += annuity
        principal_payment = annuity - state.outstanding_debt * r
        state.outstanding_debt -= principal_payment
        state.remaining_repayment_years -= 1
        if state.remaining_repayment_years <= 0:
            state.outstanding_debt = 0

    # Commitment fee on available (undrawn) balance
    available = cfg.credit_line - state.outstanding_debt
    cost += max(0, available) * cfg.commitment_fee_rate

    # Payout (state.triggered is set inside ppo_payout)
    payout = ppo_payout(cfg, state, year, loss)
    if payout > 0:
        state.outstanding_debt += payout
        state.remaining_repayment_years = cfg.repayment_years

    return cost, payout, state


# ============================================================
# CCF — Contingent Credit Facility (recurrent, population-based)
# Mirrors BID's ccf_coverage()
# ============================================================

@dataclass
class CCFState:
    """Tracks CCF debt across periods (can have multiple draws)."""
    total_outstanding_debt: float = 0.0
    draw_debts: List[dict] = field(default_factory=list)
    # Each draw: {"amount": float, "remaining_years": int}


def _ccf_empirical_payout(loss: float, ccf_person: float,
                          pop_exposed: float, ccf_maximum: float) -> float:
    """
    BID's empirical payout function for CCF.

    Uses an ad-hoc regression to estimate affected population from loss:
        affected = exp(0.001074 × ln(loss_thousands)^3.0883 + 7.9346)
    Then: payout = min(affected × ccf_person / 1e6, ccf_maximum)

    IMPORTANT: This function has R² ≈ 0.34 per BID's code comment.
    The relationship between economic loss and affected population is
    inherently noisy. Users should evaluate whether this empirical
    function is appropriate for their country context.

    Args:
        loss: fiscal loss in M USD
        ccf_person: USD per affected person
        pop_exposed: total population exposed
        ccf_maximum: maximum payout per event (M USD)
    """
    min_threshold = 0.01 * pop_exposed * ccf_person / 1e6
    if loss < min_threshold:
        return 0.0

    # BID's empirical function (loss in M USD, converted to thousands)
    loss_thousands = loss * 1000  # M USD → thousands USD
    if loss_thousands <= 0:
        return 0.0

    affected = np.exp(0.001074 * np.log(loss_thousands) ** 3.0883 + 7.9346)
    payout = affected * ccf_person / 1e6  # convert to M USD
    return min(payout, ccf_maximum)


def _ccf_layer_payout(loss: float, cfg: CCFConfig) -> float:
    """
    Alternative payout function: standard layer structure.

    For contexts where the BID's empirical population function is not
    applicable or where a simpler parametric approach is preferred.
    Uses attachment_point, exhaustion_point from CCFConfig (no ceding
    percentage — CCF covers 100% within its layer).
    """
    if loss <= cfg.attachment_point:
        return 0.0
    elif loss >= cfg.exhaustion_point:
        return cfg.exhaustion_point - cfg.attachment_point
    else:
        return loss - cfg.attachment_point


def ccf_payout(cfg: CCFConfig, loss: float) -> float:
    """
    CCF payout — dispatches to empirical or layer function.
    Unlike PPO, CCF can activate EVERY period.
    """
    if cfg.payout_function == "bid_empirical":
        return _ccf_empirical_payout(
            loss, cfg.ccf_person, cfg.pop_exposed, cfg.ccf_maximum
        )
    elif cfg.payout_function == "layer":
        return _ccf_layer_payout(loss, cfg)
    else:
        raise ValueError(f"Unknown CCF payout function: {cfg.payout_function}")


def ccf_cost_primary(
    cfg: CCFConfig, state: CCFState, year: int, loss: float
) -> tuple:
    """
    Primary cost function for CCF.

    Per IDB CCF terms:
      - No commitment fee, no up-front fee.
      - Drawdown fee: 50 bps on disbursed amount (per activation).
      - Interest on drawn amount over repayment period.

    Debt from multiple activations accumulates independently.
    Each draw has its own repayment schedule.
    """
    cost = 0.0

    # Service existing debts (each draw repays independently)
    active_debts = []
    for draw in state.draw_debts:
        if draw["remaining_years"] > 0 and draw["amount"] > 0:
            r = cfg.interest_rate
            n = draw["remaining_years"]
            if r > 0 and n > 0:
                annuity = draw["amount"] * (r * (1 + r)**n) / ((1 + r)**n - 1)
            else:
                annuity = draw["amount"] / max(n, 1)
            cost += annuity
            principal_payment = annuity - draw["amount"] * r
            draw["amount"] -= principal_payment
            draw["remaining_years"] -= 1
            if draw["remaining_years"] > 0 and draw["amount"] > 0:
                active_debts.append(draw)
        # else: debt fully repaid, drop it
    state.draw_debts = active_debts

    # Payout (can trigger every period)
    payout = ccf_payout(cfg, loss)

    # Drawdown fee + new debt
    if payout > 0:
        cost += payout * cfg.drawdown_fee_rate
        state.draw_debts.append({
            "amount": payout,
            "remaining_years": cfg.repayment_years
        })

    state.total_outstanding_debt = sum(d["amount"] for d in state.draw_debts)

    return cost, payout, state
