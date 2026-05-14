"""
LEC-CBA Configuration Module
=============================
Defines all user-configurable parameters for the cost-benefit analysis.
Instruments aligned to the BID's LEC tool base code:
  - CCRIF (standard_insurance_payout) — parametric insurance
  - PPO   (apply_ppo_coverage)        — contingent credit, single activation
  - CCF   (ccf_coverage)              — contingent credit, recurrent activation

Additionally includes configuration for:
  - DRR (Disaster Risk Reduction) — ex-ante investment in prevention

NOTE: All monetary values in M USD (millions of US dollars).
"""

from dataclasses import dataclass, field
from typing import Optional, List
import numpy as np


@dataclass
class GovernmentExposureConfig:
    """
    Government fiscal exposure as a share of total economic losses.
    Called 'resp_fiscal' in BID's code (default 0.33 in their example).
    """
    factor: float = 0.15


@dataclass
class DiscountConfig:
    """Time value of money parameters."""
    social_discount_rate: float = 0.05
    analysis_horizon: int = 10
    num_simulations: int = 1000


@dataclass
class IndirectBenefitConfig:
    """Indirect benefit multiplier for deployed resources."""
    factor: float = 0.10


@dataclass
class OMVConfig:
    """Parameters for the Optimized Money Value metric."""
    lambda_risk_adjustment: float = 0.05  # World Bank default


# ================================================================
# Instrument Configurations — aligned to BID's LEC tool
# ================================================================

@dataclass
class InsuranceConfig:
    """
    Parametric Insurance — CCRIF style.
    Maps to BID's standard_insurance_payout(loss, attachment_point,
    exhaustion_point, ceding_percentage).

    Payout: standard layer structure.
    Cost: annual premium = ROL × (EP - AP) × ceding_percentage.
    """
    attachment_point: float = 50.0
    exhaustion_point: float = 190.0
    ceding_percentage: float = 0.066
    rate_on_line: float = 0.05
    premium: Optional[float] = None
    cost_function: str = "primary"

    def __post_init__(self):
        if self.premium is None:
            coverage = (self.exhaustion_point - self.attachment_point) * self.ceding_percentage
            self.premium = self.rate_on_line * coverage


@dataclass
class PPOConfig:
    """
    Policy Payout Option — contingent credit, single activation.
    Maps to BID's apply_ppo_coverage(values, ppo_available, ppo_loss_trigger).

    Key characteristics:
      - Triggers ONCE in the analysis horizon (ppo_triggered flag).
      - Available amount may vary by year (ppo_available vector).
      - Once triggered, cannot reactivate in subsequent periods.

    Cost structure:
      - Commitment fee on undrawn balance (annual).
      - Front-end fee (one-time, year 0).
      - If drawn: annuity debt service over repayment period.
    """
    ppo_available: Optional[List[float]] = None  # available amount per year
    loss_trigger: float = 120.0
    credit_line: float = 46.0           # max available (last element of ppo_available)
    commitment_fee_rate: float = 0.005
    interest_rate: float = 0.035
    repayment_years: int = 5
    front_end_fee_rate: float = 0.0025
    cost_function: str = "primary"

    def __post_init__(self):
        if self.ppo_available is None:
            # BID default: gradual availability over 10 years
            self.ppo_available = [0, 2, 10, 25, 35, 42, 46, 46, 46, 46]
        self.credit_line = max(self.ppo_available)


@dataclass
class CCFConfig:
    """
    Contingent Credit Facility — recurrent activation, population-based.
    Maps to BID's ccf_coverage(values, ccf_maximum, ccf_person, Pop_exposed).

    Key characteristics:
      - Can activate EVERY period (unlike PPO's single activation).
      - Payout determined by empirical function estimating affected population.
      - Fixed amount per affected person, capped at maximum per event.

    Payout function (BID default):
      affected_pop = exp(0.001074 × ln(loss×1000)^3.0883 + 7.9346)
      payout = min(affected_pop × ccf_person / 1e6, ccf_maximum)
      NOTE: This empirical function has R² ≈ 0.34 (per BID's code comment).

    Alternative payout function:
      Standard layer structure like CCRIF, for contexts where the
      empirical population function is not applicable or where a
      simpler parametric approach is preferred.

    Cost structure (per IDB CCF terms):
      - No commitment fee, no up-front fee.
      - Drawdown fee: 50 bps on disbursed amount (one-time per activation).
      - Interest: SOFR + IDB spread on drawn amount.
      - Maturity: up to 25 years, grace period 5.5 years.
    """
    ccf_maximum: float = 300.0          # M USD max payout per event
    ccf_person: float = 1650.0          # USD per affected person
    pop_exposed: float = 10.83e6        # total population exposed
    drawdown_fee_rate: float = 0.005    # 0.5% on disbursed amount
    interest_rate: float = 0.04         # SOFR + spread (approx)
    repayment_years: int = 25
    grace_period_years: int = 5
    # Minimum loss threshold: 1% of pop_exposed × ccf_person / 1e6
    min_loss_threshold: Optional[float] = None
    payout_function: str = "bid_empirical"  # "bid_empirical" or "layer"
    # Layer parameters (used only if payout_function == "layer")
    attachment_point: float = 0.0
    exhaustion_point: float = 300.0
    cost_function: str = "primary"

    def __post_init__(self):
        if self.min_loss_threshold is None:
            self.min_loss_threshold = 0.01 * self.pop_exposed * self.ccf_person / 1e6


# ================================================================
# DRR Configuration
# ================================================================

@dataclass
class DRRConfig:
    """
    Disaster Risk Reduction — ex-ante investment in prevention.
    Maps to BID's DRR mechanism (calibrate_LEC_AAL).

    The user specifies annual vectors over the analysis horizon:
      - inv: investment amount per year (M USD)
      - rbc: benefit-cost ratio of each investment
      - hor: useful life of the investment (years)

    Annual AAL reduction = inv × rbc / hor, accumulated over time.
    The LEC curve is deformed downward to match each year's target.

    For the CBA module, we evaluate:
      - Direct benefit: E[L_original] - E[L_reduced]
      - Indirect benefit: E[UL_original] - E[UL_reduced]
      - Cost: PV of investment amounts
      - B/C ratios: direct and indirect
    """
    inv: Optional[List[float]] = None
    rbc: Optional[List[float]] = None
    hor: Optional[List[float]] = None
    enabled: bool = False

    def __post_init__(self):
        if self.inv is None:
            self.inv = [0, 0, 50, 0, 0, 100, 0, 0, 0, 0]
        if self.rbc is None:
            self.rbc = [4, 4, 4, 4, 4, 4, 4, 4, 4, 4]
        if self.hor is None:
            self.hor = [20, 20, 20, 20, 20, 20, 20, 20, 20, 20]

    def cumulative_reduction(self) -> List[float]:
        """Compute cumulative AAL reduction per year."""
        red, cumulative = [], 0.0
        for i in range(len(self.inv)):
            red.append(cumulative)
            value = self.inv[i] * self.rbc[i] / self.hor[i]
            cumulative += value
        return red


# ================================================================
# Master Configuration
# ================================================================

@dataclass
class LECCBAConfig:
    """Master configuration combining all components."""
    government_exposure: GovernmentExposureConfig = field(
        default_factory=GovernmentExposureConfig
    )
    discount: DiscountConfig = field(default_factory=DiscountConfig)
    indirect_benefit: IndirectBenefitConfig = field(default_factory=IndirectBenefitConfig)
    omv: OMVConfig = field(default_factory=OMVConfig)
    insurance: InsuranceConfig = field(default_factory=InsuranceConfig)
    ppo: PPOConfig = field(default_factory=PPOConfig)
    ccf: CCFConfig = field(default_factory=CCFConfig)
    drr: DRRConfig = field(default_factory=DRRConfig)

    def summary(self) -> str:
        lines = [
            "=" * 60,
            "LEC-CBA Configuration Summary",
            "=" * 60,
            f"Government exposure factor: {self.government_exposure.factor:.0%}",
            f"Analysis horizon: {self.discount.analysis_horizon} years",
            f"Simulations: {self.discount.num_simulations}",
            f"Discount rate: {self.discount.social_discount_rate:.1%}",
            f"Indirect benefit factor: {self.indirect_benefit.factor:.1%}",
            f"OMV lambda: {self.omv.lambda_risk_adjustment}",
            "",
            "--- CCRIF (Parametric Insurance) ---",
            f"  Layer: ${self.insurance.attachment_point:.0f}M"
            f" - ${self.insurance.exhaustion_point:.0f}M",
            f"  Ceding: {self.insurance.ceding_percentage:.1%}",
            f"  ROL: {self.insurance.rate_on_line:.1%}",
            f"  Annual premium: ${self.insurance.premium:.1f}M",
            "",
            "--- PPO (Contingent Credit — single activation) ---",
            f"  Trigger: ${self.ppo.loss_trigger:.0f}M",
            f"  Max available: ${self.ppo.credit_line:.0f}M",
            f"  Commitment fee: {self.ppo.commitment_fee_rate:.2%}",
            f"  Interest rate: {self.ppo.interest_rate:.1%}",
            f"  Repayment: {self.ppo.repayment_years} years",
            "",
            "--- CCF (Contingent Credit — recurrent) ---",
            f"  Max per event: ${self.ccf.ccf_maximum:.0f}M",
            f"  Per person: ${self.ccf.ccf_person:,.0f}",
            f"  Pop exposed: {self.ccf.pop_exposed/1e6:.2f}M",
            f"  Drawdown fee: {self.ccf.drawdown_fee_rate:.2%}",
            f"  Payout function: {self.ccf.payout_function}",
            "",
            "--- DRR (Disaster Risk Reduction) ---",
            f"  Enabled: {self.drr.enabled}",
        ]
        if self.drr.enabled:
            lines.append(f"  Total investment: ${sum(self.drr.inv):.0f}M")
            lines.append(f"  Final cumulative reduction: "
                         f"${self.drr.cumulative_reduction()[-1]:.1f}M AAL")
        lines.append("=" * 60)
        return "\n".join(lines)


def get_default_config() -> LECCBAConfig:
    """Returns default configuration aligned to BID's example."""
    return LECCBAConfig()


if __name__ == "__main__":
    cfg = get_default_config()
    print(cfg.summary())
