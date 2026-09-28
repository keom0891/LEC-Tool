"""
LEC-CBA Configuration Module
=============================
Defines all user-configurable parameters for the cost-benefit analysis.
Instruments aligned to the BID's LEC tool base code:
  - CCRIF (standard_insurance_payout) — parametric insurance
  - PPO   (apply_ppo_coverage)        — contingent credit, single activation
  - CCF   (ccf_coverage)              — contingent credit, recurrent activation
  - DDO   (ddo_coverage)              — deferred drawdown option, recurrent

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

    # --- Payout behaviour -------------------------------------------------
    # These two fields mirror the arguments of
    # risk_management.standard_insurance_payout and MUST be kept consistent
    # with the values passed there through drm_configs. They live here so
    # the pricing diagnostic knows which actuarial benchmark applies.
    #
    # 'proportional' pays the ceded share of the loss inside the layer, as a
    # real CCRIF policy does. 'binary' pays the full coverage limit on any
    # loss above the attachment point; it is a teaching simplification used
    # in the worked example so the mechanism can be drawn without a ramp.
    payout_mode: str = "proportional"
    # When True the policy pays at most once per policy year, cover being
    # restored only at renewal. The two options move the expected payout in
    # OPPOSITE directions and must be decided together; see
    # standard_insurance_payout for the measured magnitudes.
    one_payout_per_year: bool = False

    # --- v8 -----------------------------------------------------------------
    # Donor discount on the annual premium ($MM). Sovereign pools such as
    # CCRIF are partly funded by donors who pay part of the premium. The
    # economic cost of the insurance is the gross premium; the cost to the
    # government is the gross premium less this discount. Reported as the
    # economic and the fiscal benefit-cost ratio of the instrument.
    donor_discount: float = 0.0
    # How the premium was set: 'rol_rule', 'market_curve', 'quote' or
    # 'fixed_rol' (see insurance_layer.py). Informational.
    pricing_method: str = "fixed_rol"

    @property
    def net_premium(self) -> float:
        return self.premium - self.donor_discount

    def __post_init__(self):
        if self.premium is None:
            coverage = (self.exhaustion_point - self.attachment_point) * self.ceding_percentage
            self.premium = self.rate_on_line * coverage
        # Defensive validation: stop on absurd configuration.
        if self.exhaustion_point <= self.attachment_point:
            raise ValueError(
                f"InsuranceConfig: exhaustion_point ({self.exhaustion_point}) must "
                f"be greater than attachment_point ({self.attachment_point}). "
                f"The layer [attachment, exhaustion] would be empty or inverted."
            )
        if not (0.0 <= self.ceding_percentage <= 1.0):
            raise ValueError(
                f"InsuranceConfig: ceding_percentage must be in [0, 1], got "
                f"{self.ceding_percentage}."
            )
        if not (0.0 <= self.donor_discount <= self.premium):
            raise ValueError(
                f"InsuranceConfig: donor_discount ({self.donor_discount}) must be "
                f"between 0 and the gross premium ({self.premium})."
            )


@dataclass
class PPOConfig:
    """
    Policy Payout Option — contingent credit, single activation.
    Maps to BID's apply_ppo_coverage(values, ppo_available, ccf_applied).

    Key characteristics:
      - Triggers ONCE in the analysis horizon (ppo_triggered flag).
      - Available amount may vary by year (ppo_available vector).
      - Once triggered, cannot reactivate in subsequent periods.

    Cost structure:
      - Commitment fee on undrawn balance (annual).
      - Front-end fee (one-time, year 0).
      - If drawn: annuity debt service over repayment period.

    NOTE: PPO trigger is controlled by risk_management.apply_strategy
    (conditioned on CCF activation), not by the CBA module.
    """
    ppo_available: Optional[List[float]] = None  # available amount per year
    credit_line: float = 46.0           # max available (last element of ppo_available)
    commitment_fee_rate: float = 0.005
    # Contractual interest rate on the PPO loan (NOT the social discount rate).
    # This rate sizes the debt-service annuity. The social discount rate used
    # to bring cash flows to present value lives in DiscountConfig and is a
    # distinct, separate parameter.
    loan_interest_rate: float = 0.035
    repayment_years: int = 5
    front_end_fee_rate: float = 0.0025
    # Grace period on the PPO loan, in years. During grace only interest is
    # paid; amortisation starts afterwards. Multilateral contingent credit
    # normally carries a grace period of 2 to 5 years; a value of zero
    # describes no real contract and overstates the present value of the
    # cost. The default is kept at zero for backward compatibility with
    # earlier versions and is expected to be set by the analyst.
    grace_period_years: float = 0.0
    cost_function: str = "primary"

    def __post_init__(self):
        if self.ppo_available is None:
            # BID default: gradual availability over 10 years
            self.ppo_available = [0, 2, 10, 25, 35, 42, 46, 46, 46, 46]
        # Defensive validation: stop on absurd configuration.
        if any(a < 0 for a in self.ppo_available):
            raise ValueError(
                f"PPOConfig: ppo_available must not contain negative amounts, "
                f"got {self.ppo_available}. A negative available amount has no "
                f"meaning."
            )
        self.credit_line = max(self.ppo_available)
        if self.credit_line < 0:
            raise ValueError(
                f"PPOConfig: credit_line must be >= 0, got {self.credit_line}."
            )
        if self.repayment_years <= 0:
            raise ValueError(
                f"PPOConfig: repayment_years must be > 0, got {self.repayment_years}."
            )
        # Grace must leave a strictly positive amortisation window. With
        # grace equal to the repayment term the principal is never
        # amortised, which describes a bullet loan. Bullet repayment is not
        # modelled here, so the configuration is rejected rather than
        # silently producing a division by an empty amortisation period.
        if self.grace_period_years < 0:
            raise ValueError(
                f"PPOConfig: grace_period_years must be >= 0, got "
                f"{self.grace_period_years}."
            )
        if self.grace_period_years >= self.repayment_years:
            raise ValueError(
                f"PPOConfig: grace_period_years ({self.grace_period_years}) must "
                f"be strictly less than repayment_years ({self.repayment_years}). "
                f"With grace equal to or longer than the term there is no "
                f"amortisation window and the principal would never be repaid. "
                f"Bullet repayment is outside the scope of this module."
            )


@dataclass
class CCFConfig:
    """
    Contingent Credit Facility — recurrent activation, population-based.
    Maps to BID's ccf_coverage(values, ccf_maximum, ccf_person, Pop_exposed).

    Key characteristics:
      - Can activate EVERY period (unlike PPO's single activation).
      - Payout determined by empirical function estimating affected population.
      - Fixed amount per affected person, capped at maximum.

    ccf_maximum: TOTAL facility limit for the entire catalogue period.
      This is the cumulative depletion cap across ALL events and ALL years —
      matching Kenneth's ccf_coverage logic in risk_management.py where
      remaining_cover is tracked across the full simulation horizon.

    Payout function (BID default):
      affected_pop = exp(0.001074 × ln(loss×1000)^3.0883 + 7.9346)
      payout = min(affected_pop × ccf_person / 1e6, ccf_maximum)
      NOTE: This empirical function has R² ≈ 0.34 (per BID's code comment).

    Cost structure (per IDB CCF terms):
      - No commitment fee, no up-front fee.
      - Drawdown fee: 50 bps on disbursed amount (one-time per activation).
      - Interest: SOFR + IDB spread on drawn amount.
      - Maturity: up to 25 years, grace period 5.5 years.
    """
    ccf_maximum: float = 300.0          # M USD total facility limit (cumulative)
    ccf_person: float = 1650.0          # USD per affected person
    pop_exposed: float = 10.83e6        # total population exposed
    drawdown_fee_rate: float = 0.005    # 0.5% on disbursed amount
    # Contractual interest rate on the CCF loan (NOT the social discount rate).
    # Sizes the debt-service annuity after the grace period. The social discount
    # rate is a separate parameter in DiscountConfig.
    loan_interest_rate: float = 0.04    # SOFR + spread (approx)
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

        # Defensive validation: stop on absurd configuration.
        if self.pop_exposed <= 0:
            raise ValueError(
                f"CCFConfig: pop_exposed must be > 0, got {self.pop_exposed}. "
                f"The CCF payout scales with affected population; a non-positive "
                f"exposed population makes the instrument meaningless."
            )
        if self.ccf_person < 0:
            raise ValueError(
                f"CCFConfig: ccf_person must be >= 0, got {self.ccf_person}. "
                f"A negative payout per person has no meaning."
            )
        if self.ccf_maximum < 0:
            raise ValueError(
                f"CCFConfig: ccf_maximum must be >= 0, got {self.ccf_maximum}."
            )

        # Validate grace vs repayment term. The grace period must be strictly
        # shorter than the total repayment term: there has to be at least one
        # year of amortisation, otherwise the principal is never repaid
        # (interest-only forever), which is not a valid loan structure.
        if self.grace_period_years >= self.repayment_years:
            raise ValueError(
                f"CCF grace_period_years ({self.grace_period_years}) must be "
                f"strictly less than repayment_years ({self.repayment_years}). "
                f"With grace >= repayment there is no amortisation window and "
                f"the principal would never be repaid. Reduce the grace period "
                f"or extend the repayment term."
            )


@dataclass
class DDOConfig:
    """
    Deferred Drawdown Option — triggers on every event exceeding the threshold.
    No commitment fee. Cost = interest on drawn amount over repayment period.
    """
    # Contractual interest rate on the DDO loan (NOT the social discount rate).
    loan_interest_rate: float = 0.035
    repayment_years: int = 5
    # Grace period on the DDO loan, in years. See PPOConfig for the rationale;
    # World Bank CAT-DDO operations normally carry a grace period. The default
    # is kept at zero for backward compatibility.
    grace_period_years: float = 0.0
    cost_function: str = "primary"
    ddo_threshold: Optional[float] = None   # loss trigger ($MM)
    ddo_available: Optional[float] = None   # fixed payout per activation ($MM)

    def __post_init__(self):
        if self.repayment_years <= 0:
            raise ValueError(
                f"DDOConfig: repayment_years must be > 0, got {self.repayment_years}."
            )
        if self.grace_period_years < 0:
            raise ValueError(
                f"DDOConfig: grace_period_years must be >= 0, got "
                f"{self.grace_period_years}."
            )
        if self.grace_period_years >= self.repayment_years:
            raise ValueError(
                f"DDOConfig: grace_period_years ({self.grace_period_years}) must "
                f"be strictly less than repayment_years ({self.repayment_years}). "
                f"With grace equal to or longer than the term there is no "
                f"amortisation window and the principal would never be repaid. "
                f"Bullet repayment is outside the scope of this module."
            )


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
    ddo: DDOConfig = field(default_factory=DDOConfig)
    drr: DRRConfig = field(default_factory=DRRConfig)
    # CNC (Comparative Net Cost) — optional, modular indicator.
    # Imported lazily to keep the module fully removable.
    cnc: "object" = None

    # v6: debt-service accounting mode.
    # False (default) = full present value of debt service booked at disbursement
    #   (Clarke, Mahul, Poulter & Teh 2017). Correct treatment for loans whose
    #   repayment extends beyond the analysis horizon (notably CCF, 25 years).
    # True = legacy year-by-year annuity truncated at the horizon. Use only for
    #   cross-validation against the BID Colab outputs.
    legacy_truncation: bool = False

    def __post_init__(self):
        # Initialize CNC config lazily. If the cnc module is present, use its
        # default config; if it has been removed, leave cnc as None so the rest
        # of the pipeline keeps working (full modularity).
        if self.cnc is None:
            try:
                from cba.cnc import CNCConfig
                self.cnc = CNCConfig()
            except ImportError:
                self.cnc = None

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
            f"  Max available: ${self.ppo.credit_line:.0f}M",
            f"  Commitment fee: {self.ppo.commitment_fee_rate:.2%}",
            f"  Loan interest rate: {self.ppo.loan_interest_rate:.1%}",
            f"  Repayment: {self.ppo.repayment_years} years",
            "",
            "--- CCF (Contingent Credit — recurrent) ---",
            f"  Total facility: ${self.ccf.ccf_maximum:.0f}M (cumulative cap)",
            f"  Per person: ${self.ccf.ccf_person:,.0f}",
            f"  Pop exposed: {self.ccf.pop_exposed/1e6:.2f}M",
            f"  Drawdown fee: {self.ccf.drawdown_fee_rate:.2%}",
            f"  Grace period: {self.ccf.grace_period_years} years",
            f"  Payout function: {self.ccf.payout_function}",
            "",
            "--- DDO (Deferred Drawdown Option — recurrent) ---",
            f"  Loan interest rate: {self.ddo.loan_interest_rate:.1%}",
            f"  Repayment: {self.ddo.repayment_years} years",
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
