# -*- coding: utf-8 -*-
"""
pricing.py — Insurance layer pricing and actuarial coherence diagnostics
========================================================================

This module addresses a structural vulnerability of the LEC-CBA framework:
the Rate-on-Line (ROL) of the parametric insurance layer is an *exogenous*
parameter. Nothing in the framework checks whether the ROL the analyst types
is consistent with the loss curve that generates the payouts. A ROL that is
appropriate for a remote layer, applied to a frequent layer, produces an
artificially attractive benefit-cost ratio for insurance.

Two independent capabilities are provided.

1. MARKET ROL CURVE (``rol_from_return_period``)
   A monotone interpolation of a reinsurance pricing curve expressed as
   ROL versus the return period of the attachment point. Reinsurance
   pricing is conventionally quoted this way: the more remote the layer,
   the lower the rate per unit of limit.

   The anchors are a market indication for Central America and the
   Caribbean obtained from a practising reinsurance broker (2026-08). They
   are a point-in-time indication, not a binding quotation: reinsurance
   prices are volatile and seasonal, and the table should be refreshed at
   each renewal cycle. In v8 the table is an input in config.toml
   ([insurance_pricing] market_curve_rp / market_curve_rol / market_curve_date);
   the values below are only the defaults.

       Return period (years) :   1     5    10    25    50   100   200   500  1000
       Rate-on-Line          : 40%   35%   25%   22%    9%    5%  4.5%    4%  3.5%

   Each anchor is read as the ROL of a THIN layer that is first hit at that
   event return period. A wide layer is priced by integrating the curve
   across the layer, slice by slice (insurance_layer.price_layer); evaluating
   the curve only at the attachment point overprices a wide layer.

   Interpolation is PCHIP (monotone cubic Hermite) in log-log space.
   A high-order polynomial through these anchors was evaluated and
   rejected: it exhibits Runge oscillation and returns negative rates
   between and beyond the anchors.

   For documentation purposes a smooth closed form is also provided
   (``rol_quadratic_closed_form``), a least-squares cubic in log-log space.
   With the current anchors it reproduces them with R^2 = 0.973 in logs and
   a maximum error of 29 per cent (the drop between 1 in 25 and 1 in 50 is
   hard to follow with a smooth curve). It is indicative only; the
   interpolation is what the code uses.

2. COHERENCE DIAGNOSTIC (``diagnose_insurance_pricing``)
   Given the simulated payouts actually produced by the layer, the
   diagnostic recovers the price the analyst is implicitly assuming and
   compares it with actuarial benchmarks:

       pure ROL         = E[annual payout] / ceded limit
       implied multiple = premium / E[annual payout]

   The implied multiple is the number of currency units of premium charged
   per unit of expected loss. Values below 1.0 mean the insurer is selling
   below expected loss, which no reinsurer does. Market multiples rise with
   the remoteness of the layer: sovereign catastrophe bonds placed in Latin
   America and the Caribbean in 2023-2026 priced between about 2 and 5.5
   times expected loss (Artemis deal directory).

   Following the diagnose-do-not-stop convention already used elsewhere in
   this codebase for absurd inputs, the diagnostic never raises. It flags
   and lets the analysis run.

How the premium is set (v8)
---------------------------
The premium itself is set in insurance_layer.py, before the simulation,
from the method chosen in [insurance_pricing] ('rol_rule' by default,
'market_curve', 'fixed_rol') or from a quotation entered per instrument.
cba.engine reports the implied multiple and the protection indicators of
every insurance instrument; this module provides the market curve.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np

try:  # SciPy is already a declared dependency of the package.
    from scipy.interpolate import PchipInterpolator
    _HAS_SCIPY = True
except ImportError:  # pragma: no cover - defensive only
    _HAS_SCIPY = False


# ================================================================
# Default market pricing curve
# ================================================================

#: Return periods (years) of the attachment point.
DEFAULT_RP_ANCHORS: List[float] = [1.0, 5.0, 10.0, 25.0, 50.0, 100.0, 200.0, 500.0, 1000.0]

#: Rate-on-Line at each anchor return period (fraction of ceded limit).
DEFAULT_ROL_ANCHORS: List[float] = [0.40, 0.35, 0.25, 0.22, 0.09, 0.05, 0.045, 0.04, 0.035]

#: Coefficients of the documentation closed form, cubic in log-log space:
#:     ln(ROL) = C3 * ln(RP)^3 + C2 * ln(RP)^2 + C1 * ln(RP) + C0
CLOSED_FORM_COEFFS = (0.02809, -0.29888, 0.38320, -0.93690)

#: Bounds applied to any ROL returned by the curve. The upper bound is not a
#: theoretical maximum (a per-occurrence layer with unlimited reinstatements
#: can exceed 100 per cent annually) but a guard against extrapolating the
#: curve far below the lowest anchor.
DEFAULT_ROL_MIN = 0.005
DEFAULT_ROL_MAX = 1.00

#: Range of implied cost multiples treated as commercially plausible.
DEFAULT_MULTIPLE_RANGE = (1.2, 3.0)


# ================================================================
# ROL curve
# ================================================================

def rol_from_return_period(
    return_period: float | Sequence[float],
    rp_anchors: Optional[Sequence[float]] = None,
    rol_anchors: Optional[Sequence[float]] = None,
    rol_min: float = DEFAULT_ROL_MIN,
    rol_max: float = DEFAULT_ROL_MAX,
) -> float | np.ndarray:
    """
    Rate-on-Line implied by the market pricing curve.

    Parameters
    ----------
    return_period : float or sequence of float
        Return period, in years, of the layer's attachment point. Values
        outside the anchor range are extrapolated by the monotone
        interpolator and then clipped to ``[rol_min, rol_max]``.
    rp_anchors, rol_anchors : sequence of float, optional
        Override the default curve. Must be the same length, at least two
        points, with strictly increasing return periods.
    rol_min, rol_max : float
        Bounds applied after interpolation.

    Returns
    -------
    float or ndarray
        Rate-on-Line as a fraction of ceded limit.

    Notes
    -----
    The curve is indexed on the return period of the attachment point only.
    It does not observe layer thickness, so a wide layer and a thin layer
    attaching at the same point receive the same rate per unit of limit,
    even though the wide layer carries far more expected loss per unit of
    limit. ``diagnose_insurance_pricing`` is the intended safeguard: a layer
    whose thickness makes the curve rate implausible will show an implied
    multiple outside the plausible range and be flagged.
    """
    rp = np.asarray(rp_anchors if rp_anchors is not None else DEFAULT_RP_ANCHORS,
                    dtype=float)
    rol = np.asarray(rol_anchors if rol_anchors is not None else DEFAULT_ROL_ANCHORS,
                     dtype=float)

    if rp.size != rol.size:
        raise ValueError(
            f"rol_from_return_period: rp_anchors ({rp.size}) and rol_anchors "
            f"({rol.size}) must have the same length."
        )
    if rp.size < 2:
        raise ValueError(
            "rol_from_return_period: at least two anchor points are required "
            "to define a pricing curve."
        )
    if np.any(np.diff(rp) <= 0):
        raise ValueError(
            f"rol_from_return_period: rp_anchors must be strictly increasing, "
            f"got {rp.tolist()}."
        )
    if np.any(rp <= 0) or np.any(rol <= 0):
        raise ValueError(
            "rol_from_return_period: anchor return periods and rates must be "
            "strictly positive (interpolation is performed in log-log space)."
        )

    x_query = np.log(np.asarray(return_period, dtype=float))
    scalar_input = np.ndim(return_period) == 0

    # Below the lowest anchor the curve is not defined and must not be
    # extrapolated. A monotone cubic fitted to anchors whose first segment is
    # nearly flat produces a *decreasing* rate as the return period falls,
    # which inverts the economics: a layer that attaches more frequently would
    # be quoted more cheaply. Queries below the lowest anchor are therefore
    # held at the lowest anchor rate. The value is a floor, not a quotation,
    # and ``diagnose_insurance_pricing`` flags the layer separately.
    x_query = np.clip(x_query, np.log(rp[0]), None)

    if _HAS_SCIPY:
        interpolator = PchipInterpolator(np.log(rp), np.log(rol), extrapolate=True)
        y = interpolator(x_query)
    else:  # pragma: no cover - fallback keeps the module importable
        y = np.interp(x_query, np.log(rp), np.log(rol))

    out = np.clip(np.exp(y), rol_min, rol_max)
    return float(out) if scalar_input else out


def rol_quadratic_closed_form(return_period: float | Sequence[float]) -> float | np.ndarray:
    """
    Documentation closed form of the pricing curve.

    Least-squares cubic in log-log space fitted to the default anchors:

        ln(ROL) = 0.02809*ln(RP)^3 - 0.29888*ln(RP)^2 + 0.38320*ln(RP) - 0.93690

    Fit quality is R^2 = 0.9727 in logs but the maximum absolute error at an
    anchor is 29.5 per cent, concentrated around the steep segment between
    the 1-in-25 and 1-in-50 anchors. No smooth low-order curve reproduces
    that segment: the market indication falls by roughly 60 per cent of the
    rate as the return period doubles there, far faster than in the
    neighbouring segments.

    Because a 29.5 per cent error in the rate propagates directly to the
    premium and hence to the benefit-cost ratio, this closed form is NOT
    suitable for computing results. It is retained only as a qualitative
    description of the curve's shape. Documentation should publish the
    anchor table and state that the code interpolates between anchors.
    ``rol_from_return_period`` passes through the anchors exactly and is
    what the analysis uses.
    """
    x = np.log(np.asarray(return_period, dtype=float))
    out = np.clip(np.exp(np.polyval(CLOSED_FORM_COEFFS, x)),
                  DEFAULT_ROL_MIN, DEFAULT_ROL_MAX)
    return float(out) if np.ndim(return_period) == 0 else out


def attachment_return_period(
    losses_matrix: np.ndarray,
    attachment_point: float,
) -> float:
    """
    Return period of the attachment point, estimated from simulated losses.

    Parameters
    ----------
    losses_matrix : ndarray
        Losses arranged as (simulations x years). Each cell is one year of
        loss for one simulation.
    attachment_point : float
        Attachment point in the same units as the losses.

    Returns
    -------
    float
        Estimated return period in years, or ``inf`` if the attachment point
        is never exceeded.

    Notes
    -----
    The estimate is based on the *annual* loss aggregate, because that is
    the granularity the CBA module receives. If the insurance payout is
    applied per event upstream, the per-event return period is longer than
    the figure reported here, since the annual aggregate is at least as
    large as the largest single event in the year. The number should be read
    as a lower bound on the remoteness of the attachment point.
    """
    losses = np.asarray(losses_matrix, dtype=float)
    if losses.size == 0:
        return float("inf")
    exceedance_rate = float(np.mean(losses > attachment_point))
    if exceedance_rate <= 0.0:
        return float("inf")
    return 1.0 / exceedance_rate


def loss_at_return_period(
    losses: np.ndarray,
    return_period: float | Sequence[float],
    years_represented: Optional[float] = None,
) -> float | np.ndarray:
    """
    Loss level associated with a given return period.

    This is the inverse of ``attachment_return_period`` and exists so that
    every instrument threshold in a strategy can be expressed on the SAME
    platform: return periods rather than dollar amounts.

    Why the platform matters
    -----------------------
    A dollar threshold means nothing on its own. The same $50M means "an event
    that happens every other year" against one loss curve and "an event that
    happens once a generation" against another. When one instrument's trigger
    was calibrated on one scale and another instrument's on a different scale,
    the resulting strategy is incoherent in a way that no single number in the
    report reveals: each instrument looks individually reasonable while the
    layering between them is arbitrary.

    That is exactly what happened in this project. The analysis was decided to
    run on total economic losses rather than on a fiscal share of them, on the
    explicit condition that instrument activation thresholds be restated on
    that same scale. The insurance attachment kept a fiscal-scale value, so it
    triggered roughly once every other year instead of once in fifteen, its
    expected payout ran an order of magnitude above what its premium implied,
    and its benefit-cost ratio came out several times too high. Nothing was
    miscalculated; two instruments were simply measured with different rulers.

    Return periods are also the platform in which the decision is actually
    taken. A finance ministry choosing sovereign cover decides how rare an
    event has to be before the instrument responds; the dollar figure is an
    output of that choice and of the loss curve, not an input. Working in
    return periods therefore makes the strategy both internally coherent and
    portable across countries and across catalogue revisions: the same
    ladder of return periods can be applied to any curve, and the dollar
    thresholds regenerate themselves.

    Recommended practice for a strategy
    -----------------------------------
    Express every threshold as a return period and lay them out as an
    increasing ladder, so each instrument covers the band where the cheaper
    ones below it have run out. The worked example uses:

        1-in-5   years : DDO trigger          (frequent, cheap debt first)
        1-in-10  years : PPO trigger          (single activation, reserved)
        1-in-15  years : insurance attachment (risk transfer starts)
        1-in-50  years : insurance exhaustion (cover fully used)

    The CCF has no loss threshold: it responds through an empirical
    population-affected function and therefore participates from the bottom
    of the curve upwards.

    Parameters
    ----------
    losses : ndarray
        Event losses, or annual losses, depending on the granularity at which
        the threshold will be applied. Use EVENT losses for instruments whose
        trigger is evaluated event by event, which is the case for insurance,
        the PPO and the DDO in this codebase. Mixing granularities
        reintroduces exactly the incoherence this function exists to prevent.
    return_period : float or sequence of float
        Return period in years.
    years_represented : float, optional
        Number of years the ``losses`` sample represents. Required when
        ``losses`` is an event sample, since the number of events is not the
        number of years. If omitted, ``losses`` is assumed to be one
        observation per year.

    Returns
    -------
    float or ndarray
        Loss level exceeded on average once every ``return_period`` years.
    """
    x = np.asarray(losses, dtype=float).ravel()
    if x.size == 0:
        raise ValueError("loss_at_return_period: losses is empty.")

    rp = np.asarray(return_period, dtype=float)
    if np.any(rp <= 0):
        raise ValueError(
            f"loss_at_return_period: return periods must be positive, got "
            f"{rp.tolist() if rp.ndim else float(rp)}."
        )

    n_years = float(years_represented) if years_represented is not None else float(x.size)
    if n_years <= 0:
        raise ValueError(
            f"loss_at_return_period: years_represented must be positive, got "
            f"{years_represented}."
        )

    # Target annual exceedance rate, converted to a quantile of the sample.
    rate = 1.0 / rp
    quantile = 1.0 - rate * n_years / x.size
    out = np.where((quantile > 0) & (quantile < 1),
                   np.quantile(x, np.clip(quantile, 0.0, 1.0)),
                   np.nan)
    return float(out) if np.ndim(return_period) == 0 else out


# ================================================================
# Coherence diagnostic
# ================================================================

@dataclass
class InsurancePricingDiagnostic:
    """Result of the actuarial coherence check on an insurance layer."""

    instrument_name: str = ""
    attachment_point: float = 0.0
    exhaustion_point: float = 0.0
    ceding_percentage: float = 0.0
    ceded_limit: float = 0.0

    premium_per_year: float = 0.0
    rate_on_line_used: float = 0.0

    expected_annual_payout: float = 0.0
    pure_rate_on_line: float = 0.0
    implied_cost_multiple: float = float("nan")

    attachment_return_period: float = float("inf")
    curve_rate_on_line: float = float("nan")
    curve_premium_per_year: float = float("nan")

    # Payout behaviour, carried so the report can state which benchmark was
    # applied. Under binary payout with one payout per year the pure ROL is
    # not merely bounded by 1/RP, it EQUALS the annual trigger probability.
    payout_mode: str = "proportional"
    one_payout_per_year: bool = False
    benchmark_pure_rol: float = float("nan")
    # Return period at which the policy actually triggers, recovered from the
    # payouts. Distinct from attachment_return_period, which is measured on
    # annual aggregates and overstates trigger frequency.
    effective_trigger_return_period: float = float("nan")

    warnings: List[str] = field(default_factory=list)

    @property
    def is_coherent(self) -> bool:
        """True when no warning was raised."""
        return not self.warnings


def diagnose_insurance_pricing(
    payouts_matrix: np.ndarray,
    losses_matrix: np.ndarray,
    attachment_point: float,
    exhaustion_point: float,
    ceding_percentage: float,
    premium_per_year: float,
    rate_on_line_used: float,
    instrument_name: str = "Insurance",
    multiple_range: tuple = DEFAULT_MULTIPLE_RANGE,
    rp_anchors: Optional[Sequence[float]] = None,
    rol_anchors: Optional[Sequence[float]] = None,
    payout_mode: str = "proportional",
    one_payout_per_year: bool = False,
) -> InsurancePricingDiagnostic:
    """
    Check whether an insurance premium is coherent with the layer it covers.

    The expected annual payout is measured directly from the simulated
    payout matrix, so the diagnostic reflects the layer as the model
    actually applies it, including any per-event reinstatement behaviour.

    This function never raises on an implausible price. It records warnings
    and returns, so the analysis continues and the analyst decides.

    Parameters
    ----------
    payouts_matrix : ndarray
        Insurance payouts arranged as (simulations x years).
    losses_matrix : ndarray
        Losses arranged as (simulations x years), used for the return period.
    attachment_point, exhaustion_point, ceding_percentage : float
        Layer definition.
    premium_per_year : float
        Annual premium actually charged in the analysis.
    rate_on_line_used : float
        ROL that generated that premium, for reporting.
    instrument_name : str
        Label used in the report.
    multiple_range : tuple of float
        Lower and upper bounds of the plausible implied cost multiple.
    rp_anchors, rol_anchors : sequence of float, optional
        Override the market pricing curve.

    Returns
    -------
    InsurancePricingDiagnostic
    """
    payouts = np.asarray(payouts_matrix, dtype=float)
    ceded_limit = (exhaustion_point - attachment_point) * ceding_percentage

    diag = InsurancePricingDiagnostic(
        instrument_name=instrument_name,
        attachment_point=attachment_point,
        exhaustion_point=exhaustion_point,
        ceding_percentage=ceding_percentage,
        ceded_limit=ceded_limit,
        premium_per_year=premium_per_year,
        rate_on_line_used=rate_on_line_used,
    )

    # Expected annual payout across every simulated year.
    diag.expected_annual_payout = float(payouts.mean()) if payouts.size else 0.0

    if ceded_limit > 0:
        diag.pure_rate_on_line = diag.expected_annual_payout / ceded_limit

    if diag.expected_annual_payout > 0:
        diag.implied_cost_multiple = premium_per_year / diag.expected_annual_payout

    diag.payout_mode = payout_mode
    diag.one_payout_per_year = one_payout_per_year

    # Remoteness of the attachment point and the rate the curve would charge.
    diag.attachment_return_period = attachment_return_period(
        losses_matrix, attachment_point
    )

    # Actuarial benchmark for the rate.
    #
    # In general the expected loss of a layer attaching at return period T
    # cannot exceed limit / T, because the layer pays nothing unless it
    # attaches (probability 1/T) and pays at most the limit when it does.
    # That gives an upper bound on the pure rate but not its value, because
    # a proportional layer usually pays far less than the limit when it is
    # touched.
    #
    # Under binary payout with a single payout per policy year the bound is
    # attained exactly: whenever the policy triggers it pays the limit, and
    # it triggers at most once, so
    #
    #     pure ROL = P(trigger in a year) = 1 / RP(attachment)
    #
    # This matters for reading broker indication curves. Such curves are
    # quoted as rate against the attachment return period and are calibrated
    # on THIN layers, where touching the attachment is nearly equivalent to
    # exhausting the limit. Applying them to a wide proportional layer
    # implies implausible cost multiples; applying them to a binary policy
    # is exactly the case they were calibrated for.
    #
    # IMPORTANT MEASUREMENT NOTE. ``attachment_return_period`` above is
    # computed on ANNUAL AGGREGATE losses, because that is the granularity
    # the CBA module receives. The insurance trigger, however, is applied
    # EVENT by EVENT upstream. A year whose events sum past the attachment
    # point without any single event reaching it will be counted as an
    # exceedance by the annual measure but will not trigger the policy. The
    # annual figure therefore overstates how often the policy actually pays
    # and must not be used as the benchmark.
    #
    # Under binary payout with one payout per year the correct figure can be
    # recovered exactly from the payouts themselves, with no assumption at
    # all: every triggering year pays the limit and no year pays twice, so
    #
    #     P(trigger) = expected annual payout / ceded limit = pure ROL
    #
    # and the effective trigger return period is its reciprocal. This is
    # both the benchmark rate and the return period at which the market
    # curve should be queried.
    if diag.payout_mode == 'binary' and diag.one_payout_per_year:
        diag.benchmark_pure_rol = diag.pure_rate_on_line
        if diag.pure_rate_on_line > 0:
            diag.effective_trigger_return_period = 1.0 / diag.pure_rate_on_line
    elif np.isfinite(diag.attachment_return_period) and diag.attachment_return_period > 0:
        # Proportional payout: 1/RP is only an UPPER BOUND on the pure rate,
        # not its value, because a touched layer usually pays well below the
        # limit. Recorded as a bound, not a benchmark.
        diag.benchmark_pure_rol = float("nan")
    rp_for_curve = (diag.effective_trigger_return_period
                    if np.isfinite(diag.effective_trigger_return_period)
                    else diag.attachment_return_period)
    if np.isfinite(rp_for_curve):
        diag.curve_rate_on_line = float(
            rol_from_return_period(
                rp_for_curve,
                rp_anchors=rp_anchors,
                rol_anchors=rol_anchors,
            )
        )
        diag.curve_premium_per_year = diag.curve_rate_on_line * ceded_limit

    # --- Warnings -------------------------------------------------
    lo, hi = multiple_range

    if diag.expected_annual_payout <= 0:
        diag.warnings.append(
            "The layer never pays in any simulated year. The premium is a pure "
            "cost and the benefit-cost ratio of this instrument is zero. Review "
            "the attachment point against the loss curve."
        )
    elif diag.implied_cost_multiple < 1.0:
        diag.warnings.append(
            f"The implied cost multiple is {diag.implied_cost_multiple:.3f}, "
            f"below 1.0. The premium is smaller than the expected payout of the "
            f"layer, which means the insurer would be selling below expected "
            f"loss. No commercial market prices this way. The benefit-cost "
            f"ratio of this instrument is overstated by roughly "
            f"{(1.0 / diag.implied_cost_multiple):.1f} times."
        )
    elif diag.implied_cost_multiple < lo:
        diag.warnings.append(
            f"The implied cost multiple is {diag.implied_cost_multiple:.3f}, "
            f"below the plausible lower bound of {lo:.2f}. The premium covers "
            f"expected loss but leaves little for expenses, capital and margin."
        )
    elif diag.implied_cost_multiple > hi:
        diag.warnings.append(
            f"The implied cost multiple is {diag.implied_cost_multiple:.3f}, "
            f"above the plausible upper bound of {hi:.2f}. The premium is high "
            f"relative to the expected loss of the layer. Verify the layer "
            f"definition, in particular its thickness."
        )

    if np.isfinite(diag.attachment_return_period) and diag.attachment_return_period < 5.0:
        diag.warnings.append(
            f"The attachment point is exceeded roughly once every "
            f"{diag.attachment_return_period:.1f} years. This is a working "
            f"layer, not a catastrophe layer. Rates quoted for remote layers "
            f"do not apply at this frequency."
        )

    return diag


def format_pricing_diagnostic(diag: InsurancePricingDiagnostic) -> List[str]:
    """Render a diagnostic as report lines."""
    rp = diag.attachment_return_period
    rp_txt = "never exceeded" if not np.isfinite(rp) else f"{rp:.1f} years"
    mult_txt = ("n/a" if not np.isfinite(diag.implied_cost_multiple)
                else f"{diag.implied_cost_multiple:.3f}")
    curve_txt = ("n/a" if not np.isfinite(diag.curve_rate_on_line)
                 else f"{diag.curve_rate_on_line * 100:.1f}%")

    mode_txt = diag.payout_mode
    if diag.one_payout_per_year:
        mode_txt += ", one payout per policy year"

    lines = [
        f"  {diag.instrument_name} — layer ${diag.attachment_point:,.0f}M "
        f"to ${diag.exhaustion_point:,.0f}M, ceding "
        f"{diag.ceding_percentage * 100:.1f}%",
        f"    Payout rule:                    {mode_txt}",
        f"    Ceded limit:                    ${diag.ceded_limit:,.2f}M",
        f"    Annual premium charged:         ${diag.premium_per_year:,.3f}M "
        f"(ROL {diag.rate_on_line_used * 100:.1f}%)",
        f"    Expected annual payout:         ${diag.expected_annual_payout:,.3f}M",
        f"    Pure ROL (expected loss/limit): {diag.pure_rate_on_line * 100:.1f}%",
        f"    Implied cost multiple:          {mult_txt}",
        f"    Attachment return period:       {rp_txt}",
        f"    ROL implied by market curve:    {curve_txt}",
    ]

    # Under binary payout with one payout per year the pure rate is not just
    # bounded by 1/RP, it equals it, so the benchmark can be stated exactly.
    if (diag.payout_mode == 'binary' and diag.one_payout_per_year
            and np.isfinite(diag.benchmark_pure_rol)):
        lines.append(
            f"    Effective trigger return period:"
            f" {diag.effective_trigger_return_period:.1f} years "
            f"(event level, recovered from payouts)"
        )
        lines.append(
            f"    Benchmark pure ROL:             "
            f"{diag.benchmark_pure_rol * 100:.1f}%  "
            f"(exact under this payout rule)"
        )
        if diag.expected_annual_payout > 0:
            bc = 1.10 / diag.implied_cost_multiple
            lines.append(
                f"    Implied B/C at f_B = 10%:       {bc:.3f}"
            )
    if diag.warnings:
        lines.append("    WARNINGS:")
        for w in diag.warnings:
            lines.append(f"      - {w}")
    else:
        lines.append("    No coherence warnings.")
    return lines
