# -*- coding: utf-8 -*-
"""
insurance_layer.py — LEC Tool
==============================
Places instrument thresholds on the loss curve of the country being analysed
and prices the parametric insurance layer.

Why this module exists
----------------------
A threshold in dollars means nothing on its own: the same $50M is an event
that happens every other year in one country and once a generation in
another. Up to v7 every threshold was typed in dollars, calibrated once for
the Honduras example, and silently reused for any other catalogue. In v8 a
threshold can be given as a return period and its dollar value is derived
from the curve of the country being run, so the same configuration is
meaningful for every country in ``Country databases``.

The insurance premium has the same problem. A single Rate-on-Line applied to
every layer ignores how remote the layer is. v8 prices the layer from its
position on the loss curve.

Pipeline position
-----------------
main.py calls ``resolve_instruments`` after the LEC is computed and before
the strategy is evaluated. It returns the instrument dicts with every
threshold in dollars, the ceding percentage, the premium and the payout
floor filled in, so risk_management and cba.engine keep receiving the same
keys they always did.

Return periods here are EVENT return periods: the reciprocal of the annual
rate at which a single event exceeds the loss, which is what the LEC curve
(column 1) holds and what the simulation samples from. Instrument triggers
are evaluated event by event, so this is the consistent scale.

Pricing methods
---------------
``rol_rule`` (default)
    The layer is cut into thin slices. Each slice is priced at its own
    return period: slices hit more often than ``cutoff_rp`` use the market
    ROL curve; more remote slices use the flat rate ``flat_rol``. The
    premium is the sum over slices. The default cutoff (1 in 10 years) and
    flat rate (5%) are consistent with the pricing structure of sovereign
    risk-pooling facilities; replace them with the country's actual
    quotation whenever available (pricing = 'quote'). With the default
    layer (1 in 15 to 1 in 50 years) every slice lies above the cutoff, so
    the rule is equivalent to a flat ROL of ``flat_rol``.
``market_curve``
    Every slice priced on the market ROL curve (commercial reinsurance).
``quote``
    The premium of an actual quotation (``gross_premium``) is used as is.
``fixed_rol``
    Legacy behaviour: premium = rate_on_line x coverage limit.

The market ROL curve and both rule parameters are inputs in config.toml
([insurance_pricing]); the curve carries a date because reinsurance prices
are volatile and seasonal.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np

from cba.pricing import (
    rol_from_return_period, DEFAULT_RP_ANCHORS, DEFAULT_ROL_ANCHORS,
)

PRICING_METHODS = ('rol_rule', 'market_curve', 'quote', 'fixed_rol')

# Number of slices used to integrate a layer. The result is stable (changes
# below 0.1%) from about 100 slices; 400 leaves a wide margin.
N_SLICES = 400


class LayerError(ValueError):
    """A threshold or layer cannot be placed on the curve of this country."""


# ---------------------------------------------------------------------------
# Curve lookups
# ---------------------------------------------------------------------------

def _curve_arrays(lec_curve):
    """Loss ascending, rate descending, strictly positive values only."""
    loss = np.asarray(lec_curve[:, 0], dtype=float)
    rate = np.asarray(lec_curve[:, 1], dtype=float)
    keep = (loss > 0) & (rate > 0)
    loss, rate = loss[keep], rate[keep]
    order = np.argsort(loss)
    loss, rate = loss[order], rate[order]
    if loss.size < 2:
        raise LayerError("The loss curve has fewer than two positive points.")
    return loss, rate


def loss_at_return_period(lec_curve, return_period: float):
    """
    Loss exceeded, on average, once every ``return_period`` years by a single
    event. Log-log interpolation on the LEC curve.

    Returns (loss, in_range). ``in_range`` is False when the requested rate
    lies outside the curve; the loss is then held at the curve end.
    """
    loss, rate = _curve_arrays(lec_curve)
    target = 1.0 / float(return_period)
    # np.interp needs an increasing x: use rate ascending (loss descending).
    x = np.log(rate[::-1])
    y = np.log(loss[::-1])
    in_range = bool(rate.min() <= target <= rate.max())
    value = float(np.exp(np.interp(np.log(target), x, y)))
    return value, in_range


def return_period_at_loss(lec_curve, losses):
    """
    Event return period (years) of one or several loss levels.

    Log-log interpolation inside the curve. Above the largest loss the last
    segment is extrapolated in log-log space, so remote slices of a layer
    that extends beyond the curve keep becoming rarer instead of being held
    at the curve's smallest rate (which would overstate how often they are
    hit). Below the smallest loss the rate is held at the curve maximum.
    """
    loss, rate = _curve_arrays(lec_curve)
    q = np.atleast_1d(np.asarray(losses, dtype=float))
    lx, ly = np.log(loss), np.log(rate)
    out = np.exp(np.interp(np.log(np.maximum(q, loss[0])), lx, ly))
    above = q > loss[-1]
    if np.any(above):
        slope = (ly[-1] - ly[-2]) / (lx[-1] - lx[-2])
        slope = min(slope, -1e-6)          # rate must keep falling
        out[above] = np.exp(ly[-1] + slope * (np.log(q[above]) - lx[-1]))
    rp = 1.0 / out
    return float(rp[0]) if np.ndim(losses) == 0 else rp


# ---------------------------------------------------------------------------
# Pricing
# ---------------------------------------------------------------------------

@dataclass
class PricingSettings:
    """Contents of [insurance_pricing] in config.toml."""
    method: str = 'rol_rule'
    cutoff_rp: float = 10.0
    flat_rol: float = 0.05
    rate_on_line: float = 0.05
    curve_rp: Sequence[float] = field(default_factory=lambda: list(DEFAULT_RP_ANCHORS))
    curve_rol: Sequence[float] = field(default_factory=lambda: list(DEFAULT_ROL_ANCHORS))
    curve_date: str = '2026-08'
    payout_mode: str = 'proportional'
    one_payout_per_year: bool = True
    payout_floor: bool = True


def price_layer(lec_curve, attachment, exhaustion, ceding, method, s: PricingSettings):
    """
    Annual gross premium ($MM) of the layer [attachment, exhaustion] with the
    given ceding share, under a pricing method other than 'quote'.
    """
    limit = ceding * (exhaustion - attachment)
    if method == 'fixed_rol':
        return float(s.rate_on_line * limit)
    if method not in ('rol_rule', 'market_curve'):
        raise LayerError(f"price_layer: unsupported method '{method}'.")
    edges = np.linspace(attachment, exhaustion, N_SLICES + 1)
    mids = 0.5 * (edges[:-1] + edges[1:])
    rp = return_period_at_loss(lec_curve, mids)
    rol = np.asarray(rol_from_return_period(rp, rp_anchors=s.curve_rp,
                                            rol_anchors=s.curve_rol), dtype=float)
    if method == 'rol_rule':
        rol = np.where(rp < s.cutoff_rp, rol, s.flat_rol)
    return float(np.sum(rol * ceding * np.diff(edges)))


# ---------------------------------------------------------------------------
# Resolution of the instrument list
# ---------------------------------------------------------------------------

@dataclass
class LayerSummary:
    """What the reports print about each resolved insurance layer."""
    name: str
    attachment: float
    exhaustion: float
    attachment_rp: float
    exhaustion_rp: float
    ceding: float
    coverage_limit: float
    method: str
    gross_premium: float
    donor_discount: float
    net_premium: float
    effective_rol: float
    payout_mode: str
    one_payout_per_year: bool
    minimum_payout: float
    curve_date: Optional[str]
    reference_premiums: dict          # method -> premium, for comparison
    warnings: list


def _threshold(inst, dollar_key, rp_key, lec_curve, name, warnings):
    """Dollar threshold from either '<dollar_key>' or '<rp_key>'."""
    if rp_key in inst:
        rp = float(inst[rp_key])
        value, in_range = loss_at_return_period(lec_curve, rp)
        if not in_range:
            warnings.append(
                f"{rp_key} = {rp:g} years lies outside the loss curve of this country; "
                f"the threshold is held at the curve end ({value:,.1f} $MM)."
            )
        return value
    return float(inst[dollar_key])


def resolve_instruments(instruments, lec_curve, settings: PricingSettings):
    """
    Return (resolved_instruments, layer_summaries, notes).

    ``instruments`` are the dicts from config.toml (with the PPO schedule
    already attached). The returned dicts carry every threshold in dollars
    plus, for insurance, ceding_percentage, premium, donor_discount,
    minimum_payout, payout_mode and one_payout_per_year. ``notes`` lists
    every warning as '<instrument>: <message>'. Inputs are not modified.
    """
    resolved, summaries, notes = [], [], []
    for inst in instruments:
        cfg = copy.deepcopy(inst)
        itype = cfg['type']
        name = cfg.get('name', itype)
        warnings = []

        if itype == 'ddo' and 'ddo_threshold_rp' in cfg:
            cfg['ddo_threshold'] = _threshold(cfg, 'ddo_threshold', 'ddo_threshold_rp',
                                              lec_curve, name, warnings)

        if itype == 'ppo' and 'ppo_loss_trigger_rp' in cfg:
            cfg['ppo_loss_trigger'] = _threshold(cfg, 'ppo_loss_trigger', 'ppo_loss_trigger_rp',
                                                 lec_curve, name, warnings)
            cfg.setdefault('ppo_trigger_mode', 'loss')

        if itype == 'insurance':
            ap = _threshold(cfg, 'attachment_point', 'attachment_rp', lec_curve, name, warnings)
            ep = _threshold(cfg, 'exhaustion_point', 'exhaustion_rp', lec_curve, name, warnings)
            if not ep > ap:
                raise LayerError(
                    f"Instrument '{name}': the exhaustion point ({ep:,.1f} $MM) must be "
                    f"above the attachment point ({ap:,.1f} $MM) on this country's curve."
                )
            width = ep - ap
            if 'coverage_limit' in cfg:
                limit = float(cfg['coverage_limit'])
                ceding = limit / width
                if ceding > 1.0:
                    raise LayerError(
                        f"Instrument '{name}': coverage_limit ({limit:,.2f} $MM) is larger than "
                        f"the layer width ({width:,.2f} $MM) on this country's curve. "
                        f"Lower the limit or widen the layer."
                    )
            else:
                ceding = float(cfg['ceding_percentage'])
                limit = ceding * width
            if width < 0.02 * ap:
                warnings.append(
                    f"The layer is very narrow on this country's curve "
                    f"({ap:,.1f} to {ep:,.1f} $MM); the curve is almost flat between the "
                    f"two return periods. Check the attachment and exhaustion choice."
                )

            # --- pricing ---
            quoted = 'gross_premium' in cfg or 'premium' in cfg
            method = cfg.get('pricing', 'quote' if quoted else settings.method)
            if method not in PRICING_METHODS:
                raise LayerError(
                    f"Instrument '{name}': pricing '{method}' is not recognised. "
                    f"Use one of: {', '.join(PRICING_METHODS)}."
                )
            if method == 'quote':
                if not quoted:
                    raise LayerError(
                        f"Instrument '{name}': pricing = 'quote' requires gross_premium "
                        f"(the gross annual premium of the quotation, $MM)."
                    )
                gross = float(cfg.get('gross_premium', cfg.get('premium')))
            else:
                if quoted:
                    raise LayerError(
                        f"Instrument '{name}': gross_premium is given but pricing = '{method}'. "
                        f"Remove gross_premium or set pricing = 'quote'."
                    )
                local = copy.copy(settings)
                if 'rate_on_line' in cfg:
                    local.rate_on_line = float(cfg['rate_on_line'])
                gross = price_layer(lec_curve, ap, ep, ceding, method, local)

            if 'donor_discount' in cfg and 'donor_discount_share' in cfg:
                raise LayerError(
                    f"Instrument '{name}': give donor_discount ($MM) or donor_discount_share (0-1), not both."
                )
            if 'donor_discount_share' in cfg:
                discount = gross * float(cfg['donor_discount_share'])
            else:
                discount = float(cfg.get('donor_discount', 0.0))
            if not 0.0 <= discount <= gross:
                raise LayerError(
                    f"Instrument '{name}': the donor discount ({discount:,.3f} $MM) must be "
                    f"between 0 and the gross premium ({gross:,.3f} $MM)."
                )

            payout_mode = cfg.get('payout_mode', settings.payout_mode)
            per_year = bool(cfg.get('one_payout_per_year', settings.one_payout_per_year))
            floor = bool(cfg.get('payout_floor', settings.payout_floor))
            minimum = min(gross, limit) if floor else 0.0

            refs = {}
            for m in ('rol_rule', 'market_curve'):
                refs[m] = price_layer(lec_curve, ap, ep, ceding, m, settings)

            cfg.update({
                'attachment_point': ap,
                'exhaustion_point': ep,
                'ceding_percentage': ceding,
                'premium': gross,
                'donor_discount': discount,
                'minimum_payout': minimum,
                'payout_mode': payout_mode,
                'one_payout_per_year': per_year,
                'pricing': method,
            })
            summaries.append(LayerSummary(
                name=name, attachment=ap, exhaustion=ep,
                attachment_rp=return_period_at_loss(lec_curve, ap),
                exhaustion_rp=return_period_at_loss(lec_curve, ep),
                ceding=ceding, coverage_limit=limit, method=method,
                gross_premium=gross, donor_discount=discount, net_premium=gross - discount,
                effective_rol=gross / limit if limit > 0 else float('nan'),
                payout_mode=payout_mode, one_payout_per_year=per_year, minimum_payout=minimum,
                curve_date=settings.curve_date if method in ('rol_rule', 'market_curve') else None,
                reference_premiums=refs, warnings=warnings,
            ))
        for w in warnings:
            notes.append(f"{name}: {w}")
        resolved.append(cfg)
    return resolved, summaries, notes
