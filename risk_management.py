# -*- coding: utf-8 -*-
"""
drm.py — LEC Tool
==================
Disaster Risk Management (DRM) financial instruments.

Each instrument is implemented as a stateless payout function that maps a
sequence of event losses (in chronological order within a single simulation)
to a sequence of payouts.  A strategy-level function applies multiple
instruments in combination across all simulations.

All functions are pure (no file I/O, no plotting).

Instruments implemented
-----------------------
standard_insurance_payout   Parametric/indemnity coverage with attachment and exhaustion.
apply_ppo_coverage          Contingent credit (PPO) activated once per catalogue.
ccf_coverage                Contingent credit facility (CCF) with cumulative depletion.
ddo_coverage                Deferred Drawdown Option (DDO) with a loss trigger threshold.

Pipeline function
-----------------
apply_strategy  Apply a set of DRM instruments to every simulation in a catalogue.
"""

import numpy as np
import pandas as pd
from utils import aggregate_event_values_by_year


# ---------------------------------------------------------------------------
# Instrument functions
# ---------------------------------------------------------------------------

def standard_insurance_payout(values, attachment_point=15, exhaustion_point=50,
                               ceding_percentage=0.5,
                               payout_mode='proportional',
                               one_payout_per_year=False,
                               event_years=None,
                               minimum_payout=0.0):
    """
    Apply parametric insurance coverage to a sequence of event losses.

    Two independent policy features are configurable: how much the policy
    pays when it triggers (*payout_mode*) and how often it may trigger
    within a policy year (*one_payout_per_year*). The defaults reproduce the
    behaviour of earlier versions exactly.

    Payout modes
    ------------
    ``'proportional'`` (default)
        The insurer pays ``ceding_percentage`` of the loss lying inside the
        layer [attachment_point, exhaustion_point]:

            payout = min(max(L - AP, 0), EP - AP) * ceding_percentage

        A loss just above the attachment point produces a small payout; a
        loss at or beyond the exhaustion point produces the full coverage
        limit. This is how commercial CCRIF policies actually work, and it
        is why the exhaustion point and the ceding percentage both appear in
        a real quotation: together they define the coverage limit, and the
        distance between attachment and exhaustion defines the ramp.

    ``'binary'``
        Any loss above the attachment point pays the full coverage limit:

            payout = (EP - AP) * ceding_percentage   if L > AP, else 0

        Nothing is prorated. This is a teaching simplification, adopted for
        the worked example so that the mechanism can be explained and drawn
        without a ramp: cross the threshold, collect the cover. It should
        not be presented as the contractual behaviour of a real policy.

    Coverage reset
    --------------
    ``one_payout_per_year=False`` (default)
        Coverage resets for every individual event. A year with three
        qualifying events produces three payouts, each up to the full limit,
        with no additional premium. This is the behaviour of earlier
        versions of this codebase.

    ``one_payout_per_year=True``
        The policy pays at most once per policy year. Once it has paid, it
        is spent for the remainder of that year even if the payout was well
        below the coverage limit; cover is restored only at renewal, when
        the next annual premium is paid. This is a conservative
        approximation of the annual aggregate limit of sovereign parametric
        policies, under which a second event in the same year still pays
        while part of the limit remains. Requires *event_years*.

    Why both features matter, and why they must be decided together
    ---------------------------------------------------------------
    The two features push the expected payout in OPPOSITE directions, so
    adopting one without the other gives a misleading picture. Measured on
    the worked example's historical catalogue (Honduras hurricane, 1974-2024,
    layer $50M-$190M, ceding 6.6%, coverage limit $9.24M):

        Rule                                  Expected payout   vs default
        ----------------------------------    ---------------   ----------
        per event  + proportional (default)      $2.138M/yr         ---
        per event  + binary                      $4.711M/yr        +120%
        one/year   + proportional                $1.038M/yr         -51%
        one/year   + binary                      $2.718M/yr         +27%

    Of the 26 events that generate a payout over the 51-year record, 11 are
    the second or later qualifying event within the same calendar year. That
    concentration is what makes the reset rule worth half the expected
    payout.

    Consequence for pricing coherence
    ---------------------------------
    Under binary payout with one payout per year, the actuarially pure
    Rate-on-Line collapses to the annual probability that the policy
    triggers, which is the reciprocal of the attachment point's return
    period:

        pure ROL = P(trigger in a year) = 1 / RP(attachment)

    On the worked example that is 15 triggering years out of 51, or 29.4 per
    cent, against a return period of 3.4 years. This resolves an
    inconsistency that appeared while calibrating the premium: a broker
    indication curve quoted as ROL against attachment return period implied
    implausible cost multiples of 4 to 8 when applied to a wide proportional
    layer, because such curves are quoted for THIN layers, where touching
    the attachment point is nearly equivalent to exhausting the limit.
    Binary payout is exactly that limiting case, and under it the same
    curve implies a cost multiple of about 1.26, which is commercially
    plausible.

    It also yields a closed form for the benefit-cost ratio of insurance
    that mirrors the one for credit instruments:

        B/C = (1 + f_B) / cost multiple

    where the cost multiple is premium over expected payout. At an
    actuarially fair premium the ratio equals (1 + f_B) exactly. The
    benefit-cost ratio of an insurance instrument therefore measures the
    loading paid over expected loss, just as the ratio for a concessional
    loan measures the subsidy received. Neither ratio measures protection,
    which is why neither should be the headline indicator for judging
    whether cover is worth buying.

    Parameters
    ----------
    values : array_like
        Event loss amounts ($MM), in chronological order.
    attachment_point : float, default 15
        Loss level ($MM) at which coverage begins.
    exhaustion_point : float, default 50
        Loss level ($MM) at which coverage is fully exhausted.
    ceding_percentage : float, default 0.5
        Share of losses covered within the layer (0 - 1).
    payout_mode : {'proportional', 'binary'}, default 'proportional'
        See above.
    one_payout_per_year : bool, default False
        See above. Requires *event_years*.
    event_years : array_like of int, optional
        Policy year index of each event, same length as *values*. Only
        consulted when *one_payout_per_year* is True. The worked example
        uses calendar years for clarity; in practice CCRIF policies renew
        on 1 June, ahead of the Atlantic hurricane season, so a policy year
        straddles two calendar years. For a hurricane-only catalogue the
        two conventions rarely differ, because the season falls inside a
        single calendar year, but the choice should be stated explicitly
        whenever the peril is not seasonal.
    minimum_payout : float, default 0.0
        Payout floor ($MM) for any event that triggers the policy. Sovereign
        parametric pools pay at least the annual gross premium whenever the
        policy is triggered, even when the proportional ramp would give
        less. The floor never exceeds the coverage limit and has no effect
        under the binary rule, which already pays the full limit.

    Returns
    -------
    list of float
        Insurance payout for each event ($MM). Same length as *values*.

    Examples
    --------
    >>> standard_insurance_payout([10, 30, 100], attachment_point=20,
    ...                            exhaustion_point=80, ceding_percentage=1.0)
    [0, 10, 60]

    >>> standard_insurance_payout([10, 30, 100], attachment_point=20,
    ...                            exhaustion_point=80, ceding_percentage=1.0,
    ...                            payout_mode='binary')
    [0, 60, 60]

    >>> standard_insurance_payout([30, 100], attachment_point=20,
    ...                            exhaustion_point=80, ceding_percentage=1.0,
    ...                            payout_mode='binary', one_payout_per_year=True,
    ...                            event_years=[0, 0])
    [60, 0]
    """
    if payout_mode not in ('proportional', 'binary'):
        raise ValueError(
            f"standard_insurance_payout: payout_mode must be 'proportional' "
            f"or 'binary', got {payout_mode!r}."
        )
    if one_payout_per_year and event_years is None:
        raise ValueError(
            "standard_insurance_payout: one_payout_per_year=True requires "
            "event_years, the policy year index of each event. Without it "
            "the function cannot tell which events share a policy year."
        )
    if one_payout_per_year and len(event_years) != len(values):
        raise ValueError(
            f"standard_insurance_payout: event_years has length "
            f"{len(event_years)} but values has length {len(values)}. They "
            f"must correspond element by element."
        )

    limit = (exhaustion_point - attachment_point) * ceding_percentage
    payouts = []
    years_already_paid = set()

    for i, loss in enumerate(values):
        if loss <= attachment_point:
            payouts.append(0.0)
            continue

        # The policy is spent for this year if it has already paid, however
        # small that earlier payout was. Cover returns only at renewal.
        if one_payout_per_year:
            year = event_years[i]
            if year in years_already_paid:
                payouts.append(0.0)
                continue

        if payout_mode == 'binary':
            payout = limit
        elif loss >= exhaustion_point:
            payout = limit
        else:
            payout = (loss - attachment_point) * ceding_percentage
            if minimum_payout > 0:
                payout = min(max(payout, minimum_payout), limit)

        payouts.append(payout)
        if one_payout_per_year and payout > 0:
            years_already_paid.add(event_years[i])

    return payouts


def apply_ppo_coverage(values, ppo_available, ccf_applied,
                       ppo_loss_trigger=None,
                       trigger_mode='ccf',
                       require_available_funds=False):
    """
    Apply a Pre-arranged Parametric Option (PPO) to a sequence of event losses.

    The PPO activates at most once per catalogue.  Two activation rules are
    supported, selected with *trigger_mode*.

    ``trigger_mode='loss'``
        The PPO activates at the first event whose loss exceeds
        ``ppo_loss_trigger``.  This is the instrument's own threshold and
        makes the PPO independent of the other instruments in the strategy.

    ``trigger_mode='ccf'``
        The PPO activates at the first event that also activates the CCF.
        This reproduces the behaviour of the merged BID codebase and is the
        default so existing configurations are unaffected.  Because the CCF
        activates on small, frequent events, this rule tends to consume the
        PPO's single activation early in the horizon, while the available
        amount is still ramping up.

    In both modes, *require_available_funds* prevents the single activation
    from being consumed by an event at which no funds are available.  Under
    the original merged code the PPO could trigger against an available
    amount of zero, disburse nothing, and stay permanently disabled for the
    rest of the horizon.

    Parameters
    ----------
    values : array_like
        Event loss amounts ($MM), in chronological order.
    ppo_available : array_like
        Maximum PPO payout available at the time of each event ($MM).
        Must be the same length as *values*.  Typically a step function
        that ramps up as credit is drawn down over years.
    ccf_applied : float
        CCF payout that must be exceeded to activate the PPO ($MM).

    Returns
    -------
    list of float
        PPO payout for each event ($MM).  At most one non-zero entry.
        Same length as *values*.

    Notes
    -----
    *ppo_available* is indexed by event position within the chronological
    sequence, not by calendar year.  The caller is responsible for mapping
    each event to its year and looking up the correct available amount
    (see ``apply_strategy``).
    """
    if trigger_mode not in ('ccf', 'loss'):
        raise ValueError(
            f"apply_ppo_coverage: trigger_mode must be 'ccf' or 'loss', got "
            f"{trigger_mode!r}."
        )
    if trigger_mode == 'loss' and ppo_loss_trigger is None:
        raise ValueError(
            "apply_ppo_coverage: trigger_mode='loss' requires ppo_loss_trigger "
            "to be set. Provide the loss threshold ($MM) above which the PPO "
            "activates."
        )

    ppo_applied = []
    ppo_triggered = False

    for i, loss in enumerate(values):
        if ppo_triggered:
            ppo_applied.append(0.0)
            continue

        if trigger_mode == 'loss':
            condition_met = loss > ppo_loss_trigger
        else:
            condition_met = ccf_applied[i] > 0

        # An activation that disburses nothing still consumes the single
        # trigger.  Unless disabled, skip such events and keep the PPO
        # available for a later event that does have funds.
        if condition_met and require_available_funds and ppo_available[i] <= 0:
            condition_met = False

        if condition_met:
            ppo_applied.append(ppo_available[i])
            ppo_triggered = True
        else:
            ppo_applied.append(0.0)

    return ppo_applied


def ccf_coverage(values, ccf_maximum, ccf_person, Pop_exposed):
    """
    Apply a Contingent Credit Facility (CCF) to a sequence of event losses.

    The CCF provides a payout per affected person based on an empirical
    damage function.  The facility has a hard cap (``ccf_maximum``) that
    depletes cumulatively across events within the catalogue — there is no
    annual replenishment.

    Parameters
    ----------
    values : array_like
        Event loss amounts ($MM), in chronological order.
    ccf_maximum : float
        Total CCF coverage available for the full catalogue period ($MM).
    ccf_person : float
        Payout per affected person ($, not $MM).
    Pop_exposed : float
        Total population exposed to the hazard.

    Returns
    -------
    list of float
        CCF payout for each event ($MM).  Cumulative sum never exceeds
        ``ccf_maximum``.  Same length as *values*.

    Notes
    -----
    The number of affected persons is estimated from event loss via an
    empirical power-law damage function calibrated to the region:

        affected(L) = exp(0.001074 · ln(L·1000)^3.0883 + 7.9346)

    A minimum loss threshold is applied:
        loss < 0.01 · Pop_exposed · ccf_person / 1e6  → payout = 0
    """
    def _affected_persons(loss):
        return np.exp(0.001074 * np.log(loss * 1000) ** 3.0883 + 7.9346)

    min_loss_threshold = 0.01 * Pop_exposed * ccf_person / 1e6

    ccf_applied = []
    remaining_cover = ccf_maximum

    for loss in values:
        if loss < min_loss_threshold or remaining_cover <= 0:
            ccf_applied.append(0.0)
        else:
            payout_raw = _affected_persons(loss) * ccf_person / 1e6
            ccf_app = min(payout_raw, remaining_cover)
            remaining_cover -= ccf_app
            ccf_applied.append(ccf_app)

    return ccf_applied


def ddo_coverage(values, ddo_threshold, ddo_available):
    """
    Apply a Deferred Drawdown Option (DDO) to a sequence of event losses.

    Triggers on every event whose loss exceeds ``ddo_threshold`` and pays a
    fixed amount ``ddo_available``.  Unlike the PPO, the DDO resets each event
    (no single-activation limit).  A local cap ensures the payout for any
    single event never exceeds the event loss itself; the broader constraint
    that the *sum* of all instrument payouts cannot exceed the gross loss is
    enforced by ``apply_strategy``.

    Parameters
    ----------
    values : array_like
        Event loss amounts ($MM), in chronological order.
    ddo_threshold : float
        Loss level ($MM) that must be exceeded to trigger the DDO.
    ddo_available : float
        Fixed DDO payout ($MM) issued each time the trigger is met.

    Returns
    -------
    list of float
        DDO payout for each event ($MM).  Same length as *values*.

    Examples
    --------
    >>> ddo_coverage([50, 130, 20, 200], ddo_threshold=100, ddo_available=110)
    [0.0, 110.0, 0.0, 110.0]
    """
    applied = []
    for loss in values:
        if loss > ddo_threshold:
            applied.append(min(ddo_available, loss))
        else:
            applied.append(0.0)
    return applied


# ---------------------------------------------------------------------------
# Strategy pipeline
# ---------------------------------------------------------------------------

def apply_strategy(event_catalogue, drm_configs, catalogue_length):
    """
    Apply a combination of DRM instruments to every simulation in a catalogue.

    Iterates over all simulations, applies each configured instrument
    event-by-event in chronological order, then aggregates payouts to
    annual totals.

    Parameters
    ----------
    event_catalogue : list of dicts, length simulation_number
        As returned by ``simulation.generate_synthetic_catalogue`` in the
        'event_catalogue' key.  Each element has:
          - 'times'  : ndarray of fractional event times.
          - 'losses' : ndarray of event-level losses ($MM).

    drm_configs : list of dicts
        One dict per instrument, evaluated in list order.  Each dict must
        contain a key 'type' selecting the instrument, plus instrument-
        specific parameters:

        type 'insurance':
          attachment_point    float  ($MM)
          exhaustion_point    float  ($MM)
          ceding_percentage   float  (0–1)
          payout_mode         str, optional, default 'proportional'
              'proportional' pays the ceded share of the loss inside the
                  layer, which is how a real policy behaves.
              'binary' pays the full coverage limit on any loss above the
                  attachment point. Teaching simplification.
          one_payout_per_year bool, optional, default False
              When True the policy pays at most once per policy year and
              cover is restored only at renewal.
          See standard_insurance_payout for why these two options must be
          decided together: they move the expected payout in opposite
          directions.

        type 'ppo':
          ppo_schedule        list of float, length catalogue_length
              Available PPO payout for each year index ($MM).
          ppo_loss_trigger    float  ($MM), optional
              Own loss threshold of the PPO.  Required when
              ppo_trigger_mode is 'loss'; ignored otherwise.
          ppo_trigger_mode    str, optional, default 'ccf'
              'ccf'  activates with the CCF (merged-codebase behaviour).
              'loss' activates at the first event exceeding
                     ppo_loss_trigger, independently of other instruments.
          ppo_require_available_funds  bool, optional, default False
              When True, an event with zero available PPO funds does not
              consume the single activation.

        type 'ccf':
          ccf_maximum         float  ($MM)
          ccf_person          float  ($ per person)
          Pop_exposed         float  (number of people)

        type 'ddo':
          ddo_threshold       float  ($MM)
          ddo_available       float  ($MM)

    catalogue_length : int
        Number of years per simulation.  Must match the catalogue.

    Returns
    -------
    dict with the following keys:

    payout_dfs     list of pandas.DataFrame
        payout_dfs[j] is a DataFrame of shape (simulation_number, catalogue_length)
        with annual payouts from instrument j.  Column names: Year_1 … Year_N.
        One entry per element in *drm_configs*.

    total_coverage   pandas.DataFrame, shape (simulation_number, catalogue_length)
        Sum of all instrument payouts per year per simulation.

    Notes
    -----
    For the PPO instrument, *ppo_schedule* is indexed by calendar year
    within the catalogue (year 0 = first year).  The available amount for
    each event is looked up from the year the event falls in.

    Instruments are independent: each sees the original gross loss, not the
    loss net of prior instruments.  After all per-event payouts are computed,
    a proportional cap is applied so that the total payout for any single event
    never exceeds the gross event loss.  The strategy net loss is therefore:
        net_loss = gross_loss − total_coverage  (≥ 0 by construction)
    """
    simulation_number = len(event_catalogue)
    column_names = [f'Year_{i + 1}' for i in range(catalogue_length)]

    n_instruments = len(drm_configs)
    payout_lists = [[] for _ in range(n_instruments)]

    for i in range(simulation_number):
        times_i = event_catalogue[i]['times']
        losses_i = event_catalogue[i]['losses']
        loss_list = losses_i.tolist()
        
        # --- Compute CCF once per simulation if it exists ---
        ccf_trigger = None
        ccf_cfg = next((cfg for cfg in drm_configs if cfg['type'] == 'ccf'), None)

        if ccf_cfg is not None:
            ccf_trigger = np.array(ccf_coverage(
                loss_list,
                ccf_maximum=ccf_cfg['ccf_maximum'],
                ccf_person=ccf_cfg['ccf_person'],
                Pop_exposed=ccf_cfg['Pop_exposed'],
            ))
            
        # --- compute per-event payouts for every instrument ---
        event_payouts = []
        for cfg in drm_configs:
            instrument_type = cfg['type']

            if instrument_type == 'insurance':
                # Policy-year index of each event. Events are timed in
                # fractional years from the start of the horizon, so the
                # integer part is the year the event falls in. Only used
                # when the policy pays at most once per year.
                ins_years = (np.floor(times_i).astype(int).tolist()
                             if len(times_i) else [])
                p = np.array(standard_insurance_payout(
                    loss_list,
                    attachment_point=cfg['attachment_point'],
                    exhaustion_point=cfg['exhaustion_point'],
                    ceding_percentage=cfg['ceding_percentage'],
                    payout_mode=cfg.get('payout_mode', 'proportional'),
                    one_payout_per_year=cfg.get('one_payout_per_year', False),
                    event_years=ins_years,
                    minimum_payout=cfg.get('minimum_payout', 0.0),
                ))

            elif instrument_type == 'ppo':
                if ccf_trigger is None:
                    raise ValueError(
                        "PPO depends on CCF activation, but no 'ccf' "
                        "configuration was found in drm_configs."
                    )

                if len(times_i) == 0:
                    p = np.array([])
                else:
                    event_years = np.floor(times_i).astype(int)
                    schedule = cfg['ppo_schedule']

                    ppo_available_event = [
                        schedule[y] if 0 <= y < len(schedule) else 0.0
                        for y in event_years
                    ]

                    p = np.array(apply_ppo_coverage(
                        loss_list,
                        ppo_available=ppo_available_event,
                        ccf_applied=ccf_trigger,
                        ppo_loss_trigger=cfg.get('ppo_loss_trigger'),
                        trigger_mode=cfg.get('ppo_trigger_mode', 'ccf'),
                        require_available_funds=cfg.get(
                            'ppo_require_available_funds', False),
                    ))

            elif instrument_type == 'ccf':
                p = ccf_trigger.copy()

            elif instrument_type == 'ddo':
                p = np.array(ddo_coverage(
                    loss_list,
                    ddo_threshold=cfg['ddo_threshold'],
                    ddo_available=cfg['ddo_available'],
                ))

            else:
                raise ValueError(
                    f"Unknown instrument type '{instrument_type}'. "
                    "Supported: 'insurance', 'ppo', 'ccf', 'ddo'."
                )

            event_payouts.append(p)

        # --- cap total per-event payout to gross event loss ---
        if len(losses_i) > 0 and n_instruments > 0:
            total_p = sum(event_payouts)
            # scale factor per event: 1.0 where total ≤ loss, < 1.0 otherwise
            scale = np.minimum(np.divide(losses_i, total_p, out=np.ones_like(losses_i, dtype=float), where=total_p > 0), 1.0)
            event_payouts = [p * scale for p in event_payouts]

        # --- aggregate each instrument's payouts to annual totals ---
        for j, p in enumerate(event_payouts):
            payout_lists[j].append(
                aggregate_event_values_by_year(times_i, p, catalogue_length)
            )

    payout_dfs = [
        pd.DataFrame(payout_lists[j], columns=column_names)
        for j in range(n_instruments)
    ]

    if payout_dfs:
        total_coverage = sum(payout_dfs)
    else:
        # No instruments: coverage is identically zero (sum([]) would be the int 0).
        total_coverage = pd.DataFrame(
            np.zeros((simulation_number, catalogue_length)), columns=column_names
        )

    return {
        'payout_dfs': payout_dfs,
        'total_coverage': total_coverage,
    }