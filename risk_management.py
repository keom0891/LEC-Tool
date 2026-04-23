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

# TODO: DDO (Deferred Drawdown Option) — to be implemented when the
#       instrument specification is finalised.

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
                               ceding_percentage=0.5):
    """
    Apply parametric or indemnity insurance coverage to a sequence of event losses.

    Coverage resets for every individual event (no aggregate limit across events).
    The insurer pays ``ceding_percentage`` of the loss within the layer
    [attachment_point, exhaustion_point].

    Parameters
    ----------
    values : array_like
        Event loss amounts ($MM), in chronological order.
    attachment_point : float, default 15
        Loss level ($MM) at which coverage begins.
    exhaustion_point : float, default 50
        Loss level ($MM) at which coverage is fully exhausted.
    ceding_percentage : float, default 0.5
        Share of losses covered within the layer (0 – 1).

    Returns
    -------
    list of float
        Insurance payout for each event ($MM).  Same length as *values*.

    Examples
    --------
    >>> standard_insurance_payout([10, 30, 100], attachment_point=20,
    ...                            exhaustion_point=80, ceding_percentage=1.0)
    [0, 10, 60]
    """
    payouts = []
    for loss in values:
        if loss <= attachment_point:
            payout = 0.0
        elif loss >= exhaustion_point:
            payout = (exhaustion_point - attachment_point) * ceding_percentage
        else:
            payout = (loss - attachment_point) * ceding_percentage
        payouts.append(payout)
    return payouts


def apply_ppo_coverage(values, ppo_available, ppo_loss_trigger):
    """
    Apply a Pre-arranged Parametric Option (PPO) to a sequence of event losses.

    The PPO activates at most once per catalogue — at the first event that
    exceeds ``ppo_loss_trigger``.  Subsequent events receive no payout even
    if they also exceed the trigger.

    Parameters
    ----------
    values : array_like
        Event loss amounts ($MM), in chronological order.
    ppo_available : array_like
        Maximum PPO payout available at the time of each event ($MM).
        Must be the same length as *values*.  Typically a step function
        that ramps up as credit is drawn down over years.
    ppo_loss_trigger : float
        Loss threshold ($MM) that must be exceeded to activate the PPO.

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
    ppo_applied = []
    ppo_triggered = False

    for i, loss in enumerate(values):
        if loss > ppo_loss_trigger and not ppo_triggered:
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

    # TODO: DDO — a deferred drawdown variant of this instrument is
    #       planned; the damage function above may be shared.
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

        type 'ppo':
          ppo_schedule        list of float, length catalogue_length
              Available PPO payout for each year index ($MM).
          ppo_loss_trigger    float  ($MM)

        type 'ccf':
          ccf_maximum         float  ($MM)
          ccf_person          float  ($ per person)
          Pop_exposed         float  (number of people)

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
    loss net of prior instruments.  The strategy net loss is therefore:
        net_loss = gross_loss − total_coverage
    """
    simulation_number = len(event_catalogue)
    column_names = [f'Year_{i + 1}' for i in range(catalogue_length)]

    # Preallocate one accumulation list per instrument
    n_instruments = len(drm_configs)
    payout_lists = [[] for _ in range(n_instruments)]

    for i in range(simulation_number):
        times_i = event_catalogue[i]['times']
        losses_i = event_catalogue[i]['losses']
        loss_list = losses_i.tolist()

        for j, cfg in enumerate(drm_configs):
            instrument_type = cfg['type']

            if instrument_type == 'insurance':
                payouts = np.array(standard_insurance_payout(
                    loss_list,
                    attachment_point=cfg['attachment_point'],
                    exhaustion_point=cfg['exhaustion_point'],
                    ceding_percentage=cfg['ceding_percentage'],
                ))

            elif instrument_type == 'ppo':
                if len(times_i) == 0:
                    payouts = np.array([])
                else:
                    event_years = np.floor(times_i).astype(int)
                    schedule = cfg['ppo_schedule']
                    ppo_available_event = [
                        schedule[y] if 0 <= y < len(schedule) else 0.0
                        for y in event_years
                    ]
                    payouts = np.array(apply_ppo_coverage(
                        loss_list,
                        ppo_available=ppo_available_event,
                        ppo_loss_trigger=cfg['ppo_loss_trigger'],
                    ))

            elif instrument_type == 'ccf':
                payouts = np.array(ccf_coverage(
                    loss_list,
                    ccf_maximum=cfg['ccf_maximum'],
                    ccf_person=cfg['ccf_person'],
                    Pop_exposed=cfg['Pop_exposed'],
                ))

            else:
                raise ValueError(
                    f"Unknown instrument type '{instrument_type}'. "
                    "Supported: 'insurance', 'ppo', 'ccf'."
                )

            payout_lists[j].append(
                aggregate_event_values_by_year(times_i, payouts, catalogue_length)
            )

    payout_dfs = [
        pd.DataFrame(payout_lists[j], columns=column_names)
        for j in range(n_instruments)
    ]

    total_coverage = sum(payout_dfs)

    return {
        'payout_dfs': payout_dfs,
        'total_coverage': total_coverage,
    }