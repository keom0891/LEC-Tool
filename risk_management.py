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
from utils import aggregate_event_values_by_year, validate_insurance_layer


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
    validate_insurance_layer(attachment_point, exhaustion_point, ceding_percentage)
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


def apply_ppo_coverage(values, ppo_available, ccf_applied):
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
    ppo_applied = []
    ppo_triggered = False

    for i, loss in enumerate(values):
        if ccf_applied[i] > 0 and not ppo_triggered:
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


def ddo_coverage(values, ddo_threshold, initial_balance):
    """Draw eligible event losses from an initial balance, without a per-event cap.

    This standalone helper has no renewal schedule. apply_strategy handles
    calendar renewals, exhaustion replenishments and the combined payout cap.

    >>> ddo_coverage([50, 130, 20, 200], ddo_threshold=100, initial_balance=300)
    [0.0, 130, 0.0, 170]
    """
    remaining = initial_balance
    applied = []
    for loss in values:
        payout = min(loss, remaining) if loss > ddo_threshold else 0.0
        applied.append(payout)
        remaining -= payout
    return applied


# ---------------------------------------------------------------------------
# Strategy pipeline
# ---------------------------------------------------------------------------

class _CreditBalance:
    """Independent balance and renewal schedule for one line in one simulation."""

    def __init__(self, cfg, first_year):
        self.cfg = cfg
        self.balance = float(cfg['initial_balance'])
        self.used = 0
        self.next_renewal = cfg.get('first_renewal_year')
        if self.next_renewal is not None and self.next_renewal < first_year:
            # Preserve the supplied starting balance; historical dates only anchor the schedule.
            interval = cfg['renewal_interval_years']
            self.next_renewal += ((first_year - self.next_renewal + interval - 1)
                                  // interval) * interval

    def renew(self, year):
        if self.next_renewal is not None and year >= self.next_renewal:
            # Advance even across years without disasters; overwrite partial balances.
            interval = self.cfg['renewal_interval_years']
            elapsed = (year - self.next_renewal) // interval + 1
            self.next_renewal += elapsed * interval
            self.balance = float(self.cfg['contracted_amount'])
            self.used = 0

    def can_replenish(self):
        return (self.next_renewal is not None
                and self.used < self.cfg['max_replenishments_per_period']
                and self.cfg['contracted_amount'] > 0)

    def capacity(self):
        return self.balance

    def spend(self, amount):
        first_draw = min(amount, self.balance)
        self.balance = max(0.0, self.balance - first_draw)
        if self.balance <= 0 and self.can_replenish():
            self.balance = float(self.cfg['contracted_amount'])
            self.used += 1


def apply_strategy(event_catalogue, drm_configs, catalogue_length, first_year=None):
    """Evaluate events chronologically, renewing lines at calendar-year boundaries.

    CCF/DDO start with initial_balance, renew to contracted_amount, and allow
    one exhaustion replenishment per period. Each event can draw only its
    starting balance; replenishment is available from the next event, even
    within the same year. A zero starting balance is replenished after the
    first event, subject to the same limit. Loan has inclusive calendar
    validity and no replenishment. Balances consume only final, proportionally
    capped payouts. Each simulation owns fresh independent instrument balances.

    balance_dfs maps line names to end-of-year balances (including renewals in
    years without events). Existing payout_dfs/total_coverage outputs are unchanged.
    The standalone coverage functions above retain their original APIs.
    """
    if first_year is None:
        from datetime import datetime
        first_year = datetime.now().year + 1
    columns = [f'Year_{i + 1}' for i in range(catalogue_length)]
    payout_lists = [[] for _ in drm_configs]
    balance_lists = {cfg['name']: [] for cfg in drm_configs
                     if cfg['type'] in ('ccf', 'ddo', 'loan')}
    ccf_indices = [j for j, cfg in enumerate(drm_configs) if cfg['type'] == 'ccf']
    if any(cfg['type'] == 'ppo' for cfg in drm_configs) and not ccf_indices:
        raise ValueError("PPO requires a CCF in the same strategy.")

    for catalogue in event_catalogue:
        times = np.asarray(catalogue['times'])
        losses = np.asarray(catalogue['losses'])
        states = {j: _CreditBalance(cfg, first_year) for j, cfg in enumerate(drm_configs)
                  if cfg['type'] in ('ccf', 'ddo', 'loan')}
        ppo_used = set()
        annual = np.zeros((len(drm_configs), catalogue_length))
        balances = {j: [] for j in states}
        cursor = 0
        for y in range(catalogue_length):
            year = first_year + y
            for state in states.values():
                state.renew(year)
            while cursor < len(times) and times[cursor] < y + 1:
                loss = float(losses[cursor])
                payouts = np.zeros(len(drm_configs))
                for j, cfg in enumerate(drm_configs):
                    kind = cfg['type']
                    if kind == 'insurance':
                        payouts[j] = standard_insurance_payout(
                            [loss], cfg['attachment_point'], cfg['exhaustion_point'],
                            cfg['ceding_percentage'])[0]
                    elif kind == 'ccf':
                        threshold = max(cfg['ccf_threshold'],
                                        0.01 * cfg['Pop_exposed'] * cfg['ccf_person'] / 1e6)
                        if loss > threshold:
                            requested = (np.exp(0.001074 * np.log(loss * 1000) ** 3.0883
                                                + 7.9346) * cfg['ccf_person'] / 1e6)
                            payouts[j] = min(requested, states[j].capacity())
                    elif kind == 'ddo':
                        if loss > cfg['ddo_threshold']:
                            payouts[j] = min(loss, states[j].capacity())
                    elif kind == 'loan':
                        if cfg['start_year'] <= year <= cfg['end_year'] and loss > cfg['loan_threshold']:
                            payouts[j] = min(loss, states[j].balance)
                    elif kind != 'ppo':
                        raise ValueError(f"Unknown instrument type '{kind}'.")
                # PPO activation remains tied to the first CCF, independent of list order.
                for j, cfg in enumerate(drm_configs):
                    if cfg['type'] == 'ppo' and j not in ppo_used and payouts[ccf_indices[0]] > 0:
                        payouts[j] = cfg['ppo_schedule'][y]
                        ppo_used.add(j)
                total = payouts.sum()
                if total > loss:
                    payouts *= loss / total
                for j, state in states.items():
                    state.spend(float(payouts[j]))
                annual[:, y] += payouts
                cursor += 1
            for j, state in states.items():
                balances[j].append(state.balance)
        for j in range(len(drm_configs)):
            payout_lists[j].append(annual[j])
        for j in states:
            balance_lists[drm_configs[j]['name']].append(balances[j])

    payout_dfs = [pd.DataFrame(rows, columns=columns) for rows in payout_lists]
    total_coverage = (sum(payout_dfs) if payout_dfs else
                      pd.DataFrame(np.zeros((len(event_catalogue), catalogue_length)), columns=columns))
    return {
        'payout_dfs': payout_dfs,
        'total_coverage': total_coverage,
        'balance_dfs': {name: pd.DataFrame(rows, columns=columns)
                        for name, rows in balance_lists.items()},
    }
