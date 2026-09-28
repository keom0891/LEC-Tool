# -*- coding: utf-8 -*-
"""
reporting.py — LEC Tool
========================
Numerical summaries and the main text report of a run.

Functions
---------
strategy_statistics         Median horizon totals (loss, coverage, retention, ...) for one scenario.
statistics_table            Rows -> DataFrame in the layout of <run_id>_statistics.csv.
gap_thresholds              Absolute financing-gap thresholds from fractions of the maximum LEC loss.
gap_probabilities           P(gap > threshold) over simulations.
pml_by_return_period        PML interpolated on the LEC curve for each return period.
horizon_loss_statistics     Mean/median/percentiles of the cumulative horizon loss.
drr_payback_years           Median years until loss reduction repays the DRR investment.
build_main_report           Assemble the text report.

All functions are pure except build_main_report, which only formats text.
"""

import textwrap
from datetime import datetime

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def strategy_statistics(label, annual_df, coverage_df, resp_fiscal, horizon):
    """
    Median totals over the horizon for one scenario (same definitions as the
    historical statistics CSV of the tool).

    retention  : loss in years with no payout at all.
    uncovered  : loss net of payouts in years with some payout.
    """
    loss_med = np.round(annual_df.sum(axis=1).median(), 1)
    cov_med = np.round(coverage_df.sum(axis=1).median(), 1)
    ret_med = np.round(annual_df[coverage_df == 0].sum(axis=1).median(), 1)
    uncov_med = np.round(
        (annual_df[coverage_df != 0].sum(axis=1) - coverage_df.sum(axis=1)).median(), 1)
    return {
        'Scenario': label,
        f'Total {horizon}-year loss median ($MM)': loss_med,
        f'Total {horizon}-year coverage median ($MM)': cov_med,
        f'Total {horizon}-year retention median ($MM)': ret_med,
        f'Total {horizon}-year uncovered median ($MM)': uncov_med,
        f'Total {horizon}-year FISCAL retention median ($MM)': np.round(resp_fiscal * ret_med, 1),
        f'Total {horizon}-year FISCAL uncovered median ($MM)': np.round(resp_fiscal * uncov_med, 1),
    }


def statistics_table(rows):
    """Statistics rows -> DataFrame with one column per scenario."""
    return pd.DataFrame(rows).set_index('Scenario').T


def gap_thresholds(max_loss, fractions):
    """Absolute thresholds ($MM) = fraction x maximum loss on the LEC curve."""
    return [(float(f), float(f) * float(max_loss)) for f in fractions]


def gap_probabilities(annual_df, coverage_df, thresholds):
    """
    Financing gap per simulation = sum over the horizon of (loss - coverage),
    undiscounted. Returns {threshold: P(gap > threshold)}.
    """
    gap = (annual_df - coverage_df).clip(lower=0).sum(axis=1).to_numpy(dtype=float)
    return {thr: float(np.mean(gap > thr)) for _, thr in thresholds}


def pml_by_return_period(lec_curve, return_periods):
    """
    PML for each return period by interpolating loss vs exceedance rate on the
    LEC curve. Returns {rp: (loss, in_range)}; in_range is False when 1/rp is
    rarer than the curve's smallest rate (the value is then the curve maximum).
    """
    rates = lec_curve[:, 1][::-1]
    losses = lec_curve[:, 0][::-1]
    out = {}
    for rp in return_periods:
        target = 1.0 / float(rp)
        loss = float(np.interp(target, rates, losses))
        out[rp] = (loss, bool(target >= rates.min()))
    return out


def horizon_loss_statistics(annual_df):
    """Statistics of the cumulative loss over the horizon (one value per simulation)."""
    totals = annual_df.sum(axis=1)
    return {
        'mean': float(totals.mean()),
        'median': float(totals.median()),
        'p05': float(totals.quantile(0.05)),
        'p95': float(totals.quantile(0.95)),
        'max': float(totals.max()),
        'zero_count': int((totals == 0).sum()),
        'n': int(len(totals)),
    }


def drr_payback_years(annual_df, annual_red_df, investment):
    """
    Median (over simulations) number of years in which the cumulative loss
    reduction is still below the total investment, i.e. years to pay back the
    investment with avoided losses (undiscounted).
    """
    total_inv = float(np.sum(investment))
    cum_saving = (annual_df - annual_red_df).cumsum(axis=1)
    return float((cum_saving < total_inv).sum(axis=1).median())


# ---------------------------------------------------------------------------
# Text report
# ---------------------------------------------------------------------------

_W = 72


def _money(x):
    # Small amounts (premiums, limits) keep two decimals so they stay readable.
    return f'${x:,.2f} MM' if abs(x) < 10 else f'${x:,.1f} MM'


def _rule(char='-'):
    return char * _W


def _section(title):
    return ['', _rule('-'), title, _rule('-')]


def _notes(items, indent=2):
    """
    Footnotes under a block: items is a list of (marker, text). Each text is
    wrapped and indented under its marker so the report reads without the
    manual.
    """
    if not items:
        return []
    width = max(len(m) for m, _ in items) + 1
    pad = ' ' * indent
    out = ['']
    for marker, text in items:
        wrapped = textwrap.wrap(text, width=_W - indent - width - 1, break_on_hyphens=False) or ['']
        out.append(f"{pad}{marker:<{width}} {wrapped[0]}")
        out += [f"{pad}{'':<{width}} {w}" for w in wrapped[1:]]
    return out


def _table(header, rows, first_col_width=34, col_width=20):
    """Simple fixed-width table: header is a list of column titles."""
    lines = [f"  {header[0]:<{first_col_width}}" + ''.join(f"{h:>{col_width}}" for h in header[1:])]
    for row in rows:
        lines.append(f"  {row[0]:<{first_col_width}}" + ''.join(f"{c:>{col_width}}" for c in row[1:]))
    return lines


def build_main_report(ctx):
    """
    Assemble the main text report.

    ctx is a dict with the keys documented in main.py (write_outputs); optional
    keys ('drr', 'cba') may be None when a stage did not run.
    """
    cfg = ctx['cfg']
    sim = cfg.simulation
    horizon = sim.catalogue_length
    L = []

    # --- Header -------------------------------------------------------------
    L += [_rule('='), 'LEC TOOL - RESULTS REPORT', _rule('=')]
    L += [
        f"Run id:                      {cfg.run.id}",
        f"Generated:                   {datetime.now():%Y-%m-%d %H:%M}",
        f"Configuration file:          {cfg.source_path}",
        f"Event losses:                {cfg.inputs.event_loss_file.name} "
        f"({ctx['n_events']} events, {ctx['year_min']}-{ctx['year_max']})",
        f"Probabilistic tail:          "
        + (f"{cfg.inputs.tail_curve_file.name} (hybrid curve)" if cfg.lec.hybrid_curve
           else "not used (empirical curve only)"),
        f"Loss / frequency scale:      {cfg.lec.loss_scale_factor:g} / {cfg.lec.freq_scale_factor:g}",
        f"Simulation:                  {sim.simulation_number} catalogues x {horizon} years, "
        f"seed = {sim.random_seed if sim.random_seed is not None else 'random'}",
        f"Fiscal responsibility share: {cfg.fiscal.resp_fiscal:.0%}",
    ]

    # --- LEC ----------------------------------------------------------------
    lec = ctx['lec']
    L += _section('LOSS EXCEEDANCE CURVE')
    L.append(f"  AAL, empirical curve*:             {_money(lec['aal_empirical'])}")
    if cfg.lec.hybrid_curve:
        L.append(f"  AAL, hybrid curve (used below)**:  {_money(lec['aal'])}")
    L.append(f"  Maximum loss on the curve:         {_money(lec['max_loss'])}")
    L.append(f"  Maximum historical event loss:     {_money(lec['max_event_loss'])}")
    for rp, (loss, in_range) in lec['pml'].items():
        note = '' if in_range else '  (beyond the curve range: capped at the curve maximum)'
        label = f"  PML, return period {rp:g} years***:"
        L.append(f"{label:<37}{_money(loss)}{note}")
    lec_notes = [('*', "Average annual loss (AAL): the loss expected per year on average, "
                       "computed from the historical event record only.")]
    if cfg.lec.hybrid_curve:
        lec_notes.append(('**', "AAL after adding the probabilistic tail, which extends the curve "
                                "to rare events not seen in the historical record. All results "
                                "below use this curve."))
    lec_notes.append(('***', "Probable maximum loss (PML): the event loss exceeded on average "
                             "once in the stated number of years."))
    L += _notes(lec_notes)

    # --- Strategy -----------------------------------------------------------
    L += _section('FINANCING STRATEGY')
    if ctx['instruments']:
        L.append("  Instruments: " + ', '.join(f"{i['name']} ({i['type']})" for i in ctx['instruments']))
    else:
        L.append("  No instruments declared: losses are fully retained.")
    L += _insurance_layer_lines(ctx.get('layers'))
    L.append('')
    scenarios = ctx['scenarios']          # list of (column label, stats_row)
    header = ['Median over simulations'] + [s[0] for s in scenarios]
    metric_keys = [k for k in scenarios[0][1] if k != 'Scenario']
    rows = []
    marks = {'coverage': '*', 'retention': '**', 'uncovered': '***',
             'FISCAL retention': '+', 'FISCAL uncovered': '+'}
    for key in metric_keys:
        label = key.replace(f'Total {horizon}-year ', '').replace(' median ($MM)', '')
        rows.append([f"{horizon}-year {label}{marks.get(label, '')} ($MM)"]
                    + [f"{s[1][key]:,.1f}" for s in scenarios])
    L += _table(header, rows)
    strat_notes = [
        ('*', "Total paid by all instruments."),
        ('**', "Loss in years in which no instrument paid anything."),
        ('***', "Loss left unpaid in years in which at least one instrument paid."),
        ('+', "The same amounts multiplied by the fiscal responsibility share."),
    ]
    L += _notes(strat_notes)
    L.append("  Each row is a separate median, so the rows do not add up.")
    if len(scenarios) > 1:
        L.append("  Reduced (ex-ante): the same simulations after the risk-reduction investment.")

    # --- Financing gap ------------------------------------------------------
    L += _section('FINANCING GAP PROBABILITIES')
    L.append(f"  Thresholds are fractions of the maximum loss on the LEC curve ({_money(lec['max_loss'])}).")
    L.append('')
    header = ['P(gap > threshold)*'] + [g[0] for g in ctx['gaps']]
    rows = []
    for frac, thr in ctx['thresholds']:
        rows.append([f"{frac:.0%} of max loss ({_money(thr)})"] + [f"{g[1][thr]:.1%}" for g in ctx['gaps']])
    L += _table(header, rows)
    L += _notes([('*', f"Gap: all loss left unpaid over the {horizon}-year horizon (retention "
                       "plus uncovered), undiscounted. Each cell is the share of simulations "
                       "in which the gap exceeds the threshold.")])

    # --- DRR ----------------------------------------------------------------
    drr = ctx.get('drr')
    if drr is not None:
        L += _section('EX-ANTE RISK REDUCTION (DRR)')
        L.append(f"  Total investment:                           {_money(drr['total_investment'])}")
        L.append(f"  Annualisation rate of benefits*:            {cfg.risk_reduction.discount_rate:.1%}")
        L.append(f"  Cumulative AAL reduction, final year**:     {_money(drr['reduction'][-1])}")
        L.append(f"  Median years to pay back the investment***: {drr['payback_years']:.0f} (undiscounted)")
        L.append('')
        L.append("  Year   Investment ($MM)   AAL reduction in place ($MM)")
        for y, inv, red in zip(ctx['year_labels'], cfg.risk_reduction.investment, drr['reduction']):
            L.append(f"  {y:<6} {inv:>16,.1f}   {red:>28,.2f}")
        L += _notes([
            ('*', "Rate used to convert each investment's benefit (investment x its "
                  "benefit-cost ratio) into a constant annual reduction of the AAL over "
                  "its useful life. It should be the rate at which the benefit-cost ratio "
                  "of the investment was estimated."),
            ('**', "Reduction of the average annual loss in place in the last year, from "
                   "all investments made before it."),
            ('***', "Median number of years until the cumulative avoided loss equals the "
                    "total investment."),
        ])

    # --- Horizon losses -----------------------------------------------------
    h = ctx['horizon_stats']
    L += _section(f'CUMULATIVE LOSS OVER {horizon} YEARS (base catalogue)')
    L.append(f"  Mean:                       {_money(h['mean'])}")
    L.append(f"  Median*:                    {_money(h['median'])}")
    L.append(f"  5th / 95th percentile**:    {_money(h['p05'])} / {_money(h['p95'])}")
    L.append(f"  Maximum:                    {_money(h['max'])}")
    L.append(f"  Simulations with zero loss: {h['zero_count']} of {h['n']}")
    L += _notes([
        ('*', f"Half of the simulated {horizon}-year periods lose less than this, half "
              "more. When it is well below the mean, a few periods with extreme events "
              "are pulling the mean up."),
        ('**', f"Nine out of ten simulated {horizon}-year periods fall between these values."),
    ])

    # --- CBA headline -------------------------------------------------------
    cba = ctx.get('cba')
    if cba is not None:
        c = cba['results'].core
        L += _section('COST-BENEFIT ANALYSIS (summary)')
        L.append(f"  Loss basis:                    {cfg.cba.loss_basis}")
        L.append(f"  Expected B/C ratio:            {c.expected_bc:.3f}")
        L.append(f"  P(B/C > 1):                    {c.prob_bc_gt_1:.1%}")
        L.append(f"  Expected unpaid loss (PV):     {_money(c.expected_unpaid_loss)}")
        bul = 'n/a' if not np.isfinite(c.expected_bul) else f"{c.expected_bul:.3f}"
        L.append(f"  Expected B/UL ratio:           {bul}")
        cnc = getattr(cba['results'], 'cnc', None)
        if cnc is not None:
            L.append(f"  CNC expected net saving (PV):  {_money(cnc.expected_net_saving)} "
                     f"(median % saving {cnc.median_pct_saving:.1%}, "
                     f"P(ex-ante cheaper) = {cnc.prob_positive_saving:.1%})")
        d = getattr(cba['results'], 'drr', None)
        if d is not None:
            L.append(f"  DRR B/C direct / indirect:     {d.bc_direct:.3f} / {d.bc_indirect:.3f}")
        for name, a in getattr(cba['results'], 'insurance_analysis', {}).items():
            L.append(f"  {name}: B/C economic {a['bc_economic']:.3f}, fiscal {a['bc_fiscal']:.3f}; "
                     f"unpaid loss p99 {_money(a['unpaid_p99_without'])} -> {_money(a['unpaid_p99_with'])}")
            if a['implied_multiple'] < 1.0:
                L.append(f"  WARNING: {name} premium is below the layer's expected payout "
                         f"(multiple {a['implied_multiple']:.2f}); see the CBA report.")
        L.append(f"  Full report:                   {cba['report_file'].name}")

    # --- Files --------------------------------------------------------------
    L += _section('OUTPUT FILES')
    for p in ctx['files']:
        L.append(f"  {p}")
    L += ['', _rule('='), 'END OF REPORT', _rule('=')]
    return '\n'.join(L)


_METHOD_LABEL = {
    'rol_rule': 'layered rule (market curve below the cutoff, flat rate above)',
    'market_curve': 'commercial reinsurance curve',
    'quote': 'quotation entered by the user',
    'fixed_rol': 'fixed Rate-on-Line',
}


def _insurance_layer_lines(layers):
    """Insurance layers placed on this country's curve (v8)."""
    if not layers or not layers.get('summaries') and not layers.get('notes'):
        return []
    L = []
    for s in layers.get('summaries', []):
        L += ['', f"  Insurance '{s.name}' on this country's loss curve:"]
        L.append(f"    Attachment*:       {_money(s.attachment)}  (event return period 1 in {s.attachment_rp:,.1f} years)")
        L.append(f"    Exhaustion*:       {_money(s.exhaustion)}  (1 in {s.exhaustion_rp:,.1f} years)")
        L.append(f"    Coverage limit:    {_money(s.coverage_limit)}  (ceding {s.ceding:.3%} of the layer)")
        rule = f"{s.payout_mode}" + (", one payout per year" if s.one_payout_per_year else ", per event")
        if s.minimum_payout > 0:
            rule += f", minimum payout {_money(s.minimum_payout)}"
        L.append(f"    Payout rule**:     {rule}")
        L.append(f"    Premium source:    {_METHOD_LABEL.get(s.method, s.method)}"
                 + (f", curve dated {s.curve_date}" if s.curve_date else ""))
        L.append(f"    Gross premium:     {_money(s.gross_premium)}  (effective ROL*** {s.effective_rol:.1%})")
        if s.donor_discount > 0:
            L.append(f"    Donor discount:    {_money(s.donor_discount)}  -> premium paid by the government {_money(s.net_premium)}")
        refs = ', '.join(f"{k} {_money(v)}" for k, v in s.reference_premiums.items() if k != s.method)
        if refs:
            L.append(f"    For reference:     {refs}")
        rule_note = ("Proportional: the payout grows with the loss between attachment and "
                     "exhaustion. Binary: the full limit is paid once the attachment is "
                     "reached." if s.payout_mode == 'proportional' else
                     "Binary: the full limit is paid once the attachment is reached. "
                     "Proportional: the payout grows with the loss between attachment and "
                     "exhaustion.")
        if s.minimum_payout > 0:
            rule_note += " Minimum payout: when the policy pays, it pays at least the gross premium."
        L += _notes([
            ('*', "Event losses at which the policy starts and stops paying, shown in "
                  "dollars and as return periods on this country's curve."),
            ('**', rule_note),
            ('***', "Rate on line: premium / coverage limit."),
        ], indent=4)
    for n in layers.get('notes', []):
        L.append(f"  WARNING: {n}")
    return L
