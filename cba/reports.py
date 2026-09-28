"""
LEC-CBA Reports Module
========================
Generates formatted results: text reports and matplotlib visualizations.
Aligned to BID instruments (CCRIF, PPO, CCF) with DRR comparative table.
"""

import textwrap

import numpy as np
from typing import Optional


def _notes(items, indent=2, width=70):
    """
    Footnotes under a block: items is a list of (marker, text). The text is
    wrapped and indented under its marker so the report reads on its own.
    """
    if not items:
        return []
    mw = max(len(m) for m, _ in items) + 1
    pad = ' ' * indent
    out = [""]
    for marker, text in items:
        wrapped = textwrap.wrap(text, width=width - indent - mw - 1, break_on_hyphens=False) or ['']
        out.append(f"{pad}{marker:<{mw}} {wrapped[0]}")
        out += [f"{pad}{'':<{mw}} {w}" for w in wrapped[1:]]
    return out


def _format_config_summary(results) -> str:
    """
    Build the configuration summary from the resolved per-instrument configs
    stored in results, so it reflects actual drm_configs parameters and
    all instrument instances (including multiple DDOs).
    """
    cfg = results.config
    lines = [
        "=" * 60,
        "LEC-CBA Configuration Summary",
        "=" * 60,
        f"Analysis horizon: {results.losses_matrix.shape[1]} years",
        f"Simulations: {results.losses_matrix.shape[0]}",
        f"Loss basis: {getattr(results, 'loss_basis', 'total')} losses"
        + (f" (fiscal share {results.resp_fiscal:.0%} applied)"
           if getattr(results, 'loss_basis', 'total') == 'fiscal' else ""),
        f"Discount rate: {cfg.discount.social_discount_rate:.1%}",
        f"Indirect benefit factor: {cfg.indirect_benefit.factor:.1%}",
        f"OMV lambda: {cfg.omv.lambda_risk_adjustment}",
        "",
    ]

    for inst_name, inst_cfg in results.resolved_instrument_configs.items():
        itype = results.instrument_types.get(inst_name, '')

        if itype == 'insurance':
            lines += [
                f"--- {inst_name} (Parametric Insurance) ---",
                f"  Layer: ${inst_cfg.attachment_point:.0f}M"
                f" - ${inst_cfg.exhaustion_point:.0f}M",
                f"  Ceding: {inst_cfg.ceding_percentage:.3%}",
                f"  Coverage limit: ${(inst_cfg.exhaustion_point - inst_cfg.attachment_point) * inst_cfg.ceding_percentage:.2f}M",
                f"  Payout rule: {getattr(inst_cfg, 'payout_mode', 'proportional')}"
                + (", one payout per year" if getattr(inst_cfg, 'one_payout_per_year', False) else ", per event"),
                f"  Pricing: {getattr(inst_cfg, 'pricing_method', 'fixed_rol')}",
                f"  Annual premium (gross): ${inst_cfg.premium:.3f}M",
            ] + ([
                f"  Donor discount: ${inst_cfg.donor_discount:.3f}M"
                f" -> government pays ${inst_cfg.premium - inst_cfg.donor_discount:.3f}M",
            ] if getattr(inst_cfg, 'donor_discount', 0.0) > 0 else []) + [
                "",
            ]

        elif itype == 'ppo':
            lines += [
                f"--- {inst_name} (Contingent Credit — single activation) ---",
                f"  Max available: ${inst_cfg.credit_line:.0f}M",
                f"  Commitment fee: {inst_cfg.commitment_fee_rate:.2%}",
                f"  Loan interest rate: {inst_cfg.loan_interest_rate:.1%}",
                f"  Repayment: {inst_cfg.repayment_years} years",
                f"  Front-end fee: {inst_cfg.front_end_fee_rate:.2%}",
                "",
            ]

        elif itype == 'ccf':
            lines += [
                f"--- {inst_name} (Contingent Credit — recurrent) ---",
                f"  Total facility: ${inst_cfg.ccf_maximum:.0f}M (cumulative cap)",
                f"  Per person: ${inst_cfg.ccf_person:,.0f}",
                f"  Pop exposed: {inst_cfg.pop_exposed/1e6:.2f}M",
                f"  Drawdown fee: {inst_cfg.drawdown_fee_rate:.2%}",
                f"  Loan interest rate: {inst_cfg.loan_interest_rate:.1%}",
                f"  Grace period: {inst_cfg.grace_period_years} years",
                f"  Repayment: {inst_cfg.repayment_years} years",
                f"  Payout function: {inst_cfg.payout_function}",
                "",
            ]

        elif itype == 'ddo':
            block = [f"--- {inst_name} (Deferred Drawdown Option — recurrent) ---"]
            if inst_cfg.ddo_threshold is not None:
                block.append(f"  Trigger: ${inst_cfg.ddo_threshold:.0f}M loss")
            if inst_cfg.ddo_available is not None:
                block.append(f"  Payout per activation: ${inst_cfg.ddo_available:.0f}M")
            block += [
                f"  Loan interest rate: {inst_cfg.loan_interest_rate:.1%}",
                f"  Repayment: {inst_cfg.repayment_years} years",
                "",
            ]
            lines += block

    lines.append("=" * 60)
    return "\n".join(lines)


def _insurance_section(results):
    """
    v8 insurance block. It leads with protection, because that is what an
    insurance policy is bought for, and reads the benefit-cost ratio in that
    light: a fairly priced policy has a ratio below 1.
    """
    analysis = getattr(results, 'insurance_analysis', {}) or {}
    if not analysis:
        return []
    L = ["-" * 70, "INSURANCE: PROTECTION FIRST, THEN COST", "-" * 70]
    for name, a in analysis.items():
        L.append(f"  {name}")
        L.append("    Protection")
        L.append(f"      Probability the policy pays at least once in the horizon: "
                 f"{a['prob_pays_in_horizon']:.1%}")
        L.append(f"      Average payout in a year it pays:          ${a['mean_payout_when_paid']:,.2f}M"
                 f"  (limit ${a['coverage_limit']:,.2f}M)")
        L.append(f"      Share of that year's loss it covers:       "
                 f"{a['mean_share_of_loss_when_paid']:.1%} on average")
        L.append(f"      Unpaid loss, 1-in-20 outcome (p95, PV):    "
                 f"${a['unpaid_p95_without']:,.1f}M without this policy -> ${a['unpaid_p95_with']:,.1f}M with it")
        L.append(f"      Unpaid loss, 1-in-100 outcome (p99, PV):   "
                 f"${a['unpaid_p99_without']:,.1f}M without this policy -> ${a['unpaid_p99_with']:,.1f}M with it")
        L.append("      (the rest of the strategy is in place in both cases)")
        L.append("    Price")
        L.append(f"      Premium source: {a['pricing_method']}")
        L.append(f"      Gross premium ${a['gross_premium']:,.3f}M per year; expected payout "
                 f"${a['expected_annual_payout']:,.3f}M per year")
        L.append(f"      Implied multiple: {a['implied_multiple']:.2f} "
                 f"(premium per unit of expected payout)")
        if a['implied_multiple'] < 1.0:
            L += [
                "      WARNING: the premium is below the expected payout of this layer on this",
                "      country's curve, which no insurer would accept, and the B/C of the",
                "      policy is overstated. Under 'rol_rule' this happens on narrow layers",
                "      close to the cutoff, where the flat rate is too low for how often the",
                "      layer is exhausted. Enter a quotation (pricing = 'quote') or use",
                "      pricing = 'market_curve' for this layer.",
            ]
        if a['donor_discount'] > 0:
            L.append(f"      Donor discount ${a['donor_discount']:,.3f}M -> government pays "
                     f"${a['net_premium']:,.3f}M")
        L.append("    Benefit-cost")
        L.append(f"      B/C economic (gross premium):   {a['bc_economic']:.3f}")
        L.append(f"      B/C fiscal (premium net of donor discount): {a['bc_fiscal']:.3f}")
        L.append("")
    L += [
        "  How to read the insurance B/C: an insurer charges more than the",
        "  expected payout to hold capital for the bad years, so a fairly",
        "  priced policy has an economic B/C below 1 (roughly 1.1 divided by",
        "  the implied multiple). A ratio above 1 signals either a price below",
        "  expected loss or a donor subsidy (fiscal B/C). The policy is judged",
        "  by the protection block above, not by its B/C alone.",
        "",
    ]
    return L


def generate_text_report(results) -> str:
    """Generate a comprehensive text report of all indicators."""
    lines = []
    lines.append("=" * 70)
    lines.append("LEC-CBA ANALYSIS REPORT")
    lines.append("=" * 70)
    lines.append("")
    lines.append(_format_config_summary(results))
    lines.append("")
    lines.extend(_insurance_section(results))

    # --- Core indicators ---
    lines.append("-" * 70)
    lines.append("CORE EVALUATION METRICS")
    lines.append("-" * 70)
    c = results.core
    lines.append(f"  Expected B/C Ratio (aggregate)*:         {c.expected_bc:.3f}")
    lines.append(f"  P(B/C > 1)**:                            {c.prob_bc_gt_1:.1%}")
    lines.append(f"  Expected Unpaid Loss (PV)***:            ${c.expected_unpaid_loss:,.1f}M")
    # B/UL: Benefit over Unpaid Loss (complementarity). Sims with zero unpaid
    # loss are excluded from the mean (ratio undefined); the count is shown so
    # the exclusion is visible and not mistaken for a bug.
    if c.expected_bul == float('inf'):
        lines.append(f"  Expected B/UL Ratio (complementarity)+:  n/a (all sims fully covered)")
    else:
        lines.append(f"  Expected B/UL Ratio (complementarity)+:  {c.expected_bul:.3f}")
    if c.bul_n_full_coverage > 0:
        lines.append(
            f"    (excludes {c.bul_n_full_coverage}/{c.bul_n_total} sims "
            f"[{c.bul_pct_full_coverage:.1%}] with zero unpaid loss — full coverage)"
        )
        if c.bul_high_full_coverage:
            lines.append(
                "    NOTE: high share of fully-covered sims; interpret B/UL with care."
            )
    lines += _notes([
        ('*', "B/C = PV of benefits / PV of costs. Benefits are payouts plus the "
              "indirect benefit factor (faster recovery). Above 1, the strategy returns "
              "more than it costs."),
        ('**', "Share of simulations in which B/C exceeds 1."),
        ('***', "Average PV of the loss that no instrument covers."),
        ('+', "Payouts / unpaid loss, per simulation: dollars paid by the instruments "
              "for every dollar left unpaid. Simulations with no unpaid loss are "
              "excluded (ratio undefined)."),
    ])
    lines.append("")
    lines.append("  B/C Distribution:")
    for k, v in c.bc_percentiles.items():
        lines.append(f"    {k}: {v:.3f}")
    first = next(iter(c.bc_percentiles.items()), None)
    if first is not None and first[1] == 0.0:
        share = first[0].lstrip('p')
        lines += [
            f"    Note: {first[0]} = 0 means that in at least {share}% of simulations",
            "    no instrument paid out in the horizon: benefits are zero while",
            "    costs are not.",
        ]
    lines.append("")
    lines.append("  Unpaid Loss Distribution (PV):")
    for k, v in c.ul_percentiles.items():
        lines.append(f"    {k}: ${v:,.1f}M")
    lines.append("")
    lines.append("  B/C Ratio by Instrument*:")
    for name, ratios in c.bc_ratios_by_instrument.items():
        finite = ratios[np.isfinite(ratios)]
        if len(finite) > 0:
            note = ("  (insurance: economic ratio; see the insurance section)"
                    if results.instrument_types.get(name) == 'insurance' else "")
            lines.append(f"    {name:16s}  E[B/C]={np.mean(finite):.3f}  "
                         f"P(B/C>1)={np.mean(ratios > 1.0):.1%}{note}")
    lines += _notes([
        ('*', "Same definition, instrument by instrument. Instruments are evaluated "
              "independently, each against its own cost."),
    ])
    lines.append("")

    # --- Efficiency indicators ---
    lines.append("-" * 70)
    lines.append("EFFICIENCY METRICS (VfM)")
    lines.append("-" * 70)
    e = results.efficiency

    def _f(x):
        return f"{x:.3f}" if np.isfinite(x) else "N/A"

    lines.append(f"  {'Instrument':16s}  {'CM*':>8s}  {'MV**':>8s}  {'OMV***':>8s}")
    lines.append(f"  {'-'*16}  {'-'*8}  {'-'*8}  {'-'*8}")
    for name in e.cost_multiples:
        lines.append(f"  {name:16s}  {_f(e.cost_multiples[name]):>8s}  "
                     f"{_f(e.money_values[name]):>8s}  "
                     f"{_f(e.omv_values.get(name, float('inf'))):>8s}")
    lines.append(f"  {'Aggregate':16s}  {_f(e.aggregate_cost_multiple):>8s}  "
                 f"{_f(e.aggregate_money_value):>8s}  {_f(e.aggregate_omv):>8s}")
    lines += _notes([
        ('*', "Cost Multiple = expected cost / expected payout (both in PV). How many "
              "dollars the instrument costs for each dollar it pays out. Below 1: the "
              "instrument pays out more than it costs in PV."),
        ('**', "Money Value = expected payout / expected cost (inverse of CM). How many "
               "dollars the instrument pays out for each dollar spent."),
        ('***', "Optimized Money Value = (expected payout + lambda x std. deviation of "
                "payout) / expected cost. MV with a small reward for payouts that grow "
                "in bad years (tail responsiveness)."),
    ])
    lines += [
        "",
        "  Reading note: for loans (PPO, CCF, DDO) the \"cost\" is the PV of fees",
        "  and debt service at the social discount rate, and the \"payout\" is the",
        "  money disbursed. A CM below 1 means repaying the loan is worth less",
        "  today than the money received, because its rate (plus any grace",
        "  period) is below the social discount rate. It reflects concessional",
        "  terms, not free money. For insurance, CM is the implied multiple",
        "  (premium per unit of expected payout).",
    ]
    lines.append("")

    # --- DRR Analysis ---
    drr = getattr(results, 'drr', None)
    if drr is not None:
        lines.append("-" * 70)
        lines.append("DRR COST-EFFECTIVENESS")
        lines.append("-" * 70)
        d = drr
        lines.append(f"  PV of DRR Investment*:         ${d.pv_drr_cost:,.1f}M")
        lines.append(f"  PV Direct Benefit (ΔL)**:      ${d.pv_direct_benefit:,.1f}M")
        lines.append(f"  PV Indirect Benefit (ΔUL)***:  ${d.pv_indirect_benefit:,.1f}M")
        lines.append(f"  B/C Direct:                    {d.bc_direct:.3f}")
        lines.append(f"  B/C Indirect:                  {d.bc_indirect:.3f}")
        lines += _notes([
            ('*', "Investment by year discounted at the social rate."),
            ('**', "Average reduction in the PV of total losses between the base and "
                   "the reduced catalogue (same random draws)."),
            ('***', "Average reduction in the PV of unpaid loss, after all instruments."),
        ])
        lines.append("  B/C Direct = ** / *. B/C Indirect = *** / *.")
        lines.append("")
        lines.append("  The direct B/C measures how much loss each dollar")
        lines.append("  of prevention avoids, independent of the financial")
        lines.append("  strategy. The indirect B/C measures how much of")
        lines.append("  the financing gap (unpaid loss) is reduced per")
        lines.append("  dollar invested. The reduced-catalogue figures come")
        lines.append("  from re-evaluating the same instruments on the reduced")
        lines.append("  catalogue (same random draws, risk-reduced LEC).")
        lines.append("")
        lines.append("  Reading note: the avoided loss is not estimated by the tool. It")
        lines.append("  follows from the benefit-cost ratio entered for each investment, the")
        lines.append("  annualisation rate and the horizon, and the risk-reduced curves come")
        lines.append("  from a generic transformation of the original curve. Read these")
        lines.append("  ratios as conditional on those inputs. Benefits beyond the horizon")
        lines.append("  are not counted.")
        lines.append("")

    # --- CNC (Comparative Net Cost) — optional, modular ---
    cnc = getattr(results, 'cnc', None)
    if cnc is not None:
        lines.append("-" * 70)
        lines.append("COMPARATIVE NET COST (CNC) — ex-ante instruments vs ex-post debt")
        lines.append("-" * 70)
        lines.append(f"  Expected net saving (PV)*:  ${cnc.expected_net_saving:,.1f}M")
        lines.append(f"  Median net saving (PV):     ${cnc.median_net_saving:,.1f}M")
        lines.append(f"  P10 / P90 net saving (PV):  ${cnc.p10_net_saving:,.1f}M / ${cnc.p90_net_saving:,.1f}M")
        lines.append(f"  Expected % saving:          {cnc.expected_pct_saving:.1%}")
        lines.append(f"  Median % saving:            {cnc.median_pct_saving:.1%}")
        lines.append(f"  P(ex-ante strategy cheaper)**: {cnc.prob_positive_saving:.1%}")
        if cnc.n_overcoverage_cells > 0:
            if cnc.max_overcoverage_ratio == float('inf'):
                ratio_txt = "incl. payouts on zero-loss periods"
            else:
                ratio_txt = f"max payout/loss {cnc.max_overcoverage_ratio:.1f}x"
            lines.append(
                f"  Over-coverage cells:        {cnc.n_overcoverage_cells}"
                f"/{cnc.n_payout_cells} [{cnc.pct_overcoverage:.1%}] ({ratio_txt})"
            )
        lines += _notes([
            ('*', "PV cost of financing the same payouts with ex-post commercial debt, "
                  "minus the PV cost of the ex-ante instruments. Positive means the "
                  "instruments are cheaper."),
            ('**', "Share of simulations with a positive saving."),
        ])
        lines.append("")
        lines.append("  Positive saving means the ex-ante instrument portfolio")
        lines.append("  (Strategy A) is cheaper than financing the SAME coverage")
        lines.append("  with ex-post commercial debt (Strategy B), whose rate widens")
        lines.append("  with event size relative to GDP. Both strategies finance the")
        lines.append("  same amount, so this isolates the financing-cost differential")
        lines.append("  (not the size of coverage). Coverage adequacy is a separate")
        lines.append("  question — read the unpaid-loss indicators for that.")
        lines.append("")

    # --- Sensitivity ---
    sensitivity = getattr(results, 'sensitivity', None)
    if sensitivity is not None:
        lines.append("-" * 70)
        lines.append("SENSITIVITY ANALYSIS (TORNADO)")
        lines.append("-" * 70)
        lines.append(f"    {'Parameter':25s}  {'B/C Low':>8s}  {'B/C High':>9s}  {'Range':>8s}")
        lines.append(f"    {'-'*25}  {'-'*8}  {'-'*9}  {'-'*8}")
        for param, (low, high) in sorted(
            sensitivity.tornado_data.items(),
            key=lambda x: abs(x[1][1] - x[1][0]),
            reverse=True,
        ):
            lines.append(f"    {param:25s}  {low:8.3f}  {high:9.3f}  {high-low:8.3f}")
        lines.append("")

    lines.append("=" * 70)
    lines.append("END OF REPORT")
    lines.append("=" * 70)
    return "\n".join(lines)


def generate_plots(results):
    """
    Build the CBA results figure and return it (the caller saves or shows it;
    main.py stores it as <run_id>_09_cba_results.png).
    """
    import matplotlib.pyplot as plt

    has_drr = getattr(results, 'drr', None) is not None
    ncols = 3
    nrows = 2 if not has_drr else 3
    fig, axes = plt.subplots(nrows, ncols, figsize=(18, 5 * nrows))
    fig.suptitle("LEC-CBA Analysis Results", fontsize=14, fontweight='bold')

    colors = ['#4C72B0', '#55A868', '#C44E52']

    # 1. B/C Ratio Distribution
    ax = axes[0, 0]
    finite_bc = results.core.bc_ratios[np.isfinite(results.core.bc_ratios)]
    ax.hist(finite_bc, bins=50, color='steelblue', alpha=0.7, edgecolor='white')
    ax.axvline(x=1.0, color='red', linestyle='--', linewidth=2, label='B/C = 1')
    ax.axvline(x=results.core.expected_bc, color='green', linestyle='-',
               linewidth=2, label=f'E[B/C] = {results.core.expected_bc:.2f}')
    ax.set_title(f'B/C Ratio Distribution\nP(B/C>1) = {results.core.prob_bc_gt_1:.1%}')
    ax.set_xlabel('Benefit-Cost Ratio')
    ax.set_ylabel('Frequency')
    ax.legend(fontsize=8)

    # 2. Unpaid Loss Distribution
    ax = axes[0, 1]
    ax.hist(results.core.unpaid_losses_pv, bins=50, color='coral', alpha=0.7,
            edgecolor='white')
    ax.axvline(x=results.core.expected_unpaid_loss, color='darkred', linestyle='-',
               linewidth=2, label=f'E[Gap] = ${results.core.expected_unpaid_loss:,.0f}M')
    ax.set_title('Financing Gap Distribution (PV)')
    ax.set_xlabel('Unpaid Loss (M USD)')
    ax.legend(fontsize=8)

    # 3. B/C by Instrument (box plot)
    ax = axes[0, 2]
    inst_data, inst_labels = [], []
    for name, ratios in results.core.bc_ratios_by_instrument.items():
        finite = ratios[np.isfinite(ratios)]
        if len(finite) > 0:
            capped = np.clip(finite, 0, np.percentile(finite, 95))
            inst_data.append(capped)
            inst_labels.append(name)
    if inst_data:
        bp = ax.boxplot(inst_data, tick_labels=inst_labels, patch_artist=True)
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
    ax.axhline(y=1.0, color='red', linestyle='--', linewidth=1)
    ax.set_title('B/C Ratio by Instrument')

    # 4. CM + OMV by Instrument
    ax = axes[1, 0]
    names = list(results.efficiency.cost_multiples.keys())
    labels = names
    cms = [min(results.efficiency.cost_multiples[n], 10) for n in names]
    omvs = [min(results.efficiency.omv_values.get(n, 0), 10) for n in names]
    x = np.arange(len(names))
    w = 0.35
    ax.bar(x - w/2, cms, w, label='CM', color='steelblue', alpha=0.7)
    ax.bar(x + w/2, omvs, w, label='OMV', color='coral', alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.axhline(y=1.0, color='red', linestyle='--', linewidth=1)
    ax.set_title('CM & OMV by Instrument')
    ax.legend(fontsize=8)

    # 5. Tornado Chart
    ax = axes[1, 1]
    sensitivity = getattr(results, 'sensitivity', None)
    if sensitivity is not None and sensitivity.tornado_data:
        sorted_params = sorted(
            sensitivity.tornado_data.items(),
            key=lambda x: abs(x[1][1] - x[1][0])
        )
        param_names = [p[0].replace('_', ' ') for p in sorted_params]
        lows = [p[1][0] for p in sorted_params]
        highs = [p[1][1] for p in sorted_params]
        base = results.core.expected_bc
        y_pos = np.arange(len(param_names))
        ax.barh(y_pos, [h - base for h in highs], left=base,
                color='steelblue', alpha=0.7, label='High')
        ax.barh(y_pos, [base - l for l in lows], left=[l for l in lows],
                color='coral', alpha=0.7, label='Low')
        ax.set_yticks(y_pos)
        ax.set_yticklabels(param_names, fontsize=8)
        ax.axvline(x=base, color='black', linewidth=1)
        ax.set_title(f'Sensitivity Tornado (Base B/C = {base:.3f})')
        ax.legend(fontsize=8)
    else:
        ax.axis('off')   # sensitivity analysis not run

    # 6. Unpaid loss percentiles (PV)
    ax = axes[1, 2]
    if results.core.ul_percentiles:
        labels = list(results.core.ul_percentiles.keys())
        values = [results.core.ul_percentiles[k] for k in labels]
        ax.bar(labels, values, color='coral', alpha=0.7)
        ax.set_title('Unpaid Loss Percentiles (PV)')
        ax.set_ylabel('M USD')

    # DRR (if available)
    if has_drr:
        d = results.drr
        # 7. DRR B/C ratios
        ax = axes[2, 0]
        ax.bar(['Direct\n(loss reduction)', 'Indirect\n(gap reduction)'],
               [d.bc_direct, d.bc_indirect],
               color=['steelblue', 'coral'], alpha=0.7)
        ax.axhline(y=1.0, color='red', linestyle='--')
        ax.set_title('DRR Cost-Effectiveness')
        ax.set_ylabel('B/C Ratio')

        # 8. DRR benefit distribution
        ax = axes[2, 1]
        ax.hist(d.direct_benefit_dist, bins=50, color='steelblue', alpha=0.5,
                label=f'Direct (E=${d.pv_direct_benefit:,.0f}M)')
        ax.hist(d.indirect_benefit_dist, bins=50, color='coral', alpha=0.5,
                label=f'Indirect (E=${d.pv_indirect_benefit:,.0f}M)')
        ax.set_title('DRR Benefit Distribution (PV)')
        ax.set_xlabel('Benefit (M USD)')
        ax.legend(fontsize=8)

        # 9. empty
        axes[2, 2].axis('off')

    fig.tight_layout()
    return fig
