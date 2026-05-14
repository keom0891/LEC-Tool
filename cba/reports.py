"""
LEC-CBA Reports Module
========================
Generates formatted results: text reports and matplotlib visualizations.
Aligned to BID instruments (CCRIF, PPO, CCF) with DRR comparative table.
"""

import numpy as np
from typing import Optional


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
        f"Fiscal responsibility share (resp_fiscal): "
        f"{results.resp_fiscal:.0%}",
        f"Analysis horizon: {cfg.discount.analysis_horizon} years",
        f"Simulations: {cfg.discount.num_simulations}",
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
                f"  Ceding: {inst_cfg.ceding_percentage:.1%}",
                f"  ROL: {inst_cfg.rate_on_line:.1%}",
                f"  Annual premium: ${inst_cfg.premium:.2f}M",
                "",
            ]

        elif itype == 'ppo':
            lines += [
                f"--- {inst_name} (Contingent Credit — single activation) ---",
                f"  Max available: ${inst_cfg.credit_line:.0f}M",
                f"  Commitment fee: {inst_cfg.commitment_fee_rate:.2%}",
                f"  Interest rate: {inst_cfg.interest_rate:.1%}",
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
                f"  Interest rate: {inst_cfg.interest_rate:.1%}",
                f"  Grace period: {inst_cfg.grace_period_years} years",
                f"  Repayment: {inst_cfg.repayment_years} years",
                f"  Payout function: {inst_cfg.payout_function}",
                "",
            ]

        elif itype == 'ddo':
            lines += [
                f"--- {inst_name} (Deferred Drawdown Option — recurrent) ---",
                f"  Interest rate: {inst_cfg.interest_rate:.1%}",
                f"  Repayment: {inst_cfg.repayment_years} years",
                "",
            ]

    lines.append("=" * 60)
    return "\n".join(lines)


def generate_text_report(results) -> str:
    """Generate a comprehensive text report of all indicators."""
    lines = []
    lines.append("=" * 70)
    lines.append("LEC-CBA ANALYSIS REPORT")
    lines.append("=" * 70)
    lines.append("")
    lines.append(_format_config_summary(results))
    lines.append("")

    # --- Core indicators ---
    lines.append("-" * 70)
    lines.append("CORE EVALUATION METRICS")
    lines.append("-" * 70)
    c = results.core
    lines.append(f"  Expected B/C Ratio (aggregate):  {c.expected_bc:.3f}")
    lines.append(f"  P(B/C > 1):                      {c.prob_bc_gt_1:.1%}")
    lines.append(f"  Expected Unpaid Loss (PV):        ${c.expected_unpaid_loss:,.1f}M")
    lines.append(f"  Expected B/UL Ratio:              {c.expected_bul:.3f}")
    lines.append("")
    lines.append("  B/C Distribution:")
    for k, v in c.bc_percentiles.items():
        lines.append(f"    {k}: {v:.3f}")
    lines.append("")
    lines.append("  Financing Gap Probabilities:")
    for thr, p in c.prob_gap_gt_threshold.items():
        lines.append(f"    P(Gap > ${thr:,.0f}M): {p:.1%}")
    lines.append("")
    lines.append("  B/C Ratio by Instrument:")
    for name, ratios in c.bc_ratios_by_instrument.items():
        finite = ratios[np.isfinite(ratios)]
        if len(finite) > 0:
            lines.append(f"    {name:16s}  E[B/C]={np.mean(finite):.3f}  "
                         f"P(B/C>1)={np.mean(ratios > 1.0):.1%}")
    lines.append("")

    # --- Efficiency indicators ---
    lines.append("-" * 70)
    lines.append("EFFICIENCY METRICS (VfM)")
    lines.append("-" * 70)
    e = results.efficiency
    lines.append(f"  Aggregate Cost Multiple:  {e.aggregate_cost_multiple:.3f}")
    lines.append(f"  Aggregate Money Value:    {e.aggregate_money_value:.3f}")
    lines.append(f"  Aggregate OMV:            {e.aggregate_omv:.3f}")
    lines.append("")
    lines.append(f"  {'Instrument':16s}  {'CM':>8s}  {'MV':>8s}  {'OMV':>8s}")
    lines.append(f"  {'-'*16}  {'-'*8}  {'-'*8}  {'-'*8}")
    for name in e.cost_multiples:
        cm = e.cost_multiples[name]
        mv = e.money_values[name]
        omv = e.omv_values.get(name, float('inf'))
        cm_s = f"{cm:.3f}" if np.isfinite(cm) else "N/A"
        mv_s = f"{mv:.3f}" if np.isfinite(mv) else "N/A"
        omv_s = f"{omv:.3f}" if np.isfinite(omv) else "N/A"
        lines.append(f"  {name:16s}  {cm_s:>8s}  {mv_s:>8s}  {omv_s:>8s}")
    lines.append("")

    # --- DRR Analysis ---
    drr = getattr(results, 'drr', None)
    if drr is not None:
        lines.append("-" * 70)
        lines.append("DRR COST-EFFECTIVENESS")
        lines.append("-" * 70)
        d = drr
        lines.append(f"  PV of DRR Investment:      ${d.pv_drr_cost:,.1f}M")
        lines.append(f"  PV Direct Benefit (ΔL):    ${d.pv_direct_benefit:,.1f}M")
        lines.append(f"  PV Indirect Benefit (ΔUL): ${d.pv_indirect_benefit:,.1f}M")
        lines.append(f"  B/C Direct:                {d.bc_direct:.3f}")
        lines.append(f"  B/C Indirect:              {d.bc_indirect:.3f}")
        lines.append("")
        lines.append("  The direct B/C measures how much loss each dollar")
        lines.append("  of prevention avoids, independent of the financial")
        lines.append("  strategy. The indirect B/C measures how much of")
        lines.append("  the financing gap (unpaid loss) is reduced per")
        lines.append("  dollar invested. Both are computed on the original")
        lines.append("  instrument payouts — instruments are not re-evaluated")
        lines.append("  on the reduced catalog.")
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


def generate_plots(results, output_dir: str = "."):
    """Generate matplotlib visualizations."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available. Skipping plot generation.")
        return

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
        bp = ax.boxplot(inst_data, labels=inst_labels, patch_artist=True)
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
        ax.text(0.5, 0.5, 'No sensitivity data', ha='center', va='center',
                transform=ax.transAxes)

    # 6. Gap probabilities
    ax = axes[1, 2]
    if results.core.prob_gap_gt_threshold:
        thrs = sorted(results.core.prob_gap_gt_threshold.keys())
        probs = [results.core.prob_gap_gt_threshold[t] for t in thrs]
        ax.bar([f"${t:,.0f}M" for t in thrs], probs, color='coral', alpha=0.7)
        ax.set_title('P(Gap > Threshold)')
        ax.set_ylabel('Probability')
        ax.set_ylim(0, 1)

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

    plt.tight_layout()
    filepath = f"{output_dir}/lec_cba_results.png"
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Plots saved to {filepath}")
    return filepath
