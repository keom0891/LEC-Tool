# -*- coding: utf-8 -*-
"""
main.py — LEC Tool
===================
Single entry point of the LEC Tool. Edit ``config.toml`` and run::

    python main.py

Every input comes from the configuration file; every result is written to
``<output_dir>/<run id>/`` with the run id as file-name prefix:

    <id>_01..08_*.png       figures (see plots.py for the numbering)
    <id>_09_cba_results.png CBA figure (when the CBA is enabled)
    <id>_report.txt         main results report
    <id>_statistics.csv     strategy statistics per scenario
    <id>_cba_report.txt     cost-benefit report (when the CBA is enabled)
    <id>_config_used.toml   copy of the configuration that produced the run

Pipeline
--------
1. load_inputs         read the CSV inputs
2. compute_lec         empirical LEC + bootstrap band, optional hybrid tail
2b. resolve_layers     thresholds given as return periods -> dollars on this
                       country's curve; insurance ceding share and premium (v8)
3. simulate            synthetic catalogues (compound Poisson, CRN streams)
4. evaluate_strategy   instrument payouts on the base catalogue
5. run_risk_reduction  ex-ante DRR: reduced LEC, reduced catalogue, payouts   [risk_reduction.enabled]
6. run_cba_stage       cost-benefit analysis                                 [cba.enabled]
7. write_outputs       figures, report, CSV

Stages 5 and 6 only run when enabled in config.toml. A strategy without a PPO
never reads the PPO schedule file.
"""

import shutil
import sys

import numpy as np

from config_loader import (
    load_config, load_input_data, build_drm_configs, build_cba_config, ConfigError,
)
from lec_core import compute_empirical_lec, build_hybrid_lec
from simulation import generate_synthetic_catalogue
from risk_management import apply_strategy
from risk_reduction import compute_reduction_schedule, generate_reduced_catalogue
import reporting


def _log(msg):
    print(f'[LEC Tool] {msg}', flush=True)


# =============================================================================
# Stages
# =============================================================================

def load_inputs(cfg):
    inputs = load_input_data(cfg)
    inputs['drm_configs'] = build_drm_configs(cfg, inputs['ppo_schedule'])
    _log(f"inputs loaded: {len(inputs['event_loss_df'])} historical events, "
         f"{len(inputs['drm_configs'])} instrument(s)"
         + (", PPO schedule read" if cfg.has_ppo else ""))
    return inputs


def compute_lec(cfg, inputs):
    lec_result = compute_empirical_lec(
        inputs['event_loss_df'],
        loss_scale_factor=cfg.lec.loss_scale_factor,
        freq_scale_factor=cfg.lec.freq_scale_factor,
        B=cfg.lec.bootstrap_samples,
        random_seed=cfg.simulation.random_seed,
    )
    if cfg.lec.hybrid_curve:
        lec_curve, aal = build_hybrid_lec(lec_result['lec_curve'], inputs['tail_loss'], inputs['tail_aep'])
    else:
        lec_curve, aal = lec_result['lec_curve'], lec_result['aal']
    _log(f"LEC computed: AAL = {aal:,.2f} $MM, maximum loss on curve = {lec_curve[:, 0].max():,.1f} $MM")
    return {
        'lec_result': lec_result,
        'lec_curve': lec_curve,
        'aal': aal,
        'aal_empirical': lec_result['aal'],
        'max_loss': float(lec_curve[:, 0].max()),
        'max_event_loss': lec_result['max_loss'],
        'pml': reporting.pml_by_return_period(lec_curve, cfg.lec.pml_return_periods),
    }


def resolve_layers(cfg, lec, inputs):
    """
    v8: place every threshold on the loss curve of the country being run and
    price the insurance layers (see insurance_layer.py). Replaces the
    instrument list in *inputs* with the resolved one, in dollars.
    """
    from insurance_layer import resolve_instruments, LayerError
    try:
        resolved, summaries, notes = resolve_instruments(
            inputs['drm_configs'], lec['lec_curve'], cfg.insurance_pricing)
    except LayerError as exc:
        raise ConfigError(str(exc)) from exc
    inputs['drm_configs'] = resolved
    for s in summaries:
        _log(f"insurance '{s.name}': layer {s.attachment:,.1f}-{s.exhaustion:,.1f} $MM "
             f"(1 in {s.attachment_rp:.0f} to 1 in {s.exhaustion_rp:.0f} years), "
             f"limit {s.coverage_limit:,.2f} $MM, premium {s.gross_premium:,.3f} $MM "
             f"[{s.method}]")
    for n in notes:
        _log(f"warning: {n}")
    return {'summaries': summaries, 'notes': notes}


def simulate(cfg, lec):
    sim = generate_synthetic_catalogue(
        lec['lec_curve'], cfg.simulation.catalogue_length,
        cfg.simulation.simulation_number, cfg.simulation.random_seed,
    )
    _log(f"synthetic catalogue generated: {cfg.simulation.simulation_number} x "
         f"{cfg.simulation.catalogue_length} years")
    return sim


def evaluate_strategy(cfg, sim, inputs):
    drm = apply_strategy(sim['event_catalogue'], inputs['drm_configs'], cfg.simulation.catalogue_length)
    _log('strategy evaluated on the base catalogue')
    return {
        'drm_configs': inputs['drm_configs'],
        'payout_dfs': drm['payout_dfs'],
        'total_coverage': drm['total_coverage'],
    }


def run_risk_reduction(cfg, lec, sim, inputs):
    rr = cfg.risk_reduction
    red = compute_reduction_schedule(rr.investment, rr.benefit_cost_ratio, rr.benefit_horizon, rr.discount_rate)
    reduced = generate_reduced_catalogue(
        lec['lec_curve'], red,
        sim['N_events'], sim['U_times'], sim['U_loss'],
        cfg.simulation.catalogue_length, cfg.simulation.simulation_number,
    )
    drm_red = apply_strategy(reduced['event_catalogue_red'], inputs['drm_configs'],
                             cfg.simulation.catalogue_length)
    payback = reporting.drr_payback_years(sim['synthetic_annual_df'], reduced['synthetic_annual_red_df'],
                                          rr.investment)
    _log(f"ex-ante risk reduction evaluated: total investment {np.sum(rr.investment):,.1f} $MM, "
         f"median payback {payback:.0f} years")
    return {
        'reduction': red,
        'total_investment': float(np.sum(rr.investment)),
        'synthetic_annual_red_df': reduced['synthetic_annual_red_df'],
        'event_catalogue_red': reduced['event_catalogue_red'],
        'lec_curves_by_target': reduced['lec_curves_by_target'],
        'payout_red_dfs': drm_red['payout_dfs'],
        'total_coverage_red': drm_red['total_coverage'],
        'payback_years': payback,
    }


def run_cba_stage(cfg, sim, strat, drr, thresholds, out_dir):
    """
    Cost-benefit analysis on the base catalogue. When the ex-ante risk
    reduction ran, the CBA is repeated on the reduced catalogue (same
    instruments re-evaluated there) and the DRR cost-effectiveness
    indicators are attached to the results.
    """
    from cba.engine import run_cba
    from cba.reports import generate_text_report

    cba_config = build_cba_config(cfg)
    scale = cfg.fiscal.resp_fiscal if cfg.cba.loss_basis == 'fiscal' else 1.0
    common = dict(
        drm_configs=strat['drm_configs'],
        cba_config=cba_config,
        resp_fiscal=cfg.fiscal.resp_fiscal,
        gap_thresholds=[thr for _, thr in thresholds],
        loss_basis=cfg.cba.loss_basis,
    )

    results = run_cba(
        losses_df=sim['synthetic_annual_df'] * scale,
        payout_dfs=strat['payout_dfs'],
        **common,
    )

    results_reduced = None
    if drr is not None:
        from cba.diagnostics import compute_drr_analysis
        from cba.discounting import present_value_matrix

        results_reduced = run_cba(
            losses_df=drr['synthetic_annual_red_df'] * scale,
            payout_dfs=drr['payout_red_dfs'],
            **common,
        )
        results.drr = compute_drr_analysis(
            losses_pv_original=present_value_matrix(results.losses_matrix, cba_config.discount),
            losses_pv_reduced=present_value_matrix(results_reduced.losses_matrix, cba_config.discount),
            unpaid_pv_original=results.core.unpaid_losses_pv,
            unpaid_pv_reduced=results_reduced.core.unpaid_losses_pv,
            inv_vector=cfg.risk_reduction.investment,
            discount_rate=cfg.cba.social_discount_rate,
            cumulative_reduction=drr['reduction'],
        )

    report = generate_text_report(results)
    report_file = out_dir / f'{cfg.run.id}_cba_report.txt'
    report_file.write_text(report, encoding='utf-8')
    _log(f"cost-benefit analysis completed: E[B/C] = {results.core.expected_bc:.3f}"
         + (f", DRR B/C direct = {results.drr.bc_direct:.3f}" if results.drr is not None else ""))
    return {'results': results, 'results_reduced': results_reduced, 'report': report,
            'report_file': report_file, 'config': cba_config}


# =============================================================================
# Outputs
# =============================================================================

def write_outputs(cfg, inputs, lec, sim, strat, drr, cba, thresholds, out_dir, layers=None):
    import plots  # imported here so run() can select the backend first

    run_id = cfg.run.id
    horizon = cfg.simulation.catalogue_length
    shown = cfg.simulation.displayed_catalogue
    year_labels = np.arange(cfg.simulation.first_year, cfg.simulation.first_year + horizon)
    files = []

    def save(fig, index, slug):
        files.append(plots.save_figure(fig, out_dir, run_id, index, slug,
                                       dpi=cfg.run.figure_dpi, keep_open=cfg.run.show_figures))

    # --- Figures ------------------------------------------------------------
    save(plots.plot_lec_curve(lec['lec_result'], lec['lec_curve'], lec['aal'],
                              cfg.lec.hybrid_curve, inputs['event_loss_df']), 1, 'lec_curve')
    save(plots.plot_lec_vs_simulated(lec['lec_curve'], sim['event_catalogue'], horizon,
                                     cfg.simulation.simulation_number), 2, 'lec_analytical_vs_simulated')
    save(plots.plot_catalogue_statistics(sim['synthetic_annual_df'], horizon, shown), 3, 'catalogue_statistics')

    ylim = None
    if cfg.has_instruments:
        fig, ylim = plots.plot_strategy_payouts(
            sim['synthetic_annual_df'], strat['payout_dfs'], strat['drm_configs'], year_labels, shown,
            f'{run_id} - catalogue {shown}')
        save(fig, 4, 'strategy_payouts_base')

    if drr is not None:
        save(plots.plot_drr_investment(year_labels, cfg.risk_reduction.investment), 5, 'drr_investment')
        save(plots.plot_reduced_lec_curves(drr['lec_curves_by_target']), 6, 'reduced_lec_curves')
        if cfg.has_instruments:
            fig, _ = plots.plot_strategy_payouts(
                drr['synthetic_annual_red_df'], drr['payout_red_dfs'], strat['drm_configs'], year_labels,
                shown, f'{run_id} - reduced catalogue {shown}', ylim=ylim)
            save(fig, 7, 'strategy_payouts_reduced')

    save(plots.plot_horizon_loss_distribution(sim['synthetic_annual_df'], horizon), 8, 'horizon_loss_distribution')

    if cba is not None:
        from cba.reports import generate_plots
        save(generate_plots(cba['results']), 9, 'cba_results')

    # --- Statistics and gap probabilities -----------------------------------
    resp = cfg.fiscal.resp_fiscal
    base_label = f'{run_id} - Base catalogue'
    scenarios = [('Base catalogue', reporting.strategy_statistics(
        base_label, sim['synthetic_annual_df'], strat['total_coverage'], resp, horizon))]
    gaps = [('Base catalogue', reporting.gap_probabilities(
        sim['synthetic_annual_df'], strat['total_coverage'], thresholds))]
    if drr is not None:
        red_label = f'{run_id} - Reduced catalogue (ex-ante)'
        scenarios.append(('Reduced (ex-ante)', reporting.strategy_statistics(
            red_label, drr['synthetic_annual_red_df'], drr['total_coverage_red'], resp, horizon)))
        gaps.append(('Reduced (ex-ante)', reporting.gap_probabilities(
            drr['synthetic_annual_red_df'], drr['total_coverage_red'], thresholds)))

    stats_path = out_dir / f'{run_id}_statistics.csv'
    reporting.statistics_table([s[1] for s in scenarios]).to_csv(stats_path, encoding='utf-8-sig')
    files.append(stats_path)
    if cba is not None:
        files.append(cba['report_file'])

    config_copy = out_dir / f'{run_id}_config_used.toml'
    shutil.copyfile(cfg.source_path, config_copy)
    files.append(config_copy)

    # --- Main report --------------------------------------------------------
    report_path = out_dir / f'{run_id}_report.txt'
    files.append(report_path)
    ctx = {
        'cfg': cfg,
        'n_events': len(inputs['event_loss_df']),
        'year_min': int(inputs['event_loss_df']['year'].min()),
        'year_max': int(inputs['event_loss_df']['year'].max()),
        'lec': lec,
        'instruments': strat['drm_configs'],
        'layers': layers,
        'scenarios': scenarios,
        'thresholds': thresholds,
        'gaps': gaps,
        'drr': drr,
        'year_labels': year_labels,
        'horizon_stats': reporting.horizon_loss_statistics(sim['synthetic_annual_df']),
        'cba': cba,
        'files': [p.name for p in files],
    }
    report = reporting.build_main_report(ctx)
    report_path.write_text(report, encoding='utf-8')
    return report, files


# =============================================================================
# Entry point
# =============================================================================

def run(config_path=None):
    """Run the complete pipeline for one configuration file. Returns the output folder."""
    cfg = load_config(config_path)
    out_dir = cfg.output_path
    out_dir.mkdir(parents=True, exist_ok=True)
    _log(f"run '{cfg.run.id}' -> {out_dir}")

    import matplotlib
    if not cfg.run.show_figures:
        matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    inputs = load_inputs(cfg)
    lec = compute_lec(cfg, inputs)
    layers = resolve_layers(cfg, lec, inputs)
    sim = simulate(cfg, lec)
    strat = evaluate_strategy(cfg, sim, inputs)
    drr = run_risk_reduction(cfg, lec, sim, inputs) if cfg.risk_reduction.enabled else None

    # Financing-gap thresholds: fractions of the maximum loss on the LEC curve,
    # shared by the main report and the CBA.
    thresholds = reporting.gap_thresholds(lec['max_loss'], cfg.lec.gap_threshold_fractions)

    cba = None
    if cfg.cba.enabled:
        if cfg.has_instruments:
            cba = run_cba_stage(cfg, sim, strat, drr, thresholds, out_dir)
        else:
            _log('cost-benefit analysis skipped: no instruments declared')

    report, files = write_outputs(cfg, inputs, lec, sim, strat, drr, cba, thresholds, out_dir, layers)
    print('\n' + report)
    if cba is not None:
        print('\n' + cba['report'])
    _log(f"{len(files)} files written to {out_dir}")

    if cfg.run.show_figures:
        plt.show()
    return out_dir


if __name__ == '__main__':
    try:
        run(sys.argv[1] if len(sys.argv) > 1 else None)
    except ConfigError as exc:
        print(f'\nConfiguration error: {exc}', file=sys.stderr)
        sys.exit(2)
