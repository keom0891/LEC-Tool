# -*- coding: utf-8 -*-
"""
main.py — LEC Tool Demo Script
================================
Complete end-to-end demonstration of the LEC Tool workflow.

This script is the integration reference for the online version. It reads input
data from the data/ folder, defines all parameters, calls the four core
modules in order, and produces all diagnostic plots.

Flow
----
1.  Parameters & data loading
2.  Empirical LEC (with optional hybrid tail)
3.  Synthetic catalogue generation (CRN)
4.  DRM strategy evaluation (Estrategia 1: CCRIF + PPO + CCF + DDO)
5.  Ex-ante risk reduction + DRM on reduced catalogue

Input files (data/)
-------------------
LEC_event_loss_example.csv  columns: year (int), econ_loss (float $MM)
ppo_example.csv             single headerless row: 10 PPO amounts ($MM)
tail_curve.csv              columns: tail_loss (float $MM), tail_aep (float)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime

from lec_core import compute_empirical_lec, build_hybrid_lec
from simulation import generate_synthetic_catalogue
from risk_management import apply_strategy
from risk_reduction import compute_reduction_schedule, generate_reduced_catalogue

# =============================================================================
# 1. PARAMETERS
# =============================================================================

DATA_DIR = Path(__file__).parent / 'data'
OUTPUT_DIR = Path(__file__).parent / 'outputs'
OUTPUT_DIR.mkdir(exist_ok=True)

# --- Global ---
resp_fiscal       = 0.60   # fiscal responsibility share (0–1)
loss_scale_factor = 1.0    # multiplicative scale on all losses
freq_scale_factor = 1.0    # multiplicative scale on all exceedance rates
curva_hibrida     = True   # blend empirical LEC with probabilistic tail

# --- Simulation ---
catalogue_length  = 10     # years per synthetic catalogue
simulation_number = 1000   # number of independent catalogues (1–1000)
random_seed       = 99     # for reproducibility, set to None for random number generation

# --- Visualization ---
catalogo_visualizado = 860  # which catalogue to display in per-catalogue plots (0-(simulation_number-1))

# --- DRM strategy: Estrategia 1 (CCRIF + PPO + CCF) ---
# PPO schedule is loaded from file; placeholder here, overwritten below.
# Cost parameters can be added to any instrument dict and will override the
# cba_config defaults for that specific instrument. Examples:
#   insurance: add 'rate_on_line' (float)
#   ppo:       add 'commitment_fee_rate', 'interest_rate', 'repayment_years'
#   ccf:       add 'drawdown_fee_rate', 'interest_rate', 'grace_period_years'
#   ddo:       add 'interest_rate', 'repayment_years'
# If absent, cba/config.py defaults are used.
drm_configs = [
    {
        'name': 'CCRIF',
        'type': 'insurance',
        'attachment_point': 50,       # $MM
        'exhaustion_point': 190,      # $MM
        'ceding_percentage': 0.066,
    },
    {
        'name': 'BID PPO',
        'type': 'ppo',
        'ppo_schedule': None,         # set from ppo_example.csv (length must == catalogue_length)
    },
    {
        'name': 'BID CCF',
        'type': 'ccf',
        'ccf_maximum': 300,           # $MM total facility
        'ccf_person': 1650,           # $ per affected person
        'Pop_exposed': 10.83e6,       # exposed population
    },
    {
        'name': 'BID DDO $40MM',
        'type': 'ddo',
        'ddo_threshold': 120,         # $MM — trigger loss level
        'ddo_available': 40,          # $MM — fixed payout when triggered
    },
    {
        'name': 'WB DDO',
        'type': 'ddo',
        'ddo_threshold': 120,         # $MM — trigger loss level
        'ddo_available': 40,          # $MM — fixed payout when triggered
    },
]

id_estrategia = 'Estrategia 2'

# --- Ex-ante reduction ---
year_ini      = datetime.now().year + 1   # first year of the simulation horizon (for axis labels only)
discount_rate = 0.12                      # annual discount rate for investment cost-benefit
inv = [ 4,  0,  8,  0,  16,  0,  8,  0, 4,  0]   # $MM invested per year
rbc = [ 4,  4,  4,  4,  4,  4,  4,  4,  4,  4]  # benefit-to-cost ratio
hor = [20, 20, 20, 20, 20, 20, 20, 20, 20, 20]  # benefit horizon (years)

for name, arr in {'inv': inv, 'rbc': rbc, 'hor': hor}.items():
    if len(arr) != catalogue_length:
        raise ValueError(f"'{name}' has length {len(arr)} but catalogue_length={catalogue_length}.")

# =============================================================================
# 2. INPUT DATA
# =============================================================================

event_loss_df = pd.read_csv(DATA_DIR / 'LEC_event_loss_example.csv')

tail_curve_df = pd.read_csv(DATA_DIR / 'tail_curve_example.csv')
tail_loss = tail_curve_df['tail_loss'].to_numpy(dtype=float)
tail_aep  = tail_curve_df['tail_aep'].to_numpy(dtype=float)

# ppo_example.csv is a single headerless row: catalogue_length comma-separated amounts
ppo_schedule = pd.read_csv(DATA_DIR / 'ppo_example.csv', header=None).iloc[0].tolist()

if len(ppo_schedule) != catalogue_length:
    raise ValueError(
        f"ppo_example.csv has {len(ppo_schedule)} values "
        f"but catalogue_length={catalogue_length}."
    )

drm_configs[1]['ppo_schedule'] = ppo_schedule  # wire loaded schedule into config

# =============================================================================
# 3. EMPIRICAL LEC
# =============================================================================

lec_result = compute_empirical_lec(
    event_loss_df,
    loss_scale_factor=loss_scale_factor,
    freq_scale_factor=freq_scale_factor,
    B=1000,
    random_seed=random_seed,
)

empirical        = lec_result['empirical']
lambda_empirical = lec_result['lambda_empirical']
lec_mean         = lec_result['lec_mean']
lec_p05          = lec_result['lec_p05']
lec_p95          = lec_result['lec_p95']

if curva_hibrida:
    lec_curve, aal = build_hybrid_lec(
        lec_result['lec_curve'], tail_loss, tail_aep
    )
else:
    lec_curve = lec_result['lec_curve']
    aal       = lec_result['aal']

# --- Plot 1: LEC + bootstrap CI  |  Annual loss bar chart ---
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

axes[0].plot(empirical, lambda_empirical, 'k-',  linewidth=1.5, label='Empirical LEC')
axes[0].plot(lec_mean,  lambda_empirical, 'b-', linewidth=2,   label='Mean Bootstrapped LEC')
axes[0].fill_betweenx(lambda_empirical, lec_p05, lec_p95, color=[0.8, 0.8, 1], alpha=0.5, label='90% CI')
if curva_hibrida:
    axes[0].plot(lec_curve[:, 0], lec_curve[:, 1], 'r--', linewidth=1.5, label='Hybrid LEC')
axes[0].set_xscale('log')
axes[0].set_yscale('log')
axes[0].grid(True, which='both')
axes[0].set_xlabel('Economic Loss [$MM]')
axes[0].set_ylabel('Annual Frequency of Exceedance')
axes[0].set_title('Loss Exceedance Curve (LEC)')
axes[0].text(0.05, 0.8, f'AAL = ${aal:,.2f} MM', transform=axes[0].transAxes, fontsize=10,
             bbox=dict(facecolor='white', edgecolor='black', boxstyle='round'))
axes[0].legend()

yearly_loss = event_loss_df.groupby('year')['econ_loss'].sum().reset_index()
axes[1].bar(yearly_loss['year'], yearly_loss['econ_loss'])
axes[1].set_title('Economic Loss vs Year')
axes[1].set_xlabel('Year')
axes[1].set_ylabel('Economic Loss [$MM]')
axes[1].grid(True)

plt.tight_layout()
plt.show()

# =============================================================================
# 4. SYNTHETIC CATALOGUE
# =============================================================================

sim_result = generate_synthetic_catalogue(
    lec_curve, catalogue_length, simulation_number, random_seed
)

event_catalogue      = sim_result['event_catalogue']
synthetic_annual_df  = sim_result['synthetic_annual_df']

# --- Plot 2: Original LEC vs empirical LEC from simulated events ---
LS          = lec_curve[:, 0]
lambda_loss = lec_curve[:, 1]
all_losses  = [l for cat in event_catalogue for l in cat['losses']]

hist, _ = np.histogram(all_losses, bins=np.append(LS, np.inf))
lambda_simulated = np.flip(np.cumsum(np.flip(hist))) / (catalogue_length * simulation_number)

plt.figure()
plt.plot(LS, lambda_loss, 'b-',  linewidth=2,   label='Original (Analytical)')
plt.plot(LS, lambda_simulated, 'r--', linewidth=1.5, label='Empirical (Simulated)')
plt.xscale('log')
plt.yscale('log')
plt.xlabel('Loss [$MM]')
plt.ylabel('Annual Frequency of Exceedance')
plt.title('LEC: Analytical vs Simulated')
plt.legend(loc='lower left')
plt.grid(True, which='both', ls='--')
plt.ylim(bottom=lambda_loss.min())
plt.tight_layout()
plt.show()

# --- Plot 3: Catalogue statistics (2 × 2 grid) ---
fig, axes = plt.subplots(2, 2, figsize=(10, 8))

axes[0, 0].bar(np.arange(1, catalogue_length + 1),
               synthetic_annual_df.loc[catalogo_visualizado].values)
axes[0, 0].set_xlabel('Año')
axes[0, 0].set_ylabel('Pérdidas económicas anuales simuladas ($MM)')
axes[0, 0].set_title(f'Pérdidas anuales — catálogo {catalogo_visualizado}')

axes[1, 0].hist(synthetic_annual_df.values.flatten(), bins=50)
axes[1, 0].set_xlabel('Pérdidas económicas anuales simuladas ($MM)')
axes[1, 0].set_ylabel('Frecuencia')
axes[1, 0].set_title('Distribución pérdidas anuales (todos los catálogos)')

data_cat = synthetic_annual_df.loc[catalogo_visualizado].values
axes[0, 1].barh(['Desv. Estándar', 'Mediana', 'Máximo', 'Promedio'],
                [np.std(data_cat), np.median(data_cat),
                 np.max(data_cat), np.mean(data_cat)])
axes[0, 1].set_xlabel('Pérdida económica anual ($MM)')
axes[0, 1].set_title(f'Estadísticas — catálogo {catalogo_visualizado}')

data_all = synthetic_annual_df.values.flatten()
axes[1, 1].barh(['Desv. Estándar', 'Mediana', 'Máximo', 'Promedio'],
                [np.std(data_all), np.median(data_all),
                 np.max(data_all), np.mean(data_all)])
axes[1, 1].set_xlabel('Pérdida económica anual ($MM)')
axes[1, 1].set_title('Estadísticas — todos los catálogos')

plt.tight_layout()
plt.show()

# =============================================================================
# 5. DRM STRATEGY (base catalogue)
# =============================================================================

drm_result     = apply_strategy(event_catalogue, drm_configs, catalogue_length)
payout_dfs     = drm_result['payout_dfs']     
total_coverage = drm_result['total_coverage']
year_labels    = np.arange(year_ini, year_ini + catalogue_length)

# --- Plot 4: Stacked DRM payouts for selected catalogue ---
loss_vals = synthetic_annual_df.loc[catalogo_visualizado].values
p = [df.loc[catalogo_visualizado].values for df in payout_dfs]

fig, ax = plt.subplots(figsize=(10, 6))
ax.bar(year_labels, loss_vals, label='Pérdidas totales')
bottom = np.zeros(catalogue_length)
for cfg, payout in zip(drm_configs, p):
    ax.bar(year_labels, payout, bottom=bottom, label=cfg['name'], hatch = '//')
    bottom += payout
for x, v in zip(year_labels, loss_vals):
    if v > 0:
        ax.text(x, v, f'${v:.1f}MM', ha='center', va='bottom', fontsize=9)
Lim_axis = loss_vals.max() + 50
ax.set_ylim(0, Lim_axis)
ax.set_xticks(year_labels)
ax.set_xticklabels(year_labels)
ax.set_xlabel('Año')
ax.set_ylabel('Pérdidas económicas anuales simuladas ($MM)')
ax.set_title(f'Evaluación de {id_estrategia} — catálogo {catalogo_visualizado}')
ax.legend()
plt.tight_layout()
plt.show()

# --- Statistics: base catalogue + strategy ---
print(f'\n=== {id_estrategia} — Catálogo base ===')
print(f'Total {catalogue_length}-year loss median:            '
      f'${np.round(synthetic_annual_df.sum(axis=1).median(), 1)} MM')
print(f'Total {catalogue_length}-year coverage median:        '
      f'${np.round(total_coverage.sum(axis=1).median(), 1)} MM')
print(f'Total {catalogue_length}-year retention median:       '
      f'${np.round(synthetic_annual_df[total_coverage == 0].sum(axis=1).median(), 1)} MM')
print(f'Total {catalogue_length}-year uncovered median:       '
      f'${np.round((synthetic_annual_df[total_coverage != 0].sum(axis=1) - total_coverage.sum(axis=1)).median(), 1)} MM')
print(f'Total {catalogue_length}-year FISCAL retention median:'
      f' ${np.round(resp_fiscal * synthetic_annual_df[total_coverage == 0].sum(axis=1).median(), 1)} MM')
print(f'Total {catalogue_length}-year FISCAL uncovered median:'
      f' ${np.round(resp_fiscal * (synthetic_annual_df[total_coverage != 0].sum(axis=1) - total_coverage.sum(axis=1)).median(), 1)} MM')

# =============================================================================
# 5b. CBA ANALYSIS
# =============================================================================

from cba.config import get_default_config as get_cba_config
from cba.engine import run_cba
from cba.reports import generate_text_report

cba_config = get_cba_config()
cba_results = run_cba(
    losses_df=sim_result['synthetic_annual_df'],
    payout_dfs=drm_result['payout_dfs'],
    drm_configs=drm_configs,
    cba_config=cba_config,
)
print(generate_text_report(cba_results))
cba_report_path = OUTPUT_DIR / f'cba_report_{id_estrategia}.txt'
cba_report_path.write_text(generate_text_report(cba_results), encoding='utf-8')
print(f'CBA report saved to: {cba_report_path}')

# =============================================================================
# 6. RISK REDUCTION Mechanism
# =============================================================================

red = compute_reduction_schedule(inv, rbc, hor, discount_rate)

# --- Plot: DRR Investment per year ---
fig, ax = plt.subplots(figsize=(8, 5))
ax.bar(year_labels, inv, color='#1B5E79', label='DRR Investment')
for x, v in zip(year_labels, inv):
    label = f'${v:.1f}MM' if v > 0 else '$-'
    ax.text(x, v, label, ha='center', va='bottom', fontsize=9)
ax.yaxis.set_major_formatter(
    plt.FuncFormatter(lambda val, _: f'${val:.1f}MM' if val > 0 else '$-'))
ax.set_xticks(year_labels)
ax.set_xticklabels(year_labels)
ax.set_title('DRR Investment')
ax.legend()
plt.tight_layout()
plt.show()

reduced_result = generate_reduced_catalogue(
    lec_curve, red,
    sim_result['N_events'], sim_result['U_times'], sim_result['U_loss'],
    catalogue_length, simulation_number,
)

synthetic_annual_red_df = reduced_result['synthetic_annual_red_df']
event_catalogue_red     = reduced_result['event_catalogue_red']
lec_curves_by_target    = reduced_result['lec_curves_by_target']

# --- Plot 5: Family of reduced LEC curves ---
plt.figure()
ypos = 0.85
sorted_targets = sorted(lec_curves_by_target.keys())
for C_target in sorted_targets:
    data = lec_curves_by_target[C_target]
    plt.semilogx(data['Loss'], data['Lambda'], label=f'C_target={C_target:.1f}')
    plt.text(0.05, ypos,
             f'C={C_target:.1f}: AAL = ${data["aal"]:,.2f} MM',
             transform=plt.gca().transAxes, fontsize=9,
             bbox=dict(facecolor='white', edgecolor='black', boxstyle='round'))
    ypos -= 0.08
plt.xlabel('Loss [$MM]')
plt.ylabel('Annual Frequency of Exceedance')
plt.title('Curvas LEC reducidas (ex-ante)')
plt.legend(fontsize=8)
plt.grid(True)
plt.tight_layout()
plt.show()

# --- DRM on reduced catalogue ---
drm_red_result     = apply_strategy(event_catalogue_red, drm_configs, catalogue_length)
payout_red_dfs     = drm_red_result['payout_dfs']
total_coverage_red = drm_red_result['total_coverage']

# --- Plot 6: Stacked DRM payouts for reduced catalogue ---
loss_red_vals = synthetic_annual_red_df.loc[catalogo_visualizado].values
pr = [df.loc[catalogo_visualizado].values for df in payout_red_dfs]

fig, ax = plt.subplots(figsize=(10, 6))
ax.bar(year_labels, loss_red_vals, label='Pérdidas totales')
bottom = np.zeros(catalogue_length)
for cfg, payout in zip(drm_configs, pr):
    ax.bar(year_labels, payout, bottom=bottom, label=cfg['name'], hatch = '//')
    bottom += payout
for x, v in zip(year_labels, loss_red_vals):
    if v > 0:
        ax.text(x, v, f'${v:.1f}MM', ha='center', va='bottom', fontsize=8)
ax.set_ylim(0, Lim_axis)
ax.set_xticks(year_labels)
ax.set_xticklabels(year_labels)
ax.set_xlabel('Año')
ax.set_ylabel('Pérdidas económicas anuales simuladas ($MM)')
ax.set_title(f'Evaluación de {id_estrategia} — catálogo reducido {catalogo_visualizado}')
ax.legend()
plt.tight_layout()
plt.show()

# --- Statistics: reduced catalogue + strategy ---
print(f'\n=== {id_estrategia} — Catálogo reducido (ex-ante) ===')
print(f'Total {catalogue_length}-year loss median:            '
      f'${np.round(synthetic_annual_red_df.sum(axis=1).median(), 1)} MM')
print(f'Total {catalogue_length}-year coverage median:        '
      f'${np.round(total_coverage_red.sum(axis=1).median(), 1)} MM')
print(f'Total {catalogue_length}-year retention median:       '
      f'${np.round(synthetic_annual_red_df[total_coverage_red == 0].sum(axis=1).median(), 1)} MM')
print(f'Total {catalogue_length}-year uncovered median:       '
      f'${np.round((synthetic_annual_red_df[total_coverage_red != 0].sum(axis=1) - total_coverage_red.sum(axis=1)).median(), 1)} MM')
print(f'Total {catalogue_length}-year FISCAL retention median:'
      f' ${np.round(resp_fiscal * synthetic_annual_red_df[total_coverage_red == 0].sum(axis=1).median(), 1)} MM')
print(f'Total {catalogue_length}-year FISCAL uncovered median:'
      f' ${np.round(resp_fiscal * (synthetic_annual_red_df[total_coverage_red != 0].sum(axis=1) - total_coverage_red.sum(axis=1)).median(), 1)} MM')

drr_roi = ((synthetic_annual_df-synthetic_annual_red_df).cumsum(axis=1)<np.array(inv).sum()).sum(axis=1).median()
print(f'\nDRR Investment total: ${np.array(inv).sum():.1f} MM')
print(f'Median years until DRR investment is "paid back" by reduction in losses: {drr_roi} years (undiscounted)')

# =============================================================================
# 7. EXPORT STATISTICS TO CSV
# =============================================================================

def _stats_row(label, annual_df, coverage_df, fiscal_share):
    loss_med      = np.round(annual_df.sum(axis=1).median(), 1)
    cov_med       = np.round(coverage_df.sum(axis=1).median(), 1)
    ret_med       = np.round(annual_df[coverage_df == 0].sum(axis=1).median(), 1)
    uncov_med     = np.round(
        (annual_df[coverage_df != 0].sum(axis=1) - coverage_df.sum(axis=1)).median(), 1)
    f_ret_med     = np.round(fiscal_share * ret_med, 1)
    f_uncov_med   = np.round(fiscal_share * uncov_med, 1)
    return {
        'Scenario': label,
        f'Total {catalogue_length}-year loss median ($MM)':             loss_med,
        f'Total {catalogue_length}-year coverage median ($MM)':         cov_med,
        f'Total {catalogue_length}-year retention median ($MM)':        ret_med,
        f'Total {catalogue_length}-year uncovered median ($MM)':        uncov_med,
        f'Total {catalogue_length}-year FISCAL retention median ($MM)': f_ret_med,
        f'Total {catalogue_length}-year FISCAL uncovered median ($MM)': f_uncov_med,
    }

stats_rows = [
    _stats_row(f'{id_estrategia} — Catálogo base',
               synthetic_annual_df, total_coverage, resp_fiscal),
    _stats_row(f'{id_estrategia} — Catálogo reducido (ex-ante)',
               synthetic_annual_red_df, total_coverage_red, resp_fiscal),
]

stats_df = pd.DataFrame(stats_rows).set_index('Scenario').T
csv_path = OUTPUT_DIR / f'statistics_{id_estrategia}.csv'
stats_df.to_csv(csv_path, encoding='utf-8-sig')
print(f'\nStatistics exported to: {csv_path}')
