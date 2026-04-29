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
from risk_reduction import generate_reduced_catalogue

# =============================================================================
# 1. PARAMETERS
# =============================================================================

DATA_DIR = Path(__file__).parent / 'data'

# --- Global ---
resp_fiscal       = 0.60   # fiscal responsibility share (0–1)
loss_scale_factor = 1.0    # multiplicative scale on all losses
freq_scale_factor = 1.0    # multiplicative scale on all exceedance rates
curva_hibrida     = True   # blend empirical LEC with probabilistic tail

# --- Simulation ---
catalogue_length  = 10     # years per synthetic catalogue
simulation_number = 1000   # number of independent catalogues (1–1000)
random_seed       = 42     # for reproducibility, set to None for random number generation

# --- Visualization ---
catalogo_visualizado = 360  # which catalogue to display in per-catalogue plots (0-(simulation_number-1))

# --- DRM strategy: Estrategia 1 (CCRIF + PPO + CCF) ---
# PPO schedule is loaded from file; placeholder here, overwritten below.
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
        'ppo_loss_trigger': 120,      # $MM
    },
    {
        'name': 'BID CCF',
        'type': 'ccf',
        'ccf_maximum': 300,           # $MM total facility
        'ccf_person': 1650,           # $ per affected person
        'Pop_exposed': 10.83e6,       # exposed population
    },
    {
        'name': 'BID DDO',
        'type': 'ddo',
        'ddo_threshold': 120,         # $MM — trigger loss level
        'ddo_available': 90,          # $MM — fixed payout when triggered
    },
]

id_estrategia = 'Estrategia 1'

# --- Ex-ante reduction ---
year_ini = datetime.now().year + 1   # first year of the simulation horizon (for axis labels only)
red = [0.0, 2.7, 2.7, 8.0, 8.0, 18.7, 18.7, 24.1, 24.1, 26.8]  # per-year cumulative AAL reduction ($MM)

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
# 6. RISK REDUCTION Mechanism
# =============================================================================

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
