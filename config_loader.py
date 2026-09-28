# -*- coding: utf-8 -*-
"""
config_loader.py — LEC Tool
============================
Reads and validates the user configuration (config.toml) and turns it into
the objects the pipeline modules expect.

Public API
----------
load_config(path)            TOML file -> RunConfig (validated).
load_input_data(cfg)         Read the CSV inputs referenced by the config.
build_drm_configs(cfg, ppo)  Instrument list for risk_management.apply_strategy.
build_cba_config(cfg)        cba.config.LECCBAConfig for cba.engine.run_cba.
ConfigError                  Raised with a readable message on any invalid input.

Only this module knows the layout of config.toml; the rest of the code works
with RunConfig attributes.
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import pandas as pd


INSTRUMENT_TYPES = ('insurance', 'ppo', 'ccf', 'ddo')

# Payout parameters that risk_management.apply_strategy needs per type.
# Keys given as alternatives (dollars or return period, see ALTERNATIVE_KEYS)
# are checked separately.
REQUIRED_INSTRUMENT_KEYS = {
    'insurance': (),
    'ppo':       (),
    'ccf':       ('ccf_maximum', 'ccf_person', 'Pop_exposed'),
    'ddo':       ('ddo_available',),
}

# v8: thresholds can be given in dollars or as an event return period
# (converted with the loss curve of the country being run, see
# insurance_layer.py). Exactly one key of each pair is required.
ALTERNATIVE_KEYS = {
    'insurance': (('attachment_point', 'attachment_rp'),
                  ('exhaustion_point', 'exhaustion_rp'),
                  ('ceding_percentage', 'coverage_limit')),
    'ddo':       (('ddo_threshold', 'ddo_threshold_rp'),),
    'ppo':       (),
    'ccf':       (),
}
PPO_TRIGGER_MODES = ('ccf', 'loss')
PAYOUT_MODES = ('proportional', 'binary')

# Optional CBA cost overrides accepted per type (see cba.engine._build_instrument_config).
CBA_COST_KEYS = {
    'insurance': ('rate_on_line', 'premium'),
    'ppo':       ('commitment_fee_rate', 'loan_interest_rate', 'repayment_years',
                  'front_end_fee_rate', 'credit_line', 'grace_period_years'),
    'ccf':       ('drawdown_fee_rate', 'loan_interest_rate', 'repayment_years',
                  'grace_period_years'),
    'ddo':       ('loan_interest_rate', 'repayment_years', 'grace_period_years'),
}

_INVALID_ID_CHARS = set('<>:"/\\|?*')


class ConfigError(ValueError):
    """Invalid or missing entry in config.toml."""


# ---------------------------------------------------------------------------
# Configuration dataclasses (one per TOML section)
# ---------------------------------------------------------------------------

@dataclass
class RunSection:
    id: str
    output_dir: Path
    show_figures: bool = False
    figure_dpi: int = 130


@dataclass
class InputsSection:
    event_loss_file: Path
    tail_curve_file: Optional[Path] = None
    ppo_schedule_file: Optional[Path] = None


@dataclass
class LecSection:
    loss_scale_factor: float = 1.0
    freq_scale_factor: float = 1.0
    hybrid_curve: bool = True
    bootstrap_samples: int = 1000
    pml_return_periods: list = field(default_factory=lambda: [200])
    gap_threshold_fractions: list = field(default_factory=lambda: [0.25, 0.5, 0.75, 1.0])


@dataclass
class SimulationSection:
    catalogue_length: int
    simulation_number: int
    random_seed: Optional[int]
    displayed_catalogue: int
    first_year: int


@dataclass
class FiscalSection:
    resp_fiscal: float


@dataclass
class RiskReductionSection:
    enabled: bool = False
    discount_rate: float = 0.0
    investment: list = field(default_factory=list)
    benefit_cost_ratio: list = field(default_factory=list)
    benefit_horizon: list = field(default_factory=list)


@dataclass
class CBASection:
    enabled: bool = False
    loss_basis: str = 'total'
    social_discount_rate: float = 0.05
    indirect_benefit_factor: float = 0.10
    omv_lambda: float = 0.05
    legacy_truncation: bool = False
    defaults: dict = field(default_factory=dict)   # {'insurance': {...}, 'ppo': {...}, ...}
    cnc: Optional[dict] = None                     # None when [cba.cnc] is absent


@dataclass
class RunConfig:
    run: RunSection
    inputs: InputsSection
    lec: LecSection
    simulation: SimulationSection
    fiscal: FiscalSection
    instruments: list            # list of dicts as written in config.toml
    risk_reduction: RiskReductionSection
    cba: CBASection
    source_path: Path            # the config.toml that was read
    raw: dict                    # parsed TOML, for traceability
    insurance_pricing: Any = None  # insurance_layer.PricingSettings

    # Convenience predicates -------------------------------------------------
    @property
    def has_instruments(self) -> bool:
        return len(self.instruments) > 0

    @property
    def has_ppo(self) -> bool:
        return any(i['type'] == 'ppo' for i in self.instruments)

    @property
    def output_path(self) -> Path:
        return self.run.output_dir / self.run.id


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

_MISSING = object()


def _section(raw: dict, name: str, required: bool = True) -> dict:
    sec = raw.get(name)
    if sec is None:
        if required:
            raise ConfigError(f"Section [{name}] is missing from the configuration file.")
        return {}
    if not isinstance(sec, dict):
        raise ConfigError(f"[{name}] must be a table.")
    return sec


def _get(sec: dict, key: str, section: str, default: Any = _MISSING, types=None):
    if key not in sec:
        if default is _MISSING:
            raise ConfigError(f"[{section}] is missing the required key '{key}'.")
        return default
    value = sec[key]
    if types is not None and not isinstance(value, types):
        expected = ' or '.join(t.__name__ for t in (types if isinstance(types, tuple) else (types,)))
        raise ConfigError(f"[{section}] '{key}' must be {expected}, got {value!r}.")
    return value


def _number(sec, key, section, default=_MISSING, lo=None, hi=None, integer=False):
    types = int if integer else (int, float)
    value = _get(sec, key, section, default, types)
    if isinstance(value, bool):
        raise ConfigError(f"[{section}] '{key}' must be a number, got a boolean.")
    if lo is not None and value < lo:
        raise ConfigError(f"[{section}] '{key}' must be >= {lo}, got {value}.")
    if hi is not None and value > hi:
        raise ConfigError(f"[{section}] '{key}' must be <= {hi}, got {value}.")
    return value


def _number_list(sec, key, section, default=_MISSING, lo=None, length=None):
    value = _get(sec, key, section, default, list)
    if any(isinstance(v, bool) or not isinstance(v, (int, float)) for v in value):
        raise ConfigError(f"[{section}] '{key}' must be a list of numbers.")
    if lo is not None and any(v < lo for v in value):
        raise ConfigError(f"[{section}] every value of '{key}' must be >= {lo}.")
    if length is not None and len(value) != length:
        raise ConfigError(
            f"[{section}] '{key}' has {len(value)} values but simulation.catalogue_length = {length}."
        )
    return list(value)


def _path(sec, key, section, base: Path, required: bool) -> Optional[Path]:
    value = _get(sec, key, section, _MISSING if required else None, str)
    if value is None:
        return None
    return (base / value).resolve() if not Path(value).is_absolute() else Path(value)


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------

def load_config(path: Optional[str | Path] = None) -> RunConfig:
    """
    Read config.toml (default: the one next to main.py) and validate it.

    Raises ConfigError with a message naming the offending section and key.
    """
    path = Path(path) if path is not None else Path(__file__).parent / 'config.toml'
    if not path.exists():
        raise ConfigError(f"Configuration file not found: {path}")
    with open(path, 'rb') as fh:
        try:
            raw = tomllib.load(fh)
        except tomllib.TOMLDecodeError as exc:
            raise ConfigError(f"{path.name} is not valid TOML: {exc}") from exc
    base = path.parent

    # --- [run] --------------------------------------------------------------
    sec = _section(raw, 'run')
    run_id = _get(sec, 'id', 'run', types=str).strip()
    if not run_id:
        raise ConfigError("[run] 'id' must be a non-empty string.")
    bad = _INVALID_ID_CHARS & set(run_id)
    if bad:
        raise ConfigError(f"[run] 'id' contains characters not allowed in file names: {sorted(bad)}")
    output_dir = _get(sec, 'output_dir', 'run', 'outputs', str)
    run = RunSection(
        id=run_id,
        output_dir=(base / output_dir) if not Path(output_dir).is_absolute() else Path(output_dir),
        show_figures=_get(sec, 'show_figures', 'run', False, bool),
        figure_dpi=_number(sec, 'figure_dpi', 'run', 130, lo=30, integer=True),
    )

    # --- [lec] --------------------------------------------------------------
    sec = _section(raw, 'lec', required=False)
    lec = LecSection(
        loss_scale_factor=_number(sec, 'loss_scale_factor', 'lec', 1.0, lo=1e-12),
        freq_scale_factor=_number(sec, 'freq_scale_factor', 'lec', 1.0, lo=1e-12),
        hybrid_curve=_get(sec, 'hybrid_curve', 'lec', True, bool),
        bootstrap_samples=_number(sec, 'bootstrap_samples', 'lec', 1000, lo=1, integer=True),
        pml_return_periods=_number_list(sec, 'pml_return_periods', 'lec', [200], lo=1),
        gap_threshold_fractions=_number_list(sec, 'gap_threshold_fractions', 'lec',
                                             [0.25, 0.5, 0.75, 1.0], lo=1e-12),
    )

    # --- [simulation] -------------------------------------------------------
    sec = _section(raw, 'simulation')
    catalogue_length = _number(sec, 'catalogue_length', 'simulation', lo=1, integer=True)
    simulation_number = _number(sec, 'simulation_number', 'simulation', lo=1, integer=True)
    seed = sec.get('random_seed')
    if seed is not None and (isinstance(seed, bool) or not isinstance(seed, int)):
        raise ConfigError("[simulation] 'random_seed' must be an integer (or remove the line).")
    displayed = _number(sec, 'displayed_catalogue', 'simulation', 0, lo=0, integer=True)
    if displayed >= simulation_number:
        raise ConfigError(
            f"[simulation] 'displayed_catalogue' ({displayed}) must be smaller than "
            f"simulation_number ({simulation_number})."
        )
    simulation = SimulationSection(
        catalogue_length=catalogue_length,
        simulation_number=simulation_number,
        random_seed=seed,
        displayed_catalogue=displayed,
        first_year=_number(sec, 'first_year', 'simulation', datetime.now().year + 1, integer=True),
    )

    # --- [fiscal] -----------------------------------------------------------
    sec = _section(raw, 'fiscal')
    fiscal = FiscalSection(resp_fiscal=_number(sec, 'resp_fiscal', 'fiscal', lo=0.0, hi=1.0))

    # --- [[strategy.instruments]] -------------------------------------------
    strategy = _section(raw, 'strategy', required=False)
    instruments = strategy.get('instruments', [])
    if not isinstance(instruments, list):
        raise ConfigError("[strategy] 'instruments' must be an array of tables ([[strategy.instruments]]).")
    names = set()
    for k, inst in enumerate(instruments, start=1):
        label = f"strategy.instruments #{k}"
        if not isinstance(inst, dict):
            raise ConfigError(f"[{label}] must be a table.")
        itype = _get(inst, 'type', label, types=str)
        if itype not in INSTRUMENT_TYPES:
            raise ConfigError(f"[{label}] unknown type '{itype}'. Supported: {', '.join(INSTRUMENT_TYPES)}.")
        name = _get(inst, 'name', label, types=str).strip()
        if not name:
            raise ConfigError(f"[{label}] 'name' must be a non-empty string.")
        if name in names:
            raise ConfigError(f"Instrument name '{name}' is used more than once; names must be unique.")
        names.add(name)
        for key in REQUIRED_INSTRUMENT_KEYS[itype]:
            if key not in inst:
                raise ConfigError(f"[{label}] ('{name}', type '{itype}') is missing '{key}'.")
        if 'interest_rate' in inst:
            raise ConfigError(
                f"[{label}] ('{name}') uses 'interest_rate'; the key is 'loan_interest_rate'."
            )
        if 'ppo_schedule' in inst:
            raise ConfigError(
                f"[{label}] ('{name}') must not define 'ppo_schedule'; it is read from inputs.ppo_schedule_file."
            )
        _validate_instrument_v8(inst, itype, label, name)
    types_present = {i['type'] for i in instruments}
    ccf_triggered_ppo = any(
        i['type'] == 'ppo' and i.get('ppo_trigger_mode', 'loss' if (
            'ppo_loss_trigger' in i or 'ppo_loss_trigger_rp' in i) else 'ccf') == 'ccf'
        for i in instruments
    )
    if ccf_triggered_ppo and 'ccf' not in types_present:
        raise ConfigError(
            "A 'ppo' instrument triggered by the CCF requires a 'ccf' instrument in the "
            "same strategy. To trigger the PPO on its own loss threshold, give it "
            "'ppo_loss_trigger' ($MM) or 'ppo_loss_trigger_rp' (years)."
        )

    # --- [inputs] -----------------------------------------------------------
    sec = _section(raw, 'inputs')
    inputs = InputsSection(
        event_loss_file=_path(sec, 'event_loss_file', 'inputs', base, required=True),
        tail_curve_file=_path(sec, 'tail_curve_file', 'inputs', base, required=lec.hybrid_curve),
        ppo_schedule_file=_path(sec, 'ppo_schedule_file', 'inputs', base, required='ppo' in types_present),
    )
    if not inputs.event_loss_file.exists():
        raise ConfigError(f"[inputs] event_loss_file not found: {inputs.event_loss_file}")
    if lec.hybrid_curve and not inputs.tail_curve_file.exists():
        raise ConfigError(f"[inputs] tail_curve_file not found: {inputs.tail_curve_file}")
    if 'ppo' in types_present and not inputs.ppo_schedule_file.exists():
        raise ConfigError(f"[inputs] ppo_schedule_file not found: {inputs.ppo_schedule_file}")

    # --- [risk_reduction] ---------------------------------------------------
    sec = _section(raw, 'risk_reduction', required=False)
    rr_enabled = _get(sec, 'enabled', 'risk_reduction', False, bool)
    if rr_enabled:
        risk_reduction = RiskReductionSection(
            enabled=True,
            discount_rate=_number(sec, 'discount_rate', 'risk_reduction', lo=0.0, hi=1.0),
            investment=_number_list(sec, 'investment', 'risk_reduction', lo=0.0, length=catalogue_length),
            benefit_cost_ratio=_number_list(sec, 'benefit_cost_ratio', 'risk_reduction', lo=0.0,
                                            length=catalogue_length),
            benefit_horizon=_number_list(sec, 'benefit_horizon', 'risk_reduction', lo=0.0,
                                         length=catalogue_length),
        )
    else:
        risk_reduction = RiskReductionSection(enabled=False)

    # --- [cba] --------------------------------------------------------------
    sec = _section(raw, 'cba', required=False)
    cba_enabled = _get(sec, 'enabled', 'cba', False, bool)
    if cba_enabled:
        loss_basis = _get(sec, 'loss_basis', 'cba', 'total', str).lower()
        if loss_basis not in ('total', 'fiscal'):
            raise ConfigError("[cba] 'loss_basis' must be 'total' or 'fiscal'.")
        defaults = _get(sec, 'defaults', 'cba', {}, dict)
        for itype, table in defaults.items():
            if itype not in INSTRUMENT_TYPES or not isinstance(table, dict):
                raise ConfigError(f"[cba.defaults.{itype}] is not a recognised instrument type table.")
            for key in table:
                if key not in CBA_COST_KEYS[itype]:
                    raise ConfigError(
                        f"[cba.defaults.{itype}] unknown key '{key}'. "
                        f"Allowed: {', '.join(CBA_COST_KEYS[itype])}."
                    )
        cnc_sec = sec.get('cnc')
        cnc = None
        if cnc_sec is not None:
            if not isinstance(cnc_sec, dict):
                raise ConfigError("[cba.cnc] must be a table.")
            tiers = _get(cnc_sec, 'spread_tiers', 'cba.cnc', None, list)
            if tiers is not None:
                for t in tiers:
                    if (not isinstance(t, list) or len(t) != 2
                            or any(isinstance(v, bool) or not isinstance(v, (int, float)) for v in t)):
                        raise ConfigError("[cba.cnc] 'spread_tiers' must be a list of [threshold, spread] pairs.")
            cnc = {
                'enabled': _get(cnc_sec, 'enabled', 'cba.cnc', True, bool),
                'gdp': _number(cnc_sec, 'gdp', 'cba.cnc', lo=1e-9),
                'base_rate': _number(cnc_sec, 'base_rate', 'cba.cnc', lo=0.0, hi=1.0),
                'sovereign_base_spread': _number(cnc_sec, 'sovereign_base_spread', 'cba.cnc', lo=0.0, hi=1.0),
                'ex_post_term_years': _number(cnc_sec, 'ex_post_term_years', 'cba.cnc', 10, lo=1, integer=True),
                'spread_tiers': [tuple(float(v) for v in t) for t in tiers] if tiers is not None else None,
            }
        cba = CBASection(
            enabled=True,
            loss_basis=loss_basis,
            social_discount_rate=_number(sec, 'social_discount_rate', 'cba', 0.05, lo=0.0, hi=1.0),
            indirect_benefit_factor=_number(sec, 'indirect_benefit_factor', 'cba', 0.10, lo=0.0),
            omv_lambda=_number(sec, 'omv_lambda', 'cba', 0.05, lo=0.0),
            legacy_truncation=_get(sec, 'legacy_truncation', 'cba', False, bool),
            defaults=defaults,
            cnc=cnc,
        )
    else:
        cba = CBASection(enabled=False)

    insurance_pricing = _load_insurance_pricing(raw, cba)

    return RunConfig(
        run=run, inputs=inputs, lec=lec, simulation=simulation, fiscal=fiscal,
        instruments=[dict(i) for i in instruments], risk_reduction=risk_reduction,
        cba=cba, source_path=path.resolve(), raw=raw,
        insurance_pricing=insurance_pricing,
    )


# ---------------------------------------------------------------------------
# v8: instrument keys and [insurance_pricing]
# ---------------------------------------------------------------------------

def _validate_instrument_v8(inst, itype, label, name):
    """Checks for the v8 keys (return periods, coverage limit, pricing)."""
    from insurance_layer import PRICING_METHODS
    where = f"[{label}] ('{name}')"
    for dollars, alt in ALTERNATIVE_KEYS[itype]:
        has_d, has_a = dollars in inst, alt in inst
        if has_d and has_a:
            raise ConfigError(f"{where} gives both '{dollars}' and '{alt}'; keep only one.")
        if not (has_d or has_a):
            raise ConfigError(f"{where} needs '{dollars}' or '{alt}'.")
    for key in ('attachment_rp', 'exhaustion_rp', 'ddo_threshold_rp', 'ppo_loss_trigger_rp'):
        if key in inst:
            _number(inst, key, f"{label}", lo=1.0)
    for key in ('coverage_limit', 'gross_premium', 'premium'):
        if key in inst:
            _number(inst, key, f"{label}", lo=1e-12)
    if 'donor_discount' in inst:
        _number(inst, 'donor_discount', f"{label}", lo=0.0)
    if 'donor_discount_share' in inst:
        _number(inst, 'donor_discount_share', f"{label}", lo=0.0, hi=1.0)
    if 'ceding_percentage' in inst:
        _number(inst, 'ceding_percentage', f"{label}", lo=0.0, hi=1.0)
    if 'attachment_rp' in inst and 'exhaustion_rp' in inst and inst['exhaustion_rp'] <= inst['attachment_rp']:
        raise ConfigError(f"{where}: exhaustion_rp must be larger than attachment_rp.")
    if 'pricing' in inst and inst['pricing'] not in PRICING_METHODS:
        raise ConfigError(f"{where}: pricing must be one of {', '.join(PRICING_METHODS)}.")
    if inst.get('pricing') == 'quote' and not ('gross_premium' in inst or 'premium' in inst):
        raise ConfigError(f"{where}: pricing = 'quote' needs 'gross_premium' ($MM, from the quotation).")
    if 'payout_mode' in inst and inst['payout_mode'] not in PAYOUT_MODES:
        raise ConfigError(f"{where}: payout_mode must be 'proportional' or 'binary'.")
    for key in ('one_payout_per_year', 'payout_floor', 'ppo_require_available_funds'):
        if key in inst and not isinstance(inst[key], bool):
            raise ConfigError(f"{where}: '{key}' must be true or false.")
    if itype == 'ppo':
        mode = inst.get('ppo_trigger_mode')
        if mode is not None and mode not in PPO_TRIGGER_MODES:
            raise ConfigError(f"{where}: ppo_trigger_mode must be 'ccf' or 'loss'.")
        has_trigger = 'ppo_loss_trigger' in inst or 'ppo_loss_trigger_rp' in inst
        if 'ppo_loss_trigger' in inst and 'ppo_loss_trigger_rp' in inst:
            raise ConfigError(f"{where} gives both 'ppo_loss_trigger' and 'ppo_loss_trigger_rp'; keep only one.")
        if mode == 'loss' and not has_trigger:
            raise ConfigError(f"{where}: ppo_trigger_mode = 'loss' needs 'ppo_loss_trigger' or 'ppo_loss_trigger_rp'.")


def _load_insurance_pricing(raw, cba):
    """[insurance_pricing] -> insurance_layer.PricingSettings (all keys optional)."""
    from insurance_layer import PricingSettings, PRICING_METHODS
    sec = _section(raw, 'insurance_pricing', required=False)
    s = PricingSettings()
    legacy_rol = cba.defaults.get('insurance', {}).get('rate_on_line') if cba.enabled else None
    method = _get(sec, 'method', 'insurance_pricing', s.method, str)
    if method not in PRICING_METHODS or method == 'quote':
        raise ConfigError(
            "[insurance_pricing] 'method' must be 'rol_rule', 'market_curve' or 'fixed_rol' "
            "('quote' is set per instrument with gross_premium)."
        )
    s.method = method
    s.cutoff_rp = _number(sec, 'cutoff_rp', 'insurance_pricing', s.cutoff_rp, lo=1.0)
    s.flat_rol = _number(sec, 'flat_rol', 'insurance_pricing', s.flat_rol, lo=1e-6, hi=1.0)
    s.rate_on_line = _number(sec, 'rate_on_line', 'insurance_pricing',
                             legacy_rol if legacy_rol is not None else s.rate_on_line, lo=1e-6, hi=1.0)
    rp = _number_list(sec, 'market_curve_rp', 'insurance_pricing', list(s.curve_rp), lo=1e-9)
    rol = _number_list(sec, 'market_curve_rol', 'insurance_pricing', list(s.curve_rol), lo=1e-9)
    if len(rp) != len(rol) or len(rp) < 2:
        raise ConfigError("[insurance_pricing] 'market_curve_rp' and 'market_curve_rol' need the same length (>= 2).")
    if any(b <= a for a, b in zip(rp, rp[1:])):
        raise ConfigError("[insurance_pricing] 'market_curve_rp' must be strictly increasing.")
    if any(v > 1.0 for v in rol):
        raise ConfigError("[insurance_pricing] 'market_curve_rol' values are rates (0.25 = 25 %).")
    s.curve_rp, s.curve_rol = rp, rol
    s.curve_date = str(_get(sec, 'market_curve_date', 'insurance_pricing', s.curve_date, str))
    s.payout_mode = _get(sec, 'payout_mode', 'insurance_pricing', s.payout_mode, str)
    if s.payout_mode not in PAYOUT_MODES:
        raise ConfigError("[insurance_pricing] 'payout_mode' must be 'proportional' or 'binary'.")
    s.one_payout_per_year = _get(sec, 'one_payout_per_year', 'insurance_pricing', s.one_payout_per_year, bool)
    s.payout_floor = _get(sec, 'payout_floor', 'insurance_pricing', s.payout_floor, bool)
    return s


# ---------------------------------------------------------------------------
# Input data
# ---------------------------------------------------------------------------

def load_input_data(cfg: RunConfig) -> dict:
    """
    Read the CSV inputs referenced by the configuration.

    Returns
    -------
    dict with keys:
      event_loss_df   DataFrame with columns 'year', 'econ_loss'.
      tail_loss       ndarray or None (scaled by loss_scale_factor).
      tail_aep        ndarray or None (scaled by freq_scale_factor).
      ppo_schedule    list of floats or None (only when a ppo instrument exists).
    """
    event_loss_df = pd.read_csv(cfg.inputs.event_loss_file)
    for col in ('year', 'econ_loss'):
        if col not in event_loss_df.columns:
            raise ConfigError(f"{cfg.inputs.event_loss_file.name} must contain a '{col}' column.")
    if event_loss_df.empty:
        raise ConfigError(f"{cfg.inputs.event_loss_file.name} contains no events.")

    tail_loss = tail_aep = None
    if cfg.lec.hybrid_curve:
        tail_df = pd.read_csv(cfg.inputs.tail_curve_file)
        for col in ('tail_loss', 'tail_aep'):
            if col not in tail_df.columns:
                raise ConfigError(f"{cfg.inputs.tail_curve_file.name} must contain a '{col}' column.")
        tail_loss = cfg.lec.loss_scale_factor * tail_df['tail_loss'].to_numpy(dtype=float)
        tail_aep = cfg.lec.freq_scale_factor * tail_df['tail_aep'].to_numpy(dtype=float)

    ppo_schedule = None
    if cfg.has_ppo:
        ppo_schedule = pd.read_csv(cfg.inputs.ppo_schedule_file, header=None).iloc[0].tolist()
        n = cfg.simulation.catalogue_length
        if len(ppo_schedule) != n:
            raise ConfigError(
                f"{cfg.inputs.ppo_schedule_file.name} has {len(ppo_schedule)} values "
                f"but simulation.catalogue_length = {n}."
            )
        ppo_schedule = [float(v) for v in ppo_schedule]

    return {
        'event_loss_df': event_loss_df,
        'tail_loss': tail_loss,
        'tail_aep': tail_aep,
        'ppo_schedule': ppo_schedule,
    }


# ---------------------------------------------------------------------------
# Objects for the pipeline modules
# ---------------------------------------------------------------------------

def build_drm_configs(cfg: RunConfig, ppo_schedule: Optional[list]) -> list:
    """
    Instrument dicts for risk_management.apply_strategy (and cba.engine.run_cba).

    The PPO schedule is attached to every instrument of type 'ppo' by type
    lookup, so the position of the PPO in the list does not matter and a
    strategy without a PPO never touches the schedule file.
    """
    drm_configs = []
    for inst in cfg.instruments:
        cfg_dict = dict(inst)
        if cfg_dict['type'] == 'ppo':
            if ppo_schedule is None:
                raise ConfigError("A ppo instrument is declared but no PPO schedule was loaded.")
            cfg_dict['ppo_schedule'] = list(ppo_schedule)
        drm_configs.append(cfg_dict)
    return drm_configs


def build_cba_config(cfg: RunConfig):
    """
    Build cba.config.LECCBAConfig from the [cba] section.

    The discount horizon and simulation count are taken from [simulation] so
    present values always cover the full catalogue length.
    """
    from cba.config import (
        LECCBAConfig, DiscountConfig, IndirectBenefitConfig, OMVConfig,
        InsuranceConfig, PPOConfig, CCFConfig, DDOConfig,
    )
    from cba.cnc import CNCConfig

    c = cfg.cba
    d = c.defaults

    def _make(cls, itype):
        try:
            return cls(**d.get(itype, {}))
        except (TypeError, ValueError) as exc:
            raise ConfigError(f"[cba.defaults.{itype}]: {exc}") from exc

    if c.cnc is None:
        cnc = CNCConfig(enabled=False)
    else:
        kwargs = {k: v for k, v in c.cnc.items() if v is not None}
        try:
            cnc = CNCConfig(**kwargs)
        except ValueError as exc:
            raise ConfigError(f"[cba.cnc]: {exc}") from exc

    return LECCBAConfig(
        discount=DiscountConfig(
            social_discount_rate=c.social_discount_rate,
            analysis_horizon=cfg.simulation.catalogue_length,
            num_simulations=cfg.simulation.simulation_number,
        ),
        indirect_benefit=IndirectBenefitConfig(factor=c.indirect_benefit_factor),
        omv=OMVConfig(lambda_risk_adjustment=c.omv_lambda),
        insurance=_make(InsuranceConfig, 'insurance'),
        ppo=_make(PPOConfig, 'ppo'),
        ccf=_make(CCFConfig, 'ccf'),
        ddo=_make(DDOConfig, 'ddo'),
        cnc=cnc,
        legacy_truncation=c.legacy_truncation,
    )
