"""
LEC-CBA IO Bridge Module
=========================
Interface between the BID's LEC tool and the CBA module.

TWO OPERATING MODES:

  Mode A — PRODUCTION (read from BID's LEC tool):
    Reads the synthetic catalog and instrument payouts that the LEC
    tool already generated. No simulation is duplicated. The CBA
    module only adds cost functions, discounting, and indicators.

    Expected inputs from LEC tool:
      - synthetic_annual_df.csv  (matrix: simulations × years, annual losses)
      - drm1_payout_df.csv      (insurance payouts, same shape)
      - drm2_payout_df.csv      (PPO/contingent credit payouts)
      - drm3_payout_df.csv      (CCF/reserve fund payouts)

  Mode B — TESTING (self-generated catalog):
    When the LEC tool is not available, generates a simplified
    synthetic catalog from the LEC curve using inverse transform
    sampling. This is for development and demonstration only.

    NOTE: The testing mode uses a simplified sampling method that
    draws one loss per year directly from the LEC. The BID's tool
    uses a Poisson process that generates multiple individual events
    per year and aggregates them. Distributions will differ.
"""

import numpy as np
import pandas as pd
import csv
import os
import logging
from dataclasses import dataclass, field
from typing import Tuple, Optional, Dict

logger = logging.getLogger("lec-cba.io_bridge")


# ============================================================
# Data Container
# ============================================================

@dataclass
class LECData:
    """Standardized container for LEC tool outputs."""

    # LEC curve (optional in production mode — may not be needed)
    losses: Optional[np.ndarray] = None           # (num_points,) loss values M USD
    exceedance_probs: Optional[np.ndarray] = None  # (num_points,) P(L > x)

    # Synthetic catalog — the core input for CBA
    synthetic_annual_losses: Optional[np.ndarray] = None  # (sims, years)

    # Instrument payouts from LEC tool (production mode)
    instrument_payouts: Dict[str, np.ndarray] = field(default_factory=dict)
    # e.g. {"drm1_insurance": array, "drm2_ppo": array, "drm3_ccf": array}

    # Metadata
    mode: str = "unknown"  # "production" or "testing"

    @property
    def num_points(self) -> int:
        if self.losses is not None:
            return len(self.losses)
        return 0

    @property
    def num_simulations(self) -> int:
        if self.synthetic_annual_losses is not None:
            return self.synthetic_annual_losses.shape[0]
        return 0

    @property
    def horizon_years(self) -> int:
        if self.synthetic_annual_losses is not None:
            return self.synthetic_annual_losses.shape[1]
        return 0

    @property
    def return_periods(self) -> Optional[np.ndarray]:
        """Return periods in years (avoiding division by zero)."""
        if self.exceedance_probs is not None:
            return np.where(self.exceedance_probs > 0,
                            1.0 / self.exceedance_probs, np.inf)
        return None

    @property
    def aal(self) -> float:
        """Average Annual Loss via trapezoidal integration of LEC."""
        if self.losses is None or self.exceedance_probs is None:
            # Estimate from catalog if LEC curve not available
            if self.synthetic_annual_losses is not None:
                return float(np.mean(self.synthetic_annual_losses))
            return 0.0
        total = 0.0
        for i in range(len(self.losses) - 1):
            dp = self.exceedance_probs[i] - self.exceedance_probs[i + 1]
            avg_loss = (self.losses[i] + self.losses[i + 1]) / 2.0
            total += dp * avg_loss
        return total


# ============================================================
# MODE A: Production — Read from BID's LEC Tool
# ============================================================

def load_from_lec_tool(
    synthetic_catalog_path: str,
    drm1_payout_path: str = None,
    drm2_payout_path: str = None,
    drm3_payout_path: str = None,
    lec_curve_path: str = None,
    government_exposure_factor: float = 1.0
) -> LECData:
    """
    PRODUCTION MODE: Load pre-computed outputs from the BID's LEC tool.

    This is the primary integration path. The LEC tool has already:
      1. Built the empirical LEC curve
      2. Generated the synthetic catalog via Poisson process
      3. Calculated instrument payouts (drm1, drm2, drm3)

    We read these outputs directly. No simulation is duplicated.
    The CBA module only adds cost evaluation and indicators.

    Args:
        synthetic_catalog_path: CSV of annual losses (simulations × years)
            This is the LEC tool's 'synthetic_annual_df.csv'
        drm1_payout_path: CSV of insurance payouts (optional)
        drm2_payout_path: CSV of PPO/contingent credit payouts (optional)
        drm3_payout_path: CSV of CCF/reserve fund payouts (optional)
        lec_curve_path: CSV of LEC curve (optional, for reference)
        government_exposure_factor: fiscal share of total losses
    """
    logger.info(f"PRODUCTION MODE: Loading from LEC tool outputs")

    lec = LECData(mode="production")

    # Load synthetic catalog (required)
    logger.info(f"  Reading synthetic catalog: {synthetic_catalog_path}")
    catalog_df = pd.read_csv(synthetic_catalog_path)
    lec.synthetic_annual_losses = catalog_df.values

    logger.info(f"  Catalog shape: {lec.synthetic_annual_losses.shape} "
                f"({lec.num_simulations} sims x {lec.horizon_years} years)")

    # Load instrument payouts (optional — if not provided, CBA module
    # calculates its own payouts using cost functions)
    payout_files = {
        "drm1_insurance": drm1_payout_path,
        "drm2_ppo": drm2_payout_path,
        "drm3_ccf": drm3_payout_path,
    }
    for name, path in payout_files.items():
        if path is not None and os.path.exists(path):
            logger.info(f"  Reading {name} payouts: {path}")
            payout_df = pd.read_csv(path)
            lec.instrument_payouts[name] = payout_df.values

    # Load LEC curve (optional, for reference/AAL calculation)
    if lec_curve_path is not None and os.path.exists(lec_curve_path):
        logger.info(f"  Reading LEC curve: {lec_curve_path}")
        lec_curve = _load_lec_curve_csv(lec_curve_path)
        lec.losses = lec_curve[0]
        lec.exceedance_probs = lec_curve[1]

    # Apply government exposure factor
    if government_exposure_factor != 1.0:
        lec = apply_government_exposure(lec, government_exposure_factor)
        logger.info(f"  Applied government exposure factor: "
                    f"{government_exposure_factor:.0%}")

    logger.info(f"  Fiscal AAL: ${lec.aal:.1f}M")
    return lec


# ============================================================
# MODE B: Testing — Self-generated Catalog
# ============================================================

def load_for_testing(
    lec_curve_path: str,
    num_simulations: int = 1000,
    horizon_years: int = 10,
    government_exposure_factor: float = 1.0,
    seed: int = 42
) -> LECData:
    """
    TESTING MODE: Generate synthetic catalog from LEC curve.

    Used when the BID's LEC tool is not available. Generates a
    simplified catalog for development and demonstration.

    IMPORTANT: This method uses a simplified inverse transform
    sampling (one draw per year from the LEC). The BID's LEC tool
    uses a homogeneous Poisson process that:
      1. Generates event arrival times: exponential inter-arrivals
      2. Assigns losses to each event via inverse CDF sampling
      3. Aggregates all events within each year
    This means a single year in the BID's tool can have multiple
    events whose losses are summed. Our simplified method draws
    one aggregate loss per year directly from the LEC curve.
    Results are adequate for testing CBA indicators but the loss
    distributions will not be identical to the LEC tool's output.

    Args:
        lec_curve_path: CSV with LEC curve (loss, exceedance_prob)
        num_simulations: number of Monte Carlo runs
        horizon_years: analysis horizon in years
        government_exposure_factor: fiscal share of total losses
        seed: random seed for reproducibility
    """
    logger.info(f"TESTING MODE: Generating synthetic catalog from LEC curve")

    losses, probs = _load_lec_curve_csv(lec_curve_path)

    lec = LECData(
        losses=losses,
        exceedance_probs=probs,
        mode="testing"
    )

    # Simplified inverse transform sampling
    rng = np.random.default_rng(seed)
    sorted_idx = np.argsort(probs)
    sorted_probs = probs[sorted_idx]
    sorted_losses = losses[sorted_idx]

    u = rng.uniform(0, 1, size=(num_simulations, horizon_years))
    catalog = np.interp(u, sorted_probs, sorted_losses)
    lec.synthetic_annual_losses = catalog

    logger.info(f"  LEC points: {lec.num_points}")
    logger.info(f"  Catalog shape: {catalog.shape}")
    logger.info(f"  Total AAL: ${lec.aal:.1f}M")

    # Apply government exposure factor
    if government_exposure_factor != 1.0:
        lec = apply_government_exposure(lec, government_exposure_factor)
        logger.info(f"  Applied government exposure factor: "
                    f"{government_exposure_factor:.0%}")
        logger.info(f"  Fiscal AAL: ${lec.aal:.1f}M")

    return lec


# ============================================================
# Government Exposure Factor
# ============================================================

def apply_government_exposure(
    lec: LECData,
    factor: float
) -> LECData:
    """
    Transform total economic loss LEC into government fiscal exposure LEC.

    The LEC curve typically represents total economic losses. Only a
    fraction of these translate into direct fiscal liability. This
    function scales all loss values by the government exposure factor.

    Args:
        lec: LECData with total economic losses
        factor: government share of total losses (e.g. 0.15 for 15%)

    Returns:
        New LECData with fiscal losses = total losses x factor
    """
    fiscal_lec = LECData(mode=lec.mode)

    if lec.losses is not None:
        fiscal_lec.losses = lec.losses * factor
    if lec.exceedance_probs is not None:
        fiscal_lec.exceedance_probs = lec.exceedance_probs.copy()
    if lec.synthetic_annual_losses is not None:
        fiscal_lec.synthetic_annual_losses = lec.synthetic_annual_losses * factor

    # Scale instrument payouts too (if from production mode)
    for name, payout in lec.instrument_payouts.items():
        fiscal_lec.instrument_payouts[name] = payout * factor

    return fiscal_lec


# ============================================================
# Shared Utilities
# ============================================================

def _load_lec_curve_csv(filepath: str) -> tuple:
    """Load LEC curve CSV. Returns (losses_array, probs_array)."""
    losses, probs = [], []
    with open(filepath, 'r') as f:
        reader = csv.reader(f)
        next(reader)  # skip header
        for row in reader:
            if len(row) >= 2:
                losses.append(float(row[0]))
                probs.append(float(row[1]))
    return np.array(losses), np.array(probs)


# ============================================================
# Convenience wrapper (backward compatible)
# ============================================================

def load_and_prepare(
    lec_filepath: str,
    num_simulations: int = 1000,
    horizon_years: int = 10,
    government_exposure_factor: float = 1.0,
    seed: int = 42
) -> LECData:
    """
    Backward-compatible wrapper. Uses TESTING mode.
    For production, use load_from_lec_tool() directly.
    """
    return load_for_testing(
        lec_curve_path=lec_filepath,
        num_simulations=num_simulations,
        horizon_years=horizon_years,
        government_exposure_factor=government_exposure_factor,
        seed=seed
    )


# ============================================================
# Main — demonstrate both modes
# ============================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")

    print("=" * 60)
    print("MODE B - TESTING: Self-generated catalog")
    print("=" * 60)
    lec_test = load_for_testing(
        "LEC_curve_PER.csv",
        num_simulations=5,
        horizon_years=3,
        government_exposure_factor=0.15
    )
    print(f"  Mode: {lec_test.mode}")
    print(f"  Fiscal AAL: ${lec_test.aal:.1f}M")
    print(f"  Shape: {lec_test.synthetic_annual_losses.shape}")
    print(f"  Sample (sim 0): {lec_test.synthetic_annual_losses[0]}")
    print(f"  Instrument payouts loaded: {list(lec_test.instrument_payouts.keys())}")

    print()
    print("=" * 60)
    print("MODE A - PRODUCTION: Read from LEC tool")
    print("=" * 60)

    # Create a mock synthetic_annual_df.csv to demonstrate
    mock_catalog = np.random.default_rng(42).uniform(50, 500, size=(5, 3))
    pd.DataFrame(mock_catalog, columns=["Year_1", "Year_2", "Year_3"]
                 ).to_csv("/tmp/mock_synthetic_annual_df.csv", index=False)

    lec_prod = load_from_lec_tool(
        synthetic_catalog_path="/tmp/mock_synthetic_annual_df.csv",
        lec_curve_path="LEC_curve_PER.csv",
        government_exposure_factor=0.15
    )
    print(f"  Mode: {lec_prod.mode}")
    print(f"  Fiscal AAL: ${lec_prod.aal:.1f}M")
    print(f"  Shape: {lec_prod.synthetic_annual_losses.shape}")
    print(f"  Sample (sim 0): {lec_prod.synthetic_annual_losses[0]}")
    print(f"  Instrument payouts loaded: {list(lec_prod.instrument_payouts.keys())}")
