# CBA module v8: insurance layer by return period and layer pricing

**Branch:** `v8` of `cardenasvictor/LEC-Tool`, built on `keom0891/LEC-Tool` at `10a0702` (LEC Tool v3.0.0, 14 September 2026).

**Result:** LEC Tool v3.1.0. The configuration and the entry point are unchanged in shape (`config.toml`, `python main.py`); the insurance block gains return-period keys and a new optional section `[insurance_pricing]`.

## 1. Why

In the v3.0.0 run for Honduras the insurance layer (CCRIF) had a benefit-cost ratio of 4.79 while every other instrument sat between 0.7 and 1.2. Two causes, both visible in the new report:

1. **The layer thresholds are dollar amounts calibrated on a fiscal scale.** `attachment_point = 50` and `exhaustion_point = 190` $MM are compared with total economic losses. On the Honduras curve that layer is hit once every 1.8 years and exhausted once every 8 years, not the 1-in-15 to 1-in-50 layer it is meant to represent. The same dollar values mean something different again in every other country of `Country databases` (in Mexico the layer would be hit several times a year).
2. **The premium is a fixed 5% Rate-on-Line**, whatever the layer. For the 50-190 layer the premium is about a quarter of the expected payout (implied multiple 0.23), which no insurer accepts, and that is what inflates the ratio.

The v8 integration also brings in the CBA v7 delivery (August 2026), which was not in the repository: `v3.0.0` integrated the v6 package.

## 2. What changed

| Area | Change |
| --- | --- |
| `insurance_layer.py` (new) | Converts thresholds given as event return periods to dollars on the curve of the country being run; derives the ceding share from a coverage limit; prices the layer |
| `main.py` | New stage `resolve_layers` between `compute_lec` and `simulate` |
| `config_loader.py` | Keys `attachment_rp`, `exhaustion_rp`, `coverage_limit`, `ddo_threshold_rp`, `ppo_loss_trigger_rp`, `pricing`, `gross_premium`, `donor_discount(_share)`, `payout_mode`, `one_payout_per_year`, `payout_floor`, PPO trigger options, `grace_period_years` for PPO and DDO; section `[insurance_pricing]`; a PPO with its own trigger no longer requires a CCF |
| `risk_management.py` | Insurance payout rule (proportional or binary), one payout per policy year, payout floor equal to the gross premium (v7 + v8); PPO own trigger (v7). Your empty-strategy guard is kept |
| `cba/pricing.py` (from v7) | Market Rate-on-Line curve by return period; documentation corrected |
| `cba/config.py`, `cba/costs.py` (from v7) | Grace period for PPO and DDO; closed-form B/C of a concessional loan (diagnostic) |
| `cba/engine.py` | Donor discount; economic and fiscal B/C of the insurance; implied multiple; protection indicators |
| `cba/reports.py`, `reporting.py` | The insurance layer in years and in dollars, the source of the premium, and an insurance block that leads with protection |
| `utils.py`, `risk_reduction.py` (from v7) | NumPy 1.x / 2.x compatibility (`trapezoid`) |
| `config.toml` | Honduras example with the layer at 1 in 15 to 1 in 50 years and a 9.24 $MM limit (the same limit as before) |

All v7 changes are additive: with their defaults the code behaves exactly as v3.0.0.

### Pricing methods (`[insurance_pricing] method`, or per instrument)

| Method | Premium |
| --- | --- |
| `rol_rule` (default) | The layer is cut into thin slices, each priced at its own event return period on the country curve: slices hit more often than 1 in `cutoff_rp` years (10) on the market ROL curve, more remote slices at `flat_rol` (5%). Default parameters are consistent with the pricing structure of sovereign risk-pooling facilities. Replace them with the country's actual quotation whenever available (pricing = "quote" and gross_premium in the instrument block). |
| `market_curve` | Every slice on the market ROL curve (commercial reinsurance) |
| `quote` | The gross premium of a real quotation (`gross_premium`), with an optional donor discount. Takes precedence |
| `fixed_rol` | Premium = `rate_on_line` x coverage limit (v3.0.0 behaviour) |

The market curve is an input with a date (`market_curve_date`): reinsurance prices are volatile and seasonal.

## 3. Results, Honduras (seed 99, 1000 simulations)

| | v3.0.0 | v8 |
| --- | --- | --- |
| CCRIF layer | 50-190 $MM (1 in 1.8 to 1 in 8 years) | 430-2,557 $MM (1 in 15 to 1 in 50 years) |
| Coverage limit | 9.24 $MM | 9.24 $MM |
| Premium | 0.46 $MM (fixed 5% ROL) | 0.46 $MM (`rol_rule`: the whole layer is above the cutoff) |
| Expected annual payout | 2.01 $MM | 0.33 $MM |
| Implied multiple | 0.23 | 1.39 |
| CCRIF B/C | 4.79 | 0.79 |
| Aggregate B/C | 1.146 | 1.061 |
| PPO / CCF / DDO B/C | 0.715 / 1.227 / 1.147 | 0.716 / 1.227 / 1.147 |

The small PPO change comes from the per-event payout cap, which scales every instrument when the sum of payouts exceeds the event loss.

A fairly priced insurance policy has an economic B/C below 1: the premium exceeds the expected payout by the insurer's cost of holding capital. The report says so and leads the insurance block with protection indicators (probability that the policy pays in the horizon, average payout, share of the year's loss covered, unpaid loss at the 1-in-20 and 1-in-100 outcomes with and without the policy). When a donor pays part of the premium the fiscal B/C, on the premium the government pays, is reported next to the economic one.

## 4. Verification

| Check | Result |
| --- | --- |
| The consultant's four test batteries (costs, CNC, simulation, absurd inputs) | 105/105 pass |
| New battery for v8 (return periods, pricing, resolution, payout floor, config validation, economic and fiscal B/C) | 52/52 pass |
| Regression: v8 with the v3.0.0 mechanics (layer 50-190 $MM, `fixed_rol`, per-event payout, no floor) | every figure of both reports identical to v3.0.0 |
| Full run for the 23 countries of `Country databases` (300 simulations) | all complete; the layer lands between 1 in 15 and 1 in 50 years in every country; probability of at least one payout in 10 years between 41% and 53% |

As requested for v3.0.0, the test batteries are delivered separately and are not added to the repository.

## 5. Observations for the team

- **The default layer and `rol_rule`.** With the default layer (1-in-15 to 1-in-50 years) every slice lies above cutoff_rp, so the rule is equivalent to a flat ROL of flat_rol; the market curve applies only to the part of the layer more frequent than 1 in cutoff_rp years. For narrow layers close to the cutoff the flat ROL can fall below the expected payout; the report flags this (implied multiple < 1). In that case use a quotation or market_curve.
- **Narrow layers and `rol_rule`.** On a country whose curve is almost flat between the two return periods (Nicaragua: 1,134 to 1,213 $MM), almost every event that touches the layer exhausts it, and the 5% flat rate falls below the expected payout (multiple 0.92). The report flags any layer priced below its expected payout. For such layers use a quotation or `market_curve`.
- **Brazil** has no probabilistic tail; its empirical curve does not reach 1 in 50 years. The exhaustion point is held at the curve end and the run warns.
- **Other instruments are still calibrated for Honduras.** The DDO thresholds (120 $MM), the PPO schedule, the CCF population and payout function, and the CNC parameters are Honduras values in `config.toml`. The DDO and PPO thresholds can now be given as return periods (`ddo_threshold_rp`, `ppo_loss_trigger_rp`); the defaults were not changed.
- **The CNC must be set for each country.** `[cba.cnc]` holds the GDP, base rate and sovereign spread of Honduras. The CNC classifies each event by its size relative to GDP to set the spread of the ex-post debt, so running another country with these values misstates its CNC: with a GDP far below the real one, most events look catastrophic and the saving is overstated. In the 23-country run of section 4 the CNC used the Honduras values and its results are not valid outside Honduras; the insurance results are, because the layer is placed on each country's curve. Update `gdp` and `sovereign_base_spread` (and `base_rate` if needed) before running a country. A possible improvement is a per-country table (GDP, sovereign spread, year and source) read by the tool from the country being run.
- **One payout per year.** One payout per policy year is a conservative approximation of the annual aggregate limit of sovereign parametric policies: a second qualifying event in the same year pays nothing, even if part of the limit remains. Set one_payout_per_year = false to allow payouts per event.
- **Entering a CCRIF quotation.** Use the return periods of the attachment and of the limit of liability, the coverage limit and the gross premium. Do not copy the dollar amounts of the attachment and exhaustion points: they are on the scale of CCRIF's own model, not of the country catalogue.
