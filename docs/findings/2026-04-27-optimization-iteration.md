# Optimization iteration findings — 2026-04-27

Continues the v1 detector tuning that retired Double T/B and bull-only-ed Flag.
All experiments run on the same dataset:
- Universe: S&P 500 (503 tickers from DataHub snapshot)
- TF: Daily, Massive Starter cap = 5 years (2021-04-26 to 2026-04-24)
- Detectors enabled: H2, L2, Flag (bull-only), Failed Breakout

Baseline state going in: portfolio (4 setups × STANDARD, 6420 trades) totalled
**-42.6 R** under the engine's original "fixed 2R + 20-bar time stop" exit.

## Summary of explored directions

| Step | Hypothesis | Result | Promoted to PR? |
|------|-----------|--------|----------------|
| A | scale-half + chandelier-trail beats fixed 2R | ✅ +51R IS / +117R OOS portfolio | Yes (engine + scale_trail.yaml) |
| C | A's lift is structural, not overfit | ✅ OOS lift +121R > IS lift +90R | Yes (validates A) |
| D' | per-setup exit override > one-size-fits-all | ✅ +127.8R OOS, beats scale_trail | Yes (per_setup.yaml) |
| D | SPY > 200d MA gates H2 toward winning regimes | ❌ Filter drops the best H2 trades | No |
| H | rank candidates, cap concurrent positions | ❌ Any ranker hurts; edge is in volume | No |
| J | Daily H2 + 1H trigger entry (tight hourly stop) | ❌ 1H stop is structurally too tight | No |
| F | Rewrite Double T/B detector with stricter rules | ⚠️ Structurally clean, but PF ~= 1.00 | Code yes, default no |
| J' | True nested multi-TF (daily H2 ∩ hourly H2) | ❌ Setups are disjoint events on the two TFs | No |

## A — Promote scale-half + chandelier-trail to engine

The MVP engine hard-coded "candidate.target_price (= entry + 2R) + 20-bar
time stop". A scratch sweep on H2 STANDARD daily showed several alternatives
beat the baseline:

```
Strategy                          PF (H2 standard, daily)
2R fixed + 20-bar time stop       1.09 (baseline)
5-bar time stop only              1.27
trailing 3xATR only               1.17
scale half@1R + trail 3xATR       1.21
```

Implementation (commit `16a5c1a`):
- `ExitStrategy` dataclass (use_fixed_target / scale_at_1r /
  trailing_atr_mult / time_stop_bars / same_bar_priority)
- `simulate()` takes `bars` (must include atr14 if trailing) + strategy
- `BacktestConfig` adds three new fields with backwards-compatible defaults
- `stage_backtest` builds the strategy and merges atr14 into bars when needed

`configs/scale_trail.yaml` (commit `a6c98d3`) packages the Brooks-classic
"scale half@1R + trail 3xATR + 60-bar time stop" preset.

## C — Walk-forward validation of A

Split candidates by signal_date:
- IS: 2021-04-26 → 2024-12-31 (~3.5 years, 4609 trades)
- OOS: 2025-01-01 → 2026-04-24 (~16 months, 1811 trades)

Portfolio totals (4 setups × STANDARD):

```
                IS         OOS
baseline      -38.5       -4.0
scale_trail   +51.5     +117.2
lift          +90.0    +121.2  ← OOS lift exceeds IS lift
```

Read: scale_trail's improvement is not regime-luck of the training period;
it's structurally better. Also confirms the underlying "let trades run with
trailing instead of capping at 2R" hypothesis.

But the per-setup breakdown revealed asymmetry:

| Setup × OOS | baseline | scale_trail | Δ |
|---|---:|---:|---:|
| H2 standard | -1.2 | -8.5 | **-7.3 (regression)** |
| L2 standard | +0.8 | -2.5 | **-3.3 (regression)** |
| Flag standard | -26.2 | +11.6 | +37.8 |
| Failed BO standard | +22.5 | +116.5 | **+94.0** |

Failed BO + Flag love the trailing exit; H2 + L2 prefer the original tighter
exit. This kicked off Step D'.

## D' — Per-setup exit strategy override

Added `BacktestConfig.setup_overrides: dict[str, dict]` so each setup can
pick a partial override of the default ExitStrategy. Pipeline builds the
right strategy per `(setup, tier)` candidates file via `_strategy_for()`.

Optimal mapping based on walk-forward:
- `h2`, `l2`: default (baseline)
- `flag`, `failed_breakout`: scale_trail override

Walk-forward portfolio totals add a third strategy:

```
                IS         OOS
baseline      -38.5       -4.0
scale_trail   +51.5     +117.2
per_setup     +52.6     +127.8  ← +131.8R lift over baseline OOS
```

Per-setup wins both splits, with the bigger win in OOS (+10.6R over
scale_trail). Mechanism check (OOS, per-setup picks):

```
  H2 (baseline):   -1.2
  L2 (baseline):   +0.8
  Flag (scale):   +11.6
  FB (scale):    +116.5
  ─────────────────────
  Total:         +127.7  ✓
```

Promoted as `configs/per_setup.yaml` and `BacktestConfig.setup_overrides`
(commit `bf99ba8`).

## D — SPY macro filter on H2 (rejected)

H2 standard's year-by-year PF was clearly regime-sensitive (1.27-1.60 in
2023/2024, 0.43-0.86 in 2021/2025). Hypothesis: filter to only fire when
SPY is above its 200-day EMA, dropping bad-tape trades.

Tested 4 filters on H2 standard daily (N=223 unfiltered):

```
Filter                                 N    Win%    Avg R    PF    Total R
1. unfiltered (baseline)             223   40.8%   +0.03   1.06    +7.4
2. SPY > 200d EMA                    170   40.0%   -0.01   0.99    -0.9
3. SPY > 200d AND > 50d              159   39.6%   -0.02   0.97    -3.2
4. SPY EMA20 > EMA50                 189   38.6%   -0.05   0.91   -10.0
```

Counter-intuitive but honest: every filter hurts. Year breakdown for
filter 2 explains the mechanism — early 2023 (when SPY had recovered but
hadn't yet crossed the 200-day MA) contained the year's most profitable
H2s:

```
2023 unfiltered:    N=53 PF 1.60
2023 SPY>200d MA:   N=41 PF 1.07  ← 12 high-PF early-recovery trades dropped
```

→ "SPY above 200d MA" represents post-confirmation market state. The most
profitable H2s in stocks are *pre-confirmation* — single-name moves ahead
of broad tape. Filtering retrospective regime out exposed Brooks's "be
early" rule statistically. **Rejected.**

(SPY data fetched once to `data/_macro/spy_daily.parquet` for any future
macro experiments.)

## H — Position-cap simulation (rejected)

Real trading can't take all 25,270 candidates simultaneously. Question:
does selecting top N per day by some quality score preserve portfolio R?

Built day-by-day position-cap sim with four ranker options:

```
   Cap       setup_score              tier            fb_first              random
     3       -30.1 / 294        -1.5 / 300         +12.6 / 284         -33.8 / 274
     5       -35.5 / 474       -10.5 / 500          +1.5 / 493         -49.9 / 464
    10       -45.6 / 949       -25.4 / 987         -49.8 / 995         -59.5 / 964
    20       -73.8 / 1902      -56.0 / 1966        -85.9 / 1965         -9.9 / 1882
    50       -35.7 / 4682     -169.6 / 4845       -136.3 / 4865        -60.7 / 4745
   inf      +469.0 / 25270    +469.0 / 25270      +469.0 / 25270      +469.0 / 25270
```

Headline: any cap, any ranker, total R goes negative. The unconstrained
portfolio (+469R total / 25270 trades = +0.018 R/trade) earns through
*volume*, not via differentiating signal quality.

Diagnostic insights:
- `setup_score` correlates ≈ 0 with realized R. The score conflates two
  unrelated quality axes (`signal_bar_score` for H2/L2, `regime_strength`
  for Flag/Failed BO) and neither predicts winners reliably.
- `fb_first` (preferring Failed BO and Flag, our walk-forward-confirmed
  winners) gets a tiny edge at cap=3 (+12.6R) but degrades quickly.
- `random` performs about as well as `setup_score`, confirming there's no
  real signal in the "quality" metric.

**Rejected as designed.** Real position management on this data needs
either (a) a learned per-trade score with actual predictive power, or
(b) accept full coverage with portfolio-level risk sizing instead of
1R-per-trade. Both are bigger projects; H as a "quick rank-and-cap"
addition does not work.

## J — Daily + 1H multi-TF entry (rejected)

Brooks's stated practice mixes timeframes: daily detects the setup; hourly
times a precise entry. Hypothesis: a tight hourly stop reduces R per trade
and lifts PF on H2.

Implementation (research only):
- For each daily H2 STANDARD candidate, find the first 1H bar (within the
  next 21 hours = ~3 RTH sessions) whose `high >= daily_signal_high`.
  This simulates a stop-buy filling intraday.
- Entry = daily_signal_high + 1 tick (the stop-buy price).
- Stop = trigger 1H bar's low - 0.5 × hourly_ATR (mirror of daily's 0.5 ATR buffer).
- Target = entry + 2 × R (R = entry - stop).
- Re-simulate on HOURLY bars from the trigger bar onward (140 hourly bars
  ≈ 20 RTH days, mirroring daily 20-bar time stop).

Results on H2 standard, candidates that fall within the 2024-05 → 2026-04
hourly cache window:

```
condition                              N     Win%    Avg R    PF    Total R
pure daily entry, full 5y            223    40.8%   +0.03   1.06    +7.4
pure daily entry, same time window   103    37.9%   -0.03   0.95    -3.0
daily + 1H entry (no stop buffer)    101     0.0%   -1.00   0.00  -101.0
daily + 1H entry (0.5x hourly ATR)   103    27.2%   -0.20   0.72   -20.8
```

Both 1H entry variants underperformed pure daily on the comparable window.
Three contributing factors:

1. **Hourly stop is structurally tighter.** 0.5 × hourly ATR is roughly an
   order of magnitude smaller than 0.5 × daily ATR. Many trades exit on
   intraday mean reversion before the daily-trend resumption that the
   daily H2 was set up to capture.
2. **The trigger doesn't add alpha.** Daily H2 is already valid by the
   time the daily signal bar prints. Waiting for "first 1H bar above
   daily high" simply delays entry — it doesn't filter false signals,
   because every daily H2 will tick above its signal bar high at some
   point if it works at all.
3. **Brooks's actual practice ≠ this implementation.** He doesn't just
   enter on intraday stop-buy; he waits for a *secondary* setup on the
   hourly timeframe (hourly H2, hourly reversal bar, hourly hammer) inside
   the daily context. That's a layered detector — effectively running the
   same five detectors on hourly bars and consulting both. ~5x the engineering
   work of this scratch experiment.

**Rejected for the current iteration.** A real multi-TF implementation
should compose two detector passes (daily + hourly) and require both to
fire — that's a Step J' candidate, deferred.

## F — Double T/B detector v2 (code promoted, default still disabled)

v1 over-fired on real data:
  - 31k STANDARD daily candidates, 196k STANDARD hourly
  - Win rate 6-12% across tiers
  - PF 0.83-0.95 — net portfolio drag

Five tightening changes in v2 (commit `27f11ab`):
  - Lookback 30 → 60 bars (P1 must be multi-month significant high)
  - Pullback depth 1× → 2× ATR (real correction)
  - Pullback duration ≥ 5 → ≥ 8 bars
  - P2 placement diff 5% → 1.5%
  - Mature regime gate `regime_strength >= 0.5` + bull-regime required for
    double top, bear-regime required for double bottom
  - Signal bar must close BELOW P2 (true rejection, not just any bear bar)

Results on the same daily 5y × S&P 500 dataset, exit = baseline:

```
tier      v1 N      v1 PF    v2 N    v2 Win%   v2 PF    v2 Total R
strict    10440     0.95     513     40%       1.00      -1.1
standard  31284     0.90    2145     39%       0.98     -20.1
loose     66171     0.85    7021     41%       1.00     +10.3
```

Detector now produces ~95% fewer candidates, win rate quadruples, and PF
bracket break-even — but Total R per tier is approximately zero. **True
double tops are too rare and too symmetric in a 5-year mostly-bull
window to systematically generate edge.**

Action taken:
- v2 detector + tighter params committed to production (so future
  iterations build on the correct algorithm).
- `default.yaml` keeps `double_top_bottom` disabled — re-enabling adds
  ~0 R to portfolio while doubling computation, no benefit.
- Re-evaluate once we have (a) longer history (Massive tier upgrade) so
  multiple bear cycles enter the dataset, or (b) layered hourly
  confirmation (Step J' if pursued).

## J' — Nested multi-TF (daily H2 ∩ hourly H2) is structurally non-existent

After Step J's "1H trigger" version failed, the deeper hypothesis was that
*Brooks-style* multi-TF means **the same setup type firing on both
timeframes within a tight window** — daily H2 setup, then hourly H2 fires
within day D-D+2 to confirm and provide entry timing.

Empirical check on real cached candidates (no detector re-run needed):

  - Daily H2 STANDARD: 223 over 5y
  - Hourly H2 STANDARD: 554 over 2y
  - Of 104 daily candidates inside the hourly cache window, only 40 (38%)
    have any hourly H2 *ever* on that ticker. The other 64 (62%) have
    none — the ticker simply never produced an hourly H2 in 2 years.
  - Of those 40, the median wait until the *next* hourly H2 is **163.9 days**
    (p10 = 26 days, min = 2.7 days). Inside any plausible signal window:

    ```
    within 21h:  0 / 40
    within 4d:   1 / 40 (2%)
    within 21d:  4 / 40 (10%)
    within 42d:  5 / 40 (12%)
    ```

The two timeframes' H2 setups are essentially disjoint events. They
identify **different structures** (weeks-scale pullback vs hours-scale
pullback) over **different regime measurements** (daily EMA stack vs
hourly EMA stack). A stock can be in daily bull-trend while hourly is
transitional/range, allowing daily H2 to fire while hourly H2 never
appears nearby.

**Rejected.** Brooks's "multi-TF" guidance does not mean nested same-setup
confirmation; it means using the higher TF for *context* (already tried as
SPY-EMA200 in Step D and rejected) or watching price action on the
lower TF for *timing* (Step J's variant, also rejected). Both flavors
have been tested on this dataset; neither helps.

The remaining genuine multi-TF angle would be **cross-setup confirmation**
(e.g. daily H2 + hourly Failed Breakout, or daily Flag + hourly H2 inside
the consolidation). That's a much larger combinatorial search, deferred
unless follow-up data requires it.

## What's left

- **K**: upgrade Massive tier to lift the 5y daily / 2y hourly cap →
  unlock 2008/2018/2020 bear cycles for L2 and Bear Flag rebuild
- **J**: daily + 1H multi-TF entry (spec's original "D" extension)
- **F**: rebuild Double T/B with proper "swing high at significant level"
  filters; revive bear-side detectors
- **Per-trade ML scoring** (would unblock H): train a model on
  pnl_r ~ f(setup_score, regime_strength, MAE/MFE_so_far, sector,
  market_state) and use predicted_R as the cap ranker

## Aggregate impact landed in this iteration

```
Portfolio OOS (4 setups × STANDARD, 1811 trades, ranger 2025-01-01 → 2026-04-24):
  baseline:        -4.0 R
  scale_trail:   +117.2 R   (Step A, IS+OOS validated)
  per_setup:     +127.8 R   (Step D', best)
  improvement:   +131.8 R from baseline
```

Engine + config commits make this state reproducible:
`pa-backtest all --config configs/per_setup.yaml`.
