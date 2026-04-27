"""Parameter presets per setup x tier. Loose includes standard includes strict.

Each tier returns SetupParams; the detector reads thresholds by name.
"""

from __future__ import annotations

from pa.detectors.base import SetupParams
from pa.types import ParamTier

H2_THRESHOLDS: dict[ParamTier, dict[str, float]] = {
    ParamTier.STRICT: {
        "min_signal_score": 0.7,
        "min_first_leg_bars": 3,
        "min_second_leg_bars": 3,
        "max_second_low_below_first_low_atr": 0.3,
        "regime_strength_min": 0.6,
        "stop_atr_buffer": 0.5,
        "target_r_multiple": 2.0,
    },
    ParamTier.STANDARD: {
        "min_signal_score": 0.5,
        "min_first_leg_bars": 2,
        "min_second_leg_bars": 2,
        "max_second_low_below_first_low_atr": 1.0,
        "regime_strength_min": 0.4,
        "stop_atr_buffer": 0.5,
        "target_r_multiple": 2.0,
    },
    ParamTier.LOOSE: {
        "min_signal_score": 0.3,
        "min_first_leg_bars": 2,
        "min_second_leg_bars": 2,
        "max_second_low_below_first_low_atr": 2.0,
        "regime_strength_min": 0.2,
        "stop_atr_buffer": 0.5,
        "target_r_multiple": 2.0,
    },
}

# L2 mirrors H2 but uses the bear-direction threshold name. Built explicitly to keep
# the structure readable.
L2_THRESHOLDS: dict[ParamTier, dict[str, float]] = {
    tier: {
        "min_signal_score": vals["min_signal_score"],
        "min_first_leg_bars": vals["min_first_leg_bars"],
        "min_second_leg_bars": vals["min_second_leg_bars"],
        "max_second_high_above_first_high_atr": vals["max_second_low_below_first_low_atr"],
        "regime_strength_min": vals["regime_strength_min"],
        "stop_atr_buffer": vals["stop_atr_buffer"],
        "target_r_multiple": vals["target_r_multiple"],
    }
    for tier, vals in H2_THRESHOLDS.items()
}


def h2_params(tier: ParamTier) -> SetupParams:
    return SetupParams(tier=tier, thresholds=dict(H2_THRESHOLDS[tier]))


def l2_params(tier: ParamTier) -> SetupParams:
    return SetupParams(tier=tier, thresholds=dict(L2_THRESHOLDS[tier]))


FLAG_THRESHOLDS: dict[ParamTier, dict[str, float]] = {
    ParamTier.STRICT: {
        "min_impulse_bars": 4,
        "min_impulse_atr_mult": 2.0,
        "min_consolidation_bars": 5,
        "max_consolidation_range_ratio": 0.4,
        "regime_strength_min": 0.6,
        "stop_atr_buffer": 0.5,
        "target_r_cap": 3.0,
    },
    ParamTier.STANDARD: {
        "min_impulse_bars": 3,
        "min_impulse_atr_mult": 1.5,
        "min_consolidation_bars": 5,
        "max_consolidation_range_ratio": 0.5,
        "regime_strength_min": 0.4,
        "stop_atr_buffer": 0.5,
        "target_r_cap": 3.0,
    },
    ParamTier.LOOSE: {
        "min_impulse_bars": 3,
        "min_impulse_atr_mult": 1.0,
        "min_consolidation_bars": 4,
        "max_consolidation_range_ratio": 0.7,
        "regime_strength_min": 0.2,
        "stop_atr_buffer": 0.5,
        "target_r_cap": 3.0,
    },
}


def flag_params(tier: ParamTier) -> SetupParams:
    return SetupParams(tier=tier, thresholds=dict(FLAG_THRESHOLDS[tier]))


FAILED_BREAKOUT_THRESHOLDS: dict[ParamTier, dict[str, float]] = {
    ParamTier.STRICT: {
        "range_lookback_bars": 20,
        "min_range_atr_mult": 1.0,  # range must be at least 1 ATR wide
        "max_failure_bars": 2,
        "regime_strength_min": 0.4,
        "stop_atr_buffer": 0.5,
        "target_r_multiple": 2.0,
    },
    ParamTier.STANDARD: {
        "range_lookback_bars": 20,
        "min_range_atr_mult": 0.8,
        "max_failure_bars": 3,
        "regime_strength_min": 0.3,
        "stop_atr_buffer": 0.5,
        "target_r_multiple": 2.0,
    },
    ParamTier.LOOSE: {
        "range_lookback_bars": 20,
        "min_range_atr_mult": 0.6,
        "max_failure_bars": 5,
        "regime_strength_min": 0.1,
        "stop_atr_buffer": 0.5,
        "target_r_multiple": 2.0,
    },
}


def failed_breakout_params(tier: ParamTier) -> SetupParams:
    return SetupParams(tier=tier, thresholds=dict(FAILED_BREAKOUT_THRESHOLDS[tier]))


DOUBLE_TB_THRESHOLDS: dict[ParamTier, dict[str, float]] = {
    # v2 thresholds (see double_top_bottom.py docstring for rationale).
    # Loose >= standard >= strict for monotonicity, just inverted for "lookback"
    # since longer lookback = stricter requirement.
    ParamTier.STRICT: {
        "lookback_bars": 80,
        "min_pullback_bars": 10,
        "min_pullback_atr_mult": 2.5,
        "max_peak_diff_pct": 0.01,
        "min_signal_score": 0.7,
        "regime_strength_min": 0.7,
        "stop_atr_buffer": 0.5,
        "target_r_multiple": 2.0,
    },
    ParamTier.STANDARD: {
        "lookback_bars": 60,
        "min_pullback_bars": 8,
        "min_pullback_atr_mult": 2.0,
        "max_peak_diff_pct": 0.015,
        "min_signal_score": 0.6,
        "regime_strength_min": 0.5,
        "stop_atr_buffer": 0.5,
        "target_r_multiple": 2.0,
    },
    ParamTier.LOOSE: {
        "lookback_bars": 40,
        "min_pullback_bars": 6,
        "min_pullback_atr_mult": 1.5,
        "max_peak_diff_pct": 0.025,
        "min_signal_score": 0.5,
        "regime_strength_min": 0.3,
        "stop_atr_buffer": 0.5,
        "target_r_multiple": 2.0,
    },
}


def double_tb_params(tier: ParamTier) -> SetupParams:
    return SetupParams(tier=tier, thresholds=dict(DOUBLE_TB_THRESHOLDS[tier]))


WEDGE_THRESHOLDS: dict[ParamTier, dict[str, float]] = {
    # Three-push reversal in bear regime (long entry). Tighter ratio = stricter
    # momentum-decay requirement, longer lookback = more "established" trend.
    ParamTier.STRICT: {
        "lookback_bars": 80,
        "swing_n": 2,
        "max_push_ratio": 0.6,
        "min_pushes_total_atr": 4.0,
        "min_signal_score": 0.7,
        "regime_strength_min": 0.7,
        "stop_atr_buffer": 0.5,
        "target_r_multiple": 2.0,
    },
    ParamTier.STANDARD: {
        "lookback_bars": 60,
        "swing_n": 2,
        "max_push_ratio": 0.7,
        "min_pushes_total_atr": 3.0,
        "min_signal_score": 0.6,
        "regime_strength_min": 0.5,
        "stop_atr_buffer": 0.5,
        "target_r_multiple": 2.0,
    },
    ParamTier.LOOSE: {
        "lookback_bars": 40,
        "swing_n": 2,
        "max_push_ratio": 0.85,
        "min_pushes_total_atr": 2.0,
        "min_signal_score": 0.5,
        "regime_strength_min": 0.3,
        "stop_atr_buffer": 0.5,
        "target_r_multiple": 2.0,
    },
}


def wedge_params(tier: ParamTier) -> SetupParams:
    return SetupParams(tier=tier, thresholds=dict(WEDGE_THRESHOLDS[tier]))


CLIMACTIC_THRESHOLDS: dict[ParamTier, dict[str, float]] = {
    # Single-bar exhaustion (long-only, bottom climax in bear regime).
    ParamTier.STRICT: {
        "min_climax_atr_mult": 2.0,
        "min_body_pct": 0.8,
        "min_close_pos": 0.8,
        "new_extreme_lookback": 30,
        "min_signal_score": 0.7,
        "regime_strength_min": 0.7,
        "stop_atr_buffer": 0.5,
        "target_r_multiple": 2.0,
    },
    ParamTier.STANDARD: {
        "min_climax_atr_mult": 1.5,
        "min_body_pct": 0.7,
        "min_close_pos": 0.7,
        "new_extreme_lookback": 20,
        "min_signal_score": 0.6,
        "regime_strength_min": 0.5,
        "stop_atr_buffer": 0.5,
        "target_r_multiple": 2.0,
    },
    ParamTier.LOOSE: {
        "min_climax_atr_mult": 1.2,
        "min_body_pct": 0.5,
        "min_close_pos": 0.6,
        "new_extreme_lookback": 15,
        "min_signal_score": 0.4,
        "regime_strength_min": 0.3,
        "stop_atr_buffer": 0.5,
        "target_r_multiple": 2.0,
    },
}


def climactic_params(tier: ParamTier) -> SetupParams:
    return SetupParams(tier=tier, thresholds=dict(CLIMACTIC_THRESHOLDS[tier]))


OUTSIDE_BAR_THRESHOLDS: dict[ParamTier, dict[str, float]] = {
    # Bull outside bar at bottom (long-only, bear regime).
    ParamTier.STRICT: {
        "new_low_lookback": 30,
        "min_range_atr": 1.5,
        "min_close_pos": 0.8,
        "min_signal_score": 0.6,
        "regime_strength_min": 0.6,
        "stop_atr_buffer": 0.5,
        "target_r_multiple": 2.0,
    },
    ParamTier.STANDARD: {
        "new_low_lookback": 20,
        "min_range_atr": 1.2,
        "min_close_pos": 0.7,
        "min_signal_score": 0.4,
        "regime_strength_min": 0.4,
        "stop_atr_buffer": 0.5,
        "target_r_multiple": 2.0,
    },
    ParamTier.LOOSE: {
        "new_low_lookback": 15,
        "min_range_atr": 1.0,
        "min_close_pos": 0.6,
        "min_signal_score": 0.3,
        "regime_strength_min": 0.2,
        "stop_atr_buffer": 0.5,
        "target_r_multiple": 2.0,
    },
}


def outside_bar_params(tier: ParamTier) -> SetupParams:
    return SetupParams(tier=tier, thresholds=dict(OUTSIDE_BAR_THRESHOLDS[tier]))


TCL_THRESHOLDS: dict[ParamTier, dict[str, float]] = {
    # Trend Channel Line overshoot (long-only, bottom TCL rejection in bear regime).
    ParamTier.STRICT: {
        "lookback_bars": 80,
        "swing_n": 2,
        "min_overshoot_pct": 0.01,
        "min_signal_score": 0.7,
        "regime_strength_min": 0.7,
        "stop_atr_buffer": 0.5,
        "target_r_multiple": 2.0,
    },
    ParamTier.STANDARD: {
        "lookback_bars": 60,
        "swing_n": 2,
        "min_overshoot_pct": 0.005,
        "min_signal_score": 0.5,
        "regime_strength_min": 0.5,
        "stop_atr_buffer": 0.5,
        "target_r_multiple": 2.0,
    },
    ParamTier.LOOSE: {
        "lookback_bars": 40,
        "swing_n": 2,
        "min_overshoot_pct": 0.002,
        "min_signal_score": 0.3,
        "regime_strength_min": 0.3,
        "stop_atr_buffer": 0.5,
        "target_r_multiple": 2.0,
    },
}


def tcl_params(tier: ParamTier) -> SetupParams:
    return SetupParams(tier=tier, thresholds=dict(TCL_THRESHOLDS[tier]))
