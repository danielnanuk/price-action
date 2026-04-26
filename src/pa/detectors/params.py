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
