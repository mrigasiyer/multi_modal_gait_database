"""Utilities for detecting potentially dangerous foot pressure distributions.

The functions in this module operate purely on the insole modality that is
exported by the data extractor in this repository.  They provide a small
pipeline that

1. loads a CSV file containing the insole signals,
2. engineers a set of pressure-based features per foot, and
3. applies a handful of heuristics to flag frames whose pressure footprint may
   correspond to hazardous stances (e.g., overloading the forefoot, extreme
   asymmetry, or sudden pressure spikes).

The heuristics are intentionally conservative and rely only on normalized
pressure values, allowing the code to run even when other modalities (XSens,
gaze) are not available.  Thresholds are configurable through the
``DangerThresholds`` data class, so practitioners can adapt the detection rules
to their own environments once more data is available.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Tuple

import numpy as np
import pandas as pd


# Column groups ----------------------------------------------------------------

LEFT_PRESSURE_COLS = {
    "hallux": "Left_Hallux_norm",
    "toes": "Left_Toes_norm",
    "met1": "Left_Met1_norm",
    "met3": "Left_Met3_norm",
    "met5": "Left_Met5_norm",
    "arch": "Left_Arch_norm",
    "heel_l": "Left_Heel_L_norm",
    "heel_r": "Left_Heel_R_norm",
}

RIGHT_PRESSURE_COLS = {
    "hallux": "Right_Hallux_norm",
    "toes": "Right_Toes_norm",
    "met1": "Right_Met1_norm",
    "met3": "Right_Met3_norm",
    "met5": "Right_Met5_norm",
    "arch": "Right_Arch_norm",
    "heel_l": "Right_Heel_L_norm",
    "heel_r": "Right_Heel_R_norm",
}


FOREFOOT_KEYS: Tuple[str, ...] = ("hallux", "toes", "met1", "met3", "met5")
REARFOOT_KEYS: Tuple[str, ...] = ("arch", "heel_l", "heel_r")


@dataclass
class DangerThresholds:
    """Tunables for the danger heuristics.

    Parameters
    ----------
    max_pressure:
        Maximum single-sensor normalized pressure before we warn about an
        overload on that foot.
    min_total_pressure:
        Minimum summed normalized pressure we consider meaningful.  Below this
        value we treat the foot as unloaded when computing ratios.
    forefoot_ratio_upper / forefoot_ratio_lower:
        Bounds for how much of the total pressure is allowed on the forefoot.
        Crossing the upper bound indicates excessive load on the forefoot,
        while going below the lower bound indicates heel-only support.
    asymmetry_ratio:
        Threshold for pressure imbalance.  If one foot carries this fraction of
        the total bilateral load more than the other, we flag it.
    pressure_jump:
        Absolute change in the total pressure (per frame) that constitutes a
        sudden shift.  Values are expressed in normalized pressure units.
    """

    max_pressure: float = 0.9
    min_total_pressure: float = 0.02
    forefoot_ratio_upper: float = 0.85
    forefoot_ratio_lower: float = 0.15
    asymmetry_ratio: float = 0.6
    pressure_jump: float = 0.25


def load_insole_csv(path: Path | str) -> pd.DataFrame:
    """Load an insole CSV exported by the data extractor.

    Parameters
    ----------
    path:
        Location of the CSV file.

    Returns
    -------
    pandas.DataFrame
        Data frame with all columns from the CSV.  The column names must match
        the ones emitted by ``data_frame_extractor`` (i.e. ``Left_Hallux_norm``
        and friends).
    """

    return pd.read_csv(path)


def _get_pressure_columns(side: str) -> Dict[str, str]:
    if side == "left":
        return LEFT_PRESSURE_COLS
    if side == "right":
        return RIGHT_PRESSURE_COLS
    raise ValueError("side must be 'left' or 'right'")


def compute_pressure_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add per-foot pressure summaries to ``df``.

    The function leaves the input frame unmodified and instead returns a copy.

    Newly created columns include:

    * ``Left_Total_Pressure`` / ``Right_Total_Pressure`` – sum of all normalized
      sensors per foot.
    * ``Left_Forefoot_Ratio`` / ``Right_Forefoot_Ratio`` – share of total
      pressure that lies on the forefoot sensors.
    * ``Left_Max_Sensor_Pressure`` / ``Right_Max_Sensor_Pressure`` – highest
      normalized reading per foot.
    * ``Left_Total_Diff`` / ``Right_Total_Diff`` – absolute frame-to-frame
      change in total pressure, useful to spot sudden shifts.
    * ``Bilateral_Pressure_Total`` – total normalized pressure across both feet.
    * ``Pressure_Asymmetry`` – signed fraction indicating which foot carries
      more of the load.  Positive values mean the left foot dominates.

    Parameters
    ----------
    df:
        Data frame that contains the normalized insole columns.

    Returns
    -------
    pandas.DataFrame
        Copy of ``df`` with the engineered feature columns appended.
    """

    result = df.copy()

    for side in ("left", "right"):
        cols = _get_pressure_columns(side)
        total_col = f"{side.title()}_Total_Pressure"
        forefoot_col = f"{side.title()}_Forefoot_Pressure"
        forefoot_ratio_col = f"{side.title()}_Forefoot_Ratio"
        max_col = f"{side.title()}_Max_Sensor_Pressure"
        total_diff_col = f"{side.title()}_Total_Diff"

        total = result[list(cols.values())].sum(axis=1)
        forefoot = result[[cols[key] for key in FOREFOOT_KEYS]].sum(axis=1)
        max_sensor = result[list(cols.values())].max(axis=1)

        result[total_col] = total
        result[forefoot_col] = forefoot
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = forefoot / total
        result[forefoot_ratio_col] = ratio.fillna(0.0)
        result[max_col] = max_sensor
        result[total_diff_col] = result[total_col].diff().abs().fillna(0.0)

    result["Bilateral_Pressure_Total"] = (
        result["Left_Total_Pressure"] + result["Right_Total_Pressure"]
    )

    bilateral = result["Bilateral_Pressure_Total"].replace(0, np.nan)
    imbalance = (
        (result["Left_Total_Pressure"] - result["Right_Total_Pressure"]) / bilateral
    )
    result["Pressure_Asymmetry"] = imbalance.fillna(0.0)

    return result


def flag_dangerous_frames(
    df: pd.DataFrame, thresholds: DangerThresholds | None = None
) -> pd.DataFrame:
    """Evaluate the engineered features and add danger flags.

    Parameters
    ----------
    df:
        Data frame that already includes the engineered features (see
        :func:`compute_pressure_features`).  The function returns a copy with
        additional boolean columns for individual heuristics and an
        ``Is_Dangerous`` aggregate column.
    thresholds:
        Optional custom :class:`DangerThresholds` object.  If omitted, default
        thresholds tuned for normalized pressure data are used.
    """

    thresholds = thresholds or DangerThresholds()
    result = df.copy()

    def _foot_flags(side: str) -> Tuple[str, str, str, str]:
        return (
            f"{side.title()}_Forefoot_Ratio",
            f"{side.title()}_Total_Pressure",
            f"{side.title()}_Max_Sensor_Pressure",
            f"{side.title()}_Total_Diff",
        )

    flags: Dict[str, Iterable[bool]] = {}

    for side in ("Left", "Right"):
        ratio_col, total_col, max_col, diff_col = _foot_flags(side.lower())

        total = result[total_col]
        ratio = result[ratio_col]
        max_sensor = result[max_col]
        total_diff = result[diff_col]

        overload = (max_sensor > thresholds.max_pressure) & (
            total > thresholds.min_total_pressure
        )
        forefoot_overload = (ratio > thresholds.forefoot_ratio_upper) & (
            total > thresholds.min_total_pressure
        )
        heel_only = (ratio < thresholds.forefoot_ratio_lower) & (
            total > thresholds.min_total_pressure
        )
        sudden_shift = total_diff > thresholds.pressure_jump

        flags[f"{side}_Overload"] = overload
        flags[f"{side}_Forefoot_Overload"] = forefoot_overload
        flags[f"{side}_Heel_Only"] = heel_only
        flags[f"{side}_Sudden_Shift"] = sudden_shift

    bilateral_total = result["Bilateral_Pressure_Total"]
    asymmetry = result["Pressure_Asymmetry"].abs()
    imbalance_flag = (
        (bilateral_total > thresholds.min_total_pressure)
        & (asymmetry > thresholds.asymmetry_ratio)
    )
    flags["Severe_Imbalance"] = imbalance_flag

    for name, mask in flags.items():
        result[name] = pd.Series(mask, index=result.index)

    result["Danger_Reason_Count"] = sum(result[name] for name in flags)
    result["Is_Dangerous"] = result["Danger_Reason_Count"] > 0

    return result


def summarize_detections(df: pd.DataFrame) -> pd.DataFrame:
    """Create a compact summary of detected danger cases.

    The summary lists the number of frames per danger reason and the share of
    the data that they represent.
    """

    total_frames = len(df)
    summary_rows = []

    danger_columns = [
        col
        for col in df.columns
        if col.endswith("Overload")
        or col.endswith("Heel_Only")
        or col.endswith("Sudden_Shift")
        or col == "Severe_Imbalance"
    ]

    for column in danger_columns:
        count = int(df[column].sum())
        if count == 0:
            continue
        summary_rows.append(
            {
                "danger_condition": column,
                "frames": count,
                "fraction": count / total_frames if total_frames else 0.0,
            }
        )

    return pd.DataFrame(summary_rows)


def _format_summary(summary: pd.DataFrame) -> str:
    if summary.empty:
        return "No dangerous frames detected."

    lines = ["Danger summary:"]
    for _, row in summary.iterrows():
        lines.append(
            f"  - {row['danger_condition']}: {row['frames']} frames "
            f"({row['fraction']:.1%})"
        )
    return "\n".join(lines)


def run_analysis(
    csv_path: Path | str,
    output_path: Path | str | None = None,
    thresholds: DangerThresholds | None = None,
) -> pd.DataFrame:
    """End-to-end helper that loads, annotates, and optionally stores a CSV.

    Parameters
    ----------
    csv_path:
        Input CSV with insole data.
    output_path:
        Optional path to write the annotated data to.  When ``None``, the result
        is not written to disk.
    thresholds:
        Optional custom heuristics.
    """

    raw = load_insole_csv(csv_path)
    features = compute_pressure_features(raw)
    annotated = flag_dangerous_frames(features, thresholds=thresholds)

    summary = summarize_detections(annotated)
    print(_format_summary(summary))

    if output_path is not None:
        annotated.to_csv(output_path, index=False)

    return annotated


def _parse_args(argv: Iterable[str] | None = None):
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", type=Path, help="Insole CSV exported by the extractor")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional destination for the annotated CSV",
    )
    parser.add_argument(
        "--max-pressure",
        type=float,
        default=DangerThresholds.max_pressure,
        help="Single-sensor pressure threshold for overload detection",
    )
    parser.add_argument(
        "--pressure-jump",
        type=float,
        default=DangerThresholds.pressure_jump,
        help="Absolute change in total pressure that counts as a sudden shift",
    )
    parser.add_argument(
        "--forefoot-upper",
        type=float,
        default=DangerThresholds.forefoot_ratio_upper,
        help="Upper bound on the forefoot pressure ratio",
    )
    parser.add_argument(
        "--forefoot-lower",
        type=float,
        default=DangerThresholds.forefoot_ratio_lower,
        help="Lower bound on the forefoot pressure ratio",
    )
    parser.add_argument(
        "--asymmetry",
        type=float,
        default=DangerThresholds.asymmetry_ratio,
        help="Imbalance threshold for bilateral pressure",
    )
    parser.add_argument(
        "--min-total",
        type=float,
        default=DangerThresholds.min_total_pressure,
        help="Minimum total pressure treated as foot contact",
    )
    return parser.parse_args(argv)


def _thresholds_from_args(args) -> DangerThresholds:
    return DangerThresholds(
        max_pressure=args.max_pressure,
        min_total_pressure=args.min_total,
        forefoot_ratio_upper=args.forefoot_upper,
        forefoot_ratio_lower=args.forefoot_lower,
        asymmetry_ratio=args.asymmetry,
        pressure_jump=args.pressure_jump,
    )


def main(argv: Iterable[str] | None = None) -> None:
    args = _parse_args(argv)
    thresholds = _thresholds_from_args(args)
    run_analysis(args.csv, args.output, thresholds=thresholds)


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
