import os
from glob import glob
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from caf.toolkit.cost_utils import CostDistribution



def _load_matrix(file_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load one matrix CSV and return (origin_ids, matrix_values)."""
    df = pd.read_csv(file_path, compression="infer")
    if df.shape[1] < 2:
        raise ValueError(f"Expected ID column + data columns in: {file_path}")

    origin_ids = pd.to_numeric(df.iloc[:, 0], errors="coerce").to_numpy()
    matrix = df.iloc[:, 1:].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    return origin_ids, matrix


def _average_matrices(folder_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load all matrix CSV variants in folder and return averaged matrix."""
    file_paths = sorted(
        set(
            glob(os.path.join(folder_path, "*.csv"))
            + glob(os.path.join(folder_path, "*.csv.gz"))
            + glob(os.path.join(folder_path, "*.csv.bz2"))
        )
    )
    if not file_paths:
        raise FileNotFoundError(f"No matrix files found in: {folder_path}")

    origin_ids_ref: Optional[np.ndarray] = None
    matrices: List[np.ndarray] = []

    for file_path in file_paths:
        origin_ids, matrix = _load_matrix(file_path)

        if origin_ids_ref is None:
            origin_ids_ref = origin_ids
        elif len(origin_ids_ref) != len(origin_ids):
            raise ValueError(
                f"Origin row mismatch in {file_path}: {len(origin_ids)} vs {len(origin_ids_ref)}"
            )

        matrices.append(matrix)

    base_shape = matrices[0].shape
    for file_path, matrix in zip(file_paths, matrices):
        if matrix.shape != base_shape:
            raise ValueError(f"Matrix shape mismatch in {file_path}: {matrix.shape} vs {base_shape}")

    avg_matrix = np.nanmean(np.stack(matrices, axis=0), axis=0)
    return origin_ids_ref.astype(int), avg_matrix


def _apply_intrazonal_diagonal_adjustment(cost_matrix: np.ndarray, factor: float = 0.6) -> np.ndarray:
    """Set diagonal intrazonal costs to factor * row minimum excluding diagonal."""
    adjusted = np.asarray(cost_matrix, dtype=float).copy()
    n = min(adjusted.shape[0], adjusted.shape[1])

    for i in range(n):
        row_values = np.delete(adjusted[i, :], i)
        if row_values.size == 0 or np.all(np.isnan(row_values)):
            continue
        min_value = np.nanmin(row_values)
        adjusted[i, i] = float(factor) * float(min_value)

    return adjusted


def _load_dist_bins_config(config_value: Any) -> Dict[str, List[float]]:
    """Load dist bins from dict or YAML path and normalize to dict[key -> ordered float list]."""
    if isinstance(config_value, str):
        if not os.path.exists(config_value):
            raise FileNotFoundError(f"dist_bins YAML file not found: {config_value}")
        try:
            import yaml
        except ImportError as exc:
            raise ImportError(
                "PyYAML is required when dist_bins is a YAML path. Install with: pip install pyyaml"
            ) from exc
        with open(config_value, "r", encoding="utf-8") as file_obj:
            loaded = yaml.safe_load(file_obj)
    else:
        loaded = config_value

    if not isinstance(loaded, dict) or not loaded:
        raise ValueError("dist_bins must resolve to a non-empty dict of bin keys")

    normalized: Dict[str, List[float]] = {}
    for key, value in loaded.items():
        if isinstance(value, (list, tuple, np.ndarray)):
            max_bounds = [float(v) for v in value]
        else:
            max_bounds = [float(value)]

        if not max_bounds:
            continue

        ordered_unique: List[float] = []
        seen = set()
        for bound in max_bounds:
            if bound not in seen:
                seen.add(bound)
                ordered_unique.append(bound)

        normalized[str(key)] = ordered_unique

    if not normalized:
        raise ValueError("No usable bounds found in dist_bins")

    return normalized


def _get_min_max_bounds_from_bins(max_bounds: Sequence[float]) -> Tuple[List[float], List[float]]:
    """Build min/max bounds for one dist-bin key."""
    max_list = [float(v) for v in max_bounds]
    if not max_list:
        raise ValueError("Bin key has empty bounds")
    min_list = [0.0] + max_list[:-1]
    return min_list, max_list


def _calculate_cumulative_by_key(values: Sequence[float], key_labels: Sequence[str]) -> np.ndarray:
    """Cumulative values computed independently within each key label."""
    values_arr = np.asarray(values, dtype=float)
    labels_arr = np.asarray(key_labels, dtype=object)
    result = np.zeros_like(values_arr, dtype=float)

    for key in pd.unique(labels_arr):
        idx = np.where(labels_arr == key)[0]
        if idx.size > 0:
            result[idx] = np.cumsum(values_arr[idx])

    return result


def _load_synthetic_distribution(file_path: str) -> pd.DataFrame:
    """Load synthetic distribution and normalize expected columns."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Synthetic distribution file not found: {file_path}")

    synthetic_df = pd.read_csv(file_path)

    area_candidates = ["Tld area", ";index"]
    area_col = next((col for col in area_candidates if col in synthetic_df.columns), None)
    if area_col is None:
        raise ValueError(
            f"Synthetic CSV missing 'Tld area' column. Expected one of: {area_candidates}. Found: {list(synthetic_df.columns)}"
        )

    dist_candidates = ["Travel distance", "To Kilometres"]
    dist_col = next((col for col in dist_candidates if col in synthetic_df.columns), None)
    if dist_col is None:
        raise ValueError(
            f"Synthetic CSV missing 'Travel distance' column. Expected one of: {dist_candidates}. Found: {list(synthetic_df.columns)}"
        )

    pct_candidates = ["Trip percentage", "Distribution"]
    pct_col = next((col for col in pct_candidates if col in synthetic_df.columns), None)
    if pct_col is None:
        raise ValueError(
            f"Synthetic CSV missing 'Trip percentage' column. Expected one of: {pct_candidates}. Found: {list(synthetic_df.columns)}"
        )

    synthetic_df = synthetic_df[[area_col, dist_col, pct_col]].rename(
        columns={
            area_col: "Tld area",
            dist_col: "Travel distance",
            pct_col: "Trip percentage",
        }
    )
    synthetic_df["Tld area"] = synthetic_df["Tld area"].astype(str).str.strip()
    synthetic_df["Travel distance"] = pd.to_numeric(synthetic_df["Travel distance"], errors="coerce")
    synthetic_df["Trip percentage"] = pd.to_numeric(synthetic_df["Trip percentage"], errors="coerce")
    synthetic_df = synthetic_df.dropna(subset=["Travel distance", "Trip percentage"]).copy()
    synthetic_df = synthetic_df.groupby(["Tld area", "Travel distance"], as_index=False)["Trip percentage"].sum()
    return synthetic_df


def _write_output(out_df: pd.DataFrame, output_dir: str, output_file_name: str) -> str:
    """Write output dataframe to CSV and return saved path."""
    out_path = os.path.join(output_dir, output_file_name)
    out_df.to_csv(out_path, index=False)
    print(f"Saved: {out_path}")
    return out_path


def _apply_final_adjustment_factor_by_bin(out_df: pd.DataFrame) -> pd.DataFrame:
    """Use last factor in each bin and apply it to synthetic trip distribution."""
    if out_df.empty:
        out_df["Final Adjustment Factor"] = pd.Series(dtype=float)
        out_df["Final Adjusted Synthetic Trip Distribution"] = pd.Series(dtype=float)
        return out_df

    factor_series = pd.to_numeric(out_df["Adjustment Factor"], errors="coerce")
    out_df = out_df.copy()
    out_df["Adjustment Factor"] = factor_series

    final_factor = out_df.groupby(["Tld area", "Dist bin key"], sort=False)["Adjustment Factor"].transform("last")
    out_df["Final Adjustment Factor"] = final_factor

    synth_vals = pd.to_numeric(out_df["Synthetic Trip Distribution"], errors="coerce")
    out_df["Final Adjusted Synthetic Trip Distribution"] = synth_vals * out_df["Final Adjustment Factor"]
    return out_df


def calculate_postme_dist(
    postme_trip_matrix_folder: str = r"T:\Gaurav\Outputs\supporting files\Postme matrices\Trips\UC1_nhb",
    postme_cost_matrix_folder: str = r"T:\Gaurav\Outputs\supporting files\Postme matrices\Costs\UC1",
    zone_aggregation_file: str = r"T:\Gaurav\Tld Analysis\Inputs\Zone_Aggregation.csv",
    output_file_name: str = "Trip_distribution_UC1_nhb.csv",
    output_dir: str = r"T:\Gaurav\Tld Analysis\Outputs_test",
    dist_bins: Any = {"short": [1, 2, 5], "medium": [9, 14, 20], "medium1": [30, 45, 70], "long": [100, 140, 200, 300, 450, 700, 1000]},
    dist_bins_yaml: Optional[str] = None,
    synthetic_distribution_file: str = r"T:\Gaurav\Outputs\supporting files\TLD Distribution\synthetic_tlds\UC1_nhb\NTS_tld_m3_p12_nhb.csv",
) -> Tuple[pd.DataFrame, Dict[str, CostDistribution], Dict[str, np.ndarray]]:
    """Compute PostME distance distributions by region/bin and write the output CSV.

    Args:
        postme_trip_matrix_folder (str): Folder containing PostME trip matrices (`*.csv`, `*.csv.gz`, `*.csv.bz2`).If the folder has more than one file, it will produce an average matrix.
        postme_cost_matrix_folder (str): Folder containing cost matrices matching the trip matrix layout.If the folder has more than one file, it will produce an average matrix. 
        zone_aggregation_file (str): CSV path with `Old_zone_id` to `New_zone_name` mappings.
        output_file_name (str): Name of the output CSV file to create.
        output_dir (str): Directory where the output CSV is written.
        dist_bins (Any): Distance-bin configuration (typically `dict[str, list[float]]`) used when `dist_bins_yaml` is not provided.
        dist_bins_yaml (Optional[str]): Optional YAML path for distance-bin configuration; takes precedence over `dist_bins`. This is useful if each zone/region has a different distance-bin configuration.
        synthetic_distribution_file (str): CSV path for synthetic distribution inputs used in adjustment factor calculations.

    Returns:
        Tuple[pd.DataFrame, Dict[str, CostDistribution], Dict[str, np.ndarray]]:
            `out_df` (final output table), `cost_distributions` (per-region CostDistribution objects), and
            `trip_percentages` (per-region trip percentage arrays).
    """
    if not synthetic_distribution_file:
        raise ValueError("synthetic_distribution_file is required")

    origin_ids, trip_matrix = _average_matrices(postme_trip_matrix_folder)
    _, cost_matrix = _average_matrices(postme_cost_matrix_folder)

    if trip_matrix.shape != cost_matrix.shape:
        raise ValueError(
            f"Trip/cost shape mismatch after averaging: {trip_matrix.shape} vs {cost_matrix.shape}"
        )

    cost_matrix = _apply_intrazonal_diagonal_adjustment(cost_matrix, factor=0.6)

    bins_source = dist_bins_yaml if dist_bins_yaml is not None else dist_bins
    bins_by_key = _load_dist_bins_config(bins_source)
    synthetic_df = _load_synthetic_distribution(synthetic_distribution_file)

    combined_pairs: List[Tuple[float, str]] = []
    for bin_key, key_bounds in bins_by_key.items():
        for bound in key_bounds:
            combined_pairs.append((float(bound), str(bin_key)))

    if not combined_pairs:
        raise ValueError("No dist bin bounds found after combining keys")

    combined_pairs = sorted(combined_pairs, key=lambda item: item[0])
    max_bounds: List[float] = []
    bound_keys: List[str] = []
    seen_bounds = set()
    for bound, bin_key in combined_pairs:
        if bound not in seen_bounds:
            seen_bounds.add(bound)
            max_bounds.append(bound)
            bound_keys.append(bin_key)

    min_bounds, max_bounds = _get_min_max_bounds_from_bins(max_bounds)

    zone_map = pd.read_csv(zone_aggregation_file)
    zones = zone_map["New_zone_name"].astype(str).str.strip()
    zones = zones[zones != ""].unique().tolist()

    cost_distributions: Dict[str, CostDistribution] = {}
    trip_percentages: Dict[str, np.ndarray] = {}
    rows: List[Dict[str, Any]] = []

    for zone_name in zones:
        target_old_zones = (
            pd.to_numeric(
                zone_map.loc[zone_map["New_zone_name"] == zone_name, "Old_zone_id"],
                errors="coerce",
            )
            .dropna()
            .astype(int)
            .to_numpy()
        )

        row_mask = np.isin(origin_ids, target_old_zones)
        if row_mask.sum() == 0:
            continue

        zone_trip = trip_matrix[row_mask, :]
        zone_cost = cost_matrix[row_mask, :]

        if zone_trip.shape != zone_cost.shape:
            raise ValueError(
                f"Zone matrix mismatch for {zone_name}: {zone_trip.shape} vs {zone_cost.shape}"
            )

        cd = CostDistribution.from_data(
            matrix=zone_trip,
            cost_matrix=zone_cost,
            min_bounds=min_bounds,
            max_bounds=max_bounds,
        )
        cost_distributions[zone_name] = cd

        trip_vals = np.asarray(cd.trip_vals, dtype=float)
        trip_vals = np.nan_to_num(trip_vals, nan=0.0, posinf=0.0, neginf=0.0)
        total = trip_vals.sum()
        trip_pct = (trip_vals / total) * 100.0 if total > 0 else np.zeros_like(trip_vals)
        trip_percentages[zone_name] = trip_pct

        cumulative_trip_pct = _calculate_cumulative_by_key(trip_pct, bound_keys)

        zone_synth = synthetic_df.loc[synthetic_df["Tld area"] == zone_name].set_index("Travel distance")
        synthetic_trip_pct = np.asarray(
            [zone_synth["Trip percentage"].get(float(bound), 0.0) for bound in max_bounds],
            dtype=float,
        )

        cumulative_synthetic_pct = _calculate_cumulative_by_key(synthetic_trip_pct, bound_keys)
        adjustment_factor = np.where(
            np.isclose(cumulative_synthetic_pct, 0.0),
            np.nan,
            cumulative_trip_pct / cumulative_synthetic_pct,
        )

        for bin_key, max_b, tv, tp, ctp, sp, csp, af in zip(
            bound_keys,
            max_bounds,
            trip_vals.tolist(),
            trip_pct.tolist(),
            cumulative_trip_pct.tolist(),
            synthetic_trip_pct.tolist(),
            cumulative_synthetic_pct.tolist(),
            adjustment_factor.tolist(),
        ):
            rows.append(
                {
                    "Tld area": zone_name,
                    "Dist bin key": str(bin_key),
                    "Travel distance": float(max_b),
                    "Trip value": float(tv),
                    "Trip percentage": float(tp),
                    "Cumulative Trip percentage": float(ctp),
                    "Synthetic Trip Distribution": float(sp),
                    "Cumulative Synthetic Trip Distribution": float(csp),
                    "Adjustment Factor": float(af) if not np.isnan(af) else np.nan,
                }
            )

    out_df = pd.DataFrame(
        rows,
        columns=[
            "Tld area",
            "Dist bin key",
            "Travel distance",
            "Trip value",
            "Trip percentage",
            "Cumulative Trip percentage",
            "Synthetic Trip Distribution",
            "Cumulative Synthetic Trip Distribution",
            "Adjustment Factor",
        ],
    )

    out_df = _apply_final_adjustment_factor_by_bin(out_df)

    _write_output(out_df, output_dir, output_file_name)

    return out_df, cost_distributions, trip_percentages


# Backward-compatible alias.
run_tld_distribution_by_bin_key = calculate_postme_dist


# Sample usage:
# from caf.mat.prior_adjustment.tld_adjustment import calculate_postme_dist
#
# out_df, cost_distributions, trip_percentages = calculate_postme_dist(
#     postme_trip_matrix_folder=r"T:\Gaurav\Outputs\supporting files\Postme matrices\Trips\UC1_nhb",
#     postme_cost_matrix_folder=r"T:\Gaurav\Outputs\supporting files\Postme matrices\Costs\UC1",
#     zone_aggregation_file=r"T:\Gaurav\Tld Analysis\Inputs\Zone_Aggregation.csv",
#     output_file_name="Trip_distribution_UC1_nhb.csv",
#     output_dir=r"T:\Gaurav\Tld Analysis\Outputs_test",
#     dist_bins={
#         "short": [1, 2, 5],
#         "medium": [9, 14, 20],
#         "medium1": [30, 45, 70],
#         "long": [100, 140, 200, 300, 450, 700, 1000],
#     },
#     synthetic_distribution_file=(
#         r"T:\Gaurav\Outputs\supporting files\TLD Distribution\synthetic_tlds\UC1_nhb\NTS_tld_m3_p12_nhb.csv"
#     ),
# )
