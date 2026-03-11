class TLDAnalysisAdjusted:
    
    @staticmethod
    # Run the base TLD analysis and return unadjusted outputs and distributions.
    def _run_tld_analysis_base(
        postme_trip_matrix_folder,
        postme_cost_matrix_folder,
        zone_aggregation_file,
        output_file_name="Trip_distribution_UC1_fr.csv",
        output_dir=r"D:\\NorMITs\\Demand",
        dist_bins={"short": [1, 2, 5], "medium": [9, 14, 20], "medium1": [30, 45, 70], "long": [100, 140, 200, 300, 450, 700, 1000]},
        dist_bins_yaml=None,
        synthetic_distribution_file=None,
    ):

        from glob import glob
        import os
        import numpy as np
        import pandas as pd
        from caf.toolkit.cost_utils import CostDistribution
    
        # Collect matrix files from a folder using supported filename patterns.
        def list_files(folder_path, patterns=("*.csv", "*.csv.bz2", "*.csv.gz", "*.txt")):
            files = []
            for pattern in patterns:
                files.extend(glob(os.path.join(folder_path, pattern)))
            files = sorted(set(files))
            if not files:
                raise FileNotFoundError(f"No files found in {folder_path}")
            return files
    
        # Load a matrix file and split origin IDs from numeric matrix values.
        def load_matrix(file_path):
            df = pd.read_csv(file_path, compression="infer")
            if df.shape[1] < 2:
                raise ValueError(f"Expected matrix with ID column + data columns: {file_path}")
            origin_ids_local = pd.to_numeric(df.iloc[:, 0], errors="coerce").to_numpy()
            matrix_local = df.iloc[:, 1:].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
            return origin_ids_local, matrix_local
    
        # Read all matrix files in a folder and compute an element-wise average matrix.
        def average_matrices(folder_path):
            files = list_files(folder_path, patterns=("*.csv", "*.csv.bz2", "*.csv.gz"))
            origin_ids_ref = None
            matrices = []
    
            for file_path in files:
                origin_ids_local, matrix_local = load_matrix(file_path)
                if origin_ids_ref is None:
                    origin_ids_ref = origin_ids_local
                elif len(origin_ids_ref) != len(origin_ids_local):
                    raise ValueError(
                        f"Origin row count mismatch in {file_path}: {len(origin_ids_local)} vs {len(origin_ids_ref)}"
                    )
                matrices.append(matrix_local)
    
            base_shape = matrices[0].shape
            for file_path, matrix_local in zip(files, matrices):
                if matrix_local.shape != base_shape:
                    raise ValueError(
                        f"Matrix shape mismatch in {file_path}: {matrix_local.shape} vs {base_shape}"
                    )
    
            avg_matrix = np.nanmean(np.stack(matrices, axis=0), axis=0)
            return origin_ids_ref.astype(int), avg_matrix
    
        # Flatten nested values (dict/list/scalar) into a flat list of floats.
        def flatten_to_float_list(value):
            if value is None:
                return []
            if isinstance(value, dict):
                out = []
                for sub_val in value.values():
                    out.extend(flatten_to_float_list(sub_val))
                return out
            if isinstance(value, (list, tuple, np.ndarray)):
                out = []
                for item in value:
                    out.extend(flatten_to_float_list(item))
                return out
            return [float(value)]
    
        # Remove duplicates while preserving original value order.
        def unique_preserve_order(values):
            out = []
            seen = set()
            for value in values:
                value = float(value)
                if value not in seen:
                    seen.add(value)
                    out.append(value)
            return out
    
        # Convert bin config into a list of ordered bin groups.
        def extract_bin_groups(value):
            if value is None:
                return []
    
            if isinstance(value, dict):
                groups = []
                for _, sub_val in value.items():
                    group_vals = unique_preserve_order(flatten_to_float_list(sub_val))
                    if group_vals:
                        groups.append(group_vals)
                return groups
    
            group_vals = unique_preserve_order(flatten_to_float_list(value))
            return [group_vals] if group_vals else []
    
        # Load distance bin configuration from dict/list or YAML path.
        def load_dist_bins_config(config_value):
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
                return loaded if loaded is not None else {}, True
            return config_value, False
    
        # Resolve max distance bounds and group lengths for one zone.
        def resolve_bounds_for_zone(zone_name, loaded_bins, from_yaml):
            if loaded_bins is None:
                return [], []
    
            if from_yaml and isinstance(loaded_bins, dict):
                if zone_name in loaded_bins:
                    groups = extract_bin_groups(loaded_bins[zone_name])
                else:
                    groups = []
            else:
                groups = extract_bin_groups(loaded_bins)
    
            bounds = []
            group_lengths = []
            for group in groups:
                bounds.extend(group)
                group_lengths.append(len(group))
    
            return bounds, group_lengths
    
        # Compute cumulative percentages independently within each distance group.
        def grouped_cumulative(values, group_lengths):
            values = np.asarray(values, dtype=float).tolist()
            cummulative_values = []
            start_idx = 0
            total_len = len(values)
    
            for group_len in group_lengths:
                end_idx = min(start_idx + int(group_len), total_len)
                if end_idx <= start_idx:
                    continue
                group_vals = np.asarray(values[start_idx:end_idx], dtype=float)
                cummulative_values.extend(np.cumsum(group_vals).tolist())
                start_idx = end_idx
    
            if start_idx < total_len:
                tail_vals = np.asarray(values[start_idx:total_len], dtype=float)
                cummulative_values.extend(np.cumsum(tail_vals).tolist())
    
            return cummulative_values
    
        # Load and normalize synthetic distribution input into standard columns.
        def load_synthetic_distribution(file_path):
            if file_path is None:
                return None
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"Synthetic distribution file not found: {file_path}")
    
            synthetic_df = pd.read_csv(file_path)
    
            # Pick the first matching column name from supported aliases.
            def pick_column(df, candidates, logical_name):
                for col in candidates:
                    if col in df.columns:
                        return col
                raise ValueError(
                    f"Synthetic CSV missing '{logical_name}' column. Expected one of: {candidates}. Found: {list(df.columns)}"
                )
    
            area_col = pick_column(synthetic_df, ["Tld area", ";index"], "Tld area")
            dist_col = pick_column(synthetic_df, ["Travel distance", "To Kilometres"], "Travel distance")
            pct_col = pick_column(synthetic_df, ["Trip percentage", "Distribution"], "Trip percentage")
    
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
            synthetic_df = synthetic_df.dropna(subset=["Travel distance", "Trip percentage"])
            return synthetic_df
    
        origin_ids, pm_matrix = average_matrices(postme_trip_matrix_folder)
        _, cost_matrix = average_matrices(postme_cost_matrix_folder)
    
        if pm_matrix.shape != cost_matrix.shape:
            raise ValueError(
                f"Trip/cost matrix shape mismatch after averaging: {pm_matrix.shape} vs {cost_matrix.shape}"
            )
    
        n = min(cost_matrix.shape[0], cost_matrix.shape[1])
        for i in range(n):
            row_values = np.delete(cost_matrix[i, :], i)
            if row_values.size == 0 or np.all(np.isnan(row_values)):
                continue
            min_value = np.nanmin(row_values)
            cost_matrix[i, i] = 0.6 * min_value
    
        zone_map = pd.read_csv(zone_aggregation_file)
        bins_source = dist_bins_yaml if dist_bins_yaml is not None else dist_bins
        loaded_bins, bins_from_yaml = load_dist_bins_config(bins_source)
        synthetic_df = load_synthetic_distribution(synthetic_distribution_file)
    
        zones = zone_map["New_zone_name"].astype(str).str.strip()
        zones = zones[zones != ""].unique().tolist()
    
        cost_distributions = {}
        trip_val_percentages = {}
        zone_output_bounds = {}
        zone_output_group_lengths = {}
    
        for zn in zones:
            bounds_for_zone, group_lengths = resolve_bounds_for_zone(zn, loaded_bins, bins_from_yaml)
            if not bounds_for_zone:
                print(f"Skipping {zn}: no bounds provided")
                continue
    
            max_bounds = bounds_for_zone.copy()
            if max_bounds[-1] < 2000.0:
                max_bounds.append(2000.0)
                if group_lengths:
                    group_lengths[-1] += 1
                else:
                    group_lengths = [len(max_bounds)]
            min_bounds = [0.0] + max_bounds[:-1]
            zone_output_bounds[zn] = max_bounds
            zone_output_group_lengths[zn] = group_lengths
    
            target_old_zones = (
                pd.to_numeric(
                    zone_map.loc[zone_map["New_zone_name"] == zn, "Old_zone_id"],
                    errors="coerce",
                )
                .dropna()
                .astype(int)
                .to_numpy()
            )
    
            row_mask = np.isin(origin_ids, target_old_zones)
            if row_mask.sum() == 0:
                print(f"Skipping {zn}: no matching origins")
                continue
    
            zone_array = pm_matrix[row_mask, :]
            cost_array = cost_matrix[row_mask, :]
    
            if zone_array.shape != cost_array.shape:
                print(f"Skipping {zn}: shape mismatch {zone_array.shape} vs {cost_array.shape}")
                continue
            if cost_array.size == 0 or np.all(np.isnan(cost_array)):
                print(f"Skipping {zn}: empty/all-NaN cost array")
                continue
    
            cd = CostDistribution.from_data(
                matrix=zone_array,
                cost_matrix=cost_array,
                min_bounds=min_bounds,
                max_bounds=max_bounds,
            )
            cost_distributions[zn] = cd
    
            tv = np.asarray(cd.trip_vals, dtype=float)
            tv = np.nan_to_num(tv, nan=0.0, posinf=0.0, neginf=0.0)
            tv_sum = tv.sum()
            trip_val_percentages[zn] = (tv / tv_sum) * 100.0 if tv_sum > 0 else np.zeros_like(tv)
    
        rows = []
        for zn in cost_distributions.keys():
            max_bounds = np.asarray(zone_output_bounds.get(zn, []), dtype=float).tolist()
            trip_pcts = np.asarray(trip_val_percentages.get(zn, []), dtype=float).tolist()
            trip_values = np.asarray(cost_distributions[zn].trip_vals, dtype=float).tolist()
            group_lengths = zone_output_group_lengths.get(zn, [])
    
            min_len = min(len(max_bounds), len(trip_pcts), len(trip_values))
            if min_len == 0:
                continue
    
            cummulative_trip_pcts = grouped_cumulative(trip_pcts[:min_len], group_lengths)
    
            synthetic_trip_pcts = np.zeros(min_len, dtype=float)
            if synthetic_df is not None:
                zone_synth = synthetic_df.loc[synthetic_df["Tld area"] == zn].copy()
                if not zone_synth.empty:
                    zone_synth = zone_synth.groupby("Travel distance", as_index=False)["Trip percentage"].sum()
                    zone_synth = zone_synth.set_index("Travel distance")
                    aligned_vals = [zone_synth["Trip percentage"].get(float(bound), 0.0) for bound in max_bounds[:min_len]]
                    synthetic_trip_pcts = np.asarray(aligned_vals, dtype=float)
    
            synthetic_cummulative_trip_pcts = grouped_cumulative(synthetic_trip_pcts.tolist(), group_lengths)
    
            ratio_values = []
            for model_val, synth_val in zip(cummulative_trip_pcts[:min_len], synthetic_cummulative_trip_pcts[:min_len]):
                if np.isclose(synth_val, 0.0):
                    ratio_values.append(np.nan)
                else:
                    ratio_values.append(float(model_val) / float(synth_val))
    
            for travel_distance, trip_pct, trip_value, synthetic_trip_pct, cummulative_pct, synthetic_cummulative_pct, ratio_value in zip(
                max_bounds[:min_len],
                trip_pcts[:min_len],
                trip_values[:min_len],
                synthetic_trip_pcts[:min_len],
                cummulative_trip_pcts[:min_len],
                synthetic_cummulative_trip_pcts[:min_len],
                ratio_values[:min_len],
            ):
                rows.append(
                    {
                        "Tld area": zn,
                        "Travel distance": travel_distance,
                        "ID": f"{zn}_{travel_distance}",
                        "Trip value": trip_value,
                        "Trip percentage": trip_pct,
                        "Synthetic Trip Distribution": synthetic_trip_pct,
                        "Cumulative Postme trip Distribution": cummulative_pct,
                        "Cumulative Synthetic Trip Distribution": synthetic_cummulative_pct,
                        "Adjustment Factor": ratio_value,
                    }
                )
    
        model_df = pd.DataFrame(
            rows,
            columns=[
                "Tld area",
                "Travel distance",
                "ID",
                "Trip value",
                "Trip percentage",
                "Synthetic Trip Distribution",
                "Cumulative Postme trip Distribution",
                "Cumulative Synthetic Trip Distribution",
                "Adjustment Factor",
            ],
        )
    
        if not model_df.empty:
            required_model_cols = {"Tld area", "Travel distance", "Trip percentage"}
            missing_model = required_model_cols.difference(model_df.columns)
            if missing_model:
                raise ValueError(f"Model output data missing columns: {sorted(missing_model)}")
    
            if synthetic_distribution_file is not None:
                synthetic_post_df = pd.read_csv(synthetic_distribution_file)
    
                # Pick the first matching column name from supported aliases.
                def pick_column(df, candidates, logical_name):
                    for col in candidates:
                        if col in df.columns:
                            return col
                    raise ValueError(
                        f"Synthetic CSV missing '{logical_name}' column. Expected one of: {candidates}. Found: {list(df.columns)}"
                    )
    
                synth_area_col = pick_column(synthetic_post_df, ["Tld area", ";index"], "Tld area")
                synth_dist_col = pick_column(synthetic_post_df, ["Travel distance", "To Kilometres"], "Travel distance")
                synth_pct_col = pick_column(synthetic_post_df, ["Trip percentage", "Distribution"], "Trip percentage")
    
                synthetic_post_df = synthetic_post_df[[synth_area_col, synth_dist_col, synth_pct_col]].rename(
                    columns={
                        synth_area_col: "Tld area",
                        synth_dist_col: "Travel distance",
                        synth_pct_col: "Trip percentage",
                    }
                )
            else:
                synthetic_post_df = pd.DataFrame(columns=["Tld area", "Travel distance", "Trip percentage"])
    
            model_df["Tld area"] = model_df["Tld area"].astype(str).str.strip()
            model_df["Travel distance"] = pd.to_numeric(model_df["Travel distance"], errors="coerce")
            model_df["Trip percentage"] = pd.to_numeric(model_df["Trip percentage"], errors="coerce")
            model_df = model_df.dropna(subset=["Tld area", "Travel distance", "Trip percentage"]).copy()
    
            synthetic_post_df["Tld area"] = synthetic_post_df["Tld area"].astype(str).str.strip()
            synthetic_post_df["Travel distance"] = pd.to_numeric(synthetic_post_df["Travel distance"], errors="coerce")
            synthetic_post_df["Trip percentage"] = pd.to_numeric(synthetic_post_df["Trip percentage"], errors="coerce")
            synthetic_post_df = synthetic_post_df.dropna(subset=["Tld area", "Travel distance", "Trip percentage"]).copy()
            synthetic_post_df = synthetic_post_df.groupby(["Tld area", "Travel distance"], as_index=False)["Trip percentage"].sum()
    
            bins_cfg = loaded_bins if loaded_bins is not None else {}
    
            # Compute cumulative sums within each group label.
            def grouped_cumsum(values, group_ids):
                values = np.asarray(values, dtype=float)
                result = np.zeros_like(values, dtype=float)
                for gid in pd.unique(group_ids):
                    idx = np.where(np.asarray(group_ids) == gid)[0]
                    if idx.size > 0:
                        result[idx] = np.cumsum(values[idx])
                return result
    
            synth_lookup = {
                (row["Tld area"], float(row["Travel distance"])): float(row["Trip percentage"])
                for _, row in synthetic_post_df.iterrows()
            }
    
            model_cum_col = []
            synth_cum_col = []
            ratio_col = []
            synth_non_cum_col = []
    
            for zone_name, zone_part in model_df.groupby("Tld area", sort=False):
                zone_idx = zone_part.index.to_list()
                zone_dist = [float(v) for v in zone_part["Travel distance"].tolist()]
                zone_model_pct = np.asarray(zone_part["Trip percentage"], dtype=float)
    
                zone_bins = bins_cfg.get(zone_name, bins_cfg) if isinstance(bins_cfg, dict) else bins_cfg
                zone_groups = extract_bin_groups(zone_bins)
    
                distance_to_group = {}
                for group_number, group_values in enumerate(zone_groups):
                    for distance in group_values:
                        distance_to_group[float(distance)] = group_number
    
                default_group = len(zone_groups)
                zone_group_ids = [distance_to_group.get(dist, default_group) for dist in zone_dist]
    
                zone_model_cum = grouped_cumsum(zone_model_pct, zone_group_ids)
                zone_synth_pct = np.asarray([synth_lookup.get((zone_name, dist), 0.0) for dist in zone_dist], dtype=float)
                zone_synth_cum = grouped_cumsum(zone_synth_pct, zone_group_ids)
                zone_ratio = np.where(np.isclose(zone_synth_cum, 0.0), np.nan, zone_model_cum / zone_synth_cum)
    
                for i, idx in enumerate(zone_idx):
                    model_cum_col.append((idx, float(zone_model_cum[i])))
                    synth_cum_col.append((idx, float(zone_synth_cum[i])))
                    ratio_col.append((idx, float(zone_ratio[i]) if not np.isnan(zone_ratio[i]) else np.nan))
                    synth_non_cum_col.append((idx, float(zone_synth_pct[i])))
    
            ratio_map = dict(ratio_col)
            synth_non_cum_map = dict(synth_non_cum_col)
    
            model_df["Synthetic Trip Distribution"] = model_df.index.map(synth_non_cum_map)
            model_df["Adjustment Factor"] = model_df.index.map(ratio_map)
            #model_df["Adjusted Synthetic Trip Distribution"] = model_df["Adjustment Factor"] * model_df["Synthetic Trip Distribution"]
    
            #out_df = model_df.drop(columns=["Adjusted Synthetic Trip Distribution"], errors="ignore")
    
            return model_df, cost_distributions, trip_val_percentages

    @staticmethod
    # Apply boundary-factor based adjustment to synthetic distribution values.
    def _apply_boundary_adjustment(df, dist_bins=None, dist_bins_yaml=None):
        from bisect import bisect_left
        import numpy as np
        import pandas as pd
        import os

        adjusted_df = df.copy()
        required_cols = {"Tld area", "Travel distance", "Synthetic Trip Distribution", "Adjustment Factor"}

        if adjusted_df.empty:
            return adjusted_df
        if not required_cols.issubset(adjusted_df.columns):
            missing_cols = sorted(required_cols.difference(adjusted_df.columns))
            raise ValueError(f"Cannot apply boundary adjustment. Missing columns: {missing_cols}")

        area_vals = adjusted_df["Tld area"].astype(str).str.strip()
        b_vals = pd.to_numeric(adjusted_df["Travel distance"], errors="coerce")
        f_vals = pd.to_numeric(adjusted_df["Synthetic Trip Distribution"], errors="coerce")
        i_vals = pd.to_numeric(adjusted_df["Adjustment Factor"], errors="coerce")

        # Flatten nested values (dict/list/scalar) into a flat list of floats.
        def flatten_to_float_list(value):
            if value is None:
                return []
            if isinstance(value, dict):
                out = []
                for sub_val in value.values():
                    out.extend(flatten_to_float_list(sub_val))
                return out
            if isinstance(value, (list, tuple, np.ndarray)):
                out = []
                for item in value:
                    out.extend(flatten_to_float_list(item))
                return out
            return [float(value)]

        # Remove duplicates while preserving original value order.
        def unique_preserve_order(values):
            out = []
            seen = set()
            for value in values:
                value = float(value)
                if value not in seen:
                    seen.add(value)
                    out.append(value)
            return out

        # Convert bin config into a list of ordered bin groups.
        def extract_bin_groups(value):
            if value is None:
                return []

            if isinstance(value, dict):
                groups = []
                for sub_val in value.values():
                    group_vals = unique_preserve_order(flatten_to_float_list(sub_val))
                    if group_vals:
                        groups.append(group_vals)
                return groups

            group_vals = unique_preserve_order(flatten_to_float_list(value))
            return [group_vals] if group_vals else []

        # Load distance bin configuration from dict/list or YAML path.
        def load_dist_bins_config(config_value):
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
                return loaded if loaded is not None else {}, True
            return config_value, False

        bins_source = dist_bins_yaml if dist_bins_yaml is not None else dist_bins
        loaded_bins, bins_from_yaml = load_dist_bins_config(bins_source)

        # Determine boundary endpoints per area using bin groups.
        def resolve_boundaries_for_area(area_name):
            if loaded_bins is None:
                return []

            if bins_from_yaml and isinstance(loaded_bins, dict) and area_name in loaded_bins:
                groups = extract_bin_groups(loaded_bins[area_name])
            else:
                groups = extract_bin_groups(loaded_bins)

            boundaries_local = [group[-1] for group in groups if group]
            boundaries_local = sorted(unique_preserve_order(boundaries_local))
            return boundaries_local

        # Get the adjustment factor recorded at a specific distance boundary.
        def get_factor_at_distance(local_b_vals, local_i_vals, distance_value):
            match = np.isclose(local_b_vals, float(distance_value), equal_nan=False)
            if match.any():
                valid_i = local_i_vals.loc[match].dropna()
                if not valid_i.empty:
                    return float(valid_i.iloc[0])
            return np.nan

        area_boundary_lookup = {}
        area_boundaries = {}
        for area in area_vals.dropna().unique():
            area_mask = area_vals == area
            area_b = b_vals.loc[area_mask]
            area_i = i_vals.loc[area_mask]
            boundaries = resolve_boundaries_for_area(area)
            if not boundaries:
                boundaries = sorted(unique_preserve_order(area_b.dropna().tolist()))
            if not boundaries:
                continue

            area_boundaries[area] = boundaries

            lookup = {
                boundary: get_factor_at_distance(area_b, area_i, boundary)
                for boundary in boundaries
            }

            max_boundary = boundaries[-1]
            if pd.isna(lookup[max_boundary]):
                area_max_b = area_b.dropna().max()
                if pd.notna(area_max_b):
                    lookup[max_boundary] = get_factor_at_distance(area_b, area_i, float(area_max_b))

            area_boundary_lookup[area] = lookup

        # Map each row distance to its boundary and return the matching factor.
        def resolve_row_factor(area, distance_value):
            if pd.isna(distance_value):
                return np.nan
            lookup = area_boundary_lookup.get(area)
            boundaries = area_boundaries.get(area)
            if lookup is None or not boundaries:
                return np.nan
            idx = bisect_left(boundaries, float(distance_value))
            if idx >= len(boundaries):
                idx = len(boundaries) - 1
            chosen_boundary = boundaries[idx]
            return lookup.get(float(chosen_boundary), np.nan)

        adjusted_df["Final Adjustment Factor"] = [
            resolve_row_factor(area, distance)
            for area, distance in zip(area_vals, b_vals)
        ]
        adjusted_df["final adjusted synthetic distribution"] = adjusted_df["Final Adjustment Factor"] * f_vals

        return adjusted_df

    @classmethod
    # Execute base analysis, apply boundary adjustments, and write final output.
    def run_tld_analysis_adjusted(
        cls,
        postme_trip_matrix_folder,
        postme_cost_matrix_folder,
        zone_aggregation_file,
        output_file_name="Trip_distribution_UC1_fr.csv",
        output_dir=r"D:\\NorMITs\\Demand",
        dist_bins={"short": [1, 2, 5], "medium": [9, 14, 20], "medium1": [30, 45, 70], "long": [100, 140, 200, 300, 450, 700, 1000]},
        dist_bins_yaml=None,
        synthetic_distribution_file=None,
    ):
        import os

        out_df, cost_distributions, trip_val_percentages = cls._run_tld_analysis_base(
            postme_trip_matrix_folder=postme_trip_matrix_folder,
            postme_cost_matrix_folder=postme_cost_matrix_folder,
            zone_aggregation_file=zone_aggregation_file,
            output_file_name=output_file_name,
            output_dir=output_dir,
            dist_bins=dist_bins,
            dist_bins_yaml=dist_bins_yaml,
            synthetic_distribution_file=synthetic_distribution_file,
        )

        adjusted_df = cls._apply_boundary_adjustment(
            out_df,
            dist_bins=dist_bins,
            dist_bins_yaml=dist_bins_yaml,
        )
        adjusted_df = adjusted_df.drop(columns=["Adjusted Synthetic Trip Distribution"], errors="ignore")
        out_path = os.path.join(output_dir, output_file_name)
        adjusted_df.to_csv(out_path, index=False)
        print(f"Saved adjusted output: {out_path}")

        return adjusted_df, cost_distributions, trip_val_percentages


# Optional convenience alias for function-style calls.
run_tld_analysis_adjusted = TLDAnalysisAdjusted.run_tld_analysis_adjusted

# Usage example:
# from caf.mat.prior_adjustment.tld_adjustment import TLDAnalysisAdjusted
#
# adjusted_df, cost_distributions, trip_val_percentages = (
#     TLDAnalysisAdjusted.run_tld_analysis_adjusted(
#         postme_trip_matrix_folder=r"T:\Gaurav\Tld Analysis\Inputs\Postme matrices\Trips\UC1_fr",
#         postme_cost_matrix_folder=r"T:\Gaurav\Tld Analysis\Inputs\Postme matrices\Costs\UC1",
#         zone_aggregation_file=r"T:\Gaurav\Tld Analysis\Inputs\Zone_Aggregation.csv",
#         output_file_name="Trip_distribution_UC1_fr_trial1.csv",
#         output_dir=r"D:\NorMITs\Demand\Outputs",
#         dist_bins={
#             "short": [1, 2, 5],
#             "medium": [9, 14, 20],
#             "medium1": [30, 45, 70],
#             "long": [100, 140, 200, 300, 450, 700, 1000],
#         },
#         # Optional instead of dist_bins:
#         # dist_bins_yaml=r"T:\Gaurav\Tld Analysis\Inputs\dist_bins.yml",
#         synthetic_distribution_file=r"T:\Gaurav\Tld Analysis\Inputs\TLD Distribution\synthetic_tlds\UC1_fr\NTS_tld_m3_UC1_hb_fr.csv",
#     )
# )
#
# Notes:
# - Provide either dist_bins or dist_bins_yaml. If both are provided,
#   dist_bins_yaml takes precedence.
# - Returns a tuple: (adjusted_df, cost_distributions, trip_val_percentages).
# - The adjusted output CSV is written to output_dir/output_file_name.
