"""
Implementation of a self-calibrating multi area gravity model with 
post ME furness adjustment.
"""

# Built-Ins
import logging
import os
import pathlib

# Third Party
import caf.base as cb
import pandas as pd
from caf.distribute import cost_functions, furness, gravity_model
from caf.toolkit.concurrency import multiprocess

# Local Imports
from caf.mat.matrices import MatrixFiles, MatrixType

_ADJ_CHOICE_CONFIG = {
    (True, False, False, False): ("0_dist_adj", ["dist"]),
    (True, True, True, False):   ("1_dist_od_adj", ["dist", "origin", "dest"]),
    (True, True, True, True):    ("2_dist_od_sec_adj", ["dist", "origin", "dest", "sec"]),
    (True, False, False, True):  ("3_dist_sec_adj", ["dist", "sec"]),
    (False, True, True, False):  ("4_od_adj", ["origin", "dest"]),
    (False, True, True, True):   ("5_od_sec_adj", ["origin", "dest", "sec"]),
    (False, False, False, True): ("6_sec_adj", ["sec"]),
}

# # # CONSTANTS # # #
LOG = logging.getLogger(__name__)


def _multi_loop(
    current_slice,
    cost_distributions,
    constraint_area_trans,
    sector_target_furnessed,
    calib_gm: gravity_model.MultiAreaGravityModelCalibrator,
    out_dir,
):
    slice_name = current_slice.generate_name()
    csv_logging_path = out_dir / f"{slice_name}_log.csv"
    sectoral_inputs = furness.SectoralConstraintInputs(
        constraint_area_trans,
        from_col="normits_id",
        to_col="noham_sector_id",
        factor_col="normits_to_noham_sector",
        target_mat=sector_target_furnessed,
        zonal_zones=sorted(constraint_area_trans["normits_id"].unique()),
    )
    gravity_model_results = calib_gm.sectoral_run(
        cost_distributions,
        sectoral_inputs,
        csv_logging_path,
        True,
        gravity_model.GMCalibParams(furness_jac=True),
    )

    summary = pd.concat(
        {area: results.summary for area, results in gravity_model_results.items()}, axis=1
    )
    summary.to_csv(out_dir / f"{slice_name}_summary.csv")
    matrix = pd.DataFrame(
        calib_gm.achieved_distribution,
        index=constraint_area_trans["normits_id"],
        columns=constraint_area_trans["normits_id"],
    )
    matrix.to_csv(out_dir / f"{slice_name}_matrix.csv")

# pylint: disable=too-many-locals
def seg_furness(
    current_slice,
    cost_distributions,
    constraint_area_trans,
    sector_target_furnessed: pd.DataFrame,
    calib_gm: gravity_model.MultiAreaGravityModelCalibrator,
    out_dir: pathlib.Path,
    tld_lookup: pd.Series,
    run_gm: bool,
    run_adjust: bool,
    adj_target_options: dict[str, tuple[bool, int]],
):
    """
    Execute gravity model calibration and/or Furness adjustment for a given slice,
    outputs results and relevant diagnostic files to the specified output directory.
    Designed to be run in parallel across multiple slices.

    Parameters
    ----------
    current_slice : dict-like
        Slice object containing slice-specific information. Must have methods
        generate_name() and get(key) for extracting slice identifier and purpose.
    cost_distributions : object
        Cost distribution object with a distributions attribute containing
        distribution data for gravity model calibration.
    constraint_area_trans : pd.DataFrame
        DataFrame mapping zone IDs (normits_id) to sector IDs (noham_sector_id).
    sector_target_furnessed : pd.DataFrame
        Sector target matrix containing target values by origin and destination sector
        for the adjustment algorithm.
    calib_gm : gravity_model.MultiAreaGravityModelCalibrator
        Calibrated or pre-configured gravity model object used for calibration
        and distribution calculations.
    out_dir : pathlib.Path
        Root output directory where results will be written. Results are
        then organized by purpose subdirectories.
    tld_lookup : pd.Series
        Series mapping internal zone indices to zone names/IDs for matrix indexing.
    run_gm : bool
        If True, execute gravity model calibration.
        If False, load existing overall_matrix.csv from output directory.
    run_adjust : bool
        If True, execute Furness adjustment algorithm on seed/gravity model matrix.
    adj_target_options : dict[str, tuple[bool, int]]
        Dictionary specifying which adjustment targets to apply. Keys correspond to
        target names (e.g., 'triple_targ', 'o_target', 'd_target', 'sec_target'),
        values are tuples of (apply: bool, max_cap: value).

    Returns
    -------
    None
        Results are written directly to disk in output directories organized by
        purpose, slice and adjustment type.
    """

    slice_name = current_slice.generate_name()
    purpose = f"p{current_slice.get('p')}"
    os.makedirs(os.path.join(out_dir, purpose), exist_ok=True)
    csv_logging_path = out_dir / purpose / f"{slice_name}_log.csv"
    output_path = out_dir / purpose / slice_name
    band_targets, band_lookup = calib_gm.multi_props(cost_distributions.distributions)
    band_targets.index.names = ["area", "band_start", "band_end"]
    used = pd.MultiIndex.from_frame(
        band_lookup.reset_index()[["area", "band_start", "band_end"]].drop_duplicates()
    )
    dropped = band_targets.drop(used).sum()
    LOG.warning("Total demand dropped from band targets due to missing in band_lookup: %s", dropped.sum())

    band_targets = band_targets.loc[used]
    sec_look = constraint_area_trans.sort_values(by="normits_id").set_index("normits_id")[
        "noham_sector_id"
    ]
    full_ind = (
        band_lookup.reset_index()
        .merge(sec_look, left_on="o_zon", right_index=True)
        .rename(columns={"noham_sector_id": "o_sec"})
    )
    full_ind = full_ind.merge(sec_look, left_on="d_zon", right_index=True).rename(
        columns={"noham_sector_id": "d_sec"}
    )
    if run_gm:
        gravity_model_results, dists = calib_gm.calibrate( #pylint: disable=unused-variable
            cost_distributions,
            csv_logging_path,
            output_path,
            gravity_model.GMCalibParams(furness_jac=True, ftol=1e-2, xtol=1e-2),
            return_distributions=True,
        )
    if run_adjust:

        # check which options were set to true and output into the correction folder
        # also identify which array indices correspond to which targets in the adjust() outputs
        choice_key = tuple(option[0] for option in adj_target_options.values())
        folder_name, factor_names = _ADJ_CHOICE_CONFIG[choice_key]
        adjustment_path = output_path / folder_name
        os.makedirs(adjustment_path, exist_ok=True)
        raw_factors_output_dict = {
            i: adjustment_path / f"{slice_name}_{name}_adj_raw.csv"
            for i, name in enumerate(factor_names)
        }

        if run_gm:
            mat = pd.DataFrame(calib_gm.achieved_distribution)
            mat = mat.set_index(tld_lookup)
            mat.columns = tld_lookup
        else:
            mat = pd.read_csv(os.path.join(output_path, "overall_matrix.csv"), index_col=0)
            mat.columns = mat.columns.astype(int)

        mat = mat.stack()
        mat.index.names = ["o_zon", "d_zon"]
        mat.name = "demand"

        reindexer = full_ind.set_index(["o_zon", "d_zon"])
        seed_mat = (
            reindexer.join(mat)
            .set_index(["area", "band_start", "band_end", "o_sec", "d_sec"], append=True)
            .squeeze()
        )
        triple_targ = band_targets
        sec_target = sector_target_furnessed.stack()
        sec_target.index.names = ["o_sec", "d_sec"]

        o_target = pd.Series(calib_gm.row_targets, index=calib_gm.cost_matrix_df.index)
        o_target.index.name = "o_zon"
        d_target = pd.Series(calib_gm.col_targets, index=calib_gm.cost_matrix_df.columns)
        d_target.index.name = "d_zon"

        # pre-furness checks
        row_seed = mat.groupby("o_zon").sum()
        col_seed = mat.groupby("d_zon").sum()

        target_map = {
            "triple_targ": triple_targ,
            "o_target": o_target,
            "d_target": d_target,
            "sec_target": sec_target,
        }
        targets = [
            furness.adj_input(target_map[targ], True, option[1])
            for targ, option in adj_target_options.items()
            if option[0]
        ]
        final_mat, adj_factors = furness.adjust(seed_mat, targets)
        final_mat = final_mat.reset_index(
            level=["o_sec", "d_sec", "band_start", "band_end", "area"], drop=True
        ).unstack(level="d_zon")

        row_diff = final_mat.sum(axis=1).to_frame()
        row_diff.columns = ["final_achieved"]
        row_diff["target"] = o_target
        row_diff["seed"] = row_seed
        col_diff = final_mat.sum().to_frame()
        col_diff.columns = ["final_achieved"]
        col_diff["target"] = d_target
        col_diff["seed"] = col_seed
        te_comp = pd.concat({"rows": row_diff, "cols": col_diff}, axis=1)

        # output files
        te_comp.to_csv(adjustment_path / f"{slice_name}_te_comp.csv")
        final_mat.to_csv(adjustment_path / f"{slice_name}_matrix.csv")
        for i, out_filename in raw_factors_output_dict.items():
            adj_factors[i].to_csv(out_filename)

# pylint: disable=too-many-arguments
def _4d_constraint_gravity_model(
    row_trip_ends: cb.DVector,
    col_trip_ends: cb.DVector,
    name: str,
    tlds: dict[str, pathlib.Path],
    tld_zones: pd.Series,
    cost_matrix: MatrixFiles,
    constraint_area_trans: pd.DataFrame,
    sector_target_matrix: MatrixFiles,
    out_dir: pathlib.Path,
    run_gm: bool,
    run_adjust: bool,
    adj_target_options: dict[str, tuple[bool, int]],
    max_process: int,
):
    """
    Execute a 4D constraint gravity model with optional calibration and adjustment.
    This function processes trip distribution data using a gravity model approach,
    handling multiple time periods and applying sector-level constraints and
    adjustments. It prepares inputs for parallel processing of gravity model
    calibration and furness adjustments using seg_furness().

    Parameters
    ----------
    row_trip_ends : cb.DVector
        Origin (row) trip ends vector with segmentation information.
    col_trip_ends : cb.DVector
        Destination (column) trip ends vector with segmentation information.
    name : str
        Identifier name for the gravity model run.
    tlds : dict[str, pathlib.Path]
        Dictionary mapping segment identifiers to Trip Length Distribution (TLD)
        file paths.
    tld_zones : pd.Series
        Series containing TLD zone information indexed by zone identifiers.
    cost_matrix : MatrixFiles
        Cost matrix object containing distance file information.
    constraint_area_trans : pd.DataFrame
        Translation/mapping dataframe between normits_id and noham_sector_id for
        spatial constraint areas.
    sector_target_matrix : MatrixFiles
        MatrixFiles object for target matrices for sector-level constraint targets.
    calibrate : bool
        Flag indicating whether to calibrate the gravity model (currently unused).
    out_dir : pathlib.Path
        Output directory path for results.
    run_gm : bool
        Flag to execute gravity model; if False, disables multiprocessing.
    run_adjust : bool
        Flag to execute post-gravity model adjustments.
    adj_target_options : dict[str, tuple[bool, int]]
        Dictionary specifying which adjustment targets to apply. Keys correspond to
        target names (e.g., 'triple_targ', 'o_target', 'd_target', 'sec_target'),
        values are tuples of (apply: bool, max_cap: value).
    max_process : int
        Maximum number of processes for parallel execution.
        Will be set to 0 if run_gm is False (no calibrate) as reading multiple files
        into the same variable causes issues with multiprocessing.
        Value recommendations:
            2 is recommended for full runs due to RAM constraints (usage can spike
                to 50-60GB),
            3-4 can be used for testing individual time periods.

    Returns
    -------
    None
        Results are written to out_dir through seg_furness() and multiprocessing callback.

    Raises
    ------
    ValueError
        If run_adjust is True but no adjustment targets are enabled in
        adj_target_options.

    Warnings
    --------
    - Time period 4 (tp==4) is skipped in processing.
    - Sector target matrix is rescaled by row sum ratio, potentially overwriting
      furness adjustment results.
    - The calibrate parameter is defined but not utilized in function logic.
    - Empty adj_target_options dict will raise ValueError if run_adjust=True.

    Notes
    -----
    - Processing occurs iteratively per segment slice from row_trip_ends.
    - Distance distributions are calculated using logarithmic normal cost function.
    - Furness adjustment is applied at sector level before gravity model execution.
    - Multiprocessing count is conditional on run_gm flag.
    """
    # check if adjust was set to True and at least one of the adjustment targets was set to True, if not raise error
    if run_adjust and not any(adj_target_options.values()):
        raise ValueError(
            "At least one adjustment target must be set to True if run_adjust is True."
        )

    inputs = []
    for current_slice in row_trip_ends.segmentation.iter_slices():
        if current_slice.get("tp") == 4:
            continue
        row = row_trip_ends.get_slice(current_slice)
        col = col_trip_ends.get_slice(current_slice)
        cost = cost_matrix.get_matrix(current_slice.aggregate(cost_matrix.segmentation.naming_order))
        sector_target = sector_target_matrix.get_matrix(
            current_slice.aggregate(sector_target_matrix.segmentation.naming_order)
        )
        tld = pd.read_csv(tlds[current_slice.aggregate(["p", "direction_od"]).generate_name()])
        tld["from"] = tld["Travel distance"].shift().fillna(0)
        tld.loc[tld["from"] > tld["Travel distance"], "from"] = 0
        tld["ave_dist"] = (tld["Travel distance"] + tld["from"]) / 2
        trans = constraint_area_trans.set_index("normits_id")["noham_sector_id"].to_dict()
        col_sector = col.rename(trans).groupby("normits_id").sum().reset_index()
        col_sector.columns = ["model_zone_id", "trips"]
        row_sector = row.rename(trans).groupby("normits_id").sum().reset_index()
        row_sector.columns = ["model_zone_id", "trips"]
        sector_target_furnessed, _, _ = furness.furness_pandas_wrapper(
            sector_target.data, row_sector, col_sector, tol=0.001
        )
        sector_target_furnessed = (
            sector_target.data * row.sum() / sector_target.data.sum().sum()
        )
        LOG.info("Running Gravity Model: %s, with calibration %s", name, run_gm)

        cost_function = cost_functions.BuiltInCostFunction.LOG_NORMAL.get_cost_function()

        cost_distributions = gravity_model.MultiCostDistribution.from_pandas(
            tld,
            tld_zones,
            {i: cost_function.default_params for i in tld_zones["tld_area"].unique()},
            tld_cat_col="Tld area",
            tld_min_col="from",
            tld_max_col="Travel distance",
            tld_avg_col="ave_dist",
            tld_trips_col="final adjusted synthetic distribution",
            lookup_cat_col="tld_area",
            lookup_zone_col=";normits_v3.3_id",
        )

        calib_gm = gravity_model.MultiAreaGravityModelCalibrator(
            row,
            col,
            cost.data,
            cost_function,
        )

        inputs.append(
            (
                current_slice,
                cost_distributions,
                constraint_area_trans,
                sector_target_furnessed,
                calib_gm,
                out_dir,
                tld_zones[";normits_v3.3_id"],
                run_gm,
                run_adjust,
                adj_target_options,
            )
        )

    multiprocess(
        seg_furness, arg_list=inputs, process_count=0 if not run_gm else max_process
    )


def _use_as_list(input_list: int | list[int]) -> list[int]:
    """
    Function to ensure that input is always returned as a list,
    even if a single integer is provided. This is because the subsets
    parameter in SegmentationInput requires a dictionary with list values.

    Parameters
    ----------
    input_list : int or list of int
        The input value(s) to be converted to a list.

    Returns
    -------
    list of int
        The input value(s) as a list.
    """
    return input_list if isinstance(input_list, list) else [input_list]


def main():
    """
    Main function to execute the 4D constraint gravity model with post ME furness adjustment.
    All data loading, preprocessing, and function calls are contained within this function.
    Update the Adjust Targets Options as needed before running.
    Required:
        - Define your filters/subsets for segmentation, TLDs and trip ends loading.
        - Update all paths to matrices, TLDs, and lookup files as needed.
        - Updated filename_template where necessary to match the naming convention of input files.
        - Update the call for _4d_constraint_gravity_model with the appropriate parameters and options for your run.
    """
    # choose filter options for m, p, tp, direction_od here, will be used for all matrices to select the relevant slices
    # can be input as single integer or list of integers if multiple slices are needed
    # full lists are:
    # m_subset = list(range(1,9))
    # tp_subset = list(range(1,9))
    # p_subset = list(range(1,9)) + list(range(11,19))
    # direction_od_subset = list(range(3))
    m_subset = 3
    tp_subset = [1, 2, 3]
    p_subset = list(range(1, 9)) + list(range(11, 19))
    direction_od_subset = list(range(3))

    # define adjustment target options here
    adj_target_options = {
        "triple_targ": (True, 5),
        "o_target": (True, 10),
        "d_target": (True, 10),
        "sec_target": (True, 10),
    }

    tld_lookup = pd.read_csv(r"I:\NorMITs Distribution\voa_gb_2023_uni\NorMITs_zone.csv")
    normits = cb.ZoningSystem.get_zoning("normits")
    noham_sector = cb.ZoningSystem.get_zoning("noham_sector")
    normits_noham_sector = pd.read_csv(
        r"I:\Data\Zone Translations\cache\noham_sector_normits_v3_3\noham_sector_to_normits_spatial_.csv"
    )
    normits_noham_sector.columns = [
        "noham_sector_id",
        "normits_id",
        "noham_sector_to_normits",
        "normits_to_noham_sector",
    ]
    normits_noham_sector["noham_sector_id"] = normits_noham_sector["noham_sector_id"].replace(
        noham_sector.name_to_id
    )

    # split into HB and NHB purposes
    hb_p_subset = (
        [p for p in p_subset if p in list(range(1, 9))]
        if isinstance(p_subset, list)
        else (p_subset if p_subset in list(range(1, 9)) else None)
    )
    nhb_p_subset = (
        [p for p in p_subset if p in list(range(11, 19))]
        if isinstance(p_subset, list)
        else (p_subset if p_subset in list(range(11, 19)) else None)
    )

    full_seg_p = cb.Segmentation(
        cb.SegmentationInput(
            enum_segments=["m", "p", "direction_od", "tp"],
            naming_order=["m", "p", "tp", "direction_od"],
            subsets={
                "m": _use_as_list(m_subset),
                "p": _use_as_list(p_subset),
                "tp": _use_as_list(tp_subset),
                "direction_od": _use_as_list(direction_od_subset),
            },
        )
    )

    postme_purpose = MatrixFiles(
        full_seg_p,
        noham_sector,
        MatrixType.OD,
        pathlib.Path(r"I:\Prior adjustment\distribution\postme_p\infilled"),
        filename_template="infilled_{type}_{slice_name}.csv",
    )

    cost_seg = cb.Segmentation(
        cb.SegmentationInput(
            enum_segments=["m", "tp"],
            naming_order=["m", "tp"],
            subsets={"m": _use_as_list(m_subset), "tp": _use_as_list(tp_subset)},
        )
    )

    costs = MatrixFiles(
        cost_seg,
        normits,
        MatrixType.OD,
        pathlib.Path(r"I:\Prior adjustment\distribution\costs"),
        filename_template="normits_costs_{slice_name}.csv",
    )

    tld_seg = cb.Segmentation(
        cb.SegmentationInput(
            enum_segments=["p", "direction_od"],
            naming_order=["p", "direction_od"],
            subsets={
                "p": _use_as_list(p_subset),
                "direction_od": _use_as_list(direction_od_subset),
            },
        )
    )
    tlds = {}
    tld_dir = pathlib.Path(r"I:\Prior adjustment\postme_tlds\v2_run")
    for tld_slice in tld_seg.iter_slices():
        tld_name = tld_slice.generate_name()
        file_name = tld_name.replace("fr", "hb_fr").replace("to", "hb_to")
        tlds[tld_name] = tld_dir / f"filled_{file_name}.csv"

    dvec_map = {}
    if 0 in direction_od_subset:
        prod_nhb = (
            cb.DVector.load(
                r"I:\Prior adjustment\outputs_26_2\Core\nhb_productions\nhb_normits_tem_segmented_fr_pm_2023.dvec"
            )
            .aggregate(["p", "m", "tp"])
            .filter_segment_value("p", nhb_p_subset, keep_filtered=True)
            .filter_segment_value("m", m_subset, keep_filtered=True)
            .filter_segment_value("tp", tp_subset, keep_filtered=True)
            .aggregate_comp_zones(normits)
        )
        attr_nhb = (
            cb.DVector.load(
                r"I:\Prior adjustment\outputs_26_2\Core\nhb_attractions\nhb_normits_tem_segmented_fr_pm_2023.dvec"
            )
            .aggregate(["p", "m", "tp"])
            .filter_segment_value("p", nhb_p_subset, keep_filtered=True)
            .filter_segment_value("m", m_subset, keep_filtered=True)
            .filter_segment_value("tp", tp_subset, keep_filtered=True)
            .aggregate_comp_zones(normits)
        )
        dvec_map[0] = (prod_nhb, attr_nhb)
    if 1 in direction_od_subset:
        hb_prod_fr = (
            cb.DVector.load(
                r"I:\Prior adjustment\outputs_26_2\Core\hb_productions\hb_normits_tem_segmented_fr_pm_2023.dvec"
            )
            .aggregate(["p", "m", "tp"])
            .filter_segment_value("p", hb_p_subset, keep_filtered=True)
            .filter_segment_value("m", m_subset, keep_filtered=True)
            .filter_segment_value("tp", tp_subset, keep_filtered=True)
            .aggregate_comp_zones(normits)
        )
        hb_attr_fr = (
            cb.DVector.load(
                r"I:\Prior adjustment\outputs_26_2\Core\hb_attractions\hb_normits_tem_segmented_fr_pm_2023.dvec"
            )
            .aggregate(["p", "m", "tp"])
            .filter_segment_value("p", hb_p_subset, keep_filtered=True)
            .filter_segment_value("m", m_subset, keep_filtered=True)
            .filter_segment_value("tp", tp_subset, keep_filtered=True)
            .aggregate_comp_zones(normits)
        )
        dvec_map[1] = (hb_prod_fr, hb_attr_fr)
    if 2 in direction_od_subset:
        hb_prod_to = (
            cb.DVector.load(
                r"I:\Prior adjustment\outputs_26_2\Core\hb_productions\hb_normits_tem_segmented_to_pm_2023.dvec"
            )
            .aggregate(["p", "m", "tp"])
            .filter_segment_value("p", hb_p_subset, keep_filtered=True)
            .filter_segment_value("m", m_subset, keep_filtered=True)
            .filter_segment_value("tp", tp_subset, keep_filtered=True)
            .aggregate_comp_zones(normits)
        )
        hb_attr_to = (
            cb.DVector.load(
                r"I:\Prior adjustment\outputs_26_2\Core\hb_attractions\hb_normits_tem_segmented_to_pm_2023.dvec"
            )
            .aggregate(["p", "m", "tp"])
            .filter_segment_value("p", hb_p_subset, keep_filtered=True)
            .filter_segment_value("m", m_subset, keep_filtered=True)
            .filter_segment_value("tp", tp_subset, keep_filtered=True)
            .aggregate_comp_zones(normits)
        )
        dvec_map[2] = (hb_prod_to, hb_attr_to)

    prod = pd.concat({k: v[0].data for k, v in dvec_map.items()})
    attr = pd.concat({k: v[1].data for k, v in dvec_map.items()})
    prod.index.names = ["direction_od", "p", "m", "tp"]
    prod = cb.DVector(import_data=prod, segmentation=full_seg_p, zoning_system=normits)
    attr.index.names = ["direction_od", "p", "m", "tp"]
    attr = cb.DVector(import_data=attr, segmentation=full_seg_p, zoning_system=normits)

    _4d_constraint_gravity_model(
        prod,
        attr,
        "fullrun_06_03_v2",
        tlds,
        tld_lookup,
        costs,
        normits_noham_sector,
        postme_purpose,
        pathlib.Path(r"I:\Prior adjustment\distribution\distribute_outputs\fullrun_06_03_v2"),
        False,
        True,
        adj_target_options,
        2,
    )

    # TESTING DIFFERENT ADJUSTMENT COMBINATIONS
    # _4d_constraint_gravity_model(
    #     prod, attr, 'fullrun_06_03_v2',
    #     tlds, tld_lookup,
    #     costs, normits_noham_sector, postme_purpose,
    #     pathlib.Path(r'I:\Prior adjustment\distribution\distribute_outputs\fullrun_06_03_v2'),
    #     False, True, {
    #         'triple_targ': (True, 5),
    #         'o_target': (False, 10),
    #         'd_target': (False, 10),
    #         'sec_target': (False, 10)
    #     }, 2
    # )

    # _4d_constraint_gravity_model(
    #     prod, attr, 'fullrun_06_03_v2',
    #     tlds, tld_lookup,
    #     costs, normits_noham_sector, postme_purpose,
    #     pathlib.Path(r'I:\Prior adjustment\distribution\distribute_outputs\fullrun_06_03_v2'),
    #     False, True, {
    #         'triple_targ': (True, 5),
    #         'o_target': (True, 10),
    #         'd_target': (True, 10),
    #         'sec_target': (False, 10)
    #     }, 2
    # )

    # _4d_constraint_gravity_model(
    #     prod, attr, 'fullrun_06_03_v2',
    #     tlds, tld_lookup,
    #     costs, normits_noham_sector, postme_purpose,
    #     pathlib.Path(r'I:\Prior adjustment\distribution\distribute_outputs\fullrun_06_03_v2'),
    #     False, True, {
    #         'triple_targ': (True, 5),
    #         'o_target': (True, 10),
    #         'd_target': (True, 10),
    #         'sec_target': (True, 10)
    #     }, 2
    # )

    # _4d_constraint_gravity_model(
    #     prod, attr, 'fullrun_06_03_v2',
    #     tlds, tld_lookup,
    #     costs, normits_noham_sector, postme_purpose,
    #     pathlib.Path(r'I:\Prior adjustment\distribution\distribute_outputs\fullrun_06_03_v2'),
    #     False, True, {
    #         'triple_targ': (True, 5),
    #         'o_target': (False, 10),
    #         'd_target': (False, 10),
    #         'sec_target': (True, 10)
    #     }, 2
    # )

    # _4d_constraint_gravity_model(
    #     prod, attr, 'fullrun_06_03_v2',
    #     tlds, tld_lookup,
    #     costs, normits_noham_sector, postme_purpose,
    #     pathlib.Path(r'I:\Prior adjustment\distribution\distribute_outputs\fullrun_06_03_v2'),
    #     False, True, {
    #         'triple_targ': (False, 5),
    #         'o_target': (True, 10),
    #         'd_target': (True, 10),
    #         'sec_target': (False, 10)
    #     }, 2
    # )

    # _4d_constraint_gravity_model(
    #     prod, attr, 'fullrun_06_03_v2',
    #     tlds, tld_lookup,
    #     costs, normits_noham_sector, postme_purpose,
    #     pathlib.Path(r'I:\Prior adjustment\distribution\distribute_outputs\fullrun_06_03_v2'),
    #     False, True, {
    #         'triple_targ': (False, 5),
    #         'o_target': (True, 10),
    #         'd_target': (True, 10),
    #         'sec_target': (True, 10)
    #     }, 2
    # )

    # _4d_constraint_gravity_model(
    #     prod, attr, 'fullrun_06_03_v2',
    #     tlds, tld_lookup,
    #     costs, normits_noham_sector, postme_purpose,
    #     pathlib.Path(r'I:\Prior adjustment\distribution\distribute_outputs\fullrun_06_03_v2'),
    #     False, True, {
    #         'triple_targ': (False, 5),
    #         'o_target': (False, 10),
    #         'd_target': (False, 10),
    #         'sec_target': (True, 10)
    #     }, 2
    # )

if __name__ == "__main__":
    main()
