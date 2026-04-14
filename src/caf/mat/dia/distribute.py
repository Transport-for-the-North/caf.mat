from caf.distribute import furness
import pandas as pd
import caf.base as cb
from pathlib import Path
import caf.toolkit as ctk
from caf.mat.matrices import MatricesBase, MemoryMatrices, MatrixFiles, MatrixType
import logging
from caf.toolkit.concurrency import multiprocess
import numpy as np
import os

def smart_downcast(arr):
    """Conservatively downcast to most efficient dtype"""
    if arr.dtype == np.float64:
        # Only downcast if values are reasonable and precision loss is minimal
        arr_f32 = arr.astype(np.float32).astype(np.float64)
        max_error = np.abs(arr - arr_f32).max()
        arr_max = np.abs(arr).max()
        
        # Only downcast if max value is reasonable for float32 and error is very small
        if arr_max < 1e6 and arr_max > 1e-6 and max_error / arr_max < 1e-10:
            return arr.astype(np.float32)
    
    elif arr.dtype == np.int64:
        if arr.max() < 2**31 - 1 and arr.min() > -2**31:
            return arr.astype(np.int32)
        elif arr.max() < 2**16 - 1 and arr.min() > -2**16:
            return arr.astype(np.int16)
    
    return arr  # Return original if no downcast possible

def multi_loop(triple_inputs, mat, hdf_file, tld_ref, segmentation, zoning):
    # Create separate log file for each process/iteration
    log_dir = hdf_file.parent.parent / "logs"
    log_dir.mkdir(exist_ok=True)
    
    # Extract segment name from hdf_file path for unique log naming
    segment_name = hdf_file.stem.replace("matrices_", "")
    process_id = os.getpid()  # Get process ID for uniqueness
    log_file = log_dir / f"{segment_name}_pid{process_id}.log"
    
    # Set up per-process logging
    logger = logging.getLogger(f"distribute_{segment_name}")
    logger.setLevel(logging.INFO)
    
    # Remove any existing handlers to avoid duplicates
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Create file handler for this process
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    
    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    
    # Add handler to main process logger
    logger.addHandler(file_handler)
    
    # Also attach the same handler to furness logger so its internal logs go to this file
    furness_logger = logging.getLogger("caf.distribute.furness")
    furness_logger.setLevel(logging.INFO)
    furness_logger.addHandler(file_handler)
    furness_logger.propagate = False
    
    logger.info(f"Starting processing for segment: {segment_name}")

    furnessed, rmse, checkers = furness.segmentation_furness(triple_inputs,
                                                mat,
                                                (5430,5430),
                                                tol=1e-5,
                                                )
    logger.info(f"Finished furness with RMSE: {rmse}")
    checkers.reset_index(level='target_prop', inplace=True)
    checkers = checkers[checkers['target_prop'] > 0]
    tld_ref = tld_ref[tld_ref > 0]
    tld_ref.index.names = segmentation + ['area', 'from', 'to']
    checkers['from'] = tld_ref.index.get_level_values('from')
    checkers['to'] = tld_ref.index.get_level_values('to')
    checkers.set_index(['from','to'], append=True, inplace=True)
    furnessed.columns = zoning
    furnessed.index = furnessed.index.set_levels(zoning, level='o')
    furnessed.to_hdf(hdf_file, key='data')
    checkers.to_hdf(hdf_file, key='checks')
    
    logger.info(f"Completed processing and saved to {hdf_file}")
    
    # Clean up handlers
    logger.removeHandler(file_handler)
    furness_logger.removeHandler(file_handler)
    file_handler.close()

def dist(attr: cb.DVector, prod: cb.DVector, agg_mat: MatricesBase, tlds: pd.DataFrame, costs: pd.DataFrame, home_dir: Path, zones):
    """
    Produce new matrices at a more detailed segmentation than an existing distribution.
    Parameters
    ----------
    attr : cb.DVector
        Attraction trip ends at the more detailed level of segmentation.
    prod : cb.DVector
        Production trip ends at the more detailed level of segmentation.
    agg_mat : MatricesBase
        Existing distribution at more aggregate segmentation. The segmentation must be a subset of 
        attr and prod segmentations. All three of these first three inputs should "match" (e.g. sum 
        the same at the appropriate levels of zone and segmentation.)
    tlds : pd.Series
        Tlds at the detailed level of segmentation. These must be multiindexed with index levels matching 
        the more detailed segmentation, plus 'tld_area' and 'trav_dist'.
    costs : pd.DataFrame
        Costs at the same zoning as everything else.
    home_dir : Path
        Directory outputs will be saved to.
    zones : _type_
        _description_
    """
    # Set up logging for this distribution run
    LOG = logging.getLogger("main")
    LOG.setLevel(logging.INFO)  # Only log warnings and errors
    
    tld_seg_agg = [i for i in agg_mat.segmentation.names if i in tlds.index.names]
    tld_seg_full = [i for i in prod.segmentation.names if i in tlds.index.names]
    inputs = []
    for seg_slice in agg_mat.segmentation.iter_slices():
        mat = smart_downcast(agg_mat.get_matrix(seg_slice).data.values)
        triple_inputs = {}
        row_tot = 0
        col_tot = 0
        # Adjust to matrix
        adjustor = {}
        for area in zones['tld_area'].unique():
            tld_zones = zones[zones['tld_area'] == area].index
            tld = tlds.groupby(tld_seg_agg + ['tld_area','trav_dist']).sum().loc[seg_slice.aggregate(tld_seg_agg).as_tuple()].loc[area].to_frame()
            tld['comp'] = ctk.cost_utils.cost_distribution(mat[tld_zones], costs[tld_zones], max_bounds=tld.index, min_bounds=tld.reset_index()['trav_dist'].shift().fillna(0))
            tld /= tld.sum()
            tld['adj'] = tld['comp'] / tld['trips']
            adjustor[area] = tld['adj']
        adjustor = pd.concat(adjustor)
        adjustor[adjustor>20] = 20
        tld_dicts = {}
        for te_seg in prod.segmentation.iter_slices(filter_=seg_slice.data):
            props = {}
            tld_dict = {}
            for area in zones['tld_area'].unique():
                tld = tlds.xs(te_seg.aggregate(tld_seg_full).as_tuple(), level=tld_seg_full).loc[area].copy().sort_index().reset_index()
                tld['from'] = tld['trav_dist'].shift().fillna(0)
                tld  = tld.set_index(['from','trav_dist']).squeeze().fillna(0).mul(adjustor.loc[area])
                tld.name = 'trips'
                tld /= tld.sum()
                tld_zones = zones[zones['tld_area'] == area].index
                prop, unique = furness.cost_to_prop(
                costs[tld_zones],
                tld.reset_index(),
                'trips'
            )
                tld_dict[area] = tld
                
                props[area] = furness.PropsInput(prop, tld_zones, unique)
            tld_dicts[te_seg.as_tuple()] = pd.concat(tld_dict)
            row = prod.get_slice(te_seg).values
            row_tot += row
            col = attr.get_slice(te_seg).values
            col_tot += col
            triple_inputs[te_seg.generate_name()] = furness.SegInput(props, col_targets=col, row_targets=row)
        tld_ref = pd.concat(tld_dicts)
        rmse = furness.calc_rmse(col_tot, mat, row_tot)
        if rmse > 1e-6:
            mat_row = mat.sum(axis=1)
            mat_col = mat.sum(axis=0)
            for slice_ in triple_inputs.keys():
                triple_inputs[slice_].row_targets = np.divide(triple_inputs[slice_].row_targets, row_tot, where=row_tot != 0,out=np.ones_like(row_tot, dtype=float)) * mat_row
                triple_inputs[slice_].col_targets = np.divide(triple_inputs[slice_].col_targets, col_tot, where=col_tot != 0,out=np.ones_like(col_tot, dtype=float)) * mat_col
            LOG.warning(f"for {seg_slice.generate_name()}, rmse of tripends to target mat = {rmse}")
        hdf_file = home_dir / "outputs" / f"matrices_{seg_slice.generate_name()}.hdf"
        # seed = mat.copy()
        # seed.columns.name = 'd'
        # seed.index = seed.index.set_levels(seed.columns, level='o')
        # seed = seed.stack().to_xarray()
        inputs.append((triple_inputs, mat, hdf_file, tld_ref, prod.segmentation.naming_order, prod.zoning_system.zone_ids))
        del triple_inputs, mat, tld_ref
        # furnessed, rmse, checkers = furness.segmentation_furness(triple_inputs,
        #                                         mat,
        #                                         (5430,5430),
        #                                         tol=1e-5,
        #                                         # seed_mat=seed,
        #                                         #outer_max_iters=10,
        #                                         #max_iters=50
        #                                         )
        
        # checkers.reset_index(level='target_prop', inplace=True)
        # tld_ref.index.names = prod.segmentation.naming_order + ['area', 'from', 'to']
        # checkers['from'] = tld_ref.index.get_level_values('from')
        # checkers['to'] = tld_ref.index.get_level_values('to')
        # checkers.set_index(['from','to'], append=True, inplace=True)
        # furnessed.columns = attr.zoning_system.zone_ids
        # furnessed.index = furnessed.index.set_levels(prod.zoning_system.zone_ids, level='o')
        # furnessed.to_hdf(hdf_file, key='data')
        # checkers.to_hdf(hdf_file, key='checks')
    del agg_mat, prod, attr
    multiprocess(multi_loop, arg_list=inputs)

if __name__ == "__main__":
    # Set up main process logging
    main_log_file = r"D:\NorMITs Demand\ntem_emp_test\canca\main_process.log"
    main_logger = logging.getLogger("main")
    main_logger.setLevel(logging.INFO)
    
    # Remove any existing handlers
    for handler in main_logger.handlers[:]:
        main_logger.removeHandler(handler)
    
    main_handler = logging.FileHandler(main_log_file)
    main_handler.setLevel(logging.INFO)
    main_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    main_handler.setFormatter(main_formatter)
    main_logger.addHandler(main_handler)
    
    main_logger.info("Starting distribution process")
    
    for tp in [1,2,3,4]:
        main_logger.info(f"Processing time period: {tp}")
        segmentation_input = cb.SegmentationInput(naming_order=['direction_od','m','p','tp','ca'],
                                                enum_segments=['direction_od','m','p','tp','ca'],
                                                subsets={'m':[6],
                                                        'tp':[tp]})
        normits = cb.ZoningSystem.get_zoning('normits')
        ca_seg = cb.Segmentation(segmentation_input)
        prod = cb.DVector.load(r"D:\NorMITs Demand\ntem_emp_test\tripends\prod.dvec").aggregate(ca_seg).filter_segment_value('m', 6, keep_filtered=True).filter_segment_value('tp', tp, keep_filtered=True)
        attr = cb.DVector.load(r"D:\NorMITs Demand\ntem_emp_test\tripends\attr.dvec").aggregate(ca_seg).filter_segment_value('m', 6, keep_filtered=True).filter_segment_value('tp', tp, keep_filtered=True)
        agg_mat = MatrixFiles(ca_seg.remove_segment('ca'),
                            normits,
                            MatrixType.OD,
                            Path(r"D:\NorMITs Demand\ntem_emp_test\matrices"))
        costs = pd.read_csv(r"I:\Prior adjustment\distribution\costs\normits_costs_m6_tp1.csv", index_col=0, dtype=np.float32)
        tlds = pd.read_csv(r"D:\NorMITs Demand\ntem_emp_test\tlds\combined_tlds_canca_new.csv", index_col=[0,1,2,3,4,5])['trips']
        tlds = tlds.rename({'hb_fr':1, 'hb_to':2, 'nhb':0})
        zones = pd.read_csv(r"I:\NorMITs Distribution\voa_gb_2023_uni\NorMITs_zone.csv", index_col=0).sort_index().reset_index(drop=True)
        
        main_logger.info(f"Loaded data for tp {tp}: {len(tlds)} TLD records, {len(zones)} zones")
        
        dist(attr, prod, agg_mat, tlds, costs.values, Path(r"D:\NorMITs Demand\ntem_emp_test\canca"), zones)
        
        main_logger.info(f"Completed processing for time period {tp}")
    
    main_logger.info("All distribution processes completed")
    
    # Clean up main logger
    main_logger.removeHandler(main_handler)
    main_handler.close()
    