from caf.distribute import furness
import pandas as pd
import caf.base as cb
from pathlib import Path
import caf.toolkit as ctk
from caf.mat.matrices import MatricesBase, MemoryMatrices, MatrixFiles, MatrixType
import logging

logging.basicConfig(
    level=logging.INFO,  # Set the logging level
    format='%(asctime)s - %(levelname)s - %(message)s',  # Log message format
    filename=r"D:\NorMITs Demand\ntem_emp_test\canca\seg.log",
    filemode='a' # Append mode
)


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
    tld_seg_agg = [i for i in agg_mat.segmentation.names if i in tlds.index.names]
    tld_seg_full = [i for i in prod.segmentation.names if i in tlds.index.names]
    for seg_slice in agg_mat.segmentation.iter_slices():
        mat = agg_mat.get_matrix(seg_slice).data.values
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
        LOG.info(f"############# segment={seg_slice.generate_name} #############")
        if rmse > 1e-6:
            row_adj = mat.sum(axis=1) / row_tot
            col_adj = mat.sum(axis=0) / col_tot
            for slice_ in triple_inputs.keys():
                triple_inputs[slice_].row_targets *= row_adj
                triple_inputs[slice_].col_targets *= col_adj
            LOG.warning(f"for {seg_slice.generate_name}, rmse of tripends to target mat = {rmse}")
        hdf_file = home_dir / "outputs" / f"matrices_{te_seg.generate_name}.hdf"
        # seed = mat.copy()
        # seed.columns.name = 'd'
        # seed.index = seed.index.set_levels(seed.columns, level='o')
        # seed = seed.stack().to_xarray()
        furnessed, rmse, checkers = furness.segmentation_furness(triple_inputs,
                                                mat,
                                                (5430,5430),
                                                tol=1e-5,
                                                # seed_mat=seed,
                                                #outer_max_iters=10,
                                                #max_iters=50
                                                )
        
        checkers.reset_index(level='target_prop', inplace=True)
        tld_ref.index.names = prod.segmentation.naming_order + ['area', 'from', 'to']
        checkers['from'] = tld_ref.index.get_level_values('from')
        checkers['to'] = tld_ref.index.get_level_values('to')
        checkers.set_index(['from','to'], append=True, inplace=True)
        furnessed.columns = mat.columns
        furnessed.index = furnessed.index.set_levels(mat.index, level='o')
        furnessed.to_hdf(hdf_file, key='data')
        checkers.to_hdf(hdf_file, key='checks')

if __name__ == "__main__":
    LOG = logging.getLogger(__name__)
    segmentation_input = cb.SegmentationInput(naming_order=['direction_od','m','p','tp','ca'],
                                              enum_segments=['direction_od','m','p','tp','ca'],
                                              subsets={'m':[6],
                                                       'tp':[1,2,3,4]})
    normits = cb.ZoningSystem.get_zoning('normits')
    ca_seg = cb.Segmentation(segmentation_input)
    prod = cb.DVector.load(r"D:\NorMITs Demand\ntem_emp_test\tripends\prod.dvec").aggregate(ca_seg).filter_segment_value('m', 6, keep_filtered=True).filter_segment_value('tp', [1,2,3,4])
    attr = cb.DVector.load(r"D:\NorMITs Demand\ntem_emp_test\tripends\attr.dvec").aggregate(ca_seg).filter_segment_value('m', 6, keep_filtered=True).filter_segment_value('tp', [1,2,3,4])
    agg_mat = MatrixFiles(ca_seg.remove_segment('ca'),
                          normits,
                          MatrixType.OD,
                          Path(r"D:\NorMITs Demand\ntem_emp_test\matrices"))
    costs = pd.read_csv(r"I:\Prior adjustment\distribution\costs\normits_costs_m6_tp1.csv", index_col=0)
    tlds = pd.read_csv(r"D:\NorMITs Demand\ntem_emp_test\tlds\combined_tlds_canca_new.csv", index_col=[0,1,2,3,4,5])['trips']
    tlds = tlds.rename({'hb_fr':1, 'hb_to':2, 'nhb':0})
    zones = pd.read_csv(r"I:\NorMITs Distribution\voa_gb_2023_uni\NorMITs_zone.csv", index_col=0).sort_index().reset_index(drop=True)
    dist(attr, prod, agg_mat, tlds, costs.values, Path(r"D:\NorMITs Demand\ntem_emp_test\canca"), zones)
    