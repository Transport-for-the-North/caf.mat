from caf.distribute import furness
import pandas as pd
import caf.base as cb
from pathlib import Path
import caf.toolkit as ctk
from caf.mat.matrices import MatricesBase, MemoryMatrices, MatrixFiles
import logging

logging.basicConfig(
    level=logging.INFO,  # Set the logging level
    format='%(asctime)s - %(levelname)s - %(message)s',  # Log message format
    filename=r"E:\dia_seg\outputs\refurness.log",
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
    tlds.rename({'nca':1, 'ca':2}, inplace=True)
    tlds.rename({'NE':1, 'NW':2, 'YH':3, 'South':4, 'Scotland':5}, inplace=True)
    tlds.sort_index(inplace=True)
    for seg_slice in agg_mat.segmentation.iter_slices():
        mat = agg_mat.get_matrix(seg_slice).data
        triple_inputs = {}
        row_tot = 0
        col_tot = 0
        # Adjust to matrix
        adjustor = {}
        for area in zones['tld_area'].unique():
            tld_zones = zones[zones['tld_area'] == area].index
            tld = tlds.groupby(list(seg_slice.naming_order) + ['tld_area','trav_dist']).sum().loc[seg_slice.as_tuple()]
            tld['comp'] = ctk.cost_utils.cost_distribution(mat.values[tld_zones], costs[tld_zones], max_bounds=tld.index, min_bounds=tld.reset_index()['trav_dist'].shift().fillna(0))
            tld /= tld.sum()
            tld['adj'] = tld['comp'] / tld['trips']
            adjustor[area] = tld
        adjustor = pd.concat(adjustor)
        adjustor[adjustor>20] = 20
        tld_dicts = {}
        for te_seg in prod.segmentation.iter_slices(filter_=seg_slice.data):
            props = {}
            tld_dict = {}
            for area in [1,2,3,4,5]:
                tld = tlds.loc[te_seg.as_tuple()].copy().sort_index()
                tld['from'] = tld['trav_dist'].shift().fillna(0)
                tld  = tld.set_index(['from','trav_dist']).fillna(0).mul(adjustor.loc[area,'adj'], axis=0)
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
            row = prod.get_slice(te_seg)
            row_tot += row
            col = attr.get_slice(te_seg)
            col_tot += col
            triple_inputs[te_seg.as_tuple()] = furness.SegInput(props, col_targets=col, row_targets=row)
        tld_ref = pd.concat(tld_dicts)
        rmse = furness.calc_rmse(col_tot, mat.values, row_tot)
        LOG.info(f"############# segment={seg_slice.generate_name} #############")
        if rmse > 1e-6:
            LOG.warning(f"for uc:{uc} and ca{ca}, rmse of tripends to target mat = {rmse}")
        hdf_file = home_dir / "outputs" / f"matrices_{te_seg.generate_name}.hdf"
        seed = mat.copy()
        seed.columns.name = 'd'
        seed.index = seed.index.set_levels(seed.columns, level='o')
        seed = seed.stack().to_xarray()
        furnessed, rmse, checkers = furness.segmentation_furness(triple_inputs,
                                                mat.to_numpy(),
                                                (5430,5430),
                                                tol=1e-5,
                                                seed_mat=seed,
                                                #outer_max_iters=10,
                                                #max_iters=50
                                                )
        
        checkers.reset_index(level='target_prop', inplace=True)
        tld_ref.index.names = [seg.name, 'area', 'from', 'to']
        checkers['from'] = tld_ref.index.get_level_values('from')
        checkers['to'] = tld_ref.index.get_level_values('to')
        checkers.set_index(['from','to'], append=True, inplace=True)
        furnessed.columns = mat.columns
        furnessed.index = furnessed.index.set_levels(mat.index, level='o')
        furnessed.to_hdf(hdf_file, key='data')
        checkers.to_hdf(hdf_file, key='checks')

if __name__ == "__main__":
    LOG = logging.getLogger(__name__)
    areas = {'NE':1, 'NW':2, 'YH':3, 'South':4, 'Scotland':5}
    home_dir = Path(r"E:\dia_seg")
    zones = pd.read_csv(home_dir / "target_mats" / "zones.csv", index_col=0).sort_index().reset_index(drop=True)
    zones['tld_area'] = zones['tld_area'].replace(areas)
    costs = pd.read_csv(r"E:\costs\CSVs\ptnet_cost_ip-rail.csv", index_col=0).to_numpy()
    attr_g = cb.DVector.load(r"E:\dia_seg\tripends\attr_g_balanced.dvec")
    prod_g = cb.DVector.load(r"E:\dia_seg\tripends\prod_g.dvec")
    attr_s = cb.DVector.load(r"E:\dia_seg\tripends\attr_s_balanced.dvec")
    prod_s = cb.DVector.load(r"E:\dia_seg\tripends\prod_s.dvec")
    attr_n = cb.DVector.load(r"E:\dia_seg\tripends\attr_n_balanced.dvec")
    prod_n = cb.DVector.load(r"E:\dia_seg\tripends\prod_n.dvec")
    tlds_g = pd.read_csv(r"E:/dia_seg/adjusted_tlds_gender.csv", index_col=[1,2,3,4]).drop('uc_name',axis=1)
    tlds_s = pd.read_csv(r"E:/dia_seg/adjusted_tlds_soc.csv", index_col=[1,2,3,4]).drop('uc_name',axis=1)
    tlds_n = pd.read_csv(r"E:/dia_seg/adjusted_tlds_ns_sec.csv", index_col=[1,2,3,4]).drop('uc_name',axis=1)
    
    # dist(attr_s, prod_s, tlds_s, costs, cb.segments.SegmentsSuper('soc').get_segment(), home_dir, zones)
    dist(attr_g, prod_g, tlds_g, costs, cb.segments.SegmentsSuper('gender_3').get_segment(), home_dir, zones)
    dist(attr_n, prod_n, tlds_n, costs, cb.segments.SegmentsSuper('ns_sec').get_segment(), home_dir, zones)
    

    # tlds.rename({'nca':1, 'ca':2}, inplace=True)
    # tlds.rename({'NE':1, 'NW':2, 'YH':3, 'South':4, 'Scotland':5}, inplace=True)
    # tlds.sort_index(inplace=True)
    # for uc in [1,2,3,4,5]:
    #     for ca in [1,2]:
    #         mat = pd.read_hdf(home_dir / "target_mats" / f"uc{uc}_ca{ca}.hdf")
    #         triple_inputs = {}
    #         row_tot = 0
    #         col_tot = 0
    #         # Adjust to matrix
    #         adjustor = {}
    #         for area in [1,2,3,4,5]:
    #             tld_zones = zones[zones['tld_area'] == area].index
    #             tld = tlds.groupby(['ca','area', 'uc', 'trav_dist']).sum().loc[(ca,area,uc)]
    #             tld['comp'] = ctk.cost_utils.cost_distribution(mat.values[tld_zones], costs[tld_zones], max_bounds=tld.index, min_bounds=tld.reset_index()['trav_dist'].shift().fillna(0))
    #             tld /= tld.sum()
    #             tld['adj'] = tld['comp'] / tld['trips']
    #             adjustor[area] = tld
    #         adjustor = pd.concat(adjustor)
    #         adjustor[adjustor>20] = 20
    #         for g in [1,2,3]:
    #             props = {}
    #             for area in [1,2,3,4,5]:
    #                 tld = tlds.loc[(ca,g,area, uc)].copy().sort_index()
    #                 tld['from'] = tld['trav_dist'].shift().fillna(0)
    #                 tld  = tld.set_index(['from','trav_dist']).fillna(0).mul(adjustor.loc[area,'adj'], axis=0)
    #                 tld /= tld.sum()
    #                 tld_zones = zones[zones['tld_area'] == area].index
    #                 prop, unique = furness.cost_to_prop(
    #                 costs[tld_zones],
    #                 tld.reset_index(),
    #                 'trips'
    #             )
                    
    #                 props[area] = furness.PropsInput(prop, tld_zones, unique))
    #             row = productions.data.loc[uc, ca, g].to_numpy()
    #             row_tot += row
    #             col = attractions.data.loc[uc, ca, g].to_numpy()
    #             col_tot += col
    #             triple_inputs[g] = furness.SegInput(props, col_targets=col, row_targets=row)
    #         rmse = furness.calc_rmse(col_tot, mat.values, row_tot)
    #         LOG.info(f"############# UC={uc}, CA={ca} #############")
    #         if rmse > 1e-6:
    #             LOG.warning(f"for uc:{uc} and ca{ca}, rmse of tripends to target mat = {rmse}")
    #         furnessed, rmse, checkers = furness.segmentation_furness(triple_inputs,
    #                                                 mat.to_numpy(),
    #                                                 (5430,5430),
    #                                                 tol=1e-5)
    #         furnessed.columns = mat.columns
    #         furnessed.index = furnessed.index.set_levels(mat.index, level='o')
    #         furnessed.to_hdf(home_dir / f"gender_matrices_uc{uc}_ca{ca}.hdf", key='data')
    #         checkers.to_hdf(home_dir / f"gender_matrices_uc{uc}_ca{ca}.hdf", key='checks')
    # print('debugging')