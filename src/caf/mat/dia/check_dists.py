import caf.distribute as cd
from caf.distribute.gravity_model import core
import pandas as pd
from pathlib import Path
from caf.toolkit import cost_utils, math_utils
import caf.base as cb
import numpy as np

def weighted_difference_metric(series_a, series_b, epsilon=1e-8):
    """
    Calculate a custom weighted difference metric between two pandas Series.
    
    Parameters:
        series_a (pd.Series): First series of values.
        series_b (pd.Series): Second series of values.
        return_breakdown (bool): If True, returns a DataFrame with element-wise contributions.
        epsilon (float): Small value to avoid division by zero.
        
    Returns:
        float: Total weighted difference.
        pd.DataFrame (optional): Breakdown of contributions if return_breakdown is True.
    """
    if not series_a.index.equals(series_b.index):
        raise ValueError("Indices of the two series must match.")
    
    abs_diff = (series_a - series_b).abs()
    max_vals = np.maximum(series_a.abs(), series_b.abs()) + epsilon
    rel_diff = abs_diff / max_vals
    avg_magnitude = (series_a.abs() + series_b.abs()) / 2
    weighted = abs_diff * rel_diff
    

    breakdown = pd.DataFrame({
        'Absolute Difference': abs_diff,
        'Relative Difference': rel_diff,
        'Average Magnitude': avg_magnitude,
        'Weighted Contribution': weighted
    })
    return breakdown


def check_against_tlds(output_file, zones):
    achieved_dist = pd.read_hdf(output_file, key='checks')
    seg_lev = {}
    for seg in achieved_dist.index.get_level_values(0).unique():
        area_lev = {}
        for area in zones['tld_area'].unique():
            cost_df = achieved_dist.loc[seg, area].reset_index()
            convergence = math_utils.curve_convergence(cost_df.target_prop, cost_df.act_prop)
            area_lev[area] = convergence
        seg_lev[seg] = pd.Series(area_lev)
    df = pd.concat(seg_lev, axis=1)
    print(df)
    df.to_hdf(output_file, key='convergence')

def check_againt_tripends(output_file, prod, attr):
    mat = pd.read_hdf(output_file, key='data')
    o_mat = mat.stack().groupby(level=['seg','o']).sum()
    o_mat.index.names = prod.index.names
    d_mat = mat.stack().groupby(level=['seg','d']).sum()
    d_mat.index.names = attr.index.names
    o_diff = weighted_difference_metric(o_mat, prod)
    d_diff = weighted_difference_metric(d_mat, attr)
    # print(f"zeros = {np.count_nonzero(mat.values==0) / len(mat.stack())}")
    print(f"r squared prod = {math_utils.curve_convergence(o_mat, prod)}")
    print(f"r squared attr = {math_utils.curve_convergence(d_mat, attr)}")
    return pd.concat({'prod':o_diff, 'attr':d_diff}, axis=1)

if __name__ == '__main__':
    home_dir = Path(r"E:\dia_seg")
    outputs = home_dir / 'outputs'
    areas = {'NE': 1, 'NW': 2, 'YH': 3, 'South': 4, 'Scotland': 5}
    zones = pd.read_csv(home_dir / "target_mats" / "zones.csv",
                        index_col=0).sort_index().reset_index(drop=True)
    zones['tld_area'] = zones['tld_area'].replace(areas)
    costs = pd.read_csv(r"E:\costs\CSVs\ptnet_cost_ip-rail.csv", index_col=0).to_numpy()
    prod = cb.DVector.load(r"E:\dia_seg\tripends\prod_g.dvec").data
    attr = cb.DVector.load(r"E:\dia_seg\tripends\attr_g_balanced.dvec").data
    out = {}
    for uc in [1,2,3,4,5]:
        inner = {}
        for ca in [1,2]:
            print(f"############### uc = {uc}, ca = {ca} ###############")
            # te_check = check_againt_tripends(outputs / 'gender_3' / f"matrices_uc{uc}_ca{ca}.hdf",
            #                       prod.loc[uc,ca].stack(),
            #                       attr.loc[uc,ca].stack())
            # inner[ca] = te_check.unstack(level='gender_3').describe()
            check_against_tlds(outputs / 'ns_sec' / f"matrices_uc{uc}_ca{ca}_refurn.hdf", zones)
    #     out[uc] = pd.concat(inner)
    # pd.concat(out).to_csv(outputs / 'gender_3' / "diff_summary.csv")