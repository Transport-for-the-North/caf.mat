import pandas as pd
from pathlib import Path
gor_to_agg = {1:'NE', 2:'NW', 3:'YH', 4:'South', 5:'South', 6:'South', 7:'South', 8:'South', 9:'South', 10:'South', 11:'Scotland'}
home_dir = Path(r"E:\dia_seg")


def p_dir_to_uc(p_col, d_col, df):
    hb = df.xs('hb_fr', level=d_col).reset_index()
    nhb = df.xs('nhb', level=d_col).reset_index()
    hb['uc'] = hb[p_col].replace({1:2,2:1,3:3,4:3,5:3,6:3,7:3,8:3})
    hb['uc_name'] = hb['uc'].replace({1: 'Business', 2: 'Commuting', 3: 'Other'})
    nhb['uc'] = nhb[p_col].replace({1:5,2:4,3:5,4:5,5:5,6:5,7:5,8:5})
    nhb['uc_name'] = nhb['uc'].replace({4: 'Business', 5: 'Other'})
    return pd.concat([hb,nhb]).drop(p_col, axis=1)

def use_moira(tlds, wave_tlds, s2s_tlds, keep_ind, seg):
    s2s_int = s2s_tlds.groupby(['uc_name', 'ca','area', 'trav_dist'])['trips'].sum() / s2s_tlds.groupby(['uc_name', 'ca','area'])['trips'].sum()
    wave_int = wave_tlds.groupby(['uc_name', 'area','trav_dist'])['trips'].sum() / wave_tlds.groupby(['area','uc_name'])['trips'].sum()
    s2s_int = collapse_multiindex_to_kept(s2s_int, keep_ind, 'trav_dist')
    wave_int = collapse_multiindex_to_kept(wave_int, keep_ind, 'trav_dist')
    tlds_int = tlds.groupby(['uc_name', 'ca',seg, 'area', 'uc', 'trav_dist'])['trips'].sum() / tlds.groupby(['uc_name', 'ca',seg, 'area', 'uc'])['trips'].sum()
    factor = (wave_int / s2s_int).reorder_levels(['trav_dist','uc_name','area','ca'])
    # factor_5 = factor.loc[10].to_frame()
    # factor_5['trav_dist'] = 5
    # factor_5.set_index('trav_dist', inplace=True, append=True)
    # factor = pd.concat([factor, factor_5.reorder_levels(factor.index.names)]).squeeze()
    factor[factor > 20] = 20
    factor[factor<0.05] = 0.05
    adjusted = tlds_int * factor
    return adjusted / adjusted.groupby(['ca',seg,'area','uc']).sum()

def preprocess_tlds(tlds):
    #filter
    df = tlds.drop([5,6], level='period').drop('hb_to', level='direction').droplevel('mode')
    hb = df.xs('hb_fr', level='direction').reset_index()
    nhb = df.xs('nhb', level='direction').reset_index()
    hb['uc'] = hb['purpose'].replace({1:2,2:1,3:3,4:3,5:3,6:3,7:3,8:3})
    nhb['uc'] = nhb['purpose'].replace({1:5,2:4,3:5,4:5,5:5,6:5,7:5,8:5})
    df = pd.concat([hb,nhb]).groupby(['hh_type','gender','uc','To Kilometres', 'gor'])['trips_sum'].sum().unstack(level='gor').rename({'ca':2, 'nca':1})
    return df



def collapse_multiindex_to_kept(
    df: pd.DataFrame,
    keep: list,
    level: int | str
) -> pd.DataFrame:
    """
    Collapse a MultiIndex DataFrame by summing values from dropped indices into the next kept index
    at the specified level.

    Parameters:
    - df (pd.DataFrame): The input DataFrame with a MultiIndex.
    - keep (list): A list of values to keep at the specified index level.
    - level (int or str): The level of the MultiIndex to apply the collapsing logic.

    Returns:
    - pd.DataFrame: A new DataFrame with collapsed values and only the kept values at the specified level.
    """
    # Ensure the level is valid
    if level not in df.index.names and not isinstance(level, int):
        raise ValueError(f"Level '{level}' not found in index.")

    # Get the level values
    level_values = df.index.get_level_values(level)
    unique_values = pd.Index(level_values).unique().sort_values()

    # Sort the keep list to ensure correct mapping
    keep_sorted = sorted(keep)

    # Create a mapping from each value to the next kept value down
    mapping = {}
    current_keep = keep_sorted[0]
    for val in unique_values:
        if val in keep_sorted:
            current_keep = val
        mapping[val] = current_keep

    # Replace the level with the mapped values
    new_index = list(df.index)
    for i, idx in enumerate(new_index):
        idx_list = list(idx)
        idx_list[df.index._get_level_number(level)] = mapping[level_values[i]]
        new_index[i] = tuple(idx_list)

    # Create a new DataFrame with the updated index
    df_mapped = df.copy()
    df_mapped.index = pd.MultiIndex.from_tuples(new_index, names=df.index.names)

    # Group by the new index and sum
    df_grouped = df_mapped.groupby(level=df_mapped.index.names).sum()

    # Filter to only include rows where the specified level is in `keep`
    mask = df_grouped.index.get_level_values(level).isin(keep_sorted)
    return df_grouped[mask]


def preprocess_old_tlds(tld_dir, zones, bins: list[int]):
    uc1=[]
    uc2=[]
    uc3=[]
    uc4=[]
    uc5=[]
    min_bins = bins
    for p in [1,2,3,4,5,6,7,8]:
        for direction in ['hb_fr', 'nhb']:
            out = {}
            for ca, ca_num in {'ca':2,'nca':1}.items():
                nhan_tld = pd.read_csv(tld_dir / f"NTS_tld_m6_p{p}_{direction}_{ca}.csv")
                nhan_tld = nhan_tld.merge(zones, left_on=';index', right_on='tld_area')
                nhan_tld['tld_area'] = nhan_tld['gor'].replace({1:'NE', 2:'NW', 3:'YH', 4:'South', 5:'South', 6:'South', 7:'South', 8:'South', 9:'South', 10:'South', 11:'Scotland'})
                nhan_tld = nhan_tld.groupby(['tld_area', 'trav_dist'])['trips'].sum().unstack(level='tld_area')
                filtered_index = nhan_tld.index.intersection(bins)
                min_bins = min_bins.intersection
                out[ca_num] = collapse_to_kept_indices(nhan_tld, filtered_index)
            tld = pd.concat(out)
            if direction == 'hb_fr':
                if p == 1:
                    uc2.append(tld)
                elif p==2:
                    uc1.append(tld)
                else:
                    uc3.append(tld)
            else:
                if p == 2:
                    uc4.append(tld)
                else:
                    uc5.append(tld)
    return pd.concat({1:uc1, 2:uc2, 3:uc3, 4:uc4, 5:uc5})
        


def process_tlds(seg_tlds: dict[int, pd.DataFrame],
                 target_tld: pd.DataFrame):
    return
    
    

# for p in [1,2,3,4,5,6,7,8]:
#     for ca in ['ca','nca']:
#         nhan_tld = pd.read_csv(nhan_tlds_dir / f"NTS_tld_m6_p{p}_hb_fr_{ca}.csv")
#         nhan_tld = nhan_tld.merge(zones, left_on=';index', right_on='tld_area')
#         nhan_tld['tld_area'] = nhan_tld['gor'].replace({1:'NE', 2:'NW', 3:'YH', 4:'South', 5:'South', 6:'South', 7:'South', 8:'South', 9:'South', 10:'South', 11:'Scotland'})
#         nhan_tld = nhan_tld.groupby(['tld_area', 'trav_dist'])['trips'].sum()
#         nhan_tld = nhan_tld / nhan_tld.groupby(level='tld_area').sum()
#         alok_tld = alok_tlds.loc[ca,'hb_fr',6,p].reset_index()
#         alok_tld_seg = alok_tld.groupby(['gor', 'gender', 'To Kilometres'])['trips_sum'].sum()
#         alok_tld = alok_tld.groupby(['gor', 'To Kilometres'])['trips_sum'].sum()
#         alok_tld = alok_tld / alok_tld.groupby(level='gor').sum()
#         checks = {}
#         for area in alok_tld.index.get_level_values('gor'):
#             alok_check = alok_tld.loc[area]
#             nhan_check = nhan_tld.loc[area]
#             comp = [i for i in alok_check.index if i in nhan_check.index]
#             check = alok_check.to_frame().join(nhan_check, how='inner')
#             checks[area] = check
#         check = pd.concat(checks)
#         check.index.names = alok_tld.index.names
#         check.columns = ['Alok', 'Nhan']
#         check['adj'] = check['Nhan'] / check['Alok']
#         alok_tld_seg *= check['adj']
#         check.to_csv(home_dir / f"tld_check_{ca}_p{p}_hb_fr.csv")
if __name__ == "__main__":
    nhan_tlds_dir = Path(r"I:\NorMITs Distribution\voa_gb_2023_uni_i1\iter3a_rail_only\tld")
    alok_tlds = pd.read_csv(r"T:\AlokJain\tld\gender\Aggregated.csv", index_col=[0,1,2,3,4,5,6,7])
    alok_tlds = p_dir_to_uc('purpose','direction',alok_tlds).rename(columns={'hh_type':'ca', 'To Kilometres': 'trav_dist', 'gor':'area', 'trips_sum':'trips'})
    alok_tlds = alok_tlds[alok_tlds['period']<5]
    alok_tlds = alok_tlds.groupby(['ca','gender','uc','uc_name','area','trav_dist'])['trips'].sum()
    wave_tlds = pd.read_csv(r"I:\NorMITs Distribution\voa_gb_2023_uni_i1\iter3a_rail_only\inputs\wavelength_tld.csv", index_col=[0,1,2]).rename(gor_to_agg, level='orig tlc')
    wave_tlds.index.names = ['area','uc_name','trav_dist']
    s2s_tlds = pd.read_csv(r"I:\NorMITs Distribution\voa_gb_2023_uni_i1\iter3a_rail_only\nts_tld\trip_length_distribution_s2s_ca.csv", index_col=[0,2,3,4,5,6])
    s2s_tlds = p_dir_to_uc('purpose','direction',s2s_tlds).rename(columns={'hh_type':'ca'})
    s2s_tlds = s2s_tlds[s2s_tlds['mode'] == 6]
    s2s_tlds['area'] = s2s_tlds['triporiggor_b02id'].replace(gor_to_agg)
    keep_ind = wave_tlds.index.get_level_values('trav_dist').unique().drop(5)
    alok_tlds = collapse_multiindex_to_kept(alok_tlds, keep_ind, 'trav_dist')
    adjusted = use_moira(alok_tlds.reset_index(), wave_tlds, s2s_tlds, keep_ind, 'gender')
    adjusted.to_csv("E:/dia_seg/adjusted_tlds_gender.csv")
    
    
    zones = pd.read_csv(r"I:\NorMITs Distribution\voa_gb_2023_uni\NorMITs_zone.csv", index_col=0).sort_index().reset_index(drop=True)
    zones = zones[['tld_area', 'gor']].drop_duplicates()
    new_tlds = alok_tlds = preprocess_tlds(alok_tlds)
    preprocess_old_tlds(nhan_tlds_dir, zones, list(new_tlds.index.get_level_values('To Kilometres').unique()))

    