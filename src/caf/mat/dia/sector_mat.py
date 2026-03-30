import pandas as pd

# cb = pd.read_csv(r"I:\NTS\classified builds\cb_tfn_v2023.2.csv")
gor_to_agg = {1:'NE', 2:'NW', 3:'YH', 4:'South', 5:'South', 6:'South', 7:'South', 8:'South', 9:'South', 10:'South', 11:'Scotland'}
# hb = cb[cb['direction'] == 'hb_fr']
# nhb = cb[cb['direction'] == 'nhb']
# hb['uc'] = hb['purpose'].replace({1:2,2:1,3:3,4:3,5:3,6:3,7:3,8:3})
# nhb['uc'] = nhb['purpose'].replace({1:5,2:4,3:5,4:5,5:5,6:5,7:5,8:5})
# cb = pd.concat([hb, nhb])
# cb['orig_sector'] = cb['triporiggor_b02id'].replace(gor_to_agg)
# cb['dest_sector'] = cb['tripdestgor_b02id'].replace(gor_to_agg)
# cb['ca'] = cb['hh_type'].replace({1:1,3:1,6:1,2:2,4:2,5:2,7:2,8:2})
# cb = cb[cb['mode'] == 6]
# cb = cb[cb['period'] < 5]


for seg in ['gender', 'ns', 'soc']:
    # grouped = cb.groupby(['uc', 'ca', seg, 'orig_sector', 'dest_sector'])['trips'].sum()
    df = pd.read_csv(rf"E:\dia_seg\outputs\final_2\{seg}\convergence_checks.csv", index_col=[0,1,2])
    df.index.names = ['uc','ca', 'area']
    df.columns.name = seg
    df.reset_index(inplace=True)
    df['uc'] = df['uc'].replace({1:'hb_business', 2:'hb_commute', 3:'hb_other', 4:'nhb_business', 5:'nhb_other'})
    df['area'] = df['area'].replace(gor_to_agg)
    df.to_csv(rf"E:\dia_seg\outputs\final_2\{seg}\convergence_checks.csv", index=False)
    print('debugging')
    # df.index.names = ['uc','ca','orig_sector', 'dest_sector']
    # df_reform = df.reset_index()
    # df_reform['uc'] = df_reform['uc'].replace({1:'hb_business', 2:'hb_commute', 3:'hb_other', 4:'nhb_business', 5:'nhb_other'})
    # df_reform['orig_sector'] = df_reform['orig_sector'].replace({1:'North',2:'South',3:'Scotland'})
    # df_reform['dest_sector'] = df_reform['dest_sector'].replace({1:'North',2:'South',3:'Scotland'})
    # df_reform.to_csv(rf"E:\dia_seg\outputs\final_2\{seg}\sectorised_matrices.csv", index=False)
    # grouped.name = 'nts'
    # df = df.drop(['sum', 'Diff'], axis=1).stack()
    # df.index.names = ['uc','ca','orig_sector', 'dest_sector', seg]
    # df.name = 'dist'
    # df = df.reset_index(level=['uc','ca', seg]).rename({1:'North',2:'South',3:'Scotland'})
    # df[['uc','ca', seg]] = df[['uc','ca', seg]].astype(int)
    # df.set_index(['uc','ca', seg], append=True, inplace=True)
    # df = df.join(grouped).reorder_levels(['uc','ca',seg,'orig_sector','dest_sector']).fillna(0.00001).sort_index()
    # df /= df.groupby(level=['uc','ca', seg]).sum()
    # df.reset_index(inplace=True)
    # df['uc'] = df['uc'].replace({1:'hb_business', 2:'hb_commute', 3:'hb_other', 4:'nhb_business', 5:'nhb_other'})
    # df.to_csv(rf"E:\dia_seg\outputs\final_2\{seg}\sectorised_matrices_nts_comp.csv", index=False)