from caf.mat.matrices import MatricesBase, MatrixFiles, MemoryMatrices, MatrixType
from caf.mat.direction import od_to_pa, factors, pa_to_od
from pathlib import Path
import os
import pandas as pd
import caf.base as cb
from caf.toolkit import translation
from caf.mat.cube import CUBEMatConverter


def agg_to_pa(full_matrices: MatricesBase,
              empty_out: MatricesBase,
              tp_factors: dict[int: float | int],
              phi_factors: factors.PhiFactorsParameters,):
    segmentation = full_matrices.segmentation.remove_segment('p').add_segment('userclass')
    uc_mats= MatrixFiles(segmentation,
                            full_matrices.zoning,
                            MatrixType.OD,
                            Path(r"D:\NorMITs Demand\ntem_emp_test\canca\uc"))
    uc_mats = full_matrices.p_to_uc('uc')
    phi = factors.PhiFactors.from_csv(
        phi_factors.path,
        {i: j.value for i, j in phi_factors.segment_columns.items()},
        data_column=phi_factors.data_column,
        period_filter=tp_factors.keys(),
        period_columns=phi_factors.period_columns,
        translate_segments={
            i.value: j.value
            for i, j in phi_factors.segment_translation.items()
        },
        segment_filters=uc_mats.segmentation.input.subsets,
    )
    
    od_to_pa.od_to_pa(
        uc_mats,
        empty_out,
        '24',
        phi,
        tp_factors=tp_factors,
        calculate_tour_proportions=True
    )
    return uc_mats

def to_norms_uc(uc_mats: MatricesBase,
                uc_definitions: pd.Series,
                ie: pd.Series):
    out_seg = cb.SegmentationInput(enum_segments=['norms_uc','tp'],
                                   naming_order=['norms_uc','tp'],
                                   subsets={'tp':[1,2,3,4]})
    ucs_dict = {'1fr': 1, '1to': 2, '2fr': 3, '2to': 4, '3': 5, '4': 6, '5fr': 7, '5to': 8, '6fr': 9, '6to': 10, '7fr': 11, '7to': 12, '8fr': 13, '8to': 14, '9': 15, '10': 16, '11': 17, '12': 18, '13': 19, '14': 20, '15': 21, '16': 22, '17': 23, '18': 24, '19': 25}
    out_seg = cb.Segmentation(out_seg)
    out_mat = uc_mats.new(name='norms_uc',
                          segmentation_=out_seg)
    out_dict = {uc: 0 for uc in uc_definitions.unique()}
    out_dict = {i: out_dict for i in [1,2,3,4]}
    internal = ie[ie==1].index
    external = ie[ie==0].index
    for slice_ in uc_mats.segmentation.iter_slices():
        mat = uc_mats.get_matrix(slice_).data
        def_seg = list(uc_definitions.index.names)
        def_seg.remove('ie')
        def_slice = slice_.aggregate(def_seg)
        uc_int = uc_definitions.loc['int'].loc[def_slice.as_tuple()]
        uc_ext = uc_definitions.loc['ext'].loc[def_slice.as_tuple()]
        mat_int = mat.copy()
        mat_int.loc[external] = 0
        mat_int.loc[:, external] = 0
        mat_ext = mat.copy()
        mat_ext.loc[internal,internal] = 0
        out_dict[slice_.get('tp')][uc_int] += mat_int
        out_dict[slice_.get('tp')][uc_ext] += mat_ext
    for tp, inner in out_dict.items():
        for uc, mat in inner.items():
            out_mat.set_matrix(mat, (cb.segmentation.SegmentationSlice({'norms_uc':ucs_dict[uc],'tp':tp}, naming_order=['norms_uc','tp'])))
    return out_mat

def to_cube(converter: CUBEMatConverter, csv_dir):
    for tp in [1,2,3,4]:
        to_cube = {}
        for uc in cb.segments.SegmentsSuper('norms_uc').get_segment().values.values():
            to_cube[uc] = csv_dir / f"OD_{uc}_ts{tp}.csv"
        converter.from_csv(1325, to_cube, csv_dir / f"PT_demand_tp{tp}.mat")

def ext_tp_factors(tp_dir: Path, tot_dir: Path):
    for file in os.listdir(tot_dir):
        if 'ext' in file.lower():
            uc = file.split('.')[0]
            full_df = pd.read_csv(tot_dir / file, index_col=0)
            full_df.columns = full_df.columns.astype(int)
            for tp in [1,2,3,4]:
                tp_df = pd.read_csv(tp_dir / f"OD_{uc}_ts{tp}.csv", index_col=[0,1], names=['o','d','trips']).squeeze().unstack()
                factors = tp_df / full_df
                factors.to_csv(tp_dir / f"{uc}_factor_{tp}.csv")



def to_matrix_format(in_dir: Path,
                     out_dir: Path,
                     trans: translation.ZoneCorrespondence
                     ):
    for file in os.listdir(in_dir):
        mats = pd.read_hdf(in_dir / file, key='data')
        for seg in mats.index.get_level_values('seg').unique():
            mat = mats.loc[seg]
            mat_norms = translation.pandas_matrix_zone_translation(mat, trans)
            mat_norms.to_csv(out_dir / f"{seg}.csv.bz2", float_format="%.20f")

def to_norms_2_electric_boogaloo(pa_dir: Path,
                                 od_dir: Path,
                                 external_zones: list[int],
                                 internal_zones: list[int],
                                 norms_defs: pd.Series):
    uc_norms = cb.segments.SegmentsSuper('norms_uc_pa').get_segment()
    exts_full = {int(i):0 for i in norms_defs.loc['ext'].unique()}
    ext_def = norms_defs.loc['ext']
    int_def = norms_defs.loc['int'].copy()
    int_def = int_def.str.strip('fr').str.strip('to').drop_duplicates()
    exts_tp = {}
    for tp in [1,2,3,4]:
        exts = {int(i):0 for i in norms_defs.loc['ext'].unique()}
        for ca in [1,2]:
            for uc_num, uc_name in {1:'business', 2:'commute', 3:'other'}.items():
                for dir_num, dir_nam in {1:'hb',0:'nhb'}.items():
                    if (dir_nam == 'nhb') and (uc_name == 'commute'):
                        continue
                    df = pd.read_csv(pa_dir / f"PA_m6_ca{ca}_{uc_name}_{dir_nam}.csv.bz2", index_col=0)
                    df.columns = df.columns.astype(int)
                    df.loc[external_zones] = 0
                    df.loc[:,external_zones] = 0
                    segment = int(int_def.loc[dir_num, ca, uc_num])
                    seg_name = uc_norms.values[segment]
                    df.stack().reset_index().to_csv(pa_dir / f"{seg_name}.csv", index=False, header=False)
                for dir_num, dir_nam in {0:'nhb',1:'fr',2:'to'}.items():
                    if (uc_name == 'commute') & (dir_nam == 'nhb'):
                        continue
                    df = pd.read_csv(od_dir / f"OD_{dir_nam}_m6_ts{tp}_ca{ca}_{uc_name}.csv.bz2", index_col=0)
                    df.columns = df.columns.astype(int)
                    df.loc[internal_zones, internal_zones] = 0
                    exts[int(ext_def.loc[dir_num, ca, uc_num])] += df
                    exts_full[int(ext_def.loc[dir_num, ca, uc_num])] += df
        exts_tp[tp] = exts
    for tp, inner in exts_tp.items():
        for name, df in inner.items():
            (df / exts_full[name]).fillna(0).stack().reset_index().to_csv(pa_dir / f"{uc_norms.values[name]}_TS{tp}.csv", index=False, header=False)
    for name, df in exts_full.items():
        df.stack().reset_index().to_csv(pa_dir / f"{uc_norms.values[name]}.csv", index=False, header=False)
    

if __name__ == "__main__":
    ca = {1:'NCA',2:'CA'}
    converter = CUBEMatConverter(Path(r"C:\Program Files\Citilabs\CubeVoyager\VOYAGER.EXE"))
    base_dir = Path(r"D:\rail\distribution\canca")
    in_dir = base_dir / "outputs"
    out_dir = base_dir / "final"
    od_dir = base_dir / "matrices"
    phi_factors = factors.PhiFactorsParameters(path= r"I:\NTS\outputs\productions\hb\phi_factors\tour_proportions_PA_reg.csv",
                                               segment_columns={"purpose.fr": "p", "mode": "m"},
                                               data_column = "trips.est",
                                               segment_translation={"p": "userclass"},
                                               period_columns=["period.fr", "period"])
    
    phi = factors.PhiFactors.from_csv(
        phi_factors.path,
        {i: j.value for i, j in phi_factors.segment_columns.items()},
        data_column=phi_factors.data_column,
        period_filter=[1,2,3,4],
        period_columns=phi_factors.period_columns,
        translate_segments={
            i.value: j.value
            for i, j in phi_factors.segment_translation.items()
        },
        segment_filters={'m':[6],'tp':[1,2,3,4]},
    )
    trans = translation.ZoneCorrespondencePath(r"I:\NorMITs NoTEM\Inputs\normits_v3.3_norms_v3.7_trans.csv", from_col_name = 'normits_v3.3_id',to_col_name='norms_v3.7_id', factors_col_name='normits_v3.3_to_norms_v3.7_spatial').read()
    # to_matrix_format(in_dir, od_dir, trans)
    od_seg = cb.SegmentationInput(enum_segments=['direction_od','m','p','tp','ca'],
                                  naming_order=['direction_od','m','p','tp','ca'],
                                  subsets={'m':[6],
                                           'tp':[1,2,3,4]})
    od_seg = cb.Segmentation(od_seg)
    pa_seg = cb.SegmentationInput(enum_segments=['direction','m','userclass','ca'],
                                  naming_order=['m', 'ca', 'userclass', 'direction'],
                                  subsets={'m':[6]})
    pa_seg = cb.Segmentation(pa_seg)
    nortms = cb.ZoningSystem.get_zoning('nortms_3.7')
    od_mats = MatrixFiles(od_seg,
                          nortms,
                          MatrixType.OD,
                          od_dir)
    pa_mats = od_mats.new(name='PA',
                          segmentation_=pa_seg,
                          type_=MatrixType.PA)
    
    norms_definitions = pd.read_csv(r"D:\NorMITs Demand\ntem_emp_test\norms_uc.csv", index_col=[0,1,2,3]).squeeze().rename({'nhb':0,'hb_fr':1,'hb_to':2})
    norms_definitions.index.names = ['ie','userclass','direction_od','ca']
    norms_definitions.index = norms_definitions.index.reorder_levels(['ie','direction_od','ca','userclass'])

    ie = pd.read_csv(r"I:\NorMITs NoTEM\Inputs\normits_v3.3_norms_v3.7_trans.csv").groupby('norms_v3.7_id')['internal'].first().squeeze()
    
    # uc_mats = agg_to_pa(od_mats,
    #                     pa_mats,
    #                     {i: 1/5 for i in range(1,5)},
    #                     phi_factors
    #                     )
    # to_norms_2_electric_boogaloo(base_dir / 'pa', base_dir / 'uc', ie[ie==0].index,ie[ie==1].index, norms_definitions)

    # ext_tp_factors(csv_dir, pa_dir)
    # Convert tour props to mat
    tp_fr = cb.Segment(name='from_tp', values={1:'AM',2:'IP',3:'PM',4:'OP'}, alias='ts')
    tp_to = cb.Segment(name='to_tp', values={1:'AM',2:'IP',3:'PM',4:'OP'}, alias='ts')
    prop_seg = cb.SegmentationInput(enum_segments=['m','ca','userclass'],
                                    custom_segments=[tp_fr,tp_to],
                                  naming_order=['m', 'ca', 'userclass', 'from_tp', 'to_tp'],
                                  subsets={'m':[6]})
    prop_seg = cb.Segmentation(prop_seg)
    props = MatrixFiles(prop_seg, nortms, MatrixType.OD, Path(r"D:\NorMITs Demand\ntem_emp_test\canca\tour_proportions"))
    ucs = {1:'EB',2:'C',3:'O'}
    for uc_num, uc_nam in ucs.items():
        subbed = props.subset({'userclass':[uc_num]})
        renamer = {}
        for ca_num, ca_nam in {2:'CA',1:'NCA'}.items():
            for fh in [1,2,3,4]:
                for th in [1,2,3,4]:
                    slice_ = cb.segmentation.SegmentationSlice({'m':6, 'userclass':uc_num, 'ca':ca_num, 'from_tp':fh, 'to_tp':th},
                                                               naming_order=subbed.segmentation.naming_order)
                    renamer[slice_] = f"{uc_nam}{ca_nam}_F{fh}_T{th}"

        subbed.to_mat(out_dir / f"SplitFactors_DS{uc_num}.mat", Path(r"C:\Program Files\Citilabs\CubeVoyager\VOYAGER.EXE"), name_conversion=renamer)
    ##############################################################
    # OD_PROP
    od_prop_seg = cb.SegmentationInput(enum_segments=['m','ca','userclass','tp', 'direction_od'],
                                  naming_order=['direction_od', 'm', 'tp', 'ca', 'userclass'],
                                  subsets={'m':[6],
                                           'direction_od': [0],
                                           'tp': [1,2,3,4]})
    od_prop_seg = cb.Segmentation(od_prop_seg)
    od_prop = MatrixFiles(od_prop_seg, nortms, MatrixType.OD, base_dir / 'od_return_factors').convert_type(MemoryMatrices).to_internal()
    for tp_num, tp_name in {1:"AM",2:"IP",3:"PM",4:"OP"}.items():
        subbed = od_prop.subset({'tp':[tp_num]})
        renamer={}
        for uc_num, uc_name in {1:'EB',3:'O'}.items():
            for ca_num, ca_name in {2:'CA',1:'NCA'}.items():
                slice_ = cb.segmentation.SegmentationSlice({'m':6, 'userclass':uc_num, 'ca':ca_num, 'tp':tp_num, 'direction_od':0},
                                                               naming_order=subbed.segmentation.naming_order)
                renamer[slice_] = f"NHB{uc_name}{ca_name}"
        subbed.to_mat(out_dir / f"OD_Prop_{tp_name}_PT.mat", Path(r"C:\Program Files\Citilabs\CubeVoyager\VOYAGER.EXE"), name_conversion=renamer)
    
    norms_uc = cb.segments.SegmentsSuper('norms_uc_pa').get_segment()
    to_cube = {}
    for uc in norms_uc.values.values():
        to_cube[uc] = base_dir / 'PA' / f"{uc}.csv"
    converter.from_csv(1325, to_cube, out_dir / "PT_24hr_Demand.mat")
    for tp in [1,2,3,4]:
        csv_dict = {}
        for uc_1, uc_2 in {'EB':'EMP', 'HBW':'COM', 'O':'OTH'}.items():
            csv_dict[f"{uc_2}_NCA"] = base_dir / 'PA' / f"{uc_1}NCA_Ext_TS{tp}.csv"
            csv_dict[f"{uc_2}_FH"] = base_dir / 'PA' / f"{uc_1}CA_Ext_FM_TS{tp}.csv"
            csv_dict[f"{uc_2}_TH"] = base_dir / 'PA' / f"{uc_1}CA_Ext_TO_TS{tp}.csv"
        converter.from_csv(1325, csv_dict, out_dir / f"Time_of_Day_Factors_Zonal_{tp}.mat")
    # full_pa = MatrixFiles(norms_seg, nortms, MatrixType.PA, base_dir / 'PA')
    # full_pa.to_mat(out_dir / "PT_24hr_Demand.mat", Path(r"C:\Program Files\Citilabs\CubeVoyager\VOYAGER.EXE"))
    
    # od_seg_to = cb.Segmentation(od_seg_to)
    # hb_pa = pa_mats.convert_type(MemoryMatrices).filter_segment_value('direction', 1, keep_segment=False)
    # th_mats, fh_mats = pa_to_od.pa_to_od(hb_pa, props)
    # fr_orig = MatrixFiles(od_seg_fr, nortms, MatrixType.OD, uc_mats.folder).convert_type(MemoryMatrices).aggregate(fh_mats.segmentation)
    # to_orig = MatrixFiles(od_seg_to, nortms, MatrixType.OD, uc_mats.folder).convert_type(MemoryMatrices).aggregate(th_mats.segmentation)
    # adjustments_to = to_orig / (th_mats * 5)
    # adjustments_fr = fr_orig / (fh_mats * 5)
    # adjustments_to.save(base_dir / "adj", name='to_adj', format='cube')
    # to_mat = {}
    # adjustments_fr.save(base_dir / "adj", name='fr_adj', format='cube')
    
    # for file in os.listdir(csv_dir):
    #     pd.read_csv(csv_dir / file).to_csv(csv_dir / file, index=False, header=False)
    # to_cube(converter, csv_dir=csv_dir)