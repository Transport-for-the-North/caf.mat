from caf.mat.matrices import MatricesBase, MatrixFiles, MemoryMatrices, MatrixType
from caf.mat.direction import od_to_pa, factors
from pathlib import Path
import os
import pandas as pd
import caf.base as cb
from caf.toolkit import translation
from caf.mat.cube import CUBEMatConverter


def agg_to_pa(full_matrices: MatricesBase,
              empty_out: MatricesBase,
              tp_factors: dict[int: float | int],
              phi_factors: factors.PhiFactorsParameters,
              uc_definitions: pd.Series,
              ie: pd.Series):
    segmentation = full_matrices.segmentation.remove_segment('p').add_segment('userclass')
    uc_mats= MatrixFiles(segmentation,
                            full_matrices.zoning,
                            MatrixType.OD,
                            Path(r"D:\NorMITs Demand\ntem_emp_test\canca\uc"))
    # uc_mats = full_matrices.p_to_uc('uc')
    norms_mats = to_norms_uc(uc_mats,
                             uc_definitions,
                             ie)
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
    
    pa_mats = od_to_pa.od_to_pa(
        uc_mats,
        empty_out,
        '24',
        phi,
        tp_factors=tp_factors,
    )
    return pa_mats

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
        converter.from_csv
        for uc, mat in inner.items():
            out_mat.set_matrix(mat, (cb.segmentation.SegmentationSlice({'norms_uc':ucs_dict[uc],'tp':tp}, naming_order=['norms_uc','tp'])))
    return out_mat

def to_cube(converter: CUBEMatConverter, csv_dir):
    for tp in [1,2,3,4]:
        to_cube = {}
        for uc in cb.segments.SegmentsSuper('norms_uc').get_segment().values.values():
            to_cube[uc] = csv_dir / f"OD_{uc}_ts{tp}.csv"
        converter.from_csv(1325, to_cube, csv_dir / f"PT_demand_tp{tp}.mat")


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

if __name__ == "__main__":
    converter = CUBEMatConverter(Path(r"C:\Program Files\Citilabs\CubeVoyager\VOYAGER.EXE"))
    csv_dir=Path(r"D:\NorMITs Demand\ntem_emp_test\canca\norms_uc")
    # for file in os.listdir(csv_dir):
    #     pd.read_csv(csv_dir / file).to_csv(csv_dir / file, index=False, header=False)
    to_cube(converter, csv_dir=csv_dir)
    in_dir = Path(r"D:\NorMITs Demand\ntem_emp_test\canca\outputs")
    out_dir = Path(r"D:\NorMITs Demand\ntem_emp_test\canca\matrices")
    phi_factors = factors.PhiFactorsParameters(path= r"I:\NTS\outputs\productions\hb\phi_factors\tour_proportions_PA_reg.csv",
                                               segment_columns={"purpose.fr": "p", "mode": "m"},
                                               data_column = "trips.est",
                                               segment_translation={"p": "userclass"},
                                               period_columns=["period.fr", "period"])
    
    # phi = factors.PhiFactors.from_csv(
    #     phi_factors.path,
    #     {i: j.value for i, j in phi_factors.segment_columns.items()},
    #     data_column=phi_factors.data_column,
    #     period_filter=[1,2,3,4],
    #     period_columns=phi_factors.period_columns,
    #     translate_segments={
    #         i.value: j.value
    #         for i, j in phi_factors.segment_translation.items()
    #     },
    #     segment_filters={'m':[6],'tp':[1,2,3,4]},
    # )
    trans = translation.ZoneCorrespondencePath(r"I:\NorMITs NoTEM\Inputs\normits_v3.3_norms_v3.7_trans.csv", from_col_name = 'normits_v3.3_id',to_col_name='norms_v3.7_id', factors_col_name='normits_v3.3_to_norms_v3.7_spatial').read()
    # to_matrix_format(in_dir, out_dir, trans)
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
                          out_dir)
    pa_mats = od_mats.new(name='PA',
                          segmentation_=pa_seg,
                          type_=MatrixType.PA)
    
    norms_definitions = pd.read_csv(r"D:\NorMITs Demand\ntem_emp_test\norms_uc.csv", index_col=[0,1,2,3]).squeeze().rename({'nhb':0,'hb_fr':1,'hb_to':2})
    norms_definitions.index.names = ['ie','userclass','direction_od','ca']
    norms_definitions.index = norms_definitions.index.reorder_levels(['ie','direction_od','ca','userclass'])

    ie = pd.read_csv(r"I:\NorMITs NoTEM\Inputs\normits_v3.3_norms_v3.7_trans.csv").groupby('norms_v3.7_id')['internal'].first().squeeze()
    
    pa_mats = agg_to_pa(od_mats,
                        pa_mats,
                        {i: 1/5 for i in range(1,5)},
                        phi_factors,
                        norms_definitions,
                        ie)
