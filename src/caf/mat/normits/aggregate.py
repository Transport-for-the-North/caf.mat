from caf.mat.matrices import MatricesBase, MatrixFiles, MemoryMatrices, MatrixType
from caf.mat.direction import od_to_pa, factors
from pathlib import Path
import os
import pandas as pd
import caf.base as cb


def agg_to_pa(full_matrices: MatricesBase,
              empty_out: MatricesBase,
              tp_factors: dict[int: float | int],
              phi_factors: factors.PhiFactorsParameters):
    uc_mats= full_matrices.p_to_uc('uc')
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

def to_matrix_format(in_dir: Path,
                     out_dir: Path
                     ):
    for file in os.listdir(in_dir):
        mats = pd.read_hdf(in_dir / file, key='data')
        for seg in mats.index.get_level_values('seg').unique():
            mat = mats.loc[seg]
            mat.to_csv(out_dir / f"{seg}.csv.bz2")

if __name__ == "__main__":
    in_dir = Path(r"D:\NorMITs Demand\ntem_emp_test\canca\outputs")
    out_dir = Path(r"D:\NorMITs Demand\ntem_emp_test\canca\matrices")
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
    to_matrix_format(in_dir, out_dir)
    od_seg = cb.SegmentationInput(enum_segments=['direction_od','m','p','tp','ca'],
                                  naming_order=['direction_od','m','p','tp','ca'],
                                  subsets={'m':6,
                                           'tp':[1,2,3,4]})
    od_seg = cb.Segmentation(od_seg)
    pa_seg = cb.SegmentationInput(enum_segments=['direction','m','p','tp','ca'],
                                  naming_order=['direction','m','p','tp','ca'],
                                  subsets={'m':6,
                                           'tp':[1,2,3,4]})
    pa_seg = cb.Segmentation(pa_seg)
    normits = cb.ZoningSystem.get_zoning('normits')
    od_mats = MatrixFiles(od_seg,
                          normits,
                          MatrixType.OD,
                          out_dir)
    pa_mats = od_mats.new(name='PA',
                          segmentation_=pa_seg,
                          type_=MatrixType.PA)
    
    
    
    pa_mats = agg_to_pa(od_mats,
                        pa_mats,
                        {i: 5 for i in range(1,5)},
                        phi_factors)
