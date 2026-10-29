from caf.mat.matrices import MatricesBase, MatrixFiles, MemoryMatrices
from caf.mat.direction import factors, od_to_pa
import caf.base as cb

def pa_to_od(pa_matrices: MatricesBase,
             tour_proportions: MatricesBase,
             tp_name: str = "{}_tp",
             progress_bar: bool = True):
    fh_factor_seg = tour_proportions.segmentation.remove_segment(tp_name.format("to"))
    fh_factors = tour_proportions.aggregate(fh_factor_seg)
    th_factor_seg = tour_proportions.segmentation.remove_segment(tp_name.format("from"))
    th_factors = tour_proportions.aggregate(th_factor_seg)
    th_matrices = pa_matrices * th_factors
    fh_matrices = pa_matrices * fh_factors
    return th_matrices, fh_matrices

if __name__ == "__main__":
    pa_seg = cb.SegmentationInput(enum_segments=['direction','m','userclass','ca'],
                                  naming_order=['m', 'ca', 'userclass', 'direction'],
                                  subsets={'m':[6]})
    pa_mats = 
