from caf.mat.matrices import MatricesBase, MatrixFiles, MemoryMatrices
from caf.mat.direction import factors, od_to_pa
import tqdm

def pa_to_od(pa_matrices: MatricesBase,
             tour_proportions: MatricesBase,
             tp_name: str = "{}_tp",
             progress_bar: bool = True):
    if progress_bar:
            iterator = tqdm.tqdm(
                pa_matrices.segmentation.iter_slices(),
                total=len(pa_matrices.segmentation),
                desc=f"Aggregating {pa_matrices.name}",
            )
    else:
        iterator = pa_matrices.segmentation.iter_slices()
    tour_prop_seg = od_to_pa._tour_proportions_segmentation(pa_matrices)
    fh_factor_seg = tour_proportions.segmentation.remove_segment(tp_name.format("to"))
    fh_factors = tour_proportions.aggregate(fh_factor_seg)
    th_factor_seg = tour_proportions.segmentation.remove_segment(tp_name.format("from"))
    th_factors = tour_proportions.aggregate(th_factor_seg)
    th_matrices = pa_matrices * th_factors
    fh_factors = pa_matrices * fh_factors

