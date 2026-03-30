import caf.base as cb
from pathlib import Path
import pandas as pd

def matrix_to_tripends(segmentation: cb.Segmentation,
                       zone_system: cb.ZoningSystem,
                       matrix_path: Path):
    """Convert a set of matrices to a DVector of trip ends

    Outputs DVector at uc, ca segmentation"""
    
    p_data = {}
    a_data = {}
    for slice in segmentation.iter_slices():
        mat = pd.read_hdf(matrix_path / f"{segmentation.naming_order[0]}{slice[segmentation.naming_order[0]]}_{segmentation.naming_order[1]}{slice[segmentation.naming_order[1]]}.hdf")
        p_data[(slice[segmentation.naming_order[0]], slice[segmentation.naming_order[1]])] = mat.sum(axis=1)
        a_data[(slice[segmentation.naming_order[0]], slice[segmentation.naming_order[1]])] = mat.sum()
    p_df = pd.DataFrame.from_dict(p_data, orient='index')
    p_df.index.names = ['uc', 'ca']
    a_df = pd.DataFrame.from_dict(a_data, orient='index')
    a_df.index.names = ['uc','ca']
    a_dvec = cb.DVector(import_data=a_df, segmentation=segmentation, zoning_system=zone_system)
    p_dvec = cb.DVector(import_data=p_df, segmentation=segmentation, zoning_system=zone_system)
    return p_dvec, a_dvec

def split_tripends(hb: cb.DVector,
                   nhb: cb.DVector,
                   main:cb.DVector,
                   target_segment: str):
    translated = hb.translate_segment('p','uc').translate_segment('hh_type','ca').concat(nhb.translate_segment('p','uc').translate_segment('hh_type','ca'))
    filtered = translated.filter_segment_value('m', 6).filter_segment_value('tp', [1,2,3,4]).aggregate(['uc','ca',target_segment])
    # splitter = filtered.remove_zoni / filtered.aggregate(main.segmentation)
    split = main.split_by_other(filtered, filtered.zoning_system)
    # split = main.expand_to_other(filtered, match_props=True)
    return split

if __name__ == "__main__":
    out_dir = Path(r'E:\dia_seg\tripends')
    out_dir.mkdir(exist_ok=True)
    zoning = cb.ZoningSystem.get_zoning('normits')
    input_dir = Path(r'E:\dia_seg\target_mats')
    seg_input = cb.SegmentationInput(
        enum_segments=['ca','uc'],
        naming_order=['uc','ca']
    )
    segmentation = cb.Segmentation(seg_input)
    p, a = matrix_to_tripends(segmentation, zoning, input_dir)
    hb_prod_g = cb.DVector.load(r"E:\tem\outputs\include_nssec\Core\hb_productions\hb_normits_tem_segmented_2023_dvec.h5").aggregate_comp_zones(zoning)
    nhb_prod_g = cb.DVector.load(r"E:\tem\outputs\include_nssec\Core\nhb_productions\nhb_normits_tem_segmented_2023_dvec.h5").aggregate_comp_zones(zoning)
    hb_attr_g = cb.DVector.load(r"E:\tem\outputs\include_nssec\Core\hb_attractions\hb_normits_tem_segmented_2023_dvec.h5").aggregate_comp_zones(zoning)
    nhb_attr_g = cb.DVector.load(r"E:\tem\outputs\include_nssec\Core\nhb_attractions\nhb_normits_tem_segmented_2023_dvec.h5").aggregate_comp_zones(zoning)
    prod_g_split = split_tripends(hb_prod_g, nhb_prod_g, p, 'ns_sec')
    attr_g_split = split_tripends(hb_attr_g, nhb_attr_g, a, 'ns_sec')
    targets = [cb.data_structures.IpfTarget(data=prod_g_split.remove_zoning()),
               cb.data_structures.IpfTarget(data=a)]
    attr_g_balanced, rmse = attr_g_split.ipf(targets)
    prod_g_split.save(out_dir / "prod_n.dvec")
    attr_g_balanced.save(out_dir / "attr_n_balanced.dvec")

    
    
    
    p.save(input_dir / 'p_agg.dvec')
    a.save(input_dir / 'a_agg.dvec')