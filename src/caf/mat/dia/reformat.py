import pandas as pd
from pathlib import Path
from caf.toolkit import translation

uc_dict = {1:"hb_pa_business",
           2:"hb_pa_commute",
           3:"hb_pa_other",
           4:"nhb_pa_business",
           5:"nhb_pa_other"}


def do_it(mats_path, segs: list[int], norms_trans, sec_trans, sum_mat_dir, out_dir, name, dev_zones: list[int]):
    total_mats = {}
    total_comps = {}
    total_cons = {}
    for uc in [1,2,3,4,5]:
        ca_mats = {}
        ca_comps = {}
        ca_cons = {}
        for ca in [1,2]:
            sum_mat = pd.read_hdf(sum_mat_dir / f"uc{uc}_ca{ca}.hdf")
            sum_mat_sec = translation.pandas_matrix_zone_translation(sum_mat, sec_trans, 'normits_id', 'sector_id', 'normits_to_sector').stack()
            with pd.HDFStore(mats_path / name / f"matrices_uc{uc}_ca{ca}.hdf", 'r') as file:
                mats = file['data']
                ca_comps[ca] = file['checks']
                ca_cons[ca] = file['convergence']
            inner_mats = {}
            inner_mats['sum'] = sum_mat_sec
            for seg in segs:
                mat = mats.loc[seg]
                norms_mat = translation.pandas_matrix_zone_translation(mat, norms_trans, 'normits_v3.3_id', 'norms_v3.7_id', 'normits_v3.3_to_norms_v3.7_spatial')
                norms_mat[dev_zones] = 0
                new_rows = pd.DataFrame(data=0, index=dev_zones, columns=norms_mat.columns)
                norms_mat = pd.concat([norms_mat, new_rows])
                norms_mat.sort_index().sort_index(axis=1).to_csv(out_dir / f"{uc_dict[uc]}_m6_{name}{seg}_ca{ca}.csv.bz2")
                sec_mat = translation.pandas_matrix_zone_translation(mat, sec_trans, 'normits_id', 'sector_id', 'normits_to_sector').stack()
                sec_mat.name = seg
                inner_mats[seg] = sec_mat
            ca_mats[ca] = pd.concat(inner_mats, axis=1)
        total_mats[uc] = pd.concat(ca_mats)
        total_comps[uc] = pd.concat(ca_comps)
        total_cons[uc] = pd.concat(ca_cons)
    # pd.concat(total_mats).to_csv(out_dir / "sectorised_matrices.csv")
    # pd.concat(total_comps).to_csv(out_dir / "tld_checks.csv")
    # pd.concat(total_cons).to_csv(out_dir / "convergence_checks.csv")
        



            
            
if __name__ == "__main__":
    home_dir = Path(r"E:\dia_seg\outputs")
    out_dir = Path(r'E:\dia_seg\outputs\final_2\gender')
    sum_mats = Path(r"E:\dia_seg\target_mats")
    normits_norms = pd.read_csv(r"E:\noham\rebase\normits_v3.3_norms_v3.7_trans.csv")
    normits_sec = pd.read_csv(r"E:\shapefiles\3_sector_to_normits_spatial.csv")
    do_it(home_dir, segs=[1,2,3], norms_trans=normits_norms, sec_trans=normits_sec, out_dir=out_dir, sum_mat_dir=sum_mats, name='gender', dev_zones=[1181, 1326])