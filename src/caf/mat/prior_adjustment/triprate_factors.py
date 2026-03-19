"""
Prior adjustment module for trip rate factor calculations.
This module handles the calculation and adjustment of trip rate factors using prior
and post-ME matrices. 
It converts UFM files, calculates trip-end factors, and applies MTS adjustments, if selected.
"""

"""
OTHER COMMENTS:
mts_adj_factors has hardcoded naming order and subsets for segments assuming consistency between the post-ME and synthetic.
Is this guaranteed? This was not the case in distribute.py, so flexibility was added to the config file.
"""



import pandas as pd
import caf.base as cb
from caf.mat.matrices import MatrixFiles, MatrixType
from caf.mat.direction.od_to_pa import OD2PAParameters, disagg_and_convert
from caf.mat.ufm import UFMConverter
# import caf.tem as ctem
from pathlib import Path
import logging
from caf.toolkit.config_base import BaseConfig

LOG = logging.getLogger(__name__)
NOHAM = cb.ZoningSystem.get_zoning('noham_v3.8')
NORMITS = cb.ZoningSystem.get_zoning('normits')
NOHAM_SECTOR = cb.ZoningSystem.get_zoning('noham_sector')
normits_noham_sector = pd.read_csv(r"I:\Data\Zone Translations\cache\noham_sector_normits\noham_sector_to_normits_spatial.csv")
normits_noham_sector.columns = ['noham_sector_id','normits_id','noham_sector_to_normits','normits_to_noham_sector']
normits_noham_sector['noham_sector_id'] = normits_noham_sector['noham_sector_id'].replace(NOHAM_SECTOR.name_to_id)

class PriorAdjustmentConf(BaseConfig):
    """
    Configuration class for prior adjustment operations.
    Attributes:
        saturn_path: Path to Saturn exe for UFM conversion
        run_conversion: Flag to enable UFM conversion
        run_mts_adj: Flag to enable MTS adjustment
        post_me_dir: Input directory for post-ME UFMs
        post_me_out_dir: Directory for post-ME converted UFMs
        prior_dir: Input directory for prior UFMs
        prior_out_dir: Directory for prior converted UFMs
        main_out_dir: Main output directory for results
        post_me_od2pa_conf: Configuration parameters for post-ME OD to PA conversion
        prior_od2pa_conf: Configuration parameters for prior OD to PA conversion
        hb_prod_fr_path: HB Productions From-Home filepath
        hb_attr_fr_path: HB Attractions From-Home filepath
        hb_prod_to_path: HB Productions To-Home filepath
        hb_attr_to_path: HB Attractions To-Home filepath
        nhb_prod_path: NHB Productions filepath
        nhb_attr_path: NHB Attractions filepath
    """

    saturn_path: Path
    run_conversion: bool = False
    run_mts_adj: bool = False
    post_me_dir: Path
    post_me_out_dir: Path
    prior_dir: Path
    prior_out_dir: Path
    main_out_dir: Path
    post_me_od2pa_conf: OD2PAParameters
    prior_od2pa_conf: OD2PAParameters
    hb_prod_fr_path: Path
    hb_attr_fr_path: Path
    hb_prod_to_path: Path
    hb_attr_to_path: Path
    nhb_prod_path:Path
    nhb_attr_path:Path
    # tem_conf: ctem.MainConfig

def run_prior_adjustment(params: PriorAdjustmentConf):
    """
    Run prior adjustment process for trip rate factors.
    This function starts the prior adjustment workflow, which includes:
    1. Optional: conversion of UFMs for both post-ME and prior matrices
    2. Disaggregation and conversion of OD to PA matrices for both post-ME and prior matrices
    3. Loading HB and NHB productions/attractions
    4. Calculating trip-end factors based on the loaded data
    5. Optional: adjustment of factors based on MTS data
    Parameters
    ----------
    params : PriorAdjustmentConf
        Configuration object explained above
    Returns
    -------
    None. It runs further functions that output directly to params.main_out_dir
    """


    if params.run_conversion:
        converter = UFMConverter(params.saturn_path)
        convert_ufms(params.post_me_dir,
                    converter,
                    params.post_me_out_dir,
                    "postme_2023")
        
        convert_ufms(params.prior_dir,
                    converter,
                    params.prior_out_dir,
                    "prior_2023")

    postme_disag = disagg_and_convert(params.post_me_od2pa_conf)
    prior_disag = disagg_and_convert(params.prior_od2pa_conf)

    hb_prod_fr = cb.DVector.load(params.hb_prod_fr_path)
    hb_attr_fr = cb.DVector.load(params.hb_attr_fr_path)
    hb_prod_to = cb.DVector.load(params.hb_prod_to_path)
    hb_attr_to = cb.DVector.load(params.hb_attr_to_path)
    nhb_prod = cb.DVector.load(params.nhb_prod_path)
    nhb_attr = cb.DVector.load(params.nhb_attr_path)


    tripend_factors(prior_disag,
                    postme_disag,
                    hb_prod_fr,
                    hb_attr_fr,
                    hb_prod_to,
                    hb_attr_to,
                    nhb_prod,
                    nhb_attr,
                    params.main_out_dir)
    
    if params.run_mts_adj:
        mts_adj_factors(params.post_me_dir,
                        params.prior_out_dir,
                        params.main_out_dir)
    
def convert_ufms(ufm_dir: Path, converter: UFMConverter, out_dir: Path, rename: str):
    """
    Convert UFM files to square CSVs and organise outputs.
    This function takes a directory of UFM files, converts each to square CSV format,
    and saves the resulting files to the specified output directory.
    Parameters
    ----------
    ufm_dir : Path
        Directory containing UFM files to be converted.
    converter : UFMConverter
        Converter object used to perform the UFM to CSV conversion.
    out_dir : Path
        Directory where the converted CSV files will be saved.
    rename : str
        String used to rename the output files.
    Returns
    -------
    None. The function calls on SATURN to convert UFMs and outputs information to LOG.
    """
    matrices = list(ufm_dir.glob("*.ufm"))
    for i, path in enumerate(matrices):
        stacked, unstacked = converter.ufm_to_square_csvs(
            path, path.stem, decimal_places=8, overwrite=False
        )
        stacked.unlink()
        LOG.info("Deleted stacked matrix: %s", stacked.name)

        LOG.info(
            "Moving unstacked (%s) files to %s",
            len(unstacked),
            out_dir,
        )
        for mat_path in unstacked:
            uc = mat_path.name.split('-')[-1].split('.')[0].split('_')[-1]
            mat_path.replace(out_dir / f"OD_{rename}_m3_{path.stem}_uc{uc}.csv")
            LOG.debug("Moved %s to %s", mat_path.name, out_dir)

        LOG.info("Done %s / %s (%s)", i, len(matrices), f"{i / len(matrices):.0%}")

def tripend_factors(synthetic,
                    postme,
                    hb_prod_fr,
                    hb_attr_fr,
                    hb_prod_to,
                    hb_attr_to,
                    nhb_prod,
                    nhb_attr,
                    out_dir: Path):
    """
    Calculate trip-end factors for both prior and post-ME matrices.
    This calculates post and prior factors by dividing the interzonal post-ME OD by the
    interzonal synthetic OD, separately for productions and attractions.
    Then calculates factors by dividing the zonal synthetic OD by the HB/NHB 
    productions/attractions for each direction. 
    Finally, it multiplies the factors by the post/prior interzonal factors to get 
    adjusted factors for HB/NHB productions/attractions.
    Parameters
    ----------
    synthetic : MatrixFiles
        MatrixFiles object containing the synthetic OD matrices.
    postme : MatrixFiles
        MatrixFiles object containing the post-ME OD matrices.
    hb_prod_fr : DVector
        DVector containing home-based-from productions.
    hb_attr_fr : DVector
        DVector containing home-based-from attractions.
    hb_prod_to : DVector
        DVector containing home-based-to productions.
    hb_attr_to : DVector
        DVector containing home-based-to attractions.
    nhb_prod : DVector
        DVector containing non-home-based productions.
    nhb_attr : DVector
        DVector containing non-home-based attractions.
    out_dir : Path
        Directory where the calculated factors will be saved as DVector files.
    Returns
    -------
    None. The function outputs the calculated factors as DVector files to the specified output directory.
    """
    synthetic_sector_inter = synthetic.remove_intras().translate_zoning(NOHAM_SECTOR)
    synthetic_sector  = synthetic.translate_zoning(NOHAM_SECTOR)
    synthetic_dvectors_inter = synthetic_sector_inter.to_dvector()
    synthetic_dvecs = synthetic_sector.to_dvector()

    postme_sector_inter =  postme.remove_intras().translate_zoning(NOHAM_SECTOR)
    postme_sector = postme.translate_zoning(NOHAM_SECTOR)
    postme_dvecs_inter = postme_sector_inter.to_dvector()
    postme_dvecs = postme_sector.to_dvector()

    synthetic_dvectors_inter['O'].save(out_dir / "prior_inter_p.dvec")
    synthetic_dvectors_inter['D'].save(out_dir / "prior_inter_a.dvec")
    synthetic_dvecs['O'].save(out_dir / "prior_p.dvec")
    synthetic_dvecs['D'].save(out_dir / "prior_a.dvec")
    postme_dvecs_inter['O'].save(out_dir / "postme_inter_p.dvec")
    postme_dvecs_inter['D'].save(out_dir / "postme_inter_a.dvec")
    postme_dvecs['O'].save(out_dir / "postme_p.dvec")
    postme_dvecs['D'].save(out_dir / "postme_a.dvec")

    post_prior = {}
    post_prior['O'] = (postme_dvecs_inter['O'] / synthetic_dvectors_inter['O'])
    post_prior['D'] = (postme_dvecs_inter['D'] / synthetic_dvectors_inter['D'])
    post_prior['O'].save(out_dir / "f_pm_prior_p.dvec")
    post_prior['D'].save(out_dir / "f_pm_prior_a.dvec")

    hb_prod_fr_uc = hb_prod_fr.aggregate_comp_zones(NORMITS).translate_zoning(NOHAM_SECTOR, trans_vector=normits_noham_sector).add_segments(['userclass']).aggregate(['m', 'userclass','tp'])
    hb_attr_fr_uc = hb_attr_fr.aggregate_comp_zones(NORMITS).translate_zoning(NOHAM_SECTOR, trans_vector=normits_noham_sector).add_segments(['userclass']).aggregate(['m', 'userclass','tp'])
    hb_prod_to_uc = hb_prod_to.aggregate_comp_zones(NORMITS).translate_zoning(NOHAM_SECTOR, trans_vector=normits_noham_sector).add_segments(['userclass']).aggregate(['m', 'userclass','tp'])
    hb_attr_to_uc = hb_attr_to.aggregate_comp_zones(NORMITS).translate_zoning(NOHAM_SECTOR, trans_vector=normits_noham_sector).add_segments(['userclass']).aggregate(['m', 'userclass','tp'])
    nhb_prod_uc = nhb_prod.aggregate_comp_zones(NORMITS).translate_zoning(NOHAM_SECTOR, trans_vector=normits_noham_sector).add_segments(['userclass']).aggregate(['m', 'userclass','tp'])
    nhb_attr_uc = nhb_attr.aggregate_comp_zones(NORMITS).translate_zoning(NOHAM_SECTOR, trans_vector=normits_noham_sector).add_segments(['userclass']).aggregate(['m', 'userclass','tp'])

    prior_te_fr = {}
    prior_te_to = {}
    prior_te_nhb = {}
    prior_te_fr['O'] = (synthetic_dvecs['O'].filter_segment_value('direction_od', 1) / hb_prod_fr_uc)
    prior_te_fr['D'] = (synthetic_dvecs['D'].filter_segment_value('direction_od', 1) / hb_attr_fr_uc)
    prior_te_to['O'] = (synthetic_dvecs['O'].filter_segment_value('direction_od', 2) / hb_prod_to_uc)
    prior_te_to['D'] = (synthetic_dvecs['D'].filter_segment_value('direction_od', 2) / hb_attr_to_uc)
    prior_te_nhb['O'] = (synthetic_dvecs['O'].filter_segment_value('direction_od', 0) / nhb_prod_uc)
    prior_te_nhb['D'] = (synthetic_dvecs['D'].filter_segment_value('direction_od', 0) / nhb_attr_uc)
    prior_te_fr['O'].save(out_dir / "f_prior_te_fr_p.dvec")
    prior_te_fr['D'].save(out_dir / "f_prior_te_fr_a.dvec")
    prior_te_to['O'].save(out_dir / "f_prior_te_to_p.dvec")
    prior_te_to['D'].save(out_dir / "f_prior_te_to_a.dvec")
    prior_te_nhb['O'].save(out_dir / "f_prior_te_nhb_p.dvec")
    prior_te_nhb['D'].save(out_dir / "f_prior_te_nhb_a.dvec")


    for orig in ['O','D']:
        factor_fr = prior_te_fr[orig] * post_prior[orig]
        factor_fr.add_segments(['p']).remove_segment('userclass').save(out_dir / f"post_me_adj_fr_{orig}.dvec")
        factor_to = prior_te_to[orig] * post_prior[orig]
        factor_to.add_segments(['p']).remove_segment('userclass').save(out_dir / f"post_me_adj_to_{orig}.dvec")
        factor_nhb = prior_te_nhb[orig] * post_prior[orig]
        factor_nhb.add_segments(['p']).remove_segment('userclass').save(out_dir / f"post_me_adj_nhb_{orig}.dvec")


def mts_adj_factors(postme_dir: Path,
                    synth_dir: Path,
                    out_dir: Path):
    """
    Currently not implemented, but leaving here for future development
    """
    seg = cb.Segmentation(cb.SegmentationInput(enum_segments=['m', 'userclass', 'direction_od', 'tp'],
                               naming_order=['m', 'tp', 'userclass', 'direction_od'],
                               subsets={'m':[3], 'tp':[1,2,3]}))
    synth = MatrixFiles(segmentation_=seg,
                        zoning=NOHAM,
                        type_=MatrixType.OD,
                        folder=synth_dir)
    
    postme = MatrixFiles(segmentation_=seg,
                        zoning=NOHAM,
                        type_=MatrixType.OD,
                        folder=postme_dir)
    
    synth_sector = synth.translate_zoning(NOHAM_SECTOR)
    postme_sector = postme.translate_zoning(NOHAM_SECTOR)

    synth_dvecs = synth_sector.to_dvector()
    post_dvecs = postme_sector.to_dvector()
    od_pa = {'O':'P', 'D':'A'}
    for dir in ['O','D']:
        synth_dvec = synth_dvecs[dir].filter_segment_value('direction_od', 1, keep_filtered=True)
        postme_dvec = post_dvecs[dir].filter_segment_value('direction_od', 1, keep_filtered=True)
        post_norm = postme_dvec / postme_dvec.aggregate(['m', 'userclass', 'direction_od'])
        synth_norm = synth_dvec / synth_dvec.aggregate(['m', 'userclass', 'direction_od'])
        adj = post_norm / synth_norm
        adj = adj.add_segments(['p']).aggregate(['m','p','tp'])
        adj.save(out_dir / f"mts_adj_{od_pa[dir]}")

if __name__ == "__main__":
    post_me_conf = OD2PAParameters.load_yaml(Path(r"C:\Users\YanZhu\caf_mat\postme_adj\post_me_conf.yml"))
    prior_conf = OD2PAParameters.load_yaml(Path(r"C:\Users\YanZhu\caf_mat\postme_adj\synthetic_conf.yml"))

    conf = PriorAdjustmentConf(
        saturn_path=Path(r"C:\Program Files (x86)\Atkins\SATURN\XEXES 11.6.03E MC N4"),
        run_conversion=False,
        run_mts_adj=False,
        post_me_dir=Path(r"C:\Users\YanZhu\caf_mat\postme_adj\post_me_ufms"),
        post_me_out_dir=Path(r"C:\Users\YanZhu\caf_mat\postme_adj\post_me"),
        prior_dir=Path(r"C:\Users\YanZhu\caf_mat\postme_adj\prior_ufms"),
        prior_out_dir=Path(r"C:\Users\YanZhu\caf_mat\postme_adj\prior"),
        main_out_dir=Path(r"C:\Users\YanZhu\caf_mat\postme_adj\output_7"),
        post_me_od2pa_conf=post_me_conf,
        prior_od2pa_conf=prior_conf,
        hb_prod_fr_path=Path(r"C:\Users\YanZhu\caf_tem\Outputs\post_me_adj_7\Core\hb_productions\hb_normits_tem_segmented_fr_2023.dvec"),
        hb_attr_fr_path=Path(r"C:\Users\YanZhu\caf_tem\Outputs\post_me_adj_7\Core\hb_attractions\hb_normits_tem_segmented_fr_2023.dvec"),
        hb_prod_to_path=Path(r"C:\Users\YanZhu\caf_tem\Outputs\post_me_adj_7\Core\hb_productions\hb_normits_tem_segmented_to_2023.dvec"),
        hb_attr_to_path=Path(r"C:\Users\YanZhu\caf_tem\Outputs\post_me_adj_7\Core\hb_attractions\hb_normits_tem_segmented_to_2023.dvec"),
        nhb_prod_path=Path(r"C:\Users\YanZhu\caf_tem\Outputs\post_me_adj_7\Core\nhb_productions\nhb_normits_tem_segmented_fr_2023.dvec"),
        nhb_attr_path=Path(r"C:\Users\YanZhu\caf_tem\Outputs\post_me_adj_7\Core\nhb_attractions\nhb_normits_tem_segmented_fr_2023.dvec"),

    )

    run_prior_adjustment(conf)