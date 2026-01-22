import pandas as pd
import caf.base as cb
from caf.base import segments
from caf.mat.matrices import MatrixFiles, MatrixType, MemoryMatrices
from caf.mat.direction.od_to_pa import OD2PAParameters, main as od2pa, od_to_pa
from caf.mat.direction import factors
from caf.mat.ufm_converter import UFMConverter
# import caf.tem as ctem
from pathlib import Path
import logging
from caf.toolkit.config_base import BaseConfig

LOG = logging.getLogger(__name__)
NOHAM = cb.ZoningSystem.get_zoning('noham_v3.8')
NOHAM_SECTOR = cb.ZoningSystem.get_zoning('noham_sector')

class PriorAdjustmentConf(BaseConfig):
    saturn_path: Path
    post_me_dir: Path
    post_me_out_dir: Path
    main_out_dir: Path
    synth_24hr_pa_dir: Path
    post_me_od2pa_conf: OD2PAParameters
    prior_od2pa_conf: OD2PAParameters
    # tem_conf: ctem.MainConfig

def run_prior_adjustment(params: PriorAdjustmentConf):
    converter = UFMConverter(params.saturn_path)
    convert_ufms(params.post_me_dir,
                 converter,
                 params.post_me_out_dir)

    od2pa(params.post_me_od2pa_conf)

    tripend_factors(params.synth_24hr_pa_dir,
                    params.post_me_out_dir,
                    params.main_out_dir)

    mts_adj_factors(params.post_me_dir,
                    params.synth_24hr_pa_dir,
                    params.out_dir)
    
def convert_ufms(ufm_dir: Path, converter: UFMConverter, out_dir: Path):
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
            mat_path.rename(out_dir / f"OD_m3_{path.stem}_uc{uc}.csv")
            LOG.debug("Moved %s to %s", mat_path.name, out_dir)

        LOG.info("Done %s / %s (%s)", i, len(matrices), f"{i / len(matrices):.0%}")

def tripend_factors(synth_dir: Path,
                    postme_dir: Path,
                    hb_prod,
                    hb_attr,
                    out_dir: Path):

    seg = cb.Segmentation(cb.SegmentationInput(enum_segments=['m', 'userclass', 'direction'],
                                               naming_order=['m', 'userclass', 'direction'],
                                               subsets={'m':[3]}))
    
    synthetic = MatrixFiles(segmentation_= seg,
                zoning=NOHAM,
                folder=synth_dir,
                type_=MatrixType.PA)
    synthetic_sector_inter = synthetic.remove_intras().translate_zoning(NOHAM_SECTOR).convert_type(MemoryMatrices)
    synthetic_sector  = synthetic.translate_zoning(NOHAM_SECTOR).convert_type(MemoryMatrices)
    synthetic_dvectors_inter = synthetic_sector_inter.to_dvector()
    synthetic_dvecs = synthetic_sector.to_dvector()
    postme = MatrixFiles(segmentation_= seg,
                zoning=NOHAM,
                folder=postme_dir,
                type_=MatrixType.PA)
    postme_sector_inter =  postme.remove_intras().translate_zoning(NOHAM_SECTOR).convert_type(MemoryMatrices)
    postme_dvecs_inter = postme_sector_inter.to_dvector()

    post_prior = {}
    post_prior['P'] = (postme_dvecs_inter['P'] / synthetic_dvectors_inter['P']).add_segments(['p']).filter_segment_value('direction', 1)
    post_prior['A'] = (postme_dvecs_inter['A'] / synthetic_dvectors_inter['A']).add_segments(['p']).filter_segment_value('direction', 1)
    post_prior['P'].aggregate(['p']).save(out_dir / "post_me_adj_p.dvec")
    post_prior['A'].aggregate(['p']).save(out_dir / "post_me_adj_a.dvec")

    hb_prod_uc = hb_prod.add_segments(['userclass']).aggregate(seg)
    hb_attr_uc = hb_attr.add_segments(['userclass']).aggregate(seg)

    prior_te = {}
    prior_te['P'] = (synthetic_dvecs['P'] / hb_prod_uc)
    prior_te['A'] = (synthetic_dvecs['A'] / hb_attr_uc)

    final = {}
    for orig in ['P','A']:
        final[orig] = prior_te[orig] * post_prior[orig]
    return final


def mts_adj_factors(postme_dir: Path,
                    synth_dir: Path,
                    out_dir: Path):
    seg = cb.Segmentation(cb.SegmentationInput(enum_segments=['m', 'userclass', 'direction_od', 'tp'],
                               naming_order=['m', 'tp', 'userclass', 'direction_od'],
                               subsets={'m':[3], 'tp':[1,2,3,4]}))
    synth = MatrixFiles(segmentation_=seg,
                        zoning=NOHAM,
                        type_=synth_dir,
                        folder=MatrixType.OD)
    
    postme = MatrixFiles(segmentation_=seg,
                        zoning=NOHAM,
                        type_=postme_dir,
                        folder=MatrixType.OD)
    
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