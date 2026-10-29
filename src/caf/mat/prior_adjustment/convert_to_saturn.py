import pandas as pd
import caf.base as cb
from caf.base import segments
from caf.mat.matrices import MatrixFiles, MatrixType, MemoryMatrices
from pathlib import Path
import logging
trans = pd.read_csv(r"I:\Data\Zone Translations\cache\noham3.8_normits3.3\noham3.8_to_normits3.3_spatial.csv")
trans.columns = ['noham_v3.8_id', 'normits_id', 'noham_v3.8_to_normits', 'normits_to_noham_v3.8']
LOG = logging.getLogger(__name__)

NOHAM = cb.ZoningSystem.get_zoning('noham_v3.8')
normits = cb.ZoningSystem.get_zoning('normits')

seg = cb.Segmentation(cb.SegmentationInput(enum_segments=['m', 'p', 'tp','direction_od'],
                            naming_order=['m', 'p', 'tp', 'direction_od'],
                            subsets={'m':[3], 'tp':[2,3]}))

distributed = MatrixFiles(zoning=normits,
                          segmentation_=seg,
                          type_=MatrixType.OD,
                          folder = Path(r"D:\post_me\outputs\fullrun_06_03_v2")).convert_type(MemoryMatrices)

# noham_dist = distributed.translate_zoning(NOHAM, translation=trans).convert_type(MemoryMatrices)



occupancies = pd.read_csv(
        r"I:\NorMITs Forecast\vehicle_occupancies\car_vehicle_occupancies_uc-int.csv",
        index_col=[0, 1, 3, 4],
    ).drop('direction', axis=1)
occupancies.index.names = ['m','userclass','direction_od','tp']
occ_seg = cb.Segmentation(cb.SegmentationInput(enum_segments=['m', 'userclass', 'tp','direction_od'],
                            naming_order=['m', 'userclass', 'tp', 'direction_od'],
                            subsets={'m':[3]}))

occupancies = cb.DVector(occ_seg, occupancies, cut_read=True)

tp_dict = {1: 15, 2: 30, 3: 15}

compiled = distributed.compile_highway(occupancies, tp_dict, 'compiled')

compiled_noham = compiled.translate_zoning(NOHAM, translation=trans)

compiled_noham.save(Path(r"D:\post_me\outputs\final_v2"))
