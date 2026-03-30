from caf.distribute import furness
from pathlib import Path
import pandas as pd

mat_dir = Path(r"I:\NorMITs Distribution\voa_gb_2023_uni_i1\iter3a_rail_only\surface rail")
out_dir = Path(r"E:\dia_seg\target_mats")

# aggregate p to uc

for ca in ['ca','nca']:
    uc1 = 0
    uc2 = 0
    uc3 = 0
    uc4 = 0
    uc5 = 0
    for direction in ["hb_fr", "nhb"]:
        for p in [1, 2, 3, 4, 5, 6, 7, 8]:
            df_24 = 0
            for tp in [1,2,3,4]:
                df = pd.read_hdf(mat_dir / f"p{p}" / "ca_split" / f"gm_m6_p{p}_{direction}_ts{tp}_{ca}_estTrip.hdf")
                df = df.set_index(['o','d']).squeeze().unstack(level='d')
                df_24 += df
            if direction == 'hb_fr':
                if p == 1:
                    uc2 += df_24
                elif p == 2:
                    uc1 += df_24
                else:
                    uc3 += df_24
            else:
                if p == 2:
                    uc4 += df_24
                else:
                    uc5 += df_24
    uc1.to_hdf(out_dir / f"uc1_{ca}.hdf", key='data')
    uc2.to_hdf(out_dir / f"uc2_{ca}.hdf", key='data')
    uc3.to_hdf(out_dir / f"uc3_{ca}.hdf", key='data')
    uc4.to_hdf(out_dir / f"uc4_{ca}.hdf", key='data')
    uc5.to_hdf(out_dir / f"uc5_{ca}.hdf", key='data')
