from caf.distribute import furness
import caf.base as cb
import pandas as pd
import numpy as np
import scipy
prod = cb.DVector.load(r"E:\dia_seg\tripends\prod_s.dvec").data
attr = cb.DVector.load(r"E:\dia_seg\tripends\attr_s_balanced.dvec").data

for uc in [1,2,3,4,5]:
    for ca in [1,2]:
        mat = pd.read_hdf(rf"E:\dia_seg\target_mats\uc{uc}_ca{ca}.hdf")
        print(np.var(mat.values / mat.values.sum()))
