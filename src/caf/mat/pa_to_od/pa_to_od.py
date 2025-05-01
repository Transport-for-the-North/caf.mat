# -*- coding: utf-8 -*-
"""PA to OD conversion functionality."""

##### IMPORTS #####

# Built-Ins
import logging
from pathlib import Path
from typing import Literal, Sequence

import numpy as np

# Third Party
import pandas as pd
from caf.distribute import furness

# Local Imports
# from caf.mat.pa_to_od import matrices, factors

##### CONSTANTS #####

LOG = logging.getLogger(__name__)


##### CLASSES & FUNCTIONS #####

### THESE ARE COMMENTED OUT TO AVOID IMPORT ERRORS ETC. ###
# def tp_hb_pa_to_od(
#     matrices_: matrices.HomePA,
#     fth_factors: factors.FromToHome,
#     missing_tp_factor: factors.MissingTP | None = None,
# ) -> tuple[matrices.HomeOD]:
#     if not matrices_.has_time_periods:
#         raise ValueError("PA matrices don't have time periods so use hb_pa_to_od")
#
#     raise NotImplementedError("WIP!")
#     matrices_ = _combine_time_periods(matrices_, missing_tp_factor)
#
#
# def _combine_time_periods(
#     matrices_: matrices._Base,
#     missing_tp_factor: factors.MissingTP | None = None,
# ) -> matrices._Base:
#     if not matrices_.has_time_periods:
#         raise ValueError
#
#     raise NotImplementedError
#     return _BaseMatrices()
#
#
# def _check_factors(matrix: pd.DataFrame, factors_: pd.DataFrame, description: str = "") -> None:
#     msg = []
#     if not matrix.shape == factors_.shape:
#         msg.append(f"matrix {matrix.shape} and factors {factors_.shape} shapes")
#
#     if not matrix.index.equals(factors_.index):
#         msg.append("indices are not equal")
#
#     if not matrix.columns.equals(factors_.columns):
#         msg.append("columns are not equal")
#
#     if len(msg) > 0:
#         raise ValueError(f"errors between {description}: ".strip() + ", ".join(msg))
#
#
# def hb_pa_to_od(
#     matrices_: matrices._BasePA, fth_factors: factors.FromToHome
# ) -> tuple[matrices.ODMatricesFiles, matrices.ODMatricesFiles]:
#     if matrices_.has_time_periods:
#         raise ValueError("PA matrices split by time but expected 24hr PA")
#
#     fh_matrices = matrices.ODMatricesFiles(matrices_.segmentation, "fh", check_files=False)
#     th_matrices = matrices.ODMatricesFiles(matrices_.segmentation, "th", check_files=False)
#
#     for matrix in matrices_:
#         factors_ = fth_factors.get_from(matrix.segment)
#         _check_factors(
#             matrix, factors_, f"HB PA matrix ({matrix.segment}) and from home factors"
#         )
#
#         fh_matrix = matrix.data * factors_.data
#         fh_matrices.save_matrix(fh_matrix, matrix.segment)
#         del fh_matrix
#
#         factors_ = fth_factors.get_to(matrix.segment)
#         _check_factors(
#             matrix, factors_, f"HB PA matrix ({matrix.segment}) and from home factors"
#         )
#
#         th_matrix = matrix.data * factors_.data
#         th_matrix = th_matrix.T
#
#         th_matrices.save_matrix(th_matrix, matrix.segment)
#
#     return fh_matrices, th_matrices
#
#
# def nhb_pa_to_od(
#     matrices_: matrices._BasePA, tp_factors: factors.TimePeriod
# ) -> matrices.ODMatricesFiles:
#     if matrices_.has_time_periods:
#         raise ValueError("PA matrices are split by time periods, expected 24hr PA matrices")
#
#     od_matrices = matrices.ODMatricesFiles(matrices_.segmentation, "nhb")
#
#     for matrix in matrices_:
#         for factor in tp_factors.get(matrix.segment):
#             _check_factors(matrix.data, factor.data)
#             od_matrix = matrix.data * factor.data
#
#             od_matrices.save_matrix(od_matrix, factor.segment)
#
#     return od_matrices


def balance_fh_th_by_op(
    fh: pd.DataFrame, th: pd.DataFrame, tps: Sequence[int], seed_val: float
):
    """
    Balance fh and th, conserving all but the final time period.

    fh and th are balanced to match each other at the 24hr level. Only the final time
    period is altered in either fh or th, so the others remain identical. In this
    method the returned values for fh and th sum to greater than the input overall.

    Parameters
    ----------
    fh: From home matrices by time period. The index should be a multiindex of o
    and d, and the columns should be the time periods
    th: To home matrices, same format as fh
    tps: Sequence of ints. These must match fh and th column names.
    seed_val: float
        This is a scaler for ho much op will be increased by in balancing.

    Returns
    -------
    fh and th balanced to one another
    """
    ave = (fh.sum(axis=1) + th.sum(axis=1)) / 2
    fh_1_to_3 = fh[tps[:-1]].sum(axis=1)
    th_1_to_3 = th[tps[:-1]].sum(axis=1)
    fh[tps[-1]] = ave - fh_1_to_3
    th[tps[-1]] = ave - th_1_to_3
    th.loc[fh[tps[-1]] < 0, tps[-1]] -= (1 + seed_val) * fh[fh[tps[-1]] < 0][tps[-1]]
    fh.loc[fh[tps[-1]] < 0, tps[-1]] *= -seed_val
    fh.loc[th[tps[-1]] < 0, tps[-1]] -= (1 + seed_val) * th[th[tps[-1]] < 0][tps[-1]]
    th.loc[th[tps[-1]] < 0, tps[-1]] *= -seed_val
    return fh, th


def balance_fh_th_conserve_24hr(fh: pd.DataFrame, th: pd.DataFrame):
    """
    Balance fh and th to each other, conserving the 24hr total.

    This method calculates the average total value for fh and th, then factors both
    to match this across all time periods. This means the sum of the returned
    values for fh and th matches the sum for the inputs.

    Parameters
    ----------
    fh: From home matrices by time period. The index should be a multiindex of o
    and d, and the columns should be the time periods
    th: To home matrices, same format as fh

    Returns
    -------
    fh and th balanced
    """
    fh_24 = fh.sum(axis=1)
    th_24 = th.sum(axis=1)
    # Fix od pairs with zero in fh and non-zero in th and vice versa
    fh.loc[(fh_24 == 0) & (th_24 > 0)] = th.loc[(fh_24 == 0) & (th_24 > 0)] / 2
    th.loc[(fh_24 == 0) & (th_24 > 0)] = th.loc[(fh_24 == 0) & (th_24 > 0)] / 2
    th.loc[(th_24 == 0) & (fh_24 > 0)] = fh.loc[(th_24 == 0) & (fh_24 > 0)] / 2
    fh.loc[(th_24 == 0) & (fh_24 > 0)] = fh.loc[(th_24 == 0) & (fh_24 > 0)] / 2
    # Sum over time periods and infill zeros
    fh_24 = fh.sum(axis=1).replace(0, 1)
    th_24 = th.sum(axis=1).replace(0, 1)
    ave = (fh_24 + th_24) / 2
    # Produce factors to get both to match average
    fh_factor = ave / fh_24
    th_factor = ave / th_24
    # Balance and return
    fh_bal = fh.mul(fh_factor, axis=0)
    th_bal = th.mul(th_factor, axis=0)

    return fh_bal, th_bal


def nhb_props(
    dir: Path,
    name: str,
    tps: Sequence[int],
    occ_factors: pd.Series | None = None,
    tp_factors: dict[int, float] | None = None,
):
    tp_mats = dict()
    nhb_24 = 0
    for tp in tps:
        nhb_path = name.format(tp)
        nhb = pd.read_csv(dir / nhb_path, index_col=0)
        nhb.columns = nhb.columns.astype(int)
        nhb.index = nhb.index.astype(int)
        if occ_factors is not None:
            nhb *= occ_factors.loc["nhb", tp]
        if tp_factors is not None:
            nhb *= tp_factors[tp]
        nhb_24 += nhb
        tp_mats[tp] = nhb
    nhb_tp = pd.concat(tp_mats).stack().unstack(level=0)
    nhb_tp.index.names = ["o", "d"]
    nhb_tp.columns.name = "tp"
    nhb_24 = nhb_24.stack()
    nhb_24.index.names = ["o", "d"]
    nhb_props = nhb_tp.div(nhb_24, axis=0).fillna(1 / len(tps))
    return nhb_props, nhb_24, nhb_tp


def od_to_pa(
    od_dir: Path,
    th_name: str,
    fh_name: str,
    phi_factors: np.array,
    tp_needed: Sequence[int],
    method: Literal["op", "24"],
    occ_factors: pd.Series | None = None,
    tp_factors: dict[int, float] | None = None,
    nhb_name: str | None = None,
):
    """
    Convert od matrices by direction to PA matrices, producing tour proportions.

    Read in fh and th od matrices, balance them and produce tour proportions via
    furnessing. If nhb is provided this will also produce 24hr nhb and return
    nhb tp props.

    Parameters
    ----------
    od_dir: Path
        The directory od matrices are saved in
    th_name: str
        The name of th matrices in od_dir. Must contain curly brackets where
        time period ints will be formatted in.
    fh_name: str
        As for th_name.
    phi_factors: np.array
        Seed values for tour_proportion furnessing. These must be a square array
        of shape (len(tps), len(tps))
    tp_needed: Sequence[int]
        Time periods used here. These will be used to generate th and fh file names.
    method: Literal["op", "24"]
        Which balancing method to use. "op" uses 'balance_fh_th_by_op' and "24hr"
        uses 'balance_fh_th_conserve_24hr'.
    occ_factors: pd.Series | None = None
        Provide if input matrices need converting from pcu to people.
    tp_factors: dict[int:float] | None = None
        Provide if input matrices need converting from peak/average hour to total
        time period.
    nhb_name: str | None = None
        Name of nhb matrices if provided.
    """
    fh_mats = dict()
    th_mats = dict()
    # If provided, read in nhb and produce tp_props, 24hr_matrix and tp matrices
    if nhb_name is not None:
        nhb, nhb_24, nhb_tp = nhb_props(od_dir, nhb_name, tp_needed, occ_factors, tp_factors)
    # Read in fh and th matrices and apply occ/tp factors if needed
    for tp in tp_needed:
        th_path = th_name.format(tp)
        th = pd.read_csv(od_dir / th_path, index_col=0)
        th.columns = th.columns.astype(int)
        th.index = th.index.astype(int)
        fh_path = fh_name.format(tp)
        fh = pd.read_csv(od_dir / fh_path, index_col=0)
        fh.columns = fh.columns.astype(int)
        fh.index = fh.index.astype(int)
        if occ_factors is not None:
            th *= occ_factors.loc["hb_to", tp]
            fh *= occ_factors.loc["hb_fr", tp]
        if tp_factors is not None:
            th *= tp_factors[tp]
            fh *= tp_factors[tp]
        th_mats[tp] = th
        fh_mats[tp] = fh

    # Make sure all matrices have the same OD pairs
    n_rows, n_cols = fh_mats[list(fh_mats.keys())[0]].shape
    for mat_dict in [fh_mats, th_mats]:
        for tp, mat in mat_dict.items():
            if mat.shape != (n_rows, n_cols):
                raise ValueError(
                    "At least one of the loaded matrices does not match the "
                    "others. Expected a matrix of shape (%d, %d), got %s for tp%d."
                    % (n_rows, n_cols, str(mat.shape), tp)
                )
    # Produce compiled original matrices to compare against balanced for adj factors
    fh = pd.concat(fh_mats).stack()
    fh.index.names = ["from", "o", "d"]
    fh = fh.unstack(level="from")
    th = pd.concat(th_mats).stack()
    th.index.names = ["to", "o", "d"]
    th = th.unstack(level="to")
    orig = fh + th
    if nhb_tp is not None:
        orig += nhb_tp

    orig_vals = fh.index.get_level_values("o")
    dest_vals = fh.columns
    # Balance fh and th by chosen method
    if method == "24":
        fh, th = balance_fh_th_conserve_24hr(fh, th)
    elif method == "op":
        fh, th = balance_fh_th_by_op(fh, th, tp_needed, phi_factors[-1, -1])
    # Compare original to balanced to produce adjustment factors
    balanced = fh + th
    if nhb_tp is not None:
        balanced += nhb_tp
    adj = orig / balanced
    pa = fh.copy()
    # Refhape/reformat inputs to tour_prop furnessing
    phi = pd.DataFrame(phi_factors, index=tp_needed, columns=tp_needed).stack()
    seed_index = pd.MultiIndex.from_product(
        [tp_needed, tp_needed, orig_vals, dest_vals], names=["from", "to", "o", "d"]
    )
    # FH and TH are both normalised so that the furness returns props rather than trips
    phi = phi.reindex(seed_index)
    fh_sum = fh.sum(axis=1)
    fh = fh.div(fh_sum.replace(0, 1), axis=0)
    fh.loc[fh_sum == 0] = (0.25, 0.25, 0.25, 0.25)
    fhx = fh.stack().to_xarray()
    th_sum = th.sum(axis=1)
    th = th.div(th_sum.replace(0, 1), axis=0)
    th.loc[th_sum == 0] = (0.25, 0.25, 0.25, 0.25)
    thx = th.stack().to_xarray()

    # Call tour_prop furness
    furness_return_vals, rmse, iter = furness.numpy_ndim_furness(
        phi.to_xarray(), [fhx, thx], len(tp_needed) * len(orig_vals) * len(dest_vals)
    )
    # Put back into dataframe to return
    tour_props = furness_return_vals.to_dataframe(name="trips")
    # Return tour_props, pa matrices by tp, adjustment factors, nhb_24hr and nhb tp props
    return tour_props, pa, adj, nhb_24, nhb_props


def decomp_by_mats(
    synth_fr: pd.DataFrame,
    synth_to: pd.DataFrame,
    post_me: pd.DataFrame,
    synth_nhb: pd.DataFrame | None = None,
):
    """
    Use synthetic matrices by direction to split post-me matrices.

    Parameters
    ----------
    synth_fr: pd.DataFrame
        From home synthetic matrix
    synth_to: pd.DataFrame
        To home synthetic matrix
    post_me: pd.DataFrame
        post_me matrix to be split
    synth_nhb: pd.DataFrame | None = None
        Nhb synthetic matrix if relevant.

    """
    summed = synth_fr + synth_to
    if synth_nhb is not None:
        summed += synth_nhb
        nhb_props = synth_nhb / summed
        nhb_postme = post_me * nhb_props
    else:
        nhb_postme = None
    fr_props = (synth_fr / summed).fillna(0)
    to_props = (synth_to / summed).fillna(0)
    fr_postme = post_me * fr_props
    to_postme = post_me * to_props
    return fr_postme, to_postme, nhb_postme


if __name__ == "__main__":
    # OUTPUTS saved to I:\NorMITs Forecast\
    # Isaac's VM - DP24

    # run script ignore
    hb_to_nhb = {1: 4, 3: 5}
    uc_to_name = {1: "business", 2: "commute", 3: "other"}
    # for uc in [1,2,3]:
    #     for tp in [1,2,3,4]:
    #         synth_fr = pd.read_csv(rf"I:\NorMITs Distribution\voa_gb_2023_uni\mat\vdm\synthetic\noham_m3_ts{tp}_uc{uc}fr.csv.bz2", index_col=[0, 1], names=['o','d','trips']).squeeze().unstack()
    #         synth_to = pd.read_csv(
    #             rf"I:\NorMITs Distribution\voa_gb_2023_uni\mat\vdm\synthetic\noham_m3_ts{tp}_uc{uc}to.csv.bz2", index_col=[0, 1], names=['o','d','trips']).squeeze().unstack()
    #         if uc != 2:
    #             nhb = pd.read_csv(
    #                 rf"I:\NorMITs Distribution\voa_gb_2023_uni\mat\vdm\synthetic\noham_m3_ts{tp}_uc{hb_to_nhb[uc]}.csv.bz2", index_col=[0, 1], names=['o','d','trips']).squeeze().unstack()
    #         else:
    #             nhb=None
    #         postme = pd.read_csv(fr"E:\noham\rebase\bronze\post-me\CSVs\od_m3_{uc_to_name[uc]}_tp{tp}.csv", index_col=0)
    #         postme.columns = postme.columns.astype(int)
    #         fr_post, to_post, nhb_post = decomp_by_mats(synth_fr, synth_to, postme, nhb)
    #         fr_post.to_csv(rf"E:\noham\rebase\bronze\post-me\CSVs\by_direction\noham_m3_ts{tp}_uc{uc}fr.csv")
    #         to_post.to_csv(
    #             rf"E:\noham\rebase\bronze\post-me\CSVs\by_direction\noham_m3_ts{tp}_uc{uc}to.csv")
    #         if nhb_post is not None:
    #             nhb_post.to_csv(
    #                 rf"E:\noham\rebase\bronze\post-me\CSVs\by_direction\noham_m3_ts{tp}_uc{uc}nhb.csv")

    od_dir = Path(r"E:\noham\rebase\bronze\post-me\CSVs\by_direction")
    # phi_factors = pd.read_csv(r"I:\NorMITs Demand\import\phi_factors\v3.0\phi_factors_m3.csv", index_col=0)
    p_uc = {1: 2, 2: 1}
    p_uc.update({i: 3 for i in range(3, 9)})
    # phi_factors = phi_factors.rename(p_uc).set_index(['time_from_home', 'time_to_home'], append=True).groupby(level=['purpose_from_home','time_from_home', 'time_to_home']).mean()
    col_calc, tsx_incl = ["period.fr", "period"], [1, 2, 3, 4]
    fld = Path(r"I:\NTS\outputs\productions\hb\phi_factors")
    phi = pd.read_csv(fld / "tour_proportions_PA_reg.csv")
    phi = phi.set_index("purpose.fr").rename(index=p_uc).reset_index()
    msk = phi[col_calc[0]].isin(tsx_incl) & phi[col_calc[1]].isin(tsx_incl)
    phi = phi.loc[msk].reset_index(drop=True)
    phi[col_calc] = phi[col_calc].astype("category")
    phi = (
        phi.groupby(["mode", "purpose.fr"] + col_calc, observed=False)[["trips.est"]]
        .sum()
        .reset_index()
    )
    phi = phi.rename(
        columns={"purpose.fr": "purpose", "period.fr": "tp_from", "period": "tp_to"}
    )
    phi.set_index(["mode", "purpose", "tp_from", "tp_to"], inplace=True)
    msk = phi.groupby(["mode", "purpose"])["trips.est"].sum()
    phi["phi"] = phi["trips.est"].div(msk).fillna(0)
    phi = phi.loc[3]["phi"]

    occ_factors = (
        pd.read_csv(r"I:\NTS\outputs\occs\vehicle_occupancy_sum.csv", index_col=1)
        .rename(p_uc)
        .reset_index()
    )
    occ_factors = occ_factors.groupby(["purpose", "direction", "period"])[
        ["driver", "total"]
    ].sum()
    occ_factors = occ_factors["total"] / occ_factors["driver"]

    tp_factors = {1: 3, 2: 6, 3: 3, 4: 12}
    for uc in [1, 2, 3]:
        if uc == 2:
            nhb_name = None
        else:
            nhb_name = f"noham_m3_ts{'{}'}_uc{uc}nhb.csv"
        (tour_props, pa, adj_factors, nhb_24, nhb_props) = od_to_pa(
            od_dir,
            f"noham_m3_ts{'{}'}_uc{uc}to.csv",
            f"noham_m3_ts{'{}'}_uc{uc}fr.csv",
            phi.loc[uc].unstack().squeeze().values,
            [1, 2, 3, 4],
            "op",
            occ_factors.loc[uc],
            tp_factors,
            nhb_name,
        )

        tour_props.to_hdf(
            rf"E:\noham\rebase\bronze\post-me\pa\tour_props_uc{uc}.h5", key="data"
        )
        adj_factors.to_hdf(
            rf"E:\noham\rebase\bronze\post-me\pa\adj_factors_uc{uc}.h5", key="data"
        )
        pa.to_hdf(rf"E:\noham\rebase\bronze\post-me\pa\pa_uc{uc}.h5", key="data")
        pa.sum(axis=1).to_csv(rf"E:\noham\rebase\bronze\post-me\24hrpa_{uc}.csv")
        if isinstance(nhb_24, pd.DataFrame):
            nhb_24.to_csv(rf"E:\temp\ntem\inputs\PA\24hrpa_uc{uc}_nhb.csv")
            nhb_props.to_hdf(rf"E:\temp\ntem\inputs\PA\nhb_props_uc{uc}.h5", key="data")
