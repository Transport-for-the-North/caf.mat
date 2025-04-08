# -*- coding: utf-8 -*-
"""PA to OD conversion functionality."""

##### IMPORTS #####

# Built-Ins
import logging
from pathlib import Path
from typing import Literal

# Third Party
import pandas as pd
from caf.distribute import furness

# Local Imports
# from caf.mat.pa_to_od import matrices, factors

##### CONSTANTS #####

LOG = logging.getLogger(__name__)


##### CLASSES & FUNCTIONS #####


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


def balance_fh_th_by_op(fh, th, tps, seed_val):
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


def balance_fh_th_conserve_24hr(fh, th):
    fh_24 = fh.sum(axis=1)  # .replace(0,1)
    th_24 = th.sum(axis=1)  # .replace(0,1)
    fh.loc[(fh_24 == 0) & (th_24 > 0)] = th.loc[(fh_24 == 0) & (th_24 > 0)] / 2
    th.loc[(fh_24 == 0) & (th_24 > 0)] = th.loc[(fh_24 == 0) & (th_24 > 0)] / 2
    th.loc[(th_24 == 0) & (fh_24 > 0)] = fh.loc[(th_24 == 0) & (fh_24 > 0)] / 2
    fh.loc[(th_24 == 0) & (fh_24 > 0)] = fh.loc[(th_24 == 0) & (fh_24 > 0)] / 2
    fh_24 = fh.sum(axis=1).replace(0, 1)
    th_24 = th.sum(axis=1).replace(0, 1)
    ave = (fh_24 + th_24) / 2
    fh_factor = ave / fh_24
    th_factor = ave / th_24
    fh_bal = fh.mul(fh_factor, axis=0)
    th_bal = th.mul(th_factor, axis=0)

    return fh_bal, th_bal


def od_to_pa(
    od_dir: Path,
    th_name: str,
    fh_name: str,
    phi_factors,
    tp_needed: list,
    method: Literal["op", "24"],
    occ_factors: pd.Series | None = None,
    tp_factors: dict[int:float] | None = None,
    nhb: str | None = None,
):
    fh_mats = dict()
    th_mats = dict()
    nhb_mats = dict()
    nhb_24 = 0
    nhb_props = None
    nhb_tp = None
    for tp in tp_needed:
        th_path = th_name.format(tp)
        th_mats[tp] = pd.read_csv(od_dir / th_path, index_col=0)
        th_mats[tp].columns = th_mats[tp].columns.astype(int)
        th_mats[tp].index = th_mats[tp].index.astype(int)
        fh_path = fh_name.format(tp)
        fh_mats[tp] = pd.read_csv(od_dir / fh_path, index_col=0)
        fh_mats[tp].columns = fh_mats[tp].columns.astype(int)
        fh_mats[tp].index = fh_mats[tp].index.astype(int)
        if nhb is not None:
            nhb_name = nhb.format(tp)
            nhb_mats[tp] = pd.read_csv(od_dir / nhb_name, index_col=0)
            nhb_mats[tp].columns = nhb_mats[tp].columns.astype(int)
            nhb_mats[tp].index = nhb_mats[tp].index.astype(int)
        if occ_factors is not None:
            th_mats[tp] *= occ_factors.loc["hb_to", tp]
            fh_mats[tp] *= occ_factors.loc["hb_fr", tp]
            if nhb is not None:
                nhb_mats[tp] *= occ_factors.loc["nhb", tp]
        if tp_factors is not None:
            th_mats[tp] *= tp_factors[tp]
            fh_mats[tp] *= tp_factors[tp]
            if nhb is not None:
                nhb_mats[tp] *= tp_factors[tp]
        if nhb is not None:
            nhb_24 += nhb_mats[tp]
    if nhb:
        nhb_tp = pd.concat(nhb_mats).stack().unstack(level=0)
        nhb_tp.index.names = ["o", "d"]
        nhb_tp.columns.name = "tp"
        nhb_24 = nhb_24.stack()
        nhb_24.index.names = ["o", "d"]
        nhb_props = nhb_tp.div(nhb_24, axis=0)

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

    orig_vals = fh_mats[tp_needed[0]].index
    dest_vals = fh_mats[tp_needed[0]].columns

    fh = pd.concat(fh_mats).stack()
    fh.index.names = ["from", "o", "d"]
    fh = fh.unstack(level="from")
    th = pd.concat(th_mats).stack()
    th.index.names = ["to", "o", "d"]
    th = th.unstack(level="to")
    orig = fh + th
    if nhb_tp is not None:
        orig += nhb_tp

    if method == "24":
        fh, th = balance_fh_th_conserve_24hr(fh, th)
    elif method == "op":
        fh, th = balance_fh_th_by_op(fh, th, tp_needed, 0)

    balanced = fh + th
    if nhb_tp is not None:
        balanced += nhb_tp
    adj = orig / balanced
    pa = fh.copy()

    phi = pd.DataFrame(phi_factors, index=tp_needed, columns=tp_needed).stack()
    seed_index = pd.MultiIndex.from_product(
        [tp_needed, tp_needed, orig_vals, dest_vals], names=["from", "to", "o", "d"]
    )
    phi = phi.reindex(seed_index)
    fh_sum = fh.sum(axis=1)
    fh = fh.div(fh_sum.replace(0, 1), axis=0)
    fh.loc[fh_sum == 0] = (0.25, 0.25, 0.25, 0.25)
    fhx = fh.stack().to_xarray()
    th_sum = th.sum(axis=1)
    th = th.div(th_sum.replace(0, 1), axis=0)
    th.loc[th_sum == 0] = (0.25, 0.25, 0.25, 0.25)
    thx = th.stack().to_xarray()

    # ## CALL INNER FUNCTION ## #
    furness_return_vals, rmse, iter = furness.numpy_ndim_furness(
        phi.to_xarray(), [fhx, thx], len(tp_needed) * len(orig_vals) * len(dest_vals)
    )

    tour_props = furness_return_vals.to_dataframe(name="trips")

    return tour_props, pa, adj, nhb_24, nhb_props


def decomp_by_mats(
    synth_fr: pd.DataFrame,
    synth_to: pd.DataFrame,
    post_me: pd.DataFrame,
    synth_nhb: pd.DataFrame | None = None,
):
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
    hb_to_nhb = {1: 4, 3: 5}
    uc_to_name = {1: "business", 2: "commute", 3: "other"}
    for uc in [1,2,3]:
        for tp in [1,2,3,4]:
            synth_fr = pd.read_csv(rf"I:\NorMITs Distribution\voa_gb_2023_uni\mat\vdm\synthetic\noham_m3_ts{tp}_uc{uc}fr.csv.bz2", index_col=[0, 1], names=['o','d','trips']).squeeze().unstack()
            synth_to = pd.read_csv(
                rf"I:\NorMITs Distribution\voa_gb_2023_uni\mat\vdm\synthetic\noham_m3_ts{tp}_uc{uc}to.csv.bz2", index_col=[0, 1], names=['o','d','trips']).squeeze().unstack()
            if uc != 2:
                nhb = pd.read_csv(
                    rf"I:\NorMITs Distribution\voa_gb_2023_uni\mat\vdm\synthetic\noham_m3_ts{tp}_uc{hb_to_nhb[uc]}.csv.bz2", index_col=[0, 1], names=['o','d','trips']).squeeze().unstack()
            else:
                nhb=None
            postme = pd.read_csv(fr"E:\noham\rebase\bronze\post-me\CSVs\od_m3_{uc_to_name[uc]}_tp{tp}.csv", index_col=0)
            postme.columns = postme.columns.astype(int)
            fr_post, to_post, nhb_post = decomp_by_mats(synth_fr, synth_to, postme, nhb)
            fr_post.to_csv(rf"E:\noham\rebase\bronze\post-me\CSVs\by_direction\noham_m3_ts{tp}_uc{uc}fr.csv")
            to_post.to_csv(
                rf"E:\noham\rebase\bronze\post-me\CSVs\by_direction\noham_m3_ts{tp}_uc{uc}to.csv")
            if nhb_post is not None:
                nhb_post.to_csv(
                    rf"E:\noham\rebase\bronze\post-me\CSVs\by_direction\noham_m3_ts{tp}_uc{uc}nhb.csv")

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
    for uc in [1]:
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
            "24",
            occ_factors.loc[uc],
            tp_factors,
            nhb_name,
        )

        tour_props.to_hdf(
            rf"E:\noham\rebase\bronze\post-me\pa\24_hr\tour_props_uc{uc}.h5", key="data"
        )
        adj_factors.to_hdf(
            rf"E:\noham\rebase\bronze\post-me\pa\24_hr\adj_factors_uc{uc}.h5", key="data"
        )
        pa.to_hdf(rf"E:\noham\rebase\bronze\post-me\pa\24_hr\pa_uc{uc}.h5", key="data")
        pa.sum(axis=1).to_csv(rf"E:\noham\rebase\bronze\post-me\24hrpa_{uc}.csv")
        if isinstance(nhb_24, pd.DataFrame):
            nhb_24.to_csv(rf"E:\temp\ntem\inputs\PA\24hrpa_uc{uc}_nhb.csv")
            nhb_props.to_hdf(rf"E:\temp\ntem\inputs\PA\nhb_props_uc{uc}.h5", key="data")
