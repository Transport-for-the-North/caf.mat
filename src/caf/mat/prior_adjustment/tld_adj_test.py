import pandas as pd
import numpy as np
import os
from pathlib import Path
from typing import Union, List, Dict, Set


def dfr_capval(
    dfr: pd.Series, chg_xmax: float = 1e64, chg_xmin: float = None
) -> pd.Series:
    chg_xmin = 1 / chg_xmax if chg_xmin is None else chg_xmin
    out = np.where(dfr > chg_xmax, chg_xmax, np.where(dfr < chg_xmin, chg_xmin, dfr))
    return pd.Series(out, index=dfr.index)


def val_to_list(str_text: Union[str, float, int, List, Dict, Set, tuple, None]) -> List:
    return (
        []
        if str_text is None
        else (
            str_text
            if isinstance(str_text, list)
            else (
                list(str_text)
                if isinstance(str_text, (tuple, Set, Dict))
                else [str_text]
            )
        )
    )


def str_to_value(str_text: str | int | float) -> str | int | float:
    if type(str_text) is str:
        try:
            out_value = int(str_text)
        except ValueError:
            try:
                out_value = float(str_text)
            except ValueError:
                out_value = str_text.strip()
    else:
        out_value = str_text
    return out_value


def join(val: list | tuple | int, dct: dict = None, sep: str = "") -> str | int:
    return (
        sep.join(f"{dct[_k].lower()}" for _k in val_to_list(val))
        if dct is not None
        else str_to_value(sep.join(f"{_k}" for _k in val_to_list(val)))
    )


def read_tld(
    md: int,
    z2z_path: Path,
    s2s_path: Path,
    cons_path: Path,
    tld_fldr: Path,
    cat_dict: dict = {1: "nca", 2: "ca"},
    cat_type: str | None = None,
    tld_cons: str = "moira",
    run_code: bool = False,
) -> dict:

    tfn_purp = [1, 2, 3, 4, 5, 6, 7, 8]
    gor_id = {
        1: "NE",
        2: "NW",
        3: "YH",
        4: "EM",
        5: "WM",
        6: "EoE",
        7: "London",
        8: "SE",
        9: "SW",
        10: "Wales",
        11: "Scotland",
        -8: "NA",
        -9: "DNA",
        -10: "DEAD",
    }
    md, mdx = val_to_list(md), join(md)
    cat_type = "" if cat_type is None else f"_{cat_type}"

    if not run_code:
        csv_dict = {
            pp: {
                di: {
                    ca: tld_fldr
                    / f"NTS_tld_m{mdx}_p{pp}_{di}{'' if ca == 0 else f'_{cax}'}.csv"
                    for ca, cax in cat_dict.items()
                }
                for di in ["hb_fr", "hb_to", "nhb"]
            }
            for pp in tfn_purp
        }
        return csv_dict

    # read trip-length distribution
    csv_ntsz = pd.read_csv(z2z_path)
    dix_list = csv_ntsz["direction"].unique()
    csv_ntsz = csv_ntsz.loc[csv_ntsz["mode"].isin(md)].reset_index(drop=True)
    csv_dict = {pp: {di: {} for di in dix_list} for pp in tfn_purp}
    if len(csv_ntsz) == 0:
        return csv_dict

    nts_orig, seg_incl = "triporiggor_b02id", "ruc_o"
    if md == [6]:  # use moira/wavelength adjustment
        moi_orig, col_grby, col_purp = "orig tlc", [nts_orig, "trav_dist"], []
        tld_ntss = pd.read_csv(
            s2s_path
        )
        tld_moir = pd.read_csv(
            cons_path
        ).rename(columns={moi_orig: nts_orig}, errors="ignore")
        # calculate adj factor, normalise to 100%
        if tld_cons.lower() == "wavelength":
            col = "userclass"
            p2u = {
                1: "business",
                2: "commuting",
                3: "other",
                4: "other",
                5: "other",
                6: "other",
                7: "other",
                8: "other",
            }
            tld_ntss[col] = tld_ntss["purpose"].apply(lambda x: p2u[x])
            csv_ntsz[col] = csv_ntsz["purpose"].apply(lambda x: p2u[x])
            tld_moir[col] = tld_moir[col].str.lower()
            col_purp = [col]

        tld_ntss = (
            tld_ntss.loc[tld_ntss["mode"].isin(md)]
            .groupby(col_grby + col_purp)[["trips"]]
            .sum()
        )
        tld_ntss["trips"] = tld_ntss["trips"].div(
            tld_ntss.groupby([nts_orig] + col_purp)["trips"].transform("sum")
        )
        tld_moir = tld_moir.groupby(col_grby + col_purp)[["trips"]].sum()
        tld_moir["trips"] = tld_moir["trips"].div(
            tld_moir.groupby([nts_orig] + col_purp)["trips"].transform("sum")
        )
        tld_ntss["fact"] = (
            tld_moir["trips"].div(tld_ntss["trips"], axis="index").fillna(1)
        )
        tld_ntss["fact"] = dfr_capval(tld_ntss["fact"], 20)
        # apply adjustment
        csv_ntsz = pd.merge(
            csv_ntsz, tld_ntss["fact"].reset_index(), how="left", on=col_grby + col_purp
        )
        csv_ntsz["trips_adj"] = csv_ntsz["trips"].mul(csv_ntsz["fact"])
        tmp_grby = [nts_orig, seg_incl, "mode", "purpose", "direction"]
        tmp = (
            csv_ntsz.groupby(tmp_grby)["trips"]
            .transform("sum")
            .div(csv_ntsz.groupby(tmp_grby)["trips_adj"].transform("sum"))
        )
        csv_ntsz["trips"] = csv_ntsz["trips_adj"].mul(tmp)
        csv_ntsz = csv_ntsz.drop(
            columns=col_purp + ["fact", "trips_adj"], errors="ignore"
        )

    tld_fldr.mkdir(parents=True, exist_ok=True)

    print("hh_type unique values:", csv_ntsz["hh_type"].unique())
    print("purpose unique values:", csv_ntsz["purpose"].unique())
    print("direction unique values:", csv_ntsz["direction"].unique())

    # output tlds
    gor_list = gor_id
    csv_ntsz[nts_orig] = csv_ntsz[nts_orig].apply(lambda x: gor_list[x])
    for pp in tfn_purp:
        for di in dix_list:
            for ca, cax in cat_dict.items():
                tmp = (
                    (csv_ntsz["purpose"] == pp)
                    & (csv_ntsz["direction"] == di)
                    & (csv_ntsz["hh_type"] == cax)
                )
                tmp = csv_ntsz.loc[tmp].drop(
                    columns=["mode", "purpose", "direction", "hh_type"]
                )
                tmp[";index"] = (
                    tmp[nts_orig].astype(str) + "_" + tmp[seg_incl].astype(str)
                )
                # format required: tld_area, upper_dist, ave_dist, trips
                cax = "" if ca == 0 else f"_{cax}"
                tmp = tmp.set_index([";index", "trav_dist"])[
                    ["ave_dist", "trips"]
                ].sort_index()
                if di == "nhb":
                    curr_pp = pp + 10
                else:
                    curr_pp = pp
                tmp.to_csv(tld_fldr / f"NTS_tld_m{mdx}_p{curr_pp}_{di}{cax}.csv")
                csv_dict[pp][di][ca] = tld_fldr / f"NTS_tld_m{mdx}_p{curr_pp}_{di}{cax}.csv"

    return csv_dict

x = read_tld(
    md=6,
    z2z_path=Path(r"I:\Nhan_Data\trip_length_distribution.csv"),
    s2s_path=Path(r"I:\Nhan_Data\trip_length_distribution_s2s.csv"),
    cons_path=Path(r"I:\Nhan_Data\wavelength_tld.csv"),
    tld_fldr=Path(r"C:\Users\jared.hryszko\Documents\Task3\TLDs"),
    cat_dict={0: "all"},
    cat_type=None,
    tld_cons="wavelength",
    run_code=True,
)