# -*- coding: utf-8 -*-
"""PA to OD conversion functionality."""

##### IMPORTS #####

# Built-Ins
import logging

# Third Party
import pandas as pd

# Local Imports
from caf.mat.pa_to_od import matrices, factors

##### CONSTANTS #####

LOG = logging.getLogger(__name__)


##### CLASSES & FUNCTIONS #####


def tp_hb_pa_to_od(
    matrices_: matrices.HomePA,
    fth_factors: factors.FromToHome,
    missing_tp_factor: factors.MissingTP | None = None,
) -> tuple[matrices.HomeOD]:
    if not matrices_.has_time_periods:
        raise ValueError("PA matrices don't have time periods so use hb_pa_to_od")

    raise NotImplementedError("WIP!")
    matrices_ = _combine_time_periods(matrices_, missing_tp_factor)


def _combine_time_periods(
    matrices_: matrices._Base,
    missing_tp_factor: factors.MissingTP | None = None,
) -> matrices._Base:
    if not matrices_.has_time_periods:
        raise ValueError

    raise NotImplementedError
    return _BaseMatrices()


def _check_factors(matrix: pd.DataFrame, factors_: pd.DataFrame, description: str = "") -> None:
    msg = []
    if not matrix.shape == factors_.shape:
        msg.append(f"matrix {matrix.shape} and factors {factors_.shape} shapes")

    if not matrix.index.equals(factors_.index):
        msg.append("indices are not equal")

    if not matrix.columns.equals(factors_.columns):
        msg.append("columns are not equal")

    if len(msg) > 0:
        raise ValueError(f"errors between {description}: ".strip() + ", ".join(msg))


def hb_pa_to_od(
    matrices_: matrices._BasePA, fth_factors: factors.FromToHome
) -> tuple[matrices.ODMatricesFiles, matrices.ODMatricesFiles]:
    if matrices_.has_time_periods:
        raise ValueError("PA matrices split by time but expected 24hr PA")

    fh_matrices = matrices.ODMatricesFiles(matrices_.segmentation, "fh", check_files=False)
    th_matrices = matrices.ODMatricesFiles(matrices_.segmentation, "th", check_files=False)

    for matrix in matrices_:
        factors_ = fth_factors.get_from(matrix.segment)
        _check_factors(
            matrix, factors_, f"HB PA matrix ({matrix.segment}) and from home factors"
        )

        fh_matrix = matrix.data * factors_.data
        fh_matrices.save_matrix(fh_matrix, matrix.segment)
        del fh_matrix

        factors_ = fth_factors.get_to(matrix.segment)
        _check_factors(
            matrix, factors_, f"HB PA matrix ({matrix.segment}) and from home factors"
        )

        th_matrix = matrix.data * factors_.data
        th_matrix = th_matrix.T

        th_matrices.save_matrix(th_matrix, matrix.segment)

    return fh_matrices, th_matrices


def nhb_pa_to_od(
    matrices_: matrices._BasePA, tp_factors: factors.TimePeriod
) -> matrices.ODMatricesFiles:
    if matrices_.has_time_periods:
        raise ValueError("PA matrices are split by time periods, expected 24hr PA matrices")

    od_matrices = matrices.ODMatricesFiles(matrices_.segmentation, "nhb")

    for matrix in matrices_:
        for factor in tp_factors.get(matrix.segment):
            _check_factors(matrix.data, factor.data)
            od_matrix = matrix.data * factor.data

            od_matrices.save_matrix(od_matrix, factor.segment)

    return od_matrices
