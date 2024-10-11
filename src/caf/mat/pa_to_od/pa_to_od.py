# -*- coding: utf-8 -*-
"""PA to OD conversion functionality."""

##### IMPORTS #####

# Built-Ins
import logging

# Local Imports
from caf.mat.pa_to_od.factors import (
    FromToHomeFactors,
    MissingTPScaling,
    TimePeriodFactors,
)
from caf.mat.pa_to_od.matrices import OutputODMatrices, PAMatrices

##### CONSTANTS #####

LOG = logging.getLogger(__name__)


##### CLASSES & FUNCTIONS #####


def pa_to_od(
    matrices: PAMatrices,
    fth_factors: FromToHomeFactors,
    tp_factors: TimePeriodFactors,
) -> tuple[OutputODMatrices, OutputODMatrices]:
    if matrices.has_time_periods:
        raise ValueError("use tp_pa_to_od instead for non-24hr matrices")

    hb_outputs = hb_pa_to_od(matrices, fth_factors)
    nhb_outputs = nhb_pa_to_od(matrices, tp_factors)

    return hb_outputs, nhb_outputs


def tp_pa_to_od(
    matrices: PAMatrices,
    fth_factors: FromToHomeFactors,
    tp_factors: TimePeriodFactors,
    missing_tp_factor: MissingTPScaling | None = None,
) -> tuple[OutputODMatrices, OutputODMatrices]:
    if matrices.has_time_periods:
        matrices = _combine_time_periods(matrices, missing_tp_factor)

    raise NotImplementedError


def _combine_time_periods(
    matrices: PAMatrices,
    missing_tp_factor: MissingTPScaling | None = None,
) -> PAMatrices:
    if not matrices.has_time_periods:
        raise ValueError

    raise NotImplementedError
    return PAMatrices()


def hb_pa_to_od(matrices: PAMatrices, fth_factors: FromToHomeFactors) -> OutputODMatrices:
    if not matrices.home_based_only:
        raise ValueError

    raise NotImplementedError

    return OutputODMatrices()


def nhb_pa_to_od(matrices: PAMatrices, tp_factors: TimePeriodFactors) -> OutputODMatrices:
    if not matrices.non_home_based_only:
        raise ValueError

    raise NotImplementedError

    return OutputODMatrices()
