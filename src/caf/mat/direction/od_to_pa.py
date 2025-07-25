# -*- coding: utf-8 -*-
"""OD to PA conversion functionality."""

##### IMPORTS #####

# Built-Ins
import itertools
import logging
import pathlib
import warnings
from typing import Literal, Sequence, TypeVar

# Third Party
import caf.base as base
import caf.toolkit as ctk
import pandas as pd
import pydantic
import xarray
from caf.base import segmentation, segments
from caf.distribute import furness

# Local Imports
from caf.mat import matrices
from caf.mat.direction import factors

##### CONSTANTS #####

LOG = logging.getLogger(__name__)

_DIRECTION_VALUES = {"nhb": 0, "from": 1, "to": 2, "hb": 1}


##### CLASSES & FUNCTIONS #####

Matrices = TypeVar("Matrices", bound=matrices.MatricesBase)


def balance_fh_th_by_op(
    fh: pd.DataFrame, th: pd.DataFrame, tps: Sequence[int], seed_val: float
) -> tuple[pd.DataFrame, pd.DataFrame]:
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


def balance_fh_th_conserve_24hr(
    fh: pd.DataFrame, th: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
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


def disaggregate_postme(
    postme_folder: pathlib.Path,
    synth_folder: pathlib.Path,
    zone_system: base.ZoningSystem,
    postme_segmentation: base.Segmentation,
    disaggregation_segments: list[str],
) -> matrices.MatrixFiles:
    LOG.info(
        "Disaggregating PostME matrices (%s) using synthetic (%s)",
        postme_folder.name,
        synth_folder.name,
    )

    postme = matrices.MatrixFiles(
        postme_segmentation,
        zone_system,
        matrices.MatrixType.OD,
        postme_folder,
    )

    synth_seg_input = base.SegmentationInput(
        enum_segments=postme_segmentation.input.enum_segments + disaggregation_segments,
        naming_order=postme_segmentation.input.naming_order + disaggregation_segments,
        subsets=postme_segmentation.input.subsets,
    )
    synthetic = matrices.MatrixFiles(
        base.Segmentation(synth_seg_input),
        zone_system,
        matrices.MatrixType.OD,
        synth_folder,
        filename_template="noham_{slice_name}",
    )

    output = postme.disaggregate(synthetic)
    LOG.info("Written disaggregated matrices to: %s", output.folder)
    return output


def _matrix_multiply(
    slice_: segmentation.SegmentationSlice,
    matrices_: Matrices,
    occ_factors: base.DVector | None = None,
    tp_factors: dict[int, int] | None = None,
) -> pd.DataFrame:
    data = matrices_.get_matrix(slice_).data

    if occ_factors is not None:
        data *= occ_factors.get_slice(slice_, allow_closest=True)

    if tp_factors is not None:
        tp = slice_.get(segments.SegmentsSuper.TIMEPERIOD.value)
        if tp is None:
            raise KeyError("time period missing from slice")

        data *= tp_factors[tp]

    return data


def _validate_od_input_outputs(input_: Matrices, output: Matrices) -> None:
    """Validate inputs and outptus matrices segmentation for OD to PA.

    Checks input is OD with time periods and output is
    PA without time periods.
    """
    valid_segments = [
        ("input", input_, matrices.MatrixType.OD),
        ("output", output, matrices.MatrixType.PA),
    ]

    for name, mat, type_ in valid_segments:
        if mat.type != type_:
            raise ValueError(f"{name} matrices should be {type_.name} not {mat.type}")
        if not mat.has_type_segment:
            raise ValueError(f"{name} matrices doesn't contain the correct direction segment")

    if not input_.has_time_periods:
        raise ValueError("inputs matrices don't have time periods")
    if output.has_time_periods:
        raise ValueError("output PA matrices shouldn't have time periods")

    if input_.zoning != output.zoning:
        raise ValueError(
            "input and output zone systems are different:"
            f" {input_.zoning.name} != {output.zoning.name}"
        )


def nhb_proportions(
    input_: Matrices,
    params: dict[str, int],
    output: Matrices,
    output_proportions: Matrices,
    *,
    occ_factors: base.DVector | None = None,
    tp_factors: dict[int, int] | None = None,
):
    _validate_od_input_outputs(input_, output)

    od_params = params | {segments.SegmentsSuper.DIRECTION_OD.value: _DIRECTION_VALUES["nhb"]}

    tp_matrices: dict[segmentation.SegmentationSlice, pd.DataFrame] = {}
    for slice_ in input_.segmentation.iter_slices(od_params):
        data = _matrix_multiply(slice_, input_, occ_factors, tp_factors)
        tp_matrices[slice_] = data

    nhb_24hr = sum(tp_matrices.values())
    output.set_matrix(
        nhb_24hr,
        segmentation.SegmentationSlice(
            params | {segments.SegmentsSuper.DIRECTION.value: _DIRECTION_VALUES["nhb"]},
            naming_order=output.segmentation.naming_order,
        ),
    )

    for slice_, data in tp_matrices.items():
        output_proportions.set_matrix(data / nhb_24hr, slice_)


def _get_time_matrices(
    input_: Matrices,
    params: dict[str, int],
    occ_factors: base.DVector | None = None,
    tp_factors: dict[int, int] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame | None]:
    unstacked_matrices: dict[str, list[pd.Series]] = {"nhb": [], "from": [], "to": []}
    for tp_slice in input_.segmentation.iter_slices(params):
        tp_params = tp_slice.data

        for nm, list_ in unstacked_matrices.items():
            tp_params[input_.type.direction_segment.name] = _DIRECTION_VALUES[nm]

            try:
                square_matrix = _matrix_multiply(
                    segmentation.SegmentationSlice(tp_params, tp_slice.naming_order),
                    input_,
                    occ_factors,
                    tp_factors,
                )
            except ValueError:
                # NHB doesn't need to be included in the segmentation
                if nm == "nhb":
                    continue
                raise

            # Stack to long format so time period can be concatenated
            long_matrix: pd.Series = square_matrix.stack()
            long_matrix.name = tp_slice.get(segments.SegmentsSuper.TIMEPERIOD.value)
            list_.append(long_matrix)

    from_home = pd.concat(unstacked_matrices["from"])
    to_home = pd.concat(unstacked_matrices["to"])
    if len(unstacked_matrices["nhb"]) > 0:
        nhb = pd.concat(unstacked_matrices["nhb"])
    else:
        nhb = None
    return from_home, to_home, nhb


def _balance_fh_th(
    from_home: pd.DataFrame,
    to_home: pd.DataFrame,
    method: Literal["24", "op"],
    nhb: pd.DataFrame | None = None,
    *,
    time_periods: Sequence[int] | None = None,
    seed_value: float | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    original = from_home + to_home
    if nhb is not None:
        original += nhb

    if method == "24":
        from_home, to_home = balance_fh_th_conserve_24hr(from_home, to_home)

    elif method == "op":
        if time_periods is None or seed_value is None:
            raise ValueError(
                "time periods and seed value are required for 'op' balancing method"
            )
        from_home, to_home = balance_fh_th_by_op(
            from_home,
            to_home,
            time_periods,
            seed_val=seed_value,
        )

    balanced = from_home + to_home
    if nhb is not None:
        balanced += nhb

    adjustment = original / balanced

    return from_home, to_home, adjustment


def _set_matrices_by_time_period(
    matrices_: Matrices,
    data: pd.DataFrame,
    params: dict[str, int],
    column_segment: str = segments.SegmentsSuper.TIMEPERIOD.value,
):
    for column in data.columns:
        slice_ = segmentation.SegmentationSlice(
            params | {column_segment: int(column)},
            naming_order=matrices_.segmentation.naming_order,
        )
        matrices_.set_matrix(data[column], slice_)


def _normalise_to_xarray(data: pd.DataFrame) -> xarray.DataArray:
    total = data.sum(axis=1)
    data = data.div(total.replace(0, 1), axis=0)
    data.loc[total == 0] = (0.25, 0.25, 0.25, 0.25)
    return data.stack().to_xarray()


def _od_adjustment_segmentation(input_: Matrices) -> segmentation.Segmentation:
    """Create segmentation for OD adjustments which doesn't include NHB / HB."""
    direction = input_.type.direction_segment.name
    enum_segments = [
        i for i in input_.segmentation.input.enum_segments if i.value != direction
    ]
    names = [i for i in input_.segmentation.input.naming_order if i != direction]
    custom = [i for i in input_.segmentation.input.custom_segments if i.name != direction]

    config = segmentation.SegmentationInput(
        enum_segments=enum_segments, naming_order=names, custom_segments=custom
    )
    return segmentation.Segmentation(config)


def _tour_proportions_segmentation(input_: Matrices) -> segmentation.Segmentation:
    """Create segmentation for OD adjustments which doesn't include NHB / HB."""
    tp_name = segments.SegmentsSuper.TIMEPERIOD.value
    remove = (input_.type.direction_segment.name, tp_name)

    new_enum = [segments.SegmentsSuper.DIRECTION]
    enum_segments = [
        i for i in input_.segmentation.input.enum_segments if i.value not in remove
    ] + new_enum
    names = [i for i in input_.segmentation.input.naming_order if i not in remove]

    from_segment = segments.SegmentsSuper.TIMEPERIOD.get_segment().model_copy()
    from_segment.name = f"from_{tp_name}"
    to_segment = segments.SegmentsSuper.TIMEPERIOD.get_segment().model_copy()
    to_segment.name = f"to_{tp_name}"

    new_custom = [from_segment, to_segment]
    custom_segments = [
        i for i in input_.segmentation.input.custom_segments if i.name not in remove
    ] + new_custom

    config = segmentation.SegmentationInput(
        enum_segments=enum_segments,
        naming_order=names + [i.value for i in new_enum] + [i.name for i in new_custom],
        custom_segments=custom_segments,
    )

    return segmentation.Segmentation(config)


def _calculate_tour_proportions(
    from_home: pd.DataFrame,
    to_home: pd.DataFrame,
    phi_factors: pd.DataFrame,
    slice_params: dict[str, int],
    output: Matrices,
    tp_name: str = "{}_tp",
):
    from_array = _normalise_to_xarray(from_home)
    to_array = _normalise_to_xarray(to_home)

    time_periods = phi_factors.index.tolist()

    phi = pd.DataFrame(
        phi_factors, index=phi_factors.index.tolist(), columns=time_periods
    ).stack()
    seed_index = pd.MultiIndex.from_product(
        [time_periods, time_periods, output.zoning.zone_ids, output.zoning.zone_ids],
        names=["from", "to", "o", "d"],
    )
    phi = phi.reindex(seed_index)

    furness_return_vals: xarray.DataArray
    furness_return_vals, rmse, iter_ = furness.numpy_ndim_furness(
        phi.to_xarray(),
        [from_array, to_array],
        len(time_periods) * (len(output.zoning) ** 2),
    )
    LOG.info(
        "tour proportions furnessing complete after %s iterations with RMSE=%.0e", iter_, rmse
    )

    tour_props = furness_return_vals.to_dataframe(name="trips")
    for from_tp, to_tp in itertools.product(time_periods, time_periods):
        data = tour_props.loc[from_tp, to_tp].unstack("d")
        output.set_matrix(
            data, slice_params | {tp_name.format("from"): from_tp, tp_name.format("to"): to_tp}
        )


def od_to_pa(
    input_: Matrices,
    output: Matrices,
    balancing_method: Literal["op", "24"],
    phi: factors.PhiFactors,
    *,
    occ_factors: base.DVector | None = None,
    tp_factors: dict[int, int] | None = None,
    calculate_tour_proportions: bool = True,
):
    _validate_od_input_outputs(input_, output)

    return_factors = input_.new("od_return_factors")
    LOG.debug("OD return factors will be saved to %s", input_)

    od_adjustments = input_.new(
        "od_adjustment_factors", segmentation_=_od_adjustment_segmentation(input_)
    )
    LOG.debug("OD adjustment factors will be saved to %s", input_)

    tour_proportions = None
    if calculate_tour_proportions:
        tour_proportions = input_.new(
            "tour_proportions", segmentation_=_tour_proportions_segmentation(input_)
        )
        LOG.debug("Tour proportions will be saved to %s", input_)

    tp_name = segments.SegmentsSuper.TIMEPERIOD.value

    slices_iter: pd.DataFrame = input_.segmentation.ind().to_frame(index=False)
    slices_iter = slices_iter.drop(
        columns=[input_.type.direction_segment.name, tp_name]
    ).drop_duplicates()

    for params in slices_iter.itertuples(index=False):
        params = params._asdict()
        if not input_.home_based_only:
            nhb_proportions(
                input_,
                params,
                output,
                return_factors,
                occ_factors=occ_factors,
                tp_factors=tp_factors,
            )

        if input_.non_home_based_only:
            warnings.warn(
                "input matrices are NHB only, so OD to PA conversion is ignored for HB"
            )
            continue

        phi_factors = phi.get(segmentation.SegmentationSlice(params))

        from_home, to_home, nhb = _get_time_matrices(
            input_, params, occ_factors, tp_factors
        )
        from_home, to_home, adjustments = _balance_fh_th(
            from_home,
            to_home,
            balancing_method,
            nhb,
            time_periods=input_.segmentation.get_segment_values(tp_name),
            seed_value=phi_factors.iloc[-1, -1],
        )

        output.set_matrix(
            from_home.sum(axis=1).unstack(),
            segmentation.SegmentationSlice(
                params | {output.type.direction_segment.name: _DIRECTION_VALUES["hb"]}
            ),
        )

        # Output normalised from / to home factors
        for name, data in (("from", from_home), ("to", to_home)):
            _set_matrices_by_time_period(
                return_factors,
                data.div(data.sum(axis=1).replace(0, 1), axis=0),
                params | {return_factors.type.direction_segment.name: _DIRECTION_VALUES[name]},
                column_segment=tp_name,
            )

        _set_matrices_by_time_period(
            od_adjustments, adjustments, params, column_segment=tp_name
        )

        if not calculate_tour_proportions:
            continue

        # Can't be None at this point
        assert tour_proportions is not None
        _calculate_tour_proportions(
            from_home,
            to_home,
            phi_factors,
            params,
            tour_proportions,
            tp_name=f"{{}}_{tp_name}",
        )


class OD2PAParameters(ctk.BaseConfig):
    """Define parameters for OD to PA conversion process."""

    zone_system: str
    time_period_factors: dict[int, int]

    postme_folder: pydantic.DirectoryPath
    postme_segmentation: base.SegmentationInput
    synthetic_folder: pydantic.DirectoryPath

    phi_factors: factors.PhiFactorsParameters
    occupancy_factors: factors.OccupanciesParameters


def main(parameters: OD2PAParameters):
    """Run OD to PA conversion process."""
    zone_system = base.ZoningSystem.get_zoning(parameters.zone_system)

    tp_name = segments.SegmentsSuper.TIMEPERIOD.value
    postme_segmentation = segmentation.Segmentation(parameters.postme_segmentation)

    if tp_name not in [i.name for i in postme_segmentation.segments]:
        raise ValueError("postME matrices should contain time period segmentation")

    disaggregated = disaggregate_postme(
        parameters.postme_folder,
        parameters.synthetic_folder,
        zone_system,
        postme_segmentation,
        [segments.SegmentsSuper.DIRECTION_OD.value],
    )

    phi = factors.PhiFactors.from_csv(
        parameters.phi_factors.path,
        {i: j.value for i, j in parameters.phi_factors.segment_columns.items()},
        data_column=parameters.phi_factors.data_column,
        period_filter=postme_segmentation.input.subsets[tp_name],
        period_columns=parameters.phi_factors.period_columns,
        translate_segments={
            i.value: j.value for i, j in parameters.phi_factors.segment_translation.items()
        },
        segment_filters=postme_segmentation.input.subsets,
    )
    occupancies = factors.load_occupancies(
        parameters.occupancy_factors.path,
        segment_columns={
            i: j.value for i, j in parameters.occupancy_factors.segment_columns.items()
        },
        translate_segments={
            i.value: j.value
            for i, j in parameters.occupancy_factors.segment_translation.items()
        },
    )

    pa_segments = list(
        filter(lambda x: x != tp_name, postme_segmentation.input.naming_order)
    ) + [segments.SegmentsSuper.DIRECTION.value]
    pa_matrices = disaggregated.new(
        "pa_postme",
        segmentation_=base.Segmentation(
            base.SegmentationInput(
                enum_segments=pa_segments,
                naming_order=pa_segments,
                subsets={
                    i: j for i, j in postme_segmentation.input.subsets.items() if i != tp_name
                },
            )
        ),
        type_=matrices.MatrixType.PA,
    )

    od_to_pa(
        disaggregated,
        pa_matrices,
        "op",
        phi,
        occ_factors=occupancies,
        tp_factors=parameters.time_period_factors,
    )
