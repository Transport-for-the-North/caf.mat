# -*- coding: utf-8 -*-
"""Time period and from / to home factors required for PA and OD conversions."""

##### IMPORTS #####

# Built-Ins
import abc
import collections.abc
import logging
import pathlib
import warnings

# Third Party
import caf.toolkit as ctk
import pandas as pd
from caf.base import segmentation, segments

# Local Imports
from caf import base
from caf.mat import matrices

##### CONSTANTS #####

LOG = logging.getLogger(__name__)


##### CLASSES & FUNCTIONS #####


class UnexpectedTourProportionsWarning(UserWarning): ...


class InvalidTourProportions(ValueError): ...


class TimePeriod(abc.ABC):

    def get(self, _slice: dict[str, int]) -> list[matrices.Matrix]:
        """Get matrix of TP factors for given segment."""


class FromToHome(abc.ABC):

    def get_from(self, _slice: dict[str, int]) -> matrices.Matrix:
        """Get matrix of from home factors for each output time period."""

    def get_to(self, _slice: dict[str, int]) -> matrices.Matrix: ...


class PhiFactors:

    _tp_segment_enum = segments.SegmentsSuper.TIMEPERIOD
    _period_columns = ("tp.fr", "tp.to")

    def __init__(
        self,
        data: pd.DataFrame,
        period_filter: collections.abc.Collection[int] | None = None,
        additional_segments: collections.abc.Sequence[str] | None = None,
        segment_filters: dict[str, collections.abc.Sequence[int]] | None = None,
    ):

        self._tp_segment = self._tp_segment_enum.get_segment()

        subsets: dict[str, collections.abc.Sequence[int]] = {}
        if period_filter is not None:
            subsets[self._tp_segment.name] = tuple(period_filter)
            expected_columns = set(period_filter)
        else:
            expected_columns = set(self._tp_segment.values)

        data = self._validate_columns(data, expected_columns)

        if segment_filters is not None:
            subsets = subsets | segment_filters

        if additional_segments is not None:
            self._additional_segments: list[str] | None = list(additional_segments)
        else:
            self._additional_segments = None

        self._segmentation, self._data = self._validate_segmentation(
            data, self._additional_segments, subsets
        )

        # Normalise time period factors
        self._data = self._data / self._data.groupby(levels=self._additional_segments)

    def _validate_columns(
        self, data: pd.DataFrame, expected_columns: set[int]
    ) -> pd.DataFrame:
        columns = set(data.columns.to_list())
        if columns != expected_columns:
            missing = expected_columns - columns
            extra = columns - expected_columns
            warnings.warn(
                f"{len(missing)} expected columns missing ({missing})"
                f" and {len(extra)} extra columns (ignored)",
                UnexpectedTourProportionsWarning,
                stacklevel=2,
            )

        data = data[list(expected_columns)]

        if len(data.columns) == 0:
            raise InvalidTourProportions("no time period columns")

        return data

    def _validate_segmentation(
        self,
        data: pd.DataFrame,
        additional_segments: list[str] | None,
        subsets: dict[str, collections.abc.Sequence[int]],
    ) -> tuple[segmentation.Segmentation, pd.DataFrame]:
        if additional_segments is None:
            enum_segments = [self._tp_segment_enum]
            naming = [self._tp_segment.name]
        else:
            enum_segments = list(additional_segments) + [self._tp_segment.name]
            naming = enum_segments

        segmentation_ = segmentation.Segmentation(
            segmentation.SegmentationInput(
                enum_segments=enum_segments,
                naming_order=naming,
                subsets={i: list(j) for i, j in subsets.items()},
            )
        )
        index = segmentation_.ind()
        segmentation_, _ = segmentation.Segmentation.validate_segmentation(
            data, segmentation_, cut_read=True
        )

        data = data.reindex(index=index, fill_value=0)

        return segmentation_, data

    def get(self, slice_: segmentation.SegmentationSlice | None = None) -> pd.DataFrame:
        if self._additional_segments is None and slice_ is None:
            return self._data.copy()
        if self._additional_segments is None:
            warnings.warn("slice given when phi factors has no segmentation", stacklevel=2)
            return self._data.copy()
        if slice_ is None:
            raise ValueError("no slice given for getting phi factors with segmentation")

        self._segmentation.validate_slice(slice_)
        return self._data.loc[slice_.as_tuple(), :, :]

    @classmethod
    def from_csv(
        cls,
        path: pathlib.Path,
        segment_columns: dict[str, str],
        data_column: str = "trips.est",
        period_filter: collections.abc.Collection[int] | None = None,
        period_columns: tuple[str, str] = ("period.fr", "period"),
        translate_segments: dict[str, str] | None = None,
        segment_filters: dict[str, collections.abc.Sequence[int]] | None = None,
    ) -> "PhiFactors":
        dtypes = {
            **dict.fromkeys(tuple(segment_columns) + period_columns, int),
            data_column: float,
        }
        data = ctk.io.read_csv(
            path, "Tour Proportions", dtypes=dtypes, usecols=list(dtypes.keys())
        )
        data = data.rename(
            columns=segment_columns | dict(zip(period_columns, cls._period_columns))
        )

        if translate_segments is not None:
            data = _replace_segment_columns(data, translate_segments)

        if period_filter is not None:
            mask = data[cls._period_columns[0]].isin(period_filter) & data[
                cls._period_columns[1]
            ].isin(period_filter)
            data = data.loc[mask]

        if translate_segments is None:
            columns = tuple(segment_columns.values())
        else:
            columns = tuple(translate_segments.get(i, i) for i in segment_columns.values())

        data = data.groupby(columns + period_columns)[data_column].sum()
        data = data.unstack(period_columns[1])

        return PhiFactors(
            data,
            period_filter=period_filter,
            additional_segments=list(segment_columns.values()),
            segment_filters=segment_filters,
        )


def _replace_segment_columns(
    data: pd.DataFrame, translate_segments: dict[str, str]
) -> pd.DataFrame:
    drop_columns = []
    for from_, to in translate_segments.items():
        from_seg = segments.SegmentsSuper(from_).get_segment()
        to_seg = segments.SegmentsSuper(to).get_segment()
        lookup = from_seg.translate_segment(to_seg)[1].to_dict()

        data[to_seg.name] = data[from_seg.name].replace(lookup)
        drop_columns.append(from_seg.name)

    return data.drop(columns=drop_columns)


def load_occupancies(
    path: pathlib.Path,
    segment_columns: dict[str, str],
    driver_column: str = "driver",
    total_column: str = "total",
    translate_segments: dict[str, str] | None = None,
) -> base.DVector:
    dtypes = {
        **dict.fromkeys(tuple(segment_columns), int),
        **dict.fromkeys((driver_column, total_column), float),
    }
    data = ctk.io.read_csv(
        path, "occupancy factors", dtypes=dtypes, usecols=list(dtypes.keys())
    )
    data = data.rename(columns=segment_columns)

    if translate_segments is not None:
        data = _replace_segment_columns(data, translate_segments)

    if translate_segments is None:
        columns = list(segment_columns.values())
    else:
        columns = [translate_segments.get(i, i) for i in segment_columns.values()]

    data = data.groupby(columns)[[driver_column, total_column]].sum()
    data = data[total_column] / data[driver_column]
    data.name = "occupancies"

    segmentation_ = segmentation.Segmentation(
        segmentation.SegmentationInput(enum_segments=columns, naming_order=columns)
    )
    return base.DVector(segmentation_, data)


class MissingTP(abc.ABC):
    """Scaling factors for handling missing time periods during combining."""
