# -*- coding: utf-8 -*-
"""Time period and from / to home factors required for PA and OD conversions."""

##### IMPORTS #####

# Built-Ins
import abc
import logging
import pathlib
import warnings
from collections.abc import Collection, Mapping, Sequence
from typing import Self

# Third Party
import caf.base as base
import caf.toolkit as ctk
import numpy as np
import pandas as pd
import pydantic
from caf.base import segmentation, segments
from pydantic import dataclasses


##### CONSTANTS #####

LOG = logging.getLogger(__name__)


##### CLASSES & FUNCTIONS #####


class UnexpectedPhiFactorsWarning(UserWarning):
    """Warning for unexpected input for PhiFactors, which can be handled."""


class InvalidPhiFactors(ValueError):
    """Error for invalid input for PhiFactors."""


@dataclasses.dataclass
class PhiFactorsParameters:
    """Parameters for loading phi factors from a CSV."""

    path: pydantic.FilePath
    segment_columns: dict[str, segments.SegmentsSuper]
    data_column: str
    segment_translation: dict[segments.SegmentsSuper, segments.SegmentsSuper]
    period_columns: tuple[str, str]

    @pydantic.model_validator(mode="after")
    def _valid_segments(self) -> Self:
        """Validate no contradicting columns / segments are provided."""
        if self.data_column in self.segment_columns:
            raise ValueError(f"{self.data_column} column defined as data and segment")

        for i, j in self.segment_translation.items():
            if i not in self.segment_columns.values():
                raise ValueError(
                    f"{i} segment defined in translation but"
                    " not defined in segment column lookup"
                )
            if j in self.segment_columns.values():
                raise ValueError(
                    f"{j} segment defined as being translated to"
                    " but already found in segments columns"
                )

        return self


class PhiFactors:

    _tp_segment_enum = segments.SegmentsSuper.TIMEPERIOD
    _period_columns = ("tp", "tp.to")

    def __init__(
        self,
        data: pd.DataFrame,
        period_filter: Collection[int] | None = None,
        additional_segments: Sequence[str] | None = None,
        segment_filters: Mapping[str, Sequence[int]] | None = None,
    ):

        self._tp_segment = self._tp_segment_enum.get_segment()

        subsets: dict[str, Sequence[int]] = {}
        if period_filter is not None:
            subsets[self._tp_segment.name] = tuple(period_filter)
            expected_columns = set(period_filter)
        else:
            expected_columns = set(self._tp_segment.values)

        data = self._validate_columns(data, expected_columns)

        if segment_filters is not None:
            subsets = subsets | dict(segment_filters)

        if additional_segments is not None:
            self._additional_segments: list[str] | None = list(additional_segments)
        else:
            self._additional_segments = None

        self._segmentation, self._data = self._validate_segmentation(
            data, self._additional_segments, subsets
        )
        # Segmentation with time period remove for validating get method
        self._segmentation_no_tp = self._segmentation.remove_segment(self._tp_segment.name)

        # TODO This could be a parameter which warns user if not already sums to 1
        # Normalise time period factors, so time period from sums to 1
        # i.e. all trips leaving in 1 time period must return at some point
        self._data = self._data.div(self._data.sum(axis=1), axis=0)

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
                UnexpectedPhiFactorsWarning,
                stacklevel=2,
            )

        data = data[list(expected_columns)]

        if len(data.columns) == 0:
            raise InvalidPhiFactors("no time period columns")

        return data

    def _validate_segmentation(
        self,
        data: pd.DataFrame,
        additional_segments: list[str] | None,
        subsets: dict[str, Sequence[int]],
    ) -> tuple[segmentation.Segmentation, pd.DataFrame]:
        if additional_segments is None:
            enum_segments = [self._tp_segment_enum]
            naming = [self._tp_segment.name]
        else:
            enum_segments = list(additional_segments) + [self._tp_segment.name]
            naming = enum_segments

        mask = np.full(len(data), True)
        for seg, values in subsets.items():
            mask = mask & np.isin(data.index.get_level_values(seg), values)
        data = data.loc[mask]

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

        slice_ = self._segmentation_no_tp.validate_slice(slice_, fix_order=True)
        return self._data.loc[slice_.as_tuple(), :]

    @classmethod
    def from_csv(
        cls,
        path: pathlib.Path,
        segment_columns: dict[str, str],
        data_column: str = "trips.est",
        period_filter: Collection[int] | None = None,
        period_columns: tuple[str, str] = ("period.fr", "period"),
        translate_segments: dict[str, str] | None = None,
        segment_filters: Mapping[str, Sequence[int]] | None = None,
    ) -> "PhiFactors":
        dtypes = {
            **dict.fromkeys(tuple(segment_columns) + period_columns, int),
            data_column: float,
        }
        data = ctk.io.read_csv(
            path, "Tour Proportions", dtype=dtypes, usecols=list(dtypes.keys())
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

        data = data.groupby([*columns, *cls._period_columns])[data_column].sum()
        data = data.unstack(cls._period_columns[1])

        return PhiFactors(
            data,
            period_filter=period_filter,
            additional_segments=columns,
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


@dataclasses.dataclass
class OccupanciesParameters:
    """Parameters for loading occupancies factors from a CSV."""

    path: pydantic.FilePath
    segment_columns: dict[str, segments.SegmentsSuper]
    segment_translation: dict[segments.SegmentsSuper, segments.SegmentsSuper]

    @pydantic.model_validator(mode="after")
    def _valid_segments(self) -> Self:
        """Validate no contradicting columns / segments are provided."""
        for i, j in self.segment_translation.items():
            if i not in self.segment_columns.values():
                raise ValueError(
                    f"{i} segment defined in translation but"
                    " not defined in segment column lookup"
                )
            if j in self.segment_columns.values():
                raise ValueError(
                    f"{j} segment defined as being translated to"
                    " but already found in segments columns"
                )

        return self


def load_occupancies(
    path: pathlib.Path,
    segment_columns: dict[str, str],
    driver_column: str = "driver",
    total_column: str = "total",
    translate_segments: dict[str, str] | None = None,
) -> base.DVector:
    # TODO(MB) Reimplement this as a class which supports matrices (LongMatrices)
    dtypes = {
        **dict.fromkeys(tuple(segment_columns), int),
        **dict.fromkeys((driver_column, total_column), float),
    }
    data = ctk.io.read_csv(
        path, "occupancy factors", dtype=dtypes, usecols=list(dtypes.keys())
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
