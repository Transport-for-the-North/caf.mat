"""Time period and from / to home factors required for PA and OD conversions."""

##### IMPORTS #####

# Built-Ins
import logging
import pathlib
import warnings
from collections.abc import Collection, Mapping, Sequence
from typing import Self

# Third Party
import caf.base as cbase
import caf.toolkit as ctk
import numpy as np
import pandas as pd
import pydantic
from caf.base import segmentation, segments
from pydantic import dataclasses

##### CONSTANTS #####

LOG = logging.getLogger(__name__)
_OD_DIRECTION_LOOKUP = {"nhb": 0, "hb_fr": 1, "hb_to": 2}

##### CLASSES & FUNCTIONS #####


class UnexpectedPhiFactorsWarning(UserWarning):
    """Warning for unexpected input for PhiFactors, which can be handled."""


class InvalidPhiFactorsError(ValueError):
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
    """Managing phi factors data for OD to PA conversion.

    Parameters
    ----------
    data
        DataFrame with integer column names for time period (to home)
        and segmentation defined in the index. The segmentation
        index should at least contain time period (from home) and
        any other segments should be lists in `additional_segments`
        parameter.
    period_filter
        Optional list of time periods (columns) to keep,
        if not given all time period columns are used.
    additional_segments
        Optional list of segments in addition to the time periods,
        these should be include in the index.
    segment_filters
        Optional filters to apply to any of the segment columns
        in the index.
    """

    _tp_segment_enum = segments.SegmentsSuper.TIMEPERIOD
    _period_columns = ("tp", "tp.to")

    def __init__(
        self,
        data: pd.DataFrame,
        period_filter: Collection[int] | None = None,
        additional_segments: Sequence[str] | None = None,
        segment_filters: Mapping[str, Sequence[int]] | None = None,
    ) -> None:

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

        # TODO(MB): This could be a parameter which warns user if not already sums to 1
        # Normalise time period factors, so time period from sums to 1
        # i.e. all trips leaving in 1 time period must return at some point
        self._data = self._data.div(self._data.sum(axis=1), axis=0)

    def _validate_columns(
        self, data: pd.DataFrame, expected_columns: set[int]
    ) -> pd.DataFrame:
        """Raise error if columns are missing and warns about extras."""
        columns = set(data.columns.to_list())
        if columns != expected_columns:
            extra = columns - expected_columns
            warnings.warn(
                f"{len(extra)} extra columns are ignored: {extra}",
                UnexpectedPhiFactorsWarning,
                stacklevel=2,
            )

            missing = expected_columns - columns
            if len(missing) > 0:
                raise InvalidPhiFactorsError(
                    f"{len(missing)} expected columns missing {missing}"
                )

        data = data[list(expected_columns)]

        if len(data.columns) == 0:
            raise InvalidPhiFactorsError("no time period columns")

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
            enum_segments = [*list(additional_segments), self._tp_segment.name]  # type: ignore[list-item]
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
        """Get phi factors for a single slice as a time period matrix.

        Returns
        -------
        DataFrame
            Square matrix where the index and columns are the time
            period integers, from and to home respectively.
        """
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
        *,
        data_column: str = "trips.est",
        period_filter: Collection[int] | None = None,
        period_columns: tuple[str, str] = ("period.fr", "period"),
        translate_segments: dict[str, str] | None = None,
        segment_filters: Mapping[str, Sequence[int]] | None = None,
    ) -> "PhiFactors":
        """Load phi factors from a CSV.

        Parameters
        ----------
        path
            Path to the CSV.
        segment_columns
            Names of CSV columns (keys) and their corresponding
            segment (values).
        data_column
            Name of column containing trips data, by default "trips.est".
        period_filter
            Optional list of time periods to filter down to.
        period_columns
            Name of columns containing time periods from and to home,
            by default ("period.fr", "period").
        translate_segments
            Mapping of any segments that need translating, translates
            from keys to values. All must be names of segments not columns.
        segment_filters
            Optional segment filters, for any additional segmentation provided.
        """
        dtypes = {
            **dict.fromkeys(tuple(segment_columns) + period_columns, int),
            data_column: float,
        }
        data = ctk.io.read_csv(
            path, "Tour Proportions", dtype=dtypes, usecols=list(dtypes.keys())
        )
        data = data.rename(
            columns=segment_columns
            | dict(zip(period_columns, cls._period_columns, strict=True))
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
    segment_translation: dict[segments.SegmentsSuper, segments.SegmentsSuper] | None = None
    driver_column: str | None = None
    total_column: str | None = None
    occupancy_column: str | None = None

    @pydantic.model_validator(mode="after")
    def _valid_segments(self) -> Self:
        """Validate no contradicting columns / segments are provided."""
        if self.segment_translation is None:
            return self

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

    @pydantic.model_validator(mode="after")
    def _validate_value_columns(self) -> Self:
        """Check required value columns are given."""
        _validate_occupancy_columns(
            self.driver_column, self.total_column, self.occupancy_column
        )
        return self

    @property
    def segment_names(self) -> dict[str, str]:
        """Names of columns (key) and segments (value)."""
        return {i: j.value for i, j in self.segment_columns.items()}

    @property
    def segment_translation_names(self) -> dict[str, str] | None:
        """Names of segments to be translated."""
        if self.segment_translation is None:
            return None
        return {i.value: j.value for i, j in self.segment_translation.items()}


def _validate_occupancy_columns(
    driver_column: str | None, total_column: str | None, occupancy_column: str | None
) -> None:
    if occupancy_column is None and (total_column is None or driver_column is None):
        raise ValueError(
            "if occupancy column isn't given then total_column and"
            " driver_column are required to calculate occupancies"
        )
    if (total_column is None) ^ (driver_column is None):
        raise ValueError(
            "both total_column and driver_column are required to calculate occupancies"
        )


def load_occupancies(
    path: pathlib.Path,
    segment_columns: dict[str, str],
    *,
    driver_column: str | None = "driver",
    total_column: str | None = "total",
    occupancy_column: str | None = None,
    translate_segments: dict[str, str] | None = None,
) -> cbase.DVector:
    """Load occupancy factors from a CSV.

    Parameters
    ----------
    path
        Path to CSV.
    segment_columns
        Mapping from column names (keys) to segments (values).
    driver_column
        Name of column containing number of drivers (default "driver"),
        used for calculating occupancies. Set to None if providing
        precalculated `occupancy_column`.
    total_column
        Name of column containing total people (default "total"),
        used for calculating occupancies. Set to None if providing
        precalculated `occupancy_column`.
    occupancy_column
        Name of column containing pre-calculated occupancies, required
        if total or driver columns aren't provided.
    translate_segments
        Mapping to translate segments given (keys) to
        new segments (values) for output occupancies.

    Returns
    -------
    caf.base.DVector
        Occupancies data as a DVector.

    Raises
    ------
    ValueError
        If duplicates are found in the segmentation (often the case
        when translating) and `total_column` or `driver_column` isn't
        provided to recalculate the occupancies.
    """
    # TODO(MB): Reimplement this as a class which supports matrices (LongMatrices)
    _validate_occupancy_columns(driver_column, total_column, occupancy_column)

    dtypes = {
        **dict.fromkeys(tuple(segment_columns), int),
        **{i: float for i in (driver_column, total_column, occupancy_column) if i is not None},
    }
    data = ctk.io.read_csv(
        path, "occupancy factors", dtype=dtypes, usecols=list(dtypes.keys())
    )
    data = data.rename(columns=segment_columns)

    # direction_od segments have different names in occupancies dataset
    for col in data.columns:
        if col == segments.SegmentsSuper.DIRECTION_OD.value:
            data[col] = data[col].replace(_OD_DIRECTION_LOOKUP).astype(int)

    if translate_segments is not None:
        data = _replace_segment_columns(data, translate_segments)

    if translate_segments is None:
        columns = list(segment_columns.values())
    else:
        columns = [translate_segments.get(i, i) for i in segment_columns.values()]

    if not data.duplicated(columns).any():
        LOG.debug("Occupancies column used from input: %s", occupancy_column)
        data = data.set_index(columns)[occupancy_column]

    elif total_column is not None and driver_column is not None:
        LOG.debug(
            "Occupancies recalculated after grouping as %s / %s", total_column, driver_column
        )
        data = data.groupby(columns)[[driver_column, total_column]].sum()
        data = data[total_column] / data[driver_column]

    else:
        raise ValueError(
            f"occupancies contains duplicates within {columns}"
            f" columns but groupby recalculation cannot be done"
            f" because {total_column=} and {driver_column=}"
        )

    data.name = "occupancies"

    segmentation_ = segmentation.Segmentation(
        segmentation.SegmentationInput(
            enum_segments=columns,  # type: ignore[arg-type]
            naming_order=columns,
        )
    )

    try:
        return cbase.DVector(segmentation_, data)
    except segmentation.SegmentationError:
        LOG.exception("error creating DVector for occupancies, trying cut_read=True")
        return cbase.DVector(segmentation_, data, cut_read=True)
