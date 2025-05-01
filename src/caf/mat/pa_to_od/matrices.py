# -*- coding: utf-8 -*-
"""Handling the PA and OD matrix files."""


# Built-Ins
import abc
import collections.abc
import dataclasses
import logging
import pathlib
import re
import warnings
from typing import Iterator, Literal

# Third Party
import caf.base as bs
import caf.toolkit as ctk
import pandas as pd

# Local Imports
from caf.mat import _mat

##### CONSTANTS #####

LOG = logging.getLogger(__name__)


##### CLASSES & FUNCTIONS #####


class PAError(_mat.MatrixError):
    """Error with PA matrices or PA to OD conversion."""


@dataclasses.dataclass
class Matrix:
    data: pd.DataFrame
    segment: dict[str, int]


class _Base(abc.ABC):

    def __init__(self, segmentation_: bs.Segmentation, zoning: bs.ZoningSystem):
        self._segmentation = segmentation_
        self._zoning = zoning

    def __iter__(self) -> Iterator[Matrix]:
        """Iterate through all segments and provide the matrix.

        Time period information should be included in the segmentation.
        """
        for values in self._segmentation.iter_slices():
            yield self.get_matrix(values)

    def _segment_dict_to_tuple(self, segment: dict[str, int]) -> tuple[int, ...]:
        return tuple(segment[i] for i in self._segmentation.naming_order)

    @property
    def segmentation(self) -> bs.Segmentation:
        return self._segmentation.copy()

    @property
    def zoning(self) -> bs.ZoningSystem:
        return self._zoning.copy()

    @abc.abstractmethod
    def get_matrix(self, slice_: dict[str, int]) -> Matrix:
        raise NotImplementedError()

    @abc.abstractmethod
    def save_matrix(self, matrix: pd.DataFrame, slice_: dict[str, int]) -> None:
        raise NotImplementedError()

    @property
    def has_time_periods(self) -> bool:
        # TODO check this method works correctly with all time periods Segments
        # TODO get information about the time format, e.g. avg hour / period
        return self._segmentation.has_time_period_segments()

    def _check_purpose_subsets(self, direction: Literal["hb", "nhb"]) -> bool:
        segment, subset = _purpose_subsets()
        values = subset[direction]

        try:
            purposes = self._segmentation.seg_dict[segment.value]
        except KeyError as exc:
            raise ValueError(
                f"segmentation doesn't contain purposes '{segment.value}'"
            ) from exc

        return set(values) == set(purposes)

    def _get_purpose_subset(self, direction: Literal["hb", "nhb"]) -> bs.Segmentation:
        segment, subset = _purpose_subsets()
        return self._segmentation.update_subsets({segment.value: subset[direction]})

    @property
    def home_based_only(self) -> bool:
        """Return True if segmentation contains home-based purposes only."""
        return self._check_purpose_subsets("hb")

    @property
    def non_home_based_only(self) -> bool:
        """Return True if segmentation contains non-home-based purposes only."""
        return self._check_purpose_subsets("nhb")


def _purpose_subsets() -> tuple[bs.segments.SegmentsSuper, dict[str, list[int]]]:
    pattern = re.compile(r"\b(N?HB)\b", re.I)
    segment = bs.segments.SegmentsSuper.PURPOSE
    subsets: dict[str, list[int]] = {"hb": [], "nhb": []}

    for value, name in segment.get_segment().values.items():
        match = pattern.search(value)

        if match is None:
            raise ValueError(f"found unrecognised purpose: {name}")

        hb = match.group(1).lower()
        subsets[hb].append(value)

    return segment, subsets


class MockMatrices(_Base):

    def get_matrix(self, slice_):
        raise NotImplementedError("WIP!")

    def save_matrix(self, matrix, slice_):
        raise NotImplementedError("WIP!")


class _BasePA(_Base):
    def __init__(self, segmentation_: bs.Segmentation, home_based: bool):
        super().__init__(segmentation_)

        self._direction: Literal["hb", "nhb"] = "hb" if home_based else "nhb"

        if home_based and self.home_based_only:
            return
        if not home_based and self.non_home_based_only:
            return

        # Filter segmentation to only include relevant purposes
        warnings.warn(
            "segmentation contains NHB and HB purposes,"
            f" filtering to only include {self._direction.upper()} purposes",
            UserWarning,
        )
        self._segmentation = self._get_purpose_subset(self._direction)


class _BaseOD(_Base):
    def __init__(self, segmentation_: bs.Segmentation, direction: Literal["fh", "th", "nhb"]):
        super().__init__(segmentation_)

        self._direction = direction

        if direction in ("fh", "th") and self.home_based_only:
            return

        if direction == "nhb" and self.non_home_based_only:
            return

        # Filter segmentation to only include relevant purposes
        purp_direction: Literal["hb", "nhb"]
        if direction in ("fh", "th"):
            purp_direction = "hb"
        else:
            purp_direction = "nhb"
        warnings.warn(
            "segmentation contains NHB and HB purposes,"
            f" filtering to only include {direction.upper()} purposes",
            UserWarning,
        )
        self._segmentation = self._get_purpose_subset(purp_direction)


class ODMatricesFiles(_BaseOD):

    _file_suffixes = (".csv.bz2", ".csv")

    def __init__(
        self,
        segmentation_: bs.Segmentation,
        direction: Literal["fh", "th", "nhb"],
        folder: pathlib.Path,
        filename_template: str = "{direction}_od_{segment_name}",
        check_files: bool = True,
    ):
        super().__init__(segmentation_, direction)

        self._filename_template = filename_template.format(
            direction=direction, segment_name="{segment_name}"
        )
        self._folder = folder.resolve()

        if not self._folder.is_dir() and check_files:
            raise NotADirectoryError(folder)

        if not self._folder.is_dir():
            LOG.debug('creating empty OD matrices folder: "%s"', self._folder)
            self._folder.mkdir()

        if check_files:
            filepaths = self._segmentation.find_files(
                self._folder, self._filename_template, self._file_suffixes
            )

    def save_matrix(self, matrix: pd.DataFrame, segment: dict[str, int]):
        segment_name = self._segmentation.generate_slice_name(segment)
        filename = self._filename_template.format(segment_name=segment_name)
        path = self._folder / (filename + self._file_suffixes[0])

        matrix.to_csv(path)
        LOG.info("Written: %s", path)

    def get_matrix(self, segment: dict[str, int]) -> Matrix:
        raise NotImplementedError("WIP")
