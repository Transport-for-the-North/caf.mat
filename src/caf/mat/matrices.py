# -*- coding: utf-8 -*-
"""Handling the PA and OD matrix files."""


# Built-Ins
import abc
import dataclasses
import enum
import logging
import pathlib
from typing import Iterator

# Third Party
import caf.base as bs
import caf.toolkit as ctk
import pandas as pd
from caf.base import segments

# Local Imports
from caf.mat import _mat

##### CONSTANTS #####

LOG = logging.getLogger(__name__)


##### CLASSES & FUNCTIONS #####


class PAError(_mat.MatrixError):
    """Error with PA matrices or PA to OD conversion."""


class MatrixType(enum.Enum):
    """Type of matrix."""

    PA = enum.auto()
    OD = enum.auto()

    @property
    def direction_segment(self) -> segments.Segment:
        """Required direction segment for the matrix type."""
        lookup = {
            self.PA: segments.SegmentsSuper.DIRECTION,
            self.OD: segments.SegmentsSuper.DIRECTION_OD,
        }
        return lookup[self]


@dataclasses.dataclass
class Matrix:
    """In-memory matrix."""

    data: pd.DataFrame
    slice_: dict[str, int]


class MatricesBase(abc.ABC):
    """Abstract base class for handling matrices split by segmentation."""

    def __init__(
        self, segmentation_: bs.Segmentation, zoning: bs.ZoningSystem, type_: MatrixType
    ):
        self._segmentation = segmentation_
        self._zoning = zoning
        self._type = type_

        if type_.direction_segment() not in segmentation_.segments:
            raise ValueError(
                f"matrices with type={type_.name} should contain {type_.direction_segment().name} segment"
            )

    def __iter__(self) -> Iterator[Matrix]:
        """Iterate through all segments and provide the matrix.

        Time period information should be included in the segmentation.
        """
        for values in self._segmentation.iter_slices():
            yield self.get_matrix(values)

    @property
    def segmentation(self) -> bs.Segmentation:
        """Copy of the matrix segmentation."""
        return self._segmentation.copy()

    @property
    def zoning(self) -> bs.ZoningSystem:
        """Copy of the matrix zoning system."""
        return self._zoning.copy()

    @property
    def type(self) -> MatrixType:
        """Type of matrix, either PA or OD."""
        return self._type

    @abc.abstractmethod
    def get_matrix(self, slice_: dict[str, int]) -> Matrix:
        """Load the data for a single matrix."""
        raise NotImplementedError()

    @abc.abstractmethod
    def save_matrix(self, matrix: pd.DataFrame, slice_: dict[str, int]) -> None:
        """Save the data for a single matrix."""
        raise NotImplementedError()

    @property
    def has_time_periods(self) -> bool:
        """Return True if the segmentation contains time periods."""
        # TODO check this method works correctly with all time periods Segments
        # TODO get information about the time format, e.g. avg hour / period
        return self._segmentation.has_time_period_segments()

    def _get_direction_subset(self) -> None | set[int]:
        subsets = self._segmentation.input.subsets
        direction = self._type.direction_segment

        if direction.name not in subsets:
            return None

        return set(subsets[direction.name])

    @property
    def home_based_only(self) -> bool:
        """Return True if segmentation contains home-based direction only."""
        subset = self._get_direction_subset()

        # TODO(MB) can this hardcoding be removed and instead obtained from caf.base?
        # Does subset contain one, or both (OD), of the home-based directions
        if subset is not None and subset <= {1, 2}:
            return True
        return False

    @property
    def non_home_based_only(self) -> bool:
        """Return True if segmentation contains non-home-based direction only."""
        subset = self._get_direction_subset()

        # TODO(MB) can this hardcoding be removed and instead obtained from caf.base?
        if subset == {0}:
            return True
        return False


class MockMatrices(MatricesBase):
    """Mock matrices class for testing."""

    def get_matrix(self, slice_):
        raise NotImplementedError("WIP!")

    def save_matrix(self, matrix, slice_):
        raise NotImplementedError("WIP!")


class MatrixFiles(MatricesBase):
    """Handle matrices stored as CSVs in a single folder.

    .. todo::
        Add support for zipped folders.
    """

    _file_suffixes: tuple[str] = (".csv.bz2", ".csv")

    def __init__(
        self,
        segmentation_: bs.Segmentation,
        zoning: bs.ZoningSystem,
        type_: MatrixType,
        folder: pathlib.Path,
        *,
        filename_template: str = "{type}_{segment_name}",
        check_files: bool = True,
    ):
        super().__init__(segmentation_, zoning, type_)

        self._filename_template = filename_template.format(
            type=self.type.name, segment_name="{segment_name}"
        )
        self._folder = folder.resolve()

        if not self._folder.is_dir() and check_files:
            raise NotADirectoryError(folder)

        if not self._folder.is_dir():
            LOG.debug('creating empty matrices folder: "%s"', self._folder)
            self._folder.mkdir()

        if check_files:
            self._segmentation.find_files(
                self._folder, self._filename_template, self._file_suffixes
            )

    def save_matrix(self, matrix: pd.DataFrame, slice_: dict[str, int]) -> None:
        """Save the matrix to a CSV, with a filename based on the slice parameters."""
        slice_name = self._segmentation.generate_slice_name(slice_)
        filename = self._filename_template.format(segment_name=slice_name)
        path = self._folder / (filename + self._file_suffixes[0])

        matrix.to_csv(path)
        LOG.info("Written: %s", path)

    def get_matrix(self, slice_: dict[str, int]) -> Matrix:
        """Load the matrix from a CSV."""
        name = self.segmentation.generate_slice_name(slice_)
        filename = self._filename_template.format(segment_name=name)

        path = ctk.io.find_file_with_name(self._folder, filename, self._file_suffixes)
        LOG.info("Loading matrix file: %s", path)
        data = ctk.io.read_csv_matrix(path, format_="square")

        return Matrix(data, slice_)
