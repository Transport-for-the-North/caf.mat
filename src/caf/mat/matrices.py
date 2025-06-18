# -*- coding: utf-8 -*-
"""Handling the PA and OD matrix files."""


# Built-Ins
import abc
import collections
import dataclasses
import enum
import logging
import pathlib
from typing import Iterator, Self

# Third Party
import caf.base as bs
import caf.toolkit as ctk
import numpy as np
import pandas as pd
from caf.base import segmentation, segments

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
            self.PA: segments.SegmentsSuper.DIRECTION.get_segment(),
            self.OD: segments.SegmentsSuper.DIRECTION_OD.get_segment(),
        }
        return lookup[self]


@dataclasses.dataclass
class Matrix:
    """In-memory matrix."""

    data: pd.DataFrame
    slice: segmentation.SegmentationSlice


class MatricesBase(abc.ABC):
    """Abstract base class for handling matrices split by segmentation."""

    def __init__(
        self, segmentation_: bs.Segmentation, zoning: bs.ZoningSystem, type_: MatrixType
    ):
        self._segmentation = segmentation_
        self._zoning = zoning
        self._type = type_

        if type_.direction_segment not in segmentation_.segments:
            raise ValueError(
                f"matrices with type={type_.name} should"
                f" contain {type_.direction_segment.name} segment"
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

    @property
    @abc.abstractmethod
    def name(self) -> str:
        """Name of the matrices."""
        raise NotImplementedError()

    @abc.abstractmethod
    def get_matrix(self, slice_: "segmentation.SegmentationSlice") -> Matrix:
        """Load the data for a single matrix."""
        raise NotImplementedError()

    @abc.abstractmethod
    def save_matrix(
        self, matrix: pd.DataFrame, slice_: "segmentation.SegmentationSlice"
    ) -> None:
        """Save the data for a single matrix."""
        raise NotImplementedError()

    @abc.abstractmethod
    def new(
        self,
        name: str,
        *,
        segmentation_: bs.Segmentation | None = None,
        zoning: bs.ZoningSystem | None = None,
        type_: MatrixType | None = None,
    ) -> Self:
        """Create a new instance of the matrices class with a new name.

        The current value for segmentation, zoning, and type will be used
        if not provided.
        """
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

    def validate_slice(self, slice_: "segmentation.SegmentationSlice") -> None:
        """Raise ValueError if slice not present in segmentation."""
        if slice_ not in self.segmentation.iter_slices():
            raise ValueError(
                f"Given {slice_} not found in segmentation "
                + ", ".join(self.segmentation.names)
            )

    def validate_matrix(self, matrix: pd.DataFrame, name: str | None = None) -> None:
        """Validate the matrix zoning is correct.

        Raises
        ------
        ValueError
            If the matrix is not square, or if it is missing zones.
        """
        if name is None:
            name = "Matrix"
        else:
            name = f"{name} matrix"

        matrix = matrix.sort_index(axis=0).sort_index(axis=1)

        if not matrix.index.equals(matrix.columns):
            raise ValueError(f"{name} must be square, with the same index and columns.")

        missing = self.zoning.zone_ids[np.isin(self.zoning.zone_ids, matrix.index)]
        if len(missing) > 0:
            if len(missing) > 20:
                msg = ", ".join(map(str, missing[:20])) + "..."
            else:
                msg = ", ".join(map(str, missing))

            raise ValueError(f"{name} is missing {len(missing):,} zones: {msg}.")

    def __repr__(self) -> str:
        """Return a string representation of the matrices."""
        return (
            f"{self.__class__.__name__}(type={self.type.name}, "
            f"segmentation={self.segmentation.names}, "
            f"zoning={self.zoning.name})"
        )

    def disaggregate(
        self,
        targets: "MatricesBase",
        from_segment: segments.Segment | None = None,
        to_segment: segments.Segment | None = None,
    ) -> Self:
        """Disaggregate matrices to a target segmentation.

        Optionally a single segment can be translated to another one
        as part of the disaggregation.

        Parameters
        ----------
        targets
            Matrices at the target segmentation which are used
            to calculate disaggregation factors.
        from_segment
            Optional segment to be removed and replaced by `to_segment`,
            mandatory if `to_segment` is given.
        to_segment : segments.Segment | None, optional
            Optional segment to replace the `from_segment`,
            mandatory if `from_segment` is given.

        Returns
        -------
        Self
            Disaggregated matrices as a new instance of this
            class.

        Raises
        ------
        ValueError
            - If target matrix type or zoning are different.
            - If one, but not both, of `from_segment` and `to_segment`
              are given.
            - If `targets.segmentation` doesn't contain all segments
              found in `self.segmentation`.
        """
        if targets.type != self.type:
            raise ValueError(
                f"targets should be the same type as aggregate"
                f"({self.type}) not {targets.type}"
            )

        if self.zoning != targets.zoning:
            raise ValueError(
                f"Zoning systems between aggregate ({self.zoning.name}) and"
                f" target ({targets.zoning.name}) matrices are different"
            )

        # Calculate and validate disaggregation slices
        if from_segment is not None and to_segment is not None:
            disaggregations = _get_disaggregation_translation(
                self.segmentation, from_segment, to_segment, targets.segmentation
            )
        elif from_segment is not None or to_segment is not None:
            raise ValueError(
                "One of from/to segment is provided, both must be provided when translating"
            )
        elif not self.segmentation.is_subset(targets.segmentation):
            missing, _ = targets.segmentation.subset_difference(self.segmentation)
            raise ValueError(
                "Target segmentation doesn't include all segments"
                f" from aggregate, missing {missing}"
            )
        else:
            disaggregations = _get_slice_disaggregation(
                self.segmentation, targets.segmentation
            )

        output = self.new(self.name + "_decompiled", segmentation_=targets.segmentation)

        for from_slice, to_slices in disaggregations.items():
            disagg_matrices: list[Matrix] = []
            for slice_ in to_slices:
                disagg_matrices.append(targets.get_matrix(slice_))

            LOG.info(
                "Disaggregating %s matrix into %s matrices: %s",
                self.segmentation.generate_slice_name(from_slice),
                len(disagg_matrices),
                ", ".join(
                    map(
                        lambda i: targets.segmentation.generate_slice_name(i.slice),
                        disagg_matrices,
                    )
                ),
            )
            _disaggregate_matrix(
                self.get_matrix(from_slice),
                disagg_matrices,
                output,
            )

        return output


def _get_disaggregation_translation(
    from_segmentation: segmentation.Segmentation,
    from_segment: segments.Segment,
    to_segment: segments.Segment,
    target_segmentation: segmentation.Segmentation,
) -> dict[segmentation.SegmentationSlice, list[segmentation.SegmentationSlice]]:
    """Produce slice disaggregations with a single segment lookup."""
    try:
        to_segmentation, lookup = from_segmentation.translate_segment(from_segment, to_segment)
    except FileNotFoundError:
        to_segmentation, lookup = from_segmentation.translate_segment(
            from_segment, to_segment, reverse=True
        )
        lookup = lookup.reset_index().set_index(lookup.name)

    if not to_segmentation.is_subset(target_segmentation):
        raise ValueError(
            "Target matrices aren't the correct segmentation should be "
            f"{to_segmentation.names} not {target_segmentation.names}"
        )

    groupings: dict[int, list[int]] = lookup.groupby(level=0).agg(list).squeeze().to_dict()

    # Check if any segment values are found in multiple lists i.e. many-to-many lookup
    unique_to_segs = set()
    for to_segs in groupings.values():
        for i in to_segs:
            if i in unique_to_segs:
                raise ValueError(
                    f"{to_segment.name} segment {i} found in lookup"
                    f" for multiple {from_segment.name} segments"
                )
        unique_to_segs.update(to_segs)

    disaggregations = collections.defaultdict(list)
    for from_slice in from_segmentation.iter_slices():
        to_segments = groupings.get(from_slice[from_segment.name])
        if to_segments is None:
            raise NotImplementedError("I think this means this slice stays the same?")

        for to_seg in to_segments:
            # Replace from segment value with to segment
            slice_ = from_slice.data | {to_segment.name: to_seg}
            slice_.pop(from_segment.name)

            for to_slice in target_segmentation.iter_slices(slice_):
                disaggregations[from_slice].append(to_slice)

    return disaggregations


def _get_slice_disaggregation(
    from_: segmentation.Segmentation, to: segmentation.Segmentation
) -> dict[segmentation.SegmentationSlice, list[segmentation.SegmentationSlice]]:
    """Produce slice disaggregations without any segment lookup."""
    disaggregations = collections.defaultdict(list)
    for from_slice in from_.iter_slices():
        for to_slice in to.iter_slices(from_slice.data):
            disaggregations[from_slice].append(to_slice)

    return disaggregations


def _disaggregate_matrix(
    aggregate: Matrix,
    targets: list[Matrix],
    output: MatricesBase,
):
    """Disaggregate a single matrix and save outputs."""
    total = sum(i.data for i in targets)
    for matrix in targets:
        disaggregated = aggregate.data * matrix.data / total
        output.save_matrix(disaggregated, matrix.slice)


class MemoryMatrices(MatricesBase):
    """Stores matrices in-memory, in a dictionary.

    Intended primarily for use with few smaller matrices.
    """

    def __init__(
        self,
        segmentation_: bs.Segmentation,
        zoning: bs.ZoningSystem,
        type_: MatrixType,
        matrices: list[Matrix] | None = None,
    ):
        super().__init__(segmentation_, zoning, type_)
        self._matrices: dict[segmentation.SegmentationSlice, pd.DataFrame] = {}

        if matrices is not None:
            for matrix in matrices:
                self.save_matrix(matrix.data, matrix.slice)

    @property
    def name(self) -> str:
        """Name of the matrices."""
        return "In-memory matrices"

    def new(self, name, *, segmentation_=None, zoning=None, type_=None) -> "MemoryMatrices":
        return MemoryMatrices(
            segmentation_=self._segmentation if segmentation_ is None else segmentation_,
            zoning=self._zoning if zoning is None else zoning,
            type_=self._type if type_ is None else type_,
        )

    def get_matrix(self, slice_: segmentation.SegmentationSlice) -> Matrix:
        """Get in-memory matrix."""
        self.validate_slice(slice_)

        if slice_ not in self._matrices:
            raise KeyError(f"no matrix found for {slice_}")
        return Matrix(self._matrices[slice_], slice_)

    def save_matrix(self, matrix: pd.DataFrame, slice_: segmentation.SegmentationSlice):
        """Store matrix in class (in-memory)."""
        self.validate_slice(slice_)
        self._matrices[slice_] = matrix


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

    @property
    def name(self) -> str:
        """Name of the matrices."""
        return self._folder.name

    def save_matrix(
        self, matrix: pd.DataFrame, slice_: segmentation.SegmentationSlice
    ) -> None:
        """Save the matrix to a CSV, with a filename based on the slice parameters."""
        slice_name = self._segmentation.generate_slice_name(slice_)

        self.validate_matrix(matrix, slice_name)

        filename = self._filename_template.format(segment_name=slice_name)
        path = self._folder / (filename + self._file_suffixes[0])

        matrix.to_csv(path)
        LOG.info("Written: %s", path)

    def get_matrix(self, slice_: segmentation.SegmentationSlice) -> Matrix:
        """Load the matrix from a CSV."""
        name = self.segmentation.generate_slice_name(slice_)
        filename = self._filename_template.format(segment_name=name)

        path = ctk.io.find_file_with_name(self._folder, filename, self._file_suffixes)
        LOG.info("Loading matrix file: %s", path)
        data = ctk.io.read_csv_matrix(path, format_="square")

        self.validate_matrix(data, name)

        return Matrix(data, slice_)

    def new(
        self,
        name: str,
        *,
        segmentation_: bs.Segmentation | None = None,
        zoning: bs.ZoningSystem | None = None,
        type_: MatrixType | None = None,
    ) -> Self:
        return MatrixFiles(
            segmentation_=self._segmentation if segmentation_ is None else segmentation_,
            zoning=self._zoning if zoning is None else zoning,
            type_=self._type if type_ is None else type_,
            folder=self._folder.with_name(name),
            filename_template=self._filename_template,
            check_files=False,
        )
