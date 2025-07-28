# -*- coding: utf-8 -*-
"""Handling the PA and OD matrix files."""


# Built-Ins
import abc
import collections
import collections.abc
import dataclasses
import enum
import logging
import pathlib
import warnings
from typing import Iterator, Self

# Third Party
import caf.base as bs
import caf.toolkit as ctk
import numpy as np
import pandas as pd
import tqdm
from caf.base import segmentation, segments

# Local Imports
from caf.mat import _mat

##### CONSTANTS #####

LOG = logging.getLogger(__name__)


##### CLASSES & FUNCTIONS #####


class PAError(_mat.MatrixError):
    """Error with PA matrices or PA to OD conversion."""


class MatricesWarning(Warning):
    """Warning for matrix functionality."""


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
    """Abstract base class for handling matrices split by segmentation.

    Parameters
    ----------
    segmentation_
        Segmentation for the matrices.
    zoning
        ZoningSystem for all the matrices.
    type_ : MatrixType
        Type of the matrices.
    """

    def __init__(
        self, segmentation_: bs.Segmentation, zoning: bs.ZoningSystem, type_: MatrixType
    ):
        self._segmentation = segmentation_
        self._zoning = zoning
        self._type = type_

        if not self.has_type_segment:
            warnings.warn(
                "matrices doesn't contain direction segment "
                f"({type_.direction_segment.name}) some functionality won't be possible",
                MatricesWarning,
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
    def set_matrix(
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

    @abc.abstractmethod
    def exists(self) -> bool:
        """Check if matrices exist for all slices."""
        raise NotImplementedError()

    @property
    def has_time_periods(self) -> bool:
        """Return True if the segmentation contains time periods."""
        # TODO check this method works correctly with all time periods Segments
        # TODO get information about the time format, e.g. avg hour / period
        return self._segmentation.has_time_period_segments()

    @property
    def has_type_segment(self) -> bool:
        """Return True if segmentation contains correct direction segment."""
        segment = self.type.direction_segment.name
        return segment in (i.name for i in self.segmentation.segments)

    def _get_direction_subset(self) -> None | set[int]:
        if not self.has_type_segment:
            return None

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

    def _validate_zones(self, zones: np.ndarray, name: str) -> None:
        """Raise ValueError if zones doesn't contain only all zone IDs."""
        missing = self.zoning.zone_ids[~np.isin(self.zoning.zone_ids, zones)]
        if len(missing) > 0:
            raise ValueError(
                f"{name} is missing {len(missing):,} zones: {_short_list(missing)}"
            )

        extra = zones[~np.isin(zones, self.zoning.zone_ids)]
        if len(extra) > 0:
            raise ValueError(
                f"{name} has {len(extra):,} zones not found in"
                f" zone system ({self.zoning.name}): {_short_list(extra)}"
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

        self._check_matrix_nan(matrix, name)
        self._validate_zones(matrix.index.to_numpy(), name)

    @staticmethod
    def _check_matrix_nan(matrix: pd.DataFrame, name: str) -> None:
        """Raise ValueError if matrix contains any non-finite values."""
        if np.isfinite(matrix.to_numpy()).all():
            return

        na = np.isnan(matrix.to_numpy())
        inf = np.isinf(matrix.to_numpy())

        messages = []
        for number, array in (("NaN", na), ("Infinite", inf)):
            if array.any():
                msg = f"{np.sum(array):,} {number} cells"

                for i, nm in enumerate(("origin", "destination")):
                    zones = matrix.index.to_numpy()[np.any(array, axis=i)]
                    msg += (
                        f"\n\t\t{len(zones):,} {nm} rows with"
                        f" {number} values: {_short_list(zones)}"
                    )

                messages.append(msg)

        raise ValueError(f"{name} contains invalid values\n\t" + "\n\t".join(messages))

    def __repr__(self) -> str:
        """Return a string representation of the matrices."""
        return (
            f"{self.__class__.__name__}(name={self.name}, type={self.type.name},"
            f" segmentation={self.segmentation.names}, zoning={self.zoning.name})"
        )

    def aggregate(
        self,
        segmentation_: "segmentation.Segmentation",
        output_name: str = "{name}-aggregated",
        progress_bar: bool = True,
    ) -> Self:
        """Aggregate matrices to target segmentation.

        Outputs aggregated to a new matrices class.

        Parameters
        ----------
        segmentation_ : segmentation.Segmentation
            Segmentation to aggregate to, must be a subset of
            current segmentation.
        output_name : str
            Name for the output matrices, default "{name}-aggregated",
            where name is `self.name`.
        progress_bar
            If True display progress bar for aggregation.

        Returns
        -------
        Self
            Aggregated matrices class.

        Raises
        ------
        ValueError
            If `segmentation_` isn't a subset of `self.segmentation`.
        """
        if not segmentation_.is_subset(self.segmentation):
            raise ValueError("cannot aggregate to segmentation which isn't a subset")

        output = self.new(output_name.format(name=self.name), segmentation_=segmentation_)

        LOG.info(
            "Aggregating %s to segments %s, outputting as %s",
            self.name,
            ", ".join(segmentation_.names),
            output.name,
        )

        if progress_bar:
            iterator = tqdm.tqdm(
                segmentation_.iter_slices(),
                total=len(segmentation_),
                desc=f"Aggregating {self.name}",
            )
        else:
            iterator = segmentation_.iter_slices()

        for to_slice in iterator:
            total = 0
            for from_slice in self.segmentation.iter_slices(to_slice.data):
                total += self.get_matrix(from_slice).data

            output.set_matrix(total, to_slice)

        return output

    def disaggregate(
        self,
        targets: "MatricesBase",
        from_segment: segments.Segment | None = None,
        to_segment: segments.Segment | None = None,
        *,
        output_name: str = "{name}-disaggregated",
        progress_bar: bool = True,
        ignore_if_exists: bool = True,
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
        output_name
            Name of disaggregated matrices defaults to "{name}-disaggregated",
            will replace "{name}" with the name of this instance.
        progress_bar
            If True display progress bar for disaggregation.
        ignore_if_exists
            If True check if all disaggregated matrices already exist
            and don't attempt to recreate.

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

        output = self.new(
            output_name.format(name=self.name), segmentation_=targets.segmentation
        )

        disaggregation_segments = ", ".join(
            filter(
                lambda x: x not in self.segmentation.seg_dict,
                targets.segmentation.seg_dict,
            )
        )
        if ignore_if_exists and output.exists():
            LOG.debug(
                "Dissagregation of %s to additional segments (%s) already exists: %s",
                self.name,
                disaggregation_segments,
                output.name,
            )
            return output

        LOG.info(
            "Disaggregating %s to additional segments: %s", self.name, disaggregation_segments
        )

        if progress_bar:
            iterator = tqdm.tqdm(
                disaggregations.items(), desc=f"Disaggregating {self.name}", dynamic_ncols=True
            )
        else:
            iterator = disaggregations.items()

        for from_slice, to_slices in iterator:
            disagg_matrices: list[Matrix] = []
            for slice_ in to_slices:
                disagg_matrices.append(targets.get_matrix(slice_))

            LOG.debug(
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


def _short_list(values: collections.abc.Sequence, length: int = 10) -> str:
    if len(values) <= length:
        return ", ".join(map(str, values))

    half = length // 2
    return ", ".join(map(str, values[:half])) + "..." + ", ".join(map(str, values[-half:]))


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
    total = sum(i.data for i in targets).to_numpy()
    mask = total != 0

    for matrix in targets:
        data = matrix.data.to_numpy()
        if np.any(data[~mask] != 0):
            raise ValueError(
                f"{matrix.slice} matrix contains non-zero values in"
                " cells where the total is zero, this shouldn't be possible"
            )

        factor = np.divide(data, total, out=np.full_like(total, 0), where=mask)
        disaggregated = aggregate.data * factor
        output.set_matrix(disaggregated, matrix.slice)


class MemoryMatrices(MatricesBase):
    """Stores matrices in-memory, in a dictionary.

    Intended primarily for use with few smaller matrices.

    Parameters
    ----------
    segmentation_
        Segmentation for the matrices.
    zoning
        ZoningSystem for all the matrices.
    type_ : MatrixType
        Type of the matrices.
    matrices
        Optional list of any existing matrices already loaded,
        :meth:`get_matrix` will raise a KeyError when attempting to access
        a matrix which isn't given here or with :meth:`save_matrix`.
    """

    def __init__(
        self,
        segmentation_: bs.Segmentation,
        zoning: bs.ZoningSystem,
        type_: MatrixType,
        matrices: list[Matrix] | None = None,
        name: str | None = None,
    ):
        super().__init__(segmentation_, zoning, type_)
        self._matrices: dict[segmentation.SegmentationSlice, pd.DataFrame] = {}

        if matrices is not None:
            for matrix in matrices:
                self.set_matrix(matrix.data, matrix.slice)

        if name is None:
            name = "In-memory matrices"
        self._name = name

    @property
    def name(self) -> str:
        """Name of the matrices."""
        return self._name

    def new(self, name, *, segmentation_=None, zoning=None, type_=None) -> "MemoryMatrices":
        return MemoryMatrices(
            segmentation_=self._segmentation if segmentation_ is None else segmentation_,
            zoning=self._zoning if zoning is None else zoning,
            type_=self._type if type_ is None else type_,
            name=name,
        )

    def exists(self) -> bool:
        for slice_ in self.segmentation.iter_slices():
            if slice_ not in self._matrices:
                return False
        return True

    def get_matrix(self, slice_: segmentation.SegmentationSlice) -> Matrix:
        """Get in-memory matrix."""
        self.validate_slice(slice_)

        if slice_ not in self._matrices:
            raise KeyError(f"no matrix found for {slice_}")
        return Matrix(self._matrices[slice_], slice_)

    def set_matrix(self, matrix: pd.DataFrame, slice_: segmentation.SegmentationSlice):
        """Store matrix in class (in-memory)."""
        self.validate_slice(slice_)
        self._matrices[slice_] = matrix


class MatrixFiles(MatricesBase):
    """Handle matrices stored as CSVs in a single folder.

    Parameters
    ----------
    segmentation_
        Segmentation for the matrices.
    zoning
        ZoningSystem for all the matrices.
    type_
        Type of the matrices.
    folder
        Path to folder for storing matrices in.
    filename_template
        Template for the filenames of individual matrix CSVs, default
        "{type}_{slice_name}". Template will be infilled with the following:
        - {type} - name of the matrix `type_` e.g. "OD";
        - {slice_name} - the name of the individual slice e.g. "p1_m3_nhb"
          from :class:`SegmentationSlice`.
    check_files
        If True (default) raises an error if CSV files don't already
        exist for all slices in segmentation.

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
        filename_template: str = "{type}_{slice_name}",
        check_files: bool = True,
    ):
        super().__init__(segmentation_, zoning, type_)

        self._raw_filename_template = filename_template
        self._filename_template = filename_template.format(
            type=self.type.name, slice_name="{slice_name}"
        )
        self._folder = folder.resolve()

        if not self._folder.is_dir() and check_files:
            raise NotADirectoryError(folder)

        if not self._folder.is_dir():
            LOG.debug('creating empty matrices folder: "%s"', self._folder)
            self._folder.mkdir()

        if check_files:
            paths = self._segmentation.find_files(
                self._folder, self._filename_template, self._file_suffixes
            )
            self._filenames = {
                i: j.name.removesuffix("".join(j.suffixes)) for i, j in paths.items()
            }
        else:
            self._filenames = {}

    @property
    def name(self) -> str:
        """Name of the matrices."""
        return self._folder.name

    @property
    def folder(self) -> pathlib.Path:
        """Path to folder containing the matrices."""
        return self._folder

    def _get_filename(self, slice_: segmentation.SegmentationSlice) -> str:
        """Get filename for given slice, generates it if not already present."""
        if slice_ not in self._filenames:
            slice_name = self._segmentation.generate_slice_name(slice_)
            self._filenames[slice_] = self._filename_template.format(slice_name=slice_name)

        return self._filenames[slice_]

    def set_matrix(self, matrix: pd.DataFrame, slice_: segmentation.SegmentationSlice) -> None:
        """Save the matrix to a CSV, with a filename based on the slice parameters."""
        self.validate_slice(slice_)

        filename = self._get_filename(slice_)
        self.validate_matrix(matrix, filename)

        path = self._folder / (filename + self._file_suffixes[0])
        matrix.to_csv(path)
        LOG.debug("Written: %s", path)

    def get_matrix(self, slice_: segmentation.SegmentationSlice) -> Matrix:
        """Load the matrix from a CSV."""
        self.validate_slice(slice_)

        filename = self._get_filename(slice_)
        path = ctk.io.find_file_with_name(self._folder, filename, self._file_suffixes)

        LOG.debug("Loading matrix file: %s", path)
        data = ctk.io.read_csv_matrix(path)
        data.index.name = "origin"
        data.columns.name = "destination"

        self.validate_matrix(data, filename)
        return Matrix(data, slice_)

    def new(
        self,
        name: str,
        *,
        segmentation_: bs.Segmentation | None = None,
        zoning: bs.ZoningSystem | None = None,
        type_: MatrixType | None = None,
    ) -> Self:
        folder = self._folder.with_name(name)
        folder.mkdir(exist_ok=True)
        return MatrixFiles(
            segmentation_=self._segmentation if segmentation_ is None else segmentation_,
            zoning=self._zoning if zoning is None else zoning,
            type_=self._type if type_ is None else type_,
            folder=folder,
            filename_template=self._raw_filename_template,
            check_files=False,
        )

    def exists(self) -> bool:
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message=".*Performing in-depth search.",
                    category=RuntimeWarning,
                )
                self._segmentation.find_files(
                    self._folder,
                    self._filename_template,
                    self._file_suffixes,
                )
        except FileNotFoundError:
            return False
        return True


class LongMatrices(MatricesBase):
    """Store multiple matrices in a single long DataFrame.

    Intended only for use with smaller matrices or segmentations.
    Allows for multiple columns of data for the matrices.

    Parameters
    ----------
    segmentation_
        Segmentation for the matrices.
    zoning
        ZoningSystem for all the matrices.
    type_
        Type of the matrices.
    data
        Optional data for all matrices in a single DataFrame,
        requires index (or columns) defining the segmentation
        and origin / destination zones.
    name
        Optional name for matrices, default "LongMatrix"
    columns
        Optional list of data columns, if not given all
        columns not required for the index are used.

    .. todo::
        Override the aggregate and disaggregate methods in this
        class using DataFrame.groupby.
    """

    _origin_column: str = "origin"
    _dest_column: str = "destination"

    def __init__(
        self,
        segmentation_: bs.Segmentation,
        zoning: bs.ZoningSystem,
        type_: MatrixType,
        *,
        data: pd.DataFrame | None = None,
        name: str = "LongMatrix",
        columns: list[str] | None = None,
    ):
        super().__init__(segmentation_, zoning, type_)
        self._name = name
        self._index = self._get_index_names(segmentation_)
        self._columns = columns

        if data is None:
            self._data = self._create_empty_data()
        else:
            self._data = self._validate_data(data)

    @classmethod
    def _get_index_names(cls, segmentation_: segmentation.Segmentation) -> list[str]:
        return [*segmentation_.naming_order, cls._origin_column, cls._dest_column]

    @property
    def name(self) -> str:
        return self._name

    def _create_empty_data(self) -> pd.DataFrame:
        """Create DataFrame of NaNs with correct indices."""
        data = pd.DataFrame(
            -1,
            index=self._segmentation.ind(),
            columns=pd.MultiIndex.from_product(
                [self._zoning.zone_ids] * 2, names=[self._origin_column, self._dest_column]
            ),
            dtype=float,
        )
        data = (
            data.stack(self._origin_column, future_stack=True)
            .stack(self._dest_column, future_stack=True)
            .to_frame(name="trips")
        )
        data.loc[:] = np.nan
        return data

    def _validate_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Validate the data has correct indices and columns."""
        index = set(self._index)
        if set(data.index.names) != index:
            # If single index assume segmentation indices are columns
            if not (isinstance(data.index, pd.Index) and index <= set(data.columns.to_list())):
                raise ValueError(
                    f"expected indices {self._index} not index "
                    f"({data.index.names}) or columns ({data.columns.to_list()})"
                )

            data = data.set_index(self._index)

        assert isinstance(data.index, pd.MultiIndex)

        if not data.index.dtypes.apply(pd.api.types.is_integer_dtype).all():
            try:
                data.index = pd.MultiIndex.from_arrays(
                    [data.index.get_level_values(i).astype(int) for i in data.index.names]
                )
            except ValueError as exc:
                raise ValueError(
                    f"indices should be integers not {data.index.dtype.name}"
                ) from exc

        try:
            data = data.astype(float)
        except ValueError as exc:
            raise ValueError(
                f"matrix data should be numeric not {data.dtypes.to_list()}"
            ) from exc

        if data.index.has_duplicates:
            raise ValueError(f"duplicate indices found in {self.name}")

        if self._columns is None:
            self._columns = data.columns.to_list()
        elif set(data.columns.to_list()) != set(self._columns):
            raise ValueError(
                f"expected columns {self._columns} but given {data.columns.to_list()}"
            )

        seg_data = data.reset_index()[self.segmentation.naming_order].drop_duplicates(
            keep="first"
        )
        self.segmentation.validate_segmentation(seg_data, self.segmentation)

        for i in (self._origin_column, self._dest_column):
            self._validate_zones(
                data.index.get_level_values(i).unique().to_numpy(), f"{self.name} - {i}"
            )

        return data

    def to_frame(self, deep: bool = False) -> pd.DataFrame:
        """Return a copy of the underlying DataFrame."""
        return self._data.copy(deep)

    def get_matrix(self, slice_: segmentation.SegmentationSlice) -> Matrix | pd.DataFrame:
        self.validate_slice(slice_)
        data: pd.DataFrame = self._data.loc[slice_.as_tuple()].copy()
        if len(data.columns) > 1:
            return data

        data = data.squeeze()
        data.index.names = [None, None]
        data.name = None
        return Matrix(data.unstack(), slice_)

    def set_matrix(self, matrix: pd.DataFrame, slice_: segmentation.SegmentationSlice):
        self.validate_slice(slice_)

        if matrix.index.nlevels == 1:
            self.validate_matrix(matrix, str(slice_))
            matrix.index.name = self._origin_column
            matrix.columns.name = self._dest_column

            assert self._columns is not None
            matrix = matrix.stack().to_frame(name=self._columns[0])

        if matrix.index.names != [self._origin_column, self._dest_column]:
            raise ValueError(
                f"invalid index in matrice ({matrix.index.names}) should"
                f" be {self._origin_column}, {self._dest_column}"
            )

        if set(matrix.columns) != set(self._columns):
            raise ValueError(
                f"invalid columns in matrix ({matrix.columns}) should be {self._columns}"
            )

        matrix = matrix.copy()
        matrix.index = pd.MultiIndex.from_tuples(
            [slice_.as_tuple() + (i, j) for i, j in matrix.index]
        )

        # Cannot set multiple rows with index directly, so instead concat new columns and update
        matrix.columns = [f"{i}_set" for i in matrix.columns]
        updated = pd.concat([self._data, matrix], axis=1)
        updated.index.names = self._data.index.names
        for col in self._data.columns:
            updated[col] = np.where(
                updated[f"{col}_set"].isna(), updated[col], updated[f"{col}_set"]
            )

        self._data = updated[self._data.columns]

    def new(self, name: str, *, segmentation_=None, zoning=None, type_=None) -> "LongMatrices":
        return LongMatrices(
            segmentation_=self.segmentation if segmentation_ is None else segmentation_,
            zoning=self.zoning if zoning is None else zoning,
            type_=self.type if type_ is None else type_,
            name=name,
            columns=self._columns,
        )

    def exists(self) -> bool:
        try:
            self._validate_data(self._data)
        except ValueError:
            return False

        return not self._data.isna().any().any()

    @classmethod
    def from_csv(
        cls,
        segmentation_: bs.Segmentation,
        zoning: bs.ZoningSystem,
        type_: MatrixType,
        path: pathlib.Path,
        *,
        name: str = "LongMatrix",
        columns: list[str] | None = None,
    ) -> "LongMatrices":
        """Load matrices from a single CSV."""
        if columns is None:
            data = pd.read_csv(path)
        else:
            usecols = cls._get_index_names(segmentation_) + columns
            data = pd.read_csv(path, usecols=usecols)

        return LongMatrices(
            segmentation_,
            zoning,
            type_,
            data=data,
            name=name,
            columns=columns,
        )

    def save_csv(self, path: pathlib.Path) -> None:
        """Save matrices to a single CSV."""
        self._data.to_csv(path)
