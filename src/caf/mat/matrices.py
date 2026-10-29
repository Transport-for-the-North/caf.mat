"""Handling the PA and OD matrix files."""
# Built-InsParty
import abc
import collections
import collections.abc
import dataclasses
import enum
import logging
import pathlib
import warnings
from collections.abc import Iterator
from typing import Callable, Self, Type, TypeVar, Literal
from numbers import Real

# Third Party
import caf.base as cbase
import caf.toolkit as ctk
import h5py
import numpy as np
import pandas as pd
import tables
import tqdm
from caf.base import segmentation, segments

# Local Imports
from caf.mat import _mat, cube, omx

##### CONSTANTS #####

LOG = logging.getLogger(__name__)

MATRICES = TypeVar("MATRICES", bound="MatricesBase")


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
        lookup: dict[MatrixType, segments.Segment] = {
            MatrixType.PA: segments.SegmentsSuper.DIRECTION.get_segment(),
            MatrixType.OD: segments.SegmentsSuper.DIRECTION_OD.get_segment(),
        }
        return lookup[self]


@dataclasses.dataclass
class Matrix:
    """In-memory matrix."""

    data: pd.DataFrame
    slice: cbase.segmentation.SegmentationSlice


@dataclasses.dataclass
class _LazyOperation:
    """Stored operation in a lazy matrices pipeline."""

    name: str
    args: tuple
    kwargs: dict[str, object]


class _LazyMatricesPipeline:
    """Lazily compose matrices operations and materialise at the end."""

    def __init__(self, source: "MatricesBase") -> None:
        self._source = source
        self._operations: list[_LazyOperation] = []

    def _append(self, name: str, *args, **kwargs) -> "_LazyMatricesPipeline":
        self._operations.append(_LazyOperation(name=name, args=args, kwargs=kwargs))
        return self

    def __getattr__(self, name: str):
        """Queue unknown method calls for deferred execution."""
        if name.startswith("_"):
            raise AttributeError(name)

        method = getattr(type(self._source), name, None)
        if method is None or not callable(method):
            raise AttributeError(
                f"{type(self._source).__name__!s} has no callable method {name!r}"
            )

        def queue_operation(*args, **kwargs):
            return self._append(name, *args, **kwargs)

        return queue_operation

    @staticmethod
    def _extract_translate_args(
        operation: _LazyOperation,
    ) -> tuple[cbase.ZoningSystem, pd.DataFrame | ctk.translation.ZoneCorrespondence | None]:
        """Extract arguments from a queued translate operation."""
        if "new_zoning" in operation.kwargs:
            new_zoning = operation.kwargs["new_zoning"]
        elif operation.args:
            new_zoning = operation.args[0]
        else:
            raise ValueError("missing new_zoning for translate_zoning")

        translation = operation.kwargs.get("translation")
        if len(operation.args) > 1 and "translation" not in operation.kwargs:
            translation = operation.args[1]
        return new_zoning, translation  # type: ignore[return-value]

    @staticmethod
    def _extract_aggregate_args(
        operation: _LazyOperation,
    ) -> tuple["segmentation.Segmentation", str, bool]:
        """Extract arguments from a queued aggregate operation."""
        if "segmentation_" in operation.kwargs:
            segmentation_ = operation.kwargs["segmentation_"]
        elif operation.args:
            segmentation_ = operation.args[0]
        else:
            raise ValueError("missing segmentation_ for aggregate")

        output_name = operation.kwargs.get("output_name", "{name}-aggregated")
        progress_bar = operation.kwargs.get("progress_bar", True)
        if len(operation.args) > 1 and "output_name" not in operation.kwargs:
            output_name = operation.args[1]
        if len(operation.args) > 2 and "progress_bar" not in operation.kwargs:
            progress_bar = operation.args[2]
        return segmentation_, output_name, progress_bar  # type: ignore[return-value]

    def _compile_transform(
        self,
        operation: _LazyOperation,
        current_zoning: cbase.ZoningSystem,
    ) -> tuple[Callable[[pd.DataFrame], pd.DataFrame], cbase.ZoningSystem] | None:
        """Compile a lazy transform operation into a per-matrix callable."""
        if operation.name == "translate_zoning":
            new_zoning, translation = self._extract_translate_args(operation)

            if translation is None:
                translation = current_zoning.translate(new_zoning)
            elif isinstance(translation, pd.DataFrame):
                translation = ctk.translation.ZoneCorrespondence(
                    translation,
                    current_zoning.column_name,
                    new_zoning.column_name,
                    current_zoning.translation_column_name(new_zoning),
                )

            def transform(matrix: pd.DataFrame) -> pd.DataFrame:
                return ctk.translation.pandas_matrix_zone_translation(matrix, translation)

            return transform, new_zoning

        if operation.name == "to_internal":
            externals = current_zoning.external

            def transform(matrix: pd.DataFrame) -> pd.DataFrame:
                matrix = matrix.copy()
                matrix.loc[externals] = 0
                matrix.loc[:, externals] = 0
                return matrix

            return transform, current_zoning

        if operation.name == "to_external":
            internals = current_zoning.internal

            def transform(matrix: pd.DataFrame) -> pd.DataFrame:
                matrix = matrix.copy()
                matrix.loc[internals, internals] = 0
                return matrix

            return transform, current_zoning

        if operation.name == "remove_intras":
            def transform(matrix: pd.DataFrame) -> pd.DataFrame:
                matrix = matrix.copy()
                np.fill_diagonal(matrix.values, 0)
                return matrix

            return transform, current_zoning

        return None

    def _fused_transforms_then_aggregate(
        self,
        matrices_: "MatricesBase",
        transforms: list[_LazyOperation],
        aggregate_operation: _LazyOperation,
    ) -> "MatricesBase":
        """Execute transform-chain then aggregate in one pass."""
        compiled_transforms: list[Callable[[pd.DataFrame], pd.DataFrame]] = []
        output_zoning = matrices_.zoning

        for operation in transforms:
            compiled = self._compile_transform(operation, output_zoning)
            if compiled is None:
                raise ValueError(f"cannot fuse lazy operation {operation.name}")
            transform_fn, output_zoning = compiled
            compiled_transforms.append(transform_fn)

        segmentation_, output_name, progress_bar = self._extract_aggregate_args(aggregate_operation)

        if not segmentation_.is_subset(matrices_.segmentation):
            raise ValueError("cannot aggregate to segmentation which isn't a subset")

        output = matrices_.new(
            output_name.format(name=matrices_.name),
            segmentation_=segmentation_,
            zoning=output_zoning,
        )

        if progress_bar:
            iterator = tqdm.tqdm(
                segmentation_.iter_slices(),
                total=len(segmentation_),
                desc=f"Lazily aggregating {matrices_.name}",
            )
        else:
            iterator = segmentation_.iter_slices()

        for to_slice in iterator:
            total = 0
            for from_slice in matrices_.segmentation.iter_slices(to_slice.data):
                matrix = matrices_.get_matrix(from_slice)
                transformed = matrix.data
                for transform in compiled_transforms:
                    transformed = transform(transformed)
                total += transformed

            output.set_matrix(total, to_slice)
        return output

    def _try_fused_collect(self, current: "MatricesBase") -> "MatricesBase | None":
        """Try to execute a fused lazy plan; return None if not applicable."""
        if len(self._operations) < 2:
            return None

        aggregate_operation = self._operations[-1]
        if aggregate_operation.name != "aggregate":
            return None

        transforms = self._operations[:-1]
        try:
            return self._fused_transforms_then_aggregate(current, transforms, aggregate_operation)
        except ValueError:
            return None

    def collect(self) -> "MatricesBase":
        """Materialise the lazy pipeline to a concrete matrices object."""
        current: MatricesBase = self._source
        fused = self._try_fused_collect(current)
        if fused is not None:
            return fused

        for operation in self._operations:
            method = getattr(current, operation.name)
            current = method(*operation.args, **operation.kwargs)
        return current

    def execute(self) -> "MatricesBase":
        """Alias for collect()."""
        return self.collect()


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
        self, segmentation_: cbase.Segmentation, zoning: cbase.ZoningSystem, type_: MatrixType
    ) -> None:
        self._segmentation = segmentation_
        self._zoning = zoning
        self._type = type_

        if not self.has_type_segment:
            warnings.warn(
                "matrices doesn't contain direction segment "
                f"({type_.direction_segment.name}) some functionality won't be possible",
                MatricesWarning,
                stacklevel=2,
            )

    def __iter__(self) -> Iterator[Matrix]:
        """Iterate through all segments and provide the matrix.

        Time period information should be included in the segmentation.
        """
        for values in self._segmentation.iter_slices():
            yield self.get_matrix(slice_=values)

    @property
    def segmentation(self) -> cbase.Segmentation:
        """Copy of the matrix segmentation."""
        return self._segmentation.copy()

    @property
    def zoning(self) -> cbase.ZoningSystem:
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
    def _get_matrix(
        self,
        slice_: "segmentation.SegmentationSlice",  # type: ignore[name-defined]
    ) -> Matrix:
        """Load the data for a single matrix."""
        raise NotImplementedError()

    def get_matrix(
        self, slice_: cbase.segmentation.SegmentationSlice | tuple[int] | dict[str, int]
    ) -> Matrix:
        slice_ = self.validate_slice(slice_)
        return self._get_matrix(slice_)

    @abc.abstractmethod
    def _set_matrix(
        self,
        matrix: pd.DataFrame,
        slice_: "segmentation.SegmentationSlice",  # type: ignore[name-defined]
    ) -> None:
        """Save the data for a single matrix."""
        raise NotImplementedError()

    def set_matrix(
        self,
        matrix: pd.DataFrame,
        slice_: cbase.segmentation.SegmentationSlice | tuple[int] | dict[str, int],
    ):
        slice_ = self.validate_slice(slice_)
        self._set_matrix(matrix, slice_)

    @abc.abstractmethod
    def new(
        self,
        name: str,
        *,
        segmentation_: cbase.Segmentation | None = None,
        zoning: cbase.ZoningSystem | None = None,
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
        # TODO(MB): check this method works correctly with all time periods Segments #25
        # TODO(MB): get information about the time format, e.g. avg hour / period #24
        return self._segmentation.has_time_period_segments()

    @property
    def has_type_segment(self) -> bool:
        """Return True if segmentation contains correct direction segment."""
        segment = self.type.direction_segment.name
        return segment in (i.name for i in self.segmentation.segments)

    def _get_direction_subset(self) -> set[int] | None:
        if not self.has_type_segment:
            return None

        subsets = self._segmentation.input.subsets
        direction = self._type.direction_segment

        if direction.name not in subsets:
            return None

        return set(subsets[direction.name])

    @property
    def is_home_based_only(self) -> bool:
        """Return True if segmentation contains home-based direction only."""
        subset = self._get_direction_subset()

        # TODO(MB): can this hardcoding be removed and instead obtained from caf.base? #24
        # Does subset contain one, or both (OD), of the home-based directions
        return bool(subset is not None and subset <= {1, 2})

    @property
    def is_non_home_based_only(self) -> bool:
        """Return True if segmentation contains non-home-based direction only."""
        subset = self._get_direction_subset()

        # TODO(MB): can this hardcoding be removed and instead obtained from caf.base? #24
        return subset == {0}

    def validate_slice(
        self,
        slice_: "segmentation.SegmentationSlice" | tuple[int] | dict[str, int],  # type: ignore[name-defined]
    ) -> cbase.segmentation.SegmentationSlice:
        """Raise ValueError if slice not present in segmentation."""
        if isinstance(slice_, tuple):
            slice_ = cbase.segmentation.SegmentationSlice.from_tuple(slice_,
                                                                  self.segmentation.naming_order)
        elif isinstance(slice_, dict):
            slice_ = cbase.segmentation.SegmentationSlice(
                slice_, self.segmentation.naming_order
            )
        if slice_ not in self.segmentation.iter_slices():
            raise ValueError(
                f"Given {slice_} not found in segmentation "
                + ", ".join(self.segmentation.names)
            )
        return slice_

    def _validate_zones(self, zones: np.ndarray, name: str) -> None:
        """Raise ValueError if zones doesn't contain only all zone IDs."""
        missing = self.zoning.zone_ids[~np.isin(self.zoning.zone_ids, zones)]
        if len(missing) > 0:
            raise ValueError(
                f"{name} is missing {len(missing):,} zones: {_short_list(list(missing))}"
            )

        extra = zones[~np.isin(zones, self.zoning.zone_ids)]
        if len(extra) > 0:
            raise ValueError(
                f"{name} has {len(extra):,} zones not found in"
                f" zone system ({self.zoning.name}): {_short_list(list(extra))}"
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
    
    def to_omx(self,
               omx_path: pathlib.Path,
               name_conversion: dict[segmentation.SegmentationSlice, str] | None = None) -> None:
        """
        Save a set of matrices to an omx file

        Parameters
        ----------
        omx_path : pathlib.Path
            Path to the omx file to save to.
        name_conversion : dict[segmentation.SegmentationSlice, str] | None, optional
            A dictionary defining names for matrix levels in the omx file, if you don't want to use 
            the names generated from the slices of the segmentation. This also can control the order.
        """
        if name_conversion is None:
            iterator = self.segmentation.iter_slices()
        else:
            iterator = iter(name_conversion)
        with omx.OMXFile(omx_path,
                               mode='w',
                               shape=(len(self.zoning), len(self.zoning)),
                               zones=self.zoning.zone_ids) as omx_file:
            for slice_ in iterator:
                mat = self.get_matrix(slice_)
                if name_conversion is not None:
                    name = name_conversion[slice_]
                else:
                    name = slice_.generate_name()
                omx_file.set_matrix_level(name, mat.data)

    def to_mat(self,
               mat_path: pathlib.Path,
               cube_path: pathlib.Path,
               name_conversion: dict[segmentation.SegmentationSlice,str] | None = None,
               ) -> None:
        omx_path = mat_path.with_suffix('.omx')
        self.to_omx(omx_path,
                          name_conversion=name_conversion)
        converter = cube.CUBEMatConverter(cube_path)
        converter.from_omx(omx_path, mat_path)


    def __repr__(self) -> str:
        """Return a string representation of the matrices."""
        return (
            f"{self.__class__.__name__}(name={self.name}, type={self.type.name},"
            f" segmentation={self.segmentation.names}, zoning={self.zoning.name})"
        )

    def lazy(self) -> _LazyMatricesPipeline:
        """Create a lazy operation pipeline from this matrices object."""
        return _LazyMatricesPipeline(self)

    def aggregate(
        self,
        segmentation_: "segmentation.Segmentation",  # type: ignore[name-defined]
        output_name: str = "{name}-aggregated",
        progress_bar: bool = True,
    ) -> Self:
        """Aggregate matrices to target segmentation.

        Outputs aggregated to a new matrices class.

        Parameters
        ----------
        segmentation_ : cb.segmentation.Segmentation
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

        output = self.new(
            output_name.format(name=self.name), segmentation_=segmentation_
        )

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
    
    def filter_segment_value(self, segment: str, values: list[int] | int, progress_bar: bool=True, keep_segment: bool = True):
        if keep_segment:
            new_seg = self.segmentation.copy()
            new_seg.input.subsets['segment'] = values
            new_seg.reinit()
            iterator = new_seg.iter_slices()
        else:
            if isinstance(values, list):
                raise ValueError("To remove the segment, must filter to a single value.")
            new_seg = self.segmentation.remove_segment(segment)
            iterator = self.segmentation.iter_slices({segment: values})

        new_mats = self.new(name=f"{self.name}_filtered_{segment}",
                                segmentation_=new_seg)

        if progress_bar:
            iterator = tqdm.tqdm(
                iterator,
                total=len(new_seg),
                desc=f"Filtering {self.name}",
            )

        for slice_ in iterator:
            mat = self.get_matrix(slice_).data
            new_mats.set_matrix(mat, slice_.aggregate(new_seg))
        return new_mats
        
    
    def compile_highway(self,
                        occupancies: cbase.DVector,
                        tp_factors: dict[int, int],
                        output_name: str,
                        progress_bar: bool = True):
        """
        Compile matrices to Saturn format.

        Compile from full purpose and direction, to userclass. Also applies occupancy and time 
        period factors to go from full time period, person units to average hour pcu.

        Parameters
        ----------
        occupancies : cbase.DVector
            DVector containing occupancy factors. This should include direction_od, tp and m segments. 
            It can either contain p or userclass, but must contain one or the other.
        tp_factors : dict[int, int]
            Time period factors.
        output_name : str
            Name of output matrix.
        progress_bar : bool
            If True display progress bar for aggregation.

        Returns
        -------
        Self
        """
        new_seg_in = cbase.SegmentationInput(
            enum_segments=['m','userclass','tp'],
            naming_order=['m','userclass','tp'],
            subsets={'m': [3], 'tp': self.segmentation.input.subsets['tp']}
        )
        new_seg = cbase.Segmentation(new_seg_in)

        output = self.new(
            output_name.format(name=self.name), segmentation_=new_seg
        )

        # LOG.info(
        #     "Aggregating %s to segments %s, outputting as %s",
        #     self.name,
        #     ", ".join(segmentation_.names),
        #     output.name,
        # )

        if progress_bar:
            iterator = tqdm.tqdm(
                new_seg.iter_slices(),
                total=len(new_seg),
                desc=f"Aggregating {self.name}",
            )
        else:
            iterator = new_seg.iter_slices()

        for to_slice in iterator:
            total = 0
            # uc = to_slice.data['userclass']
            # p_filter = uc_p[uc].to_list()
            # main_filter = to_slice.data
            # main_filter.pop('userclass')
            iter_seg = self.segmentation.add_segment('userclass')
            for from_slice in iter_seg.iter_slices(to_slice.data):
                from_slice_p = from_slice.remove('userclass')
                mat = self.get_matrix(from_slice_p).data
                if 'p' in occupancies.segmentation.names:
                    mat /= occupancies.get_slice(from_slice)
                else:
                    occ_slice = to_slice.data
                    occ_slice['direction_od'] = from_slice.data['direction_od']
                    mat /= occupancies.get_slice(segmentation.SegmentationSlice(occ_slice))
                if from_slice.data['direction_od'] == 2:
                    mat = mat.T
                total += mat
            total /= tp_factors[to_slice.data['tp']]
            output.set_matrix(total, to_slice)

        return output
    
    def p_to_uc(self,
                output_name: str,
                progress_bar: bool = True):
        """
        Compile matrices to Saturn format.

        Compile from full purpose and direction, to userclass. Also applies occupancy and time 
        period factors to go from full time period, person units to average hour pcu.

        Parameters
        ----------
        occupancies : cb.DVector
            DVector containing occupancy factors. This should include direction_od, tp and m segments. 
            It can either contain p or userclass, but must contain one or the other.
        tp_factors : dict[int, int]
            Time period factors.
        output_name : str
            Name of output matrix.
        progress_bar : bool
            If True display progress bar for aggregation.

        Returns
        -------
        Self
        """
        new_seg = self.segmentation.remove_segment('p').add_segment('userclass')
        # new_seg = cb.Segmentation(new_seg)

        output = self.new(
            output_name.format(name=self.name), segmentation_=new_seg
        )

        if progress_bar:
            iterator = tqdm.tqdm(
                new_seg.iter_slices(),
                total=len(new_seg),
                desc=f"Aggregating {self.name}",
            )
        else:
            iterator = new_seg.iter_slices()

        for to_slice in iterator:
            total = 0
            iter_seg = self.segmentation.add_segment('userclass')
            for from_slice in iter_seg.iter_slices(to_slice.data):
                from_slice_p = from_slice.remove('userclass')
                mat = self.get_matrix(from_slice_p).data
                # # TODO DELETE THIS
                # if from_slice.data['direction_od'] == 2:
                #     mat = mat.T
                total += mat
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
                f"targets should be the same type as aggregate({self.type}) not {targets.type}"
            )

        if self.zoning != targets.zoning:
            raise ValueError(
                f"Zoning systems between aggregate ({self.zoning.name}) and"
                f" target ({targets.zoning.name}) matrices are different"
            )

        disaggregations = self._validate_disaggregations(
            targets, from_segment, to_segment
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
            "Disaggregating %s to additional segments: %s",
            self.name,
            disaggregation_segments,
        )

        if progress_bar:
            iterator = tqdm.tqdm(
                disaggregations.items(),
                desc=f"Disaggregating {self.name}",
                dynamic_ncols=True,
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

    def _validate_disaggregations(
        self,
        targets: "MatricesBase",
        from_segment: segments.Segment | None,
        to_segment: segments.Segment | None,
    ) -> dict["segmentation.SegmentationSlice", list["segmentation.SegmentationSlice"]]:  # type: ignore[name-defined]
        """Calculate and validate disaggregation slices."""
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

        return disaggregations

    def copy(self, **overrides):
        params = {
            "segmentation_": self._segmentation,
            "zoning": self._zoning,
            "type_": self._type,
        }
        params.update(overrides)
        return self.__class__(**params)
    
    def subset(self, seg_subset: cbase.Segmentation | dict[str, list[int]]):
        if isinstance(seg_subset, cbase.Segmentation):
            return self.copy(segmentation_=seg_subset)
        new_seg = self.segmentation.copy()
        for seg, vals in seg_subset.items():
            new_seg.input.subsets[seg] = vals
        return self.copy(segmentation_=new_seg.reinit())
    

    def convert_type(self, new_type: Type[MATRICES]) -> MATRICES:
        """
        Convert from one type inheriting from MatricesBase to another.

        Converts between the different child classes of MatricesBase.
        Parameters
        ----------
        new_type : Self
            The type to convert to. This should be different to the type of Self.

        Returns
        -------
        Self
            Self but with the new type.
        """
        if isinstance(self, new_type):
            return self
        converted = new_type(self.segmentation, self.zoning, self.type)
        for slice_ in self.segmentation.iter_slices():
            matrix = self.get_matrix(slice_)
            converted.set_matrix(matrix.data, slice_)
        return converted

    def to_external(self) -> Self:
        """
        Remove internal to internal trips leaving trips starting and/or finishing in external zones.

        Returns
        -------
        Self
            External matrices
        """
        external = self.new(name=f"{self.name}_external")
        internals = self.zoning.internal
        for slice_ in self.segmentation.iter_slices():
            matrix = self.get_matrix(slice_).data.copy()
            matrix.loc[internals, internals] = 0
            external.set_matrix(matrix, slice_)
        return external
    
    def to_internal(self) -> Self:
        """
        Remove trips starting and/or finishing in external zones, leaving internal to internal trips.

        Returns
        -------
        Self
            Internal matrices
        """
        internal = self.new(name=f"{self.name}_internal")
        externals = self.zoning.external
        for slice_ in self.segmentation.iter_slices():
            matrix = self.get_matrix(slice_).data.copy()
            matrix.loc[externals] = 0
            matrix.loc[:, externals] = 0
            internal.set_matrix(matrix, slice_)
        return internal

    def remove_intras(self) -> Self:
        """
        Remove intrazonals (i.e. the leading diagonal) from the matrix.

        Returns
        -------
        Self
            Self but with the leading diagonal set to zero.
        """
        inters = self.new(name=f"{self.name}_nointras")
        for slice_ in self.segmentation.iter_slices():
            matrix = self.get_matrix(slice_).data.copy()
            np.fill_diagonal(matrix.values, 0)
            inters.set_matrix(matrix, slice_)
        return inters

    def intras(self) -> cbase.DVector:
        """
        Return intrazonal demand as a DVector.

        Returns
        -------
        cbase.DVector
            The intrazonal demand (i.e. leading diagonal) of the matrix.
        """
        data: dict[tuple[int, ...], pd.Series] = {}
        for slice_ in self.segmentation.iter_slices():
            matrix = self.get_matrix(slice_).data
            intras = np.diagonal(matrix)
            data[slice_.as_tuple()] = pd.Series(intras, index=matrix.index)
        dvec_data = pd.concat(data, axis=1).T
        dvec_data.index.names = self.segmentation.naming_order
        return cbase.DVector(
            segmentation=self.segmentation,
            import_data=dvec_data,
            zoning_system=self.zoning,
        )

    def translate_zoning(
        self,
        new_zoning: cbase.ZoningSystem,
        translation: pd.DataFrame | ctk.translation.ZoneCorrespondence | None = None,
    ) -> Self:
        """
        Translate zoning of matrices.

        Parameters
        ----------
        new_zoning : cbase.ZoningSystem
            The zoning system to translate to.
        translation : pd.DataFrame | ctk.translation.ZoneCorrespondence | None = None
            The translation vector to use. If this is not provided, an attempt will be made to find it
            using the ZoningSystem.translate method.

        Returns
        -------
        Self
            The input matrices in the new zone system.
        """
        if translation is None:
            translation = self.zoning.translate(new_zoning)

        elif isinstance(translation, pd.DataFrame):
            translation = ctk.translation.ZoneCorrespondence(
                translation,
                self.zoning.column_name,
                new_zoning.column_name,
                self.zoning.translation_column_name(new_zoning),
            )
        translated = self.new(name=f"{self.name}_{new_zoning.name}", zoning=new_zoning)
        for slice_ in self.segmentation.iter_slices():
            matrix = self.get_matrix(slice_)
            translated_matrix = ctk.translation.pandas_matrix_zone_translation(
                matrix.data, translation
            )
            translated.set_matrix(translated_matrix, slice_)
        return translated

    def to_dvector(self) -> dict[str, cbase.DVector]:
        """
        Convert matrices to two DVectors, being sums over rows and columns.

        Returns
        -------
        dict[str, cbase.DVector]
            A dictionary of either p/a or o/d to DVectors, depending on the matrices' type.
        """
        rows = {}
        cols = {}
        for slice_ in self.segmentation.iter_slices():
            mat = self.get_matrix(slice_).data
            column = mat.sum(axis=0)
            row = mat.sum(axis=1)
            rows[slice_.as_tuple()] = row
            cols[slice_.as_tuple()] = column
        rows = pd.concat(rows, axis=1).T
        rows.index.names = self.segmentation.naming_order
        rows = cbase.DVector(
            import_data=rows, segmentation=self.segmentation, zoning_system=self.zoning
        )

        cols = pd.concat(cols, axis=1).T
        cols.index.names = self.segmentation.naming_order
        cols = cbase.DVector(
            import_data=cols, segmentation=self.segmentation, zoning_system=self.zoning
        )
        if self.type.name == "PA":
            return {"P": rows, "A": cols}
        return {"O": rows, "D": cols}

    def _generic_dunder(
        self,
        other: MATRICES | cbase.DVector | Real,
        mat_method,
        method_name,
    ):
        """
        Stop telling me to use the imperative mood pydocstyle.

        A generic dunder method which is called by each of the dunder methods.

        Parameters
        ----------
        other: MatricesBase
            Other instance of class.
        method:
            A pd.DataFrame method to be called slice by slice on self.data with other.data.
        method_name: str
            The name of the method used in naming the return object.
        """

        # if self.segmentation != other.segmentation:
        #     raise SegmentationError("Segmentations don't match.")
        # if self.zoning != other.zoning:
        #     raise ZoningError("Zoning systems don't match.")
        if isinstance(other, MatricesBase):
            if self.segmentation == other.segmentation:
                out = self.new(name=f"{self.name}_{method_name}_{other.name}")
            elif self.segmentation.overlap(other.segmentation) == set(self.segmentation.names):
                out = self.new(name=f"{self.name}_{method_name}_{other.name}", segmentation_=other.segmentation)
                for slice_ in self.segmentation.iter_slices():
                    for inner_slice in other.segmentation.iter_slices(slice_.data):
                        product = mat_method(
                            self.get_matrix(slice_).data, other.get_matrix(inner_slice).data
                        )
                        out.set_matrix(product, inner_slice)
                return out
        else:
            out = self.new(name=f"{self.name}_{method_name}_other")
        for slice_ in self.segmentation.iter_slices():
            if isinstance(other, MatricesBase):
                if self.segmentation == other.segmentation:
                    product = mat_method(
                        self.get_matrix(slice_).data, other.get_matrix(slice_).data
                    )
            elif isinstance(other, cbase.DVector):
                other_slice = other.get_slice(
                    slice_.aggregate(other.segmentation.naming_order)
                )
                if isinstance(other_slice, pd.Series):
                    product = mat_method(
                        self.get_matrix(slice_).data,
                        other_slice.squeeze(),
                    )
                else:
                    product = mat_method(self.get_matrix(slice_).data, other_slice)
            else:
                product = mat_method(self.get_matrix(slice_).data, other)
            out.set_matrix(product.fillna(0), slice_)
        return out

    def __truediv__(self, other):
        return self._generic_dunder(
            other, pd.DataFrame.__truediv__, "divide"
        )

    def __mul__(self, other):
        return self._generic_dunder(
            other, pd.DataFrame.__mul__, "multiply"
        )


def _short_list(values: collections.abc.Sequence, length: int = 10) -> str:
    if len(values) <= length:
        return ", ".join(map(str, values))

    half = length // 2
    return (
        ", ".join(map(str, values[:half])) + "..." + ", ".join(map(str, values[-half:]))
    )


def _get_disaggregation_translation(
    from_segmentation: segmentation.Segmentation,
    from_segment: segments.Segment,
    to_segment: segments.Segment,
    target_segmentation: segmentation.Segmentation,
) -> dict[segmentation.SegmentationSlice, list[segmentation.SegmentationSlice]]:
    """Produce slice disaggregations with a single segment lookup."""
    try:
        to_segmentation, lookup = from_segmentation.translate_segment(
            from_segment, to_segment
        )
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

    groupings: dict[int, list[int]] = (
        lookup.groupby(level=0).agg(list).squeeze().to_dict()
    )

    _validate_disaggregation_translation(from_segment, to_segment, groupings)

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


def _validate_disaggregation_translation(
    from_segment: segments.Segment,
    to_segment: segments.Segment,
    groupings: dict[int, list[int]],
) -> None:
    """Validate lookup groups used for disaggregation translation.

    Raises
    ------
    ValueError
        If any source segment maps to no target segments.
    """
    for from_seg, to_segs in groupings.items():
        if len(to_segs) == 0:
            raise ValueError(
                f"{from_segment.name} segment {from_seg} has no {to_segment.name}"
                " segments in lookup"
            )


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
) -> None:
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

        try:
            existing_matrix = output.get_matrix(matrix.slice)
            existing = existing_matrix.data
            if existing.isna().all().all():
                existing = None
        except (KeyError, FileNotFoundError):
            existing = None

        if existing is not None:
            disaggregated = disaggregated + existing

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
        segmentation_: cbase.Segmentation,
        zoning: cbase.ZoningSystem,
        type_: MatrixType,
        matrices: list[Matrix] | None = None,
        *,
        name: str | None = None,
        mutable: bool = True
    ) -> None:
        super().__init__(segmentation_, zoning, type_)
        self._matrices: dict[segmentation.SegmentationSlice, pd.DataFrame] = {}

        if matrices is not None:
            for matrix in matrices:
                self.set_matrix(matrix.data, matrix.slice)

        if name is None:
            name = "In-memory matrices"
        self._name = name
        self.mutable = mutable

    @property
    def name(self) -> str:
        """Name of the matrices."""
        return self._name

    def new(  # noqa: D102
        self,
        name: str,
        *,
        segmentation_: cbase.Segmentation | None = None,
        zoning: cbase.ZoningSystem | None = None,
        type_: MatrixType | None = None,
    ) -> "MemoryMatrices":
        return MemoryMatrices(
            segmentation_=self._segmentation
            if segmentation_ is None
            else segmentation_,
            zoning=self._zoning if zoning is None else zoning,
            type_=self._type if type_ is None else type_,
            name=name,
        )

    def copy(self, **overrides) -> "MemoryMatrices":
        params = {
            "segmentation_": self._segmentation,
            "zoning": self._zoning,
            "type_": self._type,
            "matrices": [
                Matrix(matrix.copy(), slice_)
                for slice_, matrix in self._matrices.items()
            ],
            "name": self._name,
            "mutable": False
        }
        params.update(overrides)
        return self.__class__(**params)

    def save(self, folder: pathlib.Path, name: str | None = None, format: Literal['long', 'wide', 'cube']='wide'):
        for slice_ in self.segmentation.iter_slices():
            mat = self.get_matrix(slice_)
            if name is None:
                name = self.name
            if format == 'cube':
                mat.data.stack().reset_index().to_csv(folder / f"{name}_{slice_.generate_name()}.csv", index=False, header=False)
            elif format == 'long':
                mat.data.stack().reset_index().to_csv(folder / f"{name}_{slice_.generate_name()}.csv")
            else:
                mat.data.to_csv(folder / f"{name}_{slice_.generate_name()}.csv")

    def exists(self) -> bool:  # noqa: D102
        return all(slice_ in self._matrices for slice_ in self.segmentation.iter_slices())

    def _get_matrix(self, slice_: segmentation.SegmentationSlice) -> Matrix:
        """Get in-memory matrix."""

        if slice_ not in self._matrices:
            raise KeyError(f"no matrix found for {slice_}")
        return Matrix(self._matrices[slice_], slice_)

    def _set_matrix(
        self, matrix: pd.DataFrame, slice_: segmentation.SegmentationSlice
    ) -> None:
        """Store matrix in class (in-memory)."""
        if not self.mutable:
            raise PermissionError("This set of matrices is immutable. This is most likely because it is a copy, and so can't be altered.")
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
        Template for the filenames of individual matrix CSVs,
        if None defaults to "{type}_{slice_name}".
        Template will be infilled with the following:
        - {type} - name of the matrix `type_` e.g. "OD";
        - {slice_name} - the name of the individual slice e.g. "p1_m3_nhb"
          from :class:`SegmentationSlice`.
    check_files
        If True (default) raises an error if CSV files don't already
        exist for all slices in segmentation.

    .. todo::
        Add support for zipped folders.
    """

    _file_suffixes: tuple[str, ...] = (".csv.bz2", ".csv")
    _default_filename_template = "{type}_{slice_name}"

    def __init__(
        self,
        segmentation_: cbase.Segmentation,
        zoning: cbase.ZoningSystem,
        type_: MatrixType,
        folder: pathlib.Path,
        *,
        filename_template: str | None = None,
        check_files: bool = True,
        mutable: bool = True
    ) -> None:
        super().__init__(segmentation_, zoning, type_)

        if filename_template is None:
            filename_template = self._default_filename_template

        self._raw_filename_template = filename_template
        self._filename_template = filename_template.format(
            type=self.type.name, slice_name="{slice_name}"
        )
        self._folder = folder.resolve()
        self._mutable = mutable

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
            self._filenames[slice_] = self._filename_template.format(
                slice_name=slice_name
            )

        return self._filenames[slice_]

    def _set_matrix(
        self, matrix: pd.DataFrame, slice_: segmentation.SegmentationSlice
    ) -> None:
        """Save the matrix to a CSV, with a filename based on the slice parameters."""
        if not self._mutable:
            raise PermissionError("This set of matrices is immutable. This is most likely because it is a copy, and so can't be altered.")
        filename = self._get_filename(slice_)
        self.validate_matrix(matrix, filename)

        path = self._folder / (filename + self._file_suffixes[0])
        matrix.to_csv(path)
        LOG.debug("Written: %s", path)

    def _get_matrix(self, slice_: segmentation.SegmentationSlice) -> Matrix:
        """Load the matrix from a CSV."""
        filename = self._get_filename(slice_)
        path = ctk.io.find_file_with_name(self._folder, filename, self._file_suffixes)

        LOG.debug("Loading matrix file: %s", path)
        data = ctk.io.read_csv_matrix(path)
        data.index.name = "origin"
        data.columns.name = "destination"

        self.validate_matrix(data, filename)
        return Matrix(data, slice_)

    def new(  # noqa: D102
        self,
        name: str,
        *,
        segmentation_: cbase.Segmentation | None = None,
        zoning: cbase.ZoningSystem | None = None,
        type_: MatrixType | None = None,
    ) -> Self:
        folder = self._folder.with_name(name)
        folder.mkdir(exist_ok=True)
        return self.__class__(
            segmentation_=self._segmentation
            if segmentation_ is None
            else segmentation_,
            zoning=self._zoning if zoning is None else zoning,
            type_=self._type if type_ is None else type_,
            folder=folder,
            filename_template=self._raw_filename_template,
            check_files=False,
        )
    
    def copy(self, **overrides) -> "MatrixFiles":
        params = {
            "segmentation_": self._segmentation,
            "zoning": self._zoning,
            "type_": self._type,
            "folder": self._folder,
            "filename_template": self._raw_filename_template,
            "check_files": False,
            "_mutable": False
        }
        params.update(overrides)
        return self.__class__(**params)

    def exists(self) -> bool:  # noqa: D102
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

    def to_hdf(
        self,
        path: pathlib.Path,
        *,
        key_template: str | None = None,
        complevel: int = 9,
        complib: str = "zlib",
    ) -> "HDFMatrices":
        """Convert CSV matrices to HDF matrices."""
        output = HDFMatrices(
            segmentation_=self._segmentation,
            zoning=self._zoning,
            type_=self._type,
            path=path,
            key_template=key_template,
            check_file=False,
            complevel=complevel,
            complib=complib,
        )
        for matrix in self:
            output.set_matrix(matrix.data, matrix.slice)
        return output


class HDFMatrices(MatricesBase):
    """Handle matrices stored in a single HDF file using OMX-compatible layout.

    Parameters
    ----------
    segmentation_
        Segmentation for the matrices.
    zoning
        ZoningSystem for all the matrices.
    type_
        Type of the matrices.
    path
        Path to HDF file for storing matrices in.
    key_template
        Template for HDF dataset keys of individual matrices,
        if None defaults to "{type}_{slice_name}".
        Template will be infilled with the following:
        - {type} - name of the matrix `type_` e.g. "OD";
        - {slice_name} - the name of the individual slice e.g. "p1_m3_nhb"
          from :class:`SegmentationSlice`.
    check_file
        If True (default) raises an error if the HDF file doesn't already
        exist with datasets for all slices in segmentation.
    complevel
        Compression level used when writing matrices.
    complib
        Compression library used when writing matrices.
    """

    _default_key_template = "{type}_{slice_name}"
    _default_save_suffix = ".dmat"
    _metadata_key = "__caf_mat_metadata__"
    _metadata_version = 1

    def __init__(
        self,
        segmentation_: cbase.Segmentation,
        zoning: cbase.ZoningSystem,
        type_: MatrixType,
        path: pathlib.Path,
        *,
        key_template: str | None = None,
        check_file: bool = True,
        complevel: int = 9,
        complib: str = "zlib",
    ) -> None:
        super().__init__(segmentation_, zoning, type_)

        if key_template is None:
            key_template = self._default_key_template

        self._raw_key_template = key_template
        self._key_template = key_template.format(
            type=self.type.name, slice_name="{slice_name}"
        )
        self._path = path.resolve()
        self._keys: dict[segmentation.SegmentationSlice, str] = {}
        self._complevel = complevel
        self._complib = complib

        self._path.parent.mkdir(parents=True, exist_ok=True)

        if check_file and not self._path.is_file():
            raise FileNotFoundError(path)

        if check_file:
            self._validate_omx_compatibility()
            with omx.OMXFile(self._path, mode="r") as file:
                levels = set(file.matrix_levels)
                for slice_ in self._segmentation.iter_slices():
                    key = self._get_key(slice_)
                    if key not in levels:
                        raise FileNotFoundError(
                            f"{self._path} does not contain matrix key: {key}"
                        )

    @property
    def name(self) -> str:
        """Name of the matrices."""
        return self._path.stem

    @property
    def path(self) -> pathlib.Path:
        """Path to HDF file containing the matrices."""
        return self._path

    def _get_key(self, slice_: segmentation.SegmentationSlice) -> str:
        """Get HDF key for given slice, generates it if not already present."""
        if slice_ not in self._keys:
            slice_name = self._segmentation.generate_slice_name(slice_)
            self._keys[slice_] = self._key_template.format(slice_name=slice_name)
        return self._keys[slice_]

    @classmethod
    def _normalise_path(cls, path: pathlib.Path, *, add_default_suffix: bool) -> pathlib.Path:
        """Return a normalised absolute path, with optional default suffix."""
        if add_default_suffix and "." not in path.name:
            path = path.with_suffix(cls._default_save_suffix)
        return path.resolve()

    @property
    def _omx_shape(self) -> tuple[int, int]:
        """Expected OMX matrix shape from zoning."""
        size = len(self._zoning.zone_ids)
        return (size, size)

    def _ensure_omx_file(self) -> None:
        """Create OMX file and required structure if it doesn't exist."""
        if self._path.is_file():
            return

        with omx.OMXFile(
            self._path,
            mode="w",
            omx_version="0.2",
            shape=self._omx_shape,
        ) as file:
            file.zones = self._zoning.zone_ids

    def _validate_omx_compatibility(self) -> None:
        """Raise ValueError if file doesn't look like expected OMX-compatible storage."""
        with omx.OMXFile(self._path, mode="r") as file:
            if file.shape != self._omx_shape:
                expected_shape = self._omx_shape
                raise ValueError(
                    f"{self._path} shape {file.shape} does not match zoning shape "
                    f"{expected_shape}"
                )
            if not np.array_equal(np.sort(file.zones), np.sort(self._zoning.zone_ids)):
                raise ValueError(f"{self._path} zones don't match zoning system")

    def _metadata_frame(self) -> pd.DataFrame:
        """Build metadata required for path-only loading."""
        return pd.DataFrame(
            [
                {
                    "version": self._metadata_version,
                    "matrix_type": self._type.name,
                    "key_template": self._raw_key_template,
                }
            ]
        )

    def _write_metadata(self, path: pathlib.Path) -> None:
        """Write metadata, segmentation and zoning to file."""
        self._metadata_frame().to_hdf(path, key=self._metadata_key, mode="a", format="fixed")
        zoning_key = f"zoning_{self._zoning.name}"
        zoning_meta_key = f"zoning_meta_{self._zoning.name}"
        with h5py.File(path, "a") as file:
            has_segmentation = "segmentation" in file
            has_zoning = zoning_key in file
            has_zoning_meta = zoning_meta_key in file

        if not has_segmentation:
            self._segmentation.save(path, "hdf")

        if not has_zoning:
            self._zoning.zones_data.reset_index().to_hdf(path, key=zoning_key, mode="a")

        if not has_zoning_meta:
            with h5py.File(path, "a") as file:
                file.create_dataset(
                    zoning_meta_key,
                    data=self._zoning.metadata.to_yaml().encode("utf-8"),
                )

    def set_matrix(self, matrix: pd.DataFrame, slice_: segmentation.SegmentationSlice) -> None:
        """Save the matrix to HDF, with a key based on the slice parameters."""
        self.validate_slice(slice_)
        self._ensure_omx_file()

        key = self._get_key(slice_)
        self.validate_matrix(matrix, key)

        with omx.OMXFile(
            self._path,
            mode="a",
            filters=tables.Filters(
                complevel=self._complevel,
                complib=self._complib,
                shuffle=False,
            ),
        ) as file:
            try:
                file.get_node("/data", key)
            except tables.NoSuchNodeError:
                pass
            else:
                file.remove_node("/data", key)
            file.set_matrix_level(key, matrix)
        self._write_metadata(self._path)
        LOG.debug("Written %s matrix to %s", key, self._path)

    def get_matrix(self, slice_: segmentation.SegmentationSlice) -> Matrix:
        """Load the matrix from HDF."""
        self.validate_slice(slice_)
        key = self._get_key(slice_)
        with omx.OMXFile(self._path, mode="r") as file:
            if key not in file.matrix_levels:
                raise KeyError(f"no matrix found for {slice_}")
            data = file.get_matrix_level(key)

        data.index.name = "origin"
        data.columns.name = "destination"
        self.validate_matrix(data, key)
        return Matrix(data, slice_)

    def new(  # noqa: D102
        self,
        name: str,
        *,
        segmentation_: cbase.Segmentation | None = None,
        zoning: cbase.ZoningSystem | None = None,
        type_: MatrixType | None = None,
    ) -> Self:
        if pathlib.Path(name).suffix:
            path = self._path.with_name(name)
        else:
            suffix = self._path.suffix or ".h5"
            path = self._path.with_name(f"{name}{suffix}")

        return self.__class__(
            segmentation_=self._segmentation if segmentation_ is None else segmentation_,
            zoning=self._zoning if zoning is None else zoning,
            type_=self._type if type_ is None else type_,
            path=path,
            key_template=self._raw_key_template,
            check_file=False,
            complevel=self._complevel,
            complib=self._complib,
        )

    def exists(self) -> bool:  # noqa: D102
        if not self._path.is_file():
            return False

        try:
            with omx.OMXFile(self._path, mode="r") as file:
                levels = set(file.matrix_levels)
                for slice_ in self._segmentation.iter_slices():
                    key = self._get_key(slice_)
                    if key not in levels:
                        return False
        except (OSError, ValueError, KeyError):
            return False
        return True

    def save(self, path: pathlib.Path) -> pathlib.Path:
        """Save matrices, segmentation and zoning to a single HDF file."""
        path = self._normalise_path(path, add_default_suffix=True)
        path.parent.mkdir(parents=True, exist_ok=True)

        with omx.OMXFile(
            path,
            mode="w",
            omx_version="0.2",
            shape=self._omx_shape,
            filters=tables.Filters(
                complevel=self._complevel,
                complib=self._complib,
                shuffle=False,
            ),
        ) as file:
            file.zones = self._zoning.zone_ids
            for matrix in self:
                key = self._get_key(matrix.slice)
                file.set_matrix_level(key, matrix.data)

        self._metadata_frame().to_hdf(path, key=self._metadata_key, mode="a", format="fixed")
        self._segmentation.save(path, "hdf")
        self._zoning.save(path, "hdf")
        return path

    @classmethod
    def load(cls, path: pathlib.Path) -> "HDFMatrices":
        """Load matrices and metadata from a single HDF file path."""
        path = cls._normalise_path(path, add_default_suffix=True)
        if not path.is_file():
            raise FileNotFoundError(path)

        try:
            metadata = pd.read_hdf(path, key=cls._metadata_key, mode="r")
        except KeyError as exc:
            raise ValueError(f"{path} does not contain HDFMatrices metadata") from exc

        if len(metadata) != 1:
            raise ValueError(f"expected 1 metadata row in {path} not {len(metadata)}")

        row = metadata.iloc[0]
        try:
            type_ = MatrixType[str(row["matrix_type"])]
        except KeyError as exc:
            raise ValueError(f"invalid matrix type in metadata: {row['matrix_type']}") from exc

        key_template = str(row["key_template"])
        segmentation_ = cbase.Segmentation.load(path, "hdf")
        zoning = cbase.ZoningSystem.load(path, "hdf")
        if not isinstance(zoning, cbase.ZoningSystem):
            raise TypeError(f"expected one zoning system in {path} not {type(zoning)}")
        loaded = cls(
            segmentation_=segmentation_,
            zoning=zoning,
            type_=type_,
            path=path,
            key_template=key_template,
            check_file=True,
        )
        loaded._validate_omx_compatibility()
        return loaded

    def to_mat(
        self,
        voyager_path: pathlib.Path,
        out_path: pathlib.Path | None = None,
        *,
        overwrite: bool = False,
    ) -> pathlib.Path:
        """Convert HDF matrices to a CUBE .mat file."""
        converter = cube.CUBEMatConverter(voyager_path)
        return converter.from_omx(self._path, out_path, overwrite=overwrite)

    def to_csv(
        self, folder: pathlib.Path, *, filename_template: str | None = None
    ) -> MatrixFiles:
        """Convert HDF matrices to CSV matrix files."""
        output = MatrixFiles(
            segmentation_=self._segmentation,
            zoning=self._zoning,
            type_=self._type,
            folder=folder,
            filename_template=filename_template,
            check_files=False,
        )
        for matrix in self:
            output.set_matrix(matrix.data, matrix.slice)
        return output


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
        segmentation_: cbase.Segmentation,
        zoning: cbase.ZoningSystem,
        type_: MatrixType,
        *,
        data: pd.DataFrame | None = None,
        name: str = "LongMatrix",
        columns: list[str] | None = None,
    ) -> None:
        super().__init__(segmentation_, zoning, type_)
        self._name = name
        self._index = self._get_index_names(segmentation_)

        if columns is not None and len(columns) == 0:
            raise ValueError("empty list given for columns")

        if data is None:
            self._data, columns = self._create_empty_data(columns)
        else:
            self._data, columns = self._validate_data(data)
        self._columns = columns

    @classmethod
    def _get_index_names(cls, segmentation_: segmentation.Segmentation) -> list[str]:
        return [*segmentation_.naming_order, cls._origin_column, cls._dest_column]

    @property
    def name(self) -> str:  # noqa: D102
        return self._name

    def _create_empty_data(
        self,
        columns: list[str] | None = None,
    ) -> tuple[pd.DataFrame, list[str]]:
        """Create DataFrame of NaNs with correct indices."""
        data = pd.DataFrame(
            -1,
            index=self._segmentation.ind(),
            columns=pd.MultiIndex.from_product(
                [self._zoning.zone_ids] * 2,
                names=[self._origin_column, self._dest_column],
            ),
            dtype=float,
        )

        if columns is None:
            columns = ["trips"]

        data = (
            data.stack(self._origin_column, future_stack=True)
            .stack(self._dest_column, future_stack=True)
            .to_frame(name=columns[0])
        )
        data.loc[:] = np.nan

        if len(columns) > 1:
            for i in columns:
                data[i] = np.nan

        return data, columns

    def _validate_index(self, data: pd.DataFrame) -> pd.DataFrame:
        """Validate the data has the correct indices."""
        index = set(self._index)
        if set(data.index.names) != index:
            # If single index assume segmentation indices are columns
            if not (
                isinstance(data.index, pd.Index)
                and index <= set(data.columns.to_list())
            ):
                raise ValueError(
                    f"expected indices {self._index} not index "
                    f"({data.index.names}) or columns ({data.columns.to_list()})"
                )

            data = data.set_index(self._index)

        if not isinstance(data.index, pd.MultiIndex):
            raise TypeError(f"data should have a MultiIndex not {type(data.index)}")

        if not data.index.dtypes.apply(pd.api.types.is_integer_dtype).all():
            try:
                data.index = pd.MultiIndex.from_arrays(
                    [
                        data.index.get_level_values(i).astype(int)
                        for i in data.index.names
                    ]
                )
            except ValueError as exc:
                raise ValueError(
                    f"indices should be integers not {data.index.dtype.name}"
                ) from exc

        if data.index.has_duplicates:
            raise ValueError(f"duplicate indices found in {self.name}")
        return data

    def _validate_data(
        self, data: pd.DataFrame, columns: list[str] | None = None
    ) -> tuple[pd.DataFrame, list[str]]:
        """Validate the data has correct indices and columns."""
        data = self._validate_index(data)

        try:
            data = data.astype(float)
        except ValueError as exc:
            raise ValueError(
                f"matrix data should be numeric not {data.dtypes.to_list()}"
            ) from exc

        if columns is None:
            columns = data.columns.to_list()
        elif set(data.columns.to_list()) != set(columns):
            raise ValueError(
                f"expected columns {columns} but given {data.columns.to_list()}"
            )

        seg_data = data.reset_index()[self.segmentation.naming_order].drop_duplicates(
            keep="first"
        )
        self.segmentation.validate_segmentation(seg_data, self.segmentation)

        for i in (self._origin_column, self._dest_column):
            self._validate_zones(
                data.index.get_level_values(i).unique().to_numpy(), f"{self.name} - {i}"
            )

        return data, columns

    def to_frame(self, deep: bool = False) -> pd.DataFrame:
        """Return a copy of the underlying DataFrame."""
        return self._data.copy(deep)

    def get_matrix(self, slice_: segmentation.SegmentationSlice) -> Matrix | pd.DataFrame:  # noqa: D102
        self.validate_slice(slice_)
        data: pd.DataFrame = self._data.loc[slice_.as_tuple()].copy()
        if len(data.columns) > 1:
            return data

        data = data.squeeze()
        data.index.names = [None, None]
        data.name = None
        return Matrix(data.unstack(), slice_)

    def set_matrix(self, matrix: pd.DataFrame, slice_: segmentation.SegmentationSlice) -> None:  # noqa: D102
        self.validate_slice(slice_)

        if matrix.index.nlevels == 1:
            self.validate_matrix(matrix, str(slice_))
            matrix.index.name = self._origin_column
            matrix.columns.name = self._dest_column

            if self._columns is None:
                raise TypeError("columns should be defined")

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
            [(*slice_.as_tuple(), i, j) for i, j in matrix.index]
        )

        # Cannot set multiple rows with index directly,
        # so instead concat new columns and update
        matrix.columns = [f"{i}_set" for i in matrix.columns]
        updated = pd.concat([self._data, matrix], axis=1)
        updated.index.names = self._data.index.names
        for col in self._data.columns:
            updated[col] = np.where(
                updated[f"{col}_set"].isna(), updated[col], updated[f"{col}_set"]
            )

        self._data = updated[self._data.columns]

    def new(  # noqa: D102
        self,
        name: str,
        *,
        segmentation_: cbase.Segmentation | None = None,
        zoning: cbase.ZoningSystem | None = None,
        type_: MatrixType | None = None,
    ) -> "LongMatrices":
        return LongMatrices(
            segmentation_=self.segmentation if segmentation_ is None else segmentation_,
            zoning=self.zoning if zoning is None else zoning,
            type_=self.type if type_ is None else type_,
            name=name,
            columns=self._columns,
        )

    def exists(self) -> bool:  # noqa: D102
        try:
            self._validate_data(self._data)
        except ValueError:
            return False

        return not self._data.isna().any().any()

    @classmethod
    def from_csv(
        cls,
        segmentation_: cbase.Segmentation,
        zoning: cbase.ZoningSystem,
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
