# -*- coding: utf-8 -*-
"""Tests for `matrices` module."""

##### IMPORTS #####

# Built-Ins
import dataclasses
import functools
import pathlib
import warnings
from typing import Generator

# Third Party
import caf.base as base
import numpy as np
import pandas as pd
import pytest
from caf.base import segmentation, zoning

# Local Imports
from caf.mat import matrices

##### CONSTANTS #####


@pytest.fixture(name="zone_system")
def fix_zone_system() -> base.ZoningSystem:
    """Test zoning system with 3 zones."""
    return base.ZoningSystem(
        "test",
        pd.DataFrame({"zone_id": [1, 2, 3]}),
        zoning.ZoningSystemMetaData(name="test"),
    )


def random_matrices(
    zones: base.ZoningSystem, max_: int = 100, seed: int = 1
) -> Generator[pd.DataFrame, None, None]:
    """Generate random square matrices with given zones."""
    rng = np.random.default_rng(seed)

    while True:
        yield pd.DataFrame(
            np.round(rng.random((len(zones), len(zones))) * max_, 4),
            columns=zones.zone_ids,
            index=zones.zone_ids,
        )


##### Tests & Fixtures for `LongMatrices` #####


@pytest.fixture(name="long_segmentation")
def fix_long_segmentation() -> segmentation.Segmentation:
    """Segmentation for LongMatrices tests."""
    config = segmentation.SegmentationInput(
        enum_segments=["p", "tp", "direction"],  # type: ignore
        naming_order=["p", "tp", "direction"],
        subsets={"p": [1, 2], "direction": [1], "tp": [1, 3]},
    )
    return segmentation.Segmentation(config)


@dataclasses.dataclass
class LongMatricesData:
    """Result dataclass for LongMatrices tests."""

    data: pd.DataFrame
    segmentation: segmentation.Segmentation
    zone_system: zoning.ZoningSystem


@pytest.fixture(name="long_data")
def fix_long_data(
    zone_system: base.ZoningSystem, long_segmentation: base.Segmentation
) -> LongMatricesData:
    """Single column of data for LongMatrices tests."""
    zones = pd.DataFrame(
        {
            "origin": np.repeat(
                np.repeat(zone_system.zone_ids, len(zone_system)), len(long_segmentation)
            ),
            "destination": np.repeat(
                np.tile(zone_system.zone_ids, len(zone_system)), len(long_segmentation)
            ),
        }
    )
    zones = pd.DataFrame(
        {
            "origin": np.tile(
                np.repeat(zone_system.zone_ids, len(zone_system)), len(long_segmentation)
            ),
            "destination": np.tile(
                np.tile(zone_system.zone_ids, len(zone_system)), len(long_segmentation)
            ),
        }
    )

    indices = long_segmentation.ind()
    seg_data = pd.DataFrame(
        {
            i: np.repeat(indices.get_level_values(i), len(zone_system) ** 2)
            for i in indices.names
        }
    )

    index_data = pd.concat([seg_data, zones], axis=1)

    rng = np.random.default_rng(1)
    data = {"trips": rng.random(len(index_data)) * 100}

    return LongMatricesData(
        pd.DataFrame(data, index=pd.MultiIndex.from_frame(index_data)),
        long_segmentation,
        zone_system,
    )


@pytest.fixture(name="long_data_csv")
def fix_long_data_csv(
    tmp_path: pathlib.Path, long_data: LongMatricesData
) -> tuple[pathlib.Path, LongMatricesData]:
    """CSV containing data for LongMatrices tests."""
    path = tmp_path / "test.csv"
    long_data.data.to_csv(path)
    return path, long_data


@pytest.fixture(name="long_matrices")
def fix_long_matrices(
    long_data: LongMatricesData,
) -> tuple[matrices.LongMatrices, LongMatricesData]:
    """Instance of LongMatrices with single column for testing."""
    long = matrices.LongMatrices(
        long_data.segmentation,
        long_data.zone_system,
        matrices.MatrixType.PA,
        data=long_data.data.reset_index(),
        columns=long_data.data.columns.to_list(),
    )

    return long, long_data


class TestLongMatrices:
    """Tests for abstract methods implemented in :class:`LongMatrices`.

    See Also
    --------
    TestMemoryMatrices
        for tests on the methods implemented in :class:`MatricesBase`.
    """

    def test_init(self, long_data: LongMatricesData):
        """Test initialising LongMatrices class with a dataframe."""
        answer = matrices.LongMatrices(
            long_data.segmentation,
            long_data.zone_system,
            matrices.MatrixType.PA,
            data=long_data.data.reset_index(),
            columns=long_data.data.columns.to_list(),
        )

        pd.testing.assert_frame_equal(long_data.data, answer._data)

    def test_from_csv(self, long_data_csv: tuple[pathlib.Path, LongMatricesData]) -> None:
        """Test loading data from a CSV with `from_csv`."""
        path, long_matrices = long_data_csv
        answer = matrices.LongMatrices.from_csv(
            long_matrices.segmentation,
            long_matrices.zone_system,
            matrices.MatrixType.PA,
            path,
            columns=long_matrices.data.columns.to_list(),
        )

        pd.testing.assert_frame_equal(long_matrices.data, answer._data)


##### Common tests for matrices classes #####


def random_matrices_data(
    zone_system: zoning.ZoningSystem,
    segmentation_: segmentation.Segmentation,
    max_: int = 100,
    seed: int = 1,
) -> dict[segmentation.SegmentationSlice, pd.DataFrame]:
    """Produce random matrices for all slices in segmentation."""
    rand = random_matrices(zone_system, max_, seed)

    data = {}
    for slice_ in segmentation_.iter_slices():
        data[slice_] = next(rand)

    return data


Matrices = matrices.MemoryMatrices | matrices.MatrixFiles | matrices.LongMatrices


@dataclasses.dataclass
class MatricesResults:
    """Store results for testing matrices classes."""

    test: Matrices
    segmentation_: segmentation.Segmentation
    zoning_: zoning.ZoningSystem
    type_: matrices.MatrixType
    data: dict[segmentation.SegmentationSlice, pd.DataFrame]


def _produce_memory_matrices(
    zone_system: zoning.ZoningSystem,
    segmentation_: segmentation.Segmentation,
    type_: matrices.MatrixType,
) -> tuple[dict[segmentation.SegmentationSlice, pd.DataFrame], matrices.MemoryMatrices]:
    """Produce instance of MemoryMatrices for testing."""
    data = random_matrices_data(zone_system, segmentation_, seed=157)
    matrices_ = matrices.MemoryMatrices(
        segmentation_,
        zone_system,
        type_,
        [matrices.Matrix(j, i) for i, j in data.items()],
    )
    return data, matrices_


def _produce_matrices_files(
    path: pathlib.Path,
    zone_system: zoning.ZoningSystem,
    segmentation_: segmentation.Segmentation,
    type_: matrices.MatrixType,
    name: str = "test matrices files",
    data: dict[segmentation.SegmentationSlice, pd.DataFrame] | None = None,
) -> tuple[dict[segmentation.SegmentationSlice, pd.DataFrame], matrices.MatrixFiles]:
    """Produce instance of MatricesFiles for testing."""
    if data is None:
        data = random_matrices_data(zone_system, segmentation_, seed=157)
    folder = path / name
    folder.mkdir()

    for slice_, df in data.items():
        df.to_csv(
            folder / f"{type_.name}_{slice_.generate_name(segmentation_.seg_dict)}.csv.bz2"
        )

    matrices_ = matrices.MatrixFiles(segmentation_, zone_system, type_, folder)

    return data, matrices_


def _produce_long_matrices(
    zone_system: zoning.ZoningSystem,
    segmentation_: segmentation.Segmentation,
    type_: matrices.MatrixType,
    data: dict[segmentation.SegmentationSlice, pd.DataFrame] | None = None,
) -> tuple[dict[segmentation.SegmentationSlice, pd.DataFrame], matrices.LongMatrices]:
    """Produce instance of LongMatrices for testing."""
    if data is None:
        data = random_matrices_data(zone_system, segmentation_, seed=2466)

    datasets = []
    for slice_, df in data.items():
        df = df.stack()
        df.index = pd.MultiIndex.from_arrays(
            [[i] * len(df) for i in slice_.as_tuple()]
            + [df.index.get_level_values(i) for i in (0, 1)],
            names=slice_.naming_order + ("origin", "destination"),
        )
        df.name = "trips"
        datasets.append(df)

    long = pd.concat(datasets, axis=0)
    if isinstance(long, pd.Series):
        long = long.to_frame()

    matrices_ = matrices.LongMatrices(segmentation_, zone_system, type_, data=long)
    return data, matrices_


@pytest.fixture(name="tp_segmentation")
def fix_tp_segmentation() -> tuple[segmentation.Segmentation, matrices.MatrixType]:
    """Simple PA segmentation containing time period for testing."""
    input_ = segmentation.SegmentationInput(
        enum_segments=["userclass", "direction", "tp"],  # type: ignore
        naming_order=["userclass", "direction", "tp"],
        subsets={"tp": [1, 2, 3]},
    )
    return segmentation.Segmentation(input_), matrices.MatrixType.PA


@pytest.fixture(name="tp_memory_matrices")
def fix_tp_memory_matrices(
    zone_system: base.ZoningSystem,
    tp_segmentation: tuple[segmentation.Segmentation, matrices.MatrixType],
) -> MatricesResults:
    """Simple PA MemoryMatrices containing time period for testing."""
    data, matrices_ = _produce_memory_matrices(zone_system, *tp_segmentation)

    return MatricesResults(
        test=matrices_,
        segmentation_=tp_segmentation[0],
        zoning_=zone_system,
        type_=tp_segmentation[1],
        data=data,
    )


@pytest.fixture(name="tp_matrices_files")
def fix_tp_matrices_files(
    tmp_path: pathlib.Path,
    zone_system: base.ZoningSystem,
    tp_segmentation: tuple[segmentation.Segmentation, matrices.MatrixType],
) -> MatricesResults:
    """Simple PA MatricesFile containing time period for testing."""
    data, matrices_ = _produce_matrices_files(tmp_path, zone_system, *tp_segmentation)

    return MatricesResults(
        test=matrices_,
        segmentation_=tp_segmentation[0],
        zoning_=zone_system,
        type_=tp_segmentation[1],
        data=data,
    )


@pytest.fixture(name="tp_long_matrices")
def fix_tp_long_matrices(
    zone_system: base.ZoningSystem,
    tp_segmentation: tuple[segmentation.Segmentation, matrices.MatrixType],
) -> MatricesResults:
    """Simple PA LongMatrices containing time period for testing."""
    data, matrices_ = _produce_long_matrices(zone_system, *tp_segmentation)

    return MatricesResults(
        test=matrices_,
        segmentation_=tp_segmentation[0],
        zoning_=zone_system,
        type_=tp_segmentation[1],
        data=data,
    )


@pytest.fixture(name="hb_segmentation")
def fix_hb_segmentation() -> tuple[segmentation.Segmentation, matrices.MatrixType]:
    """Simple PA HB segmentation without time period for testing."""
    input_ = segmentation.SegmentationInput(
        enum_segments=["p", "direction"],  # type: ignore
        naming_order=["p", "direction"],
        subsets={"p": list(range(1, 9)), "direction": [1]},
    )
    return segmentation.Segmentation(input_), matrices.MatrixType.PA


@pytest.fixture(name="hb_memory_matrices")
def fix_hb_memory_matrices(
    zone_system: base.ZoningSystem,
    hb_segmentation: tuple[segmentation.Segmentation, matrices.MatrixType],
) -> MatricesResults:
    """Simple PA HB MemoryMatrices without time period for testing."""
    data, matrices_ = _produce_memory_matrices(zone_system, *hb_segmentation)

    return MatricesResults(
        test=matrices_,
        segmentation_=hb_segmentation[0],
        zoning_=zone_system,
        type_=hb_segmentation[1],
        data=data,
    )


@pytest.fixture(name="hb_matrices_files")
def fix_hb_matrices_files(
    tmp_path: pathlib.Path,
    zone_system: base.ZoningSystem,
    hb_segmentation: tuple[segmentation.Segmentation, matrices.MatrixType],
) -> MatricesResults:
    """Simple PA HB MatricesFiles without time period for testing."""
    data, matrices_ = _produce_matrices_files(tmp_path, zone_system, *hb_segmentation)

    return MatricesResults(
        test=matrices_,
        segmentation_=hb_segmentation[0],
        zoning_=zone_system,
        type_=hb_segmentation[1],
        data=data,
    )


@pytest.fixture(name="hb_long_matrices")
def fix_hb_long_matrices(
    zone_system: base.ZoningSystem,
    hb_segmentation: tuple[segmentation.Segmentation, matrices.MatrixType],
) -> MatricesResults:
    """Simple PA HB LongMatrices without time period for testing."""
    data, matrices_ = _produce_long_matrices(zone_system, *hb_segmentation)

    return MatricesResults(
        test=matrices_,
        segmentation_=hb_segmentation[0],
        zoning_=zone_system,
        type_=hb_segmentation[1],
        data=data,
    )


@pytest.fixture(name="nhb_segmentation")
def fix_nhb_segmentation() -> tuple[segmentation.Segmentation, matrices.MatrixType]:
    """Simple PA NHB segmentation without time period for testing."""
    input_ = segmentation.SegmentationInput(
        enum_segments=["p", "direction"],  # type: ignore
        naming_order=["p", "direction"],
        subsets={"p": [11, 12, 13, 14, 15, 16, 18], "direction": [0]},
    )
    return segmentation.Segmentation(input_), matrices.MatrixType.PA


@pytest.fixture(name="nhb_memory_matrices")
def fix_nhb_memory_matrices(
    zone_system: base.ZoningSystem,
    nhb_segmentation: tuple[segmentation.Segmentation, matrices.MatrixType],
) -> MatricesResults:
    """Simple PA NHB MemoryMatrices without time period for testing."""
    data, matrices_ = _produce_memory_matrices(zone_system, *nhb_segmentation)

    return MatricesResults(
        test=matrices_,
        segmentation_=nhb_segmentation[0],
        zoning_=zone_system,
        type_=nhb_segmentation[1],
        data=data,
    )


@pytest.fixture(name="nhb_matrices_files")
def fix_nhb_matrices_files(
    tmp_path: pathlib.Path,
    zone_system: base.ZoningSystem,
    nhb_segmentation: tuple[segmentation.Segmentation, matrices.MatrixType],
) -> MatricesResults:
    """Simple PA NHB MatricesFiles without time period for testing."""
    data, matrices_ = _produce_matrices_files(tmp_path, zone_system, *nhb_segmentation)

    return MatricesResults(
        test=matrices_,
        segmentation_=nhb_segmentation[0],
        zoning_=zone_system,
        type_=nhb_segmentation[1],
        data=data,
    )


@pytest.fixture(name="nhb_long_matrices")
def fix_nhb_long_matrices(
    zone_system: base.ZoningSystem,
    nhb_segmentation: tuple[segmentation.Segmentation, matrices.MatrixType],
) -> MatricesResults:
    """Simple PA NHB LongMatrices without time period for testing."""
    data, matrices_ = _produce_long_matrices(zone_system, *nhb_segmentation)

    return MatricesResults(
        test=matrices_,
        segmentation_=nhb_segmentation[0],
        zoning_=zone_system,
        type_=nhb_segmentation[1],
        data=data,
    )


@dataclasses.dataclass
class DisaggregateDatasets:
    """Store datasets for disaggregate method tests."""

    zone_system: zoning.ZoningSystem
    type_: matrices.MatrixType
    from_segmentation: segmentation.Segmentation
    to_segmentation: segmentation.Segmentation
    input_: dict[segmentation.SegmentationSlice, pd.DataFrame]
    targets: dict[segmentation.SegmentationSlice, pd.DataFrame]
    expected: dict[segmentation.SegmentationSlice, pd.DataFrame]


@pytest.fixture(name="disaggregate_dataset")
def fix_disaggregate_dataset(zone_system: zoning.ZoningSystem) -> DisaggregateDatasets:
    """Dataset for testing disaggregate method."""
    to_segmentation = segmentation.Segmentation(
        segmentation.SegmentationInput(
            enum_segments=["userclass", "direction_od", "tp"],  # type: ignore
            naming_order=["userclass", "direction_od", "tp"],
            subsets={"tp": [1, 2, 3]},
        )
    )
    type_ = matrices.MatrixType.OD

    filter_tp = functools.partial(filter, lambda x: x != "tp")
    from_segmentation = segmentation.Segmentation(
        segmentation.SegmentationInput(
            enum_segments=list(filter_tp(to_segmentation.input.naming_order)),  # type: ignore
            naming_order=list(filter_tp(to_segmentation.input.naming_order)),
            subsets={i: j for i, j in to_segmentation.input.subsets.items() if i != "tp"},
        )
    )

    # Produce some random expected matrices and calculate inputs from these
    expected_matrices = random_matrices_data(zone_system, to_segmentation, seed=1157)

    # Divide expected matrices by constant to have a consistent ratio between them
    targets = {i: j / 3 for i, j in expected_matrices.items()}

    agg_matrices = {}
    for slice_ in from_segmentation.iter_slices():
        total = 0
        for tp in to_segmentation.input.subsets["tp"]:
            to_slice = segmentation.SegmentationSlice(
                slice_.data | {"tp": tp}, to_segmentation.naming_order
            )
            total += expected_matrices[to_slice]

        agg_matrices[slice_] = total

    return DisaggregateDatasets(
        zone_system=zone_system,
        type_=type_,
        from_segmentation=from_segmentation,
        to_segmentation=to_segmentation,
        input_=agg_matrices,
        targets=targets,
        expected=expected_matrices,
    )


@dataclasses.dataclass
class DisaggregateMatrices:
    """Store matrices and expected for disaggregate method tests."""

    zone_system: zoning.ZoningSystem
    type_: matrices.MatrixType
    from_segmentation: segmentation.Segmentation
    to_segmentation: segmentation.Segmentation
    input_: Matrices
    targets: Matrices
    expected: dict[segmentation.SegmentationSlice, pd.DataFrame]


@pytest.fixture(name="disaggregate_memory_matrices")
def fix_disaggregate_memory_matrices(
    disaggregate_dataset: DisaggregateDatasets,
) -> DisaggregateMatrices:
    """MemoryMatrices for testing disaggregate method without replace."""
    input_matrices = matrices.MemoryMatrices(
        disaggregate_dataset.from_segmentation,
        disaggregate_dataset.zone_system,
        disaggregate_dataset.type_,
        [matrices.Matrix(j, i) for i, j in disaggregate_dataset.input_.items()],
    )
    target_matrices = matrices.MemoryMatrices(
        disaggregate_dataset.to_segmentation,
        disaggregate_dataset.zone_system,
        disaggregate_dataset.type_,
        [matrices.Matrix(j, i) for i, j in disaggregate_dataset.targets.items()],
    )

    return DisaggregateMatrices(
        disaggregate_dataset.zone_system,
        disaggregate_dataset.type_,
        disaggregate_dataset.from_segmentation,
        disaggregate_dataset.to_segmentation,
        input_matrices,
        target_matrices,
        disaggregate_dataset.expected,
    )


@pytest.fixture(name="disaggregate_matrices_files")
def fix_disaggregate_matrices_files(
    tmp_path: pathlib.Path,
    disaggregate_dataset: DisaggregateDatasets,
) -> DisaggregateMatrices:
    """MatricesFiles for testing disaggregate method without replace."""
    input_matrices = _produce_matrices_files(
        tmp_path,
        disaggregate_dataset.zone_system,
        disaggregate_dataset.from_segmentation,
        disaggregate_dataset.type_,
        name="input matrices",
        data=disaggregate_dataset.input_,
    )
    target_matrices = _produce_matrices_files(
        tmp_path,
        disaggregate_dataset.zone_system,
        disaggregate_dataset.to_segmentation,
        disaggregate_dataset.type_,
        name="target matrices",
        data=disaggregate_dataset.targets,
    )
    return DisaggregateMatrices(
        disaggregate_dataset.zone_system,
        disaggregate_dataset.type_,
        disaggregate_dataset.from_segmentation,
        disaggregate_dataset.to_segmentation,
        input_matrices[1],
        target_matrices[1],
        disaggregate_dataset.expected,
    )


@pytest.fixture(name="disaggregate_long_matrices")
def fix_disaggregate_long_matrices(
    disaggregate_dataset: DisaggregateDatasets,
) -> DisaggregateMatrices:
    """LongMatrices for testing disaggregate method without replace."""
    input_matrices = _produce_long_matrices(
        disaggregate_dataset.zone_system,
        disaggregate_dataset.from_segmentation,
        disaggregate_dataset.type_,
        data=disaggregate_dataset.input_,
    )
    target_matrices = _produce_long_matrices(
        disaggregate_dataset.zone_system,
        disaggregate_dataset.to_segmentation,
        disaggregate_dataset.type_,
        data=disaggregate_dataset.targets,
    )
    return DisaggregateMatrices(
        disaggregate_dataset.zone_system,
        disaggregate_dataset.type_,
        disaggregate_dataset.from_segmentation,
        disaggregate_dataset.to_segmentation,
        input_matrices[1],
        target_matrices[1],
        disaggregate_dataset.expected,
    )


@dataclasses.dataclass
class AggregateMatrices:
    """Store matrices and expected for aggregate method tests."""

    zone_system: zoning.ZoningSystem
    type_: matrices.MatrixType
    from_segmentation: segmentation.Segmentation
    to_segmentation: segmentation.Segmentation
    input_: Matrices
    expected: dict[segmentation.SegmentationSlice, pd.DataFrame]


@pytest.fixture(name="aggregate_memory_matrices")
def fix_aggregate_memory_matrices(
    disaggregate_dataset: DisaggregateDatasets,
) -> AggregateMatrices:
    """MemoryMatrices for testing aggregate method."""
    input_matrices = matrices.MemoryMatrices(
        disaggregate_dataset.to_segmentation,
        disaggregate_dataset.zone_system,
        disaggregate_dataset.type_,
        [matrices.Matrix(j, i) for i, j in disaggregate_dataset.expected.items()],
    )

    return AggregateMatrices(
        disaggregate_dataset.zone_system,
        disaggregate_dataset.type_,
        from_segmentation=disaggregate_dataset.to_segmentation,
        to_segmentation=disaggregate_dataset.from_segmentation,
        input_=input_matrices,
        expected=disaggregate_dataset.input_,
    )


@pytest.fixture(name="aggregate_matrices_files")
def fix_aggregate_matrices_files(
    tmp_path: pathlib.Path,
    disaggregate_dataset: DisaggregateDatasets,
) -> AggregateMatrices:
    """MatricesFiles for testing aggregate method."""
    input_matrices = _produce_matrices_files(
        tmp_path,
        disaggregate_dataset.zone_system,
        disaggregate_dataset.to_segmentation,
        disaggregate_dataset.type_,
        name="input matrices",
        data=disaggregate_dataset.expected,
    )
    return AggregateMatrices(
        disaggregate_dataset.zone_system,
        disaggregate_dataset.type_,
        from_segmentation=disaggregate_dataset.to_segmentation,
        to_segmentation=disaggregate_dataset.from_segmentation,
        input_=input_matrices[1],
        expected=disaggregate_dataset.input_,
    )


@pytest.fixture(name="aggregate_long_matrices")
def fix_aggregate_long_matrices(
    disaggregate_dataset: DisaggregateDatasets,
) -> AggregateMatrices:
    """LongMatrices for testing aggregate method."""
    input_matrices = _produce_long_matrices(
        disaggregate_dataset.zone_system,
        disaggregate_dataset.to_segmentation,
        disaggregate_dataset.type_,
        data=disaggregate_dataset.expected,
    )
    return AggregateMatrices(
        disaggregate_dataset.zone_system,
        disaggregate_dataset.type_,
        from_segmentation=disaggregate_dataset.to_segmentation,
        to_segmentation=disaggregate_dataset.from_segmentation,
        input_=input_matrices[1],
        expected=disaggregate_dataset.input_,
    )


@dataclasses.dataclass
class DisaggregateReplaceDatasets:
    """Store datasets for disaggregate and replace method tests."""

    zone_system: zoning.ZoningSystem
    type_: matrices.MatrixType
    from_segmentation: segmentation.Segmentation
    to_segmentation: segmentation.Segmentation
    input_: dict[segmentation.SegmentationSlice, pd.DataFrame]
    targets: dict[segmentation.SegmentationSlice, pd.DataFrame]
    expected: dict[segmentation.SegmentationSlice, pd.DataFrame]
    segments: tuple[base.Segment, base.Segment]


@pytest.fixture(name="disaggregate_replace_datasets")
def fix_disaggregate_replace_datasets(
    zone_system: zoning.ZoningSystem,
) -> DisaggregateReplaceDatasets:
    """Datasets for testing disaggregate replace method."""
    segments = ["m"]
    subsets = {"m": [3]}
    from_segmentation = base.Segmentation(
        base.SegmentationInput(
            enum_segments=segments + ["userclass", "direction_od"],
            naming_order=["direction_od", "userclass"] + segments,
            subsets=subsets,
        )
    )
    to_segmentation = base.Segmentation(
        base.SegmentationInput(
            enum_segments=segments + ["p", "direction_od"],
            naming_order=["direction_od", "p"] + segments,
            subsets=subsets,
        )
    )

    # Produce some random expected matrices and calculate inputs from these
    expected_matrices = random_matrices_data(zone_system, to_segmentation, seed=16549)

    # Aggregate expected matrices to produce test compiled
    aggregations: list[tuple[int, int, tuple[int, ...]]] = [
        (0, 1, (11, 12)),
        (0, 3, (13, 14, 15, 16, 18)),
        (1, 1, (2,)),
        (1, 2, (1,)),
        (1, 3, tuple(range(3, 9))),
        (2, 1, (2,)),
        (2, 2, (1,)),
        (2, 3, tuple(range(3, 9))),
    ]
    agg_matrices = {}
    for direction, uc, purposes in aggregations:
        total = 0

        for p in purposes:
            slice_ = segmentation.SegmentationSlice(
                {"direction_od": direction, "p": p, "m": 3}, to_segmentation.naming_order
            )
            total += expected_matrices[slice_]

        slice_ = segmentation.SegmentationSlice(
            {"direction_od": direction, "userclass": uc, "m": 3},
            from_segmentation.naming_order,
        )
        agg_matrices[slice_] = total

    return DisaggregateReplaceDatasets(
        zone_system=zone_system,
        type_=matrices.MatrixType.OD,
        from_segmentation=from_segmentation,
        to_segmentation=to_segmentation,
        input_=agg_matrices,
        targets={i: j / 3 for i, j in expected_matrices.items()},
        expected=expected_matrices,
        segments=(
            base.segments.SegmentsSuper.USERCLASS.get_segment(),
            base.segments.SegmentsSuper.PURPOSE.get_segment(),
        ),
    )


@dataclasses.dataclass
class DisaggregateReplaceMatrices:
    """Store matrices and expected for disaggregate and replace method tests."""

    zone_system: zoning.ZoningSystem
    type_: matrices.MatrixType
    from_segmentation: segmentation.Segmentation
    to_segmentation: segmentation.Segmentation
    input_: Matrices
    targets: Matrices
    expected: dict[segmentation.SegmentationSlice, pd.DataFrame]
    segments: tuple[base.Segment, base.Segment]


@pytest.fixture(name="disaggregate_replace_memory_matrices")
def fix_disaggregate_replace_memory_matrices(
    disaggregate_replace_datasets: DisaggregateReplaceDatasets,
) -> DisaggregateReplaceMatrices:
    """MemoryMatrices for testing disaggregate method with replace."""
    input_matrices = matrices.MemoryMatrices(
        disaggregate_replace_datasets.from_segmentation,
        disaggregate_replace_datasets.zone_system,
        disaggregate_replace_datasets.type_,
        [matrices.Matrix(j, i) for i, j in disaggregate_replace_datasets.input_.items()],
    )
    target_matrices = matrices.MemoryMatrices(
        disaggregate_replace_datasets.to_segmentation,
        disaggregate_replace_datasets.zone_system,
        disaggregate_replace_datasets.type_,
        [matrices.Matrix(j, i) for i, j in disaggregate_replace_datasets.targets.items()],
    )

    return DisaggregateReplaceMatrices(
        disaggregate_replace_datasets.zone_system,
        disaggregate_replace_datasets.type_,
        disaggregate_replace_datasets.from_segmentation,
        disaggregate_replace_datasets.to_segmentation,
        input_matrices,
        target_matrices,
        disaggregate_replace_datasets.expected,
        disaggregate_replace_datasets.segments,
    )


@pytest.fixture(name="disaggregate_replace_matrices_files")
def fix_disaggregate_replace_matrices_files(
    tmp_path: pathlib.Path,
    disaggregate_replace_datasets: DisaggregateReplaceDatasets,
) -> DisaggregateReplaceMatrices:
    """MatricesFiles for testing disaggregate method with replace."""
    input_matrices = _produce_matrices_files(
        tmp_path,
        disaggregate_replace_datasets.zone_system,
        disaggregate_replace_datasets.from_segmentation,
        disaggregate_replace_datasets.type_,
        name="input matrices",
        data=disaggregate_replace_datasets.input_,
    )
    target_matrices = _produce_matrices_files(
        tmp_path,
        disaggregate_replace_datasets.zone_system,
        disaggregate_replace_datasets.to_segmentation,
        disaggregate_replace_datasets.type_,
        name="target matrices",
        data=disaggregate_replace_datasets.targets,
    )
    return DisaggregateReplaceMatrices(
        disaggregate_replace_datasets.zone_system,
        disaggregate_replace_datasets.type_,
        disaggregate_replace_datasets.from_segmentation,
        disaggregate_replace_datasets.to_segmentation,
        input_matrices[1],
        target_matrices[1],
        disaggregate_replace_datasets.expected,
        disaggregate_replace_datasets.segments,
    )


@pytest.fixture(name="disaggregate_replace_long_matrices")
def fix_disaggregate_replace_long_matrices(
    disaggregate_replace_datasets: DisaggregateReplaceDatasets,
) -> DisaggregateReplaceMatrices:
    """LongMatrices for testing disaggregate method with replace."""
    input_matrices = _produce_long_matrices(
        disaggregate_replace_datasets.zone_system,
        disaggregate_replace_datasets.from_segmentation,
        disaggregate_replace_datasets.type_,
        data=disaggregate_replace_datasets.input_,
    )
    target_matrices = _produce_long_matrices(
        disaggregate_replace_datasets.zone_system,
        disaggregate_replace_datasets.to_segmentation,
        disaggregate_replace_datasets.type_,
        data=disaggregate_replace_datasets.targets,
    )
    return DisaggregateReplaceMatrices(
        disaggregate_replace_datasets.zone_system,
        disaggregate_replace_datasets.type_,
        disaggregate_replace_datasets.from_segmentation,
        disaggregate_replace_datasets.to_segmentation,
        input_matrices[1],
        target_matrices[1],
        disaggregate_replace_datasets.expected,
        disaggregate_replace_datasets.segments,
    )


@pytest.fixture(name="disaggregate_and_replace_datasets")
def fix_disaggregate_and_replace_datasets(zone_system: base.ZoningSystem):
    """Datasets for the disaggregate and replace tests."""
    time_periods = [1, 2, 3]
    segments = ["m"]
    subsets = {"m": [3]}
    from_segmentation = base.Segmentation(
        base.SegmentationInput(
            enum_segments=segments + ["userclass", "direction_od"],
            naming_order=["direction_od", "userclass"] + segments,
            subsets=subsets,
        )
    )
    to_segmentation = base.Segmentation(
        base.SegmentationInput(
            enum_segments=segments + ["p", "direction_od", "tp"],
            naming_order=["direction_od", "p"] + segments + ["tp"],
            subsets=subsets | {"tp": time_periods},
        )
    )

    # Produce some random expected matrices and calculate inputs from these
    expected_matrices = random_matrices_data(zone_system, to_segmentation, seed=16879)

    # Aggregate expected matrices to produce test compiled
    aggregations: list[tuple[int, int, tuple[int, ...]]] = [
        (0, 1, (11, 12)),
        (0, 3, (13, 14, 15, 16, 18)),
        (1, 1, (2,)),
        (1, 2, (1,)),
        (1, 3, tuple(range(3, 9))),
        (2, 1, (2,)),
        (2, 2, (1,)),
        (2, 3, tuple(range(3, 9))),
    ]
    agg_matrices = {}
    for direction, uc, purposes in aggregations:
        total = 0

        for p in purposes:
            for tp in time_periods:
                slice_ = segmentation.SegmentationSlice(
                    {"direction_od": direction, "p": p, "m": 3, "tp": tp},
                    to_segmentation.naming_order,
                )
                total += expected_matrices[slice_]

        slice_ = segmentation.SegmentationSlice(
            {"direction_od": direction, "userclass": uc, "m": 3},
            from_segmentation.naming_order,
        )
        agg_matrices[slice_] = total

    return DisaggregateReplaceDatasets(
        zone_system=zone_system,
        type_=matrices.MatrixType.OD,
        from_segmentation=from_segmentation,
        to_segmentation=to_segmentation,
        input_=agg_matrices,
        targets={i: j / 3 for i, j in expected_matrices.items()},
        expected=expected_matrices,
        segments=(
            base.segments.SegmentsSuper.USERCLASS.get_segment(),
            base.segments.SegmentsSuper.PURPOSE.get_segment(),
        ),
    )


@pytest.fixture(name="disaggregate_and_replace_memory_matrices")
def fix_disaggregate_and_replace_memory_matrices(
    disaggregate_and_replace_datasets: DisaggregateReplaceDatasets,
) -> DisaggregateReplaceMatrices:
    """MemoryMatrices for testing disaggregate with new segment and replace."""
    input_matrices = matrices.MemoryMatrices(
        disaggregate_and_replace_datasets.from_segmentation,
        disaggregate_and_replace_datasets.zone_system,
        disaggregate_and_replace_datasets.type_,
        [matrices.Matrix(j, i) for i, j in disaggregate_and_replace_datasets.input_.items()],
    )
    target_matrices = matrices.MemoryMatrices(
        disaggregate_and_replace_datasets.to_segmentation,
        disaggregate_and_replace_datasets.zone_system,
        disaggregate_and_replace_datasets.type_,
        [matrices.Matrix(j, i) for i, j in disaggregate_and_replace_datasets.targets.items()],
    )

    return DisaggregateReplaceMatrices(
        disaggregate_and_replace_datasets.zone_system,
        disaggregate_and_replace_datasets.type_,
        disaggregate_and_replace_datasets.from_segmentation,
        disaggregate_and_replace_datasets.to_segmentation,
        input_matrices,
        target_matrices,
        disaggregate_and_replace_datasets.expected,
        disaggregate_and_replace_datasets.segments,
    )


@pytest.fixture(name="disaggregate_and_replace_matrices_files")
def fix_disaggregate_and_replace_matrices_files(
    tmp_path: pathlib.Path,
    disaggregate_and_replace_datasets: DisaggregateReplaceDatasets,
) -> DisaggregateReplaceMatrices:
    """MatricesFiles for testing disaggrate with new segment and replace."""
    input_matrices = _produce_matrices_files(
        tmp_path,
        disaggregate_and_replace_datasets.zone_system,
        disaggregate_and_replace_datasets.from_segmentation,
        disaggregate_and_replace_datasets.type_,
        name="input matrices",
        data=disaggregate_and_replace_datasets.input_,
    )
    target_matrices = _produce_matrices_files(
        tmp_path,
        disaggregate_and_replace_datasets.zone_system,
        disaggregate_and_replace_datasets.to_segmentation,
        disaggregate_and_replace_datasets.type_,
        name="target matrices",
        data=disaggregate_and_replace_datasets.targets,
    )
    return DisaggregateReplaceMatrices(
        disaggregate_and_replace_datasets.zone_system,
        disaggregate_and_replace_datasets.type_,
        disaggregate_and_replace_datasets.from_segmentation,
        disaggregate_and_replace_datasets.to_segmentation,
        input_matrices[1],
        target_matrices[1],
        disaggregate_and_replace_datasets.expected,
        disaggregate_and_replace_datasets.segments,
    )


@pytest.fixture(name="disaggregate_and_replace_long_matrices")
def fix_disaggregate_and_replace_long_matrices(
    disaggregate_and_replace_datasets: DisaggregateReplaceDatasets,
) -> DisaggregateReplaceMatrices:
    """LongMatrices for testing disaggrate with new segment and replace."""
    input_matrices = _produce_long_matrices(
        disaggregate_and_replace_datasets.zone_system,
        disaggregate_and_replace_datasets.from_segmentation,
        disaggregate_and_replace_datasets.type_,
        data=disaggregate_and_replace_datasets.input_,
    )
    target_matrices = _produce_long_matrices(
        disaggregate_and_replace_datasets.zone_system,
        disaggregate_and_replace_datasets.to_segmentation,
        disaggregate_and_replace_datasets.type_,
        data=disaggregate_and_replace_datasets.targets,
    )
    return DisaggregateReplaceMatrices(
        disaggregate_and_replace_datasets.zone_system,
        disaggregate_and_replace_datasets.type_,
        disaggregate_and_replace_datasets.from_segmentation,
        disaggregate_and_replace_datasets.to_segmentation,
        input_matrices[1],
        target_matrices[1],
        disaggregate_and_replace_datasets.expected,
        disaggregate_and_replace_datasets.segments,
    )


class TestMatrices:
    """Tests for the common functionality across all matrices."""

    @pytest.mark.parametrize(
        "matrices_", ["tp_memory_matrices", "tp_matrices_files", "tp_long_matrices"]
    )
    def test_segmentation(self, request: pytest.FixtureRequest, matrices_: str) -> None:
        """Test segmentation property returns expected Segmentation."""
        result: MatricesResults = request.getfixturevalue(matrices_)
        assert result.test.segmentation == result.segmentation_

    @pytest.mark.parametrize(
        "matrices_", ["tp_memory_matrices", "tp_matrices_files", "tp_long_matrices"]
    )
    def test_zoning(self, request: pytest.FixtureRequest, matrices_: str) -> None:
        """Test zoning property returns expected ZoningSystem."""
        result: MatricesResults = request.getfixturevalue(matrices_)
        assert result.test.zoning == result.zoning_

    @pytest.mark.parametrize(
        "matrices_", ["tp_memory_matrices", "tp_matrices_files", "tp_long_matrices"]
    )
    def test_type(self, request: pytest.FixtureRequest, matrices_: str) -> None:
        """Test type property returns expected MatrixType."""
        result: MatricesResults = request.getfixturevalue(matrices_)
        assert result.test.type == result.type_

    @pytest.mark.parametrize(
        "matrices_", ["tp_memory_matrices", "tp_matrices_files", "tp_long_matrices"]
    )
    def test_name(self, request: pytest.FixtureRequest, matrices_: str) -> None:
        """Test name property returns a string."""
        result: MatricesResults = request.getfixturevalue(matrices_)
        assert isinstance(result.test.name, str)

    @pytest.mark.parametrize(
        "matrices_", ["tp_memory_matrices", "tp_matrices_files", "tp_long_matrices"]
    )
    def test_get_matrix(self, request: pytest.FixtureRequest, matrices_: str) -> None:
        """Test `get_matrix` method returns expected DataFrame for all slices."""
        result: MatricesResults = request.getfixturevalue(matrices_)

        for slice_ in result.segmentation_.iter_slices():
            expected = result.data.get(slice_)
            got = result.test.get_matrix(slice_)

            assert got.slice == slice_
            pd.testing.assert_frame_equal(
                expected,
                got.data,
                obj=str(slice_),
                check_index_type=False,
                check_column_type=False,
            )

    @pytest.mark.parametrize(
        "matrices_", ["tp_memory_matrices", "tp_matrices_files", "tp_long_matrices"]
    )
    def test_set_matrix(self, request: pytest.FixtureRequest, matrices_: str) -> None:
        """Test `set_matrix` method for all slices."""
        result: MatricesResults = request.getfixturevalue(matrices_)
        rand = random_matrices(result.zoning_, seed=57)

        for slice_ in result.segmentation_.iter_slices():
            expected = next(rand)
            result.test.set_matrix(expected, slice_)

            got = result.test.get_matrix(slice_)
            assert got.slice == slice_
            pd.testing.assert_frame_equal(
                expected,
                got.data,
                obj=str(slice_),
                check_index_type=False,
                check_column_type=False,
                check_names=False,
            )

    @pytest.mark.parametrize(
        "matrices_", ["tp_memory_matrices", "tp_matrices_files", "tp_long_matrices"]
    )
    @pytest.mark.parametrize(
        "new_seg_input",
        [
            None,
            segmentation.SegmentationInput(
                enum_segments=["p", "m"],  # type: ignore
                naming_order=["p", "m"],
            ),
        ],
    )
    @pytest.mark.parametrize(
        "new_zoning",
        [
            None,
            zoning.ZoningSystem(
                "test 2",
                pd.DataFrame({"zone_id": [1, 2, 3, 4]}),
                zoning.ZoningSystemMetaData(name="test 2"),
            ),
        ],
    )
    @pytest.mark.parametrize("new_type", [None, matrices.MatrixType.OD])
    def test_new(
        self,
        request: pytest.FixtureRequest,
        matrices_: str,
        new_seg_input: segmentation.SegmentationInput | None,
        new_zoning: zoning.ZoningSystem | None,
        new_type: matrices.MatrixType | None,
    ) -> None:
        """Test `new` method returns a new empty instance."""
        result: MatricesResults = request.getfixturevalue(matrices_)
        name = "New Test Name"
        if new_seg_input is not None:
            new_segmentation = segmentation.Segmentation(new_seg_input)
        else:
            new_segmentation = None

        with warnings.catch_warnings(action="ignore", category=matrices.MatricesWarning):
            new = result.test.new(
                name, segmentation_=new_segmentation, zoning=new_zoning, type_=new_type
            )

        assert isinstance(new, type(result.test))
        assert new.name == name

        if new_segmentation is None:
            assert new.segmentation == result.segmentation_
        else:
            assert new.segmentation == new_segmentation

        if new_zoning is None:
            assert new.zoning == result.zoning_
        else:
            assert new.zoning == new_zoning

    @pytest.mark.parametrize(
        "matrices_, has_time_period",
        [
            ("tp_memory_matrices", True),
            ("tp_matrices_files", True),
            ("tp_long_matrices", True),
            ("hb_memory_matrices", False),
            ("hb_matrices_files", False),
            ("hb_long_matrices", False),
        ],
    )
    def test_has_time_periods(
        self, request: pytest.FixtureRequest, matrices_: str, has_time_period: bool
    ) -> None:
        """Test has_time_period method works correctly."""
        result: MatricesResults = request.getfixturevalue(matrices_)
        assert result.test.has_time_periods == has_time_period

    @pytest.mark.parametrize(
        "matrices_, home_based_only",
        [
            ("tp_memory_matrices", False),
            ("tp_matrices_files", False),
            ("hb_memory_matrices", True),
            ("hb_matrices_files", True),
            ("hb_long_matrices", True),
            ("nhb_memory_matrices", False),
            ("nhb_matrices_files", False),
            ("nhb_long_matrices", False),
        ],
    )
    def test_home_based_only(
        self, request: pytest.FixtureRequest, matrices_: str, home_based_only: bool
    ) -> None:
        """Test home_based_only method works correctly."""
        result: MatricesResults = request.getfixturevalue(matrices_)
        assert result.test.home_based_only == home_based_only

    @pytest.mark.parametrize(
        "matrices_, non_home_based_only",
        [
            ("tp_memory_matrices", False),
            ("tp_matrices_files", False),
            ("hb_memory_matrices", False),
            ("hb_matrices_files", False),
            ("hb_long_matrices", False),
            ("nhb_memory_matrices", True),
            ("nhb_matrices_files", True),
            ("nhb_long_matrices", True),
        ],
    )
    def test_non_home_based_only(
        self, request: pytest.FixtureRequest, matrices_: str, non_home_based_only: bool
    ) -> None:
        """Test non_home_based_only method works correctly."""
        result: MatricesResults = request.getfixturevalue(matrices_)
        assert result.test.non_home_based_only == non_home_based_only

    @pytest.mark.parametrize(
        "matrices_, params",
        [
            ("tp_memory_matrices", {"userclass": 3, "direction": 1, "tp": 2}),
            ("tp_matrices_files", {"userclass": 3, "direction": 1, "tp": 2}),
            ("tp_long_matrices", {"userclass": 3, "direction": 1, "tp": 2}),
        ],
    )
    def test_validate_slice(
        self, request: pytest.FixtureRequest, matrices_: str, params: dict[str, int]
    ) -> None:
        """Test `validate_slice` with valid slices."""
        result: MatricesResults = request.getfixturevalue(matrices_)

        slice_ = segmentation.SegmentationSlice(params, result.segmentation_.naming_order)
        result.test.validate_slice(slice_)

    @pytest.mark.parametrize(
        "matrices_, params",
        [
            ("tp_memory_matrices", {"userclass": 7, "direction": 3, "tp": 5}),
            ("tp_matrices_files", {"userclass": 7, "direction": 3, "tp": 5}),
            ("tp_long_matrices", {"userclass": 7, "direction": 3, "tp": 5}),
        ],
    )
    def test_validate_slice_invalid(
        self, request: pytest.FixtureRequest, matrices_: str, params: dict[str, int]
    ) -> None:
        """Test `validate_slice` correctly raises error."""
        result: MatricesResults = request.getfixturevalue(matrices_)
        slice_ = segmentation.SegmentationSlice(params, result.segmentation_.naming_order)

        with pytest.raises(ValueError):
            result.test.validate_slice(slice_)

    @pytest.mark.parametrize(
        "matrices_", ["tp_memory_matrices", "tp_matrices_files", "tp_long_matrices"]
    )
    def test_validate_matrix(self, request: pytest.FixtureRequest, matrices_: str) -> None:
        """Test `validate_matrix` for valid matrix."""
        result: MatricesResults = request.getfixturevalue(matrices_)

        rand = random_matrices(result.zoning_)
        result.test.validate_matrix(next(rand), "Valid matrix")

    @pytest.mark.parametrize(
        "matrices_", ["tp_memory_matrices", "tp_matrices_files", "tp_long_matrices"]
    )
    def test_validate_matrix_invalid(
        self, request: pytest.FixtureRequest, matrices_: str
    ) -> None:
        """Test `validate_matrix` correctly raises error."""
        result: MatricesResults = request.getfixturevalue(matrices_)

        zones = pd.DataFrame({"zone_id": result.zoning_.zone_ids.tolist() + [-1]})
        zoning_ = zoning.ZoningSystem(
            "invalid zones", zones, zoning.ZoningSystemMetaData(name="invalid zones")
        )
        rand = random_matrices(zoning_)

        with pytest.raises(ValueError):
            result.test.validate_matrix(next(rand), "Invalid matrix")

    @pytest.mark.parametrize(
        "matrices_",
        ["aggregate_memory_matrices", "aggregate_matrices_files", "aggregate_long_matrices"],
    )
    def test_aggregate(self, request: pytest.FixtureRequest, matrices_: str) -> None:
        """Test aggregating to up a segment."""
        result: AggregateMatrices = request.getfixturevalue(matrices_)

        answer = result.input_.aggregate(result.to_segmentation)

        assert answer.segmentation == result.to_segmentation

        for slice_ in result.to_segmentation.iter_slices():
            pd.testing.assert_frame_equal(
                result.expected[slice_],
                answer.get_matrix(slice_).data,
                check_column_type=False,
                check_index_type=False,
            )

    @pytest.mark.parametrize(
        "matrices_",
        [
            "disaggregate_memory_matrices",
            "disaggregate_matrices_files",
            "disaggregate_long_matrices",
        ],
    )
    def test_disaggregate(self, request: pytest.FixtureRequest, matrices_: str) -> None:
        """Test disaggregating to one new segment, with no replacements."""
        result: DisaggregateMatrices = request.getfixturevalue(matrices_)

        answer = result.input_.disaggregate(result.targets)

        assert answer.segmentation == result.to_segmentation

        for slice_ in result.to_segmentation.iter_slices():
            pd.testing.assert_frame_equal(
                result.expected[slice_],
                answer.get_matrix(slice_).data,
                check_column_type=False,
                check_index_type=False,
            )

    @pytest.mark.parametrize(
        "matrices_",
        [
            "disaggregate_replace_memory_matrices",
            "disaggregate_replace_matrices_files",
            "disaggregate_replace_long_matrices",
        ],
    )
    @pytest.mark.filterwarnings("ignore:No slices found .*:RuntimeWarning")
    def test_disaggregate_replace(
        self, request: pytest.FixtureRequest, matrices_: str
    ) -> None:
        """Test disaggregating replacements."""
        result: DisaggregateReplaceMatrices = request.getfixturevalue(matrices_)

        answer = result.input_.disaggregate(result.targets, *result.segments)

        assert answer.segmentation == result.to_segmentation

        for slice_ in result.to_segmentation.iter_slices():
            pd.testing.assert_frame_equal(
                result.expected[slice_],
                answer.get_matrix(slice_).data,
                check_column_type=False,
                check_index_type=False,
            )

    @pytest.mark.parametrize(
        "matrices_",
        [
            "disaggregate_and_replace_memory_matrices",
            "disaggregate_and_replace_matrices_files",
            "disaggregate_and_replace_long_matrices",
        ],
    )
    @pytest.mark.filterwarnings("ignore:No slices found .*:RuntimeWarning")
    def test_disaggregate_and_replace(
        self, request: pytest.FixtureRequest, matrices_: str
    ) -> None:
        """Test disaggregating with new segment and replacement."""
        result: DisaggregateReplaceMatrices = request.getfixturevalue(matrices_)

        answer = result.input_.disaggregate(result.targets, *result.segments)

        assert answer.segmentation == result.to_segmentation

        for slice_ in result.to_segmentation.iter_slices():
            pd.testing.assert_frame_equal(
                result.expected[slice_],
                answer.get_matrix(slice_).data,
                check_column_type=False,
                check_index_type=False,
            )
