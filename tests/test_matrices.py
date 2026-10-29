"""Tests for `matrices` module."""

##### IMPORTS #####

# Built-Ins
import dataclasses
import functools
import pathlib
import warnings
from collections.abc import Generator

# Third Party
import caf.base as cbase
import numpy as np
import pandas as pd
import pytest
from caf.base import segmentation, zoning

# Local Imports
from caf.mat import cube, matrices, omx

##### CONSTANTS #####


@pytest.fixture(name="zone_system")
def fix_zone_system() -> cbase.ZoningSystem:
    """Test zoning system with 3 zones."""
    return cbase.ZoningSystem(
        "test",
        pd.DataFrame({"zone_id": [1, 2, 3]}),
        zoning.ZoningSystemMetaData(name="test"),
    )


def random_matrices(
    zones: cbase.ZoningSystem, max_: int = 100, seed: int = 1
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
        enum_segments=["p", "tp", "direction"],
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
    zone_system: cbase.ZoningSystem, long_segmentation: cbase.Segmentation
) -> LongMatricesData:
    """Single column of data for LongMatrices tests."""
    zones = pd.DataFrame(
        {
            "origin": np.repeat(
                np.repeat(zone_system.zone_ids, len(zone_system)),
                len(long_segmentation),
            ),
            "destination": np.repeat(
                np.tile(zone_system.zone_ids, len(zone_system)), len(long_segmentation)
            ),
        }
    )
    zones = pd.DataFrame(
        {
            "origin": np.tile(
                np.repeat(zone_system.zone_ids, len(zone_system)),
                len(long_segmentation),
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

    def test_init(self, long_data: LongMatricesData) -> None:
        """Test initialising LongMatrices class with a dataframe."""
        answer = matrices.LongMatrices(
            long_data.segmentation,
            long_data.zone_system,
            matrices.MatrixType.PA,
            data=long_data.data.reset_index(),
            columns=long_data.data.columns.to_list(),
        )

        pd.testing.assert_frame_equal(long_data.data, answer._data)

    def test_from_csv(
        self, long_data_csv: tuple[pathlib.Path, LongMatricesData]
    ) -> None:
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


class TestHDFMatrices:
    """Tests for HDF matrix storage specific functionality."""

    def test_to_csv(
        self,
        tmp_path: pathlib.Path,
        tp_hdf_matrices: "MatricesResults",
    ) -> None:
        """Test converting HDF matrices to CSV matrix files."""
        csv = tp_hdf_matrices.test.to_csv(tmp_path / "csv")
        assert isinstance(csv, matrices.MatrixFiles)

        for slice_ in tp_hdf_matrices.segmentation_.iter_slices():
            expected = tp_hdf_matrices.data[slice_]
            got = csv.get_matrix(slice_).data
            pd.testing.assert_frame_equal(
                expected,
                got,
                check_index_type=False,
                check_column_type=False,
                check_names=False,
            )

    def test_from_csv(
        self, tmp_path: pathlib.Path, tp_matrices_files: "MatricesResults"
    ) -> None:
        """Test converting CSV matrix files to HDF matrices."""
        hdf = tp_matrices_files.test.to_hdf(tmp_path / "matrices.h5")
        assert isinstance(hdf, matrices.HDFMatrices)

        for slice_ in tp_matrices_files.segmentation_.iter_slices():
            expected = tp_matrices_files.data[slice_]
            got = hdf.get_matrix(slice_).data
            pd.testing.assert_frame_equal(
                expected,
                got,
                check_index_type=False,
                check_column_type=False,
                check_names=False,
            )

    def test_save_and_load(
        self, tmp_path: pathlib.Path, tp_hdf_matrices: "MatricesResults"
    ) -> None:
        """Test saving and loading HDF matrices with metadata."""
        save_path = tp_hdf_matrices.test.save(tmp_path / "saved_hdf_matrices")
        assert save_path.suffix == ".dmat"
        loaded = matrices.HDFMatrices.load(save_path)

        assert loaded.type == tp_hdf_matrices.type_
        assert loaded.segmentation == tp_hdf_matrices.segmentation_
        assert loaded.zoning == tp_hdf_matrices.zoning_

        for slice_ in tp_hdf_matrices.segmentation_.iter_slices():
            expected = tp_hdf_matrices.data[slice_]
            got = loaded.get_matrix(slice_).data
            pd.testing.assert_frame_equal(
                expected,
                got,
                check_index_type=False,
                check_column_type=False,
                check_names=False,
            )

    def test_load_missing_metadata(self, tmp_path: pathlib.Path) -> None:
        """Test loading HDF matrices fails when metadata is missing."""
        path = tmp_path / "missing-metadata.h5"
        pd.DataFrame({"test": [1]}).to_hdf(path, key="test", mode="w", format="fixed")

        with pytest.raises(ValueError, match="does not contain HDFMatrices metadata"):
            matrices.HDFMatrices.load(path)

    def test_omx_file_compatibility(
        self, tmp_path: pathlib.Path, tp_hdf_matrices: "MatricesResults"
    ) -> None:
        """Test saved HDFMatrices can be read by OMXFile."""
        save_path = tp_hdf_matrices.test.save(tmp_path / "omx-compatible")

        with omx.OMXFile(save_path, mode="r") as file:
            expected_shape = (
                len(tp_hdf_matrices.zoning_),
                len(tp_hdf_matrices.zoning_),
            )
            assert file.shape == expected_shape
            assert np.array_equal(file.zones, tp_hdf_matrices.zoning_.zone_ids)

            for slice_ in tp_hdf_matrices.segmentation_.iter_slices():
                key = tp_hdf_matrices.test._get_key(slice_)
                assert key in file.matrix_levels
                expected = tp_hdf_matrices.data[slice_]
                got = file.get_matrix_level(key)
                pd.testing.assert_frame_equal(
                    expected,
                    got,
                    check_index_type=False,
                    check_column_type=False,
                    check_names=False,
                )

    def test_set_matrix_twice_does_not_rewrite_metadata(
        self, tmp_path: pathlib.Path, tp_hdf_matrices: "MatricesResults"
    ) -> None:
        """Test writing twice doesn't fail due to duplicate segmentation/zoning metadata."""
        saved_path = tp_hdf_matrices.test.save(tmp_path / "repeat-write")
        loaded = matrices.HDFMatrices.load(saved_path)

        slice_ = next(tp_hdf_matrices.segmentation_.iter_slices())
        matrix = tp_hdf_matrices.data[slice_]
        loaded.set_matrix(matrix, slice_)
        loaded.set_matrix(matrix * 2, slice_)

        got = loaded.get_matrix(slice_).data
        expected = matrix * 2
        pd.testing.assert_frame_equal(
            expected,
            got,
            check_index_type=False,
            check_column_type=False,
            check_names=False,
        )

    def test_to_mat_uses_cube_converter(
        self,
        tmp_path: pathlib.Path,
        tp_hdf_matrices: "MatricesResults",
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Test converting HDF matrices to CUBE .mat using CUBEMatConverter."""
        voyager_path = tmp_path / "VOYAGER.exe"
        voyager_path.write_text("", encoding="utf-8")
        output_path = tmp_path / "converted.mat"

        called: dict[str, pathlib.Path | bool | None] = {}

        class _Converter:
            def __init__(self, path: pathlib.Path) -> None:
                called["voyager_path"] = path

            def from_omx(
                self,
                omx_file: pathlib.Path,
                out_path: pathlib.Path | None = None,
                *,
                overwrite: bool = False,
            ) -> pathlib.Path:
                called["omx_file"] = omx_file
                called["out_path"] = out_path
                called["overwrite"] = overwrite
                return output_path

        monkeypatch.setattr(cube, "CUBEMatConverter", _Converter)

        result = tp_hdf_matrices.test.to_mat(voyager_path, output_path, overwrite=True)

        assert result == output_path
        assert called["voyager_path"] == voyager_path
        assert called["omx_file"] == tp_hdf_matrices.test.path
        assert called["out_path"] == output_path
        assert called["overwrite"] is True


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


Matrices = (
    matrices.MemoryMatrices
    | matrices.MatrixFiles
    | matrices.HDFMatrices
    | matrices.LongMatrices
)


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
            folder
            / f"{type_.name}_{slice_.generate_name(segmentation_.seg_dict)}.csv.bz2"
        )

    matrices_ = matrices.MatrixFiles(segmentation_, zone_system, type_, folder)

    return data, matrices_


def _produce_hdf_matrices(
    path: pathlib.Path,
    zone_system: zoning.ZoningSystem,
    segmentation_: segmentation.Segmentation,
    type_: matrices.MatrixType,
    name: str = "test matrices hdf",
    data: dict[segmentation.SegmentationSlice, pd.DataFrame] | None = None,
) -> tuple[dict[segmentation.SegmentationSlice, pd.DataFrame], matrices.HDFMatrices]:
    """Produce instance of HDFMatrices for testing."""
    if data is None:
        data = random_matrices_data(zone_system, segmentation_, seed=157)

    hdf_path = path / f"{name}.h5"
    matrices_ = matrices.HDFMatrices(
        segmentation_, zone_system, type_, hdf_path, check_file=False
    )
    for slice_, df in data.items():
        matrices_.set_matrix(df, slice_)

    matrices_ = matrices.HDFMatrices(segmentation_, zone_system, type_, hdf_path)
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
        df = df.stack()  # noqa: PLW2901
        df.index = pd.MultiIndex.from_arrays(
            [[i] * len(df) for i in slice_.as_tuple()]
            + [df.index.get_level_values(i) for i in (0, 1)],
            names=(*slice_.naming_order, "origin", "destination"),
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
    """PA segmentation containing time period for testing."""
    input_ = segmentation.SegmentationInput(
        enum_segments=["userclass", "direction", "tp"],  # type: ignore[arg-type]
        naming_order=["userclass", "direction", "tp"],
        subsets={"tp": [1, 2, 3]},
    )
    return segmentation.Segmentation(input_), matrices.MatrixType.PA


@pytest.fixture(name="tp_memory_matrices")
def fix_tp_memory_matrices(
    zone_system: cbase.ZoningSystem,
    tp_segmentation: tuple[segmentation.Segmentation, matrices.MatrixType],
) -> MatricesResults:
    """PA MemoryMatrices containing time period for testing."""
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
    zone_system: cbase.ZoningSystem,
    tp_segmentation: tuple[segmentation.Segmentation, matrices.MatrixType],
) -> MatricesResults:
    """PA MatricesFile containing time period for testing."""
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
    zone_system: cbase.ZoningSystem,
    tp_segmentation: tuple[segmentation.Segmentation, matrices.MatrixType],
) -> MatricesResults:
    """PA LongMatrices containing time period for testing."""
    data, matrices_ = _produce_long_matrices(zone_system, *tp_segmentation)

    return MatricesResults(
        test=matrices_,
        segmentation_=tp_segmentation[0],
        zoning_=zone_system,
        type_=tp_segmentation[1],
        data=data,
    )


@pytest.fixture(name="tp_hdf_matrices")
def fix_tp_hdf_matrices(
    tmp_path: pathlib.Path,
    zone_system: cbase.ZoningSystem,
    tp_segmentation: tuple[segmentation.Segmentation, matrices.MatrixType],
) -> MatricesResults:
    """PA HDFMatrices containing time period for testing."""
    data, matrices_ = _produce_hdf_matrices(tmp_path, zone_system, *tp_segmentation)
    return MatricesResults(
        test=matrices_,
        segmentation_=tp_segmentation[0],
        zoning_=zone_system,
        type_=tp_segmentation[1],
        data=data,
    )


@pytest.fixture(name="hb_segmentation")
def fix_hb_segmentation() -> tuple[segmentation.Segmentation, matrices.MatrixType]:
    """PA HB segmentation without time period for testing."""
    input_ = segmentation.SegmentationInput(
        enum_segments=["p", "direction"],  # type: ignore[arg-type]
        naming_order=["p", "direction"],
        subsets={"p": list(range(1, 9)), "direction": [1]},
    )
    return segmentation.Segmentation(input_), matrices.MatrixType.PA


@pytest.fixture(name="hb_memory_matrices")
def fix_hb_memory_matrices(
    zone_system: cbase.ZoningSystem,
    hb_segmentation: tuple[segmentation.Segmentation, matrices.MatrixType],
) -> MatricesResults:
    """PA HB MemoryMatrices without time period for testing."""
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
    zone_system: cbase.ZoningSystem,
    hb_segmentation: tuple[segmentation.Segmentation, matrices.MatrixType],
) -> MatricesResults:
    """PA HB MatricesFiles without time period for testing."""
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
    zone_system: cbase.ZoningSystem,
    hb_segmentation: tuple[segmentation.Segmentation, matrices.MatrixType],
) -> MatricesResults:
    """PA HB LongMatrices without time period for testing."""
    data, matrices_ = _produce_long_matrices(zone_system, *hb_segmentation)

    return MatricesResults(
        test=matrices_,
        segmentation_=hb_segmentation[0],
        zoning_=zone_system,
        type_=hb_segmentation[1],
        data=data,
    )


@pytest.fixture(name="hb_hdf_matrices")
def fix_hb_hdf_matrices(
    tmp_path: pathlib.Path,
    zone_system: cbase.ZoningSystem,
    hb_segmentation: tuple[segmentation.Segmentation, matrices.MatrixType],
) -> MatricesResults:
    """PA HB HDFMatrices without time period for testing."""
    data, matrices_ = _produce_hdf_matrices(tmp_path, zone_system, *hb_segmentation)
    return MatricesResults(
        test=matrices_,
        segmentation_=hb_segmentation[0],
        zoning_=zone_system,
        type_=hb_segmentation[1],
        data=data,
    )


@pytest.fixture(name="nhb_segmentation")
def fix_nhb_segmentation() -> tuple[segmentation.Segmentation, matrices.MatrixType]:
    """PA NHB segmentation without time period for testing."""
    input_ = segmentation.SegmentationInput(
        enum_segments=["p", "direction"],  # type: ignore[arg-type]
        naming_order=["p", "direction"],
        subsets={"p": [11, 12, 13, 14, 15, 16, 17, 18], "direction": [0]},
    )
    return segmentation.Segmentation(input_), matrices.MatrixType.PA


@pytest.fixture(name="nhb_memory_matrices")
def fix_nhb_memory_matrices(
    zone_system: cbase.ZoningSystem,
    nhb_segmentation: tuple[segmentation.Segmentation, matrices.MatrixType],
) -> MatricesResults:
    """PA NHB MemoryMatrices without time period for testing."""
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
    zone_system: cbase.ZoningSystem,
    nhb_segmentation: tuple[segmentation.Segmentation, matrices.MatrixType],
) -> MatricesResults:
    """PA NHB MatricesFiles without time period for testing."""
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
    zone_system: cbase.ZoningSystem,
    nhb_segmentation: tuple[segmentation.Segmentation, matrices.MatrixType],
) -> MatricesResults:
    """PA NHB LongMatrices without time period for testing."""
    data, matrices_ = _produce_long_matrices(zone_system, *nhb_segmentation)

    return MatricesResults(
        test=matrices_,
        segmentation_=nhb_segmentation[0],
        zoning_=zone_system,
        type_=nhb_segmentation[1],
        data=data,
    )


@pytest.fixture(name="nhb_hdf_matrices")
def fix_nhb_hdf_matrices(
    tmp_path: pathlib.Path,
    zone_system: cbase.ZoningSystem,
    nhb_segmentation: tuple[segmentation.Segmentation, matrices.MatrixType],
) -> MatricesResults:
    """PA NHB HDFMatrices without time period for testing."""
    data, matrices_ = _produce_hdf_matrices(tmp_path, zone_system, *nhb_segmentation)
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
            enum_segments=["userclass", "direction_od", "tp"],  # type: ignore[arg-type]
            naming_order=["userclass", "direction_od", "tp"],
            subsets={"tp": [1, 2, 3]},
        )
    )
    type_ = matrices.MatrixType.OD

    filter_tp = functools.partial(filter, lambda x: x != "tp")
    from_segmentation = segmentation.Segmentation(
        segmentation.SegmentationInput(
            enum_segments=list(filter_tp(to_segmentation.input.naming_order)),  # type: ignore[arg-type]
            naming_order=list(filter_tp(to_segmentation.input.naming_order)),
            subsets={
                i: j for i, j in to_segmentation.input.subsets.items() if i != "tp"
            },
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
    segments: tuple[cbase.Segment, cbase.Segment]


def _build_disaggregate_inputs_from_lookup(
    from_segmentation: segmentation.Segmentation,
    to_segmentation: segmentation.Segmentation,
    expected_matrices: dict[segmentation.SegmentationSlice, pd.DataFrame],
    from_segment: cbase.Segment,
    to_segment: cbase.Segment,
) -> dict[segmentation.SegmentationSlice, pd.DataFrame]:
    """Build aggregated input matrices by grouping expected matrices via segment lookup."""
    disaggregations = _slice_disaggregations_from_lookup(
        from_segmentation,
        to_segmentation,
        from_segment,
        to_segment,
    )
    agg_matrices = {}
    sample = next(iter(expected_matrices.values()))
    for from_slice in from_segmentation.iter_slices():
        to_slices = disaggregations.get(from_slice, [])
        total = pd.DataFrame(0.0, index=sample.index, columns=sample.columns)
        for to_slice in to_slices:
            total += expected_matrices[to_slice]
        agg_matrices[from_slice] = total
    return agg_matrices


def _lookup_groupings(
    from_segmentation: segmentation.Segmentation,
    from_segment: cbase.Segment,
    to_segment: cbase.Segment,
) -> dict[int, list[int]]:
    """Get source->target segment lookup groupings from caf.base lookup data."""
    try:
        _, lookup = from_segmentation.translate_segment(from_segment, to_segment)
    except FileNotFoundError:
        _, lookup = from_segmentation.translate_segment(
            from_segment,
            to_segment,
            reverse=True,
        )
        lookup = lookup.reset_index().set_index(lookup.name)

    return lookup.groupby(level=0).agg(list).squeeze().to_dict()


def _slice_disaggregations_from_lookup(
    from_segmentation: segmentation.Segmentation,
    to_segmentation: segmentation.Segmentation,
    from_segment: cbase.Segment,
    to_segment: cbase.Segment,
) -> dict[segmentation.SegmentationSlice, list[segmentation.SegmentationSlice]]:
    """Build slice-level disaggregation map using lookup groupings and target segmentation."""
    groupings = _lookup_groupings(from_segmentation, from_segment, to_segment)
    disaggregations: dict[
        segmentation.SegmentationSlice, list[segmentation.SegmentationSlice]
    ] = {}

    for from_slice in from_segmentation.iter_slices():
        to_slices: list[segmentation.SegmentationSlice] = []
        for to_seg in groupings.get(from_slice[from_segment.name], []):
            slice_data = from_slice.data | {to_segment.name: to_seg}
            slice_data.pop(from_segment.name)
            to_slices.extend(to_segmentation.iter_slices(slice_data))
        disaggregations[from_slice] = to_slices

    return disaggregations


@pytest.fixture(name="disaggregate_replace_datasets")
def fix_disaggregate_replace_datasets(
    zone_system: zoning.ZoningSystem,
) -> DisaggregateReplaceDatasets:
    """Datasets for testing disaggregate replace method."""
    segments = ["m"]
    subsets = {"m": [3]}
    from_segmentation = cbase.Segmentation(
        cbase.SegmentationInput(
            enum_segments=[*segments, "userclass", "direction_od"],
            naming_order=["direction_od", "userclass", *segments],
            subsets=subsets,
        )
    )
    from_segment = cbase.segments.SegmentsSuper.USERCLASS.get_segment()
    to_segment = cbase.segments.SegmentsSuper.PURPOSE.get_segment()
    lookup_groupings = _lookup_groupings(from_segmentation, from_segment, to_segment)
    lookup_purpose_values = sorted(
        {purpose for values in lookup_groupings.values() for purpose in values}
    )
    to_segmentation = cbase.Segmentation(
        cbase.SegmentationInput(
            enum_segments=[*segments, "p", "direction_od"],
            naming_order=["direction_od", "p", *segments],
            subsets=subsets | {"p": lookup_purpose_values},
        )
    )

    # Produce some random expected matrices and calculate inputs from these
    expected_matrices = random_matrices_data(zone_system, to_segmentation, seed=16549)
    agg_matrices = _build_disaggregate_inputs_from_lookup(
        from_segmentation,
        to_segmentation,
        expected_matrices,
        from_segment,
        to_segment,
    )

    return DisaggregateReplaceDatasets(
        zone_system=zone_system,
        type_=matrices.MatrixType.OD,
        from_segmentation=from_segmentation,
        to_segmentation=to_segmentation,
        input_=agg_matrices,
        targets={i: j / 3 for i, j in expected_matrices.items()},
        expected=expected_matrices,
        segments=(from_segment, to_segment),
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
    segments: tuple[cbase.Segment, cbase.Segment]


@pytest.fixture(name="disaggregate_replace_memory_matrices")
def fix_disaggregate_replace_memory_matrices(
    disaggregate_replace_datasets: DisaggregateReplaceDatasets,
) -> DisaggregateReplaceMatrices:
    """MemoryMatrices for testing disaggregate method with replace."""
    input_matrices = matrices.MemoryMatrices(
        disaggregate_replace_datasets.from_segmentation,
        disaggregate_replace_datasets.zone_system,
        disaggregate_replace_datasets.type_,
        [
            matrices.Matrix(j, i)
            for i, j in disaggregate_replace_datasets.input_.items()
        ],
    )
    target_matrices = matrices.MemoryMatrices(
        disaggregate_replace_datasets.to_segmentation,
        disaggregate_replace_datasets.zone_system,
        disaggregate_replace_datasets.type_,
        [
            matrices.Matrix(j, i)
            for i, j in disaggregate_replace_datasets.targets.items()
        ],
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
def fix_disaggregate_and_replace_datasets(
    zone_system: cbase.ZoningSystem,
) -> DisaggregateReplaceDatasets:
    """Datasets for the disaggregate and replace tests."""
    time_periods = [1, 2, 3]
    segments = ["m"]
    subsets = {"m": [3]}
    from_segmentation = cbase.Segmentation(
        cbase.SegmentationInput(
            enum_segments=[*segments, "userclass", "direction_od"],
            naming_order=["direction_od", "userclass", *segments],
            subsets=subsets,
        )
    )
    from_segment = cbase.segments.SegmentsSuper.USERCLASS.get_segment()
    to_segment = cbase.segments.SegmentsSuper.PURPOSE.get_segment()
    lookup_groupings = _lookup_groupings(from_segmentation, from_segment, to_segment)
    lookup_purpose_values = sorted(
        {purpose for values in lookup_groupings.values() for purpose in values}
    )
    to_segmentation = cbase.Segmentation(
        cbase.SegmentationInput(
            enum_segments=[*segments, "p", "direction_od", "tp"],
            naming_order=["direction_od", "p", *segments, "tp"],
            subsets=subsets | {"tp": time_periods, "p": lookup_purpose_values},
        )
    )

    # Produce some random expected matrices and calculate inputs from these
    expected_matrices = random_matrices_data(zone_system, to_segmentation, seed=16879)
    agg_matrices = _build_disaggregate_inputs_from_lookup(
        from_segmentation,
        to_segmentation,
        expected_matrices,
        from_segment,
        to_segment,
    )

    return DisaggregateReplaceDatasets(
        zone_system=zone_system,
        type_=matrices.MatrixType.OD,
        from_segmentation=from_segmentation,
        to_segmentation=to_segmentation,
        input_=agg_matrices,
        targets={i: j / 3 for i, j in expected_matrices.items()},
        expected=expected_matrices,
        segments=(from_segment, to_segment),
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
        [
            matrices.Matrix(j, i)
            for i, j in disaggregate_and_replace_datasets.input_.items()
        ],
    )
    target_matrices = matrices.MemoryMatrices(
        disaggregate_and_replace_datasets.to_segmentation,
        disaggregate_and_replace_datasets.zone_system,
        disaggregate_and_replace_datasets.type_,
        [
            matrices.Matrix(j, i)
            for i, j in disaggregate_and_replace_datasets.targets.items()
        ],
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
        "matrices_",
        ["tp_memory_matrices", "tp_matrices_files", "tp_hdf_matrices", "tp_long_matrices"],
    )
    def test_segmentation(self, request: pytest.FixtureRequest, matrices_: str) -> None:
        """Test segmentation property returns expected Segmentation."""
        result: MatricesResults = request.getfixturevalue(matrices_)
        assert result.test.segmentation == result.segmentation_

    @pytest.mark.parametrize(
        "matrices_",
        ["tp_memory_matrices", "tp_matrices_files", "tp_hdf_matrices", "tp_long_matrices"],
    )
    def test_zoning(self, request: pytest.FixtureRequest, matrices_: str) -> None:
        """Test zoning property returns expected ZoningSystem."""
        result: MatricesResults = request.getfixturevalue(matrices_)
        assert result.test.zoning == result.zoning_

    @pytest.mark.parametrize(
        "matrices_",
        ["tp_memory_matrices", "tp_matrices_files", "tp_hdf_matrices", "tp_long_matrices"],
    )
    def test_type(self, request: pytest.FixtureRequest, matrices_: str) -> None:
        """Test type property returns expected MatrixType."""
        result: MatricesResults = request.getfixturevalue(matrices_)
        assert result.test.type == result.type_

    @pytest.mark.parametrize(
        "matrices_",
        ["tp_memory_matrices", "tp_matrices_files", "tp_hdf_matrices", "tp_long_matrices"],
    )
    def test_name(self, request: pytest.FixtureRequest, matrices_: str) -> None:
        """Test name property returns a string."""
        result: MatricesResults = request.getfixturevalue(matrices_)
        assert isinstance(result.test.name, str)

    @pytest.mark.parametrize(
        "matrices_",
        ["tp_memory_matrices", "tp_matrices_files", "tp_hdf_matrices", "tp_long_matrices"],
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
                check_names=False,
            )

    @pytest.mark.parametrize(
        "matrices_",
        ["tp_memory_matrices", "tp_matrices_files", "tp_hdf_matrices", "tp_long_matrices"],
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
        "matrices_",
        ["tp_memory_matrices", "tp_matrices_files", "tp_hdf_matrices", "tp_long_matrices"],
    )
    @pytest.mark.parametrize(
        "new_seg_input",
        [
            None,
            segmentation.SegmentationInput(
                enum_segments=["p", "m"],  # type: ignore[arg-type]
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

        with warnings.catch_warnings(
            action="ignore", category=matrices.MatricesWarning
        ):
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
            ("tp_hdf_matrices", True),
            ("tp_long_matrices", True),
            ("hb_memory_matrices", False),
            ("hb_matrices_files", False),
            ("hb_hdf_matrices", False),
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
            ("tp_hdf_matrices", False),
            ("hb_memory_matrices", True),
            ("hb_matrices_files", True),
            ("hb_hdf_matrices", True),
            ("hb_long_matrices", True),
            ("nhb_memory_matrices", False),
            ("nhb_matrices_files", False),
            ("nhb_hdf_matrices", False),
            ("nhb_long_matrices", False),
        ],
    )
    def test_home_based_only(
        self, request: pytest.FixtureRequest, matrices_: str, home_based_only: bool
    ) -> None:
        """Test home_based_only method works correctly."""
        result: MatricesResults = request.getfixturevalue(matrices_)
        assert result.test.is_home_based_only == home_based_only

    @pytest.mark.parametrize(
        "matrices_, non_home_based_only",
        [
            ("tp_memory_matrices", False),
            ("tp_matrices_files", False),
            ("tp_hdf_matrices", False),
            ("hb_memory_matrices", False),
            ("hb_matrices_files", False),
            ("hb_hdf_matrices", False),
            ("hb_long_matrices", False),
            ("nhb_memory_matrices", True),
            ("nhb_matrices_files", True),
            ("nhb_hdf_matrices", True),
            ("nhb_long_matrices", True),
        ],
    )
    def test_non_home_based_only(
        self, request: pytest.FixtureRequest, matrices_: str, non_home_based_only: bool
    ) -> None:
        """Test non_home_based_only method works correctly."""
        result: MatricesResults = request.getfixturevalue(matrices_)
        assert result.test.is_non_home_based_only == non_home_based_only

    @pytest.mark.parametrize(
        "matrices_, params",
        [
            ("tp_memory_matrices", {"userclass": 3, "direction": 1, "tp": 2}),
            ("tp_matrices_files", {"userclass": 3, "direction": 1, "tp": 2}),
            ("tp_hdf_matrices", {"userclass": 3, "direction": 1, "tp": 2}),
            ("tp_long_matrices", {"userclass": 3, "direction": 1, "tp": 2}),
        ],
    )
    def test_validate_slice(
        self, request: pytest.FixtureRequest, matrices_: str, params: dict[str, int]
    ) -> None:
        """Test `validate_slice` with valid slices."""
        result: MatricesResults = request.getfixturevalue(matrices_)

        slice_ = segmentation.SegmentationSlice(
            params, result.segmentation_.naming_order
        )
        result.test.validate_slice(slice_)

    @pytest.mark.parametrize(
        "matrices_, params",
        [
            ("tp_memory_matrices", {"userclass": 7, "direction": 3, "tp": 5}),
            ("tp_matrices_files", {"userclass": 7, "direction": 3, "tp": 5}),
            ("tp_hdf_matrices", {"userclass": 7, "direction": 3, "tp": 5}),
            ("tp_long_matrices", {"userclass": 7, "direction": 3, "tp": 5}),
        ],
    )
    def test_validate_slice_invalid(
        self, request: pytest.FixtureRequest, matrices_: str, params: dict[str, int]
    ) -> None:
        """Test `validate_slice` correctly raises error."""
        result: MatricesResults = request.getfixturevalue(matrices_)
        slice_ = segmentation.SegmentationSlice(
            params, result.segmentation_.naming_order
        )

        with pytest.raises(ValueError):
            result.test.validate_slice(slice_)

    @pytest.mark.parametrize(
        "matrices_",
        ["tp_memory_matrices", "tp_matrices_files", "tp_hdf_matrices", "tp_long_matrices"],
    )
    def test_validate_matrix(
        self, request: pytest.FixtureRequest, matrices_: str
    ) -> None:
        """Test `validate_matrix` for valid matrix."""
        result: MatricesResults = request.getfixturevalue(matrices_)

        rand = random_matrices(result.zoning_)
        result.test.validate_matrix(next(rand), "Valid matrix")

    @pytest.mark.parametrize(
        "matrices_",
        ["tp_memory_matrices", "tp_matrices_files", "tp_hdf_matrices", "tp_long_matrices"],
    )
    def test_validate_matrix_invalid(
        self, request: pytest.FixtureRequest, matrices_: str
    ) -> None:
        """Test `validate_matrix` correctly raises error."""
        result: MatricesResults = request.getfixturevalue(matrices_)

        zones = pd.DataFrame({"zone_id": [*result.zoning_.zone_ids.tolist(), -1]})
        zoning_ = zoning.ZoningSystem(
            "invalid zones", zones, zoning.ZoningSystemMetaData(name="invalid zones")
        )
        rand = random_matrices(zoning_)

        with pytest.raises(ValueError):
            result.test.validate_matrix(next(rand), "Invalid matrix")

    @pytest.mark.parametrize(
        "matrices_",
        [
            "aggregate_memory_matrices",
            "aggregate_matrices_files",
            "aggregate_long_matrices",
        ],
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
                check_names=False,
            )

    @pytest.mark.parametrize(
        "matrices_",
        [
            "aggregate_memory_matrices",
            "aggregate_matrices_files",
            "aggregate_long_matrices",
        ],
    )
    def test_lazy_translate_then_aggregate(
        self, request: pytest.FixtureRequest, matrices_: str
    ) -> None:
        """Test lazy translate_zoning().aggregate() matches eager execution."""
        result: AggregateMatrices = request.getfixturevalue(matrices_)

        new_zoning = cbase.ZoningSystem(
            "translated",
            pd.DataFrame({"zone_id": [10, 20]}),
            zoning.ZoningSystemMetaData(name="translated"),
        )
        source_zone = result.input_.zoning
        translation = pd.DataFrame(
            {
                source_zone.column_name: [1, 2, 3],
                new_zoning.column_name: [10, 20, 20],
                source_zone.translation_column_name(new_zoning): [1.0, 1.0, 1.0],
            }
        )

        eager = result.input_.translate_zoning(new_zoning, translation).aggregate(
            result.to_segmentation, progress_bar=False
        )
        lazy = (
            result.input_
            .lazy()
            .translate_zoning(new_zoning, translation=translation)
            .aggregate(result.to_segmentation, progress_bar=False)
            .collect()
        )

        assert isinstance(lazy, type(result.input_))
        assert lazy.segmentation == eager.segmentation
        assert lazy.zoning == eager.zoning

        for slice_ in eager.segmentation.iter_slices():
            pd.testing.assert_frame_equal(
                eager.get_matrix(slice_).data,
                lazy.get_matrix(slice_).data,
                check_column_type=False,
                check_index_type=False,
                check_names=False,
            )

    @pytest.mark.parametrize(
        "matrices_",
        [
            "aggregate_memory_matrices",
            "aggregate_matrices_files",
            "aggregate_long_matrices",
        ],
    )
    def test_lazy_default_method_queueing(
        self, request: pytest.FixtureRequest, matrices_: str
    ) -> None:
        """Test lazy pipelines queue methods via __getattr__ without explicit stubs."""
        result: AggregateMatrices = request.getfixturevalue(matrices_)

        eager = result.input_.remove_intras().aggregate(
            result.to_segmentation,
            progress_bar=False,
        )
        lazy = (
            result.input_
            .lazy()
            .remove_intras()
            .aggregate(result.to_segmentation, progress_bar=False)
            .collect()
        )

        assert isinstance(lazy, type(result.input_))
        assert lazy.segmentation == eager.segmentation
        assert lazy.zoning == eager.zoning

        for slice_ in eager.segmentation.iter_slices():
            pd.testing.assert_frame_equal(
                eager.get_matrix(slice_).data,
                lazy.get_matrix(slice_).data,
                check_column_type=False,
                check_index_type=False,
                check_names=False,
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
                check_names=False,
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
                check_names=False,
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
                check_names=False,
            )
