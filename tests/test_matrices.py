# -*- coding: utf-8 -*-
"""Tests for `matrices` module."""

##### IMPORTS #####

# Built-Ins
import itertools
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


##### Tests & Fixtures for `MemoryMatrices` #####


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


class TestMemoryMatrices:
    """Tests for the `MemoryMatrices` class.

    Using the `MemoryMatrices` sub-class to test the
    functionality implemented on the ABC.
    """

    def test_disaggregate(self, zone_system: base.ZoningSystem) -> None:
        """Test disaggregating to one new segment, with no replacements."""
        time_periods = [1, 2, 3]
        segments = ["direction_od", "userclass", "m"]
        subsets = {"m": [3]}
        from_segmentation = base.Segmentation(
            base.SegmentationInput(
                enum_segments=segments, naming_order=segments, subsets=subsets
            )
        )
        to_segmentation = base.Segmentation(
            base.SegmentationInput(
                enum_segments=segments + ["tp"],
                naming_order=segments + ["tp"],
                subsets=subsets | {"tp": time_periods},
            )
        )

        # Produce some random expected matrices and calculate inputs from these
        rng_matrices = random_matrices(zone_system)
        expected_matrices = {i: next(rng_matrices) for i in to_segmentation.iter_slices()}

        agg_matrices = []
        for slice_ in from_segmentation.iter_slices():
            total = 0
            for tp in time_periods:
                to_slice = segmentation.SegmentationSlice(
                    slice_.data | {"tp": tp}, to_segmentation.naming_order
                )
                total += expected_matrices[to_slice]

            agg_matrices.append(matrices.Matrix(total, slice_))

        aggregate = matrices.MemoryMatrices(
            from_segmentation, zone_system, matrices.MatrixType.OD, agg_matrices
        )

        # Divide expected matrices by constant to have a consistent ratio between them
        targets = matrices.MemoryMatrices(
            to_segmentation,
            zone_system,
            type_=matrices.MatrixType.OD,
            matrices=[matrices.Matrix(j / 3, i) for i, j in expected_matrices.items()],
        )

        output = aggregate.disaggregate(targets)
        assert output.zoning == zone_system
        assert output.segmentation == to_segmentation

        for slice_ in output.segmentation.iter_slices():
            pd.testing.assert_frame_equal(
                output.get_matrix(slice_).data,
                expected_matrices[slice_],
                obj=f"{slice_} DataFrame",
            )

    def test_disaggregate_replace(self, zone_system: base.ZoningSystem) -> None:
        """Test dissagregating userclasses into purposes, with no new segments."""
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
        rng_matrices = random_matrices(zone_system)
        expected_matrices = {i: next(rng_matrices) for i in to_segmentation.iter_slices()}

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
        agg_matrices = []
        for direction, uc, purposes in aggregations:
            total = 0

            for p in purposes:
                slice_ = segmentation.SegmentationSlice(
                    {"direction_od": direction, "p": p, "m": 3}, to_segmentation.naming_order
                )
                total += expected_matrices[slice_]

            agg_matrices.append(
                matrices.Matrix(
                    total,
                    segmentation.SegmentationSlice(
                        {"direction_od": direction, "userclass": uc, "m": 3},
                        from_segmentation.naming_order,
                    ),
                )
            )

        aggregate = matrices.MemoryMatrices(
            from_segmentation,
            zone_system,
            type_=matrices.MatrixType.OD,
            matrices=agg_matrices,
        )

        # Divide expected matrices by constant to have a consistent ratio between them
        targets = matrices.MemoryMatrices(
            to_segmentation,
            zone_system,
            type_=matrices.MatrixType.OD,
            matrices=[matrices.Matrix(j / 3, i) for i, j in expected_matrices.items()],
        )

        output = aggregate.disaggregate(
            targets,
            base.segments.SegmentsSuper.USERCLASS.get_segment(),
            base.segments.SegmentsSuper.PURPOSE.get_segment(),
        )
        assert output.zoning == zone_system
        assert output.segmentation == to_segmentation

        for slice_ in output.segmentation.iter_slices():
            pd.testing.assert_frame_equal(
                output.get_matrix(slice_).data,
                expected_matrices[slice_],
                obj=f"{slice_} DataFrame",
            )

    def test_disaggregate_and_replace(self, zone_system: base.ZoningSystem) -> None:
        """Test dissagregating userclasses to purposes and new time periods."""
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
        rng_matrices = random_matrices(zone_system)
        expected_matrices = {i: next(rng_matrices) for i in to_segmentation.iter_slices()}

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
        agg_matrices = []
        for direction, uc, purposes in aggregations:
            total = 0

            for p in purposes:
                for tp in time_periods:
                    slice_ = segmentation.SegmentationSlice(
                        {"direction_od": direction, "p": p, "m": 3, "tp": tp},
                        to_segmentation.naming_order,
                    )
                    total += expected_matrices[slice_]

            agg_matrices.append(
                matrices.Matrix(
                    total,
                    segmentation.SegmentationSlice(
                        {"direction_od": direction, "userclass": uc, "m": 3},
                        from_segmentation.naming_order,
                    ),
                )
            )

        aggregate = matrices.MemoryMatrices(
            from_segmentation,
            zone_system,
            type_=matrices.MatrixType.OD,
            matrices=agg_matrices,
        )

        # Divide expected matrices by constant to have a consistent ratio between them
        targets = matrices.MemoryMatrices(
            to_segmentation,
            zone_system,
            type_=matrices.MatrixType.OD,
            matrices=[matrices.Matrix(j / 3, i) for i, j in expected_matrices.items()],
        )

        output = aggregate.disaggregate(
            targets,
            base.segments.SegmentsSuper.USERCLASS.get_segment(),
            base.segments.SegmentsSuper.PURPOSE.get_segment(),
        )
        assert output.zoning == zone_system
        assert output.segmentation == to_segmentation

        for slice_ in output.segmentation.iter_slices():
            pd.testing.assert_frame_equal(
                output.get_matrix(slice_).data,
                expected_matrices[slice_],
                obj=f"{slice_} DataFrame",
            )

    def test_aggregate(self, zone_system: base.ZoningSystem) -> None:
        """Test basic case for the `aggregate` method."""
        subsets = {"direction": [1], "p": list(range(1, 9)), "tp": [1, 2, 3]}

        input_seg = segmentation.Segmentation(
            segmentation.SegmentationInput(
                enum_segments=list(subsets), naming_order=list(subsets), subsets=subsets
            )
        )
        agg_seg = segmentation.Segmentation(
            segmentation.SegmentationInput(
                enum_segments=["direction", "p"],
                naming_order=["direction", "p"],
                subsets={i: j for i, j in subsets.items() if i != "tp"},
            )
        )

        input_matrices = []
        expected_matrices = []
        rng_matrices = random_matrices(zone_system)

        for dir_, p in itertools.product(subsets["direction"], subsets["p"]):
            params = {"direction": dir_, "p": p}
            total = 0

            for tp in subsets["tp"]:
                data = next(rng_matrices)
                total += data
                input_matrices.append(
                    matrices.Matrix(
                        data,
                        segmentation.SegmentationSlice(
                            params | {"tp": tp}, input_seg.naming_order
                        ),
                    )
                )

            expected_matrices.append(
                matrices.Matrix(
                    total,
                    segmentation.SegmentationSlice(params, agg_seg.naming_order),
                )
            )

        expected = matrices.MemoryMatrices(
            segmentation_=agg_seg,
            zoning=zone_system,
            type_=matrices.MatrixType.PA,
            matrices=expected_matrices,
        )
        disaggregated = matrices.MemoryMatrices(
            segmentation_=input_seg,
            zoning=zone_system,
            type_=matrices.MatrixType.PA,
            matrices=input_matrices,
        )

        aggregated = disaggregated.aggregate(agg_seg)

        for slice_ in agg_seg.iter_slices():
            pd.testing.assert_frame_equal(
                expected.get_matrix(slice_).data, aggregated.get_matrix(slice_).data
            )
