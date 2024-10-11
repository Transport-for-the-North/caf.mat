# -*- coding: utf-8 -*-
"""Handling the PA and OD matrix files."""

##### IMPORTS #####

# Built-Ins
import abc
import copy
import dataclasses
import logging
import pathlib
import re
from typing import Iterator, Literal, Self

# Third Party
import pandas as pd
from caf.core import segmentation, segments

##### CONSTANTS #####

LOG = logging.getLogger(__name__)


##### CLASSES & FUNCTIONS #####


@dataclasses.dataclass
class Matrix:
    data: pd.DataFrame
    segment: dict[str, int]


class BasePAMatrices(abc.ABC):

    segmentation: segmentation.Segmentation

    def __iter__(self) -> Iterator[Matrix]:
        """Iterate through all segments and provide the matrix.

        Time period information should be included in the segmentation.
        """
        for values in self.segmentation.ind():
            yield self.get_matrix(values)

    def _segment_dict_to_tuple(self, segment: dict[str, int]) -> tuple[int, ...]:
        return tuple(segment[i] for i in self.segmentation.naming_order)

    @abc.abstractmethod
    def get_matrix(self, segment: dict[str, int] | tuple[int]) -> Matrix:
        raise NotImplementedError()

    @property
    def has_time_periods(self) -> bool:
        # TODO check this method works correctly with all time periods Segments
        # TODO get information about the time format, e.g. avg hour / period
        return self.segmentation.has_time_period_segments()

    def _purpose_subsets(self) -> tuple[segments.SegmentsSuper, dict[str, list[int]]]:
        pattern = re.compile(r"\b(N?HB)\b", re.I)
        segment = segments.SegmentsSuper.PURPOSE
        subsets: dict[str, list[int]] = {"hb": [], "nhb": []}

        for value, name in segment.get_segment().values.items():
            match = pattern.search(value)

            if match is None:
                raise ValueError(f"found unrecognised purpose: {name}")

            hb = match.group(1).lower()
            subsets[hb].append(value)

        return segment, subsets

    def _check_purpose_subsets(self, direction: Literal["hb", "nhb"]) -> bool:
        segment, subset = self._purpose_subsets()
        values = subset[direction]

        try:
            purposes = self.segmentation.seg_dict[segment.value]
        except KeyError as exc:
            raise ValueError(
                f"segmentation doesn't contain purposes '{segment.value}'"
            ) from exc

        return set(values) == set(purposes)

    @property
    def home_based_only(self) -> bool:
        """Return True if segmentation contains home-based purposes only."""
        return self._check_purpose_subsets("hb")

    @property
    def non_home_based_only(self) -> bool:
        """Return True if segmentation contains non-home-based purposes only."""
        return self._check_purpose_subsets("nhb")

    def get_home_based(self) -> Self:
        if self.home_based_only:
            return self
        if self.non_home_based_only:
            raise ValueError("PA matrices contains non-home-based data only")

        segment, subset = self._purpose_subsets()
        new_segmentation = self.segmentation.update_subsets({segment.value: subset["hb"]})

        new_matrices = copy.copy(self)
        new_matrices.segmentation = new_segmentation

        return new_matrices

    def get_non_home_based(self) -> Self:
        if self.non_home_based_only:
            return self
        if self.home_based_only:
            raise ValueError("PA matrices contains home-based data only")

        segment, subset = self._purpose_subsets()
        new_segmentation = self.segmentation.update_subsets({segment.value: subset["nhb"]})

        new_matrices = copy.copy(self)
        new_matrices.segmentation = new_segmentation

        return new_matrices


@dataclasses.dataclass
class OutputODMatrices:
    """Paths to all output OD matrices and HB / NHB factors."""
