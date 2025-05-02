# -*- coding: utf-8 -*-
"""Time period and from / to home factors required for PA and OD conversions."""

##### IMPORTS #####

# Built-Ins
import abc
import logging
from typing import Iterator

# Third Party
from caf.base import segments

# Local Imports
from caf.mat import matrices

##### CONSTANTS #####

LOG = logging.getLogger(__name__)


##### CLASSES & FUNCTIONS #####


class TimePeriod(abc.ABC):

    def get(self, _slice: dict[str, int]) -> list[matrices.Matrix]:
        """Get matrix of TP factors for given segment."""


class FromToHome(abc.ABC):

    def get_from(self, _slice: dict[str, int]) -> matrices.Matrix:
        """Get matrix of from home factors for each output time period."""

    def get_to(self, _slice: dict[str, int]) -> matrices.Matrix: ...


class MissingTP(abc.ABC):
    """Scaling factors for handling missing time periods during combining."""
