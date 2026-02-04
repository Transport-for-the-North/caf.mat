"""Main module for caf.mat package."""

##### IMPORTS #####

# Built-Ins
import abc
import enum
import logging
import pathlib
from typing import Self

# Third Party
import pydantic

##### CONSTANTS #####

LOG = logging.getLogger(__name__)


##### CLASSES & FUNCTIONS #####


class MatrixError(Exception):
    """Generic matrix error from caf.mat."""


class ArgumentHandler(pydantic.BaseModel, abc.ABC):
    """Define required methods for argument sub-command classes."""

    @abc.abstractmethod
    def run(self) -> None:
        """Run the sub-command functionality."""

    @property
    @abc.abstractmethod
    def log_path(self) -> pathlib.Path:
        """Define path to log file for sub-command."""


class MatrixFileFormat(enum.StrEnum):
    """File formats for matrices."""

    OMX = enum.auto()
    UFM = enum.auto()
    CUBE = enum.auto()
    SQUARE_CSV = enum.auto()
    LONG_CSV = enum.auto()

    @classmethod
    def _missing_(cls, value) -> Self | None:
        value = str(value).lower()
        for member in cls:
            if member.value == value:
                return member
        return None

    @classmethod
    def _suffix_lookup(cls) -> dict["MatrixFileFormat", str]:
        return {
            cls.OMX: ".omx",
            cls.UFM: ".ufm",
            cls.CUBE: ".mat",
            **dict.fromkeys((cls.SQUARE_CSV, cls.LONG_CSV), ".csv"),
        }

    def suffix(self) -> str:
        """File suffix for specific format."""
        lookup = self._suffix_lookup()
        if self not in lookup:
            raise ValueError(f"unknown suffix for {self}")
        return lookup[self]

    @classmethod
    def infer(cls, path: pathlib.Path) -> "MatrixFileFormat":
        """Infer matrix format from file suffix."""
        lookup = cls._suffix_lookup()

        suffix = path.suffix.strip().lower()
        if suffix in lookup.values():
            matching = [i for i, j in lookup.items() if j == suffix]
            if len(matching) == 1:
                return matching[0]
            if len(matching) > 1:
                raise ValueError(f"{suffix} matches {len(matching)} matrix formats")

        raise ValueError(f"unknown matrix format '{path.name}'")
