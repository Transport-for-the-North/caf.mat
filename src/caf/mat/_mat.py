# -*- coding: utf-8 -*-
"""Main module for caf.mat package."""

##### IMPORTS #####

# Built-Ins
import abc
import logging
import pathlib

##### CONSTANTS #####

LOG = logging.getLogger(__name__)


##### CLASSES & FUNCTIONS #####


class MatrixError(Exception):
    """Generic matrix error from caf.mat."""


class ArgumentHandler(abc.ABC):
    """Define required methods for argument sub-command classes."""

    @abc.abstractmethod
    def run(self) -> None:
        """Run the sub-command functionality."""

    @property
    @abc.abstractmethod
    def log_path(self) -> pathlib.Path:
        """Define path to log file for sub-command."""
