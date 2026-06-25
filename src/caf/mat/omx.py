# -*- coding: utf-8 -*-
"""Module for reading from and writing to OMX files."""

##### IMPORTS #####

# Built-Ins
import logging
import pathlib
import warnings
from collections.abc import Generator
from pathlib import Path
from typing import Literal, Self

# Third Party
import numpy as np
import pandas as pd
import tables

##### CONSTANTS #####

LOG = logging.getLogger(__name__)


##### CLASSES & FUNCTIONS #####


class OMXWarning(RuntimeWarning):
    """Warnings related to OMX files."""


class OMXFile(tables.File):
    """Reading and writing data to OMX files, which use the HDF5 format.

    This class wraps pytables `File` object and provides some helper
    methods and checks for working with OMX files specifically.

    Parameters
    ----------
    filename : str
        The name of the file (supports environment variable expansion).
        It is suggested that file names have any of the .h5, .hdf or
        .hdf5 extensions, although this is not mandatory.
    mode : str, default 'r'
        The mode to open the file. It can be one of the
        following:
            * *'r'*: Read-only; no data can be modified.
            * *'w'*: Write; a new file is created (an existing file
            with the same name would be deleted).
            * *'a'*: Append; an existing file is opened for reading
            and writing, and if the file does not exist it is created.
            * *'r+'*: It is similar to 'a', but the file must already
            exist.
    omx_version : str, optional
        Version of the OMX file, this option is ignored unless
        mode = 'w', when it is mandatory. Expected OMX version is '0.2'.
    shape : str, optional
        Shape of the matrices in the OMX file, this option is ignored
        unless mode = 'w', when it is mandatory.
    kwargs : Keyword arguments, optional
        All other keyword arguments are passed to `tables.File`.

    Raises
    ------
    ValueError
        If a `mode` other than those defined above is provided or
        `omx_version` and `shape` aren't provided in `mode` 'w'.
    """

    _EXPECTED_OMX_VERSION = "0.2"
    _DATA_NODE = "/data"
    _LOOKUP_NODE = "/lookup"
    _KEYS = {
        "version": "OMX_VERSION",
        "shape": "SHAPE",
        "zones": "ZoneNames",
    }

    def __init__(
        self,
        filename: Path,
        mode: str = "r",
        omx_version: str = _EXPECTED_OMX_VERSION,
        shape: tuple[int, int] | None = None,
        zones: np.ndarray | None = None
        **kwargs,
    ) -> None:
        self.mode = str(mode).strip().lower()
        if "filters" not in kwargs:
            kwargs["filters"] = tables.Filters(
                complevel=1, complib="zlib", shuffle=False
            )
        super().__init__(filename, mode=self.mode, **kwargs)

        self._path = Path(filename)
        self._zones = zones
        self._matrix_levels = None
        self._omx_version = None
        self._shape = None

        if self.mode in ("r", "a", "r+"):
            self._omx_version = self._check_omx_version(
                self.root._v_attrs[self._KEYS["version"]].decode()
            )
            self._shape = self._check_shape(self.root._v_attrs[self._KEYS["shape"]])
            self._zones = self._get_zones()
            self.get_node(self._DATA_NODE)

        elif self.mode == "w":
            if omx_version is None or shape is None:
                raise ValueError(
                    "omx_version and shape keyword arguments "
                    f"should be provided if mode is '{mode}'"
                )
            self.omx_version = omx_version
            self.shape = shape
            self._create_omx_nodes()

        else:
            raise ValueError(
                f"unknown mode '{mode}' should be one of 'r', 'a', 'r+' or 'w'"
            )

    def __enter__(self) -> Self:
        """Enter context and return the same OMXFile."""
        return self

    def _can_write(self, name: str) -> None:
        """Raise ValueError if not in a writing mode."""
        if self.mode == "r":
            raise ValueError(f"cannot set {name} in mode = {self.mode}")

    def _check_omx_version(self, value: str) -> str:
        """Warns user if OMX version isn't the expected one."""
        value = str(value).strip()
        if value != self._EXPECTED_OMX_VERSION:
            warnings.warn(
                f"OMXFile expects OMX version {self._EXPECTED_OMX_VERSION} "
                f"but got {value}, which may be incompatible"
            )
        return value
    
    def csv_to_omx(self, csv_paths: dict[str, Path]):
        for name, csv_path in csv_paths.items():
            df = pd.read_csv(csv_path, index_col=0)
            df.columns = df.columns.astype(int)
            self.set_matrix_level(name, df)

    @staticmethod
    def _check_shape(value: tuple[int, int]) -> tuple[int, int]:
        """Raise ValueError if shape isn't valid."""
        value = tuple(int(i) for i in value)
        if len(value) != 2:
            raise ValueError(
                f"shape should be a tuple of lenght 2 not length {len(value)}"
            )
        if value[0] != value[1]:
            raise ValueError(
                f"matrix should have the same number of rows and columns not {value}"
            )
        return value

    def _check_zones(self, value: np.ndarray) -> np.ndarray:
        """Raise ValueError if zones aren't the correct shape."""
        value = np.array(value)
        if value.shape != (self.shape[0],):
            raise ValueError(
                f"zones should be a 1D array with length {self.shape[0]} not {value.shape}"
            )
        return value

    def _get_zones(self) -> np.ndarray:
        """Attempt to read ZoneNames from file, otherwise uses sequential zones from 1."""
        try:
            zones = self.get_node(self._LOOKUP_NODE, self._KEYS["zones"])
        except tables.NoSuchNodeError:
            warnings.warn(
                f"no zone names found at {self._LOOKUP_NODE}/{self._KEYS['zones']},"
                " defaulting to integers starting at 1",
                OMXWarning,
                stacklevel=2,
            )
            zones = np.arange(1, self.shape[0] + 1)

        return self._check_zones(zones)

    def _create_omx_nodes(self) -> None:
        def remove_slash(value: str) -> str:
            """Remove starting slash from nodes."""
            return value.removeprefix("/")

        self.create_group("/", remove_slash(self._LOOKUP_NODE))
        self.create_group("/", remove_slash(self._DATA_NODE))

    @property
    def omx_version(self) -> str:
        """OMX version of the current file."""
        if self._omx_version is None:
            raise ValueError("unknown OMX version")
        return self._omx_version

    @omx_version.setter
    def omx_version(self, value: str) -> None:
        self._can_write("omx_version")
        value = self._check_omx_version(value)
        if value != self._omx_version:
            self._omx_version = value
            # pylint: disable=protected-access
            self.root._v_attrs[self._KEYS["version"]] = self._omx_version.encode(
                encoding="ascii"
            )

    @property
    def shape(self) -> tuple[int, int]:
        """Shape of the matrices in the current OMX file."""
        if self._shape is None:
            raise ValueError("unknown matrix shape")
        return self._shape

    @shape.setter
    def shape(self, value: tuple[int, int]) -> None:
        self._can_write("shape")
        value = self._check_shape(value)
        if value != self._shape:
            self._shape = value
            # pylint: disable=protected-access
            self.root._v_attrs[self._KEYS["shape"]] = np.array(
                self._shape, dtype=np.int32
            )

    @property
    def zones(self) -> np.ndarray:
        """Array of zone names for the current OMX file."""
        if self._zones is None:
            raise ValueError("OMX zones not yet defined")

        return self._zones

    @zones.setter
    def zones(self, value: np.ndarray) -> None:
        self._can_write("zones")
        value = self._check_zones(value)
        if np.any(value != self._zones):
            self._zones = value
            self.create_carray(self._LOOKUP_NODE, self._KEYS["zones"], obj=self._zones)

    @property
    def matrix_levels(self) -> list[str]:
        """Names of all the matrix levels in the OMX file."""
        return [n.name for n in self.list_nodes(self._DATA_NODE, "Array")]

    def get_matrix_array(self, name: str) -> np.ndarray:
        """Return a single matrix level as an array.

        Parameters
        ----------
        name : str
            Name of the matrix level to return.

        Returns
        -------
        np.ndarray
            2D square matrix for a single level.

        See Also
        --------
        :func:`OMXFile.get_matrix_level`
            to get a matrix level as a pandas DataFrame.
        """
        return self.get_node(self._DATA_NODE, name).read()

    def set_matrix_level(self, name: str, matrix: np.ndarray | pd.DataFrame) -> None:
        """Set matrix level in OMX file to given array.

        Parameters
        ----------
        name : str
            Name of matrix level to set.
        matrix : np.ndarray
            Square array of matrix values.

        Raises
        ------
        ValueError
            If `matrix.shape` isn't equal to `self.shape`.
        """
        self._can_write("matrix level")
        if matrix.shape != self.shape:
            raise ValueError(f"matrix shape should be {self.shape} no {matrix.shape}")
        if not isinstance(matrix, np.ndarray):
            if not matrix.index.equals(matrix.columns):
                raise ValueError("matrix columns and index aren't equal")
            if not np.array_equal(
                np.sort(matrix.index.to_numpy()), np.sort(self.zones)
            ):
                raise ValueError("matrix index doesn't equal OMX zones")
            matrix = matrix.reindex(index=self.zones, columns=self.zones).to_numpy()

        with warnings.catch_warnings(
            action="ignore", category=tables.NaturalNameWarning
        ):
            self.create_carray(self._DATA_NODE, str(name), obj=matrix)

    def get_matrix_level(self, name: str) -> pd.DataFrame:
        """Return a single matrix level as an DataFrame.

        Parameters
        ----------
        name : str
            Name of the matrix level to return.

        Returns
        -------
        pd.DataFrame
            2D square matrix for a single level, with
            zones used for column names and indices.
        """
        data = self.get_matrix_array(name)
        return pd.DataFrame(data, index=self.zones, columns=self.zones)

    def get_all(self) -> Generator[tuple[str, pd.DataFrame], None, None]:
        """Iterate through matrix levels as DataFrames.

        Can be used to output matrix levels in a dictionary with
        ``dict(data.get_all())``.

        Yields
        ------
        str
            Name of matrix level.
        pd.DataFrame
            2D square matrix for a single level, with
            zones used for column names and indices.

        See Also
        --------
        :func:`OMXFile.get_matrix_level`
            to read a single matrix level into a DataFrame.
        """
        for name in self.matrix_levels:
            yield name, self.get_matrix_level(name)

    def to_csvs(
        self,
        path: pathlib.Path,
        format_: Literal["square", "long"] = "square",
        *,
        overwrite: bool = False,
    ) -> list[pathlib.Path]:
        """Export each matrix level to a CSV.

        Parameters
        ----------
        path
            Base path for outputting CSVs to, actual CSV filenames
            will be `{path.stem}-{matrix level}{path.suffix}`.
        format_
            Format to write all CSVs as, can be "square" or "long".
        overwrite
            If False (default) raises a FileExistsError if any
            of the output CSVs already exists.

        Returns
        -------
        list[pathlib.Path]
            Paths to written CSVs.

        Raises
        ------
        ValueError
            If CSV format is invalid.
        FileExistsError
            If overwrite is False and one (or more) of the output
            CSVs already exists.
        """
        filepaths = [
            (i, path.with_name(path.stem + f"-{i}{path.suffix}"))
            for i in self.matrix_levels
        ]
        LOG.info("Writing %s OMX levels to CSVs", len(filepaths))

        if format_ not in ("square", "long"):
            raise ValueError(f"unknown CSV format: {format_}")

        if not overwrite:
            exists = [i for _, i in filepaths if i.is_file()]
            if len(exists) > 0:
                raise FileExistsError(", ".join(i.name for i in exists))

        for name, out_path in filepaths:
            data: pd.DataFrame | pd.Series = self.get_matrix_level(name)

            if format_ == "long":
                data = data.stack().squeeze()
                data.name = "value"

            data.to_csv(out_path)
            LOG.debug("Written: %s", out_path)

        return [i[1] for i in filepaths]
