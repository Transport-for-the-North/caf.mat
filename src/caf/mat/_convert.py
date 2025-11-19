"""Convert between various matrix formats.

See Also
--------
:class:`UFMConverter`
    for more SATURN UFM conversions and functionality.
:class:`OMXFile`
    for reading and writing OMX files.
:class:`CUBEMatConverter`
    for more CUBE MAT conversions and functionality.
"""

##### IMPORTS #####

import datetime
import logging
import pathlib
from typing import Literal, Protocol

import pydantic

from caf.mat import _mat, cube, ufm

##### CONSTANTS #####

LOG = logging.getLogger(__name__)


##### CLASSES & FUNCTIONS #####


def convert(
    path: pathlib.Path,
    from_: _mat.MatrixFileFormat,
    to: _mat.MatrixFileFormat,
    *,
    saturn_path: pathlib.Path | None = None,
    voyager_path: pathlib.Path | None = None,
    output_path: pathlib.Path | None = None,
    overwrite: bool = False,
) -> pathlib.Path:
    """Convert matrix file to a different file format.

    Not all combinations of file format conversions are supported yet.

    Parameters
    ----------
    path
        Path to matrix file.
    from_
        Format of matrix file.
    to
        Format to convert matrix file to.
    saturn_path
        Optional path to SATURN exes folder, required
        if converting from / to UFM files.
    voyager_path
        Optional path to CUBE Voyager executable, required
        if converting from / to CUBE MAT files.
    output_path
        Path to save converted matrix to, if not given
        will use `path` with a new suffix.
    overwrite
        If False (default) will raise a FileExistsError
        if the output matrix file already exists.

    Returns
    -------
    pathlib.Path
        Path to the converted matrix file.

    Raises
    ------
    ValueError
        If conversion is invalid.
    FileNotFoundError
        If the matrix file (`path`) doesn't exist.
    NotImplementedError
        If the conversion between formats hasn't been implemented.
    FileExistsError
        If the output file already exists and overwrite is False.
    """
    if from_ == to:
        raise ValueError("conversion from and to formats are the same")
    if not path.is_file():
        raise FileNotFoundError(path)
    _validate_paths(from_, to, saturn_path=saturn_path, voyager_path=voyager_path)

    # Alias to simplify case statements
    ff = _mat.MatrixFileFormat

    match from_, to:
        case (ff.UFM, ff.OMX) | (ff.OMX, ff.UFM):
            return _ufm_omx(
                path,
                from_,  # type: ignore[arg-type]
                to,  # type: ignore[arg-type]
                saturn_path=saturn_path,  # type: ignore[arg-type]
                output_path=output_path,
                overwrite=overwrite,
            )

        case (ff.CUBE, ff.OMX) | (ff.OMX, ff.CUBE):
            return _cube_omx(
                path,
                from_,  # type: ignore[arg-type]
                to,  # type: ignore[arg-type]
                voyager_path=voyager_path,  # type: ignore[arg-type]
                output_path=output_path,
                overwrite=overwrite,
            )

        case (ff.UFM, ff.CUBE) | (ff.CUBE, ff.UFM):
            return _cube_ufm(
                path,
                from_,  # type: ignore[arg-type]
                to,  # type: ignore[arg-type]
                saturn_path=saturn_path,  # type: ignore[arg-type]
                voyager_path=voyager_path,  # type: ignore[arg-type]
                output_path=output_path,
                overwrite=overwrite,
            )

    raise NotImplementedError(f"matrix conversion not implemented from {from_} to {to}")


def _validate_paths(
    *formats: _mat.MatrixFileFormat,
    saturn_path: pathlib.Path | None = None,
    voyager_path: pathlib.Path | None = None,
):
    """Raise error if paths are None when required by `formats`."""
    if _mat.MatrixFileFormat.UFM in formats:
        if saturn_path is None:
            raise ValueError("saturn path required for UFM conversions")
        if not saturn_path.is_dir():
            raise NotADirectoryError(f"SATURN path should point to exes folder: {saturn_path}")
    if _mat.MatrixFileFormat.CUBE in formats:
        if voyager_path is None:
            raise ValueError("voyager path required for CUBE MAT conversions")
        if not voyager_path.is_file():
            raise FileNotFoundError(
                f"voyager path should be an executable file not {voyager_path}"
            )


def _ufm_omx(
    path: pathlib.Path,
    from_: Literal[_mat.MatrixFileFormat.UFM, _mat.MatrixFileFormat.OMX],
    to: Literal[_mat.MatrixFileFormat.UFM, _mat.MatrixFileFormat.OMX],
    *,
    saturn_path: pathlib.Path,
    output_path: pathlib.Path | None = None,
    overwrite: bool = False,
) -> pathlib.Path:
    """Convert between UFM and OMX files."""
    if from_ == to:
        raise ValueError("conversion formats are identical")

    converter = ufm.UFMConverter(saturn_path)

    if from_ == _mat.MatrixFileFormat.UFM:
        out_path = converter.ufm_to_omx(path, overwrite=overwrite)
    else:
        out_path = converter.omx_to_ufm(path, overwrite=overwrite)

    if output_path is not None:
        out_path = out_path.rename(output_path)

    return out_path


def _cube_omx(
    path: pathlib.Path,
    from_: Literal[_mat.MatrixFileFormat.CUBE, _mat.MatrixFileFormat.OMX],
    to: Literal[_mat.MatrixFileFormat.CUBE, _mat.MatrixFileFormat.OMX],
    *,
    voyager_path: pathlib.Path,
    output_path: pathlib.Path | None = None,
    overwrite: bool = False,
) -> pathlib.Path:
    """Convert between CUBE MAT and OMX files."""
    if from_ == to:
        raise ValueError("conversion formats are identical")

    converter = cube.CUBEMatConverter(voyager_path)
    if from_ == _mat.MatrixFileFormat.CUBE:
        output_path = converter.to_omx(path, output_path, overwrite=overwrite)
    else:
        raise NotImplementedError("conversion from OMX to CUBE")

    return output_path


def _cube_ufm(
    path: pathlib.Path,
    from_: Literal[_mat.MatrixFileFormat.CUBE, _mat.MatrixFileFormat.UFM],
    to: Literal[_mat.MatrixFileFormat.CUBE, _mat.MatrixFileFormat.UFM],
    *,
    saturn_path: pathlib.Path,
    voyager_path: pathlib.Path,
    output_path: pathlib.Path | None = None,
    overwrite: bool = False,
) -> pathlib.Path:
    """Convert between CUBE MAT and UFM files, via OMX."""
    if from_ == to:
        raise ValueError("conversion formats are identical")

    cube_converter = cube.CUBEMatConverter(voyager_path)
    ufm_converter = ufm.UFMConverter(saturn_path)
    if from_ == _mat.MatrixFileFormat.CUBE:
        omx_path = cube_converter.to_omx(path, overwrite=overwrite)
        out_path = ufm_converter.omx_to_ufm(omx_path, overwrite=overwrite)

        if output_path is not None:
            out_path = out_path.rename(output_path)

    else:
        # OMX to CUBE not implemented yet, therefore UFM to CUBE isn't either
        raise NotImplementedError("UFM to CUBE conversion")

    return out_path


class ConvertArguments(_mat.ArgumentHandler):
    """Arguments for matrix conversion sub-command."""

    model_config = pydantic.ConfigDict(use_attribute_docstrings=True)

    path: pydantic.FilePath
    """Path to the matrix file for conversion."""
    to: _mat.MatrixFileFormat
    """Format to convert to."""
    from_: _mat.MatrixFileFormat | None = pydantic.Field(None, alias="from")
    """Format to convert from, if not given inferred from file extension."""
    saturn_folder: pydantic.DirectoryPath | None = None
    """Path to folder containing SATURN executables, required when converting to / from UFMs."""
    voyager_path: pydantic.FilePath | None = None
    """Path to CUBE Voyager executable file, required when converting to / from CUBE .MAT files."""
    output: pathlib.Path | None = None
    """Path to save converted matrix to, if not given uses same file name (with new extension)."""
    overwrite: bool = False
    """If given will overwrite existing output files, use with caution."""

    def run(self) -> None:
        """Run matrix conversion."""
        if self.from_ is None:
            from_ = _mat.MatrixFileFormat.infer(self.path)
        else:
            from_ = self.from_

        convert(
            self.path,
            from_,
            self.to,
            saturn_path=self.saturn_folder,
            voyager_path=self.voyager_path,
            output_path=self.output,
            overwrite=self.overwrite,
        )

    @property
    def log_path(self) -> pathlib.Path:
        """Define path to log file for conversion."""
        if self.output is not None:
            folder = self.output.parent
        else:
            folder = self.path.parent
        return folder / f"caf.mat-convert-{datetime.date.today():%Y%m%d}.log"
