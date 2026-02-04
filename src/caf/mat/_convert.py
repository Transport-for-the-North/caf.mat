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
import os
import pathlib

import pydantic

from caf.mat import _mat, cube, ufm

##### CONSTANTS #####

LOG = logging.getLogger(__name__)


def _get_env_path(name: str) -> pathlib.Path | None:
    value = os.getenv(name, None)
    if value is None:
        return value
    return pathlib.Path(value).resolve()


_VOYAGER_PATH = _get_env_path("CUBE_VOYAGER_PATH")
_SATURN_PATH = _get_env_path("SATURN_EXES_FOLDER")


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

    LOG.info('Converting "%s" from %s to %s', path.name, from_.upper(), to.upper())

    match from_, to:
        case (ff.UFM, ff.OMX) | (ff.OMX, ff.UFM):
            out_path = _ufm_to_omx(
                path,
                reverse=from_ == ff.OMX,
                saturn_path=saturn_path,  # type: ignore[arg-type]
                output_path=output_path,
                overwrite=overwrite,
            )

        case (ff.CUBE, ff.OMX) | (ff.OMX, ff.CUBE):
            out_path = _cube_to_omx(
                path,
                reverse=from_ == ff.OMX,
                voyager_path=voyager_path,  # type: ignore[arg-type]
                output_path=output_path,
                overwrite=overwrite,
            )

        case (ff.UFM, ff.CUBE) | (ff.CUBE, ff.UFM):
            out_path = _cube_to_ufm(
                path,
                reverse=from_ == ff.UFM,
                saturn_path=saturn_path,  # type: ignore[arg-type]
                voyager_path=voyager_path,  # type: ignore[arg-type]
                output_path=output_path,
                overwrite=overwrite,
            )

        case _:
            raise NotImplementedError(
                f"matrix conversion not implemented from {from_} to {to}"
            )

    LOG.info("Written: %s", out_path)
    return out_path


def _validate_paths(
    *formats: _mat.MatrixFileFormat,
    saturn_path: pathlib.Path | None = None,
    voyager_path: pathlib.Path | None = None,
) -> None:
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


def _ufm_to_omx(
    path: pathlib.Path,
    *,
    reverse: bool,
    saturn_path: pathlib.Path,
    output_path: pathlib.Path | None = None,
    overwrite: bool = False,
) -> pathlib.Path:
    """Convert between UFM and OMX files."""
    converter = ufm.UFMConverter(saturn_path)

    if reverse:
        out_path = converter.omx_to_ufm(path, overwrite=overwrite)
    else:
        out_path = converter.ufm_to_omx(path, overwrite=overwrite)

    if output_path is not None:
        out_path = out_path.rename(output_path)

    return out_path


def _cube_to_omx(
    path: pathlib.Path,
    *,
    reverse: bool,
    voyager_path: pathlib.Path,
    output_path: pathlib.Path | None = None,
    overwrite: bool = False,
) -> pathlib.Path:
    """Convert between CUBE MAT and OMX files."""
    converter = cube.CUBEMatConverter(voyager_path)
    if reverse:
        output_path = converter.from_omx(path, output_path, overwrite=overwrite)
    else:
        output_path = converter.to_omx(path, output_path, overwrite=overwrite)

    return output_path


def _cube_to_ufm(
    path: pathlib.Path,
    *,
    reverse: bool,
    saturn_path: pathlib.Path,
    voyager_path: pathlib.Path,
    output_path: pathlib.Path | None = None,
    overwrite: bool = False,
) -> pathlib.Path:
    """Convert between CUBE MAT and UFM files, via OMX."""
    cube_converter = cube.CUBEMatConverter(voyager_path)
    ufm_converter = ufm.UFMConverter(saturn_path)

    if reverse:
        omx_path = ufm_converter.ufm_to_omx(path, overwrite=overwrite)
        out_path = cube_converter.from_omx(omx_path, output_path, overwrite=overwrite)

    else:
        omx_path = cube_converter.to_omx(path, overwrite=overwrite)
        out_path = ufm_converter.omx_to_ufm(omx_path, overwrite=overwrite)

        if output_path is not None:
            out_path = out_path.rename(output_path)

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
    saturn_folder: pydantic.DirectoryPath | None = _SATURN_PATH
    """Path to folder containing SATURN executables, required when converting to / from UFMs."""
    voyager_path: pydantic.FilePath | None = _VOYAGER_PATH
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
