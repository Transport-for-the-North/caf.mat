"""Module for converting matrices to/from CUBE's .mat format."""

##### IMPORTS #####

# Built-Ins
import logging
import re
import subprocess
import warnings
from pathlib import Path

##### CONSTANTS #####

LOG = logging.getLogger(__name__)
_SCRIPT_ENCODING = "utf-8"

##### CLASSES & FUNCTIONS #####


class CUBEMatConverterError(Exception):
    """Errors when converting to/from CUBE's .mat format."""


class CUBEMatConverter:
    """Class for converting to/from CUBE's .mat format.

    Parameters
    ----------
    voyager_path : Path
        Path to the CUBE Voyager executable file.

    Raises
    ------
    FileNotFoundError
        If `voyager_path` doesn't exist, or isn't a file.
    """

    def __init__(self, voyager_path: Path) -> None:
        self.voyager_path = voyager_path
        if not self.voyager_path.is_file():
            raise FileNotFoundError(f"cannot find CUBE Voyager: {self.voyager_path}")

        if self.voyager_path.name.lower().strip() != "voyager.exe":
            warnings.warn(
                "CUBE Voyager executable is usually named "
                f"'VOYAGER.exe', is '{self.voyager_path.name}' correct?"
            )

    def from_csv(
        self,
        num_zones: int,
        csv_paths: dict[str, Path],
        mat_path: Path,
        mat_factor: float = 1,
    ) -> Path:
        """Convert CSVs to CUBE's .mat format.

        CSVs should have 3 columns: origin, destination and trips,
        with no header row.

        Parameters
        ----------
        num_zones : int
            Number of zones in the matrix.
        csv_paths : dict[str, Path]
            Paths to the CSVs to be used for creating the matrix,
            the CSVs should have 3 columns: origin, destination and trips
            with no header row. The dictionary keys provide the name for
            the matrix level in the output file.
        mat_path : Path
            Output CUBE .mat file to create.
        mat_factor : float, default
            Factor to divide matrix by upon creation.

        Returns
        -------
        Path
            CUBE matrix created.

        Raises
        ------
        FileNotFoundError
            If any of the input CSVs don't exist.
        CUBEMatConverterError
            If the process fails creating the CUBE matrix.
        """
        LOG.info("Converting CSVs to CUBE .mat format")
        for path in csv_paths.values():
            if not path.is_file():
                raise FileNotFoundError(f"cannot find CSV: {path}")

        if mat_path.suffix != ".mat":
            mat_path.with_suffix(".mat")

        # Create CUBE Voyager script
        script_text = [
            "RUN PGM=MATRIX",
            f'FILEO MATO[1]="{mat_path.resolve()}",',
            f"      mo=1-{len(csv_paths)},dec={len(csv_paths)}*d,name="
            + ",".join(str(nm) for nm in csv_paths),
        ]
        for n, path in enumerate(csv_paths.values(), 1):
            script_text.append(f'FILEI MATI[{n}]="{path.resolve()}",')
            script_text.append("      fields=#1,2,3, pattern=ij:v")
        script_text += [
            "",
            f"zones={num_zones}",
            "fillmw",
        ]
        script_text += [f"mw[{n}]=mi.{n}.1/{mat_factor}" for n in range(1, len(csv_paths) + 1)]
        script_text += ["", "ENDRUN"]

        script_path = mat_path.with_name(mat_path.stem + "-CONVERSION.s")
        script_path.write_text("\n".join(script_text), encoding=_SCRIPT_ENCODING)
        LOG.debug("Written: %s", script_path)

        self._run_script(script_path)

        if not mat_path.is_file():
            raise CUBEMatConverterError("error converting CSV to CUBE .mat")

        self._cleanup_script(script_path)
        return mat_path

    def to_omx(
        self, mat_file: Path, out_path: Path | None = None, *, overwrite: bool = False
    ) -> Path:
        """Convert Cube .MAT to .OMX.

        Parameters
        ----------
        mat_file : Path
            Full path to the .mat file.
        out_path : Path, optional
            Optional path to save output OMX to, if None
            uses `mat_file` with ".omx" extension.
        overwrite : bool, default False
            If False and output OMX already exists will
            raise FileExistsError.

        Returns
        -------
        Path
            Path to created OMX file.
        """
        out_path = self._validate_io_paths(mat_file, out_path, ".omx", overwrite=overwrite)

        mat_file = mat_file.resolve()
        out_path = out_path.resolve()

        LOG.info('Converting "%s" to OMX file', mat_file.name)

        script_path = out_path.parent / "Mat2OMX.s"
        script_path.write_text(
            f'convertmat from="{mat_file}" to="{out_path}" format=omx compression=4',
            encoding=_SCRIPT_ENCODING,
        )
        LOG.debug("Written mat2omx CUBE script: %s", script_path)

        self._run_script(script_path)

        if not out_path.is_file():
            raise CUBEMatConverterError(f"failed creating {out_path.name}")

        self._cleanup_script(script_path)

        return out_path

    def _run_script(self, path: Path) -> None:
        """Run script with CUBE Voyager."""
        args = [
            str(self.voyager_path.resolve()),
            str(path.resolve()),
            "-Pcmat",
            "/Start",
            "/Hide",
            "/HideScript",
        ]

        LOG.debug("Running CUBE Voyager command: %s", " ".join(args))
        comp_proc = subprocess.run(args, capture_output=True, check=False)
        LOG.debug(
            "CUBE output:%s%s",
            _stdout_decode(comp_proc.stdout),
            _stdout_decode(comp_proc.stderr),
        )

    def _validate_io_paths(
        self, path: Path, out_path: Path | None, suffix: str, *, overwrite: bool = False
    ) -> Path:
        """Check `path` exists and validate `out_path`.

        Will create `out_path` based on `path` if not given.
        """
        if not suffix.startswith("."):
            suffix = "." + suffix

        if not path.is_file():
            raise FileNotFoundError(path.resolve())

        if out_path is None:
            out_path = path.with_suffix(suffix)
        if out_path.is_dir():
            out_path = out_path / f"{path.stem}{suffix}"
        if out_path.suffix != suffix:
            out_path = out_path.with_suffix(suffix)
        if not overwrite and out_path.is_file():
            raise FileExistsError(out_path)

        return out_path

    def _cleanup_script(self, script_path: Path) -> None:
        """Cleanup script file and logs."""
        script_path.unlink()
        script_path.with_name("TPPL.PRJ").unlink()
        del_pat = re.compile(r"(cmat.*)\.(prn|var)", re.IGNORECASE)
        for path in script_path.parent.iterdir():
            match = del_pat.match(path.name)
            if match:
                path.unlink()

    def folder_to_omx(self, folder: Path, glob: str = "*.mat") -> list[Path]:
        """Conver all ".mat" files in `folder` to OMX."""
        # TODO(MB): This could be made more efficient by writing a single script
        # to convert all files instead of iteratively calling to_omx
        omx_paths = []
        for path in folder.glob(glob):
            out_path = self.to_omx(path)
            omx_paths.append(out_path)

        return omx_paths

    def from_omx(
        self, omx_file: Path, out_path: Path | None = None, *, overwrite: bool = False
    ) -> Path:
        """Convert OMX to Cube .MAT.

        Parameters
        ----------
        omx_file : Path
            Full path to the OMX file.
        out_path : Path, optional
            Optional path to save output Cube .MAT to, if
            None uses `omx_file` with ".mat" extension.
        overwrite : bool, default False
            If False and output MAT already exists will
            raise FileExistsError.

        Returns
        -------
        Path
            Path to created Cube MAT file.
        """
        out_path = self._validate_io_paths(omx_file, out_path, ".mat", overwrite=overwrite)

        omx_file = omx_file.resolve()
        out_path = out_path.resolve()

        LOG.info(
            'Converting "%s" to MAT file with Cube Voyager, this might take several minutes',
            omx_file.name,
        )

        script_path = out_path.parent / "OMX2Mat.s"
        script_path.write_text(
            f'convertmat from="{omx_file}" to="{out_path}" format=TPP',
            encoding=_SCRIPT_ENCODING,
        )
        LOG.debug("Written OMX2Mat CUBE script: %s", script_path)

        self._run_script(script_path)

        if not out_path.is_file():
            raise CUBEMatConverterError(f"failed creating {out_path.name}")

        self._cleanup_script(script_path)

        return out_path


def _stdout_decode(stdout: bytes) -> str:
    """Convert bytes to string starting with a newline, or return empty string."""
    stdout = stdout.decode().strip()
    if stdout != "":
        stdout = "\n" + stdout
    return stdout
