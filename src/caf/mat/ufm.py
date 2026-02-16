# -*- coding: utf-8 -*-
"""Module containing functionality to convert CSV matrices into SATURN's UFM files."""

##### IMPORTS #####

# Built-Ins
import enum
import logging
import os
import pathlib
import subprocess
from typing import Literal

# Third Party
import pandas as pd

##### CONSTANTS #####
LOG = logging.getLogger(__name__)


##### CLASSES #####
class CSVFormat(enum.Enum):
    """Available CSV formats when converting between UFMs and CSV.

    Not all format defined are available for conversions in both
    directions.

    Enumerations
    ------------
    TUBA2
        TUBA 2 format is 3 columns without a header row containing
        the origin zone number, destination zone number and the
        matrix value. This format excludes any cells which are 0.
    TUBA3
        TUBA 3 format is a fixed width format with 4 columns starting at
        the following positions: 0, 9, 17 and 25. The columns include the
        origin zone, destination zone, matrix level and the matrix value.
        This format excludes any cells which are 0.
    SQUARE
        Square matrix with no header row and the first
        column containing zone name.
    """

    TUBA2 = "tuba2"
    TUBA3 = "tuba3"
    SQUARE = "square"


class UFMConverter:
    """Class for converting matrices to and from SATURN's UFM file format.

    Parameters
    ----------
    saturn_folder : pathlib.Path
        Path to the folder containing SATURN's EXES and batch files.
    """

    def __init__(self, saturn_folder: pathlib.Path) -> None:
        self._saturn_folder = pathlib.Path(saturn_folder).resolve()
        if not self._saturn_folder.is_dir():
            raise NotADirectoryError(
                f"saturn_folder isn't an existing folder: {self._saturn_folder}"
            )
        self._environment: dict[str, str] | None = None

    @property
    def environment(self) -> dict[str, str]:
        """Environment variables dictionary with `saturn_folder` added to "PATH"."""
        if self._environment is None:
            self._environment = update_env(self._saturn_folder)
        return self._environment

    def _run_cmd(
        self,
        *args: str | int | pathlib.Path,
        cwd: pathlib.Path | None = None,
    ) -> tuple[str, str]:
        if cwd is None:
            cwd = pathlib.Path()

        arguments = [
            str(i.resolve()) if isinstance(i, pathlib.Path) else str(i) for i in args
        ]

        LOG.debug(
            "Running: %s\nWorking directory: %s",
            " ".join(i for i in arguments),
            cwd.resolve(),
        )
        comp_proc = subprocess.run(
            arguments,
            capture_output=True,
            env=self.environment,
            cwd=cwd,
            check=False,
            shell=True,
        )
        stdout = _cmd_strip(comp_proc.stdout)
        stderr = _cmd_strip(comp_proc.stderr)
        return stdout, stderr

    @staticmethod
    def _write_key(path: pathlib.Path, *data: str | int) -> pathlib.Path:
        """Write `data` to SATURN KEY file."""
        with path.open("wt", encoding="utf-8") as file:
            file.writelines(f"{i}\n" for i in data)
        LOG.debug("Written KEY file: %s", path)
        return path

    def ufm_to_csv(
        self,
        ufm: pathlib.Path,
        csv: pathlib.Path | None = None,
        csv_format: CSVFormat = CSVFormat.TUBA2,
    ) -> pathlib.Path:
        """Convert UFM file to a CSV in specific format.

        Parameters
        ----------
        ufm
            Path to UFM file to convert.
        csv
            Optional path to output CSV file, defaults to `ufm.csv`
            if not given.
        csv_format
            Format that the output CSV should be saved in,
            default CSVFormat.TUBA2.

        Returns
        -------
        Path
            Path to output CSV file.

        Raises
        ------
        FileNotFoundError
            If the output CSV file isn't created.
        NotImplementedError
            If `csv_format` isn't 'TUBA2' or 'TUBA3'.

        See Also
        --------
        :meth:`ufm_to_square_csvs`
            for conversion to square CSV format.
        """
        ufm = ufm.resolve()
        if csv is None:
            csv = ufm.with_suffix(".csv")
        else:
            csv = csv.resolve()

        if csv_format == CSVFormat.TUBA2:
            args = ["UFM2TBA2", str(ufm), str(csv)]
        elif csv_format == CSVFormat.TUBA3:
            args = ["UFM2TBA3", str(ufm), str(csv.with_suffix(""))]
        elif csv_format == CSVFormat.SQUARE:
            raise NotImplementedError(
                f"use `ufm_to_square_csvs` instead of `ufm_to_csv` for {csv_format}"
            )
        else:
            raise NotImplementedError(f"ufm_to_csv not implemented for {csv_format}")

        LOG.debug("Converting UFM to %s CSV: %s", csv_format.value, ufm)
        msg_data = self._run_cmd(*args, cwd=ufm.parent)

        for suff in (csv.suffix, ".CSV", ".TXT"):
            if csv.with_suffix(suff).exists():
                LOG.debug("Created CSV: %s\n%s\n%s", csv.with_suffix(suff), *msg_data)
                break
        else:
            LOG.error("Failed to create CSV: %s\n%s\n%s", csv, *msg_data)
            raise FileNotFoundError(f"error creating {csv}")
        return csv

    def ufm_to_square_csvs(
        self,
        ufm: pathlib.Path,
        csv_name: str,
        decimal_places: int = 5,
        overwrite: bool = True,
    ) -> tuple[pathlib.Path, list[pathlib.Path]]:
        """Convert UFM to separate square CSVs for each level.

        Dumps UFM to stacked square CSV and then splits that
        into separate CSVs for each level.

        Parameters
        ----------
        ufm
            Path to UFM file.
        csv_name : str
            Base name for CSV files to output, '-level_N.csv'
            will be appended to each.
        decimal_places
            Number of decimal places for SATURN to export to CSV,
            more decimal places can sometimes break SATURN's CSV
            export functionality. Defaults to 5.
        overwrite
            Default True, if the CSV exists already it will be deleted before running.
            If False an error will be raised.

        Returns
        -------
        pathlib.Path
            Path to CSV containing full stacked matrices.
        list[pathlib.Path]
            Paths to the CSVs which contain each level.

        Raises
        ------
        FileNotFoundError
            If the UFM file doesn't exist or the CSV file isn't
            exported from SATURN.
        FileExistsError
            If the CSV file already exists and overwrite is False.

        See Also
        --------
        :meth:`ufm_to_csv`
            for conversion to other CSV formats.
        """
        ufm = ufm.resolve()
        if not ufm.is_file():
            raise FileNotFoundError(f"UFM file doesn't exist: {ufm}")

        csv_path = ufm.parent / csv_name
        csv_path = csv_path.with_suffix(".csv")

        # Remove output if it already exists
        if csv_path.is_file():
            if overwrite:
                LOG.debug("Removed existing CSV file: %s", csv_path)
                csv_path.unlink()
            else:
                raise FileExistsError(
                    f"CSV file already exists and overwrite is False: {csv_path}"
                )

        key_path = self._write_key(
            ufm.with_suffix(".KEY"),
            13,  # Dump to text file
            5,  # CSV format
            csv_path.name,
            1,  # Change decimal places
            int(decimal_places),  # Number of decimal places to output
            # 7,  # Include row numbers
            8,  # Include zone names
            0,  # Do it!
            0,  # Back
            0,  # Exit
            "y",
        )

        LOG.info("Exporting UFM (%s) to CSV", ufm.name)
        msg_data = self._run_cmd(
            "MX",
            str(ufm),
            "KEY",
            str(key_path),
            "VDU",
            str(key_path.with_suffix(".VDU")),
            cwd=key_path.parent,
        )
        if ufm.exists():
            LOG.debug("Created CSV: %s\n%s\n%s", csv_path, *msg_data)
        else:
            LOG.error("Failed creating UFM: %s\n%s\n%s", csv_path, *msg_data)
            raise FileNotFoundError(f"error creating {ufm}")

        # Load CSV and split it into chunks for each level
        level_csvs = self._unstack_csv(csv_path)

        return csv_path, level_csvs

    def _unstack_csv(self, csv_path: pathlib.Path) -> list[pathlib.Path]:
        """Split CSV in stacked square format into separate CSVs for each level."""
        data = pd.read_csv(csv_path, header=None, index_col=0)
        data.insert(0, "level", 0)
        data["level"] = data.groupby(level=0)["level"].transform(
            lambda x: range(1, len(x) + 1)
        )
        data = data.reset_index(names="origin").set_index(["level", "origin"])

        level_csvs = []
        level_names = data.index.get_level_values("level").unique()
        LOG.info(
            "Found %s levels in matrix, writing to separate CSVs", len(level_names)
        )
        for level_name in level_names:
            level: pd.DataFrame = data.loc[level_name, :]

            if len(level) != len(level.columns):
                raise ValueError(f"matrix isn't square for level {level_name}")

            level.columns = level.index
            out_path = csv_path.with_name(csv_path.stem + f"-level_{level_name}.csv")
            level.to_csv(out_path)
            LOG.info("Written: %s", out_path)
            level_csvs.append(out_path)

        return level_csvs

    def _square_csv_to_ufm(
        self,
        csv: pathlib.Path,
        ufm: pathlib.Path | None = None,
        title: str | None = None,
    ) -> pathlib.Path:
        """Convert CSV in square format to UFM file.

        CSV should be a square matrix with no header row and the first
        column containing zone name.

        Parameters
        ----------
        csv
            Path to CSV file.
        ufm
            Optional path to output CSV file, defaults to
            `csv.UFM` if not given.
        title
            Optional title of the matrix (UFM metadata), defaults
            to `csv` name if not given.

        Returns
        -------
        Path
            Path to output UFM.

        Raises
        ------
        FileNotFoundError
            If the output UFM file isn't created.
        """
        # Should be square matrix with zone names in first column and no header rows
        LOG.debug('Converting CSV in square format to UFM: "%s"', csv)
        if ufm is None:
            ufm = csv.with_suffix(".UFM")
        if title is None:
            title = csv.stem

        key_path = self._write_key(
            csv.with_suffix(".KEY"),
            1,  # Select input data
            str(csv.resolve()),
            1,  # Read file with default format (spreadsheet)
            5,  # First column is zone name
            0,  # Read data
            14,  # Dump to UFM file
            1,  # Output as is to UFM
            str(ufm.resolve()),
            title,  # Matrix title
            1,  # Close output file
            0,  # Exit
            "Y",
        )

        msg_data = self._run_cmd(
            "MX",
            "I",
            "KEY",
            str(key_path),
            "VDU",
            str(key_path.with_suffix(".VDU")),
            cwd=key_path.parent,
        )

        if ufm.exists():
            LOG.debug("Created UFM: %s\n%s\n%s", ufm, *msg_data)
        else:
            LOG.error("Failed creating UFM: %s\n%s\n%s", ufm, *msg_data)
            raise FileNotFoundError(f"error creating {ufm}")
        return ufm

    def csv_to_ufm(
        self,
        csv: pathlib.Path,
        format_: CSVFormat = CSVFormat.SQUARE,
        ufm: pathlib.Path | None = None,
        title: str | None = None,
    ) -> pathlib.Path:
        """Conver `csv` file to UFM format.

        Parameters
        ----------
        csv : pathlib.Path
            Path to CSV file to convert
        format_ : CSVFormat
            Format of the CSV file, default `CSVFormat.SQUARE`.
        ufm : pathlib.Path, optional
            Optional path to UFM to create, defaults to `'{csv}.ufm'`.
        title : str, optional
            Optional title of the matrix, used for logging.

        Returns
        -------
        pathlib.Path
            Path to UFM file created.

        Raises
        ------
        NotImplementedError
            Conversion is only implemented for the SQUARE :class:`CSVFormat`,
            any others will currently raise an error.
        """
        csv = csv.resolve()
        if format_ == CSVFormat.SQUARE:
            return self._square_csv_to_ufm(csv, ufm, title)
        raise NotImplementedError(f"csv_to_ufm not yet implemented for {format_}")

    def stack(self, matrices: list[pathlib.Path], ufm: pathlib.Path) -> pathlib.Path:
        """Run SATURN's UFMSTACK to stack `matrices` into a single UFM.

        Parameters
        ----------
        matrices
            Paths to all UFM files to be stacked together.
        ufm
            Path to the output UFM file to create.

        Returns
        -------
        Path
            Path to output UFM created.

        Raises
        ------
        FileNotFoundError
            If any files in `matrices` don't exist, or aren't files.
            If their is an error creating the stacked UFM.
        """
        LOG.debug('Stacking UFMs to "%s"', ufm)
        matrices = [pathlib.Path(m) for m in matrices]
        missing = list(filter(lambda p: not p.is_file(), matrices))
        if missing:
            raise FileNotFoundError(f"cannot find matrices: {missing}")

        # Control file contains path of output then path to all input matrices, without suffix
        control_data = [
            ufm.resolve(),
            *[p.resolve().with_name(p.stem) for p in matrices],
        ]
        control_path = ufm.with_name(ufm.stem + "-STACK.dat")
        with control_path.open("wt") as file:
            file.writelines(f"{i}\n" for i in control_data)
        LOG.debug("Written control file: %s", control_path)

        msg_data = self._run_cmd(
            "UFMSTACK", str(control_path.resolve()), cwd=control_path.parent
        )

        if ufm.exists():
            LOG.debug("Created UFM: %s\n%s\n%s", ufm, *msg_data)
        else:
            LOG.error("Failed creating UFM: %s\n%s\n%s", ufm, *msg_data)
            raise FileNotFoundError(f"error creating {ufm}")
        return ufm

    def _ufm_omx_conversion(
        self,
        path: pathlib.Path,
        from_: Literal["OMX", "UFM"],
        to: Literal["OMX", "UFM"],
        *,
        overwrite: bool = False,
    ) -> pathlib.Path:
        """Perform UFM to OMX, and reverse, conversion."""
        LOG.info('Converting "%s" to %s', path.name, to)

        def check_from_to(value: str) -> Literal["OMX", "UFM"]:
            value = value.upper().strip()
            if value not in ("OMX", "UFM"):
                raise ValueError(f"from / to should be OMX or UFM not '{value}'")
            return value  # type: ignore[return-value]

        from_ = check_from_to(from_)
        to = check_from_to(to)

        path = pathlib.Path(path).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"{from_} doesn't exist: {path}")
        out = path.with_suffix(f".{to}")

        if not overwrite and out.is_file():
            raise FileExistsError(out)

        LOG.debug("Converting %s to %s: %s", from_, to, path)
        msg_data = self._run_cmd(
            f"{from_}2{to}", str(path.with_suffix("")), cwd=path.parent
        )

        if out.exists():
            LOG.debug("Created %s: %s\n%s\n%s", to, out, *msg_data)
        else:
            LOG.error("Failed to create %s: %s\n%s\n%s", to, out, *msg_data)
            raise FileNotFoundError(f"error creating: {out}")
        return out

    def ufm_to_omx(self, ufm: pathlib.Path, *, overwrite: bool = False) -> pathlib.Path:
        """Convert a UFM file to the OMX format.

        Parameters
        ----------
        ufm
            Path to existing UFM file.
        overwrite
            If False (default) will raise error
            if output OMX already exists.

        Returns
        -------
        Path
            Output OMX file, will be '{ufm}.OMX'.

        Raises
        ------
        FileNotFoundError
            If `ufm` doesn't exist or OMX file isn't created.
        FileExistsError
            If `overwrite` is False and output OMX already exists.
        """
        return self._ufm_omx_conversion(ufm, "UFM", "OMX", overwrite=overwrite)

    def omx_to_ufm(self, omx: pathlib.Path, *, overwrite: bool = False) -> pathlib.Path:
        """Convert a OMX file to a UFM file.

        Parameters
        ----------
        omx
            Path to existing UFM file.
        overwrite
            If False (default) will raise error
            if output UFM already exists.

        Returns
        -------
        Path
            Output UFM file, will be '{omx}.UFM'.

        Raises
        ------
        FileNotFoundError
            If `omx` doesn't exist or UFM file isn't created.
        FileExistsError
            If `overwrite` is False and output UFM already exists.
        """
        return self._ufm_omx_conversion(omx, "OMX", "UFM", overwrite=overwrite)


##### FUNCTIONS #####
def update_env(saturn_path: pathlib.Path) -> dict[str, str]:
    """Create a copy of environment variables and adds SATURN path.

    Parameters
    ----------
    saturn_path
        Path to folder containing SATURN batch files.

    Returns
    -------
    dict[str, str]
        A copy of `os.environ` with the `saturn_path` added to the
        "PATH" variable.

    Raises
    ------
    NotADirectoryError
        If `saturn_path` isn't an existing folder.
    """
    if not saturn_path.is_dir():
        raise NotADirectoryError(saturn_path)
    new_env = os.environ.copy()
    sat_paths = rf'"{saturn_path.resolve()}";"{saturn_path.resolve()}\BATS";'
    new_env["PATH"] = sat_paths + new_env["PATH"]
    return new_env


def _cmd_strip(stdout: bytes) -> str:
    """Convert to str and strip newlines from subprocess `stdout` or `stderr`."""
    return stdout.decode().strip().replace("\r\n", "\n")
