# -*- coding: utf-8 -*-
"""Module for converting matrices to/from CUBE's .mat format."""

##### IMPORTS #####

# Built-Ins
import logging
import re
import subprocess
import warnings
from pathlib import Path
from caf.mat import omx
import pandas as pd

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
    cube_path : Path
        Path to the CUBE Voyager executable file.

    Raises
    ------
    FileNotFoundError
        If `cube_voyager_path` doesn't exist, or isn't a file.
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

        if len(csv_paths) > 20:
            warnings.warn("Mat files cannot be created from more than 20 files. Creating via omx.")
            omx_folder = mat_path.parent
            omx_file_path = omx_folder / 'temp_omx.omx'
            omx_file = omx.OMXFile(omx_file_path,
                                   mode='w',
                                   omx_version=omx.OMXFile._EXPECTED_OMX_VERSION,
                                   shape=[num_zones, num_zones])
            omx_file.zones = pd.read_csv(list(csv_paths.values())[0], index_col=0).index
            omx_file.csv_to_omx(csv_paths)
            self.from_omx(omx_file_path, mat_path)
            omx_file_path.unlink()

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
        script_text += [
            f"mw[{n}]=mi.{n}.1/{mat_factor}" for n in range(1, len(csv_paths) + 1)
        ]
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
        out_path = self._validate_io_paths(
            mat_file, out_path, ".omx", overwrite=overwrite
        )

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
        del_pat = re.compile(r"(cmat.*)\.(prn|var)", re.I)
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
        out_path = self._validate_io_paths(
            omx_file, out_path, ".mat", overwrite=overwrite
        )

        omx_file = omx_file.resolve()
        out_path = out_path.resolve()

        LOG.info(
            'Converting "%s" to MAT file with Cube Voyager,'
            " this might take several minutes",
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

if __name__ == "__main__":

    # import pandas as pd
    # # 24hr_pa
    demand_segments = demand_segments = {
        "HBEBCA_Int": "business",
        "HBEBNCA_Int": "business",
        "NHBEBCA_Int": "business",
        "NHBEBNCA_Int": "business",
        "HBWCA_Int": "commute",
        "HBWNCA_Int": "commute",
        "HBOCA_Int": "leisure",
        "HBONCA_Int": "leisure",
        "NHBOCA_Int": "leisure",
        "NHBONCA_Int": "leisure",
        "EBCA_Ext_FM": "business",
        "EBCA_Ext_TO": "business",
        "EBNCA_Ext": "business",
        "HBWCA_Ext_FM": "commute",
        "HBWCA_Ext_TO": "commute",
        "HBWNCA_Ext": "commute",
        "OCA_Ext_FM": "leisure",
        "OCA_Ext_TO": "leisure",
        "ONCA_Ext": "leisure",
    }
    
    # in_dir = Path(r"D:\NorMITs Demand\ntem_emp_test\canca\PA")
    # csvs = {}
    # for key, path in demand_segments.items():
    #     # df = pd.read_csv(path, index_col=0)
    #     # df.stack().reset_index().to_csv(in_dir / "to_mat" / f"{key}.csv", index=False, header=False)
    #     csvs[key] = in_dir / "to_mat" / f"{key}.csv"
    converter = CUBEMatConverter(Path(r"C:\Program Files\Citilabs\CubeVoyager\VOYAGER.EXE"))
    # converter.from_csv(1325, csvs, in_dir / "24hr_pa.mat")
    # ###
    adj_dir = Path(r"D:\NorMITs Demand\ntem_emp_test\canca\try_2\norms_uc")
    tod_dir = Path(r"D:\NorMITs Demand\ntem_emp_test\canca\TOD")
    full_dir = Path(r"D:\NorMITs Demand\ntem_emp_test\canca\PA\to_mat")
    tps = {1:'AM', 2:'IP', 3:'PM', 4:'OP'}
    full_mats = {}
    for uc in demand_segments.keys():
        if 'ext' in uc.lower():
            df = pd.read_csv(full_dir / f"{uc}.csv", index_col=[0,1], names=['o','d','trips'])
            # df *= 5
            # df.reset_index().to_csv(full_dir / f"{uc}.csv", index=False, header=False)
            full_mats[uc] = df

    ### TOD
    for tp in [1,2,3,4]:
        csv_paths = {}
        for uc_1, uc_2 in {'EB':'EMP', 'HBW':'COM', 'O':"OTH"}.items():
            nca = pd.read_csv(tod_dir / f"{uc_1}NCA_Ext_TS{tp}.csv", index_col=[0,1], names=['o','d','trips'])
            (nca / (5*full_mats[f"{uc_1}NCA_Ext"])).fillna(0).reset_index().to_csv(tod_dir / f"{uc_2}NCA_Ext_TS{tp}_long.csv", index=False, header=False)
            csv_paths[f"{uc_2}_NCA"] = tod_dir / f"{uc_2}NCA_Ext_TS{tp}_long.csv"
            fr = pd.read_csv(tod_dir / f"{uc_1}CA_Ext_FM_TS{tp}.csv", index_col=[0,1], names=['o','d','trips'])
            (fr / full_mats[f"{uc_1}CA_Ext_FM"]).reset_index().to_csv(tod_dir / f"{uc_2}CA_Ext_FM_TS{tp}_long.csv", index=False, header=False)
            csv_paths[f"{uc_2}_FH"] = tod_dir / f"{uc_2}CA_Ext_FM_TS{tp}_long.csv"
            to = pd.read_csv(tod_dir / f"{uc_1}CA_Ext_TO_TS{tp}.csv", index_col=[0,1], names=['o','d','trips'])
            (to / full_mats[f"{uc_1}CA_Ext_TO"]).reset_index().to_csv(tod_dir / f"{uc_2}CA_Ext_TO_TS{tp}_long.csv", index=False, header=False)
            csv_paths[f"{uc_2}_TH"] = tod_dir / f"{uc_2}CA_Ext_TO_TS{tp}_long.csv"
            
        converter.from_csv(1325, csv_paths, tod_dir / f"Time_of_Day_Factors_Zonal_{tps[tp]}.mat")

    ### nhb_od_prop
    od_prop_dir = Path(r"D:\NorMITs Demand\ntem_emp_test\canca\od_return_factors")
    for tp_num, tp_nam in {1:'AM', 2:'IP', 3:'PM', 4:'OP'}.items():
        csvs = {}
        for uc_1, uc_2 in {'business':'EB', 'other':"O"}.items():
            for ca_num, ca_nam in {2:'CA', 1:'NCA'}.items():
                csvs[f"NHB{uc_2}{ca_nam}"] = od_prop_dir / f"OD_nhb_m6_ts{tp_num}_ca{ca_num}_{uc_1}.csv"
        converter.from_csv(1325, csvs, od_prop_dir / f"OD_Prop_{tp_nam}_PT.mat")


