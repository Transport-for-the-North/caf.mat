"""
Interacting with OMX Files
==========================

Example showing how to convert a UFM to and OMX file and
access the data within Python.
"""

# %%
# Import required modules from Python's standard library

import os
import pathlib
import shutil
from typing import Any

# %%
# Import require third-party packages
import numpy as np

# %%
# Import required modules from caf.mat
from caf.mat import omx, ufm

# %%
# Get SATURN executable path from an environment variable
saturn_path = pathlib.Path(os.getenv("SATURN_EXES_PATH"))

if not saturn_path.is_dir():
    raise NotADirectoryError(saturn_path)
print(f'SATURN exes folder: "{saturn_path}"')


# %%
# Convert a UFM file to OMX by first initialising the :class:`UFMConverter` with
# a path to the SATURN executables folder and then running :meth:`UFMConverter.ufm_to_omx`.

ufm_path = pathlib.Path("example.ufm")
converter = ufm.UFMConverter(saturn_path)
omx_path = converter.ufm_to_omx(ufm_path)
print(f"Created OMX file: {omx_path}")

# %%
# Define a function which formats the outputs for the matrix level statistics.


def format_numeric(value: Any) -> str:  # noqa: ANN401
    """Format numeric `value`, non-numeric return as `str(value)`."""
    if not isinstance(value, (int, float)):
        return str(value)
    integer_cutoff = 10
    if value > integer_cutoff:
        return f"{value:,.0f}"
    if value > 1:
        return f"{value:,.2f}"
    return f"{value:.1e}"


# %%
# Read the OMX file using :class:`OMXFile`, iterate through the matrix levels and print
# out some summary statistics for each one. Using Python's `with` statement ensures the file
# is closed automatically at the end.
#
# The :meth:`OMXFile.get_all` method iterates through all matrix levels in the file and
# provides the level name and the data (as a :class:`pd.DataFrame`).

with omx.OMXFile(omx_path) as omx_file:
    print(
        f"OMX file is v{omx_file.omx_version} with shape "
        f"{omx_file.shape} and {len(omx_file.matrix_levels)} levels"
        f"\nMatrices contain {len(omx_file.zones):,} zones,"
        f" first 5 are: {omx_file.zones[:5]}"
    )

    for name, data in omx_file.get_all():
        array = data.to_numpy()
        stats = {
            "Total": np.sum(array),
            "Mean": np.mean(array),
            "Min": np.min(array),
            "Median": np.median(array),
            "Max": np.max(array),
            "Shape": data.shape,
        }

        print(
            name,
            "-" * len(name),
            "\n".join(f"{i:<10.10}: {format_numeric(j)}" for i, j in stats.items()),
            "",
            sep="\n",
        )

# %%
# Read the same OMX file as above, factor the data and output to a new OMX file.

omx_out = omx_path.with_name(omx_path.stem + "-factored.omx")
factor = 2.5

with (
    omx.OMXFile(omx_path) as read,
    omx.OMXFile(omx_out, mode="w", omx_version=read.omx_version, shape=read.shape) as write,
):
    write.zones = read.zones
    for name, data in read.get_all():
        print(f"Multiplying {name} by {factor}")
        factored = data * factor
        write.set_matrix_level(name, factored)

print(f"Written: {omx_out}")

# %%
# Convert the both the original and factored OMX files back to UFMs using :class:`UFMFile`.
# The OMX file is copied to a new folder before conversion to avoid overwriting the
# original UFM.

folder = omx_path.parent / "To UFM"
folder.mkdir(exist_ok=True)

for path in (omx_path, omx_out):
    out_path = folder / path.name
    print(f"Copying {path.name} to {folder}")
    shutil.copy(path, out_path)

    print(f"Converting {out_path.name} to UFM")
    out_path = converter.omx_to_ufm(out_path)
    print(f"Written: {out_path}")
