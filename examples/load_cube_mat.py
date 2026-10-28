"""
Load CUBE .mat
==============

Showing how to convert CUBE matrix files (.mat) into the Open Matrix (OMX)
format so the data can be loaded in Python.
"""

import pathlib

from caf.mat import cube, omx

# %%
# MAT Conversion
# --------------
#
# Initialise :class:`~caf.mat.cube.CUBEMatConverter` with the path to the
# CUBE Voyager executable file, usually named "voyager.exe".
voyager_path = pathlib.Path(r"path/to/cube/voyager.exe")
converter = cube.CUBEMatConverter(voyager_path)

# %%
# Convert all the '.mat' files found in a folder to OMX files
# (:meth:`~caf.mat.cube.CUBEMatConverter.folder_to_omx`),  new files will be created
# with the same filename but with the '.omx' suffix.
#
# Alternatively the :meth:`~caf.mat.cube.CUBEMatConverter.to_omx` method can
# be used to convert a single CUBE matrix to OMX.

mat_folder = pathlib.Path(r"path/to/folder/containing/matrices")
print(f"Converting MAT to OMX in {mat_folder}")

omx_paths = converter.folder_to_omx(mat_folder)
print(f"Converted {len(omx_paths)} files")

# %%
# Read OMX
# --------
#
# Open the first converted file using :class:`~caf.mat.omx.OMXFile`
# and load the first matrix level using :meth:`~caf.mat.omx.OMXFile.get_matrix_level`,
# which returns the matrix as a :class:`~pandas.DataFrame`.

print(f"Reading {omx_paths[0].name}")
with omx.OMXFile(omx_paths[0]) as omx_file:
    level = omx_file.matrix_levels[0]
    data = omx_file.get_matrix_level(level)
    print(f"{omx_paths[0].name} - {level}", data, sep="\n")

# %%
# Iterate through all matrix levels with :meth:`~caf.mat.omx.OMXFile.get_all`.

with omx.OMXFile(omx_paths[0]) as omx_file:
    for level, data in omx_file.get_all():
        print(f"{level} total: {data.sum().sum():,}")

# %%
# Write OMX
# ---------
#
# Load matrix levels from one OMX file, edit the values and write them
# to a new OMX file.

out_path = omx_paths[0].with_name(omx_paths[0].stem + "-factored.omx")
print(f"Factoring {omx_paths[0].name}")

# Open file to read from and file to write to
with (
    omx.OMXFile(omx_paths[0]) as read,
    omx.OMXFile(
        out_path,
        mode="w",
        omx_version=read.omx_version,
        shape=read.shape,
    ) as write,
):
    write.zones = read.zones

    for level, data in read.get_all():
        write.set_matrix_level(level, data * 2)
        print(f"\tWritten {level} level to {out_path.name}")

print(f"Written: {out_path}")

# %%
# Convert OMX to MAT
# ------------------
#
# Convert the newly created OMX file back to CUBE's matrix format.

print(f"Converting {out_path.name} to CUBE MAT")
mat_path = converter.from_omx(out_path)
print(f"Written: {mat_path}")
