# -*- coding: utf-8 -*-
"""
Created on: 4/22/2024
Updated on:

Original author: Matteo Gravellu
Last update made by:
Other updates made by:

File purpose:

"""
# Built-Ins
# Third Party
import numpy as np
from openmatrix import open_file
import typing as typ
import subprocess as sp

# Local Imports
# pylint: disable=import-error,wrong-import-position
# Local imports here
# pylint: enable=import-error,wrong-import-position

# # # CONSTANTS # # #

# # # CLASSES # # #

# # # FUNCTIONS # # #
def proc_single(cmd_list: typ.List):
    for ts in cmd_list:
        pr = sp.Popen(ts, creationflags=sp.CREATE_NEW_CONSOLE, shell=True)
        pr.wait()

# CUBE process to convert MAT to OMX format
def cube_mat2omx(exe_cube: str,
                 app_path: str,
                 app_name: str,
                 mat_file: str,
                 del_file=False):

    app_path = app_path.replace('/', '\\').strip()
    mat_file = mat_file.replace('/', '\\').strip()

    to_write = [
        f'convertmat from="{mat_file}" to="{app_path}\\{app_name}.omx" format=omx compression=4']
    print(to_write)
    with open(f'{app_path}\\{app_name}.s', 'w') as script:
        for line in to_write:
            print(line, file=script)
    proc_single([f'"{exe_cube}" "{app_path}\\{app_name}.s" -Pvdmi /Start /Hide /HideScript',
                 f'del "{app_path}\\{app_name}.s"', f'del "{app_path}\\vdmi*.prn"',
                 f'del "{app_path}\\vdmi*.var"', f'del "{app_path}\\TPPL.prj"'])
    proc_single([f'del "{mat_file}"']) if del_file else None


# import omx format to python
def func_omx2ram(omx_file: str,
                 skm_dict: typ.Dict,
                 del_file: bool = False) -> typ.Dict:

    omx_open = open_file(omx_file)
    omx_list = omx_open.list_matrices()
    skm_dict = skm_dict if len(skm_dict) > 0 else {skm: [skm] for skm in omx_list}
    omx_dict = {roh.lower().strip(): 0 for roh in skm_dict}
    for key in omx_list:
        omx_data = np.array(omx_open[key], dtype=float)
        for roh in skm_dict:
            rox = roh.lower().strip()
            for skm in skm_dict[roh]:
                omx_dict[rox] = omx_dict[rox] + (
                    omx_data if skm.lower().strip() == key.lower().strip() else 0)
    omx_open.close()
    proc_single([f'del "{omx_file}"']) if del_file else None
    return omx_dict




