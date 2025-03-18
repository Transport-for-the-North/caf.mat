import dataclasses
import pathlib
import re

import pandas as pd
import caf.toolkit as ctk

from caf.mat import ufm_converter, omx_file


@dataclasses.dataclass
class Translation:
    path: pathlib.Path
    from_col: str
    to_col: str
    factors_col: str|None = None


class UFMComparison(ctk.BaseConfig):
    matrix_a_path: pathlib.Path
    matrix_b_path: pathlib.Path
    matrix_sector_system_path: Translation
    cost_matrix_path:pathlib.Path
    tld_sector_system_path: Translation|None = None
    out_path: pathlib.Path

    def run(self, saturn_folder:pathlib.Path)->None:
        stacked_matrix_a = read_ufm(self.matrix_a_path, saturn_folder)
        stacked_matrix_b = read_ufm(self.matrix_b_path, saturn_folder)

        if stacked_matrix_a.keys() != stacked_matrix_b.keys():
            raise ValueError("Read in matrices do not contain the same keys")
        
        matrix_sector_system = pd.read_csv(self.matrix_sector_system_path.path)

        tld_sector_system = None
        cost_matrix = None
        
        if self.cost_matrix_path is not None:
            cost_matrix = pd.read_csv(self.cost_matrix_path)

        if self.tld_sector_system_path is not None:
            tld_sector_system = pd.read_csv(self.tld_sector_system_path.path)


def compare_matrix(output_path: pathlib.Path, matrix_a:pd.DataFrame, matrix_b:pd.DataFrame, matrix_sector_system: pd.DataFrame, cost_matrix: pd.DataFrame|None, tld_sector_system: pd.DataFrame|None)->None:
    matrix_report_a = ctk.pandas_utils.MatrixReport(matrix=matrix_a, translation_factors=matrix_sector_system, )

        


        

def read_ufm(matrix_path:pathlib.Path, saturn_folder:pathlib.Path)->dict[int, pd.DataFrame]:
    converter = ufm_converter.UFMConverter(saturn_folder)
    omx_path = converter.ufm_to_omx(matrix_path)

    omx_reader = omx_file.OMXFile(omx_path)
    output = {}
    for level_name in omx_reader.matrix_levels:
        matched = re.match(r"^l(\d{2})", l, flags=re.IGNORECASE)
        if matched is None:
            continue
        level = int(matched.group(1))
        
        output[level] = omx_reader.get_matrix_level_dataframe(level_name)
    
    return output