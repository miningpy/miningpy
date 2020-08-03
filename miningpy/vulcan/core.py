"""
This module is used to directly integrate with Maptek Vulcan by using its default executables.
For the functions to run correctly, you must have a valid license installed and the dongle attached.
This code has been tested using Maptek Vulcan 10.1 & Vulcan 8.1.
"""

import sys
import os
import subprocess
import pandas as pd


def vulcan_dir_to_path(parent_path):
    """
    adds Vulcan installation directory to path so that the correct executables can be found.
    """
    full_path = parent_path + '/bin/exe'
    sys.path.append("%r" % full_path)
    os.environ['VULCAN'] = full_path
    return


def bdf_from_bmf(input_bmf_path,
                 output_bdf_path):
    """
    Use this to create a new block model definition file (.bdf) from an existing block model.
    """

    exe = os.environ['VULCAN'] + '/breverse'
    args = [exe, input_bmf_path, output_bdf_path]
    subprocess.call(args, shell=True)
    return


def create_bdf(output_bdf_path):
    """
    Create a basic regular Vulcan bdf that defines a block model framework.
    Used in conjunction with the csv_to_bmf / pandas_to_bmf functions when creating a bmf file.
    """

    exe = os.environ['VULCAN'] + '/bmine'

    return


def bmf_to_csv(input_bmf_path,
               output_csv_path,
               bounding_triangulation=None,
               mask_variable=[None, None],
               test_condition=None):
    """
    Use this to export block values to a nominated CSV file. 
    The resulting CSV file can be edited through software packages that are familiar with the CSV format, such as Microsoft Excel and Microsoft Access. 
    The resulting file will be named after the original block model and stored in the same directory.
    """

    exe = os.environ['VULCAN'] + '/btocsv'

    # check bounding_triangulation, mask_variable, and test_condition
    args = []
    args.append(exe)
    if bounding_triangulation != None:
        args.append(f'-t {bounding_triangulation}')
    if mask_variable != [None, None]:
        args.append(f'-v {mask_variable[0]} {mask_variable[1]}')
    if test_condition != None:
        args.append(f'-C {test_condition}')

    args.append(input_bmf_path)
    args.append(output_csv_path)

    subprocess.call(args, shell=True)
    return


def csv_to_bmf(input_bmf_path):
    """

    """

    exe = os.environ['VULCAN'] + '/bmine'

    return


def pandas_to_bmf(input_bmf_path):
    """

    """

    exe = os.environ['VULCAN'] + '/bmine'

    return


def bmf_to_pandas(input_bmf_path,
                  remove_header_rows=True,
                  bounding_triangulation=None,
                  mask_variable=[None, None],
                  test_condition=None):
    """
    Read a regular block model file directly to a pandas dataframe in a single function
    """
    bmf_to_csv(input_bmf_path, 'temp.csv', bounding_triangulation=bounding_triangulation,
               mask_variable=mask_variable, test_condition=test_condition)

    if remove_header_rows == True:
        df = pd.read_csv('temp.csv', skiprows=[1, 2, 3])
    else:
        df = pd.read_csv('temp.csv')

    os.remove('temp.csv')

    return df


def mine_block_model(input_bmf_path):
    """
    Use this to report on the proportion of a block that falls in a nominated triangulation(s). 
    The 'mined-out' value, which can be reported as a percentage or fraction, will be written to a specified block model variable.
    """

    exe = os.environ['VULCAN'] + '/bmine'

    return


def set_variable(input_bmf_path,
                 variable,
                 value):
    """
    Use this to set a numeric block model variable. i.e. for resetting to 0 before flagging, etc...
    """

    value = float(value)
    variable = str(variable)

    with open("temp.bcf", "w") as file:
        file.write(f"{variable} = {value}")

    exe = os.environ['VULCAN'] + '/bcalc'
    args = [exe, input_bmf_path, "temp.bcf"]
    subprocess.call(args, shell=True)
    os.remove('temp.bcf')

    return


def create_variable(input_bmf_path,
                    variable,
                    value):
    """
    Use this to create a numeric block model variable.
    Data type may be "float" or "int".
    """

    exe = os.environ['VULCAN'] + '/DGD2DWG'

    return


def strings_to_dxf():
    """
    Use this to export the design data contained in a Vulcan design database into a nominated drawing file (.dwg and .dxf).
    """

    exe = os.environ['VULCAN'] + '/DGD2DWG'

    return


def dxf_to_strings():
    """
    Use this to export the design data contained in a Vulcan design database into a nominated drawing file (.dwg and .dxf).
    """

    exe = os.environ['VULCAN'] + '/DWG2DGD'

    return
