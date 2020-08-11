# import libraries
import pyvista as pv
import pandas as pd
import numpy as np
from pandas.core.base import PandasObject
from typing import Union, Tuple, List
import miningpy.visualisation.core
import base64
import os
import io
import shutil


def export_html(blockmodel:  pd.DataFrame,
                path:        str = None,
                xyz_cols:    Tuple[str, str, str] = ('x', 'y', 'z'),
                dims:        Tuple[Union[int, float], Union[int, float], Union[int, float]] = None,
                rotation:    Tuple[Union[int, float], Union[int, float], Union[int, float]] = (0, 0, 0),
                cols:        List[str] = None,
                split_by:    str = None) -> bool:
    """
    exports blocks and attributes of block model
    and embeds the data in a paraview glance html app
    to visualise and distribute

    Parameters
    ----------
    blockmodel: pd.DataFrame
        pandas dataframe of block model
    path: str
        filename for vtk file
    xyz_cols: tuple of strings
        names of x,y,z columns in model
    dims: tuple of floats or ints
        x,y,z dimension of regular parent blocks
    rotation: tuple of floats or ints
        rotation of block model grid around x,y,z axis, -180 to 180 degrees
    cols: list of strings
        columns of attributes to visualise using vtk. If None then exports all columns
    split_by: str
        column that is used to split up the block model into components in the Paraview Glance
        app HTML file.

    Returns
    -------
    True if .html file is exported with no errors
    """

    # create temporary directory to work in
    if os.path.exists('__tempMining__'):
        shutil.rmtree("__tempMining__")

    os.mkdir("__tempMining__")

    list_vtu = []
    # create temporary .vtu file(s)
    if split_by is None:
        vtufile = "__tempMining__\\blockModel.vtu"
        list_vtu.append(vtufile)

        blockmodel.blocks2vtk(
            path=vtufile,
            xyz_cols=xyz_cols,
            dims=dims,
            rotation=rotation,
            cols=cols
        )

    else:
        # unique values in split_by column
        uniques = blockmodel[split_by].unique()
        if str(blockmodel[split_by].dtype)[:5] == 'float' or str(blockmodel[split_by].dtype)[:3] == 'int':
            uniques = np.sort(uniques)

        for val in uniques:
            vtufile = f"__tempMining__\\blockModel_{val}.vtu"
            list_vtu.append(vtufile)

            mask = blockmodel[split_by] == val

            blockmodel[mask].blocks2vtk(
                path=vtufile,
                xyz_cols=xyz_cols,
                dims=dims,
                rotation=rotation,
                cols=cols
            )

    # convert .vtu file into polydata
    list_vtp = []
    for file in list_vtu:
        vtp = file[:-4] + '.vtp'  # need to have vtp extension so PyVista doesn't shit the bed.
        list_vtp.append(vtp)
        pv_vtu = pv.read(file)
        wire = pv_vtu.extract_geometry()
        wire.save(vtp)

    # pv glance html template path
    __location__ = os.path.realpath(
        os.path.join(os.getcwd(), os.path.dirname(__file__)))

    template = os.path.join(__location__, r'ParaViewGlance.html')

    # embed vtp into paraview glance html file
    addDataToViewer(list_vtp, template, path)

    # delete temp files
    shutil.rmtree("__tempMining__")

    return True


def addDataToViewer(dataPathList, srcHtmlPath, dstHtmlPath):
    # Extract data as base64
    base64dict = dict()
    for vtp in dataPathList:
        with open(vtp, 'rb') as data:
            dataContent = data.read()
            base64Content = base64.b64encode(dataContent)
            base64Content = base64Content.decode().replace('\n', '')
            base64dict[vtp] = base64Content

    # Create new output file
    with io.open(srcHtmlPath, mode='r', encoding="utf-8") as srcHtml:
        with io.open(dstHtmlPath, mode='w', encoding="utf-8") as dstHtml:
            for line in srcHtml:
                if '</body>' in line:
                    for file, content in base64dict.items():
                        dstHtml.write('<script>\n')
                        dstHtml.write('var contentToLoad = "%s";\n\n' % content);
                        dstHtml.write('Glance.importBase64Dataset("%s" , contentToLoad, glanceInstance.proxyManager);\n' % os.path.basename(file));
                        dstHtml.write('glanceInstance.showApp();\n');
                        dstHtml.write('</script>\n')

                dstHtml.write(line)

    return True


def extend_pandas_html():
    """
    Extends pandas' PandasObject (Series,
    DataFrame) with functions defined in this file.
    """

    PandasObject.export_html = export_html
