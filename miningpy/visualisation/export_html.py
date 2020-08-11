# import libraries
import pyvista as pv
import pandas as pd
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
                cols:        List[str] = None) -> bool:
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

    Returns
    -------
    True if .html file is exported with no errors
    """

    # create temporary .vtu file
    vtu = r"__temp__.vtu"
    vtufile = vtu

    blockmodel.blocks2vtk(
        path=vtufile,
        xyz_cols=xyz_cols,
        dims=dims,
        rotation=rotation,
        cols=cols
    )

    # convert .vtu file into polydata
    vtp = r"__temp__.vtp"
    vtpfile = vtp
    pv_vtu = pv.read(vtufile)
    wire = pv_vtu.extract_geometry()
    wire.save(vtpfile)

    # make a copy of the paraview glance html template
    # to where the user wants to save it (i.e. path parameter)

    # template path
    __location__ = os.path.realpath(
        os.path.join(os.getcwd(), os.path.dirname(__file__)))

    template = os.path.join(__location__, r'ParaViewGlance.html')

    # copy pv glance
    shutil.copy(template, path)

    # embed vtp into paraview glance html file
    addDataToViewer(vtpfile, path)

    # delete temp files
    os.remove(vtufile)
    os.remove(vtpfile)

    return True


def addDataToViewer(dataPath, srcHtmlPath):
    if os.path.isfile(dataPath) and os.path.exists(srcHtmlPath):
        dstDir = os.path.dirname(dataPath)
        dstHtmlPath = os.path.join(dstDir, '%s.html' % os.path.basename(dataPath)[:-6])

        # Extract data as base64
        with open(dataPath, 'rb') as data:
            dataContent = data.read()
            base64Content = base64.b64encode(dataContent)
            base64Content = base64Content.decode().replace('\n', '')

        # Create new output file
        with io.open(srcHtmlPath, mode='r', encoding="utf-8") as srcHtml:
            with io.open(dstHtmlPath, mode='w', encoding="utf-8") as dstHtml:
                for line in srcHtml:
                    if '</body>' in line:
                        dstHtml.write('<script>\n')
                        dstHtml.write('var contentToLoad = "%s";\n\n' % base64Content);
                        dstHtml.write('Glance.importBase64Dataset("%s" , contentToLoad, glanceInstance.proxyManager);\n' % os.path.basename(dataPath));
                        dstHtml.write('glanceInstance.showApp();\n');
                        dstHtml.write('</script>\n')

                    dstHtml.write(line)


def extend_pandas_html():
    """
    Extends pandas' PandasObject (Series,
    DataFrame) with functions defined in this file.
    """

    PandasObject.export_html = export_html
