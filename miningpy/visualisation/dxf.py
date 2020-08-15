import pandas as pd
import numpy as np
from pandas.core.base import PandasObject
from typing import Union, List, Tuple
import ezdxf
import pyvista as pv


def plot3D_dxf(path:              str = None,
               colour:            Tuple[float] = (0.666667, 1, 0.498039),
               show_wireframe:    bool = False,
               show_grid:         bool = False,
               cross_section:     bool = False,
               show_plot:         bool = True) -> pv.Plotter:
    """
    create activate 3D vtk plot of dxf that is fully interactive

    Parameters
    ----------
    path: str
        path to dxf file to visualise
    colour: tuple of floats
        default solid colouring of the triangulation
    show_wireframe: bool
        whether to show the edges/lines of a wireframe or not
    show_grid: bool
        add x,y,z grid to see coordinates on plot
    cross_section: bool
        add widget cross-section to plot
    show_plot: bool
        whether to open active window or just return pyvista.Plotter object
        to .show() later

    Returns
    -------
    pyvista.Plotter object & active window of dxf visualisation
    """

    # read in dxf file
    doc = ezdxf.readfile(path)

    # get dxf data
    msp = doc.modelspace()
    faces = msp.query('3DFACE')

    # extract points and faces from dxf object
    num3Dfaces = len(faces)
    pointArray = np.zeros((num3Dfaces * 3, 3))
    cellArray = np.zeros((num3Dfaces, 4), dtype='int')
    cellData = []

    i = 0
    faceCounter = 0
    name = []
    for e in faces:
        pointArray[i, 0] = e.dxf.vtx0[0]
        pointArray[i, 1] = e.dxf.vtx0[1]
        pointArray[i, 2] = e.dxf.vtx0[2]

        pointArray[i + 1, 0] = e.dxf.vtx1[0]
        pointArray[i + 1, 1] = e.dxf.vtx1[1]
        pointArray[i + 1, 2] = e.dxf.vtx1[2]

        pointArray[i + 2, 0] = e.dxf.vtx2[0]
        pointArray[i + 2, 1] = e.dxf.vtx2[1]
        pointArray[i + 2, 2] = e.dxf.vtx2[2]

        cellArray[faceCounter, :] = [3, i, i + 1, i + 2]  # need the 3 at the start to tell pyvista its a triangulation
        cellData.append(e.dxf.layer)
        i = i + 3
        faceCounter = faceCounter + 1

    cellArray = np.hstack(cellArray)
    surf = pv.PolyData(pointArray, cellArray)

    # set theme
    pv.set_plot_theme("ParaView")  # just changes colour scheme

    # create polydata pyvista object
    # plot dxf in pyvista window
    p = pv.Plotter(notebook=False, title="MiningPy DXF Viewer")

    if not cross_section:
        p.add_mesh(mesh=surf,
                   show_edges=show_wireframe,
                   color=colour)

    if cross_section:
        p.add_mesh_clip_plane(mesh=surf,
                              show_edges=show_wireframe,
                              color=colour)

    if show_grid:
        p.show_grid()

    p.show_axes()

    if show_plot:
        p.show(full_screen=True)

    return p


def face_position_dxf():
    return


def extend_pandas_dxf():
    """
    Extends pandas' PandasObject (Series,
    DataFrame) with functions defined in this file.
    """

    PandasObject.face_position_dxf = face_position_dxf
