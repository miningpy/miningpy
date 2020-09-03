import pandas as pd
import numpy as np
import miningpy
import pyvista as pv
import matplotlib

# block model example (zuck small - MineLib)
url = "https://drive.google.com/uc?export=download&id=1SOrYhqiu5Tg8Zjb7be4fUWhbFDTU1sEk"

# read in block model from link
blockModel = pd.read_csv(url, compression='zip')

# remove blocks not in schedule
mask = blockModel['period'] >= 0
blockModel = blockModel[mask].copy()

# make period into an int
blockModel['period'] = blockModel['period'].astype(int)

blockModel['test'] = 'waste'

mask = blockModel['value'] > 0
blockModel.loc[mask, 'test'] = 'ore'

blockModel['test'] = blockModel['test'].astype('string')

blockModel['bool'] = False
mask = blockModel['value'] > 0
blockModel.loc[mask, 'bool'] = True

test = blockModel['test'].values
test2 = blockModel['bool'].values

# grid = blockModel.blocks2vtk(
#     dims=(1, 1, 1),
#     output_file=False,
# )
#
# pvgrid = pv.UnstructuredGrid(grid)
#
# # set theme
# pv.set_plot_theme("ParaView")  # just changes colour scheme
#
# p = pv.Plotter(notebook=False, title="Block Model 3D Plot")
#
# # legend settings
# sargs = dict(interactive=True)
#
# p.add_mesh(grid,
#            style='surface',
#            show_edges=True,
#            scalars='test',
#            scalar_bar_args=sargs,
#            cmap='bwr')
#
# p.show_axes()
# p.show(full_screen=True)


def callback(value, widget):
    _rounded_value = int(round(float(value), 0))
    print(_rounded_value)
    widget.GetRepresentation().SetValue(_rounded_value)


plot = blockModel.plot3D(col='period',
                         dims=(1, 1, 1),
                         show_plot=False)

range = (blockModel['period'].min(), blockModel['period'].max())

plot.add_slider_widget(callback=callback,
                       rng=range,
                       title='int-test',
                       value=range[0],
                       event_type='always',
                       pass_widget=True)

plot.show(full_screen=True)
