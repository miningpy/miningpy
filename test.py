import mining_utils
import pandas as pd

block_model = 'letlhakane_kimberlite_flat_mod.csv'

model = pd.read_csv(block_model)

model.blocks2vtk(path='letlhakane',
                 xcol='x',
                 ycol='y',
                 zcol='z',
                 xsize=25,
                 ysize=25,
                 zsize=7)

model.blocks2dxf(path='letlhakane',
                 xcol='x',
                 ycol='y',
                 zcol='z',
                 xsize=25,
                 ysize=25,
                 zsize=7)

'''
model.rotate_grid(xcol='xc',
                         ycol='yc',
                         zcol='zc',
                         xorigin=0,
                         yorigin=0,
                         zorigin=0,
                        z_rotation=45,
                        inplace=True)
'''