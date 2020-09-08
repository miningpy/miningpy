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

mask = blockModel['value'] > 300_000
blockModel.loc[mask, 'test'] = 'ore_highgrade'

blockModel['test'] = blockModel['test'].astype('string')

blockModel['bool'] = False
mask = blockModel['value'] > 0
blockModel.loc[mask, 'bool'] = True

test = blockModel['test'].values
test2 = blockModel['bool'].values

blockModel['test_int'] = 0
mask = blockModel['test'] == 'waste'
blockModel.loc[mask, 'test_int'] = 1

plot = blockModel.plot3D(col='value',
                         dims=(1, 1, 1),
                         show_plot=True,
                         widget='section')
