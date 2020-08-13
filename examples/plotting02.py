import pandas as pd
import miningpy

blockModelData = {
    'x': [5, 5, 15],
    'y': [5, 15, 25],
    'z': [5, 5, 5],
    'tonnage': [50, 100, 50],
}

blockModel = pd.DataFrame(blockModelData)
blockModel.plot3D(
    xyz_cols=('x', 'y', 'z'),
    dims=(5, 5, 5),  # block dimensions (5m * 5m * 5m)
    col='tonnage',  # block attribute to colour by
    widget='COG'
)
