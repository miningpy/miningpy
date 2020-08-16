import pandas as pd
import miningpy
import json

with open("examples/data_links.json", 'r') as file:
    data = json.load(file)

url = data['blockmodel']['mclaughlin']
blockModel = pd.read_csv(url, compression='zip')

blockModel.plot3D(
    xyz_cols=('x', 'y', 'z'),
    dims=(1, 1, 1),  # block dimensions (5m * 5m * 5m)
    col='blockvalue',  # block attribute to colour by
)
