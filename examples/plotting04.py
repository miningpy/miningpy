import pandas as pd
import miningpy
import json

with open("examples/data_links.json", 'r') as file:
    data = json.load(file)

url = data['blockmodel']['zuck_small']
blockModel = pd.read_csv(url, compression='zip')

blockModel.export_html(
    path='blockmodel.html',
    xyz_cols=('x', 'y', 'z'),
    dims=(1, 1, 1),  # block dimensions (5m * 5m * 5m)
)
