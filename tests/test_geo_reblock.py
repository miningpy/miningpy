# standard automated tests for miningpy->core->gemometric_reblok=ck function
import pytest
import pandas as pd
import miningpy


# block model example (zuck small - MineLib)
url = "https://drive.google.com/uc?export=download&id=1SOrYhqiu5Tg8Zjb7be4fUWhbFDTU1sEk"

# read in block model from link
data = pd.read_csv(url, compression='zip')


reblock = data.geometric_reblock(
    dims=(1, 1, 1),
    xyz_cols=('x', 'y', 'z'),
    origin=(-0.5, -0.5, -0.5)
)

