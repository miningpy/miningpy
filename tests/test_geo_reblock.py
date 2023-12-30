# standard automated tests for miningpy->core->gemometric_reblock function
import pytest
import pandas as pd
import miningpy

# block model example (zuck small - MineLib)
url = "https://drive.google.com/uc?export=download&id=1SOrYhqiu5Tg8Zjb7be4fUWhbFDTU1sEk"

# read in block model from link
data = pd.read_csv(url, compression='zip')

# testing reblock dimensions
list_of_reblocking_multis = [(2, 2, 5),         # superblock
                             (0.5, 0.5, 1),     # subblock
                             (1, 1, 1),         # the same
                             (1, 2, 1),         # superblock starting with 1
                             (1.0, 0.5, 1),     # subblock starting with 1 as float
]


def test_geo_reblock():
    # listing attributes to carry through (n.b. dropping ID column)
    # keys are what to weight by and values are lists of attributes to be weighted
    varlist_agg = {
        'rock_tonnes': ['cost', 'value'],
        'ore_tonnes': [],
    }

    # take the max or min value of reblock
    min_cols = ['final_pit']
    max_cols = ['period']

    for multi in list_of_reblocking_multis:
        # reblock function
        reblock = data.geometric_reblock(
            dims=(1, 1, 1),  # original dims of model
            xyz_cols=('x', 'y', 'z'),
            origin=(-0.5, -0.5, -0.5),  # bottom left corner
            reblock_multiplier=multi,  # doubling x and y dim and keeping z dim the same
            varlist_agg=varlist_agg,
            min_cols=min_cols,
            max_cols=max_cols,
        )

        # check tonnages sum
        assert (data.rock_tonnes.sum() - reblock.rock_tonnes.sum() < 1) and \
               (data.rock_tonnes.sum() - reblock.rock_tonnes.sum() > -1), 'rock_tonnes lost superblocking'

        # check dims
        new_dims = reblock.block_dims(xyz_cols=('x', 'y', 'z'),
                                      origin=(-0.5, -0.5, -0.5))
        assert new_dims == multi, 'reblocking dims error'

# testing breaking code with broken reblock multiplier
# def test_mytest():
#     with pytest.raises(AssertionError):
#         test_geo_reblock()
