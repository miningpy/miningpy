# standard automated tests for mining_utils->core->ijk function

import pandas as pd
import pytest
import mining_utils
import os

# test data
testdata = {
    'x': [5, 5, 15],
    'y': [5, 15, 25],
    'z': [5, 5, 5],
    'xdim': [5, 5, 5],
    'ydim': [5, 5, 5],
    'zdim': [5, 5, 5],
}


def test_ijk_1():
    # test ijk inplace
    # all params specified
    data = pd.DataFrame(testdata)
    result = {
        'i': [1, 1, 3],
        'j': [1, 3, 5],
        'k': [1, 1, 1],
    }
    result = pd.DataFrame(result)
    data.ijk(
        method='ijk',
        indexing=1,
        xyz_cols=('x', 'y', 'z'),
        origin=(2.5, 2.5, 2.5),
        dims=(5, 5, 5),
        rotation=(0, 0, 0),
        ijk_cols=('i', 'j', 'k'),
        inplace=True
    )
    data = data.astype(float)  # make sure dtypes same
    result = result.astype(float)  # make sure dtypes same
    assert data[['i', 'j', 'k']].equals(result)


def test_ijk_2():
    # test ijk not inplace
    # all params specified
    data = pd.DataFrame(testdata)
    result = {
        'i': [1, 1, 3],
        'j': [1, 3, 5],
        'k': [1, 1, 1],
    }
    result = pd.DataFrame(result)
    data = data.ijk(
            method='ijk',
            indexing=1,
            xyz_cols=('x', 'y', 'z'),
            origin=(2.5, 2.5, 2.5),
            dims=(5, 5, 5),
            rotation=(0, 0, 0),
            ijk_cols=('i', 'j', 'k'),
    )
    data = data.astype(float)  # make sure dtypes same
    result = result.astype(float)  # make sure dtypes same
    assert data[['i', 'j', 'k']].equals(result)


def test_ijk_3():
    # test ijk inplace
    # using default params specified
    data = pd.DataFrame(testdata)
    result = {
        'i': [0, 0, 2],
        'j': [0, 2, 4],
        'k': [0, 0, 0],
    }
    result = pd.DataFrame(result)
    data.ijk(
        method='ijk',
        xyz_cols=('x', 'y', 'z'),
        origin=(2.5, 2.5, 2.5),
        dims=(5, 5, 5),
        rotation=(0, 0, 0),
        ijk_cols=('i', 'j', 'k'),
        inplace=True
    )
    data = data.astype(float)  # make sure dtypes same
    result = result.astype(float)  # make sure dtypes same
    assert data[['i', 'j', 'k']].equals(result)


def test_ijk_4():
    # test ijk inplace
    # all params specified
    # only calculating i
    data = pd.DataFrame(testdata)
    result = {
        'i': [1, 1, 3],
    }
    result = pd.DataFrame(result)
    data.ijk(
        method='i',
        indexing=1,
        xyz_cols=('x', 'y', 'z'),
        origin=(2.5, 2.5, 2.5),
        dims=(5, 5, 5),
        rotation=(0, 0, 0),
        ijk_cols=('i', 'j', 'k'),
        inplace=True
    )
    data = data.astype(float)  # make sure dtypes same
    result = result.astype(float)  # make sure dtypes same
    assert data[['i']].equals(result)


def test_ijk_5():
    # test ijk inplace
    # all params specified
    # only calculatig j
    data = pd.DataFrame(testdata)
    result = {
        'j': [1, 3, 5],
    }
    result = pd.DataFrame(result)
    data.ijk(
        method='j',
        indexing=1,
        xyz_cols=('x', 'y', 'z'),
        origin=(2.5, 2.5, 2.5),
        dims=(5, 5, 5),
        rotation=(0, 0, 0),
        ijk_cols=('i', 'j', 'k'),
        inplace=True
    )
    data = data.astype(float)  # make sure dtypes same
    result = result.astype(float)  # make sure dtypes same
    assert data[['j']].equals(result)


def test_ijk_6():
    # test ijk inplace
    # all params specified
    # only calculatig k
    data = pd.DataFrame(testdata)
    result = {
        'k': [1, 1, 1],
    }
    result = pd.DataFrame(result)
    data.ijk(
        method='k',
        indexing=1,
        xyz_cols=('x', 'y', 'z'),
        origin=(2.5, 2.5, 2.5),
        dims=(5, 5, 5),
        rotation=(0, 0, 0),
        ijk_cols=('i', 'j', 'k'),
        inplace=True
    )
    data = data.astype(float)  # make sure dtypes same
    result = result.astype(float)  # make sure dtypes same
    assert data[['k']].equals(result)


def test_ijk_7():
    # test ijk inplace
    # using default params specified
    # zero indexing
    data = pd.DataFrame(testdata)
    result = {
        'i': [0, 0, 2],
        'j': [0, 2, 4],
        'k': [0, 0, 0],
    }
    result = pd.DataFrame(result)
    data.ijk(
        method='ijk',
        indexing=0,
        xyz_cols=('x', 'y', 'z'),
        origin=(2.5, 2.5, 2.5),
        dims=(5, 5, 5),
        rotation=(0, 0, 0),
        ijk_cols=('i', 'j', 'k'),
        inplace=True
    )
    data = data.astype(float)  # make sure dtypes same
    result = result.astype(float)  # make sure dtypes same
    assert data[['i', 'j', 'k']].equals(result)


def test_ijk_8():
    # test ijk inplace
    # using default params specified
    # specifying wrong method, to see if test fails
    data = pd.DataFrame(testdata)
    with pytest.raises(Exception):
        for wrong_method in ['kj', 'l', 'kji', 'ji']:
            data.ijk(
                method=wrong_method,
                indexing=0,
                xyz_cols=('x', 'y', 'z'),
                origin=(2.5, 2.5, 2.5),
                dims=(5, 5, 5),
                rotation=(0, 0, 0),
                ijk_cols=('i', 'j', 'k'),
                inplace=True
            )


def test_ijk_9():
    # test ijk inplace
    # using default params specified
    # specifying indexing other than 1 or 0, to see if fail
    data = pd.DataFrame(testdata)
    with pytest.raises(Exception):
        data.ijk(
            method='ijk',
            indexing=5,
            xyz_cols=('x', 'y', 'z'),
            origin=(2.5, 2.5, 2.5),
            dims=(5, 5, 5),
            rotation=(0, 0, 0),
            ijk_cols=('i', 'j', 'k'),
            inplace=True
        )
    with pytest.raises(Exception):
        data.ijk(
            method='ijk',
            indexing=-5,
            xyz_cols=('x', 'y', 'z'),
            origin=(2.5, 2.5, 2.5),
            dims=(5, 5, 5),
            rotation=(0, 0, 0),
            ijk_cols=('i', 'j', 'k'),
            inplace=True
        )


def test_ijk_10():
    # test ijk inplace
    # using default params specified
    # specifying rotation out of bounds to see if test fails
    data = pd.DataFrame(testdata)
    with pytest.raises(Exception):
        data.ijk(
            method='ijk',
            indexing=1,
            xyz_cols=('x', 'y', 'z'),
            origin=(2.5, 2.5, 2.5),
            dims=(5, 5, 5),
            rotation=(300, 0, 0),
            ijk_cols=('i', 'j', 'k'),
            inplace=True
        )
    with pytest.raises(Exception):
        data.ijk(
            method='ijk',
            indexing=1,
            xyz_cols=('x', 'y', 'z'),
            origin=(2.5, 2.5, 2.5),
            dims=(5, 5, 5),
            rotation=(0, -300, 0),
            ijk_cols=('i', 'j', 'k'),
            inplace=True
        )
    with pytest.raises(Exception):
        data.ijk(
            method='ijk',
            indexing=1,
            xyz_cols=('x', 'y', 'z'),
            origin=(2.5, 2.5, 2.5),
            dims=(5, 5, 5),
            rotation=(0, 0, 200),
            ijk_cols=('i', 'j', 'k'),
            inplace=True
        )


# test rotation



