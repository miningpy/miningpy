# standard automated tests for mining_utils core functions

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
    result = pd.read_csv(dir_result + 'test_ijk_1.csv')
    data.ijk(
        method='ijk',
        indexing=1,
        xcol='x',
        ycol='y',
        zcol='z',
        xorigin=2.5,
        yorigin=2.5,
        zorigin=2.5,
        xsize=5,
        ysize=5,
        zsize=5,
        x_rotation=0,
        y_rotation=0,
        z_rotation=0,
        icol='i',
        jcol='j',
        kcol='k',
        inplace=True
    )
    data = data.astype(float)  # make sure dtypes same
    result = result.astype(float)  # make sure dtypes same
    assert data.equals(result)


def test_ijk_2():
    # test ijk not inplace
    # all params specified
    data = pd.read_csv(dir_input + 'test_ijk_2.csv')
    result = pd.read_csv(dir_result + 'test_ijk_2.csv')
    data = data.ijk(
            method='ijk',
            indexing=1,
            xcol='x',
            ycol='y',
            zcol='z',
            xorigin=2.5,
            yorigin=2.5,
            zorigin=2.5,
            xsize=5,
            ysize=5,
            zsize=5,
            x_rotation=0,
            y_rotation=0,
            z_rotation=0,
            icol='i',
            jcol='j',
            kcol='k',
            inplace=False
        )
    data = data.astype(float)  # make sure dtypes same
    result = result.astype(float)  # make sure dtypes same
    assert data.equals(result)


def test_ijk_3():
    # test ijk inplace
    # using default params specified
    data = pd.read_csv(dir_input + 'test_ijk_3.csv')
    result = pd.read_csv(dir_result + 'test_ijk_3.csv')
    data.ijk(
        xcol='x',
        ycol='y',
        zcol='z',
        xorigin=2.5,
        yorigin=2.5,
        zorigin=2.5,
        xsize=5,
        ysize=5,
        zsize=5,
        inplace=True
        )
    data = data.astype(float)  # make sure dtypes same
    result = result.astype(float)  # make sure dtypes same
    assert data.equals(result)


def test_ijk_4():
    # test ijk inplace
    # all params specified
    # only calculatig i
    data = pd.read_csv(dir_input + 'test_ijk_4.csv')
    result = pd.read_csv(dir_result + 'test_ijk_4.csv')
    data.ijk(
        method='i',
        indexing=1,
        xcol='x',
        ycol='y',
        zcol='z',
        xorigin=2.5,
        yorigin=2.5,
        zorigin=2.5,
        xsize=5,
        ysize=5,
        zsize=5,
        x_rotation=0,
        y_rotation=0,
        z_rotation=0,
        icol='i',
        jcol='j',
        kcol='k',
        inplace=True
    )
    data = data.astype(float)  # make sure dtypes same
    result = result.astype(float)  # make sure dtypes same
    assert data.equals(result)


def test_ijk_5():
    # test ijk inplace
    # all params specified
    # only calculatig j
    data = pd.read_csv(dir_input + 'test_ijk_5.csv')
    result = pd.read_csv(dir_result + 'test_ijk_5.csv')
    data.ijk(
        method='j',
        indexing=1,
        xcol='x',
        ycol='y',
        zcol='z',
        xorigin=2.5,
        yorigin=2.5,
        zorigin=2.5,
        xsize=5,
        ysize=5,
        zsize=5,
        x_rotation=0,
        y_rotation=0,
        z_rotation=0,
        icol='i',
        jcol='j',
        kcol='k',
        inplace=True
    )
    data = data.astype(float)  # make sure dtypes same
    result = result.astype(float)  # make sure dtypes same
    assert data.equals(result)


def test_ijk_6():
    # test ijk inplace
    # all params specified
    # only calculatig k
    data = pd.read_csv(dir_input + 'test_ijk_6.csv')
    result = pd.read_csv(dir_result + 'test_ijk_6.csv')
    data.ijk(
        method='k',
        indexing=1,
        xcol='x',
        ycol='y',
        zcol='z',
        xorigin=2.5,
        yorigin=2.5,
        zorigin=2.5,
        xsize=5,
        ysize=5,
        zsize=5,
        x_rotation=0,
        y_rotation=0,
        z_rotation=0,
        icol='i',
        jcol='j',
        kcol='k',
        inplace=True
    )
    data = data.astype(float)  # make sure dtypes same
    result = result.astype(float)  # make sure dtypes same
    assert data.equals(result)


def test_ijk_7():
    # test ijk inplace
    # using default params specified
    # zero indexing
    data = pd.read_csv(dir_input + 'test_ijk_7.csv')
    result = pd.read_csv(dir_result + 'test_ijk_7.csv')
    data.ijk(
        xcol='x',
        ycol='y',
        zcol='z',
        indexing=0,
        xorigin=2.5,
        yorigin=2.5,
        zorigin=2.5,
        xsize=5,
        ysize=5,
        zsize=5,
        inplace=True
        )
    data = data.astype(float)  # make sure dtypes same
    result = result.astype(float)  # make sure dtypes same
    assert data.equals(result)


def test_ijk_8():
    # test ijk inplace
    # using default params specified
    # specifying wrong method, to see if test fails
    data = pd.read_csv(dir_input + 'test_ijk_8.csv')
    with pytest.raises(Exception):
        for wrong_method in ['kj', 'l', 'kji', 'ji']:
            data.ijk(
                method = wrong_method,
                xcol='x',
                ycol='y',
                zcol='z',
                xorigin=2.5,
                yorigin=2.5,
                zorigin=2.5,
                xsize=5,
                ysize=5,
                zsize=5,
                inplace=True
                )

def test_ijk_9():
    # test ijk inplace
    # using default params specified
    # specifying indexing other than 1 or 0, to see if fail
    data = pd.read_csv(dir_input + 'test_ijk_9.csv')
    with pytest.raises(Exception):
        data.ijk(
            xcol='x',
            ycol='y',
            zcol='z',
            indexing=5,
            xorigin=2.5,
            yorigin=2.5,
            zorigin=2.5,
            xsize=5,
            ysize=5,
            zsize=5,
            inplace=True
            )
    with pytest.raises(Exception):
        data.ijk(
            xcol='x',
            ycol='y',
            zcol='z',
            indexing=-5,
            xorigin=2.5,
            yorigin=2.5,
            zorigin=2.5,
            xsize=5,
            ysize=5,
            zsize=5,
            inplace=True
            )


def test_ijk_10():
    # test ijk inplace
    # using default params specified
    # specifying rotation out of bounds to see if test fails
    data = pd.read_csv(dir_input + 'test_ijk_10.csv')
    with pytest.raises(Exception):
        data.ijk(
            xcol='x',
            ycol='y',
            zcol='z',
            xorigin=2.5,
            yorigin=2.5,
            zorigin=2.5,
            x_rotation=300,
            xsize=5,
            ysize=5,
            zsize=5,
            inplace=True
            )
    with pytest.raises(Exception):
        data.ijk(
            xcol='x',
            ycol='y',
            zcol='z',
            xorigin=2.5,
            yorigin=2.5,
            zorigin=2.5,
            y_rotation=-300,
            xsize=5,
            ysize=5,
            zsize=5,
            inplace=True
        )
    with pytest.raises(Exception):
        data.ijk(
            xcol='x',
            ycol='y',
            zcol='z',
            xorigin=2.5,
            yorigin=2.5,
            zorigin=2.5,
            z_rotation=200,
            xsize=5,
            ysize=5,
            zsize=5,
            inplace=True
        )


def test_ijk_11():
    # test ijk inplace
    # using default params specified
    # test x rotation
    data = pd.read_csv(dir_input + 'test_ijk_11.csv')
    result = pd.read_csv(dir_result + 'test_ijk_11.csv')
    data.ijk(
        xcol='x',
        ycol='y',
        zcol='z',
        indexing=0,
        xorigin=2.5,
        yorigin=2.5,
        zorigin=2.5,
        x_rotation=45,
        xsize=5,
        ysize=5,
        zsize=5,
        inplace=True
        )
    data = data.astype(float)  # make sure dtypes same
    result = result.astype(float)  # make sure dtypes same
    assert data.equals(result)



