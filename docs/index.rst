.. MiningPy documentation master file, created by
   sphinx-quickstart on Mon Aug  3 11:24:38 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.


.. title:: MiningPy


.. raw:: html

    <div class="banner">
        <img src="_static/miningpy_logo.png" alt="MiningPy">
    </div>


Version: |release|

Documentation: https://miningpy.readthedocs.io/en/latest/

Repository: https://bitbucket.org/incitron/miningpy

Stable Release:

* `Anaconda Cloud Stable <https://anaconda.org/miningpy/miningpy>`_
* `PyPi Stable <https://pypi.org/project/miningpy/>`_

Nightly Release: Version Format: `version.version.version.yyyyMMddHHmm`

* `Anaconda Cloud Nightly <https://anaconda.org/miningpy_nightly/miningpy>`_
* `PyPi Nightly <https://test.pypi.org/project/miningpy/>`_

Testing Pipelines (Azure DevOps): https://dev.azure.com/Iain123/MiningPy

.. toctree::
   :name: Introduction
   :hidden:

   Introduction<self>


About
-----
MiningPy is intended to help mining engineers harness the full power of the Python ecosystem to solve routine mine planning problems.
This package includes tools to help with:

* Block model manipulation:
   * Indexing (ijk)
   * Reblocking (geometric & attribute based)
   * Rotations
   * Calculating the model framework (origin, dimensions, rotation, extents, etc...)
   * Validating the block model (missing internal blocks, checking the model is regular, etc...)
   * Creating bench reserves
   * Aggregatng blocks for scheduling
   * Haulage modelling & encoding to the block model
* Interfacing with commercial mine planning packages, such as:
   * Maptek Vulcan
   * GEOVIA Whittle
   * COMET
   * Minemax Scheduler/Tempo
   * Datamine
* Visualisation:
   * Previewing block models directly in Python for fast reviewing of work
   * Previewing designs (.dxf) directly in Python
   * Exporting block models in `ParaView <https://www.paraview.org/>`_ compatible format

Why MiningPy?
-------------
There are numerous geological packages that have been written in Python, such as `GemPy <https://www.gempy.org/>`_, `PyGSLIB <https://opengeostat.github.io/pygslib/>`_, and `GeostatsPy <https://github.com/GeostatsGuy/GeostatsPy>`_.
However, none of these packages directly provide any tools to handle mining engineering specific problems.
MiningPy aims to provide a simple API to mining engineers that extends existing data science tools like `Pandas <https://pandas.pydata.org/>`_, without having to re-invent the wheel every time they need to interface with commercial mine planning software or manipulate mining data.

Installation
------------

MiningPy is distributed using:

* `conda-forge <https://conda-forge.org/>`_
* `Anaconda Cloud <https://anaconda.org/miningpy/miningpy>`_
* `PyPi <https://pypi.org/project/miningpy/>`_

Conda
^^^^^

MiningPy can be installed using the Conda package manager.
To install using `conda`, you need to add the `conda-forge` channel
so that all dependencies are installed correctly:

.. code-block::

   conda config --add channels conda-forge

To install from `Anaconda Cloud <https://anaconda.org/miningpy/miningpy>`_:

.. code-block::

   conda install -c miningpy miningpy

To install from Conda-Forge:

Pip
^^^

MiningPy can be installed using the Pip package manager:

.. code-block::

   pip install miningpy

Example
-------

.. code-block:: python

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
   )

.. raw:: html

    <div class="banner">
        <img src="_static./plot3D_example.png" alt="plot3D">
    </div>
|

Documentation
-------------

Auto-generated documentation is hosted at
`Read The Docs <https://miningpy.readthedocs.io/en/latest/>`_.

You may also build the documentation yourself:

.. code-block::

   git clone https://bitbucket.org/incitron/miningpy/miningpy.git
   cd miningpy/docs
   make html

The documention can then be found in `miningpy/docs/_build/html/index.html`.

Supported Platforms & Testing
-----------------------------

Platforms
^^^^^^^^^

MiningPy is only tested on Microsoft Windows 10.

Testing
^^^^^^^

The package is built and tested nightly using environments based on `Virtualenv <https://virtualenv.pypa.io/>`_ and `Conda <https://docs.conda.io>`_ (with the current base Anaconda packages).

`Azure DevOps <https://dev.azure.com/Iain123/MiningPy>`_
hosts and runs and the testing pipelines.

MiningPy is tested to be compatible with the following versions of Python:

* Python 3.6
* Python 3.7
* Python 3.8

VTK is a dependency of MiningPy and there are known issues with the current Linux version of VTK published on `PyPi <https://pypi.org/project/vtk/>`_.

The package is also automatically deployed nightly to `TestPyPi <https://test.pypi.org/project/miningpy/>`_, to ensure that official package releases are stable.
The versioning format used on TestPyPi is: `version.version.version.yyyyMMddHHmm`.

API Reference
-------------
The MiningPy API currently includes:

* :ref:`core-api`.
* :ref:`visualisation-api`.
* :ref:`vulcan-api`.

.. toctree::
   :maxdepth: 2
   :caption: API Reference
   :hidden:

   core/core
   visualisation/visualisation
   vulcan/vulcan


Author
------
The creator of MiningPy is a mining engineer that primarly works in long-term strategic mine planning.


License
-------
MiningPy is licensed under the very liberal MIT-License_.

.. _MIT-License: http://opensource.org/licenses/mit-license.php

