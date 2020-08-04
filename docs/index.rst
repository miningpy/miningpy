.. MiningPy documentation master file, created by
   sphinx-quickstart on Mon Aug  3 11:24:38 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

..
   ========================================
   MiningPy documentation
   ========================================

.. image:: ./_static/miningpy_logo.png
    :alt: MiningPy

Version: |release|

Documentation:  https://miningpy.readthedocs.io/en/latest/

Repository:     https://bitbucket.org/incitron/miningpy

.. toctree::
   :name: Introduction
   :hidden:

   self

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
MiningPy is distributed using `PyPi <https://pypi.org>`_ and can be installed through Pip::

    $ pip install miningpy


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

.. image:: ./_static./plot3D_example.png
    :alt: plot3D


Supported Platforms & Testing
-----------------------------

Platforms
^^^^^^^^^

MiningPy is only tested on Microsoft Windows 10.

Testing
^^^^^^^

The package is built and tested nightly using environments based on `Virtualenv <https://virtualenv.pypa.io/>`_ and `Conda <https://docs.conda.io>`_ (with the current base Anaconda packages).

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

