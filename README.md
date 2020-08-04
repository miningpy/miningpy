![alt text](https://bitbucket.org/incitron/miningpy/src/master/docs/_static/miningpy_logo.png)

Version:        0.1.0

Documentation:  https://miningpy.readthedocs.io/en/latest/

Repository:     https://bitbucket.org/incitron/miningpy


## About
MiningPy is intended to help mining engineers harness the full power of the Python ecosystem to solve routine mine planning problems.
This package includes tools to help with:

- Block model manipulation:
    - Indexing (ijk)
    - Reblocking (geometric & attribute based)
    - Rotations
    - Calculating the model framework (origin, dimensions, rotation, extents, etc...)
    - Validating the block model (missing internal blocks, checking the model is regular, etc...)
    - Creating bench reserves
    - Aggregatng blocks for scheduling
    - Haulage modelling & encoding to the block model
- Interfacing with commercial mine planning packages, such as:
    - Maptek Vulcan
    - GEOVIA Whittle
    - COMET
    - Minemax Scheduler/Tempo
    - Datamine
- Visualisation:
    - Previewing block models directly in Python for fast reviewing of work
    - Previewing designs (.dxf) directly in Python
    - Exporting block models in [Paraview](https://www.paraview.org/) compatible format


## Why MiningPy?

There are numerous geological packages that have been written in Python, such as [GemPy](https://www.gempy.org), [PyGSLIB](https://opengeostat.github.io/pygslib), and [GeostatsPy](https://github.com/GeostatsGuy/GeostatsPy).
However, none of these packages directly provide any tools to handle mining engineering specific problems.
MiningPy aims to provide a simple API to mining engineers that extends existing data science tools like [Pandas](https://pandas.pydata.org), without having to re-invent the wheel every time they need to interface with commercial mine planning software or manipulate mining data.


## Installation

MiningPy is distributed using [PyPi](https://pypi.org) and can be installed through Pip:

    $ pip install miningpy


## Documentation

Auto-generated documentation is hosted at [Read The Docs](https://miningpy.readthedocs.io/en/latest/)

You may also build the documentation yourself:

```bash
git clone https://bitbucket.org/incitron/miningpy/miningpy.git
cd miningpy/docs
make html
```

The documention can then be found in miningpy/docs/_build/html/index.html


## Supported Platforms & Testing

### Platforms

MiningPy is only tested on Microsoft Windows 10.

### Testing

The package is built and tested nightly using environments based on [Virtualenv](https://virtualenv.pypa.io/) and [Conda](https://docs.conda.io) (with the current base Anaconda packages).

MiningPy is tested to be compatible with the following versions of Python:

- Python 3.6
- Python 3.7
- Python 3.8

VTK is a dependency of MiningPy and there are known issues with the current Linux version of VTK published on [PyPi](https://pypi.org/project/vtk/).

The package is also automatically deployed nightly to [TestPyPi](https://test.pypi.org/project/miningpy/), to ensure that official package releases are stable.
The versioning format used on TestPyPi is: `version.version.version.yyyyMMddHHmm`.

## API Reference


## Author

The creator of MiningPy is a mining engineer consultant that primarly works in long-term strategic mine planning.


## Contributing

Contributions to the MiningPy package are welcome. 
Please refer to the [Contributing Guide](CONTRIBUTING.md) for detailed information.


## License
MiningPy is licensed under the very liberal [MIT License](http://opensource.org/licenses/mit-license.php).
