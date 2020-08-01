import setuptools 
import sys

with open("README.md", "r") as fh:
    long_description = fh.read()

install_requires_win = ['ezdxf>=0.13.1',
                        'numpy',
                        'pandas>=1.1.0',
                        'pyvista',
                        'vtk>=9.0',
                        ]

if sys.platform == 'win32':
    setuptools.setup(
        name="mining_utils",
        version="2020-08-02",
        author="Iain Fullelove",
        author_email="fullelove.iain@gmail.com",
        description="set of tools for mining engineering purposes",
        long_description=long_description,
        long_description_content_type="text/markdown",
        url="",
        packages=setuptools.find_packages(),
        install_requires=install_requires_win,
        classifiers=[
            "Programming Language :: Python :: 3",
            "Development Status :: 3 - Alpha",
            "Natural Language :: English",
            "License :: OSI Approved :: MIT License",
            "Operating System :: Microsoft :: Windows",
        ],
    )

install_requires_lin = ['ezdxf>=0.13.1',
                        'numpy',
                        'pandas>=1.1.0',
                        'pyvista',
                        'vtk @ https://vtk.org/files/release/9.0/vtk-9.0.0-cp38-cp38-linux_x86_64.whl',
                        ]

if sys.platform == 'linux':
    setuptools.setup(
        name="mining_utils",
        version="2020-08-02",
        author="Iain Fullelove",
        author_email="fullelove.iain@gmail.com",
        description="set of tools for mining engineering purposes",
        long_description=long_description,
        long_description_content_type="text/markdown",
        url="",
        packages=setuptools.find_packages(),
        install_requires=install_requires_lin,
        classifiers=[
            "Programming Language :: Python :: 3",
            "Development Status :: 3 - Alpha",
            "Natural Language :: English",
            "License :: OSI Approved :: MIT License",
        ],
    )
