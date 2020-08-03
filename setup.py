import setuptools 

with open("README.rst", "r") as fh:
    long_description = fh.read()

install_requires = ['ezdxf>=0.13.1',
                    'numpy',
                    'pandas>=1.1.0',
                    'pyvista',
                    'vtk>=9.0',
                    ]

with open('miningpy/VERSION') as version_file:
    ver = version_file.read().strip()

setuptools.setup(
    name="miningpy",
    version=ver,
    author="Iain Fullelove",
    author_email="fullelove.iain@gmail.com",
    description="set of tools for mining engineering",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="",
    packages=setuptools.find_packages(),
    install_requires=install_requires,
    classifiers=[
        "Programming Language :: Python :: 3",
        "Development Status :: 3 - Alpha",
        "Natural Language :: English",
        "License :: OSI Approved :: MIT License",
        "Operating System :: Microsoft :: Windows",
    ],
)
