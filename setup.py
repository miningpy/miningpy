import setuptools 

with open("README.md", "r") as fh:
    long_description = fh.read()

install_requires = ['ezdxf>=0.13.1',
                    'numpy',
                    'pandas>=1.1.0',
                    'pyvista',
                    'vtk>=9.0',
                    ]

setuptools.setup(
    name="miningpy",
    version="0.1.0",
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
