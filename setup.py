import setuptools 
import sys

with open("README.md", "r") as fh:
    long_description = fh.read()

install_requires = [
    'ezdxf>=1.0',
    'pandas>=2.1',
    'numpy>=1.26',
    'pyvista>=0.43',
    'pyvistaqt',
    'pyqt5',
    'vtk>=9.2',
    'matplotlib'
]

with open('miningpy/VERSION') as version_file:
    ver = version_file.read().strip()

# for test deployments only
if '--testversion' in sys.argv:
    index = sys.argv.index('--testversion')
    sys.argv.pop(index)  # Removes the '--testversion'
    ver = sys.argv.pop(index)  # Returns the element after the '--testversion'

    # be careful! this is only intended for automated testing on azure pipelines - do not use this in production
    with open('miningpy/VERSION', 'w') as version_file:
        version_file.write(ver)

setuptools.setup(
    name="miningpy",
    version=ver,
    author="Iain Fullelove",
    author_email="fullelove.iain@gmail.com",
    description="python tools for mining engineering",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="",
    packages=setuptools.find_packages(),
    install_requires=install_requires,
    include_package_data=True,
    classifiers=[
        "Programming Language :: Python :: 3",
        "Development Status :: 3 - Alpha",
        "Natural Language :: English",
        "Operating System :: Microsoft :: Windows",
        "License :: OSI Approved :: MIT License"
    ],
)
