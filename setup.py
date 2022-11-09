import sys, os, pathlib

try:
    from skbuild import setup
    from setuptools import find_packages
    import pybind11
except ImportError:
    print("The preferred way to invoke 'setup.py' is via pip, as in 'pip "
          "install .'. If you wish to run the setup script directly, you must "
          "first install the build dependencies listed in pyproject.toml!",
          file=sys.stderr)
    raise

pathlib.Path("./luisa").mkdir(exist_ok=True)
pathlib.Path("./luisa/dylibs").mkdir(exist_ok=True)

setup(
    name="luisa",
    version="0.1.0",  # TBD
    author="Luisa Group",  # TBD
    author_email="zcyjim@outlook.com",  # TBD
    description="A High-Performance Rendering Framework with Layered and Unified Interfaces on Stream Architectures",
    url="https://github.com/LuisaGroup/LuisaCompute",
    license="BSD",
    long_description="A High-Performance Rendering Framework with Layered and Unified Interfaces on Stream Architectures",  # TBD
    long_description_content_type="text/markdown",
    packages=['luisa'],
    cmake_args=[
        '-DLUISA_PROJECT_ROOT=luisa',
        '-DLUISA_PROJECT_DYLIB_PATH=luisa/dylibs',
        '-DCMAKE_INSTALL_LIBDIR=luisa',
        '-DCMAKE_INSTALL_BINDIR=luisa',
        "-DCMAKE_BUILD_TYPE=Release",
        "-DLUISA_COMPUTE_ENABLE_ISPC=ON",
        "-DLUISA_COMPUTE_BUILD_TESTS=OFF"
    ],
    # cmake_install_target="lcapi-packaging",
    python_requires=">=3.9",  # Requires type hint
)
