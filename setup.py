import sys, os, pathlib

try:
    from skbuild import setup
    from skbuild.constants import CMAKE_INSTALL_DIR
    from setuptools.command import install_lib
except ImportError:
    print("The preferred way to invoke 'setup.py' is via pip, as in 'pip "
          "install .'. If you wish to run the setup script directly, you must "
          "first install the build dependencies listed in pyproject.toml!",
          file=sys.stderr)
    raise


def filter_install_files(cmake_manifest):
    return [f for f in cmake_manifest if
            "include/" not in f and
            "cmake/" not in f and
            "pkgconfig/" not in f]


setup(name="luisa",
      version="0.1.0",  # TBD
      author="Luisa Group",  # TBD
      author_email="zcyjim@outlook.com",  # TBD
      description="A High-Performance Rendering Framework with Layered and Unified Interfaces on Stream Architectures",
      url="https://github.com/LuisaGroup/LuisaCompute",
      license="BSD",
      long_description="A High-Performance Rendering Framework with Layered and Unified Interfaces on Stream Architectures",  # TBD
      long_description_content_type="text/markdown",
      packages=['luisa'],
      package_dir={'luisa': 'src/py/luisa'},
      cmake_args=["-DCMAKE_BUILD_TYPE=Release",
                  "-DLUISA_COMPUTE_ENABLE_ISPC=OFF",
                  "-DLUISA_COMPUTE_ENABLE_LLVM=OFF",
                  "-DLUISA_COMPUTE_BUILD_TESTS=OFF"],
      python_requires=">=3.9",  # Requires type hint
      include_package_data=False,
      cmake_process_manifest_hook=filter_install_files)
