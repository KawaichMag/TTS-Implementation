from setuptools import setup
from Cython.Build import cythonize
import numpy
from pathlib import Path

# Ensure the output folder exists
Path("monotonic_align").mkdir(exist_ok=True)

setup(
    name="monotonic_align",
    ext_modules=cythonize(
        "core.pyx",  # correct path
        compiler_directives={"boundscheck": False, "wraparound": False},
    ),
    include_dirs=[numpy.get_include()],
)
