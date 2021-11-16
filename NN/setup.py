from distutils.core import setup, Extension
from Cython.Build import cythonize, build_ext
import numpy
import os

file_names = []
for file in os.listdir("."):
    if file.endswith(".pyx"):
        file_names.append(file)
if len(file_names) == 0:
    raise ValueError('File pyx not found!')

module_name = 'NN'

e1 = Extension('NN', ['NN.pyx'], include_dirs=[numpy.get_include(), '.'],)
e2 = Extension('topology', ['topology.pyx'], include_dirs=[numpy.get_include(), '.'],)
e3 = Extension('training', ['training.pyx'],include_dirs=[numpy.get_include(), '.'],)
#               extra_compile_args = ["-O3", "-ffast-math", "-march=native", "-fopenmp" ],
#               extra_link_args=['-fopenmp'])

ext_modules = [e1, e2, e3]

for e in ext_modules:
    e.cython_directives = {'language_level': "3"} #all are Python-3

print(file_names)
setup(
    name = 'NN',
    cmdclass = {'build_ext': build_ext},
    ext_modules=ext_modules
    )
