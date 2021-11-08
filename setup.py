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

ext_modules = [Extension('NN', ['NN.pyx'],
                        include_dirs=[numpy.get_include(), '.'],
                        compiler_directives={'language_level' : "3"})]

ext_modules += [Extension('topology', ['topology.pyx'],
                         include_dirs=[numpy.get_include(), '.'],
                         compiler_directives={'language_level' : "3"})]
print(file_names)
setup(
    name = 'NN',
    cmdclass = {'build_ext': build_ext},
    ext_modules=ext_modules
    )
