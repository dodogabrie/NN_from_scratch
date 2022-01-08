from distutils.core import setup, Extension
from Cython.Build import build_ext
import numpy

module_name = 'cyNeural'

e1 = Extension('cyNeural', ['NN.pyx'], include_dirs=[numpy.get_include(), '.'],)
e2 = Extension('topology.topology', ['topology/topology.pyx'], include_dirs=[numpy.get_include(), '.'],)
e3 = Extension('training.training', ['training/training.pyx'],include_dirs=[numpy.get_include(), '.'],)

ext_modules = [e1, e2, e3]

for e in ext_modules:
    e.cython_directives = {'language_level': "3"} #all are Python-3

setup(
    name = module_name,
    cmdclass = {'build_ext': build_ext},
    ext_modules=ext_modules
    )
