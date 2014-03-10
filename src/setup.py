from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
import numpy as np

#---Cython extensions
common_faster = Extension('oscail.common.faster',
                          ['oscail/common/faster.pyx'],
                          include_dirs=[np.get_include()],
                          extra_compile_args=['-fopenmp'],  # '-O3'
                          extra_link_args=['-fopenmp'])


setup(
    name='ccl_malaria',  # Cricket-Chorus-Learn (malaria)
    version='0.1rc1',
    packages=['malaria',
              'malaria.sandbox',
              'malaria.sandbox.threeD',
              'minioscail',
              'minioscail.common',
              'integration'],
    package_dir={'': 'src'},
    url='',
    license='',
    author='Santi Villalba, Floriane Montanari',
    author_email='sdvillal@gmail.com',
    description='',
    requires=['joblib', 'h5py', 'pandas', 'argh', 'scipy', 'cython'],
    cmdclass={'build_ext': build_ext},
    ext_modules=[common_faster],
    script_args=['build_ext', '--inplace']
)