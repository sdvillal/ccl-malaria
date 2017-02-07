from setuptools import setup

setup(
    name='ccl_malaria',  # Cricket-Chorus-Learn (malaria)
    version='0.2-dev0',
    license='BSD 3 clause',
    url='https://github.com/sdvillal/ccl-malaria',
    packages=['ccl_malaria',
              'minioscail',
              'minioscail.common',
              'minioscail.common.tests'],
    author='Santi Villalba, Floriane Montanari',
    author_email='sdvillal@gmail.com',
    description='Entry for the TDT Malaria 2014 Challenge',
    entry_points={
        'console_scripts': [
            'ccl-malaria = ccl_malaria.cli:main',
        ]
    },
    install_requires=['future',
                      'numpy',
                      'scipy',
                      'pandas',
                      'joblib',
                      'cython',
                      'scikit-learn',
                      # 'rdkit',
                      'h5py',
                      'ntplib',
                      'argh'],
    tests_require=['pytest', 'pytest-cov', 'pytest-pep8'],
    classifiers=[
        'Intended Audience :: Science/Research',
        'Intended Audience :: Developers',
        'Topic :: Software Development',
        'Topic :: Scientific/Engineering',
        'License :: OSI Approved',
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6'],
)
