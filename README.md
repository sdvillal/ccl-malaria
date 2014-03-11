# CCL-Malaria - Modelling the Activity of Molecules against Malaria

[First see why?](http://www.tdtproject.org/challenge-1---malaria-hts.html).

## Installation

First we need to install all the dependencies:

- python (scientific) libraries:
   * numpy >= 1.7
   * scipy >= 0.13
   * cython >= 0.19
   * scikit-learn >= 0.14
   * pandas >= 0.13
   * joblib >= 0.8
   * h5py >= 2.2.1
   * argh >=0.21
- for cheminformatics, [rdkit >= R2013_09](http://www.rdkit.org/docs/Install.html)

The easiest way to get (1) done is to install the free
[anaconda python scientific distribution](https://store.continuum.io/cshop/anaconda/)
and then complete the installation of the missing libraries by running:

```sh
pip install argh joblib
```

Then, we just need to install CCL-Malaria by decompressing a distribution package or
cloning the [latest version from github](https://github.com/sdvillal/ccl-malaria).

## Running

The library provides a lot of entangled cheminformatics and machine learning functionality,
ready to be used in a programmatic way (highly recommended) or using the command line.
The scripts with the exemplified workflow are designed to work on bash shells in linux, but they
should be easy to adapt to other environments. These scripts take care of downloading the data,
so there is nothing else to do other than running them in order.

## Contact

This is work in progress by Santi Villalba and Floriane Montanari. Feel free to use the issue tracker
or github messages to contact us.
