#!/bin/bash

pushd `dirname $0` > /dev/null
myDir=`pwd`
popd > /dev/null

echo "We will compute tree ensembles using the rdkit descriptors."
echo "This will create files blah..."
echo "Here we illustrate:"
echo "  - feature selection via ensembles of trees"
echo "  - the importance of model parameter selection"
echo "  - the goodness of OOB-based performance estimation; the miseries of SAR bias"
echo "  - the need for further postprocessing"
echo "  - ..."
echo "  - and something about chemistry!"
echo "This will require quite a bit of memory and time..."

PYTHONPATH="${myDir}/src:${PYTHONPATH}" /usr/bin/time -v python2 -u ${myDir}/src/malaria/trees.py fit-trees
