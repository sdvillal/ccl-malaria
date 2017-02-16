#!/bin/bash

pushd `dirname $0` > /dev/null
myDir=`pwd`
popd > /dev/null

echo "We will compute a few descriptors using rdkit."
echo "Many (but not all) of these are referenced here:"
echo "    http://www.rdkit.org/docs/GettingStartedInPython.html#list-of-available-descriptors"
echo "Go to sleep, (this took around 32 hours in our machine)."

for molsSet in unl lab scr; do
  outFile="${myDir}/../data/rdkit/rdkfs/${molsSet}rdkf.h5"
  params="--start 0 --step 1 --mols ${molsSet} --output-file \"$outFile\""
  PYTHONUNBUFFERED=1 ccl-malaria features rdkfs ${params}
done