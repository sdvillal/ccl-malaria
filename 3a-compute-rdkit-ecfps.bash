#!/bin/bash

pushd `dirname $0` > /dev/null
myDir=`pwd`
popd > /dev/null

echo "We will compute all Morgan fingerprints (ECFP-like and FCFP-like) in rdkit."
echo "This is a computationally intensive work, so we do it in two steps."
echo "Here we compute them, possibly in parallel, generating several plain-text files."
echo "In the next step (3b-munge-rdkit-ecfps.bash) we merge and make these faster to access from our programs."
echo "Go to sleep, (this took around 9 hours in 44 cores)."

time PYTHONPATH="${myDir}/src:${PYTHONPATH}" python2 -u ${myDir}/src/malaria/features.py ecfps-mp

