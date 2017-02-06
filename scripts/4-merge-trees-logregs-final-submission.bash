#!/bin/bash

pushd `dirname $0` > /dev/null
myDir=`pwd`
popd > /dev/null

echo "
The following command generates the final submission that is based on an average (or stacking with linear regression)
of the predictions computed in steps 2c (submissions based on trees + rdkit features) and 3d
(submissions based on logistic regression + unfolded morgan fingerprints).

This should take just a few minutes... and we are done with the challenge!
"

# L1-regularized models for submission
PYTHONPATH="${myDir}/src:${PYTHONPATH}" python2 -u ${myDir}/src/malaria/results.py final-merged-submissions
