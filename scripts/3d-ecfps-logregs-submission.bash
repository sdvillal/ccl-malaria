#!/bin/bash

pushd `dirname $0` > /dev/null
myDir=`pwd`
popd > /dev/null

echo "
After running as many experiments as we want with logistic regression, it is time to deploy
(generate predictions for the unlabelled sets of molecules). Here we use the following approach:
we average (or stack) the (calibrated) predictions on the test sets for different models for the hit-lists,
competition hold-out dataset and emolecules screening dataset.

In particular, with this command we select many l1 and l2 regularized logistic regressions with parameters
that exhibit high performance (of course if we have evaluated them first).

It is important to note that in no case we use predictions on any molecule that was part of a training set.
"

# L1-regularized models for submission
PYTHONPATH="${myDir}/src:${PYTHONPATH}" python2 -u ${myDir}/src/malaria/logregs_analysis.py do-logreg-submissions
