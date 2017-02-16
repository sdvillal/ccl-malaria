#!/bin/bash

echo "
The following command generates the final submission that is based on an average (or stacking with linear regression)
of the predictions computed in steps 2c (submissions based on trees + rdkit features) and 3d
(submissions based on logistic regression + unfolded morgan fingerprints).

This should take just a few minutes... and we are done with the challenge!
"

PYTHONUNBUFFERED=1 ccl-malaria blending merge-submissions
