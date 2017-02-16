#!/bin/bash

echo "We will compute tree ensembles using the rdkit descriptors."
echo "Here we illustrate:"
echo "  - feature selection via ensembles of trees"
echo "  - the importance of model parameter selection"
echo "  - the goodness of OOB-based performance estimation; the miseries of SAR bias"
echo "  - the need for further postprocessing"
echo "  - ..."
echo "  - and something about chemistry!"
echo "This will require quite a bit of memory and time..."

PYTHONUNBUFFERED=1 ccl-malaria trees fit
