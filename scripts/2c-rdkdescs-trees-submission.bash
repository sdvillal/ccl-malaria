#!/bin/bash

echo "
After running as many experiments as we want with ensembles of trees, it is time to deploy
(generate predictions for the unlabelled sets of molecules). Here we use the following approach:
we average (or stack) the (calibrated) predictions on the test sets for different models for the hit-lists,
competition hold-out dataset and emolecules screening dataset.

In particular, with this command we select all forests with 6000 trees (of course if we have evaluated them first),
a parameter that exhibits higher performance in our tests.

It is important to note that in no case we use predictions on any molecule that was part of a training set.
"

PYTHONUNBUFFERED=1 ccl-malaria trees submit
