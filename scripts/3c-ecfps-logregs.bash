#!/bin/bash

echo "
Once we have computed all the fingerprints and stored them in a suitable,
high performance format, we can quickly evaluate linear models that, hopefully,
exploit sparsity.

Here we will fit logistic regression models using both:
   - scikit learn wrapper over LibLinear (all data need to be in memory)
   - vowpal wabbit (blazingly fast out of core learning)

This is an example on how to perform a cross validation experiment with logistic regression and folding.
Many parameters are available:
  - logistic regression parameters: here we use l1 regularization with C=1,
    and we ask for automatic class weights to tackle imbalance problems
  - evaluation parameters: here we ask to perform 6 times a 10-fold cross validation,
    and we request that the evaluation will stop whenever the roc-auc of a fold do not reach 0.88
  - fingerprint folding: here we request a fold size of 1023 (we use modulus folding so better
    do not use powers of 2 for the fold sizes), with a random seed of 0. No folding is requested
    using '--fingerprint-fold-size 0'.

For each fold, this program will store the test scores and fitted parameters, together with all the provenance
information needed to use these for feature selection and model deployment purposes.
"

# L1-regularized models for submission
PYTHONUNBUFFERED=1 ccl-malaria logregs fit \
--penalty l1 --C 1 --tol 1E-4 --class-weight-auto \
--num-cv-folds 10 --cv-seeds 0 1 2 3 4 5 --min-fold-auc 0.88 \
--fingerprint-folder-seed 0 --fingerprint-fold-size 1023

# L1-regularized models for submission over non-folded fingerprints
PYTHONUNBUFFERED=1 ccl-malaria logregs fit \
--penalty l1 --C 1 --tol 1E-4 --class-weight-auto \
--num-cv-folds 10 --cv-seeds 0 1 2 3 4 5 --min-fold-auc 0.88 \
--fingerprint-folder-seed 0 --fingerprint-fold-size 0

# L2-regularized models for submission over non-folded fingerprints
PYTHONUNBUFFERED=1 ccl-malaria logregs fit \
--penalty l2 --C 5 --tol 1E-4 --class-weight-auto \
--num-cv-folds 10 --cv-seeds 0 1 2 3 4 5 --min-fold-auc 0.88 \
--fingerprint-folder-seed 0 --fingerprint-fold-size 0
