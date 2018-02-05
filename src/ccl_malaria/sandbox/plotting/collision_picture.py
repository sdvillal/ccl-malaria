# coding=utf-8
"""Generates plots illustrating the randomness of collisions.
Collisions are bad because:
  - They preclude proper interpretation on models that provide feature importance information (like linear models)
  - Generate spurius interactions between unrelated molecules and substructures
On the other hand, collisions are good because:
  - They can act as a form of regulariztion (create simpler models with better generalization capabilities) via the
    hashing trick (in machine learning parlance)
  - And they allow to build models that cannot cope with high dimensional inputs (most of the traditionally used
    statistical models (implementations), from random forests to SVMs and many others).

We believe that the advantages are well worth exploring. Modern modelling techniques
 leveraging the use of big datasets (millions of compounds
and millions of mollecules) and to do so,...
"""
from __future__ import print_function
from functools import partial
from itertools import izip
from rdkit import Chem
from rdkit.Chem import Draw
from ccl_malaria.features import MalariaFingerprintsManager
from ccl_malaria.logregs_analysis import logreg_results_to_pandas
import numpy as np

# Plot one: select an important feature
#           select a folding size (e.g. 2047)
#           check the colliding features there; select like 3
#           select molecules in which the selected features appear (like 3 also per feature)
#           generate a few and make the case we want if we can

# 1. Important features in blah
from ccl_malaria.molscatalog import MalariaCatalog
from ccl_malaria_sandbox.drawing_ala_rdkit import draw_in_a_grid_aligned_according_to_pattern
from integration.smartsviewer_utils import SmartsViewerRunner
from minioscail.common.eval import rank_sort

df = logreg_results_to_pandas(common_molids_cache=False)
print(df.columns)

# Select a folder, many ways, one quick one
u2f = df[df.folder_size == 4095].head().result[0].folded2unfolded()

# Collect importances

conds = ((df.num_cv_folds == 10) &
         (df.tol == 1E-4) &
         # (df.cv_seed == 0) & \
         (df.class_weight == 'auto') &
         (df.penalty == 'l1') &
         (df.C == 1) &
         (df.folder_seed < 1) &
         (df.folder_size == 0))

results_for_fs = df[conds].result

importances = []
for res in results_for_fs:
    importances += [res.logreg_coefs(fold).ravel() for fold in res.present_folds()]

mean_importance = np.mean(importances, axis=0)
std_importance = np.std(importances, axis=0)
features = np.arange(len(mean_importance))

# Importance is in the absolute value, sign indicates positive/negative feature
ranks, (sfeatures, smean_importance, sstd_importance) =\
    rank_sort(mean_importance, (features, mean_importance, std_importance))

# Negative
print('Super-negative features')
for f, mi, si in izip(sfeatures[:10], smean_importance[:10], sstd_importance[:10]):
    print('Feature: %d (%.2f +/- %.2f)' % (f, mi, si))

# Positives
print('Super-positive features')
for f, mi, si in izip(sfeatures[-10:], smean_importance[-10:], sstd_importance[-10:]):
    print('Feature: %d (%.2f +/- %.2f)' % (f, mi, si))

# Some preparations...
rng = np.random.RandomState(52)
mfm = MalariaFingerprintsManager(dset='lab')
mc = MalariaCatalog()

# Feature to fold and back
one_feature = sfeatures[-1]
# The feature was mapped to...
fold = u2f[one_feature]
# And these are other features that were also mapped to that fold
in_fold = np.where(u2f == fold)[0]
# Let's select a small subsample of colliding features
colliding_features = rng.choice(in_fold, 3)

cols = [one_feature] + colliding_features.tolist()
print('Selected features: ', cols)

# Now we will check which molecules contain these features using the sparse matrix
X = mfm.XCSC()
rows = [X[:, f].indices.tolist() for f in cols]

# These are the actual substructures
features = map(mfm.i2s, cols)


def i2ms(rows, top=3):
    return map(mfm.i2m, rows[:top])


molids = list(map(i2ms, rows))
print(molids)

mols = map(mc.molids2mols, molids)
classes = map(partial(mc.molids2labels, as01=True), molids)

print(mols)
print(classes)

for i, (s, ms) in enumerate(zip(features, mols)):
    # draw_in_a_grid_aligned_according_to_pattern(ms, s, '/home/santi/%d.png' % i, symbols=range(1000))
    Draw.MolsToGridImage(ms).save('/home/santi/%d.png' % i)
    svr = SmartsViewerRunner()
    svr.depict(s, '/home/santi/%d-sub.svg' % i)
