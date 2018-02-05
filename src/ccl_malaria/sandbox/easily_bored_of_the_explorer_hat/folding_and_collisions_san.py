# coding=utf-8
from array import array
from collections import defaultdict

from scipy.sparse import vstack
import numpy as np
from sklearn.cross_validation import StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.utils import shuffle

from ccl_malaria.molscatalog import MalariaCatalog
from ccl_malaria.features import MalariaFingerprintsManager, MurmurFolder, fold_csr


def benchmark_test_faster_slower():

    from time import time
    mfm = MalariaFingerprintsManager(dset='lab')
    print('Reader')
    X = mfm.X()
    folder = MurmurFolder()

    start = time()
    slower = fold_csr(X, folder)
    print('Slow: %.2f s' % (time() - start))

    start = time()
    faster = fold_csr(X, folder)
    print('Hopefully faster: %.2f s' % (time() - start))

    assert np.allclose(slower.indices, faster.indices)
    assert np.allclose(slower.indptr, faster.indptr)
    assert np.allclose(slower.data, faster.data)


#####################
# TODO: manage correlated (or implicitly equal) features due mainly by the "growing" nature of Morgan fps.
# Correlated features can have an important effect on logreg.
#  http://en.wikipedia.org/wiki/Multicollinearity
#  http://www.statisticalhorizons.com/multicollinearity
#  http://stats.stackexchange.com/questions/74138/correlated-features-produce-strange-weights-in-logistic-regression
#  http://www.talkstats.com/showthread.php/18371-Logistic-regression-and-correlation
#  http://www.analyticbridge.com/group/analyticaltechniques/forum/topics/excluding-variables-from-a-logistic-regression-model-based-on
#  http://www.researchgate.net/post/How_to_test_multicollinearity_in_logistic_regression
#
#  http://metaoptimize.com/qa/questions/5205/when-to-use-l1-regularization-and-when-l2
#  http://metaoptimize.com/qa/questions/9463/selecting-features-correlated-with-those-chosen-by-l1-regularization-or-other-sparse-methods
#  http://metaoptimize.com/qa/questions/8763/compressive-sensing-conditions-for-recovery
#
#  Of course regularization helps, but what if we do not want to regularize? Options:
#    - Build a hierarchy of features using the centers information from rdkit (but I removed the centers ATM!)
#    - Look for (near) duplicates in the matrix (use LSH or something like that)
#    - Use non-regularized LogReg coefs to detect (near) duplicates
#    - Some other clever chemical approach? What if rdk canonicalisation is bogus?
#####################

# ~10¹² comparisons is quite unfeasible with less than 10 years to go to the competition
# for i, j in combinations(range(nf), 2):
#     xi = set(X.indices[X.indptr[i]:X.indptr[i+1]:])
#     xj = set(X.indices[X.indptr[j]:X.indptr[j+1]:])
#     if len(xi - xj) == 0:
#         print 'Same shitza because they are in the same mols????????'

# Let's try to base detection on the design matrix(ces)

# Option 1: plain old hashing to detect full duplicates
# X.indices.flags.writeable = False  # Make the views from this array hashable
# groups = defaultdict(list)
# for i in xrange(nf):
#     xi = X.indices[X.indptr[i]:X.indptr[i+1]:]
#     groups[xi.data].append(i)
#     if i > 0 and not i % 1000000:
#         print '%d of %d substructures hashed according to the molecules they pertain' % (i, nf)
# print len(groups)
# for molidxs, bitnos in groups.iteritems():
#     # print len(molidxs), len(bitnos)
#     if len(bitnos) == 40:
#         mols = []
#         for bitno in bitnos:
#             try:
#                 smiles = mfm.i2s(bitno)
#                 mol = AllChem.MolFromSmiles(smiles, sanitize=True)
#                 # http://comments.gmane.org/gmane.science.chemistry.rdkit.user/881
#                 # https://github.com/rdkit/rdkit/issues/46
#                 # AllChem.SanitizeMol(mol,
#                 #                     sanitizeOps=AllChem.SanitizeFlags.SANITIZE_ALL ^
#                 #                                 AllChem.SanitizeFlags.SANITIZE_KEKULIZE)
#                 #mol.UpdatePropertyCache(strict=False)
#                 #mol = AllChem.AddHs(mol, addCoords=True)
#                 AllChem.Compute2DCoords(mol)
#                 mols.append((smiles, mol))
#             except Exception, e:
#                 pass
#         img = Draw.MolsToGridImage([mol for _, mol in mols],
#                                    molsPerRow=4,
#                                    subImgSize=(200, 200),
#                                    legends=[smiles for smiles, _ in mols])
#         img.save('/home/santi/mistery-%s.png' % mols[0][0])
#         print '*' * 80


def duplicate_features(verbose=False):

    Xlab = MalariaFingerprintsManager(dset='lab', keep_ambiguous=False).X()
    Xunl = MalariaFingerprintsManager(dset='unl', keep_ambiguous=True).X()
    Xscr = MalariaFingerprintsManager(dset='scr', keep_ambiguous=True).X()

    bigX = vstack((Xlab, Xunl, Xscr))
    print('We are dealing with a matrix as big as %d molecules and %d features' % bigX.shape)

    def detect_duplicates(X):
        ne, nf = X.shape
        X = X.tocsc()
        X.indices.flags.writeable = False  # Make the views from this array hashable
        groups = defaultdict(lambda: array('I'))
        for i in range(nf):
            xi = X.indices[X.indptr[i]:X.indptr[i+1]:]
            groups[xi.data].append(i)
            if verbose and i > 0 and not i % 1000000:
                print('%d of %d substructures hashed according to the molecules they pertain' % (i, nf))
        # Represent each group by the first substructure
        return {v[0]: v for v in groups.values()}

    duplicates_in_labelled = detect_duplicates(Xlab)
    duplicates_in_all = detect_duplicates(bigX)

    print('Number of non-unique in labelled =  %d' % len(duplicates_in_labelled))
    print('Number of non-unique in all =      %d' % len(duplicates_in_all))

    # Now go back: feature to group in both of them,
    # stablish group correspondence and highlight not similar feats in test
    # We could learn with no-duplicates in labelling and bring this correspondance to testing interpretation
    # Would that be feasible?

    return duplicates_in_labelled.values(), duplicates_in_all.values()


def more_dupes_shitza():
    import joblib
    dlab, dall = duplicate_features()
    dlab = np.array([d[0] for d in dlab])
    dall = np.array([d[0] for d in dall])
    joblib.dump((dlab, dall), '/home/santi/dall.pickle', compress=3)
    dlab, dall = joblib.load('/home/santi/dall.pickle')
    print(len(dlab))
    print(len(dall))

    X, y = MalariaFingerprintsManager(dset='lab').Xy()
    X = X.tocsc()
    X = X[:, dall]
    X = X.tocsr()
    print(X.shape)

    num_cvfolds = 10
    X, y = shuffle(X, y, random_state=0)
    skf = StratifiedKFold(y, n_folds=num_cvfolds)

    estimator = LogisticRegression(penalty='l1', C=1., class_weight='auto')

    print('Running Full 0/1')
    aucs = cross_val_score(estimator, X, y=y, scoring='roc_auc', cv=skf, n_jobs=10)
    # noinspection PyStringFormat
    print('DALL FullCounts: %.2f +/- %.2f' % (np.mean(aucs), np.std(aucs)))

    X, y = MalariaFingerprintsManager(dset='lab').Xy()
    X = X.tocsc()
    X = X[:, dlab]
    X = X.tocsr()
    print(X.shape)

    num_cvfolds = 10
    X, y = shuffle(X, y, random_state=0)
    skf = StratifiedKFold(y, n_folds=num_cvfolds)

    estimator = LogisticRegression(penalty='l1', C=1., class_weight='auto')

    print('Running Full 0/1')
    aucs = cross_val_score(estimator, X, y=y, scoring='roc_auc', cv=skf, n_jobs=10)
    # noinspection PyStringFormat
    print('DLAB FullCounts: %.2f +/- %.2f' % (np.mean(aucs), np.std(aucs)))


if __name__ == '__main__':

    mfm = MalariaFingerprintsManager(dset='lab')
    mc = MalariaCatalog()
    X, y = mfm.Xy()
    molids = mfm.molids()

    # X = X[:4000, :]
    # y = y[:4000]

    num_cvfolds = 3
    X, y = shuffle(X, y, random_state=0)
    skf = StratifiedKFold(y, n_folds=num_cvfolds)

    # estimator = LogisticRegression(penalty='l1', C=1., class_weight=None)
    # estimator.fit(X, y)
    # np.save('/home/santi/coefs.npy', estimator.coef_)
    # np.save('/home/santi/intercept.npy', estimator.intercept_)

    # X = mfm.XCSC()
    # nf = X.shape[1]
    # print 'There are %d features' % nf
    #
    # coefs = np.load('/home/santi/coefs.npy').ravel()
    # intercept = np.load('/home/santi/intercept.npy')
    # print coefs.shape
    # important_shitza = np.where(coefs > 0.)[0]
    # for f in important_shitza:
    #     smiles = mfm.i2s(f)
    #     try:
    #         mol_sub = AllChem.MolFromSmiles(smiles)
    #         AllChem.Compute2DCoords(mol_sub)
    #     except:
    #         # mol_sub = pybel.readstring('smi', smiles)
    #         # with open('/home/santi/%s-mol.png' % smiles, 'w') as writer:
    #         #     writer.write(mol_sub.write('png'))
    #         mol_sub = AllChem.MolFromSmiles('O')
    #         AllChem.Compute2DCoords(mol_sub)
    #     molids = mfm.mols_with_feature(f)
    #     print smiles, len(molids)
    #     if len(molids) < 200:
    #         mols = mc.molids2mols(molids) + [mol_sub]
    #         molids += ['substructure']
    #         for m in mols:
    #             AllChem.Compute2DCoords(m)
    #             print AllChem.MolToSmiles(m)
    #         # img = Draw.MolsToGridImage(mols,
    #         #                            molsPerRow=4,
    #         #                            subImgSize=(200, 200),
    #         #                            legends=molids)
    #         # img.save('/home/santi/%s.png' % mfm.i2s(f))
    #         draw_in_a_grid_aligned_according_to_pattern(mols, smiles,
    #                                                     '/home/santi/%s.png' % mfm.i2s(f), legends=None)
    #
    estimator = LogisticRegression(penalty='l1', C=1., class_weight=None)
    print('Running Full 0/1')
    aucs = cross_val_score(estimator, X, y=y, scoring='roc_auc', cv=skf, n_jobs=3)
    # noinspection PyStringFormat
    print('FullCounts: %.2f +/- %.2f' % (np.mean(aucs), np.std(aucs)))
    #
    #
    # print '0/1 with folding'
    # for fs in (257, 513, 1025, 2047, 4097, 16383, 32767, 65537, 131073):
    #     # TODO: initialize to splitter, ask for randomisation
    #     folder = MurmurFolder(fold_size=fs)
    #     aucs = cross_val_score(estimator, fold_csr(X, folder), y=y, scoring='roc_auc', cv=skf, n_jobs=4)
    #     print '\t0/1Folded to %d: %.2f +/- %.2f' % (fs, np.mean(aucs), np.std(aucs))
