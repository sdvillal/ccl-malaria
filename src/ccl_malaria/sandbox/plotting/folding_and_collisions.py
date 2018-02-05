from __future__ import print_function, division
import os.path as op
import gzip
import time
try:
    import cPickle as pickle
except ImportError:
    import pickle

import argh
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Draw
from scipy.sparse import coo_matrix, csr_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.utils.murmurhash import murmurhash3_32 as murmur

from minioscail.common.misc import ensure_dir

from ccl_malaria.sandbox.plotting.models_with_rdkit_descs import give_cross_val_folds

from ccl_malaria import MALARIA_DATA_ROOT, MALARIA_EXPS_ROOT
from ccl_malaria.features import MalariaFingerprintsManager
from ccl_malaria.molscatalog import MalariaCatalog, read_labelled_smiles

import matplotlib.pyplot as plt

###########
# --- Flo code
###########
#
# from malaria.vowpal_wabbit_utils import cross_validate_vw_classification_model
#
# def exp_collisions(start=0, cv_seed=0):
#     bs = range(start, 23)
#     for b in bs:
#         _, _, auc = cross_validate_vw_classification_model(op.join(MALARIA_DATA_ROOT,
#                                                                    'vowpal', 'rdkit-ecfp100.vw.gz'),
#                                                            b=b,
#                                                            working_dir=op.join(MALARIA_EXPS_ROOT, 'collision_exps',
#                                                                                'b=%i_seed=%i' % (b, cv_seed)),
#                                                            verbose=False, seed=cv_seed)
#         print 'AUC obtained for b = %i: %.2f' % (b, auc)


# AUC obtained for b = 0: 0.43              # AUC obtained for b = 12: 0.87
# AUC obtained for b = 1: 0.45              # AUC obtained for b = 13: 0.87
# AUC obtained for b = 2: 0.45              # AUC obtained for b = 14: 0.87
# AUC obtained for b = 3: 0.47              # AUC obtained for b = 15: 0.87
# AUC obtained for b = 4: 0.50              # AUC obtained for b = 16: 0.87
# AUC obtained for b = 5: 0.57              # AUC obtained for b = 17: 0.88
# AUC obtained for b = 6: 0.60              # AUC obtained for b = 18: 0.90
# AUC obtained for b = 7: 0.66              # AUC obtained for b = 19: 0.90
# AUC obtained for b = 8: 0.72              # AUC obtained for b = 20: 0.91
# AUC obtained for b = 9: 0.79              # AUC obtained for b = 21: 0.91
# AUC obtained for b = 10: 0.84             # RESULTS FOR CV SEED=0
# AUC obtained for b = 11: 0.86

# Only for vowpal features
def manual_folding(hasher=murmur,
                   vw_file=op.join(MALARIA_DATA_ROOT, 'vowpal', 'rdkit-ecfp100.vw.gz'),
                   fold_sizes=None):
    if fold_sizes is None:
        fold_sizes = [257, 513, 1025, 2047, 4097, 16383, 32767, 65537, 131073]

    non_folded_feats = set()
    with gzip.open(vw_file, 'r') as reader:
        for line in reader:
            feats = set([f.split(':')[0] for f in line.split('|')[1].split()])
            non_folded_feats.update(feats)
    total_n_susbtructs = len(non_folded_feats)
    print('Total number of different substructures: %i.' % total_n_susbtructs)
    print('Hashing....')
    beg = time.time()
    hashed_subs = [hasher(sub) for sub in non_folded_feats]
    print('Hashing done in %f s.' % (time.time() - beg))
    collisions_by_hash = total_n_susbtructs - len(hashed_subs)
    print('The hashing produced %i collisions.' % collisions_by_hash)
    for fs in fold_sizes:
        hashed_fpt = [hashed_sub % fs for hashed_sub in hashed_subs]
        collisions_by_folding = len(hashed_subs) - len(np.unique(np.array(hashed_fpt)))
        print('Folding to a size of %i produced %i collisions' % (fs, collisions_by_folding))


FOLD_SIZES = [513, 1025, 2047, 4097, 16383, 32767, 65537, 131073, 262143, 524287, 1048577, 2097151]


def manual_folding_rdkit_feats(hasher=murmur, fold_size=513, dset='lab'):
    """Returns a triplet of lists:
    - list 1 contains all the substructures present in the dataset
    - list 2 contains the hashed codes for each of those substructures, in the same order
    - list 3 contains the folded bits for each of those hashed codes, in the same order
    All lists have the same length.
    Some information about collisions gets printed on the way.
    """
    mfm = MalariaFingerprintsManager(dset=dset)
    non_folded_feats = mfm.substructures()
    total_n_susbtructs = len(non_folded_feats)
    print('Total number of different substructures: %i.' % total_n_susbtructs)
    print('Hashing....')
    beg = time.time()
    hashed_subs = [hasher(sub, seed=0) for sub in non_folded_feats]
    print('Hashing done in %f s.' % (time.time() - beg))
    collisions_by_hash = total_n_susbtructs - len(set(hashed_subs))
    print('The hashing produced %i collisions.' % collisions_by_hash)
    hashed_fpt = [hashed_sub % fold_size for hashed_sub in hashed_subs]
    collisions_by_folding = len(hashed_subs) - len(np.unique(np.array(hashed_fpt)))
    print('Folding to a size of %i produced %i collisions' % (fold_size, collisions_by_folding))
    return non_folded_feats, hashed_subs, hashed_fpt


def get_folded_fpt_sparse(dset, folding_size=513):
    print(dset)
    mfm = MalariaFingerprintsManager(dset=dset)
    rows = []  # TODO: use python array: http://docs.python.org/2/library/array.html
    cols = []
    data = []
    # for each mol of the dataset, get the list of non-folded fpts, fold it and return it in a sparse matrix format
    with open(mfm.original_file, 'r') as reader:
        for i, line in enumerate(reader):
            info = line.split('\t')
            feats = [feat.split()[0] for feat in info[1:]]  # get rid of molid, and counts
            for feat in feats:
                # We hash and fold on the way
                hashed = murmur(feat, seed=0)
                folded = hashed % folding_size
                cols.append(folded)
                rows.append(i)
                data.append(1)
    cols = np.array(cols)
    data = np.array(data)
    rows = np.array(rows)
    # We ensure an adecuate shape in case a small dataset does not have the last bit set
    return coo_matrix((data, (rows, cols)), shape=(len(np.unique(rows)), folding_size)).tocsr()


#####################

# Factorization of the experiments.

####################

def start_exp(dest_dir, regularization, c, keep_ambiguous):
    """
    Define the classifier. Get the non-folded data.
    """
    ensure_dir(dest_dir)
    classifier = LogisticRegression(penalty=regularization, C=c, class_weight='auto')
    mfp = MalariaFingerprintsManager(dset='lab')
    X, y = mfp.Xy()
    if not keep_ambiguous:
        labelled = ~np.isnan(y)
        X = X[labelled, :]
        y = y[labelled]
    else:
        labelled = np.ones(len(y))
    return X, y, classifier, labelled


def run_cv(dest_dir, X, y, classifier, Xunl, Xscr):
    folds = give_cross_val_folds(y, 10)
    aucs = []
    for i, fold in enumerate(folds):
        fold_dir = op.join(dest_dir, 'fold=%i' % i)
        ensure_dir(fold_dir)
        scores_unl = None
        scores_scr = None
        train_indices = np.array([j for j in range(len(y)) if j not in fold])
        yte = y[fold]
        ytr = y[train_indices]
        Xte = X[fold, :]
        Xtr = X[train_indices, :]
        print('Training the classifier...')
        classifier.fit(Xtr, ytr)
        scores = classifier.predict_proba(Xte)[:, 1]
        if Xunl is not None:
            print('Scoring the unlabelled dataset...')
            scores_unl = classifier.predict_proba(Xunl)
        if Xscr is not None:
            print('Scoring the screening dataset...')
            scores_scr = classifier.predict_proba(Xscr)
        auc = roc_auc_score(yte, scores)
        aucs.append(auc)
        print('AUC for fold %i: %.2f' % (i, auc))
        print('********************')
        result = [classifier, scores, fold, auc, scores_unl, scores_scr]
        with open(op.join(fold_dir, 'results.pkl'), 'w') as writer:
            pickle.dump(result, writer)
    # noinspection PyStringFormat
    print('Average AUC: %.2f +/- %.2f' % (np.mean(np.array(aucs)), np.std(np.array(aucs))))
    print('********************')


def run_full_model(dest_dir, X, y, classifier, Xunl, Xscr):
    classifier.fit(X, y)
    scores_unl = None
    scores_scr = None
    if Xunl is not None:
        print('Scoring the unlabelled dataset...')
        scores_unl = classifier.predict_proba(Xunl)
    if Xscr is not None:
        print('Scoring the screening dataset...')
        scores_scr = classifier.predict_proba(Xscr)
    result = [classifier, scores_unl, scores_scr]
    with open(op.join(dest_dir, 'results.pkl'), 'w') as writer:
        pickle.dump(result, writer)


# --- Experiment with folding

def master_experiment(fold_sizes=513, compute_scores_unlabelled=True, compute_scores_screening=False, cv=True,
                      regularization='l1', c=1.0, keep_ambiguous=False,
                      dest_dir=op.join(MALARIA_EXPS_ROOT, 'folding_rdkit')):
    """
    Here we compute a scikit learn Logistic Regression for different folding sizes of the rdkit ecfp fingerprints.
    If asked, we do a 10-fold cross-validation and also apply the built model to the screening and/or unlabelled sets.
    """
    dest_dir = op.join(dest_dir, 'fs=%i' % fold_sizes)
    print('Starting the experiment...')
    _, y, classifier, labelled = start_exp(dest_dir, regularization, c, keep_ambiguous)
    X = get_folded_fpt_sparse('lab', folding_size=fold_sizes)
    X.data = np.ones(X.data.shape)
    X = X[labelled, :]
    print(X.shape)
    print('Got the training set.')
    Xunl = None
    Xscr = None
    if compute_scores_unlabelled:
        Xunl = get_folded_fpt_sparse('unl', folding_size=fold_sizes)
        print('Got the unlabelled set.')
        print(Xunl.shape)
    if compute_scores_screening:
        Xscr = get_folded_fpt_sparse('scr', folding_size=fold_sizes)
    if cv:
        print('Cross-validating the model...')
        print(run_cv(dest_dir, X, y, classifier, Xunl, Xscr))

    # Train the full model and predict if necessary the unlabelled and screening sets
    global_dir = op.join(dest_dir, 'full_model')
    ensure_dir(global_dir)
    print('Training the global classifier...')
    print(run_full_model(global_dir, X, y, classifier, Xunl, Xscr))


##################

# Experiment without folding

##################

def experiment_no_folding_01(compute_scores_unlabelled=True, compute_scores_screening=False, cv=True,
                             regularization='l1', c=1.0, keep_ambiguous=False,
                             dest_dir=op.join(MALARIA_EXPS_ROOT, 'folding_rdkit')):
    # We train the same models but without folding, and not using the counts but just 0 and 1
    dest_dir = op.join(dest_dir, 'no_folding')
    X, y, classifier, _ = start_exp(dest_dir, regularization, c, keep_ambiguous)
    X.data = np.ones(X.data.shape)
    Xunl = None
    Xscr = None
    if compute_scores_unlabelled:
        mfpunl = MalariaFingerprintsManager(dset='unl')
        Xunl = mfpunl.X()
        Xunl = csr_matrix((np.ones(Xunl.data.shape), Xunl.indices, Xunl.indptr), shape=(Xunl.shape[0], X.shape[1]))
    if compute_scores_screening:
        mfscr = MalariaFingerprintsManager(dset='scr')
        Xscr = mfscr.X()
        Xscr = csr_matrix((np.ones(Xscr.data.shape), Xscr.indices, Xscr.indptr), shape=(Xscr.shape[0], X.shape[1]))
    if cv:
        print(X.shape, Xunl.shape)
        print('Cross-validating the model...')
        print(run_cv(dest_dir, X, y, classifier, Xunl, Xscr))

    # Train the full model and predict if necessary the unlabelled and screening sets
    global_dir = op.join(dest_dir, 'full_model')
    ensure_dir(global_dir)
    print('Training the global classifier...')
    print(run_full_model(global_dir, X, y, classifier, Xunl, Xscr))


##################

# Analysis

##################

def get_AUC(fold_size):
    """
    Reads the auc from the logfile
    """
    if fold_size is None:
        # then we want AUC without folding
        logfile = op.join(op.expanduser('~'), 'collisions_nofolding.log' % fold_size)
        with open(logfile, 'r') as reader:
            text = reader.readall()  # FIXME: make py3
            auc = float(text.partition('Average AUC: ')[2].partition(' +/-')[0])
    else:
        logfile = op.join(op.expanduser('~'), 'collisions_fold_size=%i.log' % fold_size)
        with open(logfile, 'r') as reader:
            text = reader.read()
            auc = float(text.partition('Average AUC: ')[2].partition(' +/-')[0])
    return auc


def plot_AUC_f_folding():
    """
    Plots AUC as a function of the log of the folding size.
    """
    fold_sizes = np.log(np.array(FOLD_SIZES))
    aucs = []
    for fs in FOLD_SIZES:
        aucs.append(get_AUC(fs))
    aucs = np.array(aucs)
    plt.plot(fold_sizes, aucs, marker='o')
    plt.xlabel('log(folding size)')
    plt.ylabel('AUC')
    plt.ylim(0.5, 1)
    plt.show()


def plot_mispredictions_f_folding():
    """
    Plots the number of mispredicted compounds as a function of the log of the folding size.
    """
    fold_sizes = np.log(np.array(FOLD_SIZES))
    misses = []
    for fs in FOLD_SIZES:
        misses.append(len(mispredicted_compounds(fs)))
    misses = np.array(misses)
    plt.plot(fold_sizes, misses, marker='o')
    plt.xlabel('log(folding size)')
    plt.ylabel('Number of mispredicted compounds during CV')
    # plt.ylim(0.5,1)
    plt.show()


def get_feature_importances_nofolding_01():
    """
    Retrieves from the non-null coefficients of the logistic regression the corresponding substructures and saves them
    in a file.
    """
    if not op.exists(op.join(MALARIA_EXPS_ROOT, 'folding_rdkit', 'no_folding', 'full_model',
                             'important_features.txt')):
        with open(op.join(MALARIA_EXPS_ROOT,
                          'folding_rdkit',
                          'no_folding',
                          'full_model',
                          'results.pkl'), 'r') as reader:
            classifier, scores_unl, scores_scr = pickle.load(reader)
        coefs = classifier.coef_.ravel()
        f = MalariaFingerprintsManager(dset='lab')
        structs = [f.i2s(i) for i, c in enumerate(coefs) if c != 0]
        with open(op.join(MALARIA_EXPS_ROOT, 'folding_rdkit', 'no_folding', 'full_model',
                          'important_features.txt'), 'w') as writer:
            for c, s in zip(coefs[coefs != 0], structs):
                writer.write('%g\t%s\n' % (c, s))
        return zip(coefs[coefs != 0], structs)
    else:
        important_feats = []
        with open(op.join(MALARIA_EXPS_ROOT, 'folding_rdkit', 'no_folding', 'full_model',
                          'important_features.txt'), 'r') as reader:
            for line in reader:
                importance = float(line.split('\t')[0])
                subst = line.split('\t')[1].strip()
                important_feats.append((importance, subst))
        return important_feats


def get_n_most_important_feats_ref(n=20):
    """
    Keeps only the features for which the coefficients are the n/2 top highest and the n/2 top lowest.
    """
    sorted_important_feats = sorted(get_feature_importances_nofolding_01(), key=lambda tup: tup[0])
    return sorted_important_feats[:n/2] + sorted_important_feats[-n/2:]


def get_feature_importances_folded(folding_size):
    """Returns a list of tuples (coeff, list of structures assigned to the bit).

    Parameters:
      - fs: folding_size
    """
    if not op.exists(op.join(MALARIA_EXPS_ROOT,
                             'folding_rdkit', 'fs=%i' % folding_size,
                             'full_model',
                             'important_features.txt')):
        with open(op.join(MALARIA_EXPS_ROOT,
                          'folding_rdkit',
                          'fs=%i' % folding_size,
                          'full_model',
                          'results.pkl'), 'r') as reader:
            classifier, scores_unl, scores_scr = pickle.load(reader)
        # These are the coefficients from the log-reg
        coefs = classifier.coef_.ravel()
        # Get the corresponding folded indices
        weight_indices = [i for i, c in enumerate(coefs) if c != 0]
        # Get the "potentially" corresponding substructures (because of collisions)
        non_folded_feats, _, folded_fpt = manual_folding_rdkit_feats(fold_size=folding_size)
        non_folded_feats = np.array(non_folded_feats)
        folded_fpt = np.array(folded_fpt)
        equivalent_substructures = []
        for fi in weight_indices:
             equivalent_substructures.append(list(non_folded_feats[folded_fpt == fi]))
        structs = zip(coefs[coefs != 0], equivalent_substructures)
        with open(op.join(MALARIA_EXPS_ROOT, 'folding_rdkit', 'fs=%i' % folding_size,
                          'full_model', 'important_features.txt'), 'w') as writer:
            for c, s in structs:
                writer.write('%g\t%s\n' % (c, '\t'.join(s)))
        return structs
    else:
        important_feats = []
        with open(op.join(MALARIA_EXPS_ROOT, 'folding_rdkit', 'fs=%i' % folding_size,
                          'full_model', 'important_features.txt'), 'r') as reader:
            for line in reader:
                importance = float(line.split('\t')[0])
                subst = [s.strip() for s in line.split('\t')[1:]]
                important_feats.append((importance, subst))
        return important_feats


def get_n_most_important_feats_folded(fs, n=20):
    sorted_important_feats = sorted(get_feature_importances_folded(fs), key=lambda tup: tup[0])
    return sorted_important_feats[:n/2] + sorted_important_feats[-n/2:]


def compare_n_most_important_feats(n=20):
    reference = get_n_most_important_feats_ref(n)
    for fs in FOLD_SIZES:
        print('Comparing non folded results with results from folding to a size of %i' % fs)
        folded_imp = get_n_most_important_feats_folded(fs, n)
        for i, ref in enumerate(reference):
            print('Structure %i: %s   Coefficient: %.2f' % (i, ref[1], ref[0]))
            if ref[1] in folded_imp[i][1]:
                print('The structure was also found at the position %i among %i other features' % (
                      i, len(folded_imp[i][1])))
            else:
                for j in range(len(folded_imp)):
                    if ref[1] in folded_imp[j][1]:
                        print('The structure was found at the position %i '
                              'among %i other features with coefficient %.2f' % (j,
                                                                                 len(folded_imp[j][1]),
                                                                                 folded_imp[j][0]))


# noinspection PyUnusedLocal
def where_is_my_favorit_feat(feat, folding_size=513, dset='lab'):
    non_folded_feats, hashed_feats, folded_feats = manual_folding_rdkit_feats(fold_size=folding_size)
    hashed = murmur(feat, seed=0)
    folded = hashed % folding_size
    hashed_feats = np.array(hashed_feats)
    folded_feats = np.array(folded_feats)
    collision_at_hashing = hashed_feats[hashed_feats == hashed]
    collisions_at_folding = folded_feats[folded_feats == folded]
    if len(collision_at_hashing) == 1:
        print('There were no collisions at hashing time for substructure %s' % feat)
    if len(collisions_at_folding) > 1:
        print('Substructure %s collided with %i others at bit %i at folding time' %
              (feat, len(collisions_at_folding) - 1, folded))
    return non_folded_feats[collisions_at_folding], collision_at_hashing, collisions_at_folding


def feat2score(smis):
    with open(op.join(MALARIA_EXPS_ROOT, 'folding_rdkit', 'no_folding', 'full_model', 'results.pkl'), 'r') as reader:
        classifier, _, _ = pickle.load(reader)
        coefs = classifier.coef_.ravel()
        mfm = MalariaFingerprintsManager(dset='lab')
        indices = [mfm.s2i(smi) for smi in smis]
    return [coefs[ind] for ind in indices]


def from_feat_back_to_mols(dset, smi):
    """
    Retrieves the list of molecules that contain the given feature in the given dataset. THIS IS EXTREMELY SLOW!!!!!!!
    """
    mols = []
    molids = []
    indices = []
    classes = []
    mfm = MalariaFingerprintsManager(dset=dset)
    print('I am here')
    with open(mfm.original_file, 'r') as reader:
        for i, line in enumerate(reader):
            info = line.split('\t')
            molid = info[0]
            ecfps = [feat.split()[0] for feat in info[1:]]
            if smi in ecfps:
                print(molid)
                molids.append(molid)
                indices.append(i)
                mc = MalariaCatalog()
                classes.append(mc.label(molid, as01=True))
                mols.append(None)
    print('Got the first lists')
    molids = np.array(molids)
    indices = np.array(indices)
    classes = np.array(classes)
    mols = np.array(mols)
    # Now we need to retrieve the real rdkit mols for each molid.
    if dset == 'lab':
        for molid, _, _, _, _, smiles in read_labelled_smiles():
            if molid in molids:
                mols[molids == molid] = smiles
    return zip(mols, molids, indices, classes)


def from_feat_back_to_mols_faster(dset, smi):
    """
    Retrieves the list of molecules that contain the given feature in the given dataset.
    """
    # The non-folded version is easy
    mfm = MalariaFingerprintsManager(dset=dset)
    X = mfm.X()
    col = mfm.s2i(smi)  # the column where we have to look for in the X matrix
    cooX = X.tocoo()
    indices_mols = cooX.row[cooX.col == col]
    molids = [mfm.i2m(i) for i in indices_mols]
    mc = MalariaCatalog()
    activities = [mc.label(molid, as01=True) for molid in molids]
    mols = mc.molids2mols(molids)
    return zip(mols, molids, indices_mols, activities)


def mispredicted_compounds(folding_size=None):
    """
    At each fold, collect the list of mispredicted compounds and assemble it into one list of molids
    """
    FOLDS = range(10)
    mfm = MalariaFingerprintsManager(dset='lab')
    mispredicted = []
    if folding_size is None:
        path = op.join(MALARIA_EXPS_ROOT, 'folding_rdkit', 'no_folding')
    else:
        path = op.join(MALARIA_EXPS_ROOT, 'folding_rdkit', 'fs=%i' % folding_size)
    for fold in FOLDS:
        with open(op.join(path, 'fold=%i' % fold, 'results.pkl'), 'r') as reader:
            _, scores, fold, _, _, _ = pickle.load(reader)
            scores = scores >= 0.5    # dummy threshold
            molids_test = [mfm.i2m(i) for i in fold]
            mc = MalariaCatalog()
            classes_test = [mc.label(molid, as01=True) for molid in molids_test]
            for i, mol in enumerate(molids_test):
                if scores[i] != classes_test[i] and not np.isnan(classes_test[i]):
                    mispredicted.append(mol)
    return mispredicted


def molid_to_rankings(molid):
    """
    Compare accross the models (different folding sizes) how the ranking of a precise molecule can vary. Returns a list
    of rankings from the smallest folding size up to the non-folded model.
    """
    rankings = []
    cv_folds = range(10)
    mfm = MalariaFingerprintsManager(dset='lab')
    raw_id = mfm.m2i(molid)
    print(raw_id)
    # Now we will cumulate the results of CV and get the ranking
    for fs in FOLD_SIZES:
        print(fs)
        path = op.join(MALARIA_EXPS_ROOT, 'folding_rdkit', 'fs=%i' % fs)
        scores = []
        folds = []
        for cvf in cv_folds:
            with open(op.join(path, 'fold=%i' % cvf, 'results.pkl'), 'r') as reader:
                _, score, fold, _, _, _ = pickle.load(reader)
                scores += list(score)
                folds += list(fold)
        ordered = sorted(zip(folds, scores), key=lambda t: t[1], reverse=True)
        ordered_folds, _ = zip(*ordered)
        rankings.append(ordered_folds.index(raw_id))
    path = op.join(MALARIA_EXPS_ROOT, 'folding_rdkit', 'no_folding')
    scores = []
    folds = []
    for cvf in cv_folds:
        with open(op.join(path, 'fold=%i' % cvf, 'results.pkl'), 'r') as reader:
            _, score, fold, _, _, _ = pickle.load(reader)
            scores += list(score)
            folds += list(fold)
    ordered = sorted(zip(folds, scores), key=lambda t: t[1], reverse=True)
    ordered_folds, _ = zip(*ordered)
    rankings.append(ordered_folds.index(raw_id))
    return rankings


def molid_to_ranking(molid, fs=513):
    """
    Returns the ranking of the specified molecule in the specified model
    """
    cv_folds = range(10)
    mfm = MalariaFingerprintsManager(dset='lab')
    raw_id = mfm.m2i(molid)
    if fs is not None:
        path = op.join(MALARIA_EXPS_ROOT, 'folding_rdkit', 'fs=%i' % fs)
        if not op.isfile(op.join(path, 'ranking.pkl')):
            scores = []
            folds = []
            for cvf in cv_folds:
                with open(op.join(path, 'fold=%i' % cvf, 'results.pkl'), 'r') as reader:
                    _, score, fold, _, _, _ = pickle.load(reader)
                    scores += list(score)
                    folds += list(fold)
            ordered = sorted(zip(folds, scores), key=lambda t: t[1], reverse=True)
            ordered_folds, _ = zip(*ordered)
            with open(op.join(path, 'ranking.pkl'), 'w') as writer:
                pickle.dump(ordered, writer)
        else:
            with open(op.join(path, 'ranking.pkl'), 'r') as reader:
                ordered = pickle.load(reader)
                ordered_folds, _ = zip(*ordered)
    else:
        path = op.join(MALARIA_EXPS_ROOT, 'folding_rdkit', 'no_folding')
        if not op.isfile(op.join(path, 'ranking.pkl')):
            scores = []
            folds = []
            for cvf in cv_folds:
                with open(op.join(path, 'fold=%i' % cvf, 'results.pkl'), 'r') as reader:
                    _, score, fold, _, _, _ = pickle.load(reader)
                    scores += list(score)
                    folds += list(fold)
            ordered = sorted(zip(folds, scores), key=lambda t: t[1], reverse=True)
            ordered_folds, _ = zip(*ordered)
            with open(op.join(path, 'ranking.pkl'), 'w') as writer:
                pickle.dump(ordered, writer)
        else:
            with open(op.join(path, 'ranking.pkl'), 'r') as reader:
                ordered = pickle.load(reader)
                ordered_folds, _ = zip(*ordered)
    return ordered_folds.index(raw_id)


def plot_mols_having_a_common_feat(mols, cf):
    """
    Mols are given as a list of (smiles, molids, ids, classes) (as in output of from_feat_back_to_mols())
    cf: the common feature, also given as smile string
    Returns: a grid plot with the different molecules having highlighted the common feature
    """
    rdkit_mols = []
    for mol, _, _, _ in mols:
        rdkit_mols.append(Chem.MolFromSmiles(mol))

    p = Chem.MolFromSmarts(cf)
    subms = [x for x in rdkit_mols if x.HasSubstructMatch(p)]
    AllChem.Compute2DCoords(p)
    for m in subms:
        AllChem.GenerateDepictionMatching2DStructure(m, p)
    subms.insert(0, p)
    legends = [mol[1] + ' ' + str(mol[-1][0]) for mol in mols]
    legends.insert(0, 'Common feature')
    img = Draw.MolsToGridImage(subms, molsPerRow=4, subImgSize=(200, 200),
                               legends=legends, kekulize=False)
    img.save(op.join(MALARIA_EXPS_ROOT, 'folding_rdkit', 'Mols_having_%s.png' % cf))


############
# Elaborated analysis
############


def exp_with_best_feature():
    best_feat = get_n_most_important_feats_ref()[-1]
    molset = from_feat_back_to_mols('lab', best_feat[1])
    # to be continued...
    return best_feat, molset


def ranking_diffs_folding(fs=513, index_bit=3):
    """
    We take the example of fold 513, where "best bit" 4 (coef=-0.63) contains collisions with features
    having both positive and negative coefs in the non-folded model.
    """
    ranking_differences = []
    score, feats = get_n_most_important_feats_folded(fs)[index_bit]
    nonfolded_coefs, nonfolded_feats = zip(*get_feature_importances_nofolding_01())
    for feat in feats:
        # what is the coef value in non folded model?? (only if it is non 0)
        if feat in nonfolded_feats:
            print(feat)
            print(nonfolded_coefs[nonfolded_feats.index(feat)])
            mols = from_feat_back_to_mols_faster('lab', feat)
            molids = [mol[1] for mol in mols]
            classes = [mol[-1] for mol in mols]
            print(len(molids))
            for i, molid in enumerate(molids):
                if not np.isnan(classes[i]):
                    print(molid, classes[i])
                    rank_folded = molid_to_ranking(molid)
                    # noinspection PyTypeChecker
                    rank_nonfolded = molid_to_ranking(molid, fs=None)
                    ranking_differences.append(rank_folded - rank_nonfolded)
    return ranking_differences


def show_effect_folding_ranking():
    ranking_differences_bad = ranking_diffs_folding()
    # this one is only corresponding to 2 positive features in the non-folded model
    ranking_differences_good = ranking_diffs_folding(index_bit=14)
    plt.boxplot([ranking_differences_bad, ranking_differences_good])
    plt.xticks([1, 2], ['problematic collision', 'non problematic collision'])
    plt.ylabel('Ranking in folded model - Ranking in non-folded model')
    plt.savefig(op.join(MALARIA_EXPS_ROOT, 'folding_rdkit', 'Differences_of_ranking.png'))


def coefs_distributions_post_folding(fs=513):
    populations = []
    top_features = get_n_most_important_feats_folded(fs)
    for score, feats in top_features:
        print(score)
        print(len(feats))
        # Now for each substructure in feats, we have to find its index and the score value in non-folded
        coefs_colliding = feat2score(feats)
        print(len(coefs_colliding), max(coefs_colliding), min(coefs_colliding))
        populations.append(coefs_colliding)
    # plot each population as a boxplot
    plt.boxplot(populations)
    plt.rcParams['figure.figsize'] = 15, 30
    plt.xticks(range(1, len(top_features) + 1),
               ['%.2f' % score for score in [top_feature[0] for top_feature in top_features]])
    plt.xlabel('Coefficients of the best features in folding size %i' % fs)
    plt.ylabel('Coefficient population for the corresponding\n colliding features in the non-folded model')
    plt.show()


def cl():
    for fold in FOLD_SIZES:
        print('python2 -u malaria/folding_and_collisions.py master-experiment --fold-sizes %i '
              '&>~/collisions_fold_size=%i.log' % (fold, fold))


if __name__ == '__main__':
    parser = argh.ArghParser()
    parser.add_commands([cl,
                         # exp_collisions,
                         manual_folding,
                         manual_folding_rdkit_feats,
                         master_experiment,
                         experiment_no_folding_01,
                         show_effect_folding_ranking])
    parser.dispatch()

#
# Here we check what happened during the master experiment: how does AUC evolved with folding? What happens to the
# most important features identified with the non-folded model? (how many collisions do they suffer). What are the
# newly predicted most important substructures? How did the order of molecules containing the important features has
# changed compared to the non-folded version?
#
