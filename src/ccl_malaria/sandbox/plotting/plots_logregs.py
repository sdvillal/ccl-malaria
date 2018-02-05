from __future__ import print_function, division

import collections
import os.path as op

import matplotlib.pyplot as plt
import numpy as np

from ccl_malaria.features import MalariaFingerprintsManager
from ccl_malaria.logregs_analysis import logreg_results_to_pandas
from ccl_malaria.sandbox.plotting.drawing_ala_rdkit import draw_in_a_grid_aligned_according_to_pattern
from ccl_malaria import MALARIA_EXPS_ROOT
from ccl_malaria.molscatalog import MalariaCatalog


####################
# TASK 1: results on the parameters selection experiment.
####################

def task1_res():
    """Results on the parameter selection experiment (task1)."""

    df = logreg_results_to_pandas()

    task1_res = df[
        # Cross-validation
        (df.num_cv_folds == 10) &
        (df.cv_seed == 0) &
        # Logistic regression
        ((df.penalty == 'l1') |
         (df.penalty == 'l2')) &
        ((df.C == 0.001) |
         (df.C == 0.01) |
         (df.C == 0.1) |
         (df.C == 0.5) |
         (df.C == 1) |
         (df.C == 5) |
         (df.C == 10) |
         (df.C == 100) |
         (df.C == 1000)) &
        ((df.class_weight == 'auto') |
         (df.class_weight == 'uniform')) &
        ((df.tol == 1E-2) |
         (df.tol == 1E-4)) &
        (~df.dual) &
        # Fingerprint Folders
        ((((df.folder_size == 1023) |
           (df.folder_size == 2047) |
           (df.folder_size == 4095) |
           (df.folder_size == 8191) |
           (df.folder_size == 16383)) &
          (df.folder_seed == 0)) |
         ((df.folder_size == 0) &
          (df.folder_seed < 0)))]

    assert len(task1_res) == 432

    return task1_res


def plot_regularization_exp(to_show='auc_mean', name='Mean AUC value', ylim=(0.5, 1)):
    """How do the different values of C influence the AUC? Does that change for different folding sizes?..."""
    df = task1_res()
    df_auto_l1 = df[(df.class_weight == 'auto') & (df.penalty == 'l1')]
    df_auto_l2 = df[(df.class_weight == 'auto') & (df.penalty == 'l2')]
    df_uniform_l1 = df[(df.class_weight == 'uniform') & (df.penalty == 'l1')]
    df_uniform_l2 = df[(df.class_weight == 'uniform') & (df.penalty == 'l2')]

    # Get AUCs and Cs for l1, 'auto'
    cs_l1_auto = collections.defaultdict(list)
    aucs_l1_auto = collections.defaultdict(list)
    for (C, fs), group in df_auto_l1.groupby(['C', 'folder_size']):
        if fs in [0, 1023, 2047, 4095, 8191, 16383]:  # Those that were planned in Task1
            print(C, fs, len(group), '%.2f' % group[to_show].mean())
            cs_l1_auto[fs].append(C)
            aucs_l1_auto[fs].append(group[to_show].mean())

    # Get AUCs and Cs for l2, 'auto'
    cs_l2_auto = collections.defaultdict(list)
    aucs_l2_auto = collections.defaultdict(list)
    for (C, fs), group in df_auto_l2.groupby(['C', 'folder_size']):
        if fs in [0, 1023, 2047, 4095, 8191, 16383]:  # Those that were planned in Task1
            print(C, fs, len(group), '%.2f' % group[to_show].mean())
            cs_l2_auto[fs].append(C)
            aucs_l2_auto[fs].append(group[to_show].mean())

    # Get AUCs and Cs for l1, 'uniform'
    cs_l1_uni = collections.defaultdict(list)
    aucs_l1_uni = collections.defaultdict(list)
    for (C, fs), group in df_uniform_l1.groupby(['C', 'folder_size']):
        if fs in [0, 1023, 2047, 4095, 8191, 16383]:  # Those that were planned in Task1
            print(C, fs, len(group), '%.2f' % group[to_show].mean())
            cs_l1_uni[fs].append(C)
            aucs_l1_uni[fs].append(group[to_show].mean())

    # Get AUCs and Cs for l2, 'uniform'
    cs_l2_uni = collections.defaultdict(list)
    aucs_l2_uni = collections.defaultdict(list)
    for (C, fs), group in df_uniform_l2.groupby(['C', 'folder_size']):
        if fs in [0, 1023, 2047, 4095, 8191, 16383]:  # Those that were planned in Task1
            print(C, fs, len(group), '%.2f' % group[to_show].mean())
            cs_l2_uni[fs].append(C)
            aucs_l2_uni[fs].append(group[to_show].mean())

    # First plot: L1, class_eight = 'auto
    plt.subplot(2, 2, 1)
    for k in sorted(cs_l1_auto.keys()):
        a = np.log(np.array(cs_l1_auto[k]))
        b = np.array(aucs_l1_auto[k])
        plt.plot(a, b, 'o-')
    plt.xlabel('Log(C)')
    plt.ylim(ylim)
    plt.ylabel(name)
    plt.legend(['folding: %i' % fs if fs != 0 else
                'No folding' for fs in sorted(cs_l1_auto.keys())], loc='lower right')
    plt.title('L1 regularization, Automatic class weight')

    # Second plot: L2, class_eight = 'auto'
    plt.subplot(2, 2, 3)
    for k in sorted(cs_l2_auto.keys()):
        a = np.log(np.array(cs_l2_auto[k]))
        b = np.array(aucs_l2_auto[k])
        plt.plot(a, b, 'o-')
    plt.xlabel('Log(C)')
    plt.ylim(ylim)
    plt.ylabel(name)
    plt.legend(['folding: %i' % fs if fs != 0 else
                'No folding' for fs in sorted(cs_l1_auto.keys())], loc='lower right')
    plt.title('L2 regularization, Automatic class weight')

    # Third plot: L1, class_eight = 'uniform'
    plt.subplot(2, 2, 2)
    for k in sorted(cs_l1_uni.keys()):
        a = np.log(np.array(cs_l1_uni[k]))
        b = np.array(aucs_l1_uni[k])
        plt.plot(a, b, 'o-')
    plt.xlabel('Log(C)')
    plt.ylim(ylim)
    plt.ylabel(name)
    plt.legend(['folding: %i' % fs if fs != 0 else
                'No folding' for fs in sorted(cs_l1_auto.keys())], loc='lower right')
    plt.title('L1 regularization, Uniform class weight')

    # Fourth plot: L2, class_eight = 'uniform'
    plt.subplot(2, 2, 4)
    for k in sorted(cs_l2_uni.keys()):
        a = np.log(np.array(cs_l2_uni[k]))
        b = np.array(aucs_l2_uni[k])
        plt.plot(a, b, 'o-')
    plt.xlabel('Log(C)')
    plt.ylim(ylim)
    plt.ylabel(name)
    plt.legend(['folding: %i' % fs if fs != 0 else
                'No folding' for fs in sorted(cs_l1_auto.keys())], loc='lower right')
    plt.title('L2 regularization, Uniform class weight')

    plt.show()


####################
# TASK 2: focus on fingerprint size exploration
####################

def task2_res():
    """Results on the fingerprint strategy exploration experiment (task2)."""

    df = logreg_results_to_pandas()

    task2_res = df[
        # Cross-validation
        (df.num_cv_folds == 10) &
        (df.cv_seed == 0) &
        # Logistic regression
        (df.penalty == 'l1') &
        (df.C == 1) &
        (df.class_weight == 'auto') &
        (df.tol == 1E-4) &
        (~df.dual) &
        # Fingerprint Folders
        ((((df.folder_size == 255) |
         (df.folder_size == 511) |
         (df.folder_size == 1023) |
         (df.folder_size == 2047) |
         (df.folder_size == 4095) |
         (df.folder_size == 8191) |
         (df.folder_size == 16383) |
         (df.folder_size == 32767) |
         (df.folder_size == 65537) |
         (df.folder_size == 131073)) &
          ((df.folder_seed == 0) |
           (df.folder_seed == 1) |
           (df.folder_seed == 2) |
           (df.folder_seed == 3))) |
         ((df.folder_size < 1) &
          (df.folder_seed < 0)))]

    assert len(task2_res) == 41  # N.B. not-folded is just one experiment, the rest 4

    return task2_res


def plot_auc_f_folding_size():
    """
    How do the different folding sizes influence the AUC? Does that change for different folding sizes? We analyze
    task 2 here.
    """
    df_t2 = task2_res()

    log_fss = []
    fss = []
    aucs = []
    stds = []
    for fs, group in df_t2.groupby(['folder_size']):
        if fs != 0:
            print(fs, len(group), '%.2f' % group.auc_mean.mean(), '%.2f' % group.auc_mean.std())
            log_fss.append(np.log(fs))
            fss.append(fs)
            aucs.append(group.auc_mean.mean())
            stds.append(group.auc_mean.std())

    plt.errorbar(log_fss, aucs, yerr=stds, fmt='o')
    plt.xlabel('Folding size')
    plt.xticks(log_fss, fss)
    plt.ylabel('Average AUC over 4 experiments')
    plt.show()


####################
# TASK 3: results on the deployment classifiers (explotation of good parameter combinations after task 1)
####################

def task3_res():
    """Results on the deployment classifiers computation experiment (task3)."""

    df = logreg_results_to_pandas()

    task3_res = df[
        # Cross-validation
        ((df.num_cv_folds == 3) |
         (df.num_cv_folds == 5) |
         (df.num_cv_folds == 7) |
         (df.num_cv_folds == 10)) &
        ((df.cv_seed == 0) |
         (df.cv_seed == 1) |
         (df.cv_seed == 2) |
         (df.cv_seed == 3) |
         (df.cv_seed == 4)) &
        # Logistic regression
        ((df.penalty == 'l1') | (df.penalty == 'l2')) &
        ((df.C == 1) | (df.C == 5)) &
        (df.class_weight == 'auto') &
        (df.tol == 1E-4) &
        (~df.dual) &
        # Fingerprint Folders
        (df.folder_size == 0) &
        (df.folder_seed < 0)]

    assert len(task3_res) == 80

    return task3_res


def average_feat_importance(penalty='l1'):

    df_t3 = task3_res()

    num_folds = []
    coefs_l = []
    stds_l = []
    for (cv_folds, c), group in df_t3.groupby(['num_cv_folds', 'C']):
        if c in [5, 1] and len(group) == 5:   # 5 different cv seeds were used
            print(cv_folds)
            coefs = []
            # Iterate over the 5 cv seeds for the same num_cv_folds:
            for _, gr in group.result.items():
                # average over the different folds
                coefs.append(np.mean(np.array([gr.logreg_coefs(i).ravel() for i in range(cv_folds)]), axis=0))
            num_folds.append(cv_folds)

            coefs_l.append(np.mean(np.array(coefs), axis=0))
            stds_l.append(np.std(np.array(coefs), axis=0))

    plt.subplot(2, 1, 1)          # better to have a rectangular shape anyway
    non_zeros = coefs_l[0] > 0.7  # only select those descriptors that have coefficients over 0.7

    num_non_zeros = np.sum(non_zeros)
    for i in range(len(coefs_l)):
        # noinspection PyTypeChecker
        plt.errorbar(range(num_non_zeros), coefs_l[i][non_zeros], yerr=stds_l[i][non_zeros], fmt='o')
    mfm = MalariaFingerprintsManager(dset='lab')
    # What are the substructures for which the coefficients are p
    non_zero_substruct = [mfm.i2s(i) for i, c in enumerate(coefs_l[0]) if c > 0.7]
    # noinspection PyTypeChecker
    plt.xticks(range(num_non_zeros), non_zero_substruct, rotation=90)
    plt.ylabel('Logistic regression coefficient')
    plt.title('Features with high coefficient, %s regularization' % penalty)
    plt.legend(['%i folds' % i for i in num_folds], loc='upper left')
    plt.show()


# noinspection PyUnusedLocal
def mols_having_best_feat(penalty='l1', c=1, num_folds=10):

    df_t3 = task3_res()

    coefs = []
    for cv_seed, group in df_t3.groupby(['cv_seed']):
        print(cv_seed, len(group))
    # Iterate over the 5 cv seeds for the same num_cv_folds:
    for _, gr in df_t3.result.items():
        # average over the different folds
        # print df_t3.C
        # print df_t3.num_cv_folds
        # print df_t3.cv_seed
        coefs.append(np.mean(np.array([gr.logreg_coefs(i).ravel() for i in range(num_folds)]), axis=0))

    av_coefs = np.mean(np.array(coefs), axis=0)
    index_of_best = np.argmax(av_coefs)
    mfm = MalariaFingerprintsManager(dset='lab')
    feat = mfm.i2s(index_of_best)
    print(feat)
    # feat = 'n1c(S(C)(=O)=O)sc(N)c1S(c)(=O)=O'
    molids = mfm.mols_with_feature(feat)
    mc = MalariaCatalog()
    mols = mc.molids2mols(molids)
    labels = mc.molids2labels(molids, as01=True)

    print(len(mols))
    draw_in_a_grid_aligned_according_to_pattern(mols, feat,
                                                op.join(MALARIA_EXPS_ROOT, 'logregs', 'Mols_having_best_fpt.png'),
                                                legends=molids, classes=labels)
    # img = MolsToGridImage(mols, legends=['%s %i'%(molids[k], labels[k]) for k, _ in enumerate(molids)], MolsPerRow=4)
    # img.show()


#################
# TODO Feature selection via logreg odds
#################


#################
# TODO Logistic regression parameter selection
#################

def auc_f_C(df):
    df = df[(df.folder_seed < 1)
            & (df.num_cv_folds == 10)
            & (df.cv_seed == 0)
            # (df.folder_size == 0) &
            # & (df.penalty == penalty)
            # & (df.class_weight == 'auto')
            & (df.tol == 1E-4)]
    for (penalty, C, fs, class_weight), group in df.groupby(['penalty',
                                                             'C',
                                                             'folder_size',
                                                             'class_weight']):
        print(class_weight, penalty, C, fs, len(group), '%.2f' % group.auc_mean.mean())


#################
# TODO Effect of folding on learning and model interpretation
#################

# fold size vs. auc
def auc_f_fold_size(df):
    df = df[(df.folder_seed < 1) &
            (df.num_cv_folds == 10) &
            (df.cv_seed == 0) &
            (df.penalty == 'l1') &
            (df.C == 1.) &
            (df.tol == 1E-4) &
            (df.class_weight == 'auto')]
    for size, group in df.groupby(['folder_size']):
        print(size, len(group), group.auc_mean.mean())


if __name__ == '__main__':
    plot_regularization_exp(ylim=(0, 1))
    plot_regularization_exp(to_show='enrichement5_mean', name='Mean enrichment at 5%', ylim=(0, 1))
    plot_auc_f_folding_size()
