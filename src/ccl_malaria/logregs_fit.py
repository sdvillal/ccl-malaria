# coding=utf-8
"""Experiments with Morgan fingerprints and logistic regression (sklearn and vowpal wabbit)."""
from __future__ import print_function, division
from collections import OrderedDict
from copy import copy
import hashlib
from itertools import product
import os.path as op
import json
from time import time

import argh
import h5py
import joblib
from sklearn.base import clone
from sklearn.linear_model.logistic import LogisticRegression
from sklearn.metrics import roc_auc_score
import numpy as np

from ccl_malaria import MALARIA_EXPS_ROOT, info
from minioscail.common.eval import cv_splits, enrichment_at
from ccl_malaria.features import MurmurFolder, MalariaFingerprintsExampleSet
from ccl_malaria.results import predict_malaria_unlabelled, save_molids
from minioscail.common.config import mlexp_info_helper
from minioscail.common.misc import ensure_dir


MALARIA_LOGREGS_EXPERIMENT_ROOT = op.join(MALARIA_EXPS_ROOT, 'logregs')


#######################################
# The data version we will work with
#######################################

def malaria_logreg_fpt_providers(folder):
    """Returns a tuple (rf_lab, rf_amb, rf_unl, rf_scr) with the example-sets used in the logreg experiments."""
    rf_lab = MalariaFingerprintsExampleSet(dset='lab', remove_ambiguous=True, zero_dupes='all', folder=folder)
    rf_unl = MalariaFingerprintsExampleSet(dset='unl', remove_ambiguous=False, zero_dupes='all', folder=folder)
    rf_amb = MalariaFingerprintsExampleSet(dset='amb', zero_dupes='all', folder=folder)
    rf_scr = MalariaFingerprintsExampleSet(dset='scr',
                                           remove_ambiguous=False,
                                           zero_dupes=None,  # N.B. dupes do not matter with logreg,
                                           # faster not to do it (at least when streaming in my tests)
                                           folder=folder)
    return rf_lab, rf_amb, rf_unl, rf_scr


#######################################
# FIT the logistic regression models
#######################################

@argh.arg('--cv-seeds', nargs='+', type=int)
def fit(dest_dir=MALARIA_LOGREGS_EXPERIMENT_ROOT,
        # Logreg params
        penalty='l1',
        C=1.0,
        class_weight_auto=False,
        dual=False,
        tol=1e-4,
        fit_intercept=True,
        intercept_scaling=1,
        # CV params
        num_cv_folds=10,
        cv_seeds=(0,),
        save_unlabelled_predictions=False,
        save_fold_model=False,
        min_fold_auc=0.88,
        # Fingerprint folding params
        fingerprint_folder_seed=0,
        fingerprint_fold_size=1023,
        # Computational requirements params
        force=False,
        chunksize=1000000,
        max_logreg_tol=1E-5):
    """Logistic regression experiment using the liblinear wrapper in sklearn.
    Generates cross-val results
    """

    if max_logreg_tol is not None and tol < max_logreg_tol:
        info('Ignoring long intolerant experiments')
        return

    info('Malaria logregs experiment')

    # Command line type inference is rotten...
    C = float(C)
    tol = float(tol)
    intercept_scaling = float(intercept_scaling)
    num_cv_folds = int(num_cv_folds)
    min_fold_auc = float(min_fold_auc)
    fingerprint_folder_seed = int(fingerprint_folder_seed)
    fingerprint_fold_size = int(fingerprint_fold_size)
    chunksize = int(chunksize)

    # Example providers
    folder = None if fingerprint_fold_size < 1 else MurmurFolder(seed=fingerprint_folder_seed,
                                                                 fold_size=fingerprint_fold_size)
    rf_lab, rf_amb, rf_unl, rf_scr = malaria_logreg_fpt_providers(folder)
    info('Data description: %s' % rf_lab.configuration().id(nonids_too=True))

    # Experiment context: data
    data_id = rf_lab.configuration().id(nonids_too=True)
    data_dir = op.join(dest_dir, data_id)
    ensure_dir(data_dir)

    for cv_seed in cv_seeds:

        # Command line type inference is rotten...
        cv_seed = int(cv_seed)

        # Deterministic randomness
        my_rng = np.random.RandomState(seed=cv_seed)

        # Experiment context: model
        logreg_params = OrderedDict((
            ('penalty', penalty),
            ('C', C),
            ('class_weight', 'auto' if class_weight_auto else None),
            ('dual', dual),
            ('tol', tol),
            ('fit_intercept', fit_intercept),
            ('intercept_scaling', intercept_scaling),
            ('random_state', my_rng.randint(low=0, high=4294967294)),
            # Changed, from original 1000**4, to make liblinear happy
        ))
        model_setup = LogisticRegression(**logreg_params)
        model_id = 'skllogreg__%s' % '__'.join(['%s=%s' % (k, str(v)) for k, v in logreg_params.items()])
        model_dir = op.join(data_dir, model_id)
        ensure_dir(model_dir)
        info('Model: %s' % model_id)

        # Experiment context: eval
        eval_id = 'cv__cv_seed=%d__num_folds=%d' % (cv_seed, num_cv_folds)
        eval_dir = op.join(model_dir, eval_id)
        ensure_dir(eval_dir)
        info('Eval: %d-fold cross validation (seed=%d)' % (num_cv_folds, cv_seed))

        # Already done?
        info_file = op.join(eval_dir, 'info.json')
        if op.isfile(info_file) and not force:
            info('\tAlready done, skipping...')
            return  # Oh well, a lot have been done up to here... rework somehow

        # Anytime we see this file, we know we need to stop
        stop_computing_file = op.join(eval_dir, 'STOP_BAD_FOLD')

        # --- Time to work!

        # Save model config
        joblib.dump(model_setup, op.join(model_dir, 'model_setup.pkl'), compress=3)

        # Read labelled data in
        info('Reading data...')
        X, y = rf_lab.Xy()
        info('ne=%d; nf=%d' % rf_lab.X().shape)

        # Save molids... a bit too ad-hoc...
        save_molids(data_dir, 'lab', rf_lab.ids())
        if save_unlabelled_predictions:
            save_molids(data_dir, 'unl', rf_unl.ids())
            save_molids(data_dir, 'scr', rf_scr.ids())
            save_molids(data_dir, 'amb', rf_amb.ids())

        # Save folding information.
        # By now, all the folds have already been computed:
        #   - because we cached X
        #   - and in this case we are warranted that no new unfolded features will appear at test time
        if folder is not None:
            info('Saving the map folded_features -> unfolded_feature...')
            folded2unfolded_file = op.join(data_dir, 'folded2unfolded.h5')
            if not op.isfile(folded2unfolded_file):
                with h5py.File(folded2unfolded_file) as h5:
                    h5['f2u'] = folder.folded2unfolded()
            folder_light_file = op.join(data_dir, 'folder.pkl')
            if not op.isfile(folder_light_file):
                folder_light = copy(folder)  # Shallow copy
                folder_light.clear_cache()
                joblib.dump(folder_light, folder_light_file, compress=3)

        # Cross-val splitter
        cver = cv_splits(num_points=len(y),
                         Y=y,
                         num_folds=num_cv_folds,
                         rng=my_rng,
                         stratify=True)

        # Fit and classify
        for cv_fold_num in range(num_cv_folds):

            fold_info_file = op.join(eval_dir, 'fold=%d__info.json' % cv_fold_num)
            if op.isfile(fold_info_file):
                info('Fold %d already done, skipping' % cv_fold_num)
                continue

            if op.isfile(stop_computing_file):
                info('Bad fold detected, no more computations required')
                break

            # Split into train/test
            train_i, test_i = cver(cv_fold_num)
            Xtrain, ytrain = X[train_i, :], y[train_i]
            Xtest, ytest = X[test_i, :], y[test_i]
            assert len(set(train_i) & set(test_i)) == 0

            # Copy the model...
            model = clone(model_setup)

            start = time()
            info('Training...')
            model.fit(Xtrain, ytrain)
            train_time = time() - start
            info('Model fitting has taken %.2f seconds' % train_time)

            if save_fold_model:
                info('Saving trained model')
                joblib.dump(model, op.join(eval_dir, 'fold=%d__fitmodel.pkl' % cv_fold_num), compress=3)

            info('Predicting and saving results...')
            with h5py.File(op.join(eval_dir, 'fold=%d__scores.h5' % cv_fold_num), 'w') as h5:

                start = time()

                # Test indices
                h5['test_indices'] = test_i

                # Model
                h5['logreg_coef'] = model.coef_
                h5['logreg_intercept'] = model.intercept_

                # Test examples
                info('Scoring test...')
                scores_test = model.predict_proba(Xtest)
                fold_auc = roc_auc_score(ytest, scores_test[:, 1])
                fold_enrichment5 = enrichment_at(ytest, scores_test[:, 1], percentage=0.05)
                info('Fold %d ROCAUC: %.3f' % (cv_fold_num, fold_auc))
                info('Fold %d Enrichment at 5%%: %.3f' % (cv_fold_num, fold_enrichment5))
                h5['test'] = scores_test.astype(np.float32)

                if save_unlabelled_predictions:
                    predict_malaria_unlabelled(model,
                                               h5,
                                               rf_amb=rf_amb,
                                               rf_scr=rf_scr,
                                               rf_unl=rf_unl,
                                               chunksize=chunksize)

                test_time = time() - start
                info('Predicting has taken %.2f seconds' % test_time)

                # Finally save meta-information for the fold
                metainfo = mlexp_info_helper(
                    title='malaria-trees-oob',
                    data_setup=data_id,
                    model_setup=model_id,
                    exp_function=fit,
                )
                metainfo.update((
                    ('train_time', train_time),
                    ('test_time', test_time),
                    ('auc', fold_auc),
                    ('enrichment5', fold_enrichment5),
                ))
                with open(fold_info_file, 'w') as writer:
                    json.dump(metainfo, writer, indent=2, sort_keys=False)

                # One last thing, should we stop now?
                if fold_auc < min_fold_auc:
                    stop_message = 'The fold %d was bad (auc %.3f < %.3f), skipping the rest of the folds' % \
                                   (cv_fold_num, fold_auc, min_fold_auc)
                    info(stop_message)
                    with open(stop_computing_file, 'w') as writer:
                        writer.write(stop_message)

        # Summarize cross-val in the info file
        metainfo = mlexp_info_helper(
            title='malaria-logregs-cv',
            data_setup=data_id,
            model_setup=model_id,
            exp_function=fit,
        )
        metainfo.update((
            ('num_cv_folds', num_cv_folds),
            ('cv_seed', cv_seed),
        ))
        metainfo.update(logreg_params.items())
        with open(info_file, 'w') as writer:
            json.dump(metainfo, writer, indent=2, sort_keys=False)


#######################################
# Generate command lines for many logreg experiments
#######################################

def sha_for_cl(cl):
    params = cl.partition('fit-logregs ')[2].partition(' &>')[0]
    return hashlib.sha256(params).hexdigest()


def cl():
    """Generate command lines for different experiments."""

    all_commands = []

    def gen_cl(num_foldss=(10,),
               cv_seedsss=((0, 1, 2, 3, 4),),  # FIXME: do not use a special case, it breaks parameters shas
               penalties=('l1', 'l2'),
               Cs=(0.001, 0.01, 0.1, 0.5, 1, 5, 10, 100, 1000),
               class_autos=(True, False),
               tols=(1E-4,),
               duals=(False,),
               fingerprint_fold_sizes=(0, 255, 511, 1023, 2047, 4095, 8191, 16383, 32767, 65537, 131073),
               fingerprint_folder_seeds=(0, 1)):
        """
        Generates command lines for the logistic regression tasks.
        The default params are these used for the "not so crazy" experiment (stopped on Sunday morning)
        """

        for i, (num_folds, cv_seeds, penalty, C, class_auto, tol, dual, ff_size, ff_seed) in \
                enumerate(product(num_foldss, cv_seedsss, penalties, Cs, class_autos, tols, duals,
                                  fingerprint_fold_sizes, fingerprint_folder_seeds)):
            params = (
                '--num-cv-folds %d' % num_folds,
                '--cv-seeds %s' % ' '.join(map(str, cv_seeds)),
                '--logreg-penalty %s' % penalty,
                '--logreg-C %g' % C,
                '--logreg-class-weight-auto' if class_auto else None,
                '--logreg-tol %g' % tol,
                '--logreg-dual' if dual else None,
                '--fingerprint-fold-size %d' % ff_size,
                '--fingerprint-folder-seed %d' % ff_seed
            )
            params = ' '.join(filter(lambda x: x is not None, params))
            cl = 'PYTHONPATH=.:$PYTHONPATH /usr/bin/time -v python2 -u malaria/logregs_fit.py fit-logregs '
            cl += params
            cl += ' &>~/logreg-%s.log' % hashlib.sha256(params).hexdigest()
            all_commands.append(cl)

    #########################
    #
    # There are three basic tasks we want to do:
    #
    #   1- Logreg param selection:
    #       For this we would only need 1 cv seeds and 1 fpt seed, as the results are clearly consistent
    #       accross folds. Probably no need to do together with fp exploration (can reduce the number
    #       of fold sizes greatly). We would like to explore also at least tolerances and duals.
    #       We might want to use less number of folds (e.g. just 5 --> from 90% to 75% train size).
    #
    #   2- Fingerprint strategy exploration:
    #       We would just stick with what is done in the previous. An alternative that would be faster
    #       is to do parameter selection just with unfolded fingerprints (as anyway that is what we
    #       plan to do) and then apply the best logreg parameters to this phase. We could miss
    #       interactions but, oh well, that is life. This second faster way is what Flo planned.
    #
    #   3- Final model explotation and interpretation:
    #       For this we would need (a) unfolded feature vectors only (b) maybe more cvseeds (c) maybe boosting.
    #       This phase only depends on phase 1 and it is what we need to generate the predictions and interpretations.
    #       We could stick with Flo's insights and just use a few big Cs, l1 and class weights.
    #
    #########################
    #
    # From sklearn implementation:
    #    dual : boolean
    #           Dual or primal formulation. Dual formulation is only implemented for l2 penalty.
    #           Prefer dual=False when n_samples > n_features.
    #  So we should use dual when not using folding and regularizing via l2.
    #
    #########################

    # Task 1: logreg parameter selection
    gen_cl(num_foldss=(10,),
           cv_seedsss=((0,),),
           penalties=('l1', 'l2'),
           Cs=(0.001, 0.01, 0.1, 0.5, 1, 5, 10, 100, 1000),
           class_autos=(True, False),
           tols=(1E-2, 1E-4),  # 1E-6 Takes really long
           duals=(False,),
           fingerprint_fold_sizes=(0, 1023, 2047, 4095, 8191, 16383,),
           fingerprint_folder_seeds=(0,))

    # Task 2: fingerprint strategy exploration
    gen_cl(num_foldss=(10,),
           cv_seedsss=((0,),),
           penalties=('l1',),
           Cs=(1,),
           class_autos=(True,),
           tols=(1E-4,),
           duals=(False,),
           fingerprint_fold_sizes=(0, 255, 511, 1023, 2047, 4095, 8191, 16383, 32767, 65537, 131073),
           fingerprint_folder_seeds=(0, 1, 2, 3))

    # Task 3: deployment classifiers computation - only one long job...
    gen_cl(num_foldss=(3, 5, 7, 10,),
           cv_seedsss=((0,), (1,), (2,), (3,), (4,)),
           penalties=('l1', 'l2',),
           Cs=(1, 5,),
           class_autos=(True,),
           tols=(1E-4,),
           duals=(False,),
           fingerprint_fold_sizes=(0,),
           fingerprint_folder_seeds=(0,))

    # ---- Save the cls to files

    all_commands = list(set(all_commands))    # Remove duplicates

    # Proper balance of workloads between machines
    destinies = (
        ('galileo', [], 0.30196078),  # machine, cls, probability to be picked
        ('zeus', [], 0.25882353),
        ('str22', [], 0.18431373),
        ('strz', [], 0.25490196),
    )

    p_choice = [p for _, _, p in destinies]
    rng = np.random.RandomState(2147483647)
    for cl in all_commands:
        _, your_destiny, _ = destinies[rng.choice(len(destinies), p=p_choice)]
        your_destiny.append(cl)

    # Save the selections
    for name, cls, _ in destinies:
        with open(op.join(op.dirname(__file__), '..', name), 'w') as writer:
            writer.write('\n'.join(cls))

    # Summary
    total_cls = sum(len(cl) for _, cl, _ in destinies)
    print('Total number of commands: %d' % total_cls)
    for name, cls, p in destinies:
        print('\t%s\t%d %g %g' % (name.ljust(30), len(cls), p, len(cls) / (total_cls + 1.)))


if __name__ == '__main__':
    parser = argh.ArghParser()
    parser.add_commands([cl, fit])
    parser.dispatch()

# TODO: bring back from oscail configurable to model (urgent!) and eval (unnecessary, but good for consistency)
# TODO: use SGDClassifier to be able to use elastic net
# TODO: vowpal wabbit back to scene - it was the original idea for the tutorial!
