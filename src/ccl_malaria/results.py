# coding=utf-8
import json
import os.path as op
from glob import glob
from itertools import izip

import h5py
import joblib
import numpy as np
import pandas as pd
from pandas import DataFrame, Series
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LinearRegression
from sklearn.metrics import roc_auc_score

from ccl_malaria import info, MALARIA_EXPS_ROOT
from ccl_malaria.molscatalog import MalariaCatalog
from minioscail.common.eval import kendalltau, rank_sort
from minioscail.common.results import OOBResult, ResultInDisk


###################
# Helper functions
###################

def predict_malaria_unlabelled(model, h5, rf_amb=None, rf_scr=None, rf_unl=None, chunksize=0):
    """Use the model to cast predictions for the datasets, storing them where appropriate in the h5 file and
    allowing predicition on streams of the screening dataset.
    """

    # Ambiguous examples
    if rf_amb is not None:
        info('Scoring amb...')
        h5['amb'] = model.predict_proba(rf_amb.X()).astype(np.float32)
    # Unlabelled (competition) examples
    if rf_unl is not None:
        info('Scoring unl...')
        h5['unl'] = model.predict_proba(rf_unl.X()).astype(np.float32)
    # Screening examples
    if rf_scr is not None:
        info('Scoring scr...')
        if chunksize <= 0:
            h5['scr'] = model.predict_proba(rf_scr.X()).astype(np.int32)
        else:
            scr = h5.create_dataset('scr', shape=(rf_scr.ne_stream(), 2), dtype=np.float32)
            for i, x in enumerate(rf_scr.X_stream(chunksize=chunksize)):
                base = i * chunksize
                info('\t num_scr_examples: %d' % base)
                scr[base:base + chunksize] = model.predict_proba(x)


def save_molids(data_dir, name, molids, overwrite=False):
    """Save molids in plain text in the data directory."""
    molids_file = op.join(data_dir, '%s.ids' % name)
    if not op.isfile(molids_file) or overwrite:
        info('Saving molids...')
        with open(molids_file, 'w') as writer:
            for molid in molids:
                writer.write(molid)
                writer.write('\n')


###################
# Results specials
###################

def malaria_result_factory(root_dir):
    # OOB experiments
    if 'oob' == op.basename(root_dir):
        return OOBResult(root_dir)
    if op.basename(root_dir).startswith('cv__'):
        return MalariaCVResult(root_dir)
    return None


class MalariaCVResult(ResultInDisk):
    def __init__(self, root_dir, ids_cache=None):
        """A result for a model evaluated using cross validation, practical for the malaria competition."""
        # TODO: refactor common parts to CVResult
        # TODO: no tests, no joy (need for bad_fold, unfinished and hinished results)
        super(MalariaCVResult, self).__init__(root_dir, ids_cache)

    def is_done(self):
        return super(MalariaCVResult, self).is_done() or self.num_present_folds() > 0
        # There was once upon a time a bug
        # No info.json was written
        # if a fold was not good...
        # Fix at filesystem level!

    def logreg_coefs(self, fold_num):
        with h5py.File(self.fold_h5(fold_num)) as h5:
            return h5['logreg_coef'][:]

    def num_folds(self):
        """Returns the number of expected folds (as int).
        This does not need to agree with the number of computed folds.
        """
        return int(op.basename(self.eval_dir).partition('num_folds=')[2])

    def num_present_folds(self):
        return len(self.present_folds())

    def num_examples(self):
        return len(self.ids('lab'))

    def cv_seed(self):
        """Returns the cross val seed (as int) for this experiment."""
        return int(op.basename(self.eval_dir).partition('cv_seed=')[2].partition('__')[0])

    @staticmethod
    def fold_from_fn(fn):
        """Returns a int corresponding to the fold number in the file name."""
        fn = op.basename(fn)
        return int(fn.split('__')[0][len('fold='):])

    def has_bad_fold(self):
        """Has the result computation stopped because a the model did not achieve a minimum performance in a vold?."""
        return op.isfile(op.join(self.eval_dir, 'STOP_BAD_FOLD'))

    def present_folds(self):
        """Returns a list of integers with the folds that have finished."""
        return sorted(map(self.fold_from_fn, glob(op.join(self.eval_dir, 'fold=*__info.json'))))

    def fold_h5(self, fold_num):
        if fold_num not in self.present_folds():
            return None
        return op.join(self.eval_dir, 'fold=%d__scores.h5' % fold_num)

    def fold_test_indices(self, fold_num):
        if fold_num not in self.present_folds():
            return None
        with h5py.File(self.fold_h5(fold_num)) as h5:
            return h5['test_indices'][:]

    def fold_scores(self, fold_num, dset='test'):
        if fold_num not in self.present_folds():
            return None
        with h5py.File(self.fold_h5(fold_num)) as h5:
            if dset not in h5:
                raise Exception('Dataset %s have not been scored yet in fold %d' % (dset, fold_num))
            return h5[dset][:]

    def fold_info_file(self, fold_num):
        if fold_num not in self.present_folds():
            return None
        return op.join(self.eval_dir, 'fold=%d__info.json' % fold_num)

    def fold_info(self, fold_num):
        if fold_num not in self.present_folds():
            return None
        with open(self.fold_info_file(fold_num)) as reader:
            return json.load(reader)

    def fold_auc(self, fold_num):
        info = self.fold_info(fold_num)
        return None if info is None else info['auc']

    def fold_enrichement5(self, fold_num):
        info = self.fold_info(fold_num)
        return None if info is None else info['enrichment5']

    def auc_mean(self):
        aucs = [self.fold_auc(fold) for fold in self.present_folds()]
        return np.mean(aucs)

    def auc_std(self):
        aucs = [self.fold_auc(fold) for fold in self.present_folds()]
        return np.std(aucs)

    def enrichement5_mean(self):
        riches = [self.fold_enrichement5(fold) for fold in self.present_folds()]
        return np.mean(riches)

    def enrichement5_std(self):
        riches = [self.fold_enrichement5(fold) for fold in self.present_folds()]
        return np.std(riches)

    def scores(self, dset='test'):
        scores = None
        for fold in self.present_folds():
            fold_scores = self.fold_scores(fold)
            if scores is None:
                scores = np.empty((self.num_examples(), fold_scores.shape[1]))
                scores.fill(np.nan)
            fold_indices = self.fold_test_indices(fold)
            scores[fold_indices] = fold_scores
        return scores

    def fold_model(self, fold_num, with_bug=False):  # Logreg-only, but abstract in superclass
        """Returns the classifier for a fold."""
        with h5py.File(self.fold_h5(fold_num)) as h5:
            model = self.model_setup(with_bug=with_bug)
            model.coef_ = h5['logreg_coef'][:]
            model.intercept_ = h5['logreg_intercept'][:]
        return model

    def fingerprint_folder(self):
        """Returns the object used to fold the feature vectors."""
        folder_file = op.join(self.data_dir, 'folder.pkl')
        return joblib.load(folder_file) if op.isfile(folder_file) else None

    def folded2unfolded(self):
        """Returns an array, the ith element is the fold (aka bucket) for the ith unfolded feature."""
        h5 = op.join(self.data_dir, 'folded2unfolded.h5')
        with h5py.File(h5) as h5:
            return h5['f2u'][:]

    def remove(self):
        import shutil
        shutil.rmtree(self.eval_dir)


########
# DEPLOYMENT AND SUBMISSION TOOLS
########

def compute_confirmatory(deployers,
                         molids_provider,
                         outfile,
                         y_provider=None,
                         select_top=500,
                         mc=None):
    """Scores and rankings on plain-average for the labelled / ambiguous dataset."""

    # Labelled
    Xlab, f_names = deployers(dset='lab')
    info('AUC after plain averaging (bagging like): %.3f' % roc_auc_score(y_provider(),
                                                                          np.nanmean(Xlab, axis=1)))
    # Ambiguous
    Xamb, _ = deployers(dset='amb')
    # All together
    X = np.vstack((Xlab, Xamb))

    # Scores are just plain averages
    scores = np.nanmean(X, axis=1)

    # Get the molids, smiles, labels, pec50
    lab_molids = molids_provider(dset='lab')
    amb_molids = molids_provider(dset='amb')
    molids = np.hstack((lab_molids, amb_molids))

    if mc is None:
        mc = MalariaCatalog()
    labels = mc.molids2labels(molids)
    pec50s = mc.molids2pec50s(molids)
    smiles = mc.molids2smiless(molids)

    # Rankings
    ranks, (sscores, smolids, slabels, spec50s, ssmiles) = \
        rank_sort(scores, (scores, molids, labels, pec50s, smiles), reverse=True, select_top=select_top)

    # N.B.
    # if analyzing ranking variability, use instead
    # scores2rankings()

    # Save for submission
    with open(outfile, 'w') as writer:
        for molid, smiles, score in izip(smolids, ssmiles, sscores):
            writer.write('%s,%s,%.6f\n' % (molid, smiles, score))

    # Create and save a pandas series to allow further stacking
    s = Series(data=scores, index=molids)
    s.to_pickle(op.join(op.splitext(outfile)[0] + '.pkl'))

    # TODO Flo: create a molecules plot, interpret it
    return molids, scores


def compute_heldout(dset,
                    deployers,
                    molids_provider,
                    outfile,
                    y_provider=None,
                    stacker=None,
                    select_top=None,
                    mc=None):
    """Predictions for the held-out sets."""
    X, _ = deployers(dset=dset)

    # Stacking or averaging?
    if stacker is not None:
        Xlab, _ = deployers(dset='lab')
        y = y_provider()
        stacker.fit(Xlab, y)  # Careful: Xlab columns can be extremelly collinear...
        if True:
            scores = stacker.predict(X)
        else:
            scores = stacker.predict_proba(X)[:, 1]
    else:
        scores = np.nanmean(X, axis=1)

    # Get the molids, smiles
    if mc is None:
        mc = MalariaCatalog()
    molids = molids_provider(dset=dset)
    smiles = mc.molids2smiless(molids)

    # Rankings
    ranks, (sscores, smolids, ssmiles) = \
        rank_sort(scores, (scores, molids, smiles), reverse=True, select_top=select_top)

    # Save for submission
    with open(outfile, 'w') as writer:
        for molid, smiles, score in izip(smolids, ssmiles, sscores):
            writer.write('%s,%s,%.6f\n' % (molid, smiles, score))

    # Create and save a pandas series to allow further stacking
    s = Series(data=scores, index=molids)
    s.to_pickle(op.join(op.splitext(outfile)[0] + '.pkl'))

    return molids, scores


def compute_submissions(prefix,
                        dest_dir,
                        deployers,
                        molids_provider,
                        y_provider,
                        do_confirmatory=True,
                        do_heldout=True,
                        do_screening=True):
    info('Computing submissions for %s' % prefix)

    mc = MalariaCatalog()  # For performance, maybe this should be singleton...

    if do_confirmatory:
        compute_confirmatory(deployers,
                             molids_provider,
                             outfile=op.join(dest_dir, '%s_hitSelection.txt' % prefix),
                             y_provider=y_provider,
                             select_top=500)

    def do_predict(dset, select_top=None):

        info('Computing predictions for %s: %s' % (prefix, dset))

        _, scores_averaged = compute_heldout(dset,
                                             deployers,
                                             molids_provider,
                                             op.join(dest_dir, '%s_%s-averaged.txt' % (prefix, dset)),
                                             y_provider=y_provider,
                                             mc=mc,
                                             select_top=select_top)

        _, scores_linr = compute_heldout(dset,
                                         deployers,
                                         molids_provider,
                                         op.join(dest_dir, '%s_%s-stacker=linr.txt' % (prefix, dset)),
                                         y_provider=y_provider,
                                         stacker=LinearRegression(),
                                         mc=mc,
                                         select_top=select_top)

        info('Computing kendall-tau (go take a nap if there are a lot of examples...)')
        info('%s:%s - Kendall-tau avg vs linr: %.2f' % (prefix, dset, kendalltau(scores_linr, scores_averaged)))

    if do_heldout:
        do_predict('unl')

    if do_screening:
        do_predict('scr', select_top=1000)


#########
# Computation of averaged and stacked final submissions
#########

# noinspection PyTypeChecker
def final_merged_submissions(calibrate=False,
                             select_top_scr=None,
                             with_bug=False,
                             dest_dir=MALARIA_EXPS_ROOT):
    """Very ad-hoc merge of submissions obtained with trees and logistic regressors."""

    #####
    # 0 Preparations
    #####

    # Avoid circular imports
    from ccl_malaria.logregs_fit import MALARIA_LOGREGS_EXPERIMENT_ROOT
    from ccl_malaria.trees_fit import MALARIA_TREES_EXPERIMENT_ROOT

    mc = MalariaCatalog()

    def save_submission(sub, outfile, select_top=500):
        # Get the smiles
        smiles = mc.molids2smiless(sub.index)

        # Rankings
        ranks, (sscores, smolids, ssmiles) = \
            rank_sort(sub.values, (sub.values,
                                   sub.index.values,
                                   smiles), reverse=True, select_top=select_top)
        # Save for submission
        with open(outfile, 'w') as writer:
            for molid, smiles, score in izip(smolids, ssmiles, sscores):
                writer.write('%s,%s,%.6f\n' % (molid, smiles, score))

    #####
    # 1 Robust merge using pandas
    #####
    def read_average_merge(root, prefix):
        hit = pd.read_pickle(op.join(root, '%s_hitSelection.pkl' % prefix))
        labels = mc.molids2labels(hit.index, as01=True)
        lab = hit[~np.isnan(labels)]
        amb = hit[np.isnan(labels)]
        unl = pd.read_pickle(op.join(root, '%s_unl-averaged.pkl' % prefix))
        scr = pd.read_pickle(op.join(root, '%s_scr-averaged.pkl' % prefix))
        return lab, amb, unl, scr

    tlab, tamb, tunl, tscr = read_average_merge(MALARIA_TREES_EXPERIMENT_ROOT, 'trees')
    llab, lamb, lunl, lscr = read_average_merge(MALARIA_LOGREGS_EXPERIMENT_ROOT,
                                                malaria_logreg_file_prefix(with_bug=with_bug))

    lab = DataFrame({'trees': tlab, 'logregs': llab})
    lab['labels'] = mc.molids2labels(lab.index, as01=True)
    assert np.sum(np.isnan(lab['labels'])) == 0
    amb = DataFrame({'trees': tamb, 'logregs': lamb})
    unl = DataFrame({'trees': tunl, 'logregs': lunl})
    scr = DataFrame({'trees': tscr, 'logregs': lscr})

    # ATM we take it easy and just drop any NA
    lab.dropna(inplace=True)
    amb.dropna(inplace=True)
    unl.dropna(inplace=True)
    scr.dropna(inplace=True)

    #####
    # 2 Calibration on labelling - careful with overfitting for hitList, do it in cross-val fashion
    #####
    def calibrate_row(row):
        calibrator = IsotonicRegression(y_min=0, y_max=1)
        x = lab[~np.isnan(lab[row])][row].values
        y = lab[~np.isnan(lab[row])]['labels'].values
        calibrator.fit(x, y)
        lab[row] = calibrator.predict(lab[row].values)
        amb[row] = calibrator.predict(amb[row].values)
        unl[row] = calibrator.predict(unl[row].values)
        scr[row] = calibrator.predict(scr[row].values)

    if calibrate:
        calibrate_row('trees')
        calibrate_row('logregs')

    #####
    # 3 Average for the submission in lab-amb
    #####
    submission_lab = (lab.trees + lab.logregs) / 2
    submission_amb = (amb.trees + amb.logregs) / 2
    submission_hts = pd.concat((submission_lab, submission_amb))

    submission_options = '%s-%s' % (
        'calibrated' if calibrate else 'nonCalibrated',
        'lastFold' if with_bug else 'averageFolds')

    outfile = op.join(dest_dir, 'final-merged-%s-hitSelection.csv' % submission_options)
    save_submission(submission_hts, outfile)

    #####
    # 4 Average predictions for unlabelled
    #####
    submission_unl_avg = (unl.trees + unl.logregs) / 2
    outfile = op.join(dest_dir, 'final-%s-avg-unl.csv' % submission_options)
    save_submission(submission_unl_avg, outfile, select_top=None)

    submission_scr_avg = (scr.trees + scr.logregs) / 2
    outfile = op.join(dest_dir, 'final-%s-avg-scr.csv' % submission_options)
    save_submission(submission_scr_avg, outfile, select_top=select_top_scr)

    #####
    # 5 Stacked (linear regression) for unlabelled
    #####
    stacker = LinearRegression()
    stacker.fit(lab[['trees', 'logregs']], lab['labels'])

    # noinspection PyArgumentList
    submission_unl_st = Series(data=stacker.predict(unl[['trees', 'logregs']]), index=unl.index)
    outfile = op.join(dest_dir, 'final-%s-stacker=linr-unl.csv' % submission_options)
    save_submission(submission_unl_st, outfile, select_top=None)

    # noinspection PyArgumentList
    submission_scr_st = Series(data=stacker.predict(scr[['trees', 'logregs']]), index=scr.index)
    outfile = op.join(dest_dir, 'final-%s-stacker=linr-scr.csv' % submission_options)
    save_submission(submission_scr_st, outfile, select_top=select_top_scr)

    # TODO: document that:
    #   - Averaging averages averages
    #   - Stacking stacks stacks
    # read the code to understand what I mean


if __name__ == '__main__':
    import argh
    parser = argh.ArghParser()
    parser.add_commands([final_merged_submissions])
    parser.dispatch()
