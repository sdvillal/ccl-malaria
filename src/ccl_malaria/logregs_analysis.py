# coding=utf-8
"""Generating plots and stats from the Morgan fingerprint results."""
from copy import copy
import os.path as op
import h5py
from functools import partial
from pandas import DataFrame

from ccl_malaria import info
from ccl_malaria.logregs_fit import MALARIA_LOGREGS_EXPERIMENT_ROOT, malaria_logreg_fpt_providers
from ccl_malaria.molscatalog import MalariaCatalog
from ccl_malaria.results import malaria_result_factory, compute_submissions
from minioscail.common.results import ResultInDisk
import numpy as np
import pandas as pd


#################
# Results presented in a convenient way...
#################

def logreg_results_to_pandas(common_molids_cache=False):
    """Collects all the results in disk and place them in record-format in a pandas dataframe.
    Allows convenient reporting, grouping and filtering of results.
    """
    results = ResultInDisk.collect_results_under_dir(MALARIA_LOGREGS_EXPERIMENT_ROOT,
                                                     factory=malaria_result_factory)

    # --- molids cache
    molids_cache = None
    if common_molids_cache:
        rf_lab, rf_amb, rf_unl, rf_scr = malaria_logreg_fpt_providers(None)
        # Labelled molids
        lab_molids = rf_lab.ids()
        amb_molids = rf_amb.ids()  # To prioritize confirmatory tests on labelled data
        # Unlabelled molids
        unl_molids = rf_unl.ids()
        scr_molids = rf_scr.ids()
        # Let's avoid the need to reread them...
        molids_cache = {
            'lab': lab_molids,
            'amb': amb_molids,
            'unl': unl_molids,
            'scr': scr_molids
        }

    results_dict_of_dicts = {}
    for result in results:
        if common_molids_cache:
            result.ids_cache = molids_cache    # dodgy, rework with a copying constructor
        rdict = copy(result.info())
        rdict['result'] = result
        rdict['class_weight'] = 'uniform' if rdict['class_weight'] is None else rdict['class_weight']
        # Some more ad-hoc keys for the model
        rdict['num_present_folds'] = result.num_present_folds()
        rdict['auc_mean'] = result.auc_mean()
        rdict['enrichement5_mean'] = result.enrichement5_mean()
        # Some more ad-hoc keys for the fingerprint folder
        folder = result.fingerprint_folder()
        rdict['folder_seed'] = int(folder.seed) if folder is not None else -1
        rdict['folder_size'] = int(folder.fold_size) if folder is not None else 0
        # Add this result to the data frame
        results_dict_of_dicts[result.root_key()] = rdict

    return DataFrame(results_dict_of_dicts).T


def clean_results_pre_infojson_bug_fix():
    results = ResultInDisk.collect_results_under_dir(MALARIA_LOGREGS_EXPERIMENT_ROOT,
                                                     factory=malaria_result_factory)
    bad_results = [res for res in results if not op.isfile(op.join(res.eval_dir, 'info.json'))]
    for res in bad_results:
        info('Bye %s' % res.eval_dir)


#################
# "DEPLOYMENT" (Get scores from selected models and merge them in (not so) clever ways).
#################

# noinspection PyUnresolvedReferences
def logreg_experiments_to_deploy():
    """
    Returns a pandas dataframe with a selection of the logistic regression
    experiments to be included on deployment.
    """

    # A dataframe with all the results (parameters, performance, stored models...)
    df = logreg_results_to_pandas()

    # Choose a few good results (maybe apply diversity filters or ensemble selection or...)
    # These decisions where informed by some plotting (see tutorial)
    # (e.g., keep number maneageble, keep most regularized amongst these with higher performance...)
    deployment_cond_1 = df.query('cv_seed < 5 and '
                                 'num_present_folds == num_cv_folds and '
                                 'penalty == "l1" and '
                                 'C == 1 and '
                                 'class_weight == "auto" and '
                                 'tol == 1E-4 and '
                                 'folder_size < 1 and '
                                 'folder_seed == -1 and '
                                 'auc_mean > 0.92')

    deployment_cond_2 = df.query('num_present_folds == num_cv_folds and '
                                 'penalty == "l2" and '
                                 'C == 5 and '
                                 'class_weight == "auto" and '
                                 'tol == 1E-4 and '
                                 'folder_size < 1 and '
                                 'folder_seed == -1 and '
                                 'auc_mean > 0.93')

    deployers = pd.concat([deployment_cond_1, deployment_cond_2]).reset_index()

    return deployers


def deployment_models(with_bug=False):
    df = logreg_experiments_to_deploy()
    return [res.fold_model(fold, with_bug=with_bug)
            for res in df.result for fold in res.present_folds()]
    # But we need also to specify what the input to the model looks like...
    # So this should infer feature extraction and preprocessing (all possible from the df)


def malaria_logreg_file_prefix(with_bug=False):
    return 'logreg-only-last-fold' if with_bug else 'logreg-folds-average'


def malaria_logreg_deployers_file(with_bug=False):
    return op.join(MALARIA_LOGREGS_EXPERIMENT_ROOT,
                   malaria_logreg_file_prefix(with_bug=with_bug) + '.h5')


def logreg_deploy(dest_file=None, with_bug=False):
    """
    Generates predictions for the competition unlabelled datasets, saving them in HDF5 files.

    Generates one prediction per molecule and cross-validation experiment:

      - For the labelled set, the prediction is given by the model of the
        run where the molecule was in the testing set.

      - For the other sets, the predictions are averages of all the models
        built during cross-validation. Note that at the time of submitting
        there was a bug that made these predictions be just the one of the
        last fold (see `with_bug` parameter).


    Parameters
    ----------
    dest_file : string or None, default None
      Path to the HDF5 to store the prediction values.
      There will be as many groups in there as deployed models.
      Each group will contain 4 datasets:
        - lab: predicitions on the labelled dataset
        - amb: predictions on the ambiguously labelled compounds
        - unl: predictions in the held-out competition set
        - scr: predictions in the screening dataset

    with_bug : bool, default False
      If True, predictions will be generated as for the competion
      (taking only the last fold of each experiment into account).
      If False, predictions will be generated as initially intended
      (averaging all the folds for each experiment).
      This bug does not affect the labelled scores.

    Returns
    -------
    The path to the HDF5 file where the scores have been saved.

    Side effects
    ------------
    The HDF5 file is created
    """

    if dest_file is None:
        dest_file = malaria_logreg_deployers_file(with_bug=with_bug)

    results = logreg_experiments_to_deploy().result

    info('Deploying %d logistic regression experiments (%d classifiers)' % (
        len(results),
        sum(len(result.present_folds()) for result in results)))

    # We will have a few "features" for each deployer
    # For lab it will just be the test scores
    # For amb, unl and scr it will be the average of the scores for each cv fold

    rf_lab, rf_amb, rf_unl, rf_scr = malaria_logreg_fpt_providers(None)

    with h5py.File(dest_file, 'w') as h5:

        for i, res in enumerate(results):

            # Deployer id
            f_name = '%s__%s' % (res.model_setup_id(), res.eval_setup_id())

            # Lab
            if '%s/lab' % f_name not in h5:
                h5['%s/lab' % f_name] = res.scores()[:, 1].astype(np.float32)

            # Get result models
            models = [res.fold_model(fold, with_bug=with_bug) for fold in res.present_folds()]

            # Amb
            if '%s/amb' % f_name not in h5:
                h5['%s/amb' % f_name] = np.nanmean([model.predict_proba(rf_amb.X())[:, 1]
                                                    for model in models], axis=0).astype(np.float32)
            # Unl
            if '%s/unl' % f_name not in h5:
                h5['%s/unl' % f_name] = np.nanmean([model.predict_proba(rf_unl.X())[:, 1]
                                                    for model in models], axis=0).astype(np.float32)
            # Scr
            if '%s/scr' % f_name not in h5:
                h5['%s/scr' % f_name] = np.nanmean([model.predict_proba(rf_scr.X())[:, 1]
                                                    for model in models], axis=0).astype(np.float32)

    return dest_file


def logreg_deployers(dset='lab', with_bug=False):
    """
    Returns a tuple (scores, f_names).
      - scores is a numpy array, each column are the scores for a fold
      - f_names is the id of the corresponding models.
    """
    deployers_file = malaria_logreg_deployers_file(with_bug=with_bug)
    if not op.isfile(deployers_file):
        logreg_deploy(with_bug=with_bug)
    with h5py.File(deployers_file) as h5:
        f_names = sorted(h5.keys())
        return np.array([h5['%s/%s' % (feature, dset)][:] for feature in f_names]).T, f_names


def logreg_molids(dset='lab'):
    # No need to do this on a per-result basis because
    # atm we are warranted that they are the same accross all evaluations.
    rf_lab, rf_amb, rf_unl, rf_scr = malaria_logreg_fpt_providers(None)
    rf = (rf_lab if dset == 'lab' else
          rf_amb if dset == 'amb' else
          rf_unl if dset == 'unl' else
          rf_scr if dset == 'scr' else
          None)
    if rf is None:
        raise Exception('Unknown dataset %s' % dset)
    return rf.ids()


def logreg_y():
    return MalariaCatalog().molids2labels(logreg_molids(dset='lab'), as01=True)


def submit(no_confirmatory=False,
           no_heldout=False,
           no_screening=False,
           with_bug=False,
           confirmatory_top=500,
           scr_top=1000):
    compute_submissions(prefix=malaria_logreg_file_prefix(with_bug=with_bug),
                        dest_dir=MALARIA_LOGREGS_EXPERIMENT_ROOT,
                        deployers=partial(logreg_deployers, with_bug=with_bug),
                        molids_provider=logreg_molids,
                        y_provider=logreg_y,
                        do_confirmatory=not no_confirmatory,
                        do_heldout=not no_heldout,
                        do_screening=not no_screening,
                        confirmatory_top=confirmatory_top,
                        scr_top=scr_top)


if __name__ == '__main__':

    import argh
    parser = argh.ArghParser()
    parser.add_commands([submit])
    parser.dispatch()


#
#######################
#
# # Labelled rankings - we will just average, maybe we should first calibrate
# rf_lab, rf_amb, rf_unl, rf_scr = malaria_logreg_fpt_providers(None)
# Xlab = logreg_deployers(dset='lab')
# Xamb = logreg_deployers(dset='amb')
# print Xlab.shape, Xamb.shape
# molids = np.hstack((rf_lab.ids(), rf_amb.ids()))
# mc = MalariaCatalog()
# mols = np.array(mc.molids2mols(molids))
# pec50s = mc.molids2pec50s(molids)
# scores = np.hstack((np.mean(Xlab, axis=1), np.mean(Xamb, axis=1)))
#
# _, (scores, molids, mols, pec50s) = rank_sort(scores, (scores, molids, mols, pec50s),
#                                               reverse=True, select_top=500)
#
# for molid, mol, score in izip(molids, mols, scores):
#     print molid, mc.molid2label(molid), '%.4f' % score
#     # for mol in mols:
#     #     AllChem.
# image = Draw.MolsToGridImage(mols, molsPerRow=5, legends=['%s (%s, %s)' %
#                                                           (molid, mc.molid2label(molid),
#                                                            'T' if not np.isnan(mc.molid2pec50(molid)) else '')
#                                                           for molid in molids])
# image.save('/home/santi/notepetes.png')
