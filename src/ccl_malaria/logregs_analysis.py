# coding=utf-8
"""Generating plots and stats from the Morgan fingerprint results."""
from copy import copy
import os.path as op
import h5py
from pandas import DataFrame
from ccl_malaria import info
from ccl_malaria.logregs_fit import MALARIA_LOGREGS_EXPERIMENT_ROOT, malaria_logreg_fpt_providers
from ccl_malaria.molscatalog import MalariaCatalog
from ccl_malaria.results import malaria_result_factory, compute_submissions
from minioscail.common.results import ResultInDisk
import numpy as np


MALARIA_LOGREGS_DEPLOYMENT_H5 = op.join(MALARIA_LOGREGS_EXPERIMENT_ROOT, 'logreg-deployers.h5')


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


def logreg_deploy(dest_file=MALARIA_LOGREGS_DEPLOYMENT_H5):
    """Generates predictions for unlabelled datasets."""

    df = logreg_results_to_pandas()

    h5 = h5py.File(dest_file, 'w')

    # Choose a few good results (maybe apply diversity filters or ensemble selection or...)
    deployment_cond_1 = (df.cv_seed < 5) & \
                        (df.num_present_folds == df.num_cv_folds) & \
                        (df.penalty == 'l1') & \
                        (df.C == 1) & \
                        (df.class_weight == 'auto') & \
                        (df.tol == 1E-4) & \
                        (df.folder_size < 1) & \
                        (df.folder_seed == -1) & \
                        (df.auc_mean > 0.92)

    deployment_cond_2 = (df.num_present_folds == df.num_cv_folds) & \
                        (df.penalty == 'l2') & \
                        (df.C == 5) & \
                        (df.class_weight == 'auto') & \
                        (df.tol == 1E-4) & \
                        (df.folder_size < 1) & \
                        (df.folder_seed == -1) & \
                        (df.auc_mean > 0.93)

    deployers = df[deployment_cond_1 | deployment_cond_2]

    info('Deploying %d logistic regressors' % len(deployers))

    # We will have 40 "features", one for each deployer
    # For lab it will just be the test scores
    # For amb, unl and scr it will be the average of the scores for each cv fold

    rf_lab, rf_amb, rf_unl, rf_scr = malaria_logreg_fpt_providers(None)

    for i, res in enumerate(deployers.result):
        f_name = '%s__%s' % (res.model_setup_id(), res.eval_setup_id())  # What about the data setup?
                                                                         # Here it works but in general not
                                                                         # Save it all...
                                                                         # (a new dataset with all the coords
                                                                         # and the result path)
        print f_name

        # Lab
        if '%s/lab' % f_name not in h5:
            h5['%s/lab' % f_name] = res.scores()[:, 1].astype(np.float32)

        # Amb
        models = [res.fold_model(fold) for fold in res.present_folds()]
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

    h5.close()


def logreg_deployers(dset='lab'):
    """Returns a tuple (scores, f_names).
    scores is a numpy array, each column are the scores for a
    f_names is the id of the corresponding models.
    """
    if not op.isfile(MALARIA_LOGREGS_DEPLOYMENT_H5):
        logreg_deploy()
    with h5py.File(MALARIA_LOGREGS_DEPLOYMENT_H5) as h5:
        f_names = sorted(h5.keys())
        return np.array([h5['%s/%s' % (feature, dset)][:] for feature in f_names]).T, f_names


def logreg_molids(dset='lab'):
    # No need to do this on a per-result basis because
    # atm we are warranted that they are the same accross all evaluations.
    rf_lab, rf_amb, rf_unl, rf_scr = malaria_logreg_fpt_providers(None)
    rf = rf_lab if dset == 'lab' else \
         rf_amb if dset == 'amb' else \
         rf_unl if dset == 'unl' else \
         rf_scr if dset == 'scr' else \
         None
    if rf is None:
        raise Exception('Unknown dataset %s' % dset)
    return rf.ids()


def logreg_y():
    return MalariaCatalog().molids2labels(logreg_molids(dset='lab'), as01=True)


def do_logreg_submissions(do_confirmatory=True,
                          do_heldout=True,
                          do_screening=True):
    compute_submissions(prefix='logreg',
                        dest_dir=MALARIA_LOGREGS_EXPERIMENT_ROOT,
                        deployers=logreg_deployers,
                        molids_provider=logreg_molids,
                        y_provider=logreg_y,
                        do_confirmatory=do_confirmatory,
                        do_heldout=do_heldout,
                        do_screening=do_screening)


#################
# Feature selection via logreg odds
#################


#################
# Logistic regression parameter selection
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
        print class_weight, penalty, C, fs, len(group), '%.2f' % group.auc_mean.mean()


#################
# Effect of folding on learning and model interpretation
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
        print size, len(group), group.auc_mean.mean()


if __name__ == '__main__':
    import argh
    parser = argh.ArghParser()
    parser.add_commands([do_logreg_submissions])
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
