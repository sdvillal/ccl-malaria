# coding=utf-8
"""Analysis/plots and deployment of trees OOB results."""
import os.path as op
from copy import copy

import h5py
import numpy as np
from pandas import DataFrame

from ccl_malaria import info
from ccl_malaria.features import MalariaRDKFsExampleSet
from ccl_malaria.molscatalog import MalariaCatalog
from ccl_malaria.results import malaria_result_factory, compute_submissions
from ccl_malaria.trees_fit import MALARIA_TREES_EXPERIMENT_ROOT
from minioscail.common.config import parse_id_string
from minioscail.common.results import ResultInDisk

MALARIA_TREES_DEPLOYMENT_H5 = op.join(MALARIA_TREES_EXPERIMENT_ROOT, 'trees-deployers.h5')


#################
# Results presented in a convenient way...
#################


def trees_results_to_pandas(common_molids_cache=False):
    """Collects all the results in disk and place them in record-format in a pandas dataframe.
    Allows convenient reporting, grouping and filtering of results.
    """
    results = ResultInDisk.collect_results_under_dir(MALARIA_TREES_EXPERIMENT_ROOT,
                                                     factory=malaria_result_factory)

    # --- molids cache
    molids_cache = None
    if common_molids_cache:
        a_result = results[0]
        # Labelled molids
        lab_molids = a_result.ids('lab')
        amb_molids = a_result.ids('amb')  # To prioritize confirmatory tests on labelled data
        # Unlabelled molids
        unl_molids = a_result.ids('unl')
        scr_molids = a_result.ids('scr')
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
            result.ids_cache = molids_cache  # dodgy, rework with a copying constructor
        rdict = copy(result.info())
        rdict['result'] = result
        # Some more ad-hoc keys for the model
        model_params = parse_id_string(result.model_setup_id())[1]
        rdict['model_num_trees'] = int(model_params['num_trees'])
        rdict['model_seed'] = int(model_params['seed'])
        rdict['model_type'] = 'ExtraTrees' if model_params['etc'] else 'RandomForest'
        # Add this result to the data frame
        results_dict_of_dicts[result.root_key()] = rdict

    return DataFrame(results_dict_of_dicts).T


#################
# "DEPLOYMENT" (Get scores from selected models and merge them in (not so) clever ways).
#################


def fix_streaming_scoring_bug_results(scores):
    """
    When we computed the scores for the screening dataset, we did it in a streaming fashion.

    Unfortunately, at the moment this fails to coordinate well with the molecules because there
    was a bug in streaming: molecules with some missing value were passed.

    Trees had no problem on classifying those: sklearn implementation makes no checks and all
    predicates result just in False.

    To fix we just need to check what are the non-legit scores, which is simple (albeit time consuming).
    This function does this.
    """
    return scores[MalariaRDKFsExampleSet(dset='scr', remove_ambiguous=False).rows_to_keep()]


def trees_deploy(dest_file=MALARIA_TREES_DEPLOYMENT_H5):
    """Generates predictions for unlabelled datasets."""

    df = trees_results_to_pandas()

    h5 = h5py.File(dest_file, 'w')

    # Choose a few good results (maybe apply diversity filters or ensemble selection or...)
    # noinspection PyUnresolvedReferences
    deployers = df[(df['model_num_trees'] == 6000)]

    info('Deploying %d tree ensembles' % len(deployers))

    for i, res in enumerate(deployers.result):
        f_name = '%s__%s' % (res.model_setup_id(), res.eval_setup_id())
        # What about the data setup?
        # Here it works but in general not
        # Save it all (a new dataset with all the coords and the result path)
        info(f_name)

        # Lab
        if '%s/lab' % f_name not in h5:
            h5['%s/lab' % f_name] = res.scores('lab')[:, 1].astype(np.float32)

        # Amb
        if '%s/amb' % f_name not in h5:
            h5['%s/amb' % f_name] = res.scores('amb')[:, 1].astype(np.float32)

        # Unl
        if '%s/unl' % f_name not in h5:
            h5['%s/unl' % f_name] = res.scores('unl')[:, 1].astype(np.float32)

        # Scr
        if '%s/scr' % f_name not in h5:
            h5['%s/scr' % f_name] = fix_streaming_scoring_bug_results(res.scores('scr')[:, 1].astype(np.float32))
            assert h5['%s/scr' % f_name].hape[0] == 5488144, 'Streaming rdkf bug striking back...'

    h5.close()


def trees_deployers(dset='lab', rewrite=False):
    """Returns a tuple (scores, f_names).
    scores is a numpy array, each column are the scores for a
    f_names is the id of the corresponding models.
    """
    if not op.isfile(MALARIA_TREES_DEPLOYMENT_H5) or rewrite:
        trees_deploy()
    with h5py.File(MALARIA_TREES_DEPLOYMENT_H5) as h5:
        f_names = sorted(h5.keys())
        return np.array([h5['%s/%s' % (feature, dset)][:] for feature in f_names]).T, f_names


def trees_molids(dset='lab'):
    # No need to do this on a per-result basis because
    # atm we are warranted that they are the same accross all evaluations.
    a_result = ResultInDisk.collect_results_under_dir(MALARIA_TREES_EXPERIMENT_ROOT)[0]
    return a_result.ids(dset=dset)


def trees_y():
    return MalariaCatalog().molids2labels(trees_molids(dset='lab'), as01=True)


def do_trees_submissions(do_confirmatory=True,
                         do_heldout=True,
                         do_screening=True):
    compute_submissions(prefix='trees',
                        dest_dir=MALARIA_TREES_EXPERIMENT_ROOT,
                        deployers=trees_deployers,
                        molids_provider=trees_molids,
                        y_provider=trees_y,
                        do_confirmatory=do_confirmatory,
                        do_heldout=do_heldout,
                        do_screening=do_screening)
    info('Submissions computed!')


if __name__ == '__main__':
    import argh

    parser = argh.ArghParser()
    parser.add_commands([do_trees_submissions])
    parser.dispatch()
