# coding=utf-8
import json
import os.path as op
from itertools import product
from time import time

import h5py
from joblib import cpu_count
import joblib
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.metrics import roc_auc_score

from ccl_malaria import info
from ccl_malaria import MALARIA_EXPS_ROOT
from minioscail.common.eval import enrichment_at
from ccl_malaria.features import MalariaRDKFsExampleSet
from ccl_malaria.results import save_molids
from minioscail.common.config import mlexp_info_helper
from minioscail.common.misc import ensure_dir, giveupthefunc, fill_missing_scores


MALARIA_TREES_EXPERIMENT_ROOT = op.join(MALARIA_EXPS_ROOT, 'trees')


def fit(dest_dir=MALARIA_TREES_EXPERIMENT_ROOT,
        seeds=(0, 1, 2, 3, 4),
        num_treess=(10, 6000, 4000, 2000, 1000, 500, 20, 50, 100),
        save_trained_models=False,
        chunksize=200000,
        num_threads=None,
        force=False):

    # Generates OOB results

    info('Malaria trees experiment')

    # Guess the number of threads
    if num_threads is None:
        num_threads = cpu_count()
    info('Will use %d threads' % num_threads)

    # Example providers
    info('Reading data...')
    rf_lab = MalariaRDKFsExampleSet()
    X, y = rf_lab.Xy()
    rf_unl = MalariaRDKFsExampleSet(dset='unl', remove_ambiguous=False)
    rf_scr = MalariaRDKFsExampleSet(dset='scr', remove_ambiguous=False)
    rf_amb = MalariaRDKFsExampleSet(dset='amb')
    # A bit of logging
    info('Data description: %s' % rf_lab.configuration().id(nonids_too=True))
    info('ne=%d; nf=%d' % rf_lab.X().shape)

    # Experiment context: data
    data_id = rf_lab.configuration().id(nonids_too=True)  # TODO: bring hashing from oscail
    data_dir = op.join(dest_dir, data_id)
    ensure_dir(data_dir)

    # Save molids... a bit too ad-hoc...
    info('Saving molids...')

    save_molids(data_dir, 'lab', rf_lab.ids())
    save_molids(data_dir, 'unl', rf_unl.ids())
    save_molids(data_dir, 'scr', rf_scr.ids())
    save_molids(data_dir, 'amb', rf_amb.ids())

    # Main loop - TODO: robustify with try and continue
    for etc, seed, num_trees in product((True, False), seeds, num_treess):

        # Configure the model
        if etc:
            model = ExtraTreesClassifier(n_estimators=num_trees,
                                         n_jobs=num_threads,
                                         bootstrap=True,
                                         oob_score=True,
                                         random_state=seed)
        else:
            model = RandomForestClassifier(n_estimators=num_trees,
                                           n_jobs=num_threads,
                                           oob_score=True,
                                           random_state=seed)

        # Experiment context: model
        model_id = 'trees__etc=%r__num_trees=%d__seed=%d' % (etc, num_trees, seed)  # TODO: bring self-id from oscail
        model_dir = op.join(data_dir, model_id)
        ensure_dir(model_dir)
        info('Model: %s' % model_id)

        # Experiment context: eval
        eval_id = 'oob'
        eval_dir = op.join(model_dir, eval_id)
        ensure_dir(eval_dir)
        info('Eval: OOB (Out Of Bag)')

        # Already done?
        info_file = op.join(eval_dir, 'info.json')
        if op.isfile(info_file) and not force:
            info('\tAlready done, skipping...')
            continue

        # Save model config
        joblib.dump(model, op.join(model_dir, 'model_setup.pkl'), compress=3)

        # Train-full
        info('Training...')
        start = time()
        model.fit(X, y)
        train_time = time() - start  # This is also test-time, as per OOB=True

        # Save trained model? - yeah, lets do it under oob
        if save_trained_models:
            joblib.dump(model, op.join(eval_dir, 'model_trained.pkl'), compress=3)

        # OOB score, auc and enrichment
        oob_score = model.oob_score_
        oob_scores = model.oob_decision_function_
        oob_scores_not_missing = fill_missing_scores(oob_scores[:, 1])

        auc = roc_auc_score(y,  oob_scores_not_missing)
        enrichment5 = enrichment_at(y, oob_scores_not_missing, percentage=0.05)

        info('OOB AUC: %.2f' % auc)
        info('OOB Enrichment at 5%%: %.2f' % enrichment5)
        info('OOB Accuracy: %.2f' % oob_score)

        # Save scores and importances
        info('Saving results...')
        with h5py.File(op.join(eval_dir, 'oob_auc=%.2f__scores.h5' % auc), 'w') as h5:

            start = time()

            # Feature importances
            h5['f_names'] = rf_lab.fnames()
            h5['f_importances'] = model.feature_importances_

            # Labelled (development) examples
            info('Scoring lab...')
            h5['lab'] = oob_scores.astype(np.float32)

            info('Scoring amb...')
            h5['amb'] = model.predict_proba(rf_amb.X()).astype(np.float32)

            # Unlabelled (competition) examples
            info('Scoring unl...')
            h5['unl'] = model.predict_proba(rf_unl.X()).astype(np.float32)

            # Unlabelled (screening) examples
            info('Scoring scr...')
            if chunksize <= 0:
                h5['scr'] = model.predict_proba(rf_scr.X()).astype(np.int32)
            else:
                scr = h5.create_dataset('scr', shape=(rf_scr.ne_stream(), 2), dtype=np.float32)
                for i, x in enumerate(rf_scr.X_stream(chunksize=chunksize)):
                    base = i * chunksize
                    info('\t num_scr_examples: %d' % base)
                    scr[base:base + chunksize] = model.predict_proba(x)

            test_time = time() - start

        # Finally save meta-information
        metainfo = mlexp_info_helper(
            title='malaria-trees-oob',
            data_setup=data_id,
            model_setup=model_id,
            exp_function=fit,
        )
        metainfo.update((
            ('train_time', train_time),
            ('test_time', test_time),
            ('oob_auc', auc),
            ('oob_enrichment5', enrichment5),
            ('oob_accuracy', oob_score),
        ))
        with open(info_file, 'w') as writer:
            json.dump(metainfo, writer, indent=2, sort_keys=False)


if __name__ == '__main__':

    import argh
    parser = argh.ArghParser()
    parser.add_commands([fit])
    parser.dispatch()
