# coding=utf-8
"""
Management of predictive modelling evaluation and deployment results.

Digression (but should go into a design document / proper docstring...)
We need a way to specify coordinates for a single experiment, carrying context information.

A experiment is identified by:
  - The data preprocessing configuration (e.g. ecfp-fingerprints folded as...)
  - The model configuration (e.g. Normalization+LogReg(C=1))
  - The evaluation coordinates (seed and number of folds)

Reflecting this simple setup, the layout of results in disk should look as follows:
- prep_short_string
  |-- ..files-related-to-preprocessing..
  |-- model1-short-string
    |-- ..files-related-to-classifier..
    |-- eval1-short-string
      |-- ..files-related-to-evaluation..
"""
from glob import glob
import json
import os
import os.path as op
import h5py
import joblib


class ResultInDisk(object):

    def __init__(self, root_dir, ids_cache=None):
        """Access to predictive modelling results from evaluation experiments.
        We store:
          - The untrained model setup.
          - Possibly the trained (models)
          - Scores for the development dataset and possibly unlabelled datasets.
          - Possibly extra information like feature importances out of RFs or example weights out of boosting.
          - Other metainformation like the time taken in train and test, the date, the host...
        """
        super(ResultInDisk, self).__init__()
        self.root = op.abspath(root_dir)
        self.container_dir = op.abspath(op.join(self.root, '..', '..', '..'))
        self.data_dir = op.abspath(op.join(self.root, '..', '..'))
        self.model_dir = op.abspath(op.join(self.root, '..'))
        self.eval_dir = self.root

        self._ids_cache = ids_cache if ids_cache is not None else {}
        self._info_cache = None
        self._model_cache = None

    ########
    # Context information.
    ########

    def is_done(self):
        return op.isfile(op.join(self.eval_dir, 'info.json'))

    def check_done(self):
        if not self.is_done():
            raise Exception('The result at %s have not been computed yet...' % self.eval_dir)

    def ids(self, dset):
        if dset not in self._ids_cache:
            with open(op.join(self.data_dir, '%s.ids' % dset)) as reader:
                ids = [line.strip() for line in reader]
                self._ids_cache[dset] = ids
        return self._ids_cache[dset]

    def model_setup(self):
        if self._model_cache is None:
            self._model_cache = joblib.load(op.join(self.model_dir, 'model_setup.pkl'))
        return self._model_cache

    def root_key(self):
        return self.eval_dir[len(self.container_dir) + 1:]

    ########
    # Metainfo. My IDE will help me remember...
    ########

    def info(self):
        self.check_done()
        if self._info_cache is None:
            with open(op.join(self.eval_dir, 'info.json')) as reader:
                self._info_cache = json.load(reader)
                self._info_cache['eval_setup'] = op.basename(self.eval_dir)
        return self._info_cache

    def title(self):
        return self.info()['title']

    def data_setup_id(self):
        return self.info()['data_setup']

    def model_setup_id(self):
        return self.info()['model_setup']

    def eval_setup_id(self):
        return self.info()['eval_setup']

    def fsource(self):
        return self.info()['fsource']

    def comments(self):
        return self.info()['comments']

    def date(self):
        return self.info()['date']

    def host(self):
        return self.info()['host']

    def train_time(self):
        return self.info()['train_time']

    def test_time(self):
        return self.info()['test_time']

    def meta_keys(self):
        return ['title', 'data_setup', 'model_setup', 'eval_setup', 'fsource',
                'comments', 'date', 'host', 'train_time', 'test_time']

    def meta_as_lists(self):
        return self.meta_keys(), [self.info()[key] for key in self.meta_keys()]

    ########
    # Any result should provide scores for different datasets.
    ########

    def scores(self, dset):
        raise NotImplementedError

    ########
    # Results collection.
    ########

    @staticmethod
    def collect_results_under_dir(root, filters=(), factory=None):
        """Returns all results under a root directory."""
        results = []
        for dire, _, _ in os.walk(root):
            if any(not one_filter(dire) for one_filter in filters):  # ignore?
                continue
            result = factory(dire) if factory is not None else ResultInDisk(dire)
            if result is not None and result.is_done():
                results.append(result)
        return sorted(results, key=lambda result: result.root)


class OOBResult(ResultInDisk):

    def __init__(self, root_dir, ids_cache=None):
        """
        A result for a model evaluated using OOB estimates
        (anything based on resampling, from bagging to random forests).
        """
        super(OOBResult, self).__init__(root_dir, ids_cache)

        self._h5_file = None

        self._f_names_cache = None
        self._f_importances_cache = None
        self._scores_cache = None

    ########
    # Metainfo. My IDE will help me remember...
    ########

    def oob_auc(self):
        return self.info()['oob_auc']

    def oob_enrichment5(self):
        return self.info()['oob_enrichment5']

    def oob_accuracy(self):
        return self.info()['oob_accuracy']

    def meta_keys(self):
        return super(OOBResult, self).meta_keys() + ['oob_auc', 'oob_enrichment5', 'oob_accuracy']

    ########
    # Model results...
    ########

    def _h5(self):
        self.check_done()
        if self._h5_file is None:
            h5s = glob(op.join(self.eval_dir, '*__scores.h5'))
            if len(h5s) != 1:
                raise Exception('Unexpected number of files with scores: %r' % h5s)
            self._h5_file = h5s[0]
        return self._h5_file

    def f_names(self):
        with h5py.File(self._h5()) as h5:
            return h5['f_names'][:]

    def f_importances(self):
        with h5py.File(self._h5()) as h5:
            return h5['f_importances'][:]

    def scores(self, dset):
        with h5py.File(self._h5()) as h5:
            return h5[dset][:]
