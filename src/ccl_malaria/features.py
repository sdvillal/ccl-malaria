# coding=utf-8
"""Feature generation and munging."""
from array import array
from glob import glob
from operator import itemgetter
import os.path as op
from collections import defaultdict
from itertools import islice, izip, product, chain
import gzip
import os

import h5py
from joblib import Parallel, cpu_count, delayed
import numpy as np
import argh
from rdkit.Chem import AllChem
from scipy.sparse import coo_matrix, csr_matrix, vstack
from sklearn.utils.murmurhash import murmurhash3_32 as murmur

from ccl_malaria import info
from ccl_malaria import MALARIA_DATA_ROOT
from ccl_malaria.molscatalog import read_smiles_ultraiterator, MOLS2MOLS, MalariaCatalog
from ccl_malaria.rdkit_utils import explain_circular_substructure, RDKitDescriptorsComputer, to_rdkit_mol
from minioscail.common.configuration import Configurable
from minioscail.common.misc import ensure_dir, is_iterable


##################################################
# Configurable example (molecule representations) provider.
##################################################

class ExampleSet(Configurable):

    def __init__(self):
        """An example set provides examples, ids, features and labels."""
        super(ExampleSet, self).__init__()

    def ids(self):
        raise NotImplementedError

    def X(self):
        raise NotImplementedError

    def y(self):
        raise NotImplementedError

    def Xy(self):
        return self.X(), self.y()

    def iXy(self):
        return self.ids(), self.X(), self.y()

    def fnames(self):
        raise NotImplementedError

    def ids_stream(self, chunksize=1):
        raise NotImplementedError

    def X_stream(self, chunksize=1):
        raise NotImplementedError

    def y_stream(self, chunksize=1):
        raise NotImplementedError

    def ne_stream(self):
        return np.inf

    def Xy_stream(self, chunksize=1):
        chain(self.X_stream(chunksize=chunksize),
              self.y_stream(chunksize=chunksize))

    def iXy_stream(self, chunksize=1):
        chain(self.ids_stream(chunksize=chunksize),
              self.X_stream(chunksize=chunksize),
              self.y_stream(chunksize=chunksize))


##################################################
# COMPUTE STUFF IN STREAMS OF SMILES
# A sort of visitor pattern implementation
##################################################


_END_MOLID = None  # Marker smiles for end of iteration.


def _molidsmiles_it(start=0, step=46, mols=None, processor=None, logeach=500):
    """Iterates (molindex, molid, smiles) triplets skipping step molecules in each iteration.
    This is useful for evenly splitting workloads between processors / machines.
    Parameters:
      - start: the index of the first pair to consider
      - step: how many molecules are skipped on each iteration
      - mols: an iterator (molid, smiles)
      - processor: a function that gets called for each pair;
                   when the iterator is exhausted, (_END_MOLID, None) is sent.
    """
    if mols is None:
        mols = read_smiles_ultraiterator()
    for molindex, (molid, smiles) in enumerate(islice(mols, start, None, step)):
        if logeach > 0 and molindex > 0 and not molindex % logeach:
            info('Molecule %d' % molindex)
        processor(molid, smiles)
    processor(_END_MOLID, None)


def _sort_by_start(fns):
    """Sorts filenames by their "start" field."""
    def start(fn):
        return int(fn.partition('start=')[2].partition('__')[0])
    return sorted(fns, key=start)


def _process_molecule_data(moldata_it, processors):
    """Apply a processor to each item in an iterator.

    Parameters:
      - moldata_it: an iterator over general items
      - processors: a list of processors, that must define a method
                    "process(item)" that gets called for each item and
                    another method "done()" that gets called when the iterator
                    gets exhausted.
    """
    for moldata in moldata_it:
        for proc in processors:
            proc.process(moldata)
    for proc in processors:
        proc.done()


##################################################
# Computation of rdkit features (these in rdk.Descriptor)
##################################################

def _rdkfeats_writer(output_file=None, features=None):
    """Returns a (molindex, molid, smiles) processor that computes descriptors using RDKit and stores then in a h5 file.

    Parameters:
      - output_file: where the descriptors will be written; this file will be overwritten.
      - features: a list of the names of the RDKit features that will be computed
                  (by default all the descriptors exposed by the Descriptor class in RDKit)

    Returns:
      - a processor function ready to be used as a parameter to _molidsmiles_it.

    The h5 file has the following data:
      - 'rdkdescs': a float matrix num_mols x num_descs
                    this will all be nans if the computation failed completely
      - 'fnames': the name of the feature in each column (num_cols)
      - 'molids': the molid corresponding to each row (num_rows)
    """
    ensure_dir(op.dirname(output_file))
    h5 = h5py.File(output_file, mode='w', dtype=np.float32)
    computer = RDKitDescriptorsComputer(features)
    fnames = computer.fnames()
    nf = len(fnames)
    descs = h5.create_dataset('rdkdescs', (0, nf), maxshape=(None, nf), compression='lzf')
    str_type = h5py.new_vlen(str)
    h5.create_dataset('fnames', data=fnames)
    molids = h5.create_dataset('molids', shape=(0,), maxshape=(None,), dtype=str_type)

    def process(molid, smiles):
        if molid is _END_MOLID:
            h5.close()
            return
        ne = len(molids)
        try:
            molids.resize((ne + 1,))
            molids[ne] = molid
            mol = to_rdkit_mol(smiles)
            descs.resize((ne + 1, nf))
            descs[ne, :] = computer.compute(mol)[0]
        except:
            info('Failed molecule %s: %s' % (molid, smiles))
            descs[ne, :] = [np.nan] * nf

    return process


def rdkfs(start=0, step=46, mols='all', output_file=None):
    """Entry point for the command line to generate rdkit descriptors.
    Parameters:
      - start: the index of the first molecule to consider
      - step: how many molecules are skipped in each iteration
      - mols: an iterator over pairs (molid, smiles) or a string
              ('lab'|'unl'|'scr'|'all') to use one of TDT malaria's iterators
      - output_file: the file to which the fingerprints will be written, an hdf5 file.

    This is fairly fast and can finish in a few hours even for the screening dataset, so we recommend not to bother
    with multithreaded computation (and we do not support yet merging of multiple runs). In other words, this is
    better called as:
    rdkfs --start 0 --step 1 --mols scr --output-file ~/scr-rdkfs.h5
    """
    if isinstance(mols, basestring):
        mols = MOLS2MOLS[mols]()
    _molidsmiles_it(start=start, step=step,
                    mols=mols,
                    processor=_rdkfeats_writer(output_file))


class MalariaRDKFsExampleSet(ExampleSet):

    def __init__(self, dset='lab', remove_ambiguous=True, remove_with_nan=True, streaming_in_ram=False):
        super(MalariaRDKFsExampleSet, self).__init__()
        self.dset = dset
        self.remove_ambiguous = remove_ambiguous
        self.remove_with_nan = remove_with_nan
        self._streaming_in_ram = streaming_in_ram
        # ATM all in memory...
        self._X = None
        self._y = None
        self._molids = None
        self._fnames = None
        self._h5 = op.join(MALARIA_DATA_ROOT, 'rdkit', 'rdkfs', '%srdkf.h5' %
                                                                (self.dset if self.dset != 'amb' else 'lab'))
        # Dirty, time allowing we should redesign to better take into account ambiguous

    def _populate(self):
        if self._molids is None or self._X is None or self._y is None:
            with h5py.File(self._h5) as h5:
                X = h5['rdkdescs'][:]
                molids = h5['molids'][:]
                fnames = h5['fnames'][:]
                y = MalariaCatalog().labels(molids, as01=True, asnp=True)
                if self.dset == 'amb':
                    rows_with_label = np.isnan(y)  # Dirty hack
                else:
                    rows_with_label = ~np.isnan(y) if self.remove_ambiguous else np.array([True] * len(y))
                rows_without_nan = ~np.isnan(X).any(axis=1) if self.remove_with_nan else np.array([True] * len(y))
                rows_to_keep = rows_without_nan & rows_with_label
                self._X = X[rows_to_keep]
                self._y = y[rows_to_keep]
                self._molids = molids[rows_to_keep]
                self._fnames = fnames

    def rows_to_keep(self):
        """Temporary workaround to streaming not removing with nan bug."""
        with h5py.File(self._h5) as h5:
            X = h5['rdkdescs'][:]
            molids = h5['molids'][:]
            y = MalariaCatalog().labels(molids, as01=True, asnp=True)
            if self.dset == 'amb':
                rows_with_label = np.isnan(y)  # Dirty hack
            else:
                rows_with_label = ~np.isnan(y) if self.remove_ambiguous else np.array([True] * len(y))
            rows_without_nan = ~np.isnan(X).any(axis=1) if self.remove_with_nan else np.array([True] * len(y))
            return rows_without_nan & rows_with_label

    def ids(self):
        self._populate()
        return self._molids

    def X(self):
        self._populate()
        return self._X

    def y(self):
        self._populate()
        return self._y

    def fnames(self):
        return self._fnames

    def ne_stream(self):
        if self.dset == 'lab' or self.dset == 'amb':
            return self.X().shape[0]  # Caveat: disallows streaming from labelled.
        with h5py.File(self._h5) as h5:
            return h5['rdkdescs'].shape[0]

    def X_stream(self, chunksize=10000):
        if self._streaming_in_ram or self.dset == 'lab' or self.dset == 'amb':  # Dodgy lab and amb, remove corner case
            for start in xrange(0, self.ne_stream(), chunksize):
                yield self.X()[start:start+chunksize]
        else:
            h5 = h5py.File(self._h5)['rdkdescs']
            for start in xrange(0, self.ne_stream(), chunksize):
                yield h5[start:start+chunksize]  # FIXME: bug, do not skip some of the molecules
                                                 # that need to be skipped remove ambiguous and
                                                 # guys with nans, it is simple
            h5.close()


##################################################
# Computation of ECFPs and FCFPs
##################################################

_MALARIA_ECFPS_DIR = op.join(MALARIA_DATA_ROOT, 'rdkit', 'ecfps')
_MALARIA_ECFPS_PARALLEL_RESULTS_DIR = op.join(_MALARIA_ECFPS_DIR, 'from_workers')


def _ecfp_writer(output_file=None, max_radius=200, fcfp=False, write_centers=False):
    """Returns a (molid, smiles) processor that computes ecfps on the smiles and stores then in a text file.

    Parameters:
      - output_file: where the fingerprints will be written; this file will be overwritten and gzipped.
      - max_radius: the maximum radius for the found circular substructures.
      - fcfp: whether to generate FCFP or ECFP like fingerprints.
      - write_centers: whether to also save all the centers of each substructure or not.

    Returns:
      - a processor function ready to be used as a parameter to _molidsmiles_it.

    We call the format generated "weird fp format". On each line we have either:
        molid[\tcansmi count [center radius]+]+
    or, if there is an error with the molecule:
        molindex\molid\t*FAILED*
    Note that parsing this format is quite consuming, so we recommend to simplify it if a subset of this information
    is to be read repeatly.
    """
    writer = gzip.open(output_file, 'w')

    def process(molid, smiles):
        if molid is _END_MOLID:
            writer.close()
            return
        try:
            mol = to_rdkit_mol(smiles)
            fpsinfo = {}
            # N.B. We won't actually use rdkit hash, so we won't ask for nonzero values...
            # Is there a way of asking rdkit to give us this directly?
            AllChem.GetMorganFingerprint(mol, max_radius, bitInfo=fpsinfo, useFeatures=fcfp)
            counts = defaultdict(int)
            centers = defaultdict(list)
            for bit_descs in fpsinfo.values():
                for center, radius in bit_descs:
                    cansmiles = explain_circular_substructure(mol, center, radius)
                    counts[cansmiles] += 1
                    centers[cansmiles].append((center, radius))
            if write_centers:
                features_strings = ['%s %d %s' % (cansmiles,
                                                  count,
                                                  ' '.join(['%d %d' % (c, r) for c, r in centers[cansmiles]]))
                                    for cansmiles, count in counts.iteritems()]
            else:
                features_strings = ['%s %d' % (cansmiles, count) for cansmiles, count in counts.iteritems()]
            writer.write('%s\t%s\n' % (molid, '\t'.join(features_strings)))
        except:
            info('Failed molecule %s: %s' % (molid, smiles))
            writer.write('%s\t*FAILED*\n' % molid)
    return process


def ecfps(start=0, step=46, mols='lab', output_file=None, fcfp=True):
    """Entry point for the command line to generate fingerprints.
    Parameters:
      - start: the index of the first molecule to consider
      - step: how many molecules are skipped in each iteration
      - mols: an iterator over pairs (molid, smiles) or a string
              ('lab'|'unl'|'scr'|'all') to use one of TDT malaria's iterators
      - fcfp: generate FCFPs or ECFPs
      - output_file: the file to which the fingerprints will be written, in
                     weird fp format(TM).
    """
    if isinstance(mols, basestring):
        mols = MOLS2MOLS[mols]()
    ensure_dir(op.dirname(output_file))
    _molidsmiles_it(start=start, step=step,
                    mols=mols,
                    processor=_ecfp_writer(output_file=output_file, fcfp=fcfp))


def _molidsmiles_it_ecfp(output_file, start=0, step=46, fcfp=True, logeach=5000):
    """Q&D variant to allow Parallel work (cannot pickle closures or reuse iterators...)."""
    processor = _ecfp_writer(output_file=output_file, fcfp=fcfp)
    mols = read_smiles_ultraiterator()
    for molindex, (molid, smiles) in enumerate(islice(mols, start, None, step)):
        if logeach > 0 and molindex > 0 and not molindex % logeach:
            info('Molecule %d' % molindex)
        processor(molid, smiles)
    processor(_END_MOLID, None)


def ecfps_mp(numjobs=None, dest_dir=None):
    """Python-parallel computation of ECFPs.
    Parameters:
      - numjobs: the number of threads to use (None=all in the machine).
      - dest_dir: the directory to which the fingerprints will be written, in weird fp format(TM).
    """
    dest_dir = _MALARIA_ECFPS_PARALLEL_RESULTS_DIR if dest_dir is None else dest_dir
    ensure_dir(dest_dir)
    numjobs = cpu_count() if numjobs is None else int(numjobs)
    Parallel(n_jobs=numjobs)(delayed(_molidsmiles_it_ecfp)
                             (start=start,
                              step=numjobs,
                              output_file=op.join(dest_dir, 'all__fcfp=%r__start=%d__step=%d.weirdfps' %
                                                            (fcfp, start, numjobs)),
                              fcfp=fcfp)
                             for start, fcfp in product(range(numjobs), (True, False)))


def munge_ecfps():

    #####---Step 1: put all these together in 3 files, lab, unl and scr.
    #####     - ECFPs and FCFPs for the same mol are together
    #####     - The order is the same as in the original file
    #####     - Optionally delete the workers files

    def parse_weirdfpformat_line(line):
        """Returns a tuple (molid, [cansmi, count, [(center, radius)]+]+)."""
        def _parse_weird_feature(feature):
            vals = feature.split()
            cansmi = vals[0]
            count = int(vals[1])
            if len(vals) > 2:
                a = iter(vals[2:])
                centers = [(center, radius) for center, radius in izip(a, a)]
                return cansmi, count, centers
            return cansmi, count, ()
        values = line.strip().split('\t')
        molid = values[0]
        if '*FAILED*' in values[1]:
            return molid, None
        return molid, map(_parse_weird_feature, values[1:])

    def malaria_ecfp_parallel_results_iterator(prefix='', log=True):
        """Iterates over the files resulting from the computation of ecfps using the function ecfp."""
        weirdfps = glob(op.join(_MALARIA_ECFPS_PARALLEL_RESULTS_DIR, '%s*.weirdfps' % prefix))
        weirdfps = _sort_by_start(weirdfps)
        for fn in weirdfps:
            if log:
                info(fn)
            with gzip.open(fn) as reader:
                for line in reader:
                    yield line

    class Chihuahua(object):
        """A data processor that takes weirdfp lines, hunk them in disk and then merge them sorted in a big file.
        It can be setup to be easy on memory usage (at the cost of doubling disk space usage).
        """
        def __init__(self, molid2i, root, prefix, data2molid, chunksize=10000):
            super(Chihuahua, self).__init__()
            self.chunksize = chunksize
            self.molid2i = molid2i
            self.num_mols = len(self.molid2i)
            self.temp_fns = [op.join(root, '%s-%d' % (prefix, base)) for base in xrange(0, self.num_mols, chunksize)]
            self.temp_files = [open(fn, 'w') for fn in self.temp_fns]
            self.data2molid = data2molid
            self.root = root
            self.prefix = prefix
            ensure_dir(self.root)

        def process(self, moldata):
            index = self.molid2i.get(self.data2molid(moldata), None)
            if index is None:
                return
            goes_to = index / self.chunksize
            self.temp_files[goes_to].write(moldata)
            if not moldata.endswith('\n'):
                self.temp_files[goes_to].write('\n')

        def done(self):
            # Sort in memory each chunk
            for tmp in self.temp_files:
                tmp.close()
            with open(op.join(self.root, self.prefix), 'w') as writer:
                for fn in self.temp_fns:
                    with open(fn) as reader:
                        lines = sorted(reader.readlines(), key=lambda line: self.molid2i[self.data2molid(line)])
                        for line in lines:
                            writer.write(line)
            for fn in self.temp_fns:
                os.remove(fn)

    mc = MalariaCatalog()

    labproc = Chihuahua(molid2i={molid: i for i, molid in enumerate(mc.lab())},
                        root=_MALARIA_ECFPS_DIR,
                        prefix='lab',
                        data2molid=lambda line: line[0:line.find('\t')],
                        chunksize=100000)

    unlproc = Chihuahua(molid2i={molid: i for i, molid in enumerate(mc.unl())},
                        root=_MALARIA_ECFPS_DIR,
                        prefix='unl',
                        data2molid=lambda line: line[0:line.find('\t')],
                        chunksize=100000)

    scrproc = Chihuahua(molid2i={molid: i for i, molid in enumerate(mc.scr())},
                        root=_MALARIA_ECFPS_DIR,
                        prefix='scr',
                        data2molid=lambda line: line[0:line.find('\t')],
                        chunksize=100000)

    _process_molecule_data(malaria_ecfp_parallel_results_iterator(), (labproc, unlproc, scrproc))

    #####---Step 2: recode ECFPs and FCFPs from the file at step 1. After this:
    #####  - ECFPs and FCFPs duplicates get merged.
    #####  - A unique assignment for each substructure in the dataset to a int [0, ...] (column number).
    #####  - A unique assignment for each molid in the dataset for wich Morgan DID NOT FAIL (row number).

    def ecfps_recode(dset='lab'):
        """Merges ECFPs and FCFPs into a single line and gets rid of the centers information if present."""
        with open(op.join(_MALARIA_ECFPS_DIR, dset)) as reader, \
                open(op.join(_MALARIA_ECFPS_DIR, dset + '.merged'), 'w') as writer:
            for ecfp in reader:
                fcfp = reader.next()
                molide, subse = parse_weirdfpformat_line(ecfp)
                molidf, subsf = parse_weirdfpformat_line(fcfp)
                assert molide == molidf
                if subse is not None:
                    uniques = set((sub, count) for sub, count, _ in subse + subsf)
                    writer.write('%s\t%s' % (molide, '\t'.join(['%s %d' % (sub, count) for sub, count in uniques])))
                    writer.write('\n')
    ecfps_recode('lab')
    ecfps_recode('unl')
    ecfps_recode('scr')

    def sub2i():
        """Generates a map {labelled_substructure -> column}
        This produces a unique assignment for all the features in the dataset, in three files:
          - lab: the indices for all features that appear in labelled
          - unl: the indices for features that do not appear in labelled but appear in unlabelld
          - scr: the indices for the features that appear in screening but not on labelled or unlabelled
        Of course, keep track of changes to the map as needed while creating models.

        Note that this keeps all the substructures in memory (which shoould be ok for any recent machine).
        """
        def all_subs(dset):
            info(dset)
            subs = set()
            with open(op.join(_MALARIA_ECFPS_DIR, dset + '.merged')) as reader:
                for line in reader:
                    subs.update(sub.split()[0] for sub in line.split('\t')[1:])  # TODO sort by frequency
            return subs
        lab_subs = all_subs('lab')
        unl_subs = all_subs('unl')
        scr_subs = all_subs('scr')
        with open(op.join(_MALARIA_ECFPS_DIR, 'lab.merged.s2i'), 'w') as writer:
            for i, sub in enumerate(sorted(lab_subs)):
                writer.write('%s %d\n' % (sub, i))
        num_written = len(lab_subs)
        with open(op.join(_MALARIA_ECFPS_DIR, 'unl.merged.s2i'), 'w') as writer:
            new_subs = unl_subs - lab_subs
            for i, sub in enumerate(sorted(new_subs)):
                writer.write('%s %d\n' % (sub, i + num_written))
            num_written += len(new_subs)
        with open(op.join(_MALARIA_ECFPS_DIR, 'scr.merged.s2i'), 'w') as writer:
            new_subs = scr_subs - (unl_subs | lab_subs)
            for i, sub in enumerate(sorted(new_subs)):
                writer.write('%s %d\n' % (sub, i + num_written))
        with open(op.join(_MALARIA_ECFPS_DIR, 'trans.merged.s2i'), 'w') as writer:
            for sub in sorted(lab_subs & unl_subs | lab_subs & scr_subs):
                writer.write('%s\n' % sub)
    sub2i()

    def mol2i(dset='lab'):
        """Generates a map {molid -> row}.
        Molecules for which RDKIT could not generate the fingerprints are not in this map,
        nor in hte final sparse matrices.
        In any case we will need to keep track of changes on the map as we do, for example, cross-val.
        """
        with open(op.join(_MALARIA_ECFPS_DIR, dset + '.merged.m2i'), 'w') as writer:
            with open(op.join(_MALARIA_ECFPS_DIR, dset + '.merged')) as reader:
                for line in reader:
                    writer.write('%s\n' % line[0:line.find('\t')])
    mol2i('lab')
    mol2i('unl')
    mol2i('scr')

    #####---Step 3: write sparse matrices with the recoded information of step 2. After this:
    #####  - We get a h5 file for each dataset, with a sparse matrix in CSR format.
    #####  - Note that this is a memory intense procedure, can be done lightweight by using 2 passes.

    def to_sparse_chihuahua(dset='lab', two_pass=False):
        """Generates sparse CSR matrices using as features only these in the labelled dataset,
        with the column index and the row index as computed previously.
        They get stored in a h5 file with the following datasets:
          - data
          - indices
          - indptr
          - shape
        """
        if two_pass:
            # First pass: shape and number of nonzeros
            # Second pass: h5 file with the proper sizes of indices, indptr and data, write on the fly
            raise NotImplementedError
        # mol2row, smiles2col
        m2i = {mol.strip(): i for i, mol in enumerate(open(op.join(_MALARIA_ECFPS_DIR, dset + '.merged.m2i')))}
        s2i = {}
        with open(op.join(_MALARIA_ECFPS_DIR, 'lab.merged.s2i')) as reader:
            for line in reader:
                sub, i = line.strip().split()
                i = int(i)
                s2i[sub] = i
        rows = array('I')
        cols = array('I')
        data = array('I')
        # gather data
        with open(op.join(_MALARIA_ECFPS_DIR, dset + '.merged')) as reader:
            for line in reader:
                values = line.split('\t')
                molid = values[0]
                row = m2i[molid]
                for fc in values[1:]:
                    sub, count = fc.split()
                    count = int(count)
                    col = s2i.get(sub, None)
                    if col is not None:
                        rows.append(row)
                        cols.append(col)
                        data.append(count)
        # save as CSR sparse matrix
        M = coo_matrix((data, (rows, cols)), dtype=np.int32).tocsr()
        with h5py.File(op.join(_MALARIA_ECFPS_DIR, dset + '.sparse.h5'), 'w') as h5:
            h5['indices'] = M.indices
            h5['indptr'] = M.indptr
            h5['data'] = data
            h5['shape'] = np.array([M.shape[0], len(s2i)])
    to_sparse_chihuahua('lab')
    to_sparse_chihuahua('unl')
    to_sparse_chihuahua('scr')

    #####---Step 4: lame feature duplicate detection to tackle partially multicolliniarity
    MalariaFingerprintsManager(zero_dupes='lab').X()
    MalariaFingerprintsManager(zero_dupes='all').X()


def detect_duplicate_features(transductive=False, verbose=False):
    """Detect exact duplicated features in the malaria dataset, returning a list of duplicated groups (column indices).
    Here duplicated is very practically defined as "appearing in the same molecules accross the malaria dataset".
    """

    # TODO: this is really memory intensive, make streaming (over the columns...)
    # TODO: manage ambiguous...

    # Are there many singleton features collapsed?

    if transductive:
        Xlab = MalariaFingerprintsManager(dset='lab', keep_ambiguous=False).X()
        Xunl = MalariaFingerprintsManager(dset='unl', keep_ambiguous=True).X()
        Xscr = MalariaFingerprintsManager(dset='scr', keep_ambiguous=True).X()
        X = vstack((Xlab, Xunl, Xscr))
    else:
        X = MalariaFingerprintsManager(dset='lab', keep_ambiguous=False).X()

    info('MatrixMol Feature Duplicate detection')
    info('We are dealing with a matrix as big as %d molecules and %d features' % X.shape)

    ne, nf = X.shape
    X = X.tocsc()
    X.indices.flags.writeable = False  # Make the views from this array hashable
    groups = defaultdict(lambda: array('I'))
    for i in xrange(nf):
        xi = X.indices[X.indptr[i]:X.indptr[i+1]:]
        groups[xi.data].append(i)
        if verbose and i > 0 and not i % 1000000:
            info('%d of %d substructures hashed according to the molecules they pertain' % (i, nf))

    return groups.values()


def zero_columns(X, columns, zero_other=False):
    X = X.tocsc()  # Duplicate memory!
    num_cols = X.shape[1]
    if zero_other:
        # Equivalent but 400 times faster than sorted(set(xrange(num_cols)) - set(columns))
        to_zero = np.ones(num_cols)
        to_zero[columns] = 0
        columns = np.where(to_zero)[0]
    for col in columns:  # This for is the hotspot ATM
        X.data[X.indptr[col]:X.indptr[col+1]] = 0
    X.eliminate_zeros()
    return X.tocsr()


class MalariaFingerprintsManager(object):

    # Q&D cache. Next: give a thought to caching and probably use weakrefs; use a wrapper.

    def __init__(self,
                 root=_MALARIA_ECFPS_DIR,
                 dset='lab',
                 only_labelled=True,         # Keep only labelled instances
                 only01=True,                # y as {0, 1}
                 keep_ambiguous=False,       # Keep ambiguous instances? Ignored for any dataset other than 'lab'
                 zero_dupes=None,            # Make zero all but one columns in each duplicated column
                 streaming_from_ram=False):  # Streaming is from memory
        """Ad-hoc class to manage fingerprinting for the malaria competition.

        N.B.: The current implementation can hit hard memory usage,
        but it is not hard to make it more conservative.
        """
        super(MalariaFingerprintsManager, self).__init__()

        # The dataset we are looking at (lab|unl|scr)
        self.dset = dset

        # Files
        # Maybe we should consider a few h5s for some of these...
        self._root = root
        self._original_file = op.join(root, '%s.merged' % (self.dset if self.dset != 'amb' else 'lab'))
        self._m2i_file = op.join(root, '%s.merged.m2i' % (self.dset if self.dset != 'amb' else 'lab'))
        # N.B. s2i_file ignored ATM
        self._s2i_file = op.join(root, '%s.merged.s2i' % (self.dset if self.dset != 'amb' else 'lab'))
        self._lab_s2i_file = op.join(root, 'lab.merged.s2i')
        self._trans_s2i_file = op.join(root, 'trans.merged.s2i')
        self._sparse_file = op.join(root, '%s.sparse.h5' % (self.dset if self.dset != 'amb' else 'lab'))

        self._only01 = only01
        self._keep_ambiguous = keep_ambiguous
        self._only_labelled = only_labelled
        self._streaming_from_ram = streaming_from_ram
        if not self._only_labelled:
            raise NotImplementedError
        if zero_dupes and not zero_dupes in {'all', 'lab'}:
            raise Exception('Zero dupes in lab must be one of {None, all, lab}, but %r' % zero_dupes)
        # Something in {None|lab|all} (recommended all)
        self._zero_dupes = zero_dupes

        # Q&D caches
        self._m2i = None
        self._s2i = None
        self._i2m = None
        self._i2s = None
        self._XCSR = None
        self._XCSC = None
        self._y = None
        self._labelled_mask = None
        self._features = defaultdict(set)

    def h5(self):
        """Returns the path to the hdf5 file containing the sparse matrix in CSR format."""
        return self._sparse_file

    def _s2i_cache(self):
        """Caches and returns the map {substructure->col}."""
        if self._s2i is None:
            def pl(line):
                values = line.strip().split()
                return values[0], int(values[1])
            with open(self._lab_s2i_file) as reader:  # N.B. HARDCODED ONLY LAB ATM
                self._s2i = {s: i for s, i in map(pl, reader)}
        return self._s2i

    def s2i(self, s):
        """Maps a substructure to the column number (or None if the substructure is unknow)."""
        return self._s2i_cache().get(s, None)

    def _i2s_cache(self):
        """Caches and return a numpy array of known substructures."""
        if self._i2s is None:
            self._i2s = np.array([s for s, _ in sorted(self._s2i_cache().items(), key=itemgetter(1))])
        return self._i2s

    def i2s(self, i):
        """Returns the substructure corresponding to the column i."""
        return self._i2s_cache()[i]

    def substructures(self):
        """Returns a numpy of substructures corresponding to columns in the matrix."""
        return self._i2s_cache()

    def duplicate_features_representatives(self, transductive=True):
        """Returns a numpy array with one representative (column) for each duplicated feature."""
        representatives_file = op.join(self._root, 'duplicated_representatives.h5')
        dset = 'dall' if transductive else 'dlab'
        with h5py.File(representatives_file) as h5:
            if dset not in h5:
                dfeats = detect_duplicate_features(transductive=transductive)
                h5[dset] = np.array([dfeat[0] for dfeat in dfeats], dtype=np.int32)
                # Back map
                back = np.zeros(self.X().shape[1], dtype=np.int32)
                for group in dfeats:
                    back[np.array(group)] = group[0]
                h5[dset+'_back'] = back
            return h5[dset][:], h5[dset+'_back'][:]

    def transductive_features(self):
        """Returns a set with the features that appear in the labelled set and also in either unl or scr."""
        with open(self._trans_s2i_file) as reader:
            return set([molid.rstrip() for molid in reader])

    def mols_with_feature(self, bit_or_smiles):
        """Returns a list with the molids of the molecules that contains one feature."""
        column = self.s2i(bit_or_smiles) if isinstance(bit_or_smiles, basestring) else bit_or_smiles
        molindices = self.XCSC().indices[self.XCSC().indptr[column]:self.XCSC().indptr[column+1]]
        return [self.i2m(index) for index in molindices]

    def m2i(self, molid):
        """Returns the row index for a molid (or None if the molid is unknown)."""
        # if self._m2i is None:
        #     with open(self._m2i_file) as reader:
        #         self._m2i = {m.rstrip(): i for i, m in enumerate(reader)}
        if self._m2i is None:
            self._m2i = {molid: i for i, molid in enumerate(self.molids())}
        return self._m2i.get(molid, None)

    def _i2m_cache(self):
        """Caches and returns a numpy array with all the known molids, in the proper order."""
        if self._i2m is None:
            with open(self._m2i_file) as reader:
                self._i2m = np.array([m.rstrip() for m in reader])
                if self.dset == 'amb':
                    self._i2m = self._i2m[~self._labelled()]
                elif self.dset == 'lab' and not self._keep_ambiguous:
                    self._i2m = self._i2m[self._labelled()]
        return self._i2m

    def i2m(self, i):
        """Returns the molid corresponding to a row."""
        return self._i2m_cache()[i]

    def molids(self):
        """Returns a numpy array with the molids corresponding to each row in the matrix."""
        return self._i2m_cache()

    def _labelled(self):
        if self._labelled_mask is None:
            with open(self._m2i_file) as reader:
                molids = np.array([m.rstrip() for m in reader])
                y = MalariaCatalog().labels(molids, as01=True, asnp=True)
                self._labelled_mask = ~np.isnan(y)
        return self._labelled_mask

    def y(self):
        """Returns the labels (0 or 1) corresponding to the molecules in this matrix."""
        if self._y is None:
            self._y = MalariaCatalog().labels(self._i2m_cache(), as01=True, asnp=True)
        return self._y

    def X(self):
        """Returns a CSR sparse matrix with the fingerprints for each molecule."""
        if self._XCSR is None:
            with h5py.File(self._sparse_file, 'r') as h5:
                indices = h5['indices'][:]
                indptr = h5['indptr'][:]
                data = np.ones(len(indices), dtype=np.int8) if self._only01 else h5['data'][:]  # bools are ints in np
                                                                                          # smallest datatype is int8
                                                                                          # TODO implicit sparse 1/0
                                                                                          # (ala oscail-java)
                shape = h5['shape'][:]
                self._XCSR = csr_matrix((data, indices, indptr), shape=shape)
                if self.dset == 'amb':
                    self._XCSR = self._XCSR[np.where(~self._labelled())[0], :]
                elif not self._keep_ambiguous:
                    self._XCSR = self._XCSR[np.where(self._labelled())[0], :]
                # Zero duplicated columns
                # Dirty trick to remove duplicates without changing the map {col->structure)
                if self._zero_dupes:
                    transductive = self._zero_dupes == 'all'
                    self._XCSR = zero_columns(self._XCSR,
                                              self.duplicate_features_representatives(transductive=transductive)[0],
                                              zero_other=True)

        return self._XCSR

    def ne_stream(self):
        if self._streaming_from_ram or self.dset == 'lab' or self.dset == 'amb':
            return self.X().shape[0]  # Caveat: disallows streaming from labelled.
        with h5py.File(self._sparse_file, 'r') as h5:
            return h5['shape'][:][0]

    def X_stream(self, chunksize=10000):
        if self._streaming_from_ram or self.dset == 'lab' or self.dset == 'amb':
            for start in xrange(0, self.ne_stream(), chunksize):
                yield self.X()[start:start+chunksize, :]
                # Warn in the docstring that lab and amb will in any case still require all in ram
                # Better, implement streaming in this case properly,
                # or even better, split at the beginning between lab and amb!
                # Will get rid of this special case headache
        else:
            h5 = h5py.File(self._sparse_file, 'r')
            indptr = h5['indptr']
            indices = h5['indices']
            data = h5['data']
            shape = h5['shape'][:]
            columns_to_zero = None
            if self._zero_dupes:
                not_to_zero = self.duplicate_features_representatives(transductive=self._zero_dupes == 'all')[0]
                columns_to_zero = np.ones(shape[1])
                columns_to_zero[not_to_zero] = 0
                columns_to_zero = np.where(columns_to_zero)[0]
            for start in xrange(0, self.ne_stream(), chunksize):
                chunk_ptrs = indptr[start:start+chunksize+1]
                chunk_indices = indices[chunk_ptrs[0]:chunk_ptrs[-1]]
                chunk_data = data[chunk_ptrs[0]:chunk_ptrs[-1]] if not self._only01 \
                    else np.ones(chunk_ptrs[-1] - chunk_ptrs[0])
                X = csr_matrix((chunk_data, chunk_indices, chunk_ptrs - chunk_ptrs[0]),
                               shape=(len(chunk_ptrs) - 1, shape[1]))
                if columns_to_zero is not None:
                    yield zero_columns(X, columns_to_zero, zero_other=False)
                yield(X)
            h5.close()

    def XCSC(self):
        """Same as X, but returns a CSC matrix."""
        if self._XCSC is None:
            self._XCSC = self.X().tocsc()
        return self._XCSC

    def Xy(self):
        """Returns the fingerprints and the labels."""
        return self.X(), self.y()

    def clear_cache(self):
        self._features = defaultdict(set)


def fold_csr(X, folder, slow=False):
    """A convenience method to fold a CSR sparse matrix using a folder (hashing + bucket assignment method)."""
    X = X.tocsr()
    # Is there an efficient way of doing this for the whole matrix?
    # This is tempting:
    #   folded = murmur(X.indices) % fold_size
    #   X.indices = folded
    # But then the indices for each row can be non-unique and non-sorted
    # Let's do a lot of work here...
    if slow:
        cols = [np.unique(folder.assign_feature(X[row, :].indices)) for row in xrange(X.shape[0])]  # Fold
    else:
        cols = []
        for row in xrange(X.shape[0]):
            indices = X.indices[X.indptr[row]:X.indptr[row + 1]]
            cols.append(np.unique(folder.assign_feature(indices)))

    def rowindices(row_num, num_cols):
        rows = np.empty(num_cols, dtype=np.int)
        rows.fill(row_num)
        return rows
    rows = np.hstack([rowindices(i, len(c)) for i, c in enumerate(cols)])
    data = np.ones(len(rows), dtype=np.float)  # dtype=np.bool
    return coo_matrix((data, (rows, np.hstack(cols))), shape=(X.shape[0], folder.fold_size)).tocsr()


class MurmurFolder(Configurable):

    def __init__(self, seed=0, positive=True, fold_size=1023, save_map=True):
        """Hashes and folds substructures into features in R^d, keeping track of what goes where."""
        super(MurmurFolder, self).__init__()
        self.seed = seed
        self.positive = positive
        self.fold_size = fold_size
        self._save_map = save_map
        self._features = defaultdict(set)  # What happens when we serialize? This will be 2E6 big...

    def features(self):
        return self._features

    def hash(self, substructure):
        return murmur(substructure, seed=self.seed, positive=self.positive)

    def assign_feature(self, substructures):
        if not is_iterable(substructures):
            substructures = (substructures,)
        features = self.hash(substructures) % self.fold_size
        if self._save_map:
            for feature, key in izip(features, substructures):
                self._features[feature].add(key)
        return features

    def substructures_for_feature(self, feature):
        return sorted(self._features[feature])

    def fold(self, X):
        return fold_csr(X, self)

    def folded2unfolded(self):
        """Return a numpy array with bucket assignments for each original feature.
        That is csfold()[4] gives the fold for unfolded feature 4.
        """
        max_unfolded_feature = max(max(group) for group in self._features.itervalues())
        fold2unfold = np.empty(max_unfolded_feature + 1, dtype=np.int32)
        fold2unfold.fill(-1)
        for folded_feature, group in self._features.iteritems():
            fold2unfold[list(group)] = folded_feature
        return fold2unfold

    def clear_cache(self):
        self._features = defaultdict(set)


class MalariaFingerprintsExampleSet(ExampleSet):
    # TODO: consider merge this with MalariaFingerprintsManager

    def __init__(self,
                 dset='lab',
                 only_labelled_features=True,
                 only01=True,
                 remove_ambiguous=True,
                 zero_dupes='all',
                 folder=None):
        super(MalariaFingerprintsExampleSet, self).__init__()
        self.dset = dset
        self.only_labelled = only_labelled_features
        self.only01 = only01
        self.keep_ambiguous = not remove_ambiguous
        self.zero_dupes = zero_dupes
        self.folder = folder  # N.B. this can carry a lot of data, danger for serialization...
        self._mfm = None      # N.B. this can carry a lot of data, danger for serialization...
        self._X = None        # N.B. this can carry a lot of data, danger for serialization...

    def mfm(self):
        if self._mfm is None:
            self._mfm = MalariaFingerprintsManager(dset=self.dset,
                                                   only_labelled=self.only_labelled,
                                                   only01=self.only01,
                                                   keep_ambiguous=self.keep_ambiguous,
                                                   zero_dupes=self.zero_dupes)
        return self._mfm

    def X(self):
        if self.folder is None:
            return self.mfm().X()
        if self._X is None:
            self._X = self.folder.fold(self.mfm().X())
        return self._X

    def y(self):
        return self.mfm().y()

    def ids(self):
        return self.mfm().molids()

    def ne_stream(self):
        return self.mfm().ne_stream()

    def X_stream(self, chunksize=10000):
        Xchunk = self.mfm().X_stream(chunksize=chunksize)
        if self.folder is None:
            return Xchunk
        return self.folder.fold(Xchunk)


##################################################
# Command line generation for workload split
##################################################

def cl(step=46, for_what='rdkf'):
    """Generates example command lines for feature generation and munging.
     Parameters:
       - step: in how many steps does the work need to be divided
       - for_what: specify which command will be used;
                   one of (ecfps|rdkfs)
    """
    starts = range(step)
    if for_what == 'ecfps':
        for start, fcfp in product(starts, (True, False)):
            runid = 'fcfp=%r__start=%d__step=%d' % (fcfp, start, step)
            destfile = '~/tdtmalaria/data/rdkit/ecfps/from_workers/%s.weirdfps' % runid
            logfile = '~/tdtmalaria/data/rdkit/ecfps/from_workers/%s.log' % runid
            print 'PYTHONPATH=.:$PYTHONPATH python2 -u malaria/features.py ecfps ' \
                  '--start %d --step %d --output-file %s %s &>%s' %\
                  (start, step, destfile, '--fcfp' if fcfp else '', logfile)
    if for_what == 'rdkfs':
        for start, mols in product(starts, ('lab', 'unl', 'scr')):
            runid = 'rdkdescs__mols=%s__start=%d__step=%d' % (mols, start, step)
            destfile = '~/tdtmalaria/data/rdkit/rdkfs/from_workers/%s.h5' % runid
            logfile = '~/tdtmalaria/data/rdkit/rdkfs/from_workers/%s.log' % runid
            print 'PYTHONPATH=.:$PYTHONPATH python2 -u malaria/features.py rdkf ' \
                  '--start %d --step %d --mols %s --output-file %s &>%s' %\
                  (start, step, mols, destfile, logfile)


##################################################
# ENTRY POINT
##################################################


if __name__ == '__main__':

    dall = detect_duplicate_features(transductive=True)
    exit(69)

    parser = argh.ArghParser()
    parser.add_commands([cl, ecfps, ecfps_mp, munge_ecfps, rdkfs])
    parser.dispatch()