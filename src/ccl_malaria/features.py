# coding=utf-8
"""Feature generation and munging."""
from __future__ import print_function, division
from future.utils import string_types

import time
from array import array
from glob import glob
from operator import itemgetter
import os.path as op
from collections import defaultdict
from itertools import islice, product, chain, zip_longest
import gzip
import os
import argh

import h5py
from joblib import Parallel, delayed
import numpy as np
import pandas as pd
from natsort import natsorted
from scipy.sparse import coo_matrix, csr_matrix, vstack
from scipy.stats import rankdata
from sklearn.utils.murmurhash import murmurhash3_32 as murmur

from ccl_malaria import info
from ccl_malaria import MALARIA_DATA_ROOT
from ccl_malaria.molscatalog import read_smiles_ultraiterator, MOLS2MOLS, MalariaCatalog
from ccl_malaria.rdkit_utils import RDKitDescriptorsComputer, to_rdkit_mol, morgan_fingerprint

from minioscail.common.config import Configurable
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


_END_MOLID = None  # Marker for end of iteration


def _molidsmiles_it(start=0, step=46, mols=None, processor=None, log_each=500):
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
        mols = read_smiles_ultraiterator()  # This could benefit quite a bit of async IO
    t0 = time.time()
    for molindex, (molid, smiles) in enumerate(islice(mols, start, None, step)):
        if log_each > 0 and molindex > 0 and not molindex % log_each:
            taken = time.time() - t0
            info('Molecule %d (%.2fs, %.2fmols/s)' %
                 (molindex, taken, molindex / taken))
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
    h5.create_dataset('fnames', data=[n.encode("ascii", "ignore") for n in fnames])
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
        except Exception as ex:
            info('Failed molecule %s: %s (%s)' % (molid, smiles, str(ex)))
            descs[ne, :] = [np.nan] * nf

    return process


def rdkfs(start=0, step=1, mols='all', output_file=None):
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
    if isinstance(mols, string_types):
        mols = MOLS2MOLS[mols]()
    _molidsmiles_it(start=start, step=step,
                    mols=mols,
                    processor=_rdkfeats_writer(output_file))


def rdkfs_mp(num_jobs=1, mols='all', dest_dir=None):
    if dest_dir is None:
        dest_dir = ensure_dir(MALARIA_DATA_ROOT, 'rdkit', 'rdkfs', 'from_workers')
    Parallel(n_jobs=num_jobs)(
        delayed(rdkfs)
        (start=start, step=num_jobs,
         mols=mols,
         output_file=op.join(dest_dir, 'rdkfs__%s__start=%d__step=%d.h5' % (
             mols, start, num_jobs
         )))
        for start in range(num_jobs)
    )


def _h52pandas(h5_path):
    """
    Returns a pandas dataframe with all the data in the hdf5 file.
    Columns are features and molids are in the index.
    """
    with h5py.File(h5_path, mode='r') as h5:
        return pd.DataFrame(
            data=h5['rdkdescs'][()],
            columns=h5['fnames'][()].astype(str),
            index=h5['molids'][()].astype(str)
        )


def _h5s(src_dir=None, num_jobs=28, mol_set_id='all', strict=False):
    if src_dir is None:
        src_dir = op.join(MALARIA_DATA_ROOT, 'rdkit', 'rdkfs', 'from_workers')
    h5s = []
    missing = []
    for i in range(num_jobs):
        h5src = op.join(src_dir,
                        'rdkfs__%s__start=%d__step=%d.h5' % (mol_set_id, i, num_jobs))
        if strict and not op.isfile(h5src):
            missing.append(h5src)
        h5s.append(h5src)
    if strict and missing:
        raise Exception('Missing the results from some workers (%r)' % missing)
    return h5s


def merge_rdkfs(src_dir=None, num_jobs=28, src_mol_set_id='all',  # coordinates of inputs
                strict=True,
                dest_dir=op.join(MALARIA_DATA_ROOT, 'rdkit', 'rdkfs'),
                **dest_mol_sets):

    # This does not stream copy, but for simplicity holds everything in memory
    # Will use ~16GB RAM for the screening dataset

    if not dest_mol_sets:
        dest_mol_sets[src_mol_set_id] = None

    # Merge everything (in memory, takes at most total + 1 temporary space)
    dfs = {}
    info('Merging hdf5 files')
    for h5 in _h5s(src_dir=src_dir,
                   num_jobs=num_jobs,
                   mol_set_id=src_mol_set_id,
                   strict=strict):
        df = _h52pandas(h5)
        for dset, molids in dest_mol_sets.items():
            # Get the subset of molecules in this dataframe
            if molids is not None:
                molids = df.index.intersection([molids])
            else:
                molids = df.index
            if dset not in dfs:
                dfs[dset] = df.loc[molids]
            else:
                dfs[dset] = dfs[dset].append(df.loc[molids])

    # Respect requested molecule order
    def reorder(dset):
        if dest_mol_sets[dset] is None:
            return dfs[dset]
        return dfs[dset].loc[dest_mol_sets[dset]].copy()
    info('Reordering dataframes')
    dfs = {dset: reorder(dset) for dset in dfs}

    # Store back (if requested, for compatibility with rushy competition code)
    if dest_dir is not None:
        info('Saving back to clumsy competition format')
        dest_dir = ensure_dir(dest_dir)
        for dset, df in dfs.items():
            dest_file = op.join(dest_dir, '%srdkfs.h5' % dset)
            with h5py.File(dest_file, mode='w', dtype=np.float32) as h5:
                h5.create_dataset('fnames', data=[n.encode("ascii", "ignore") for n in df.columns])
                h5.create_dataset('molids', data=[n.encode("ascii", "ignore") for n in df.index])
                h5.create_dataset('rdkdescs', data=df.values, compression='lzf')

    # Return
    return dfs


def munge_rdkfs():
    from ccl_malaria.molscatalog import lab_molids, unl_molids, scr_molids
    merge_rdkfs(lab=lab_molids(), unl=unl_molids(), scr=scr_molids())


# noinspection PyAbstractClass
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
                molids = h5['molids'][()].astype(str)
                fnames = h5['fnames'][()].astype(str)
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
        self._populate()
        return self._fnames

    def ne_stream(self):
        if self.dset == 'lab' or self.dset == 'amb':
            return self.X().shape[0]  # Caveat: disallows streaming from labelled.
        with h5py.File(self._h5) as h5:
            return h5['rdkdescs'].shape[0]

    def X_stream(self, chunksize=10000):
        if self._streaming_in_ram or self.dset == 'lab' or self.dset == 'amb':  # Dodgy lab and amb, remove corner case
            for start in range(0, self.ne_stream(), chunksize):
                yield self.X()[start:start+chunksize]
        else:
            h5 = h5py.File(self._h5)['rdkdescs']
            for start in range(0, self.ne_stream(), chunksize):
                yield h5[start:start+chunksize]
                # FIXME: bug, does not skip some of the molecules that need to be skipped,
                # remove ambiguous and nans
            h5.close()
    # FIXME: streaming is clunky and error prone for lab / amb, base on molids

    def df(self):
        return _h52pandas(self._h5)


##################################################
# Computation of ECFPs and FCFPs
##################################################

_MALARIA_ECFPS_DIR = op.join(MALARIA_DATA_ROOT, 'rdkit', 'ecfps')
_MALARIA_ECFPS_PARALLEL_RESULTS_DIR = op.join(_MALARIA_ECFPS_DIR, 'from_workers')


def _ecfp_writer(output_file=None, max_radius=200, fcfp=False,
                 write_centers=True, write_radius=True, write_bit_keys=True,
                 write_error_msgs=False):
    """
    Returns a (molid, smiles) processor that computes ecfps on the smiles and stores them in a text file.

    By default, the saved file (format) contains quite a bit or redundant or useless information.

    Parameters:
    -----------
    output_file : string
      path to the file where the fingerprints will be written.
      This file will be overwritten and gzipped.

    max_radius : int, default 200
      the maximum radius for the found circular substructures.

    fcfp : bool, default False
      whether to generate FCFP or ECFP like fingerprints.

    write_centers : bool, default True
      whether to also save all the centers of each substructure or not.

    write_radius : bool, default True
      whether to also save all the radius generating each substructure or not.

    write_bit_keys : bool, default True
      whether to also save all the corresponding rdkit hash for each feature or not.

    write_error_msgs : bool, default False
      whether to write also error messages to the file

    Returns:
      - a processor function ready to be used as a parameter to `_molidsmiles_it`.


    File format
    -----------
    We call the format generated "weird fp format".
    The file starts with a header like this:
      v1\t[|C|R|K|CR|CK|RK|CRK]
    For each molecule for which fingerprinting works we have a line like this:
      molid\t[cansmi count [center radius hash]+\t]+\n
      For example:
        'v1\tCRK\tmol1\tC 4 1 1 33 2 1 33\tB 1 1 1 22\tBC 1 1 2 1789\n'
      Parses to:
        - v1: version tag of the format
        - CRK: for each feature we additionally store C=Center, R=Radius, K=bit Key
        - mol1: the molecule id
        - C 2 1 1 33 2 1 33: feature explained by smiles C appears twice
                              centered at atoms 1 and 2, with radius 1, with hash 33
        - B 3 1 1 22: feature explained by smiles D appears once at atom 3, radius 1,
                      hashes to 22
        - BC 1 1 2 1789: feature explained by smiles BC appears once at atom 1, radius 2,
                         hashes to 1789.
    If there is an error with the molecule, a line like this is written:
        FAIL\tmolid\n
    Optionally, error lines can end with the error message
        FAIL\tmolid\tCannot kekulize mol\n

    Note that parsing this format is quite consuming, so munge this information
    into more efficient formats for reading, specially if (a subset of) it is to
    be read repeatly.
    """
    writer = gzip.open(output_file, 'wt')

    def process(molid, smiles):
        if molid is _END_MOLID:
            writer.close()
            return
        extra_info = '%s%s%s' % ('C' if write_centers else '',
                                 'R' if write_radius else '',
                                 'K' if write_bit_keys else '')
        header = 'v1\t%s\t%s' % (extra_info, molid)
        try:
            cansmi2fptinfo = morgan_fingerprint(smiles,
                                                max_radius=max_radius,
                                                fcfp=fcfp)
            features_strings = []
            for cansmiles, feature_info in cansmi2fptinfo.items():
                feature_string = '%s %d' % (cansmiles, len(feature_info))
                if 0 < len(extra_info):
                    extra_info = []
                    for center, radius, bit_key in feature_info:
                        if write_centers:
                            extra_info.append(str(center))
                        if write_radius:
                            extra_info.append(str(radius))
                        if write_bit_keys:
                            extra_info.append(str(bit_key))
                    feature_string += ' ' + ' '.join(extra_info)
                features_strings.append(feature_string)
            writer.write('%s %s\n' % (header, '\t'.join(features_strings)))
        except Exception as ex:
            info('Failed molecule %s: %s; exception: %s' % (molid, smiles, str(ex)))
            if write_error_msgs:
                writer.write('FAIL\t%s\t%s\n' % (header, str(ex).replace('\n', ' => ')))
            writer.write('FAIL\t%s\n' % header)
    return process


def parse_weirdfpt_v1_line(line, check_dumb_tab=False):
    # Check for error line
    if 'FAILED' in line:
        molid, _, reason = line.strip().split('\t')
        return molid, None
    # Dumb workaround to broken lines due to exceptions containing newlines
    # This is already fixed in the writer
    if not line.startswith('\t'):
        return None, None
    # Check for starting tab
    if check_dumb_tab and not line.startswith('\t') and len(line.strip()) > 0:
        # This dumb tab can be removed when we convert competition data to the new format
        # or do not care anymore about the TDT competition data.
        raise ValueError('At the moment, line should start with an spurious tab (%r)' % line)
    # Split
    line = line.strip()
    # Header + molid
    version, extra_info, rest = line.split('\t', 2)
    # Check version
    if version != 'v1':
        raise ValueError('Wrong format version: %r' % version)
    molid, rest = rest.split(' ', 1)

    # --- Parse features
    if True:
        # Check extra info description
        valid_extra_infos = {'', 'C', 'R', 'K', 'CR', 'CK', 'RK', 'CRK'}  # partition set of {'', 'C', 'R', 'K'}
        if extra_info not in valid_extra_infos:
            raise ValueError('Wrong extra info spec %r, must be one of %r' % (extra_info, valid_extra_infos))
        extra_info = [{'C': 'center', 'R': 'radius', 'K': 'rdkit_hash'}[ei]
                      for ei in extra_info]
        parsed_features = []
        while rest:
            s = rest.split('\t', 1)
            feature, rest = s if len(s) == 2 else (s[0], '')
            ffields = feature.split()
            cansmi = ffields[0]
            count = int(ffields[1])
            feature = dict(
                cansmi=cansmi,
                count=count,
            )
            centers = ffields[2:]
            if (0 == len(centers) and extra_info) or len(centers) % len(extra_info) != 0:
                raise ValueError('Corrupt extra info for molecule %s, feature %r' % (molid, cansmi))
            # feature['infos'] = ta.array([int(c) for c in centers])
            feature['infos'] = array.array('I', [int(c) for c in centers])
            parsed_features.append(feature)

        return molid, parsed_features
    return molid

# OF COURSE THIS IS EXPLOSION AND WE ANYWAY HAVE THIS IN THE MOLECULE
# IS RDKIT ALL INT32?


def _grouper(iterable, n, fill_value=None):
    """Collect data into fixed-length chunks or blocks"""
    # https://docs.python.org/3/library/itertools.html
    #   _grouper('ABCDEFG', 3, 'x') --> ABC DEF Gxx"
    # like `partition` in (cy)toolz
    # see also `partition_all` (a tad faster in cytoolz)
    args = [iter(iterable)] * n
    return zip_longest(*args, fillvalue=fill_value)


def ext_based_open(path, mode='rt'):
    if path.endswith('.gz'):
        return gzip.open(path, mode=mode)
    return open(path, mode=mode)


class IterationSignal:
    def __init__(self, name):
        self.name = name


IGNORE_RESULT = IterationSignal('IGNORE_RESULT')
END_FILE = IterationSignal('END_FILE')
END_ALL = IterationSignal('END_ALL')


def wfpts_files(src_dir=None):
    if src_dir is None:
        src_dir = op.join(_MALARIA_ECFPS_PARALLEL_RESULTS_DIR)
    return natsorted(glob(op.join(src_dir, '*.weirdfps.gz')))


def line_iterator(visitor,
                  file_paths=None,
                  file_start=0, file_step=1,
                  line_start=0, line_step=1):
    if file_paths is None:
        file_paths = wfpts_files()
    for file_path in islice(file_paths, file_start, None, file_step):
        with ext_based_open(file_path, 'rt') as reader:
            line_iterator = islice(reader, line_start, None, line_step)
            for line in line_iterator:
                result = visitor(line)
                if result is IGNORE_RESULT:
                    continue
                elif result is END_FILE:
                    yield result
                    break
                elif result is END_ALL:
                    yield result
                    return
                else:
                    yield result


def identity(x):
    return x


def up_to_n(visitor, n=1000):
    # noinspection PyDefaultArgument
    def up_to(x, counter=[0]):
        if counter[0] == n:
            return END_ALL
        counter[0] += 1
        return visitor(x)
    return up_to


def sharded_parse(x, shard=0, num_shards=127):
    molid, feats = parse_weirdfpt_v1_line(x)
    if murmur(molid) % num_shards == shard:
        print('Not ignored')
        return molid, feats
    return IGNORE_RESULT


def to_new(dest_dir='/home/santi/testit'):
    ensure_dir(dest_dir)
    fpt_writer = ext_based_open(op.join(dest_dir, 'fingerprints.bin'), 'wb')  # TODO: ensure it is closed
    cansmi_writer = ext_based_open(op.join(dest_dir, 'features.txt'), 'wt')
    molids_writer = ext_based_open(op.join(dest_dir, 'molids.txt'), 'wt')
    failed_molids_writer = ext_based_open(op.join(dest_dir, 'molids_failed.txt'), 'wt')
    h2i = {}

    def to_new(x):
        molid, features = parse_weirdfpt_v1_line(x)
        pos = fpt_writer.tell()
        if molid is None:
            return  # Should not happen, but we where saving verbatim error messages with newlines
        if features is None:
            failed_molids_writer.write(molid + '\n')
            return
        array.array('I', [len(features)]).tofile(fpt_writer)
        for feature in features:
            cansmi = feature['cansmi']
            count = feature['count']
            infos = feature['infos']
            assert len(infos) / count == 3
            h3 = murmur(cansmi, seed=0), murmur(cansmi, seed=1), murmur(cansmi, seed=2)
            if h3 not in h2i:
                h2i[h3] = len(h2i)
                cansmi_writer.write(cansmi)
                cansmi_writer.write('\n')
            array.array('I', [h2i[h3], count]).tofile(fpt_writer)
            infos.tofile(fpt_writer)
        molids_writer.write(molid + ' ' + str(pos) + '\n')  # TODO: also save position in binary file
    return to_new


def interpret_fpt(x):
    nfeats = x[0]
    base = 1
    feats = []
    for f in range(nfeats):
        cansmi_hash, count = x[base:base+2]
        occurrences = np.frombuffer(x[base+2:base+2+3*count], dtype=x.typecode).reshape(-1, 3)
        feats.append([cansmi_hash, occurrences])
        base = base+2+3*count
    return feats


def parse_new(dest_dir='/home/santi/testit'):
    fpt = ext_based_open(op.join(dest_dir, 'fingerprints.bin'), 'rb')
    molids = ext_based_open(op.join(dest_dir, 'molids.txt'), 'rt')

    def read_one_fpt():
        x = array.array('I')
        x.fromfile(fpt, 1)              # number of features (redundant once read)
        for _ in range(x[0]):
            x.fromfile(fpt, 2)          # hash, count_in_mol
            x.fromfile(fpt, 3 * x[-1])  # (center, radius, rdkit_hash) * count
        return x

    for _ in molids:
        yield read_one_fpt()


def _read_full_file(x, path):
    """Reads the full contentes of file path into array x."""
    with open(path, 'rb') as reader:
        reader.seek(0, 2)
        size = reader.tell()
        reader.seek(0, 0)
        if size % x.itemsize != 0:
            raise Exception('Truncated file')
        x.fromfile(reader, size // x.itemsize)
        return x


class FlyWeightDict(dict):
    # Kinda FlyWeight pattern, without the magic on types
    # (e.g. not defining str subclass that on construction gets the canonical / intrinsics)...
    #   http://www.boost.org/doc/libs/1_60_0/libs/flyweight/doc/tutorial/index.html
    # N.B. rdkit uses the pattern for Morgan fingerprinting, as implemented in boost
    #   http://www.boost.org/doc/libs/1_60_0/libs/flyweight/doc/tutorial/index.html
    def __getitem__(self, k):
        try:
            return super(FlyWeightDict, self).__getitem__(k)
        except KeyError:
            self[k] = k
            return k


def parse_wfps(start=0, step=4):
    t0 = time.time()
    hash2smi = defaultdict(set)  # maybe set is too much mem, usually cardinality will be very low
    smi2hash = defaultdict(set)  # maybe set is too much mem, usually cardinality will be very low
    smis = FlyWeightDict()    # kinda flyweight
    hashes = FlyWeightDict()  # kinda flyweight
    wfpts = glob(op.join(_MALARIA_ECFPS_PARALLEL_RESULTS_DIR, '*.weirdfps.gz'))
    lines = 0
    for wfpt in islice(wfpts, start, None, step):  # Could parallelize too at the line level...
        info(wfpt)
        with gzip.open(wfpt, 'rt') as reader:
            for line in reader:
                if lines > 0 and lines % 10000 == 0:
                    taken = time.time() - t0
                    info('Parsed %d lines (%.2fs, %.2fmols/s), %d features, %d hashes, %s' %
                         (lines, taken, lines / taken, len(smis), len(hashes), wfpt))
                molid, feats = parse_weirdfpt_v1_line(line)
                if feats is not None:
                    for feat in feats:
                        cansmi = smis[feat['cansmi']]
                        for finfo in feat['infos']:
                            rdkit_hash = hashes[finfo['rdkit_hash']]
                            hash2smi[rdkit_hash].add(cansmi)
                            smi2hash[cansmi].add(rdkit_hash)
                lines += 1
    return hash2smi, smi2hash


def morgan(start=0, step=46, mols='lab', output_file=None, fcfp=True):
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
    if isinstance(mols, string_types):
        mols = MOLS2MOLS[mols]()
    ensure_dir(op.dirname(output_file))
    _molidsmiles_it(start=start, step=step,
                    mols=mols,
                    processor=_ecfp_writer(output_file=output_file, fcfp=fcfp))


def _molidsmiles_it_ecfp(output_file, start=0, step=46,
                         write_centers=True, write_radius=True, write_bit_keys=True,
                         fcfp=True,
                         log_each=5000):
    """Q&D variant to allow Parallel work (cannot pickle closures or reuse iterators...)."""
    processor = _ecfp_writer(output_file=output_file, fcfp=fcfp,
                             write_centers=write_centers,
                             write_radius=write_radius,
                             write_bit_keys=write_bit_keys)
    mols = read_smiles_ultraiterator()
    t0 = time.time()
    for molindex, (molid, smiles) in enumerate(islice(mols, start, None, step)):
        if log_each > 0 and molindex > 0 and not molindex % log_each:
            taken = time.time() - t0
            info('Molecule %d (%.2fs, %.2f mol/s)' % (molindex, taken, molindex / taken))
        processor(molid, smiles)
    processor(_END_MOLID, None)


def morgan_mp(num_jobs=1, dest_dir=None,
              no_write_centers=False, no_write_radius=True, no_write_bit_keys=True,
              no_ecfps=False,
              no_fcfps=False,
              log_each=5000):
    """Python-parallel computation of ECFPs.
    Parameters:
      - num_jobs: the number of threads to use (joblib semantics for negative numbers).
      - dest_dir: the directory to which the fingerprints will be written, in weird fp format(TM).
    """
    # Dest dir
    dest_dir = _MALARIA_ECFPS_PARALLEL_RESULTS_DIR if dest_dir is None else dest_dir
    ensure_dir(dest_dir)
    # What to compute: nothing, ECFPS (contains False), FCFPS (contains True)
    invariants = [] + ([False] if not no_ecfps else []) + ([True] if not no_fcfps else [])
    # Parallel run
    Parallel(n_jobs=num_jobs)(delayed(_molidsmiles_it_ecfp)
                              (start=start,
                               step=num_jobs,
                               output_file=op.join(dest_dir, 'all__fcfp=%r__start=%d__step=%d.weirdfps.gz' %
                                                   (fcfp, start, num_jobs)),
                               fcfp=fcfp,
                               write_centers=not no_write_centers,
                               write_radius=not no_write_radius,
                               write_bit_keys=not no_write_bit_keys,
                               log_each=log_each)
                              for start, fcfp in product(range(num_jobs), invariants))


def munge_morgan():

    # --- Step 1: put all these together in 3 files, lab, unl and scr.
    #       - ECFPs and FCFPs for the same mol are together
    #       - The order is the same as in the original file
    #       - Optionally delete the workers files

    def parse_weirdfpformat_legacy_line(line):
        """Returns a tuple (molid, [cansmi, count, [(center, radius)]+]+)."""
        # Legacy file format
        # ------------------
        #
        # The legacy, competition version "the weird fp format" looked like this
        # We call the format generated "weird fp format". On each line we have either:
        #     molid[\tcansmi count [center radius]+]+
        # or, if there is an error with the molecule:
        #     molindex\molid\t*FAILED*
        # It can be detected because the line does not start by a tab. Remove that ugly
        # tab when the competition data is not relevant anymore (or just rewrite that data
        # into the new format).
        def _parse_weird_feature(feature):
            vals = feature.split()
            cansmi = vals[0]
            count = int(vals[1])
            if len(vals) > 2:
                a = iter(vals[2:])
                centers = [(center, radius) for center, radius in zip(a, a)]
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
        """
        A data processor that takes weirdfp lines, chunk them in disk
        and then merge them sorted in a big file.
        It can be setup to be easy on memory usage (at the cost of doubling disk space usage).
        """
        def __init__(self, molid2i, root, prefix, data2molid, chunksize=10000):
            super(Chihuahua, self).__init__()
            self.chunksize = chunksize
            self.molid2i = molid2i
            self.num_mols = len(self.molid2i)
            self.temp_fns = [op.join(root, '%s-%d' % (prefix, base)) for base in range(0, self.num_mols, chunksize)]
            self.temp_files = [open(fn, 'w') for fn in self.temp_fns]
            self.data2molid = data2molid
            self.root = root
            self.prefix = prefix
            ensure_dir(self.root)

        def process(self, moldata):
            index = self.molid2i.get(self.data2molid(moldata), None)
            if index is None:
                return
            goes_to = int(index / self.chunksize)
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

    # ---Step 2: recode ECFPs and FCFPs from the file at step 1. After this:
    #  - ECFPs and FCFPs duplicates get merged.
    #  - A unique assignment for each substructure in the dataset
    #    to an int [0, ...] (column number) is generated.
    #  - A unique assignment for each molid in the dataset
    #    for wich Morgan DID NOT FAIL (row number) is generated.

    def ecfps_recode(dset='lab'):
        """Merges ECFPs and FCFPs into a single line and gets rid of the centers information if present."""
        with open(op.join(_MALARIA_ECFPS_DIR, dset)) as reader, \
                open(op.join(_MALARIA_ECFPS_DIR, dset + '.merged'), 'w') as writer:
            for ecfp in reader:
                fcfp = next(reader)
                molide, subse = parse_weirdfpformat_legacy_line(ecfp)
                molidf, subsf = parse_weirdfpformat_legacy_line(fcfp)
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

    # ---Step 3: write sparse matrices with the recoded information of step 2. After this:
    #      - We get a h5 file for each dataset, with a sparse matrix in CSR format.
    #      - Note that this is a memory intense procedure, can be done lightweight by using 2 passes.

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

    # --- Step 4: lame feature duplicate detection to tackle partially multicolliniarity
    MalariaFingerprintsManager(zero_dupes='lab').X()
    MalariaFingerprintsManager(zero_dupes='all').X()


# noinspection PyNoneFunctionAssignment
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
    for i in range(nf):
        xi = X.indices[X.indptr[i]:X.indptr[i+1]:]
        groups[xi.data].append(i)  # We should hash xi.data to make the dictionary easier on memory
        if verbose and i > 0 and not i % 1000000:
            info('%d of %d substructures hashed according to the molecules they pertain' % (i, nf))

    return groups.values()


def zero_columns(X, columns, zero_other=False):
    # N.B. always returns copy under current implementation
    X = X.tocsc()  # Duplicate memory!
    num_cols = X.shape[1]
    if zero_other:
        # Equivalent but 400 times faster than sorted(set(xrange(num_cols)) - set(columns))
        to_zero = np.ones(num_cols)
        to_zero[columns] = 0
        columns = np.where(to_zero)[0]
    for col in columns:  # This for is the hotspot ATM
        if col < num_cols:
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
        if zero_dupes and zero_dupes not in {'all', 'lab'}:
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
        """Returns a numpy array of substructures corresponding to columns in the matrix."""
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
        column = self.s2i(bit_or_smiles) if isinstance(bit_or_smiles, string_types) else bit_or_smiles
        molindices = self.XCSC().indices[self.XCSC().indptr[column]:self.XCSC().indptr[column+1]]
        return [self.i2m(index) for index in molindices]

    def m2i(self, molid):
        """Returns the row index for a molid (or None if the molid is unknown)."""
        # if self._m2i is None:
        #     with open(self._m2i_file) as reader:
        #         self._m2i = {m.rstrip(): i for i, m in enumerate(reader)}
        if self._m2i is None:
            # noinspection PyTypeChecker
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
                data = np.ones(len(indices), dtype=np.int8) if self._only01 else h5['data'][:]
                # bools are ints in np, smallest datatype is int8
                # TODO implicit sparse 1/0 (ala oscail-java)
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
                    # noinspection PyTypeChecker
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
            for start in range(0, self.ne_stream(), chunksize):
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
            for start in range(0, self.ne_stream(), chunksize):
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

    @property
    def original_file(self):
        return self._original_file


def fold_csr(X, folder, safe=True, binary=True, as_float=False):
    """A convenience method to fold a CSR sparse matrix using a folder (hashing + bucket assignment method)."""

    X = X.tocsr()
    if safe:
        X.eliminate_zeros()

    #
    # Is there a more efficient way of doing this for the whole matrix?
    # This is tempting:
    #   folded = murmur(X.indices) % fold_size
    #   X.indices = folded
    # But then the indices for each row can be non-unique and non-sorted
    #
    # Also, just remember this is much slower
    # if slow:
    #     cols = [np.unique(folder.assign_feature(X[row, :].indices))
    #             for row in range(X.shape[0])]  # Fold
    #
    # Let's do a lot of work here.
    # Move maybe this implementation down to numba/cython if ever a problem,
    # and then make it without intermediate arrays and all that heavy
    # machinery.
    #

    # Create a COO Matrix
    if not binary:
        rows = []
        cols = []
        data = []
        for row in range(X.shape[0]):
            # Cols that are non-zero for the row
            non_zero_cols = X.indices[X.indptr[row]:X.indptr[row + 1]]
            # Hash + fold the columns
            hashed_cols = folder.assign_feature(non_zero_cols)
            # There might have been collisions...
            unique_folded_cols = np.unique(hashed_cols)
            # Coordinate collisions
            # For folded coords like [11, 33, 11, 33, 11, 11], coords will be [0, 1, 0, 1, 0, 0]
            folded_data_coords = rankdata(hashed_cols, method='dense') - 1
            # Merge colliding data by summing
            folded_data = np.bincount(folded_data_coords, weights=X.data[X.indptr[row]:X.indptr[row + 1]])
            # We are done for this row
            rows.append(np.full(len(unique_folded_cols), fill_value=row, dtype=np.int))
            cols.append(unique_folded_cols)
            data.append(folded_data.astype(X.dtype if not as_float else np.float))
        # Merge data for all rows
        rows = np.hstack(rows)
        cols = np.hstack(cols)
        data = np.hstack(data)
    else:
        rows = []
        cols = []
        for row in range(X.shape[0]):
            # Cols that are non-zero for the row
            non_zero_cols = X.indices[X.indptr[row]:X.indptr[row + 1]]
            # Hash + fold the columns
            hashed_cols = folder.assign_feature(non_zero_cols)
            # There might have been collisions...
            unique_folded_cols = np.unique(hashed_cols)
            rows.append(np.full(len(unique_folded_cols), fill_value=row, dtype=np.int))
            cols.append(unique_folded_cols)
        # Merge data for all rows
        rows = np.hstack(rows)
        cols = np.hstack(cols)
        data = np.ones(len(rows), dtype=(np.float if as_float else np.bool))

    # COO -> CSR, return
    return coo_matrix((data, (rows, cols)), shape=(X.shape[0], folder.fold_size)).tocsr()


class MurmurFolder(Configurable):

    def __init__(self, seed=0, positive=True, fold_size=1023,
                 safe=True, as_binary=True, as_float=False,
                 save_map=True):
        """
        Hashes and folds substructures into features in R^d, possibly keeping track of what goes where.
        """

        #
        # TODO: allow these different modes:
        #   - fold to target density
        #   - fold to fixed number of bits per substructure
        #   - fold to a mask; allow counts, binary and "binary counts"
        #

        #
        # TODO: allow a fold_size auto mode, where the nearest prime is looked for
        # (e.g. use lookup table for common values and a brute force approach + primality (Miller-Rabin)
        # for values not in the lookup table)
        #

        #
        # TODO: fold online
        #

        super(MurmurFolder, self).__init__()
        self.seed = seed
        self.positive = positive
        self.fold_size = fold_size
        self.as_binary = as_binary
        self.as_float = as_float
        self._safe = safe
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
            for feature, key in zip(features, substructures):
                self._features[feature].add(key)
        return features

    def substructures_for_feature(self, feature):
        return sorted(self._features[feature])

    def fold(self, X):
        return fold_csr(X, self, safe=self._safe, binary=self.as_binary, as_float=self.as_float)

    def folded2unfolded(self):
        """Return a numpy array with bucket assignments for each original feature.
        That is csfold()[4] gives the fold for unfolded feature 4.
        """
        max_unfolded_feature = max(max(group) for group in self._features.values())
        # noinspection PyTypeChecker
        fold2unfold = np.empty(max_unfolded_feature + 1, dtype=np.int32)
        fold2unfold.fill(-1)
        for folded_feature, group in self._features.items():
            fold2unfold[list(group)] = folded_feature
        return fold2unfold

    def clear_cache(self):
        self._features = defaultdict(set)


# noinspection PyAbstractClass
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
            print('PYTHONUNBUFFERED=1 ccl-malaria features morgan '
                  '--start %d --step %d --output-file %s %s &>%s' %
                  (start, step, destfile, '--fcfp' if fcfp else '', logfile))
    if for_what == 'rdkfs':
        for start, mols in product(starts, ('lab', 'unl', 'scr')):
            runid = 'rdkdescs__mols=%s__start=%d__step=%d' % (mols, start, step)
            destfile = '~/tdtmalaria/data/rdkit/rdkfs/from_workers/%s.h5' % runid
            logfile = '~/tdtmalaria/data/rdkit/rdkfs/from_workers/%s.log' % runid
            print('PYTHONUNBUFFERED=1 ccl-malaria features rdkfs '
                  '--start %d --step %d --mols %s --output-file %s &>%s' %
                  (start, step, mols, destfile, logfile))


##################################################
# ENTRY POINT
##################################################


if __name__ == '__main__':

    parser = argh.ArghParser()
    parser.add_commands([cl, morgan, morgan_mp, munge_morgan, rdkfs, rdkfs_mp, munge_rdkfs])
    parser.dispatch()
