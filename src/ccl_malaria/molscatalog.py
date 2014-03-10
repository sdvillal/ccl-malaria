# coding=utf-8
"""Read the original smiles data, create convenient molecules catalogs."""
from collections import Counter
import gzip
import itertools
import mmap
import os.path as op
from time import time

import numpy as np
from rdkit import Chem

from minioscail.common.misc import download, ensure_dir, is_iterable
from ccl_malaria import info, warning
from ccl_malaria import MALARIA_DATA_ROOT, MALARIA_ORIGINAL_DATA_ROOT, MALARIA_INDICES_ROOT
from ccl_malaria.rdkit_utils import to_rdkit_mol


####
####--- Lazy data download
####

def _labelled_smiles_file():
    return download('http://www.tdtproject.org/uploads/6/8/2/0/6820495/malariahts_trainingset.txt.gz',
                    op.join(MALARIA_ORIGINAL_DATA_ROOT, 'malariahts_trainingset.txt.gz'),
                    info=info)


def _unlabelled_smiles_file():
    return download('http://www.tdtproject.org/uploads/6/8/2/0/6820495/malariahts_externaltestset.txt',
                    op.join(MALARIA_ORIGINAL_DATA_ROOT, 'malariahts_externaltestset.txt'),
                    info=info)


def _screening_smiles_file():
    return download('http://downloads.emolecules.com/ordersc/2014-01-01/parent.smi.gz',
                    op.join(MALARIA_ORIGINAL_DATA_ROOT, '20140101-parent.smi.gz'),
                    info=info)


def _malaria_screening_sdf_file():
    return download('http://downloads.emolecules.com/ordersc/2014-01-01/parent.sdf.gz',
                    op.join(MALARIA_ORIGINAL_DATA_ROOT, '20140101-parent.sdf.gz'),
                    info=info)


#####
#####---Data in, relabelling
#####


def relabel_pEC50(sample, threshold=5):
    """Whenever present we use pEC50 confirmatory data to set the label of compounds.
    With a threshold of 5, this produces the following changes:
      - 296 molecules move from positive to negative
      - 192 molecules move from ambiguous to negative
      - 52 molecules move from ambiguous to positive
      - 4 molecules move from negative to positive
    In some papers (e.g. Tropsha's) they us 2 micromolar instead of our 10 as a threshold.
    """
    sample_id, Pf3D7_ps_green, Pf3D7_ps_red, Pf3D7_ps_hit, Pf3D7_pEC50, smiles = sample
    if not np.isnan(Pf3D7_pEC50):
        Pf3D7_ps_hit = 'true' if Pf3D7_pEC50 >= threshold else 'false'
    return sample_id, Pf3D7_ps_green, Pf3D7_ps_red, Pf3D7_ps_hit, Pf3D7_pEC50, smiles


def read_labelled_smiles(relabel_func=relabel_pEC50):
    """Returns a generator of the molecules used to develop the model for the competition.
    Each element yielded is a tuple containing in this order:
      - molid
      - green fluorescence intensity
      - red fluorescence intensity
      - hit? [true|false|ambiguous]
      - confirmatory pEC50
      - smiles
    The original file contains a header:
      SAMPLE Pf3D7_ps_green Pf3D7_ps_red Pf3D7_ps_hit Pf3D7_pEC50 Canonical_Smiles

    For example:
    >>> generator = read_labelled_smiles()
    >>> print generator.next()
    ('SJ000241685-1', -5.42492139537131, -22.9358967463118, 'false', nan, 'Cc1ccccc1c2nsc(SCC(=O)Nc3ccc(Br)cc3)n2')
    >>> print generator.next()
    ('SJ000241765-1', 62.8825610415562, 28.7677982751592, 'ambiguous', nan, 'O=C(Nc1nc(cs1)c2ccccn2)c3ccc(cc3)C#N')
    >>> ids = {'SJ000241685-1', 'SJ000241765-1'}
    >>> for row in generator:
    ...    ids.add(row[0])
    >>> num_molecules = 305569 - 1  # As given by zcat | wc -l
    >>> print 'Are there repeated ids in the labelled set? %s.' % 'No' if len(ids) == num_molecules else 'Yes'
    Are there repeated ids in the labelled set? No.
    """
    with gzip.open(_labelled_smiles_file()) as reader:
        reader.next()  # Skip header
        for line in reader:
            sample_id, Pf3D7_ps_green, Pf3D7_ps_red, Pf3D7_ps_hit, Pf3D7_pEC50, smiles = line.split()
            sample = (sample_id,
                      float(Pf3D7_ps_green),
                      float(Pf3D7_ps_red),
                      Pf3D7_ps_hit,  # true, false or ambiguous
                      float(Pf3D7_pEC50) if Pf3D7_pEC50 != 'NA' else np.nan,   # can be NA or a float
                      smiles)
            yield relabel_func(sample)


def _count_label_changes(relabel_func=relabel_pEC50):
    """Returns information on the number of label changes induced by the relabelling function."""
    c = Counter()
    for raw, rel in itertools.izip(read_labelled_smiles(lambda x: x),
                                   read_labelled_smiles(relabel_func)):
        class_raw = raw[3]
        class_rel = rel[3]
        c[class_raw + class_rel] += 1
    return c


def read_labelled_only_smiles():
    """Returns a generator [(molid, smiles)] of the labelled molecules."""
    for molid, _, _, _, _, smiles in read_labelled_smiles():
        yield molid, smiles


def read_unlabelled_smiles():
    """Returns a generator [(molid, smiles)] of the molecules to rank for the competition.
    The original file contains a header:
      SAMPLE CANONICAL_SMILES

    For example:
    >>> generator = read_unlabelled_smiles()
    >>> print generator.next()
    ('SJ000551065-1', 'Cc1ccc(OCC(O)CNC2CCCCC2)cc1')
    >>> print generator.next()
    ('SJ000551074-1', 'CCCCN(CCCC)CC(O)COc1ccccc1[N+](=O)[O-]')
    >>> ids = {'SJ000241685-1', 'SJ000241765-1'}
    >>> for molid, _ in generator:
    ...    ids.add(molid)
    >>> num_molecules = 1057 - 1  # As given by wc -l
    >>> print 'Are there repeated ids in the unlabelled set? %s.' % 'No' if len(ids) == num_molecules else 'Yes'
    Are there repeated ids in the unlabelled set? No.
    """
    with open(_unlabelled_smiles_file()) as reader:
        reader.next()  # Skip header
        for line in reader:
            sample_id, smiles = line.split()
            yield (sample_id, smiles)


def read_screening_smiles():
    """Returns a generator [(molid, smiles)] of the purchasable molecules to screen for the competition.
    The original file contains a header:
      isosmiles parent_id

    For example:
    >>> generator = read_screening_smiles()
    >>> print generator.next()
    ('10019', 'COC(=O)C12NCC3(C2(C)CCC3C1)C')
    >>> print generator.next()
    ('10023', 'Oc1noc(c1)C1CCNCC1')
    >>> ids = {'10019', '10023'}
    >>> for molid, _ in generator:
    ...    ids.add(molid)
    >>> num_molecules = 5488504 - 1  # As given by zcat | wc -l
    >>> print 'Are there repeated ids in the screening set? %s.' % 'No' if len(ids) == num_molecules else 'Yes'
    Are there repeated ids in the screening set? No.
    """
    with gzip.open(_screening_smiles_file()) as reader:
        reader.next()  # Ignore header
        for line in reader:
            smiles, molid = line.split()
            yield (molid, smiles)


def read_smiles_ultraiterator():
    """Iterates over all pairs (molid, smiles), first labelled, then unlabelled, finally screening."""
    return itertools.chain(read_labelled_only_smiles(),
                           read_unlabelled_smiles(),
                           read_screening_smiles())


# 3-letter codes to molecules streams...
MOLS2MOLS = {
    'all': read_smiles_ultraiterator,
    'lab': read_labelled_only_smiles,
    'unl': read_unlabelled_smiles,
    'scr': read_screening_smiles
}


#####
#####--- Faster molid->label and molids present in the original files.
#####

def molid_lab():
    """Returns a map {molid->label} for the labelled molecules."""
    cachefile = op.join(MALARIA_INDICES_ROOT, 'lab_molids_label.txt')
    if not op.isfile(cachefile):
        with open(cachefile, 'w') as writer:
            writer.write('\n'.join(['%s %s' % (molid, label[0].lower())
                                    for molid, _, _, label, _, _ in read_labelled_smiles()]))
    with open(cachefile) as reader:
        return {line.rstrip().split()[0]: line.rstrip().split()[1] for line in reader}


def _molids_cache(molidit, cachefile):
    """Caches the molids in text files."""
    if not op.isfile(cachefile):
        with open(cachefile, 'w') as writer:
            writer.write('\n'.join([molid for molid, _ in molidit]))
    with open(cachefile) as reader:
        return [line.rstrip() for line in reader]


def lab_molids():
    """Returns a list with the labelled molids as they appear in the original file."""
    return _molids_cache(read_labelled_only_smiles(), op.join(MALARIA_INDICES_ROOT, 'lab_molids.txt'))


def unl_molids():
    """Returns a list with the unlabelled molids as they appear in the original file."""
    return _molids_cache(read_unlabelled_smiles(), op.join(MALARIA_INDICES_ROOT, 'unl_molids.txt'))


def scr_molids():
    """Returns a list with the screening molids as they appear in the original file."""
    return _molids_cache(read_screening_smiles(), op.join(MALARIA_INDICES_ROOT, 'scr_molids.txt'))


#####
#####--- Memmapped catalog of rdkit molecules
#####
#
# Some numbers:
#   - Total number of molecules: 5795127
#   - Total number of molecules failed to read by rdkit: 46
#   - Total size of all ToBinay combined: 2227099601 bytes
#   - Mean size of ToBinary: 384.305572768 bytes
#   - Max size of ToBinary: 2933 bytes
#   - Min size of ToBinary: 0 bytes
#
# Just for fun, this could be better done with a full-blown DB (e.g. using the postgres rdkit cartridge).
# But we do not want postgres as a dependency for this competition...
######

class MemMappedMols(object):
    # TODO: maybe periodically reopen the memmapped handle to avoid memory leaks
    #       probably with implementin reopen functionality wired in _memmapped_data
    #       Should not be necessary:
    #       http://stackoverflow.com/questions/20713063/writing-into-a-numpy-memmap-still-loads-into-ram-memory
    # TODO: make this a context manager
    # TODO: allow to do regular, no memmapped I/O with a "memmap" flag

    def __init__(self, root_dir):
        """Quick random access to collections of molecules in disk, using molids.
        Caveat: great for random access, not suitable for streaming purposes.
                All read molecules stay in memory until (all) handles to this memmap is closed.
        """
        # Where the index resides...
        self._root = root_dir
        ensure_dir(self._root)

        # Index {molid -> (start, numbytes)}
        self._molids_file = op.join(self._root, 'molids.txt')
        self._coords_file = op.join(self._root, 'coords.npy')
        self._molids = None
        self._coords = None
        self._molid2coords = None

        # The serialized molecules
        self._data_file = op.join(self._root, 'molsdata')
        self._filehandle = None
        self._molsdata = None

    def has_catalog(self):
        return op.isfile(self._molids_file) and op.isfile(self._coords_file) and op.isfile(self._data_file)

    def _memmapped_data(self):
        """Returns a memmapped version of the molecules data."""
        if self._molsdata is None:
            self._filehandle = open(self._data_file, 'r+b')
            self._molsdata = mmap.mmap(self._filehandle.fileno(), 0)
        return self._molsdata

    def molids(self):
        """Returns the molids present in the catalog, sorted in the order they are written in the data file."""
        if self._molids is None:
            with open(self._molids_file) as reader:
                self._molids = [line.strip() for line in reader]
        return self._molids

    def _index(self):
        """Returns the index {molid->(start, num_bytes)}."""
        if self._molid2coords is None:
            self._molid2coords = {}
            self._coords = np.load(self._coords_file)
            for i, molid in enumerate(self.molids()):
                self._molid2coords[molid] = self._coords[i]
        return self._molid2coords

    def contains_mol(self, molid):
        """Does this catalog contain the molecule with index molid?."""
        return molid in self._index

    def mol(self, molid):
        """Return the molecule associated to the molid, or None if no such molecule is in the catalog."""
        base, length = self._index().get(molid, (-1, 0))
        if base < 0:
            return None
        return Chem.Mol(self._memmapped_data()[base:base+length])

    def mols(self, molids):
        """Returns a list of molecules (None for molecules not in the catalog) associated to the molids."""
        return map(self.mol, molids)

    def close(self):
        """Close the open resourcer."""
        self._molsdata.close()

    def save_from_smiles_iterator(self, it):
        """Creates the catalog from the (molid, smiles) iterator, possibly overwriting the present files."""
        molids = []
        coords = []
        base = 0
        with open(op.join(self._root, 'molsdata'), 'w') as writer:
            for molid, smiles in it:
                mol = to_rdkit_mol(smiles, molid=molid)
                if mol is None:
                    molids.append(molid)
                    coords.append((-1, 0))
                else:
                    moldata = mol.ToBinary()
                    molids.append(molid)
                    coords.append((base, len(moldata)))
                    base += len(moldata)
                    writer.write(moldata)
        with open(self._molids_file, 'w') as writer:
            for molid in molids:
                writer.write(molid + '\n')
        np.save(self._coords_file, np.array(coords))


def unl_molid2mol_memmapped_catalog():
    """Returns a memmapped catalog {molid->rdkit mol} for the unlabelled (competition) molecules."""
    return MemMappedMols(op.join(MALARIA_DATA_ROOT, 'rdkit', 'mols', 'unl'))


def lab_molid2mol_memmapped_catalog():
    """Returns a memmapped catalog {molid->rdkit mol} for the labelled molecules."""
    return MemMappedMols(op.join(MALARIA_DATA_ROOT, 'rdkit', 'mols', 'lab'))


def scr_molid2mol_memmapped_catalog():
    """Returns a memmapped catalog {molid->rdkit mol} for the unlabelled (screening) molecules."""
    return MemMappedMols(op.join(MALARIA_DATA_ROOT, 'rdkit', 'mols', 'scr'))


def build_benchmark_check_rdkmols_catalog(mmapdir, molit=read_labelled_only_smiles, checks=False, overwrite=False):
    """Builds a memmapped catalog {molid->rdkbytes} from a (molid, smiles) iterator.
    tests it and compares to sequential recreation of the molecules from smiles.
    """

    # Build the catalog
    info('Building %s catalog...' % mmapdir)
    start = time()
    mmm = MemMappedMols(mmapdir)
    if not overwrite and mmm.has_catalog():
        info('Already computed, skipping.')
    else:
        mmm.save_from_smiles_iterator(molit())
    info('Time taken to build the memmapped file: %.2f seconds' % (time() - start))

    if not checks:
        return

    # Load the catalog
    mmms = MemMappedMols(mmapdir)

    # Lame benchmark - memmapped contiguous
    info('Benchmarking contiguous memmap reading')
    start = time()
    molcount = 0
    for molid in mmms.molids():
        mmms.mol(molid)
        molcount += 1
    info('Time taken to read the memmapped %d mols (contiguous): %.2f seconds' % (molcount, time() - start))

    info('Benchmarking random memmap reading')
    start = time()
    molcount = 0
    for molid in set(mmms.molids()):
        mmms.mol(molid)
        molcount += 1
    info('Time taken to read the memmapped %d mols (random): %.2f seconds' % (molcount, time() - start))

    # Lame benchmark - from smiles
    info('Benchmarking reading from the original file')
    start = time()
    molcount = 0
    for _, smiles in molit():
        Chem.MolFromSmiles(smiles)
        molcount += 1
    info('Time taken to read the smiled %d mols: %.2f seconds' % (molcount, time() - start))

    # Exhaustive linear test that all mols are correctly stored
    info('Making sure that all is OKish')
    for molid, smiles in molit():
        emol = Chem.MolFromSmiles(smiles)
        if emol is None:
            if not mmms.mol(molid) is None:
                warning('Molecule %s with original smiles %s should not be parsed from the binary store' %
                        (molid, smiles))
        else:
            if not Chem.MolToSmiles(emol) == Chem.MolToSmiles(mmms.mol(molid)):
                warning('Molecule %s with original smiles %s do not reconstruct properly: \n\t(%s != %s)' %
                        (molid, smiles, Chem.MolToSmiles(emol), Chem.MolToSmiles(mmms.mol(molid))))
    info('All is OKish')


def catalog_malaria_mols(overwrite=False, checks=False):
    """Bootstrap the malaria catalogs."""

    to_catalog = (
        (op.join(MALARIA_DATA_ROOT, 'rdkit', 'mols', 'unl'), read_unlabelled_smiles),
        (op.join(MALARIA_DATA_ROOT, 'rdkit', 'mols', 'lab'), read_labelled_only_smiles),
        (op.join(MALARIA_DATA_ROOT, 'rdkit', 'mols', 'scr'), read_screening_smiles),
    )

    for path, molit in to_catalog:
        build_benchmark_check_rdkmols_catalog(path, molit=molit, checks=checks, overwrite=overwrite)

    info('ALL DONE')


#####
#####--- Fast retrieval of information from molids
#####


class MalariaCatalog(object):

    def __init__(self):
        super(MalariaCatalog, self).__init__()
        self._lm = None        # Labelled molids
        self._um = None        # Unlabelled molids
        self._sm = None        # Screening molids
        self._lm2index = None  # Labelled molid -> index (in the original file)
        self._um2index = None  # Unlabelled molid -> index (in the original file)
        self._sm2index = None  # Screening molid -> index (in the original file)
        self._m2index = None   # Any molid -> index (as per concatenation of lab+unl+scr)
        self._m2label = None   # Any molid -> label ('t'->positive, 'n'->negative, 'a'->ambiguous, None->unlabelled)
        self._lm2mol = lab_molid2mol_memmapped_catalog()   # Labelled molid -> rdkit mol
        self._um2mol = unl_molid2mol_memmapped_catalog()   # Unlabelled molid -> rdkit mol
        self._sm2mol = scr_molid2mol_memmapped_catalog()   # Screening molid -> rdkit mol
        self._m2pec50 = None   # molid -> ec50
        self._m2smiles = None  # molid -> original smiles

    def lab(self):
        """Returns a tuple with the molids of the labelled molecules in the order of the original file."""
        if self._lm is None:
            self._lm = tuple(lab_molids())
        return self._lm

    def unl(self):
        """Returns a tuple with the molids of the unlabelled molecules in the order of the original file."""
        if self._um is None:
            self._um = tuple(unl_molids())
        return self._um

    def scr(self):
        """Returns a tuple with the molids of the "screening" molecules in the order of the original file."""
        if self._sm is None:
            self._sm = tuple(scr_molids())
        return self._sm

    def label(self, molid, as01=False):
        """Returns the label of a molid.
        The label can be:
          - 't': positive (active against plasmodium falciparum)
          - 'n': negative (non active against plasmodium falciparum)
          - 'a': ambiguous (contradictory assay results)
          - None: unlabelled (no experimental information on activity)
        If as01 is True, then the possible labels are:
          - 1 if it is a positive
          - 0 if it is a negative
          - nan if ambiguous or unlabelled
        """
        if self._m2label is None:
            self._m2label = molid_lab()
        label = self._m2label.get(molid, None)
        if as01:
            return 1 if label == 't' else 0 if label == 'f' else np.nan
        return label

    def molid2label(self, molid, as01=False):
        """Same as label."""
        return self.label(molid, as01=as01)

    def labels(self, molids, as01=False, asnp=True):
        """Returns the labels of the molids, as per the label function."""
        labels = [self.label(molid, as01=as01) for molid in molids]
        return np.array(labels) if asnp else labels

    def molids2labels(self, molids, as01=False, asnp=True):
        """Same as labels."""
        return self.labels(molids, as01=as01, asnp=asnp)

    def molid2pec50(self, molid):
        if self._m2pec50 is None:
            self._m2pec50 = {sample_id: Pf3D7_pEC50 for
                             sample_id, _, _, _, Pf3D7_pEC50, _ in read_labelled_smiles()}
        return self._m2pec50.get(molid, np.nan)

    def molids2pec50s(self, molids):
        return map(self.molid2pec50, molids)

    def molid2smiles(self, molid):
        if self._m2smiles is None:
            self._m2smiles = {sample_id: smiles for
                              sample_id, smiles in read_smiles_ultraiterator()}
        return self._m2smiles.get(molid, None)

    def molids2smiless(self, molids):
        return map(self.molid2smiles, molids)

    def lab2i(self, molid):
        """Returns the index of a labelled molecule in the original file."""
        if self._lm2index is None:
            self._lm2index = {molid: i for i, molid in enumerate(self.lab())}
        return self._lm2index.get(molid, None)

    def unl2i(self, molid):
        """Returns the index of an unlabelled molecule in the original file."""
        if self._um2index is None:
            self._um2index = {molid: i for i, molid in enumerate(self.unl())}
        return self._um2index.get(molid, None)

    def scr2i(self, molid):
        """Returns the index of a screening molecule in the original file."""
        if self._sm2index is None:
            self._sm2index = {molid: i for i, molid in enumerate(self.scr())}
        return self._sm2index.get(molid, None)

    def any2i(self, molid):
        """Returns the index of a screening molecule in the original file."""
        if self._m2index is None:
            self._m2index = {molid: i
                             for i, molid in enumerate(self.lab() + self.unl() + self.scr())}
        return self._m2index.get(molid, None)

    def is_lab(self, molid):
        """Returns True iff the molid corresponds to a labelled molecule."""
        return self.lab2i(molid) is not None

    def is_unl(self, molid):
        """Returns True iff the molid corresponds to a unlabelled molecule (of the competition set)."""
        return self.unl2i(molid) is not None

    def is_scr(self, molid):
        """Returns True iff the molid corresponds to a unlabelled screening molecule."""
        return self.scr2i(molid) is not None

    def is_known(self, molid):
        """Returns True iff the molid corresponds to a molecule in the catalog."""
        return self.any2i(molid) is not None

    def num_lab(self):
        """Returns the number of labelled molecules."""
        return len(self.lab())

    def num_unl(self):
        """Returns the number of unlabelled (competition) molecules."""
        return len(self.unl())

    def num_scr(self):
        """Returns the number of unlabelled (screening) molecules."""
        return len(self.scr())

    def num_known(self):
        """Returns the number of molecules in the catalog."""
        return self.num_lab() + self.num_unl() + self.num_scr()

    def is_positive(self, molid):
        """Return True iff the molecule is positive."""
        return self.label(molid) == 't'

    def is_negative(self, molid):
        """Return True iff the molecule is negative."""
        return self.label(molid) == 'n'

    def is_ambiguous(self, molid):
        """Return True iff the molecule is ambiguous."""
        return self.label(molid) == 'a'

    def _molid2mol_catalog(self, molid):
        """Returns a {molid->rdkit_mol} map for the molid (mmap at the moment, but this should be configurable)."""
        if self.is_lab(molid):
            return self._lm2mol
        if self.is_unl(molid):
            return self._um2mol
        return self._sm2mol

    def molid2mol(self, molid):
        """Returns an rdkit molecule for the molid (barenaked, without any property set) or None."""
        return self._molid2mol_catalog(molid).mol(molid)

    def molids2mols(self, molids):
        """Returns a list of rdkit molecules (None also possible) for the molids."""
        if not is_iterable(molids):
            molids = (molids,)
        return map(self.molid2mol, molids)


if __name__ == '__main__':
    def init(checks=False):
        """Downloads the original files if necessary and builds the molecules catalogue."""
        # Indices from molids
        molid_lab()
        lab_molids()
        unl_molids()
        scr_molids()
        # Rdkit indices
        catalog_malaria_mols(checks=checks)

    def doctest():
        """Runs tests."""
        import doctest
        doctest.testmod(verbose=True)

    import argh
    parser = argh.ArghParser()
    parser.add_commands([init, doctest])
    parser.dispatch()