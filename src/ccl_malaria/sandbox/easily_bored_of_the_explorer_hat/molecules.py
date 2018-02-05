# coding=utf-8
"""Molecule i/o and preprocessing."""
from collections import defaultdict
import gzip
import os.path as op
import random

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem

from amazonsec.model.vowpal import VWOscailAdaptor, vw_from_porpoise
from ccl_malaria import MALARIA_DATA_ROOT
from ccl_malaria.dataio import read_unlabelled_smiles, read_labelled_smiles
from minioscail.common.evaluation import fast_auc
from thesis_code.descriptors.jcompoundmapper import JCompoundMapperCLIDriver
from ccl_malaria.rdkit_utils import explain_circular_substructure


MALARIA_VOWPAL_POC_DATA = op.join(MALARIA_DATA_ROOT, 'vowpal', 'rdkit-ecfp100.vw.gz')
MALARIA_VOWPAL_POC_FEATURES = op.join(MALARIA_DATA_ROOT, 'vowpal', 'rdkit-ecfp100.features.pkl')
MALARIA_VOWPAL_POC_DATA_RECODED = op.splitext(MALARIA_VOWPAL_POC_DATA)[0] + '.recoded.gz'


def features_from_vw(vwfile=MALARIA_VOWPAL_POC_DATA):
    """Extract the features from a simple VW file.
    N.B. At the moment we assume there is only one unnamed namespace.
    """
    with gzip.open(vwfile) if vwfile.endswith('gz') else open(vwfile) as reader:
        features = set()
        for line in reader:
            features.update([feature.split(':')[0] for feature in line.split('|')[1].split()])
        return features


def vw_poc_features():
    try:
        import cPickle as pickle
    except ImportError:
        import pickle
    if not op.exists(MALARIA_VOWPAL_POC_FEATURES):
        features = features_from_vw()
        with open(MALARIA_VOWPAL_POC_FEATURES, 'w') as writer:
            pickle.dump(sorted(features), writer, pickle.HIGHEST_PROTOCOL)
    with open(MALARIA_VOWPAL_POC_FEATURES) as reader:
        return pickle.load(reader)


def vwinput_from_file(vwfile=MALARIA_VOWPAL_POC_DATA):
    with gzip.open(vwfile) if vwfile.endswith('gz') else open(vwfile) as reader:
        return [x.strip() for x in reader.readlines()]


# 26258685 <- VW 1 pass
# 25953119 <- Ours set
#  2209406 <- RDKIT
#  2140435 <- Pybel recoded

def f2i(features):
    return {f: i for i, f in enumerate(sorted(features))}


def vw_recode00rollo(vwfile=MALARIA_VOWPAL_POC_DATA, dest=None):
    if dest is None:
        dest = op.splitext(vwfile)[0] + '.recoded.gz'
    recoding_shitza = f2i(vw_poc_features())
    # print recoding_shitza
    with gzip.open(vwfile) as reader, gzip.open(dest, 'w') as writer:
        for line in reader:
            meta, features = line.split('|')
            new_features = []
            for feature in sorted(features.split()):
                feature, value = feature.split(':')
                new_features.append('%d:%s' % (recoding_shitza[feature] + 1, value))
            writer.write('%s| %s\n' % (meta, ' '.join(new_features)))

# vw_recode00rollo()


def pybelshitza():
    import pybel
    features = vw_poc_features()
    pybel_feats = set()
    for i, feature in enumerate(features):
        if not i % 10000:
            print(i)
        pybel_feats.add(pybel.readstring('smi', feature).write('can'))
    return pybel_feats


def entropy_mistery():
    X = vwinput_from_file()

    # shuffle
    random.seed(0)
    random.shuffle(X)
    with gzip.open('/home/santi/mistery.gz', 'w') as writer:
        for x in X:
            writer.write(x + '\n')
    print('Let coding measure entropy and artifacts!!!')


def cross_val_10minutes():
    def labels(lines):
        def label_from_line(line):
            meta = line.split('|')[0].split()
            if len(meta) == 1:
                return np.nan
            return float(meta[0])
        return np.array(map(label_from_line, lines))
    X = vwinput_from_file()
    # X = vwinput_from_file(MALARIA_VOWPAL_POC_DATA_RECODED)
    # shuffle
    random.seed(0)
    random.shuffle(X)
    num_folds = 10
    for fold in range(num_folds):
        # train/test split
        train = [X[i] for i in range(len(X)) if not (i % num_folds == fold)]
        test = [X[i] for i in range(len(X)) if (i % num_folds == fold)]
        print('Writing...')
        # with gzip.open('/home/santi/%d.train.gz' % fold, 'w') as writer:
        #     for x in train:
        #         writer.write(x + '\n')
        # with gzip.open('/home/santi/%d.test.gz' % fold, 'w') as writer:
        #     for x in test:
        #         writer.write(x + '\n')
        train_labels = labels(train)
        train_labels[train_labels == -1] = 0
        train_labelled = ~np.isnan(train_labels)
        test_labels = labels(test)
        test_labels[test_labels == -1] = 0
        test_labelled = ~np.isnan(test_labels)
        print('NumTrain=%d; NumTest=%d' % (len(train), len(test)))
        # noinspection PyStringFormat
        print('TrainNumbers: pos=%d, neg=%d, unk=%d' % (np.sum(train_labels == 1),
                                                        np.sum(train_labels == 0),
                                                        np.sum(np.isnan(train_labels))))
        # noinspection PyStringFormat
        print('TestNumbers: pos=%d, neg=%d, unk=%d' % (np.sum(test_labels == 1),
                                                       np.sum(test_labels == 0),
                                                       np.sum(np.isnan(test_labels))))

        # Model setup
        vw = VWOscailAdaptor(vw_from_porpoise(bits=22, passes=5, l1=1E-8, l2=1E-8))
        # l1=1E-6, quadratic='::', cubic=':::',

        print('Training...')
        vw.train(train)
        print('Predicting and evaluating...')

        train_scores = vw.scores(train)[:, 1][train_labelled]
        train_auc = fast_auc(train_labels[train_labelled], train_scores)

        test_scores = vw.scores(test)[:, 1][test_labelled]
        test_auc = fast_auc(test_labels[test_labelled], test_scores)

        print('TrainAUC=%.2f, TestAUC=%.2f' % (train_auc, test_auc))
        print('-' * 80)


def isomeric_smiles_shitza():
    for molid, smiles in read_unlabelled_smiles():
        # print >> sys.stderr, smiles
        mol = Chem.MolFromSmiles(smiles)
        Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol)
        AllChem.UFFOptimizeMolecule(mol)
        Chem.AssignStereochemistry(mol)
        rdksmiles = Chem.MolToSmiles(mol, isomericSmiles=True)
        # print smiles
        # print rdksmiles
        if '@' in smiles and '@' not in rdksmiles or '@' not in smiles and '@' in rdksmiles:
            print('\t ******')
            print('\t\t', smiles)
            print('\t\t', rdksmiles)
        if '@' in smiles:
            print('\t\t', smiles)
            print('\t\t', rdksmiles)
        #   exit(77)
        # assert Chem.MolToSmiles(Chem.MolFromSmiles(smiles)) == Chem.MolToSmiles(Chem.MolFromSmiles(rdksmiles))
        # print Chem.MolToInchi(Chem.MolFromSmiles(rdksmiles))
        # print >> sys.stderr, Chem.MolToInchi(mol)
        # print >> sys.stderr, pybel.readstring('smi', rdksmiles).write('inchi')


if __name__ == '__main__':

    # TOTAL = 305568
    # false 293608
    # ambiguous 10432
    # true 1528
    # confirmed 1524
    # NA 304044
    # false and NA 293517
    # ambiguous and NA 10188
    # true and NA 339

    # There are no ID duplicates
    # TODO: check for molecular duplicates
    def jcmp():

        # jcpm = JCompoundMapperCLIDriver(
        #     jcompoundmapperjar='/home/santi/Build/--chem/jcompoundmapper-code/bin/jCMapperCLI.jar')
        jcpm = JCompoundMapperCLIDriver(
            jcompoundmapperjar='/home/santi/Downloads/jCMapperCLI.jar')
        jcpm.fingerprint('/home/santi/onlypos.sdf', '/home/santi/mistery.txt',
                         output_format='FULL_TAB_UNFOLDED', fingerprint='LSTAR')  # output_format='STRING_PATTERNS',


    # https://github.com/JohnLangford/vowpal_wabbit/wiki/Input-format
    def mols2vw(output_file):
        with gzip.open(output_file, 'w') as writer:
            for i, (molid, _, _, cl, _, smiles) in enumerate(read_labelled_smiles()):
                if not i % 1000:
                    print(i)
                mol = Chem.MolFromSmiles(smiles)
                if mol is None:
                    print('Failed molecule %s: %s' % (molid, smiles))
                    continue  # Just report it...
                info = {}
                fp = AllChem.GetMorganFingerprint(mol, 100, bitInfo=info).GetNonzeroElements()

                # FCFP (useFeatures=True)
                # Do not use bond type (useBondTypes=False)

                counts = defaultdict(int)
                for bit_descs in info.values():
                    for bd in bit_descs:
                        counts[explain_circular_substructure(mol, bd[0], bd[1])] += 1

                # All to VW format
                features = ['%s:%d' % (sub, count) for sub, count in sorted(counts.items())]
                label = '1 ' if cl == 'true' else '-1 ' if cl == 'false' else ''
                writer.write('%s%s| %s\n' % (label, molid, ' '.join(features)))

    mols2vw(MALARIA_VOWPAL_POC_DATA)
# 2 -> ECFP4, 3 -> ECFP6...
# TODO: is there any way to avoid rdkit computing the hash?
#       other than generating ourselves the strings back...
