import os.path as op
import os
import gzip
import subprocess
import random

import numpy as np

from ccl_malaria import dataio
from ccl_malaria import MALARIA_DATA_ROOT, MALARIA_EXPS_ROOT
from sandbox import feature_selection_analysis as featsel
from sandbox.molecules import vwinput_from_file
from amazonsec.model.vowpal import VWOscailAdaptor, vw_from_porpoise
from vowpal_porpoise.sklearn import VW_Regressor


def create_vw_training_set(full_vw_training, new_file):
    confirmatory = {}
    for molinfo in dataio.read_labelled_smiles():
        if molinfo[4] != 'NA':
            confirmatory[molinfo[0]] = float(molinfo[4])
    with gzip.open(new_file, 'w') as writer, gzip.open(full_vw_training, 'r') as reader:
        for line in reader:
            molid = line.split('|')[0].split()[-1]
            if molid in confirmatory:
                new_line = str(confirmatory[molid]) + ' ' + molid + '|' + line.split('|')[1]
                writer.write(new_line)


# DOES NOT WORK WELL
# noinspection PyUnusedLocal
def run_vw(path_to_vw_varinfo, training_file, output_file, b=None, passes=None, l1=None, l2=None):
    default_command = [path_to_vw_varinfo, training_file]   # , ' >', output_file]
    if b is not None:
        default_command.insert(1, '-b ' + str(b))
    if passes is not None:
        default_command.insert(1, '-c --passes ' + str(passes))
    if l1 is not None:
        default_command.insert(1, '--l1 ' + str(l1))
    if l2 is not None:
        default_command.insert(1, '--l2 ' + str(l2))
    print(default_command)
    subprocess.check_call(default_command)


def plot_top_N_good(vw_result_file, training_file, N=10):
    with open(vw_result_file, 'r') as reader:
        reader.readline()  # skip the header
        feats = set()
        while len(feats) < N:
            feat = reader.readline().split()[0][1:]
            feats.add(feat)
    print(feats)
    for feat in feats:
        molids = featsel.mols_having_this_feature(feat, training_file)
        molid_activity = featsel.mols2activity(molids, training_file)
        mols = featsel.get_mols_from_molids(molids)
        direct = op.join(MALARIA_DATA_ROOT, 'plots', 'confirmatory_' + feat)
        if not op.exists(direct):
            os.makedirs(direct)
        try:
            featsel.plot_mols_and_substructure(feat, mols, molids, direct, legends=molid_activity)
        except Exception:
            print('could not plot')


def rmse(y, yhat):
    return np.sqrt(np.sum((y-yhat) ^ 2)/float(len(y)))


def cross_validate_model(num_folds=10, seed=0):
    def labels(lines):
        def label_from_line(line):
            meta = line.split('|')[0].split()
            if len(meta) == 1:
                return np.nan
            return float(meta[0])
        return np.array(map(label_from_line, lines))
    X = vwinput_from_file(op.join(MALARIA_DATA_ROOT, 'vowpal', 'confirmatory-rdkit-ecfp100.vw.gz'))
    # X = vwinput_from_file(MALARIA_VOWPAL_POC_DATA_RECODED)
    # shuffle
    random.seed(seed)
    random.shuffle(X)
    for fold in range(num_folds):
        # train/test split
        train = [X[i] for i in range(len(X)) if not (i % num_folds == fold)]
        test = [X[i] for i in range(len(X)) if (i % num_folds == fold)]
        print('Writing...')
        with gzip.open(op.join(MALARIA_EXPS_ROOT, 'confirmatory', '%d.train.gz' % fold), 'w') as writer:
            for x in train:
                writer.write(x + '\n')
        with gzip.open(op.join(MALARIA_EXPS_ROOT, 'confirmatory', '%d.test.gz' % fold), 'w') as writer:
            for x in test:
                writer.write(x + '\n')
        train_labels = labels(train)
        test_labels = labels(test)
        print('NumTrain=%d; NumTest=%d' % (len(train), len(test)))

        # Model setup
        vw = VWOscailAdaptor(vw_from_porpoise(bits=22, passes=5, l1=1E-8, l2=1E-8))
        # l1=1E-6, quadratic='::', cubic=':::',

        print('Training...')
        vw.train(train)
        print('Predicting and evaluating...')
        train_scores = vw.scores(train)[:, 1]
        train_rmse = rmse(train_labels, train_scores)

        test_scores = vw.scores(test)[:, 1]
        test_rmse = rmse(test_labels, test_scores)

        print('TrainRMSE=%.2f, TestRMSE=%.2f' % (train_rmse, test_rmse))
        print('-' * 80)


def CV_with_vowpal_porpoise(training_set):

    def get_data_into_proper_format():
        X = []
        y = []
        with gzip.open(training_set, 'r') as reader:
            for line in reader:
                label = line.split('|')[0].split()[0]
                y.append(float(label))
                features = line.split('|')[1].strip()
                feat_dict = {}
                for feat_count in features.split():
                    feat_dict[feat_count.split(':')[0]] = int(feat_count.split(':')[1])
                X.append(feat_dict)
        return X, y

    X, y = get_data_into_proper_format()
    Xtrain = X[:1]
    ytrain = y[:1]
    vw = VW_Regressor(loss='square', moniker='confirmed_data',
                      passes=10, silent=True, learning_rate=10).fit(Xtrain, ytrain)

    # scores = cross_validation.cross_val_score(vw, X, np.array(y), cv=10)

    return vw


def relabel(pec50_threshold=5):
    old_inactives_to_actives = 0
    old_actives_to_inactives = 0
    old_ambiguous_to_actives = 0
    old_ambiguous_to_inactives = 0
    nb_inactives = 0
    nb_actives = 0
    nb_ambiguous = 0
    for sample_id, bla, bli, cl, pec50, smiles in dataio.read_labelled_smiles():
        if cl == 'ambiguous' and np.isnan(pec50):
            nb_ambiguous += 1
        elif np.isnan(pec50):
            if cl == 'true':
                nb_actives += 1
            else:
                nb_inactives += 1
            # yield (sample_id, bla, bli, cl, pec50, smiles)
        elif pec50 >= pec50_threshold:
            nb_actives += 1
            if cl == 'false':
                print('Class for mol %s changed to 1 after reading confirmatory results' % sample_id)
                old_inactives_to_actives += 1
            elif cl == 'amibguous':
                old_ambiguous_to_actives += 1
            # yield (sample_id, bla, bli, 'true', pec50, smiles)
        else:
            nb_inactives += 1
            if cl == 'true':
                old_actives_to_inactives += 1
                print('Class for mol %s changed to 0 after reading confirmatory results' % sample_id)
            elif cl == 'ambiguous':
                old_ambiguous_to_inactives += 1
            # yield (sample_id, bla, bli, 'false', pec50, smiles)
    return (nb_actives, nb_inactives, nb_ambiguous, old_actives_to_inactives,
            old_inactives_to_actives, old_ambiguous_to_actives, old_ambiguous_to_inactives)


if __name__ == '__main__':
    # create_vw_training_set(op.join(MALARIA_DATA_ROOT, 'vowpal', 'rdkit-ecfp100.vw.gz'),
    #                        op.join(MALARIA_DATA_ROOT, 'vowpal', 'confirmatory-rdkit-ecfp100.vw.gz'))

    # run_vw('/home/flo/Proyectos/vowpal_wabbit/utl/vw-varinfo',
    #        op.join(MALARIA_DATA_ROOT, 'vowpal', 'confirmatory-rdkit-ecfp100.vw.gz'),
    #        op.join(MALARIA_EXPS_ROOT, 'confirmatory-rdkit-ecfp100_vw_default.txt'))

    # plot_top_N_good(op.join(MALARIA_EXPS_ROOT, 'confirmatory-rdkit-ecfp100_vw_default.txt'),
    #                 op.join(MALARIA_DATA_ROOT, 'vowpal', 'confirmatory-rdkit-ecfp100.vw.gz'))

    # cross_validate_model()
    # print CV_with_vowpal_porpoise(op.join(MALARIA_DATA_ROOT, 'vowpal', 'confirmatory-rdkit-ecfp100.vw.gz'))
    print(relabel())

#  ACT    INACT   AMB
# (1288, 294092, 10188)
# (1528, 293608, 10432)
# We lose 240 actives
# We gain 484 inactives
# We lose 244 ambiguous
