import gzip
from itertools import product
import random
import os.path as op
import shutil
import glob

import argh
import pandas as pd
import numpy as np
from rdkit.Chem import AllChem
from rdkit import Chem

from amazonsec.model.vowpal import VWOscailAdaptor, vw_from_porpoise
from minioscail.common.evaluation import fast_auc
from sandbox.molecules import vwinput_from_file
from ccl_malaria import MALARIA_DATA_ROOT, MALARIA_EXPS_ROOT
from misc import home


def add_weights_per_class(vw_file, weights, output_file):
    """
    Transform the input file to add a weight information. The weight will correspond to a class. Ex: class 1 has
    weight 0.5 and class -1 has weight 1
    """
    nb_lines = 0
    with gzip.open(vw_file, 'r') as reader:
        with gzip.open(output_file, 'w') as writer:
            for line in reader:
                info = line.strip().split(' ')
                label = info[0] if info[0] == '-1' or info[0] == '1' else ''
                if label == '-1' or label == '1':
                    rest = ' '.join(info[1:])
                    new_line = ' '.join([label, str(weights[label]), rest])
                else:
                    new_line = line.strip()
                writer.write(new_line)
                nb_lines += 1
                writer.write('\n')
    print(nb_lines)


def get_confirmed_mols(original_file, output_file):
    w = Chem.SDWriter(output_file)
    df = pd.read_csv(original_file, compression='gzip', sep='\t')
    confirmatory = df['Pf3D7_pEC50']
    molids = df['SAMPLE']
    molecules = df['Canonical_Smiles']
    for i, (molid, smiles) in enumerate(zip(molids, molecules)):
        if not np.isnan(confirmatory[i]) and confirmatory[i] is not None:
            print(confirmatory[i])
            mol = AllChem.MolFromSmiles(smiles)
            if mol is not None:
                AllChem.Compute2DCoords(mol)
                mol.SetProp('ec50', str(confirmatory[i]))
                mol.SetProp('_Name', molid)
                w.write(mol)


def cross_validate_vw_classification_model(training_set, seed=0, num_folds=10, b=22, l1=1e-8, l2=1e-8, verbose=True,
                                           working_dir=None):
    def labels(lines):
        def label_from_line(line):
            meta = line.split('|')[0].split()
            if len(meta) == 1:
                return np.nan
            return float(meta[0])
        return np.array(map(label_from_line, lines))
    X = vwinput_from_file(training_set)
    # shuffle
    random.seed(seed)
    random.shuffle(X)
    final_test_scores = []
    real_test = []
    for fold in range(num_folds):
        train = [X[i] for i in range(len(X)) if not (i % num_folds == fold)]
        test = [X[i] for i in range(len(X)) if (i % num_folds == fold)]
        train_labels = labels(train)
        train_labels[train_labels == -1] = 0
        train_labelled = ~np.isnan(train_labels)  # to remove the unlabelled examples
        test_labels = labels(test)
        test_labels[test_labels == -1] = 0
        test_labelled = ~np.isnan(test_labels)  # to remove the unlabelled examples
        if verbose:
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
        vw = vw_from_porpoise(loss='logistic', bits=b, passes=5, l1=l1, l2=l2, working_dir=working_dir)
        wd = vw.working_directory
        vw = VWOscailAdaptor(vw)

        print('Training...')
        vw.train(train)
        print('Predicting and evaluating...')

        train_scores = vw.scores(train)[:, 1][train_labelled]
        train_auc = fast_auc(train_labels[train_labelled], train_scores)

        test_scores = vw.scores(test)[:, 1][test_labelled]
        final_test_scores.append(test_scores)
        real_test.append(test_labels[test_labelled])
        test_auc = fast_auc(test_labels[test_labelled], test_scores)

        print('TrainAUC=%.2f, TestAUC=%.2f' % (train_auc, test_auc))
        print('-' * 80)
        # Removing at the end the working directory for not to fill up the /tmp in flo's machine
        if working_dir is None:
            shutil.rmtree(wd)
    total_predicted = np.array([item for sublist in final_test_scores for item in sublist])
    total_real = np.array([item for sublist in real_test for item in sublist])
    return total_predicted, total_real, fast_auc(total_real, total_predicted)


def compare_models_by_cv(training_set1, training_set2):
    set1_test_scores, set1_real_scores, set1_auc = cross_validate_vw_classification_model(training_set1)
    set2_test_scores, set2_real_scores, set2_auc = cross_validate_vw_classification_model(training_set2)
    return set1_auc, set2_auc
# --- Result: AUC non weighted = 0.91329171082074678,
# ---         AUC weighted = 0.95329387358421824


def which_regularization(l1, l2):
    l1 = float(l1)
    l2 = float(l2)
    _, _, auc = cross_validate_vw_classification_model(op.join(MALARIA_DATA_ROOT, 'vowpal', 'rdkit-ecfp100.vw.gz'),
                                                       l1=l1, l2=l2, verbose=False, num_folds=10,
                                                       working_dir=op.join(MALARIA_EXPS_ROOT,
                                                                           'regularization_vw',
                                                                           'l1=%f_l2=%f' % (l1, l2)))
    log = op.join(home(), 'auc_l1=%g_l2=%g.log' % (l1, l2))
    with open(log, 'w') as writer:
        writer.write(str(auc))
    print('AUC obtained for l1 = %f and l2 = %f: %.2f' % (l1, l2, auc))


def command_line():
    l1s = [0, 1e-10, 1e-8, 1e-6, 1e-4, 1e-3, 1e-2, 0.05, 0.1]
    l2s = [0, 1e-10, 1e-8, 1e-6, 1e-4, 1e-3, 1e-2, 0.05, 0.1]
    for l1, l2 in product(l1s, l2s):
        print('python2 -u malaria/vowpal_wabbit_utils.py which-regularization %g %g &> ~/regul_%g_%g.log' %
              (l1, l2, l1, l2))


def interpret_results_regularization(resdir=None):
    res_dict = {}
    l1s = set()
    l2s = set()
    for f in glob.glob(op.join(resdir, '*.log')):
        fname = op.basename(f)
        if fname.startswith('auc'):
            # noinspection PyTypeChecker
            l1 = fname.partition('_l1=')[2].partition('_l2=')[0]
            l1s.add(float(l1))
            # noinspection PyTypeChecker
            l2 = fname.partition('_l2=')[2].partition('.log')[0]
            l2s.add(float(l2))
            with open(f, 'r') as reader:
                auc = reader.readline().strip()
                res_dict[(l1, l2)] = auc
    print(res_dict.keys())
    l1s = sorted(l1s)
    l2s = sorted(l2s)
    print('l1\l2\t' + '\t'.join(map(lambda x: '%g' % x, l2s)))
    for i, l1 in enumerate(l1s):
        l1 = '%g' % l1
        print(l1 + '\t' + '\t'.join([res_dict[(l1, '%g' % l2s[j])] for j in range(len(l2s))]))
    # Let's print that nicely
    return res_dict


if __name__ == '__main__':
    #
    # add_weights_per_class('/home/flo/Proyectos/malaria/data/chorrrada.vw.gz', {'-1':0.005, '1':1.0, '':1.0},
    #                       '/home/flo/Proyectos/malaria/data/chorrrada_weighted.vw.gz')
    # print get_confirmed_mols(MALARIA_ORIGINAL_TRAINING_SET,
    #                          '/home/santi/confirmed_data.sdf')
    # cross_validate_vw_classification_model(op.join(MALARIA_DATA_ROOT, 'vowpal', 'rdkit-ecfp100.vw.gz'))
    # print compare_models_by_cv(op.join(MALARIA_DATA_ROOT, 'vowpal', 'rdkit-ecfp100.vw.gz'),
    #                            op.join(MALARIA_DATA_ROOT, 'vowpal', 'rdkit-ecfp100_weighted.vw.gz'))
    parser = argh.ArghParser()
    parser.add_commands([which_regularization, command_line, interpret_results_regularization])
    parser.dispatch()
