import gzip
import os
import os.path as op
from collections import defaultdict
try:
    import cPickle as pickle
except ImportError:
    import pickle
import h5py
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem.Draw import MolDrawing
from ccl_malaria import MALARIA_EXPS_ROOT, MALARIA_DATA_ROOT
from ccl_malaria.misc import home
from ccl_malaria.storage_experiments import memmaps


def mols_having_this_feature(feature, vw_training_set):
    molids = []
    with gzip.open(vw_training_set, 'r') as reader:
        for line in reader:
            feats = set([f.split(':')[0] for f in line.split('|')[1].split()])
            if feature in feats:
                molids.append(line.split('|')[0].split()[-1])
    return molids


def get_mols_from_molids(molids):
    lm = memmaps.lab_mols_memmapped()
    return [lm.mol(molid) for molid in molids]


def plot_mols_and_substructure(feat, mols, molids, folder, legends=None):
    patt = Chem.MolFromSmarts(feat)
    for mol, molid in zip(mols, molids):
        m = mol.GetSubstructMatch(patt)
        # By default the RDKit colors atoms by element in depictions.
        # We can turn this off by replacing the element dictionary in MolDrawing
        MolDrawing.elemDict = defaultdict(lambda: (0, 0, 0))
        Draw.MolToImageFile(mol, op.join(folder, molid + '.png'), size=(400, 400), highlightAtoms=m,
                            legend=legends[molid])
    # It seems that the highlighting is not working with grid images??
    # image = Draw.MolsToGridImage(mols, molsPerRow=min(len(mols), 5), subImgSize=(400,400),
    #                             legends=molids, highlightAtoms=matchings)
    # return image


def mols2activity(molids, vw_file):
    molid_activity = {}
    with gzip.open(vw_file, 'r') as reader:
        for line in reader:
            instance_info = line.split('|')[0]
            molid = instance_info.split()[-1]
            if molid in molids:
                molid_activity[molid] = instance_info.split()[0]
    return molid_activity


def compare_selected_feats(pickled_files, hd5file_with_descs=op.join(home(), 'labrdkf.h5'), topN=10):
    top_feat = []
    with h5py.File(hd5file_with_descs, mode='r') as h5:
        fnames = h5['fnames'][:]
        for fi in pickled_files:
            with open(fi, 'r') as reader:
                feat_imp = sorted((zip(fnames, pickle.load(reader))), key=lambda x: x[1], reverse=True)
                top_feat.append(feat_imp[:topN])
    return top_feat[2]


if __name__ == '__main__':
    # vw_result_file = op.join(MALARIA_EXPS_ROOT, 'output_features.txt')
    # with open(vw_result_file, 'r') as reader:
    #     reader.readline()
    #     feat = reader.readline().split()[0][1:]
    #     print feat
    #     molids = mols_having_this_feature(feat, op.join(MALARIA_DATA_ROOT, 'vowpal', 'rdkit-ecfp100_weighted.vw.gz'))
    #     print molids
    #     mols = get_mols_from_molids(molids)
    #     direct = op.join(MALARIA_DATA_ROOT, 'plots', feat)
    #     if not op.exists(direct):
    #         os.makedirs(direct)
    #     plot_mols_and_substructure(feat, mols, molids, direct)
    #     #image.save('/home/flo/image_chorra.png')

    print(compare_selected_feats([op.join(MALARIA_EXPS_ROOT, 'ERT_4000trees_feat_imp.pkl'),
                                  op.join(MALARIA_EXPS_ROOT, 'RF_4000trees_feat_imp.pkl'),
                                  op.join(MALARIA_EXPS_ROOT, 'GBC_500trees_feat_imp.pkl')]))
