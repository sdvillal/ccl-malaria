"""Let's explore simple descriptor spaces in positive, negative, and test compounds.
Descriptors we want to explore:
- MW --> ok
- HBA, HBD --> ok
- TPSA --> ok
- logP
- nb of halogens
- aromaticity
- rings
- nb rotatable bonds  --> ok
- nb of double bonds
- Lipinski failures
...
"""
from collections import Counter
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
import numpy as np
import matplotlib.pyplot as plt
from ccl_malaria import dataio
from numpy import linspace, zeros, ones_like
import pandas as pd


def compute_descs_rdkit(mol):
    # We can always add more later on
    # noinspection PyProtectedMember
    MW = rdMolDescriptors._CalcMolWt(mol)
    HBA = rdMolDescriptors.CalcNumHBA(mol)
    HBD = rdMolDescriptors.CalcNumHBD(mol)
    TPSA = rdMolDescriptors.CalcTPSA(mol)
    aromatic_rings = rdMolDescriptors.CalcNumAromaticRings(mol)
    nb_heteroatoms = rdMolDescriptors.CalcNumHeteroatoms(mol)
    nb_rot_bonds = rdMolDescriptors.CalcNumRotatableBonds(mol)
    return MW, HBA, HBD, TPSA, aromatic_rings, nb_heteroatoms, nb_rot_bonds


def cdf(X):
    # Brute force brutal
    counter = Counter(X)
    xv = sorted(counter.items())
    sortedX = [x for x, _ in xv]
    cumsums = np.cumsum([v for _, v in xv]).astype(np.float)
    return sortedX, cumsums / cumsums[-1]


def plot_cdfs(xs, ys, labels, desc):
    ax = plt.axes()
    for x, y, l in zip(xs, ys, labels):
        plt.plot(x, y, label=l)
        plt.legend()
    ax.set_xlabel(desc)
    ax.set_ylabel('Cumulative distribution')
    plt.show()


def do_the_job():
    descriptors_positives = {'MW': [], 'HBA': [], 'HBD': [], 'TPSA': [], 'aromatic_rings': [], 'nb_heteroatoms': [],
                             'nb_rot_bonds': []}
    descriptors_negatives = {'MW': [], 'HBA': [], 'HBD': [], 'TPSA': [], 'aromatic_rings': [], 'nb_heteroatoms': [],
                             'nb_rot_bonds': []}
    descriptors_ambiguous = {'MW': [], 'HBA': [], 'HBD': [], 'TPSA': [], 'aromatic_rings': [], 'nb_heteroatoms': [],
                             'nb_rot_bonds': []}
    descriptors_test = {'MW': [], 'HBA': [], 'HBD': [], 'TPSA': [], 'aromatic_rings': [], 'nb_heteroatoms': [],
                        'nb_rot_bonds': []}
    print('Computing descriptors for the training set...')
    for molinfo in dataio.read_labelled_smiles():
        mol = Chem.MolFromSmiles(molinfo[-1])
        if mol is None:
            continue
        descs = compute_descs_rdkit(mol)
        if molinfo[3] == 'true':
            descriptors_positives['MW'].append(descs[0])
            descriptors_positives['HBA'].append(descs[1])
            descriptors_positives['HBD'].append(descs[2])
            descriptors_positives['TPSA'].append(descs[3])
            descriptors_positives['aromatic_rings'].append(descs[4])
            descriptors_positives['nb_heteroatoms'].append(descs[5])
            descriptors_positives['nb_rot_bonds'].append(descs[6])
        elif molinfo[3] == 'false':
            descriptors_negatives['MW'].append(descs[0])
            descriptors_negatives['HBA'].append(descs[1])
            descriptors_negatives['HBD'].append(descs[2])
            descriptors_negatives['TPSA'].append(descs[3])
            descriptors_negatives['aromatic_rings'].append(descs[4])
            descriptors_negatives['nb_heteroatoms'].append(descs[5])
            descriptors_negatives['nb_rot_bonds'].append(descs[6])
        else:
            descriptors_ambiguous['MW'].append(descs[0])
            descriptors_ambiguous['HBA'].append(descs[1])
            descriptors_ambiguous['HBD'].append(descs[2])
            descriptors_ambiguous['TPSA'].append(descs[3])
            descriptors_ambiguous['aromatic_rings'].append(descs[4])
            descriptors_ambiguous['nb_heteroatoms'].append(descs[5])
            descriptors_ambiguous['nb_rot_bonds'].append(descs[6])
    print('Computing descriptors for the test set...')
    for molinfo in dataio.read_unlabelled_smiles():
        mol = Chem.MolFromSmiles(molinfo[-1])
        if mol is None:
            continue
        descs = compute_descs_rdkit(mol)
        descriptors_test['MW'].append(descs[0])
        descriptors_test['HBA'].append(descs[1])
        descriptors_test['HBD'].append(descs[2])
        descriptors_test['TPSA'].append(descs[3])
        descriptors_test['aromatic_rings'].append(descs[4])
        descriptors_test['nb_heteroatoms'].append(descs[5])
        descriptors_test['nb_rot_bonds'].append(descs[6])
    for desc in descriptors_positives.keys():
        print(desc)
        x_pos, y_pos = cdf(np.array(descriptors_positives[desc]))
        x_neg, y_neg = cdf(np.array(descriptors_negatives[desc]))
        x_amb, y_amb = cdf(np.array(descriptors_ambiguous[desc]))
        x_test, y_test = cdf(np.array(descriptors_test[desc]))
        plot_cdfs([x_pos, x_neg, x_amb, x_test], [y_pos, y_neg, y_amb, y_test],
                  labels=['pos', 'neg', 'amb', 'test'],
                  desc=desc)


# --- Using pieces and bits from http://www.mglerner.com/blog/?p=28 #####

def draw_gaussian_kde(data, kernel_width=10, showpts=False, descriptor='MW'):

    def gaussian(x, sigma, mu):
        return (1/np.sqrt(2*np.pi*sigma**2)) * np.exp(-((x-mu)**2)/(2*sigma**2))

    width = data.max() - data.min()
    left, right = data.min(), data.min() + width
    ax = plt.axes()
    # left, right = left - width, right + width   # dunno why the guy is doing that...
    if showpts:
        plt.plot(data, ones_like(data)*0.1, 'go')
    numpts = len(np.unique(data))
    x = linspace(left, right, numpts)
    dx = (right-left)/(numpts-1)
    print(numpts - 1)
    print(dx)
    kernelpts = int(kernel_width/dx)
    kernel = gaussian(linspace(-3, 3, kernelpts), 1, 0)
    y = zeros(numpts)
    for d in data:
        center = d - left
        centerpts = int(center/dx)
        bottom = centerpts - int(kernelpts/2)
        top = centerpts+int(kernelpts/2)
        if top - bottom < kernelpts:
            top = top + (kernelpts - (top - bottom))
        elif top - bottom > kernelpts:
            top = top - ((top - bottom) - kernelpts)
        try:
            y[bottom:top] += kernel
        except Exception:
            print('Could not take care of this datapoint.')
    ax.set_xlim(x[np.where(y > 0)[0][0]], x[np.where(y > 0)[0][-1]])
    plt.plot(x, y)
    ax.set_xlabel(descriptor)
    ax.set_ylabel('Gaussian kernel density')
    plt.title('With kernel width=%i' % kernel_width)
    ax.set_ylim(min(0, y.min()), 1.1*y.max())
    plt.show()


# --- Using pandas' Series.plot() functionalities

def draw_pandas_kde(data, descriptor='MW'):
    s = pd.Series(data=data)
    s.plot(kind='kde', label=descriptor)
    plt.show()


# noinspection PyUnusedLocal
def draw_multiple_pandas_kde(data, labels=('pos', 'neg', 'amb', 'test'), descriptor='MW'):
    df = pd.DataFrame(data=data)  # , columns=list(labels)
    df.plot(kind='kde', label=descriptor)
    plt.legend()
    plt.show()


def do_the_kde_job_with_pandas():
    descriptors_positives = {'MW': [], 'HBA': [], 'HBD': [], 'TPSA': [], 'aromatic_rings': [], 'nb_heteroatoms': [],
                             'nb_rot_bonds': []}
    descriptors_negatives = {'MW': [], 'HBA': [], 'HBD': [], 'TPSA': [], 'aromatic_rings': [], 'nb_heteroatoms': [],
                             'nb_rot_bonds': []}
    descriptors_ambiguous = {'MW': [], 'HBA': [], 'HBD': [], 'TPSA': [], 'aromatic_rings': [], 'nb_heteroatoms': [],
                             'nb_rot_bonds': []}
    descriptors_test = {'MW': [], 'HBA': [], 'HBD': [], 'TPSA': [], 'aromatic_rings': [], 'nb_heteroatoms': [],
                        'nb_rot_bonds': []}
    print('Computing descriptors for the training set...')
    for molinfo in dataio.read_labelled_smiles():
        mol = Chem.MolFromSmiles(molinfo[-1])
        if mol is None:
            continue
        descs = compute_descs_rdkit(mol)
        if molinfo[3] == 'true':
            descriptors_positives['MW'].append(descs[0])
            descriptors_positives['HBA'].append(descs[1])
            descriptors_positives['HBD'].append(descs[2])
            descriptors_positives['TPSA'].append(descs[3])
            descriptors_positives['aromatic_rings'].append(descs[4])
            descriptors_positives['nb_heteroatoms'].append(descs[5])
            descriptors_positives['nb_rot_bonds'].append(descs[6])
        elif molinfo[3] == 'false':
            descriptors_negatives['MW'].append(descs[0])
            descriptors_negatives['HBA'].append(descs[1])
            descriptors_negatives['HBD'].append(descs[2])
            descriptors_negatives['TPSA'].append(descs[3])
            descriptors_negatives['aromatic_rings'].append(descs[4])
            descriptors_negatives['nb_heteroatoms'].append(descs[5])
            descriptors_negatives['nb_rot_bonds'].append(descs[6])
        else:
            descriptors_ambiguous['MW'].append(descs[0])
            descriptors_ambiguous['HBA'].append(descs[1])
            descriptors_ambiguous['HBD'].append(descs[2])
            descriptors_ambiguous['TPSA'].append(descs[3])
            descriptors_ambiguous['aromatic_rings'].append(descs[4])
            descriptors_ambiguous['nb_heteroatoms'].append(descs[5])
            descriptors_ambiguous['nb_rot_bonds'].append(descs[6])
    print('Computing descriptors for the test set...')
    for molinfo in dataio.read_unlabelled_smiles():
        mol = Chem.MolFromSmiles(molinfo[-1])
        if mol is None:
            continue
        descs = compute_descs_rdkit(mol)
        descriptors_test['MW'].append(descs[0])
        descriptors_test['HBA'].append(descs[1])
        descriptors_test['HBD'].append(descs[2])
        descriptors_test['TPSA'].append(descs[3])
        descriptors_test['aromatic_rings'].append(descs[4])
        descriptors_test['nb_heteroatoms'].append(descs[5])
        descriptors_test['nb_rot_bonds'].append(descs[6])
    for desc in descriptors_positives.keys():
        draw_multiple_pandas_kde(np.array((descriptors_positives[desc], descriptors_negatives[desc],
                                           descriptors_ambiguous[desc], descriptors_test[desc])), descriptor=desc)


if __name__ == '__main__':
    # do_the_job()
    # Test the KDE thing
    # descriptors_test = {'MW': [], 'HBA': [], 'HBD': [], 'TPSA': [], 'aromatic_rings': [], 'nb_heteroatoms': [],
    #                     'nb_rot_bonds': []}
    # for molinfo in dataio.read_unlabelled_smiles():
    #     mol = Chem.MolFromSmiles(molinfo[-1])
    #     if mol is None:
    #         continue
    #     descs = compute_descs_rdkit(mol)
    #     descriptors_test['MW'].append(descs[0])
    #     descriptors_test['HBA'].append(descs[1])
    #     descriptors_test['HBD'].append(descs[2])
    #     descriptors_test['TPSA'].append(descs[3])
    #     descriptors_test['aromatic_rings'].append(descs[4])
    #     descriptors_test['nb_heteroatoms'].append(descs[5])
    #     descriptors_test['nb_rot_bonds'].append(descs[6])
    #
    # best_width = {'MW': 95,
    #               'HBA': 5,
    #               'HBD': 5,
    #               'TPSA': 35,
    #               'aromatic_rings': 5,
    #               'nb_heteroatoms': 5,
    #               'nb_rot_bonds': 5}
    #
    # for desc in sorted(descriptors_test.keys()):
    #     draw_pandas_kde(np.array(descriptors_test[desc]), descriptor=desc)
    # #draw_pandas_kde(np.array(descriptors_test['HBA']), descriptor='HBA')

    do_the_kde_job_with_pandas()
