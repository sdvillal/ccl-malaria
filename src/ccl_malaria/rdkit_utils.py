# coding=utf-8
"""Molecule manipulation using rdkit."""
from __future__ import print_function, division
from future.utils import string_types
from collections import Iterable
from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem, Descriptors3D
import numpy as np
from rdkit.Chem.PropertyMol import PropertyMol
from ccl_malaria import warning
from collections import defaultdict


##################################################
# DESCRIPTOR COMPUTATION
# For a list of available descriptors see:
#  - http://www.rdkit.org/docs/GettingStartedInPython.html#list-of-available-descriptors
#  - http://code.google.com/p/rdkit/wiki/DescriptorsInTheRDKit
#  - http://www.rdkit.org/docs/api/rdkit.Chem.Descriptors-module.html
##################################################


def discover_rdk_descriptors(verbose=False, no_3D=True, ignores=None):
    """Returns a list of the names descriptors (other than fps, ultrashape and possibly others) present in RDKIT."""
    if ignores is None:
        ignores = set()

    # noinspection PyUnresolvedReferences
    descriptors = tuple(desc_name
                        for desc_name, func in Descriptors.descList
                        if desc_name not in ignores)
    if no_3D:
        descriptors = tuple(desc_name for desc_name in descriptors
                            if not hasattr(Descriptors3D, desc_name))

    if verbose:
        print('Discovered RDKIT descriptors...')
        print('\n'.join(descriptors))
        print('-' * 80)
        print('Members of class Descriptors that are not descriptors...')
        print('\n'.join(sorted(set(Descriptors.__dict__.keys()) -
                               set(descriptors))))
    return descriptors


def desc_wrapper(desc_func):  # TODO: allow logging
    def wrapper(mol):
        try:
            return desc_func(mol)
        except:
            return np.nan
    return wrapper


class RDKitDescriptorsComputer(object):

    def __init__(self, descriptors=None):
        super(RDKitDescriptorsComputer, self).__init__()
        self.descriptors = discover_rdk_descriptors() if descriptors is None else descriptors
        self._dfs = [desc_wrapper(getattr(Descriptors, descriptor)) for descriptor in self.descriptors]

    def _compute_for_mol(self, mol):
        return np.array([computer(mol) for computer in self._dfs])

    def compute(self, mols):
        if not isinstance(mols, Iterable):
            mols = [mols]
        X = np.empty((len(mols), len(self._dfs)))
        for i, mol in enumerate(mols):
            X[i, :] = self._compute_for_mol(mol)
        return X

    def fnames(self, prefix='rdkit-'):
        return ['%s%s' % (prefix, d) for d in self.descriptors]


##################################################
# MORGAN FINGERPRINTS
# See:   http://www.rdkit.org/docs/GettingStartedInPython.html#morgan-fingerprints-circular-fingerprints
##################################################

def explain_circular_substructure(mol,
                                  center,
                                  radius,
                                  use_hs=False,
                                  canonical=True, isomeric=False, kekule=False, all_bonds_explicit=False):
    """Returns a SMILES description of the circular structure defined by a center and a topological radius."""
    atoms = {center}
    env = Chem.FindAtomEnvironmentOfRadiusN(mol, radius, center, useHs=use_hs)
    for bidx in env:
        bond = mol.GetBondWithIdx(bidx)
        atoms.add(bond.GetBeginAtomIdx())
        atoms.add(bond.GetEndAtomIdx())
    return Chem.MolFragmentToSmiles(mol,
                                    atomsToUse=list(atoms),
                                    bondsToUse=env,
                                    rootedAtAtom=center,
                                    isomericSmiles=isomeric,
                                    kekuleSmiles=kekule,
                                    canonical=canonical,
                                    allBondsExplicit=all_bonds_explicit)


def morgan_fingerprint(mol,
                       max_radius=100,
                       fcfp=False,
                       use_chirality=False,
                       use_bond_types=True):
    """
    Computes morgan fingerprints for a molecule.

    Returns also bit explanations and enables flexible hashing and folding in python land.

    Parameters
    ----------
    mol : rdkit.Mol or smiles string
      The molecule.

    max_radius : int
      The maximum radius go take into account.

    fcfp : bool, default False
      Build ECFP-like (False) of FCFP-like descriptors.

    use_chirality : bool, default False
      If True, stereoisomers generate different features.

    use_bond_types : bool, default True
      If True, bonding information is taken into account.

    Returns
    -------
    A dictionary {cansmi: [(center, radius, bit_key)]+}, where:
      - cansmi are the canonical smiles of the substructure
        (note that we lose context that might be used when generating bit_key)
      - center is the atom id where the feature is centered at
      - radius is the depth of the wide-first search used to span the feature
      - bit_key is the hash assigned by rdkit to the substructure/feature
        (should have little to no collisions)
    """
    if isinstance(mol, string_types):
        mol = to_rdkit_mol(mol)
    fpsinfo = {}
    # N.B. We won't actually use rdkit hash, so we won't ask for nonzero values...
    # Is there a way of asking rdkit to give us this directly?
    _ = AllChem.GetMorganFingerprint(mol,
                                     max_radius,
                                     bitInfo=fpsinfo,
                                     useFeatures=fcfp,
                                     useChirality=use_chirality,
                                     useBondTypes=use_bond_types)
    smi2centers = defaultdict(list)  # {smiles: [(atom_center_id, radius, bit)]}

    for bit_key, bit_descs in fpsinfo.items():
        for center, radius in bit_descs:
            cansmiles = explain_circular_substructure(mol, center, radius)
            smi2centers[cansmiles].append((center, radius, bit_key))
    return smi2centers


##################################################
# Instantiation of molecules
##################################################


def to_rdkit_mol(smiles, molid=None, sanitize=True, to2D=False, to3D=False, toPropertyMol=False):
    """Converts a smiles string into an RDKit molecule."""
    mol = Chem.MolFromSmiles(smiles, sanitize=sanitize)
    if mol is None:
        if molid is None:
            warning('RDKit cannot create a molecule from smiles %s' % smiles)
        else:
            warning('RDKit cannot create molecule %s from smiles %s' % (molid, smiles))
        return mol
    if to3D:
        AllChem.EmbedMolecule(mol)
        AllChem.UFFOptimizeMolecule(mol)
    elif to2D:
        AllChem.Compute2DCoords(mol)
    if toPropertyMol:
        return PropertyMol(mol)
    return mol
