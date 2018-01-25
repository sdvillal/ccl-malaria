# coding=utf-8
"""Molecule manipulation using rdkit."""
from __future__ import print_function, division
from future.utils import string_types

from itertools import combinations
from functools import partial
from collections import Iterable, defaultdict
from copy import deepcopy

import numpy as np

from ccl_malaria import warning
from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem, Descriptors3D
from rdkit.Chem.PropertyMol import PropertyMol

import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)  # FIXME
info = logger.info
debug = logger.debug

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
        except Exception:
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
                       use_bond_types=True,
                       fold_size=None,
                       explainer=explain_circular_substructure):
    """
    Computes morgan fingerprints for a molecule.

    This is a thin wrapper over rdkit GetMorganFingerprint.

    By default it returns bit explanations as smiles, that can be useful to explain
    features without resorting to concrete molecules containing them or to detect
    "growing" versions of a substructure that could be catched by SMILES canonicalization
    (e.g. CO -> COO...). Note that the same explanation can correspond to several features
    (because rdkit also use atom context to compute the hash, that sometimes cannot be represented
    by smiles).

    Disabling folding enable flexible hashing and folding in python land with minimal
    collision probability.

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

    fold_size : int or None, default None
      If provided, folding will already be done by rdkit.

    explainer : f(mol, center, radius) -> cansmiles or None, default `explain_circular_substructure`

      If provided, each present feature ("bit") will be converted to a SMILES string,
      possibly losing some atom context information on the way. Usually you will want to
      use a `partial` application of `explain_circular_substructure` here.

      If False, only the rdkit hash (possibly folded) will be used to represent
      the feature.

    Returns
    -------
    If `explained` is True, a dictionary {cansmi: [(center, radius, bit_key)]+}, where:
      - cansmi are the canonical smiles of the substructure
        (note that we lose context that might be used when generating bit_key)
      - center is the atom id where the feature is centered at
      - radius is the depth of the wide-first search used to span the feature
      - bit_key is the hash assigned by rdkit to the substructure/feature
        (should have little to no collisions if using unfolded fingerprints)

    If `explained` is False, a dictionary {hash: count}, where:
      - Hash is the int value assigned by rdkit hash (+ possibly folding) to a feature.
        If fold_size is None, collisions will happen very rarely.

    Examples
    --------
    # Lets have a fun molecule
    >>> smiles = 'CC1CCC/C(C)=C1/C=C/C(C)=C/C=C/C(C)=C/C=C/C=C(C)/C=C/C=C(C)/C=C/C2=C(C)/CCCC2(C)C'

    # By default we get unfolded, explained fingerprints
    >>> explained_unfolded_fpt = morgan_fingerprint(smiles)
    >>> print(explained_unfolded_fpt['C(=CC(C)=CC=C)C=C(C)C=CC'])
    [(13, 5, 1652796936), (24, 5, 1652796936)]

    # We can ask rdkit to fold the fingerprint for us
    >>> explained_folded_fpt = morgan_fingerprint(smiles, fold_size=1024)
    >>> print(explained_folded_fpt['C(=CC(C)=CC=C)C=C(C)C=CC'])
    [(13, 5, 520), (24, 5, 520)]

    # We can also just ask for rdkit feature hashes (explainer=None)...
    >>> unfolded_fpt = morgan_fingerprint(smiles, fold_size=None, explainer=None)
    >>> print(unfolded_fpt[1652796936])
    2
    # ...and hash after folding (note the collision within the same molecule)
    >>> folded_fpt = morgan_fingerprint(smiles, fold_size=1024, explainer=None)
    >>> print(folded_fpt[520])
    4

    The smiles representation of the feature explanation can also
    be tuned with the explainer parameter
    >>> from functools import partial
    >>> to_isomeric_smiles = partial(explain_circular_substructure, isomeric=True)
    >>> explained_folded_fpt_isomeric = morgan_fingerprint(smiles, fold_size=1024, explainer=to_isomeric_smiles)
    >>> print(explained_folded_fpt_isomeric['C(=C/C=C)\\C(C)=C\\C'])
    [(14, 3, 397), (23, 3, 397)]

    Note that ECFP and FCFP can generate features that, while being represented
    by different SMILES, actually correspond to the same substructure. For example,
    let's badly cut an aromatic ring here:
    >>> smiles = 'CCC(=O)Oc1ccc(/C=C(\C#N)S(=O)(=O)c2ccccc2)cc1OC'
    >>> same_substructure = [
    ...     'c1c(C=C(C#N)S(=O)(=O)c(c)c)ccc(OC(=O)CC)c1OC',
    ...     'c1cc(OC(=O)CC)c(OC)cc1C=C(C#N)S(=O)(=O)c(c)c'
    ... ]
    >>> smi1 = AllChem.MolToSmiles(AllChem.MolFromSmarts(same_substructure[0]))
    >>> smi2 = AllChem.MolToSmiles(AllChem.MolFromSmarts(same_substructure[1]))
    >>> smi1 == smi2
    True
    >>> ecfp = morgan_fingerprint(smiles, fcfp=False)
    >>> fcfp = morgan_fingerprint(smiles, fcfp=False)
    >>> same_substructure[0] in ecfp, same_substructure[1] in ecfp
    True, False
    >>> same_substructure[0] in fcfp, same_substructure[1] in fcfp
    False, True
    """

    #
    # Notes:
    #
    # 1) The bitkey (feature hash) is computed as follows in rdkit-C-land:
    #  - compute the feature based on the subgraph and atom contextual information
    #  - hash the (description of the) feature => random_seed
    #  - use that hash to seed a rng and use it to set randomly a given number of bits
    #  - these bits are then interpreted as an uint
    #
    # This, unfortunately, is not the same as assigning several bitkeys to
    # one feature, at least in terms of collision resolution.
    # So it is a step behind similar ML solutions to alleviate collisions,
    # (for example, assign several hashes to a single feature).
    # See, e.g., vowpal wabbit hashing-trick tricks.
    #
    # In any case, hopefully this will be irrelevant it terms of number of collisions.
    # Of course folding would also provoke more undistinguishable collisions.
    #
    # 2) Unexplained fingerprints include more differentiating information,
    # as by using explanation (converting to SMILES) we are possibly collapsing
    # together features that, even if mapping to the same SMILES, might have had
    # different rdkit hash due to different atom contextual information.
    #
    # 3) To keep better statistical guarantees of uniform distribution
    # of hashes regardless of the quality of the hashing function, it is safer
    # not to use powers of two for `fold_size` (as it is customary in
    # cheminformatics). For example, instead of 1024, use better 1023 or,
    # even better, a prime number (1021) close to your desired size. At
    # the same time, power of two sizes also have their advantages
    # (e.g. faster address computation), and bad quality hash functions
    # can be corrected to avoid key clustering. I have not checked which
    # hash function does rdkit use. See e.g. https://gist.github.com/badboy/6267743.
    #

    if isinstance(mol, string_types):
        mol = to_rdkit_mol(mol)
    fpsinfo = {}
    smi2centers = defaultdict(list)  # {smiles: [(atom_center_id, radius, bit)]}
    if fold_size is None:
        fpt = AllChem.GetMorganFingerprint(mol,
                                           max_radius,
                                           bitInfo=fpsinfo,
                                           useFeatures=fcfp,
                                           useChirality=use_chirality,
                                           useBondTypes=use_bond_types)
    else:
        fpt = AllChem.GetHashedMorganFingerprint(mol,
                                                 max_radius,
                                                 nBits=fold_size,
                                                 bitInfo=fpsinfo,
                                                 useFeatures=fcfp,
                                                 useChirality=use_chirality,
                                                 useBondTypes=use_bond_types)
    if explainer:
        for bit_key, bit_descs in fpsinfo.items():
            for center, radius in bit_descs:
                cansmiles = explainer(mol, center, radius)
                smi2centers[cansmiles].append((center, radius, bit_key))
        return smi2centers
    else:
        return fpt.GetNonzeroElements()


##################################################
# Instantiation of molecules and patterns
##################################################


def best_effort_sanitization(mol, copy=False, ops=Chem.SANITIZE_ALL):

    # Some literature:
    #   http://www.rdkit.org/docs/RDKit_Book.html#molecular-sanitization
    #   http://www.rdkit.org/docs/Cookbook.html#using-a-different-aromaticity-model

    if copy:
        mol = deepcopy(mol)

    failed_op = None
    while failed_op != Chem.SANITIZE_NONE:
        # FIXME: is repeated sanitization idempotent, or should we always depart from the original mol?
        failed_op = Chem.SanitizeMol(mol, catchErrors=True, sanitizeOps=ops)
        if failed_op == Chem.SANITIZE_NONE:
            return mol
        ops ^= failed_op

    return None


def to_rdkit_mol(mol_repr, molid=None, instantiator=Chem.MolFromSmiles, to2D=False, to3D=False, toPropertyMol=False):
    """
    Converts a molecular representation (e.g. smiles string) into an RDKit molecule.
    Allows to perform common postprocessing operations on the resulting molecule.
    """
    if not isinstance(mol_repr, Chem.Mol):
        mol = instantiator(mol_repr)
    else:
        mol = mol_repr
    if mol is None:
        if molid is None:
            warning('RDKit cannot create a molecule from %r' % mol_repr)
        else:
            warning('RDKit cannot create molecule %s from %r' % (molid, mol_repr))
        return mol
    if to3D:
        AllChem.EmbedMolecule(mol)
        AllChem.UFFOptimizeMolecule(mol)
    elif to2D:
        AllChem.Compute2DCoords(mol)
    if toPropertyMol:
        return PropertyMol(mol)
    return mol


mol_from_smiles = to_rdkit_mol
unsanitized_mol_from_smiles = partial(to_rdkit_mol, instantiator=partial(Chem.MolFromSmiles, sanitize=False))
mol_from_smarts = partial(to_rdkit_mol, instantiator=AllChem.MolFromSmarts)


def to_rdkit_mols(mol_reprs, instantiator=unsanitized_mol_from_smiles):
    # Convenience to allow single mols or strings
    if isinstance(mol_reprs, (string_types, Chem.Mol)):
        mol_reprs = [mol_reprs]
    # Generate...
    for mol_repr in mol_reprs:
        yield instantiator(mol_repr)


def has_substruct_match(mol, pattern,
                        recursion_possible=True,
                        use_chirality=False,
                        use_query_query_matches=False):
    """Thin wrapper over `Mol.HasSubstructMatch`."""

    #
    # Here we can use either MolFromXXX or MolFromSMARTS. Matching semantics change. See:
    #   http://www.rdkit.org/docs/GettingStartedInPython.html#substructure-searching
    #   http://www.rdkit.org/docs/RDKit_Book.html#atom-atom-matching-in-substructure-queries
    #

    error_messages = []
    if not isinstance(mol, Chem.Mol):
        error_messages.append('mol must be an `rdkit.Mol`, it is %r')
    if not isinstance(pattern, Chem.Mol):
        error_messages.append('pattern must be an `rdkit.Mol`, it is %r')
    if error_messages:
        raise ValueError('; '.join(error_messages))

    return mol.HasSubstructMatch(pattern,
                                 recursionPossible=recursion_possible,
                                 useChirality=use_chirality,
                                 useQueryQueryMatches=use_query_query_matches)


has_non_recursive_query_query_match = partial(has_substruct_match,
                                              use_query_query_matches=True,
                                              recursion_possible=False)
has_query_query_match = partial(has_substruct_match,
                                use_query_query_matches=True)


def group_substructures(mols, patterns=None,
                        mol_instantiator=unsanitized_mol_from_smiles,
                        pattern_instantiator=mol_from_smarts,
                        matcher=has_query_query_match,
                        reduce=True):

    import networkx as nx

    # Instantiate mols and their "pattern" representation
    # Must document that, when already provided Chem.Mol objects, instantiators usually are no-ops
    if pattern_instantiator is not None:
        patterns = list(to_rdkit_mols(mols, pattern_instantiator))
    if mol_instantiator is not None:
        mols = list(to_rdkit_mols(mols, mol_instantiator))

    if patterns is None:
        patterns = mols

    # Sort substructures by decreasing number of atoms
    num_atoms = [mol.GetNumAtoms() for mol in mols]
    descending_number_of_atoms_order = np.argsort(num_atoms)[::-1]

    representative = [None] * len(mols)  # For duplicates
    graph = nx.DiGraph()                 # Directed graph, if (p1, p2) on it,

    # Nasty stuff that would not happen if cheminformatics were logical
    # noinspection PyUnusedLocal
    has_equal_nonequal = has_cycles = False

    for p1, p2 in combinations(descending_number_of_atoms_order, 2):
        p2_in_p1, p1_in_p2 = matcher(mols[p1], patterns[p2]), matcher(mols[p2], patterns[p1])
        representative[p1] = representative[p1] or p1
        representative[p2] = representative[p2] or p2
        if p2_in_p1 and p1_in_p2:
            representative[p2] = representative[p1]
        elif p2_in_p1:
            if num_atoms[p1] == num_atoms[p2] and not has_equal_nonequal:
                has_equal_nonequal = True
                info('mindblowingly, with equal number of atoms, one contains the other but not viceversa')
            graph.add_edge(representative[p1], representative[p2])
        elif p1_in_p2:
            if num_atoms[p1] == num_atoms[p2] and not has_equal_nonequal:
                has_equal_nonequal = True
                info('mindblowingly, with equal number of atoms, one contains the other but not viceversa')
            graph.add_edge(representative[p2], representative[p1])
        else:
            graph.add_node(representative[p1])
            graph.add_node(representative[p2])

    # Cycles?
    try:
        nx.find_cycle(graph)
        has_cycles = True
        info('containment graph has cycles')
    except nx.NetworkXNoCycle:
        has_cycles = False

    if reduce:
        graph = nx.transitive_reduction(graph)

    groups = list(nx.weakly_connected_components(graph))
    # noinspection PyCallingNonCallable
    roots = [node for node, degree in graph.in_degree() if 0 == degree]
    # noinspection PyCallingNonCallable
    leaves = [node for node, degree in graph.out_degree() if 0 == degree]

    return graph, groups, representative, roots, leaves, num_atoms, has_cycles, has_equal_nonequal


if __name__ == '__main__':
    p1 = 'C(=C(C#N)C(=O)NC)c1c(C)[nH]c(c)c1cc'
    p2 = 'c1(C=C(C#N)C(=O)NC)c(C)[nH]c2ccccc21'
    print(group_substructures([p1, p2]))


# candidates = [
#     'C(=NNC(=O)c1ccccc1)c1cc(OC)c(OC)cc1OC',
#     'N(=Cc1cc(OC)c(O)cc1OC)NC(=O)c1ccccc1',
# ]
#
# #                i     r                                                  s
# # 1913        1913  1913   C(=C(C#N)S(=O)(=O)c1ccccc1)c1ccc(OC(C)=O)c(OC)c1
# # 1594291  1594291  1913    c1(C=C(C#N)S(=O)(=O)c(cc)cc)ccc(OC(C)=O)c(OC)c1
# # 1594358  1594358  1913  c1(C=C(C#N)S(=O)(=O)c2ccccc2)ccc(OC(=O)CC)c(OC)c1
# # 1816229  1816229  1913           c1(OC)cc(C=C(C#N)S(c)(=O)=O)ccc1OC(=O)CC
# # 2025829  2025829  1913       c1c(C=C(C#N)S(=O)(=O)c(c)c)ccc(OC(=O)CC)c1OC
# # 2025839  2025839  1913     c1c(C=C(C#N)S(=O)(=O)c(cc)cc)ccc(OC(=O)CC)c1OC
# # 2025852  2025852  1913            c1c(C=C(C#N)S(c)(=O)=O)ccc(OC(C)=O)c1OC
# # 2134013  2134013  1913       c1cc(OC(=O)CC)c(OC)cc1C=C(C#N)S(=O)(=O)c(c)c
#


# candidates = [
#     'C(=C(C#N)S(=O)(=O)c1ccccc1)c1ccc(OC(C)=O)c(OC)c1',
#     'c1(C=C(C#N)S(=O)(=O)c(cc)cc)ccc(OC(C)=O)c(OC)c1',
#     'c1(C=C(C#N)S(=O)(=O)c2ccccc2)ccc(OC(=O)CC)c(OC)c1',
#     'c1(OC)cc(C=C(C#N)S(c)(=O)=O)ccc1OC(=O)CC',
#     'c1c(C=C(C#N)S(=O)(=O)c(c)c)ccc(OC(=O)CC)c1OC',
#     'c1c(C=C(C#N)S(=O)(=O)c(cc)cc)ccc(OC(=O)CC)c1OC',
#     'c1c(C=C(C#N)S(c)(=O)=O)ccc(OC(C)=O)c1OC',
#     'c1cc(OC(=O)CC)c(OC)cc1C=C(C#N)S(=O)(=O)c(c)c',
# ]

# print(group_patterns(candidates))
