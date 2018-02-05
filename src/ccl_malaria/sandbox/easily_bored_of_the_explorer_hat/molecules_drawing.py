# Let's try to use other stuff than rdkit...
# http://ctr.wikia.com/wiki/Align_the_depiction_using_a_fixed_substructure

import indigo
import indigo_renderer
from ccl_malaria_sandbox.folding_and_collisions import from_feat_back_to_mols_faster
from molscatalog import MalariaCatalog


# noinspection PyUnusedLocal
def draw_grid(mols, pattern, output_file, legends=None, classes=None, values=None, ranking=None):
    """
    mols is a list of molecules we want to plot in a grid. Given as smiles string
    pattern is a smiles or smarts string that all the mols are supposed to contain
    legends: what to print under each picture (molid or so...)
    classes: if binry, can be translated into green/red background color
    vales, ranking: for the slider
    """
    indig = indigo.Indigo()
    renderer = indigo_renderer.IndigoRenderer(indig)
    query = indig.loadSmarts(pattern)

    xyz = []
    collection = indig.createArray()
    refatoms = []
    collection.arrayAdd(query)
    refatoms.append(0)
    for i, smi in enumerate(mols):
        structure = indig.loadMolecule(smi)
        structure.foldHydrogens()
        match = indig.substructureMatcher(structure).match(query)
        if i == 0:
            for atom in query.iterateAtoms():
                xyz.extend(match.mapAtom(atom).xyz())
        else:
            atoms = [match.mapAtom(atom).index() for atom in query.iterateAtoms()]
            structure.alignAtoms(atoms, xyz)
        refatoms.append(match.mapAtom(query.getAtom(0)).index())
        collection.arrayAdd(structure)

    indig.setOption('render-output-format', 'png')
    indig.setOption('render-image-size', '400, 400')
    # indig.setOption("render-grid-title-property", "PUBCHEM_COMPOUND_CID");
    indig.setOption('render-grid-title-font-size', '10')
    indig.setOption('render-grid-title-offset', '2')
    indig.setOption('render-coloring', 'true')
    indig.setOption('render-coloring', 'true')
    indig.setOption('render-background-color', 1.0, 1.0, 1.0)
    renderer.renderGridToFile(collection, refatoms, 4, output_file)


if __name__ == '__main__':
    pattern = 'c(CN)(cc)c(c)O'
    molecules = from_feat_back_to_mols_faster('lab', pattern)
    molids = [molecule[1] for molecule in molecules][:10]
    mc = MalariaCatalog()
    smis = mc.molids2smiless(molids)
    draw_grid(smis, pattern, '/home/flo/bla4.png')
    # classes = [1,1,0,0,1,0,1,0,0,0],
    # values=[[0.9, 0.87, 0.8, 0.67, 0.65, 0.63, 0.55, 0.23] for mol in molecules[:10]],
    # index=np.random.randint(7, size=10))
