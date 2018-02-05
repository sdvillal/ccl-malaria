"""
The normal rdkit drawing tools for creating grid images has its limitations, essentially when it comes to: fragments
(substructures that could not possibly be transformed to a proper molecules), coloring substructures,
adding backgrounds. Here we try to solve some of these problems
"""
from __future__ import print_function, division

import os
import os.path as op

import numpy as np

from ccl_malaria.sandbox.plotting.folding_and_collisions import from_feat_back_to_mols_faster
from rdkit import Chem
from rdkit.Chem import AllChem, Draw
from rdkit.Chem.Draw.MolDrawing import MolDrawing, DrawingOptions, Font
from rdkit.Chem.GraphDescriptors import periodicTable


def draw_in_a_grid_aligned_according_to_pattern(mols, pattern, image_file, legends=None, classes=None, molsPerRow=4,
                                                symbols=None, values=None, index=None, matching=True):
    """
    Mols is a list of rdkit mols sharing all the pattern, that is given as smiles string. The final grid image is saved
    in .png format on the required image_file path.
    """
    p = Chem.MolFromSmiles(pattern)
    if p is None:
        p = Chem.MolFromSmiles(pattern, sanitize=False)
        # See https://www.mail-archive.com/rdkit-discuss@lists.sourceforge.net/msg02387.html
        # it will hopefully remove the error message about "Precondition Violation"
        # p.UpdatePropertyCache(strict=False)
        # ph=Chem.AddHs(p, addCoords=True)
    try:
        subms = [x for x in mols if x.HasSubstructMatch(p)]
        AllChem.Compute2DCoords(p)
    except Exception:
        print('Error when getting 2D coords of the pattern.')
        p = Chem.MolFromSmarts(pattern)
        subms = [x for x in mols if x.HasSubstructMatch(p)]
        AllChem.Compute2DCoords(p)
    for m in subms:
        AllChem.GenerateDepictionMatching2DStructure(m, p)
    if matching:
        matchings = [mol.GetSubstructMatch(p) for mol in subms]
        matchings.insert(0, None)
    else:
        matchings = [None for _ in range(len(subms))]
        matchings.insert(0, None)
    subms.insert(0, p)

    if classes is None:
        # Simple version; just taking properly care of depicting the pattern that can be an improper molecule
        img = gridplot_no_class(subms, pattern, molsPerRow, legends, matchings, symbols)
    elif values is None and index is None:
        # Version with colored bounding boxes according to the class
        img = gridplot_classes(subms, pattern, molsPerRow, legends, matchings, symbols, classes)

    elif values is not None and index is not None:
        # Let's create a matplotlib image instead of those boring canvas:
        import matplotlib.pyplot as plt
        from matplotlib.widgets import Slider
        from PIL import Image

        if legends is None:
            legends = [None for _ in range(len(subms))]
        legends[0] = pattern
        nRows = len(subms) // molsPerRow
        if len(subms) % molsPerRow:
            nRows += 1

        colors = ['white']
        for clas in classes:
            # noinspection PyTypeChecker
            colors.append((200, 0, 0) if clas == 1 else (0, 200, 0))

        img = Image.new("RGBA", (molsPerRow * 200, nRows * 200), (0, 0, 0, 0))

        for i, mol in enumerate(subms):
            row = i // molsPerRow
            col = i % molsPerRow
            if i == 0:
                # Let's first take care of the problematic substructure. No need to add pretty border nor slider
                subimg = MolToImage(mol, (200, 200), kekulize=False, ignoreHs=True,
                                    legend=legends[i], highlightAtoms=matchings[i], background=colors[i], )
            else:
                # One day it will work...
                rank = float(len(values[i - 1][:index[i - 1] + 1])) / len(values[i - 1])
                subimg = MolToImage(mol, (200, 200), legend=str(rank), highlightAtoms=matchings[i],
                                    background=colors[i], mol_adder=MolDrawing.AddMol, symbol=None)
                # draw_mol_slider(mol, legends[i], matchings[i], symbols[i], colors[i], values[i-1], index[i-1])
                # subimg = Image.open('/home/flo/temp.png')
                # subimg.show()
            img.paste(subimg, (col * 200, row * 200))
    else:
        raise ValueError('Nasty here')

    img.save(image_file)

    #  f0 = plt.figure(0)
    #  sub = plt.subplot(nRows, molsPerRow, 1)
    #  sub.imshow(pattern_img)
    #  sub.axis('off')
    # plt.tight_layout()
    # plt.show()
    # for i, mol in enumerate(subms[1:]):
    #     draw_mol_slider(mol, legends[i+1], matchings[i+1], symbols[i+1], colors[i], values[i], index[i])
    #     sub2 = plt.subplot(nRows, molsPerRow, i+1)
    #     # We need a float array between 0-1, rather than
    #     # a uint8 array between 0-255
    #     #print np.array(subimg.make_image())
    #     from matplotlib.image import FigureImage
    #     #im = np.array(subimg.figure).astype(np.float) / 255
    #     subim = plt.imread('/home/flo/temp.png')
    #     sub2.imshow(subim)
    #     sub2.axis('off')
    # plt.tight_layout()
    # f0.show()

    # for i, (vals, ind) in enumerate(zip(values, index)):
    #     # Then we will integrate a slider showing the relative ranking of the molecule
    #     rank = float(len(vals[:ind+1]))/len(vals)
    #     alpha_axis  = plt.axes([0.5, 0.5, 0.20, 0.03], axisbg='grey')
    #     img2 = Slider(alpha_axis, '1(+)', 0, 1, valinit=rank, facecolor='w', valfmt='(-)%.0f')
    #     fig.figimage(img2, 0)
    #     #plt.savefig(file, bbox_inches='tight')
    # plt.show()
    # img.save(image_file)


# noinspection PyUnusedLocal
def draw_in_a_grid(mols, image_file, legends=None, classes=None, molsPerRow=4,
                   symbols=None, values=None, index=None, matching=True):
    """
    Mols is a list of rdkit mols. The final grid image is saved in .png format on the required image_file path.
    NO PATTERN!
    """
    if classes is None:
        # Simple version
        img = Draw.MolsToGridImage(mols, molsPerRow=molsPerRow, subImgSize=(200, 200), kekulize=False, legends=legends)
    elif values is None and index is None:
        # Version with colored bounding boxes according to the class
        raise NotImplementedError()

    elif values is not None and index is not None:
        # Let's create a matplotlib image instead of those boring canvas:
        raise NotImplementedError()
    else:
        raise NotImplementedError()

    img.save(image_file)


def gridplot_no_class(subms, pattern, molsPerRow, legends, matchings, symbols):
    try:
        # Let's first try the rdkit way
        img = Draw.MolsToGridImage(subms, molsPerRow=molsPerRow, subImgSize=(200, 200), kekulize=False, legends=legends,
                                   highlights=matchings)

    except RuntimeError:
        try:
            import Image
        except ImportError:
            from PIL import Image
        if legends is None:
            legends = [None] * len(subms)
        legends[0] = pattern
        nRows = len(subms) // molsPerRow
        if len(subms) % molsPerRow:
            nRows += 1

        img = Image.new("RGBA", (molsPerRow * 200, nRows * 200), (255, 255, 255, 0))

        try:
            for i, mol in enumerate(subms):
                row = i // molsPerRow
                col = i % molsPerRow
                subimg = Draw.MolToImage(mol, (200, 200), legend=legends[i], highlightAtoms=matchings[i])
                img.paste(subimg, (col * 200, row * 200))
        except Exception:
            # Let's first take care of the problematic substructure
            pattern_img = MolToImage(subms[0], molsPerRow=4, subImgSize=(200, 200), kekulize=False, ignoreHs=True,
                                     legend=legends[0])
            img.paste(pattern_img, (0, 0))
            for i, mol in enumerate(subms[1:]):
                row = (i + 1) // molsPerRow
                col = (i + 1) % molsPerRow
                subimg = Draw.MolToImage(mol, (200, 200), legend=legends[i + 1], highlightAtoms=matchings[i + 1],
                                         symbol=symbols[i + 1])
                img.paste(subimg, (col * 200, row * 200))
    return img


def gridplot_classes(subms, pattern, molsPerRow, legends, matchings, symbols, classes):
    try:
        import Image
    except ImportError:
        from PIL import Image
    if legends is None:
        legends = [None for _ in range(len(subms))]
        legends[0] = pattern if len(pattern) < 20 else ''
    else:
        legends.insert(0, pattern if len(pattern) < 20 else '')
    if symbols is None:
        symbols = [None for _ in range(len(subms))]

    nRows = len(subms) // molsPerRow
    if len(subms) % molsPerRow:
        nRows += 1
    img = Image.new("RGBA", (molsPerRow * 200, nRows * 200), (0, 0, 0, 0))
    colors = ['white']
    for clas in classes:
        # noinspection PyTypeChecker
        colors.append((200, 0, 0) if clas == 1 else (0, 200, 0))
    for i, mol in enumerate(subms):
        row = i // molsPerRow
        col = i % molsPerRow
        # symbols[i] = '*'  # TODO change that awful hardcoding!!!
        if i == 0:
            subimg = MolToImage(mol, (200, 200), legend=legends[i], highlightAtoms=matchings[i], background=colors[i],
                                kekulize=False, ignoreHs=True, symbol=symbols[i])
        else:
            subimg = MolToImage(mol, (200, 200), legend=legends[i], highlightAtoms=matchings[i],
                                background=colors[i], mol_adder=MolDrawing.AddMol, symbol=symbols[i])
        img.paste(subimg, (col * 200, row * 200))
    return img


def draw_mol_slider(mol, legend, matching, symbol, color, values, index):
    import matplotlib.pyplot as plt
    from matplotlib.widgets import Slider
    if op.isfile('/home/flo/temp.png'):
        os.remove('/home/flo/temp.png')
    subimg = MolToImage(mol, (200, 200), legend=legend, highlightAtoms=matching, symbol=symbol, background=color,
                        mol_adder=MolDrawing.AddMol)
    # Then we will integrate a slider showing the relative ranking of the molecule
    rank = float(len(values[:index + 1])) / len(values)
    alpha_axis = plt.axes([0.37, 0.59, 0.14, 0.018], axisbg='grey')
    Slider(alpha_axis, '1(+)', 0, 1, valinit=rank, facecolor='w', valfmt='(-)%.0f')
    # Manually set to look good on the svg: [0.37, 0.59, 0.14, 0.018] for alpha_axis and 180, 70 for subimg
    # Manually set to look good on the png: [0.37, 0.59, 0.14, 0.018] for alpha_axis and 247, 172 for subimg
    # Now we need to combine subimg and slider in one image
    plt.figimage(subimg, 247, 172)
    plt.show()
    plt.savefig('/home/flo/temp.png')  # , bbox_inches='tight')


def MyAddMol(drawer, mol, centerIt=True, molTrans=None, drawingTrans=None,
             highlightAtoms=(), confId=-1, flagCloseContactsDist=2,
             highlightMap=None, ignoreHs=False, highlightBonds=(), **kwargs):
    """Set the molecule to be drawn.

    Parameters:
      hightlightAtoms -- list of atoms to highlight (default [])
      highlightMap -- dictionary of (atom, color) pairs (default None)

    Notes:
      - specifying centerIt will cause molTrans and drawingTrans to be ignored
    """
    conf = mol.GetConformer(confId)
    if 'coordScale' in kwargs:
        drawer.drawingOptions.coordScale = kwargs['coordScale']

    drawer.currDotsPerAngstrom = drawer.drawingOptions.dotsPerAngstrom
    drawer.currAtomLabelFontSize = drawer.drawingOptions.atomLabelFontSize
    if centerIt:
        drawer.scaleAndCenter(mol, conf, ignoreHs=ignoreHs)
    else:
        if molTrans is None:
            molTrans = (0, 0)
        drawer.molTrans = molTrans
        if drawingTrans is None:
            drawingTrans = (0, 0)
        drawer.drawingTrans = drawingTrans

    font = Font(face=drawer.drawingOptions.atomLabelFontFace, size=drawer.currAtomLabelFontSize)

    obds = None
    if not mol.HasProp('_drawingBondsWedged'):
        # this is going to modify the molecule, get ready to undo that
        obds = [x.GetBondDir() for x in mol.GetBonds()]
        Chem.WedgeMolBonds(mol, conf)

    includeAtomNumbers = kwargs.get('includeAtomNumbers', drawer.drawingOptions.includeAtomNumbers)
    drawer.atomPs[mol] = {}
    drawer.boundingBoxes[mol] = [0] * 4
    drawer.activeMol = mol
    drawer.bondRings = mol.GetRingInfo().BondRings()
    labelSizes = {}
    for atom in mol.GetAtoms():
        labelSizes[atom.GetIdx()] = None
        if ignoreHs and atom.GetAtomicNum() == 1:
            drawAtom = False
        else:
            drawAtom = True
        idx = atom.GetIdx()
        pos = drawer.atomPs[mol].get(idx, None)
        if pos is None:
            pos = drawer.transformPoint(conf.GetAtomPosition(idx) * drawer.drawingOptions.coordScale)
            drawer.atomPs[mol][idx] = pos
            if drawAtom:
                drawer.boundingBoxes[mol][0] = min(drawer.boundingBoxes[mol][0], pos[0])
                drawer.boundingBoxes[mol][1] = min(drawer.boundingBoxes[mol][1], pos[1])
                drawer.boundingBoxes[mol][2] = max(drawer.boundingBoxes[mol][2], pos[0])
                drawer.boundingBoxes[mol][3] = max(drawer.boundingBoxes[mol][3], pos[1])
        if not drawAtom:
            continue
        nbrSum = [0, 0]
        for bond in atom.GetBonds():
            nbr = bond.GetOtherAtom(atom)
            if ignoreHs and nbr.GetAtomicNum() == 1:
                continue
            nbrIdx = nbr.GetIdx()
            if nbrIdx > idx:
                nbrPos = drawer.atomPs[mol].get(nbrIdx, None)
                if nbrPos is None:
                    nbrPos = drawer.transformPoint(conf.GetAtomPosition(nbrIdx) * drawer.drawingOptions.coordScale)
                    drawer.atomPs[mol][nbrIdx] = nbrPos
                    drawer.boundingBoxes[mol][0] = min(drawer.boundingBoxes[mol][0], nbrPos[0])
                    drawer.boundingBoxes[mol][1] = min(drawer.boundingBoxes[mol][1], nbrPos[1])
                    drawer.boundingBoxes[mol][2] = max(drawer.boundingBoxes[mol][2], nbrPos[0])
                    drawer.boundingBoxes[mol][3] = max(drawer.boundingBoxes[mol][3], nbrPos[1])
            else:
                nbrPos = drawer.atomPs[mol][nbrIdx]
            nbrSum[0] += nbrPos[0] - pos[0]
            nbrSum[1] += nbrPos[1] - pos[1]

        iso = atom.GetIsotope()

        # Write down atoms that are not C, that have some charge, or if we ask specifically for writing
        labelIt = (not drawer.drawingOptions.noCarbonSymbols or
                   atom.GetAtomicNum() != 6 or
                   atom.GetFormalCharge() != 0 or
                   atom.GetNumRadicalElectrons() or
                   includeAtomNumbers or
                   iso or
                   atom.HasProp('molAtomMapNumber') or
                   atom.GetDegree() == 0)

        orient = ''
        if labelIt:
            baseOffset = 0
            if includeAtomNumbers:
                symbol = str(atom.GetIdx())
                # noinspection PyUnusedLocal
                symbolLength = len(symbol)
            else:
                base = atom.GetSymbol()
                symbolLength = len(base)
                # i_v = atom.calcImplicitValence()
                # nHs = atom.GetTotalNumHs()
                # if nHs > 0:
                #    if nHs > 1:
                #        hs='H<sub>%d</sub>'%nHs
                #        symbolLength += 1 + len(str(nHs))
                #    else:
                #        hs ='H'
                #        symbolLength += 1
                # else:
                # hs = ''
                hs = ''
                chg = atom.GetFormalCharge()
                if chg != 0:
                    if chg == 1:
                        chg = '+'
                    elif chg == -1:
                        chg = '-'
                    elif chg > 1:
                        chg = '+%d' % chg
                    elif chg < - 1:
                        chg = '-%d' % chg
                    symbolLength += len(chg)
                else:
                    chg = ''
                if chg:
                    chg = '<sup>%s</sup>' % chg

                if atom.GetNumRadicalElectrons():
                    rad = drawer.drawingOptions.radicalSymbol * atom.GetNumRadicalElectrons()
                    rad = '<sup>%s</sup>' % rad
                    symbolLength += atom.GetNumRadicalElectrons()
                else:
                    rad = ''

                isotope = ''
                isotopeLength = 0
                if iso:
                    isotope = '<sup>%d</sup>' % atom.GetIsotope()
                    isotopeLength = len(str(atom.GetIsotope()))
                    symbolLength += isotopeLength
                mapNum = ''
                mapNumLength = 0
                if atom.HasProp('molAtomMapNumber'):
                    mapNum = ':' + atom.GetProp('molAtomMapNumber')
                    mapNumLength = 1 + len(str(atom.GetProp('molAtomMapNumber')))
                    symbolLength += mapNumLength
                deg = atom.GetDegree()
                # This should be done in a better way in the future:
                # 'baseOffset' should be determined by getting the size of 'isotope' and the size of 'base', or the
                # size of 'mapNum' and the size of 'base' (depending on 'deg' and 'nbrSum[0]') in order to determine
                # the exact position of the base
                if deg == 0:
                    if periodicTable.GetElementSymbol(atom.GetAtomicNum()) in (
                            'O', 'S', 'Se', 'Te', 'F', 'Cl', 'Br', 'I', 'At'):
                        symbol = '%s%s%s%s%s%s' % (hs, isotope, base, chg, rad, mapNum)
                    else:
                        symbol = '%s%s%s%s%s%s' % (isotope, base, hs, chg, rad, mapNum)
                elif deg > 1 or nbrSum[0] < 1:
                    symbol = '%s%s%s%s%s%s' % (isotope, base, hs, chg, rad, mapNum)
                    baseOffset = 0.5 - (isotopeLength + len(base) / 2.) / symbolLength
                else:
                    symbol = '%s%s%s%s%s%s' % (rad, chg, hs, isotope, base, mapNum)
                    baseOffset = -0.5 + (mapNumLength + len(base) / 2.) / symbolLength
                if deg == 1:
                    if abs(nbrSum[1]) > 1:
                        islope = nbrSum[0] / abs(nbrSum[1])
                    else:
                        islope = nbrSum[0]
                    if abs(islope) > .3:
                        if islope > 0:
                            orient = 'W'
                        else:
                            orient = 'E'
                    elif abs(nbrSum[1]) > 10:
                        if nbrSum[1] > 0:
                            orient = 'N'
                        else:
                            orient = 'S'
                    else:
                        orient = 'C'
                if highlightMap and idx in highlightMap:
                    color = highlightMap[idx]
                elif highlightAtoms and idx in highlightAtoms:
                    color = drawer.drawingOptions.selectColor
                else:
                    color = drawer.drawingOptions.elemDict.get(atom.GetAtomicNum(), (0, 0, 0))
                try:
                    # noinspection PyProtectedMember
                    labelSize = drawer._drawLabel(symbol, pos, baseOffset, font, color=color, orientation=orient)
                except TypeError:
                    # noinspection PyNoneFunctionAssignment
                    labelSize = drawLabel(drawer, symbol, pos, font, color, orientation=orient)
                labelSizes[atom.GetIdx()] = [labelSize, orient]

            for bond in mol.GetBonds():
                atom, idx = bond.GetBeginAtom(), bond.GetBeginAtomIdx()
                nbr, nbrIdx = bond.GetEndAtom(), bond.GetEndAtomIdx()
                pos = drawer.atomPs[mol].get(idx, None)
                nbrPos = drawer.atomPs[mol].get(nbrIdx, None)
                if highlightBonds and bond.GetIdx() in highlightBonds:
                    width = 2.0 * drawer.drawingOptions.bondLineWidth
                    color = drawer.drawingOptions.selectColor
                    color2 = drawer.drawingOptions.selectColor
                elif highlightAtoms and idx in highlightAtoms and nbrIdx in highlightAtoms:
                    width = 2.0 * drawer.drawingOptions.bondLineWidth
                    color = drawer.drawingOptions.selectColor
                    color2 = drawer.drawingOptions.selectColor
                elif highlightMap is not None and idx in highlightMap and nbrIdx in highlightMap:
                    width = 2.0 * drawer.drawingOptions.bondLineWidth
                    color = highlightMap[idx]
                    color2 = highlightMap[nbrIdx]
                else:
                    width = drawer.drawingOptions.bondLineWidth
                    if drawer.drawingOptions.colorBonds:
                        color = drawer.drawingOptions.elemDict.get(atom.GetAtomicNum(), (0, 0, 0))
                        color2 = drawer.drawingOptions.elemDict.get(nbr.GetAtomicNum(), (0, 0, 0))
                    else:
                        color = drawer.drawingOptions.defaultColor
                        color2 = color
                try:
                    # noinspection PyProtectedMember
                    drawer._drawBond(bond, atom, nbr, pos, nbrPos, conf, color=color, width=width, color2=color2,
                                     labelSize1=labelSizes[idx], labelSize2=labelSizes[nbrIdx])
                except TypeError:
                    # noinspection PyProtectedMember
                    drawer._drawBond(bond, atom, nbr, pos, nbrPos, conf, color=color, width=width, color2=color2)
                except KeyError:
                    if pos is not None and nbrPos is not None:
                        # noinspection PyProtectedMember
                        drawer._drawBond(bond, atom, nbr, pos, nbrPos, conf, color=color, width=width, color2=color2)

            # if we modified the bond wedging state, undo those changes now
            if obds:
                for i, d in enumerate(obds):
                    mol.GetBondWithIdx(i).SetBondDir(d)

            if flagCloseContactsDist > 0:
                tol = flagCloseContactsDist * flagCloseContactsDist
                for i, atomi in enumerate(mol.GetAtoms()):
                    try:
                        pi = np.array(drawer.atomPs[mol][i])
                        for j in range(i + 1, mol.GetNumAtoms()):
                            try:
                                pj = np.array(drawer.atomPs[mol][j])
                                d = pj - pi
                                dist2 = d[0] * d[0] + d[1] * d[1]
                                if dist2 <= tol:
                                    drawer.canvas.addCanvasPolygon(((pi[0] - 2 * flagCloseContactsDist,
                                                                     pi[1] - 2 * flagCloseContactsDist),
                                                                    (pi[0] + 2 * flagCloseContactsDist,
                                                                     pi[1] - 2 * flagCloseContactsDist),
                                                                    (pi[0] + 2 * flagCloseContactsDist,
                                                                     pi[1] + 2 * flagCloseContactsDist),
                                                                    (pi[0] - 2 * flagCloseContactsDist,
                                                                     pi[1] + 2 * flagCloseContactsDist)),
                                                                   color=(1., 0, 0), fill=False, stroke=True)
                            except KeyError:
                                continue
                    except KeyError:
                        continue


def MycreateCanvas(size, color='white'):
    # noinspection PyProtectedMember
    useAGG, useCairo, Canvas = Draw._getCanvas()
    if useAGG or useCairo:
        try:
            import Image
        except ImportError:
            from PIL import Image
        img = Image.new("RGBA", size, color)
        canvas = Canvas(img)
        return img, canvas
    return None, None


# noinspection PyUnusedLocal
def MolToImage(mol, size=(300, 300), kekulize=True, wedgeBonds=True,
               fitImage=False, options=None, canvas=None, background='white',
               values=None, index=None, canvas_creater=MycreateCanvas,
               mol_adder=MyAddMol, **kwargs):
    """ returns a PIL image containing a drawing of the molecule

    Keyword arguments:
    kekulize -- run kekulization routine on input `mol` (default True)
    size -- final image size, in pixel (default (300,300))
    wedgeBonds -- draw wedge (stereo) bonds (default True)
    highlightAtoms -- list of atoms to highlight (default [])
    highlightMap -- dictionary of (atom, color) pairs (default None)
    highlightBonds -- list of bonds to highlight (default [])
    background: can be given as a string (default = 'white') or as a color code: (250,0,0,0)
    """
    if not mol:
        raise ValueError('Null molecule provided')
    if canvas is None:
        img, canvas = canvas_creater(size, color='white')
    else:
        img = None

    if options is None:
        options = DrawingOptions()
    if fitImage:
        options.dotsPerAngstrom = int(min(size) / 10)
    options.wedgeDashedBonds = wedgeBonds
    drawer = MolDrawing(canvas=canvas, drawingOptions=options)

    if kekulize:
        from rdkit import Chem
        mol = Chem.Mol(mol.ToBinary())
        Chem.Kekulize(mol)

    # noinspection PyArgumentList
    if not mol.GetNumConformers():
        from rdkit.Chem import AllChem
        AllChem.Compute2DCoords(mol)

    if 'legend' in kwargs:
        legend = kwargs['legend']
        del kwargs['legend']
    else:
        legend = ''

    if 'symbol' in kwargs:
        symbol = kwargs['symbol']
        del kwargs['symbol']
    else:
        symbol = ''

    # Add a colored border
    if not background == 'white':
        position = [(0, 0), (size[0], 0), (size[0], size[1]), (0, size[1])]
        canvas.addCanvasPolygon(position, fill=True, color=background)
        position = [(5, 5), (size[0] - 5, 5), (size[0] - 5, size[1] - 5), (5, size[1] - 5)]
        canvas.addCanvasPolygon(position, fill=True, color=(250, 250, 250))

    mol_adder(drawer, mol, **kwargs)

    if legend:
        # noinspection PyUnusedLocal
        bbox = drawer.boundingBoxes[mol]
        pos = size[0] / 2, int(.94 * size[1]), 0  # the 0.94 is extremely empirical
        # canvas.addCanvasPolygon(((bbox[0],bbox[1]),(bbox[2],bbox[1]),(bbox[2],bbox[3]),(bbox[0],bbox[3])),
        # color=(1,0,0),fill=False,stroke=True)
        # canvas.addCanvasPolygon(((0,0),(0,size[1]),(size[0],size[1]),(size[0],0) ),
        # color=(0,0,1),fill=False,stroke=True)
        font = Font(face='sans', size=12)
        canvas.addCanvasText(legend, pos, font)

    if symbol:
        # Let's put it (them) in the top left corner of the canas
        pos = size[0] / 10, size[1] / 10, 0
        font = Font(face='sans', size=12)
        canvas.addCanvasText(symbol, pos, font)

    if kwargs.get('returnCanvas', False):
        return img, canvas, drawer
    else:
        canvas.flush()
        return img


def drawLabel(drawer, label, pos, font, color, **kwargs):
    if color is None:
        color = drawer.drawingOptions.defaultColor
    x1 = pos[0]
    y1 = pos[1]
    drawer.canvas.addCanvasText(label, (x1, y1), font, color, **kwargs)


if __name__ == '__main__':
    pattern = 'c(CN)(cc)c(c)O'
    molecules = from_feat_back_to_mols_faster('lab', pattern)
    draw_in_a_grid_aligned_according_to_pattern([mol[0] for mol in molecules[:10]], pattern, '/home/flo/bla4.png',
                                                classes=[1, 1, 0, 0, 1, 0, 1, 0, 0, 0],
                                                values=[[0.9, 0.87, 0.8, 0.67, 0.65, 0.63, 0.55, 0.23] for mol in
                                                        molecules[:10]],
                                                index=np.random.randint(7, size=10))
    # p = Chem.MolFromSmiles(pattern)
    # if p is None:
    #    p = Chem.MolFromSmiles(pattern, sanitize=False)
    #    print p
    # drawColorbar([0.9, 0.87, 0.8, 0.67, 0.65, 0.63, 0.55, 0.23], 1, '/home/flo/bla.png')
