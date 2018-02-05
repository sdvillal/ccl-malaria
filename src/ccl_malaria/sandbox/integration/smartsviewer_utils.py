# coding=utf-8
"""Pythonic interface to smartsviewer:
   http://www.smartsviewer.de/
   http://www.biosolveit.de/SMARTStools/

Output of running "SmartsViewer -h" for version 0.9.0
    SYNOPSIS:
        smartsviewer [OPTIONS]

    OPTIONS:
        -h              : Print this help text.
        -s <smarts>     : The input smarts for visualization. Either -s or -f have to be given.
        -f <file>       : A file containing the smarts. Either -s or -f have to be given.
        -o <outfile>    : Prints the diagram to <outfile>
                          possible file formats: .pdf, .ps, .svg
        -d <w> <h>      : Dimension of the  .svg output file. (100 <= w|h <= 1000)
        -p              : Set default parameter.
                        : Eight values have to be given, range and defaults:
                        : 1. Display options: 0-3 <0>
                        :   (0=Complete Visualization, 1= IDs, 2= Element symbols, 3=Structure Diagram-like)
                        : 2. Default bond options: 0-1 <0>
                        :   (0=Single bond, 1=Single or aromatic bond
                        : 3. Show Userlabels?: 0-1 <0>
                        :   (0=No, 1=Yes)
                        : 4. Trim-errorcheck?: 0-1 <0>
                        :   (0=Yes, 1=No)
                        : 5. Trim-simplification?: 0-1 <0>
                        :   (0=Yes, 1=No)
                        : 6. Trim-interpretation?: 0-1 <0>
                        :   (0=Yes, 1=No)
                        : 7. Show Legend?: 0-3 <0>
                        :   (0=No, 1=Dynamic legend, 2=Static Legend 3=Both)
                        : 8. Print SMARTS string into picture?: 0-1 <0>
                        :   (0=YES, 1=NO)
"""
import os.path as op


class SmartsViewerRunner(object):

    def __init__(self,
                 sv_root=None,
                 w=500, h=500,  # Dimension of the  .svg output file. (100 <= w|h <= 1000)
                 display_style=0,            # (0=Complete Visualization,
                                             #  1= IDs, 2= Element symbols, 3=Structure Diagram-like)
                 also_aromatic_bonds=False,  # (0=Single bond, 1=Single or aromatic bond)
                 user_labels=False,          # (0=No, 1=Yes)
                 trim_errorcheck=True,       # (0=Yes, 1=No)
                 trim_simplification=True,   # (0=Yes, 1=No)
                 trim_interpretation=True,   # (0=Yes, 1=No)
                 legend_style=0,             # (0=No, 1=Dynamic legend, 2=Static Legend 3=Both)
                 show_smarts=True):          # (0=YES, 1=NO)
        super(SmartsViewerRunner, self).__init__()
        self.sv_root = sv_root if sv_root is not None else \
            op.abspath(op.join(op.dirname(__file__), 'thirdparty', 'smartsviewer'))
        self.w = w
        self.h = h
        self.display_style = display_style
        self.also_aromatic = also_aromatic_bonds
        self.user_labels = user_labels
        self.trim_errorcheck = trim_errorcheck
        self.trim_simplification = trim_simplification
        self.trim_interpretation = trim_interpretation
        self.legend_style = legend_style
        self.show_smarts = show_smarts
        self.cl = op.join(self.sv_root, 'SmartsViewer -d %d %d -p %d %d %d %d %d %d %d %d' %
                                        (self.w, self.h, self.display_style,
                                         0 if not self.also_aromatic else 1,
                                         0 if not self.user_labels else 1,
                                         0 if self.trim_errorcheck else 1,
                                         0 if self.trim_simplification else 1,
                                         0 if self.trim_interpretation else 1,
                                         self.legend_style,
                                         0 if self.show_smarts else 1))

    def depict(self, smarts, dest_file):

        # Smiles or smarts
        # Destination image (.pdf, .ps or .svg)

        cl = self.cl + ' -s \"%s\" -o \"%s\"' % (smarts, dest_file)
        # Use subprocess.call or the like
        import subprocess
        subprocess.call(cl, shell=True)  # TODO: eat the output

        return cl  # to remove, just to debug


if __name__ == '__main__':
    svr = SmartsViewerRunner()
    print(svr.depict('CCCC#CC1=CC(=CC(=C1)C#CC2=CC(=C(C=C2C#CC(C)(C)C)C3OCCO3)C#CC(C)(C)C)C#CCCC',
                     op.join(op.expanduser('~'), 'example-smartsviewer.svg')))
