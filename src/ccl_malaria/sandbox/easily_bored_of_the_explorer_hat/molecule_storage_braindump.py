# coding=utf-8

# Stuff we "need" to save per molecule:
#  - original smiles
#  - original meta-data (assays results)
#  - fingerprints and other features
#  - neighbors
#  - model scores and participation in model evaluation
#
# Use DB or raw files?
#
# Inefficiencies with many files (e.g. linear search for a directory)
# http://linux-xfs.sgi.com/projects/xfs/papers/xfs_white/xfs_white_paper.html


def shitza2():
    pass

    # A quick own format:
    # One 1D big-fat numpy array memmapped in memory, plus an index CPD -> (base, size)
    # Some interesting reading:
    #   - http://useless-factor.blogspot.co.at/2011/05/why-not-mmap.html
    #   - http://nextmovesoftware.com/blog/2012/10/17/lazy-file-reading-with-mmap/
    #
    # Nice post comparing memmap and pytables
    # https://www.mail-archive.com/pytables-users@lists.sourceforge.net/msg01861.html
    # Memmapped files can be the way to go, but numpy does not support incremental writing.
    # For storing rdkit molecules we do not even need numpy's machinery
    # Let's just use plain python memmapped files

    # Option 3: hdfstore
    # How to:
    #  - Make it incremental for writing?
    #  - How to allow incremental reading and queries?
    #  - Add an index for molid?
    #  - Make it space-efficient?
    #
    # See: https://github.com/PyTables/PyTables/issues/198
    #      http://comments.gmane.org/gmane.comp.python.pytables.user/3148
    # Something from Umit:
    #   http://stackoverflow.com/questions/16907195/
    #   what-is-a-better-approach-of-storing-and-querying-a-big-dataset-of-meteorologica
    # VLTypes penalties and the like:
    #   http://www.hdfgroup.org/HDF5/doc/TechNotes/VLTypes.html

    # Option 4: A DB like mongodb
    #           might be ok, but overkilling at the moment


if __name__ == '__main__':
    shitza2()