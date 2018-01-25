"""
Seeking for better performance on large {smiles: column_index} associative maps.
memmapped Marisa Tries are an interesting tool for multiproc and not too sucking performance without mem explosion

By extension, looking also at python dict alternatives for large {int: int} and {str: int} mappings.
Have an advanced cython wrapper for both sparsepp and standard unordered_map in the works.
"""
import string
import os.path as op
from tqdm import tqdm

import marisa_trie
import dawg
import datrie

from ccl_malaria.logregs_fit import malaria_logreg_fpt_providers


def trie_bench():

    #
    # Needs to be properly done, but we already know:
    #
    #   - In terms of disk space, no option beats vanilla gzipping
    #     on plain text data.
    #
    #   - Marisa trie can be useful to avoid memory problems: it can memmap
    #     making multiprocessing easy, it reduces dramatically the
    #     size. Speed is also fastest for this dataset (informal benchmarks
    #     not in here).
    #
    #   - The smallest memory footprint is given by plain marisa.
    #     But that is only suitable if we could let marisa decide the mapping
    #     to the column index and if we do not mind these indices changing
    #     each time we update the feature collection. We could always keep
    #     an extra int array with the actual index of the column and update it
    #     when we need to rewrite the trie upon new features arrival.
    #
    #   - Most probably, python dict is way faster on remapping - but that is also
    #     probably irrelevant here. We need to measure how much space these consume
    #     and how other alternatives (e.g. sparsepp) fare.
    #

    rf_lab, rf_amb, rf_unl, rf_scr = malaria_logreg_fpt_providers(None)

    def subs(n=None, return_index='simple'):
        substructures = rf_lab.mfm().substructures()
        if n is None:
            n = len(substructures)
        for i in tqdm(range(min(n, len(substructures)))):
            if return_index == 'simple':
                yield str(substructures[i]), i
            elif return_index == 'tuple':
                yield str(substructures[i]), (i,)
            elif return_index == 'no':
                yield str(substructures[i])
            else:
                raise ValueError('return_index must be one of ["simple", "tuple", "no"], it is %r' % return_index)

    # Uncompressed: 11697K. Can mmap.
    # Mapping to index becomes arbitrary, so we either need to define a format
    # with an auxiliary mapping (int by marisa -> int by insertion) and keep it
    # constant with new additions or we just not use this.
    # Need to measure speed.
    trie = marisa_trie.Trie()
    trie.load(op.expanduser('~/substructures.marisa'))
    for k, v in trie.iteritems():
        print(k, v)

    trie = marisa_trie.Trie(subs(return_index='no'))
    trie.save(op.expanduser('~/substructures.marisa'))

    fmt = 'I'
    trie = marisa_trie.RecordTrie(fmt, subs(return_index='tuple'))
    trie.save(op.expanduser('~/substructures.intMarisa'))

    trie = dawg.DAWG(subs(return_index='simple'))
    trie.save(op.expanduser('~/substructures.dawg'))

    trie = dawg.IntCompletionDAWG(subs(return_index='simple'))
    trie.save(op.expanduser('~/substructures.intCompletionDawg'))

    trie = dawg.IntDAWG(subs(return_index='simple'))
    trie.save(op.expanduser('~/substructures.intDawg'))

    trie = datrie.Trie(string.printable)
    for s, i in subs(return_index='simple'):
        trie[s] = i
    trie.save(op.expanduser('~/substructures.datrie'))

    trie = datrie.BaseTrie(string.printable)
    for s, i in subs(return_index='simple'):
        trie[s] = i
    trie.save(op.expanduser('~/substructures.basedatrie'))
