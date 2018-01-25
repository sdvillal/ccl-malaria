import tinyarray as ta
import array

# from cytoolz import partition_all, partition
# Also, for example, boltons chunked

# Check base64; or encode directly in a string buffer that we encode binary, or...

# print(murmur('cataract'), murmur('periti'), murmur('periti', seed=3))
# exit(33)

# print('loading')
# rdk_hashes = _read_full_file(rdk_hashes, '/home/santi/testit/rdk.array')
# cansmi_hashes = _read_full_file(cansmi_hashes, '/home/santi/testit/cansmi.array')
#
# print('np')
# rdk = np.frombuffer(rdk_hashes, dtype=rdk_hashes.typecode)
# cansmi = np.frombuffer(cansmi_hashes, dtype=cansmi_hashes.typecode)
#
# print(len(cansmi))
# print(cansmi.max())
# print(rdk.max())

# cansmi % 255 == 1

# print('numba')
# from numba import jit, vectorize, boolean, uint32
# from blist import blist


# @vectorize([boolean(uint32, uint32, uint32)],
#            nopython=True,
#            target='parallel')
# def shard(x, s=0, ns=255):
#     return x % ns == s
#
# print('sharding')
# # sharded = shard(cansmi, 0, 255)
# sharded = shard(rdk, 200, 255)
#
# print('indexing')
# df = pd.DataFrame({'cs': cansmi[sharded], 'rdk': rdk[sharded]})
# print(df.groupby(['rdk', 'cs']).size().reset_index())
# exit(22)

# from pandas import kh


# @jit
# def shard(rdk, cansmi, shard=0, num_shards=255):
#     x = array.array('I')
#     for i in range(len(rdk)):
#         cs = cansmi[i]
#         if cs % num_shards == shard:
#             x.append(cs)
#             x.append(cansmi[i])

# This is probably running in python mode
# print('numba')
# shard(rdk, cansmi)

# print('df')
# df = pd.DataFrame({'rdk': rdk_hashes, 'cansmi': cansmi_hashes})
#
# print('gc')
# import gc
# rdk_hashes = None
# cansmi_hashes = None
# gc.collect()
#
# print('groupby')
# print(df.groupby(['cansmi', 'rdk']).size().reset_index().shape)

# print(df)

# exit(22)


# import mmap


class FptDonor:

    def __init__(self, dest_dir='/home/santi/testit'):
        self.fpt = ext_based_open(op.join(dest_dir, 'fingerprints.bin'), 'rb')
        self.molids = OrderedDict()
        self.mmap = mmap.mmap(self.fpt.fileno(), 0, access=mmap.ACCESS_READ)
        with ext_based_open(op.join(dest_dir, 'molids.txt'), 'rt') as molids:
            for molid in molids:
                molid, pos = molid.strip().split()
                if molid not in self.molids:
                    self.molids[molid] = []
                self.molids[molid].append(int(pos))

    def _read(self, pos=None):
        x = array.array('I')
        if pos is not None:
            self.mmap.seek(pos)
        x.fromfile(self.mmap, 1)  # number of features (redundant once read)
        for _ in range(x[0]):
            x.fromfile(self.mmap, 2)          # hash, count_in_mol
            x.fromfile(self.mmap, 3 * x[-1])  # (center, radius, rdkit_hash) * count
        return x

    def get(self, molid):
        return [self._read(pos) for pos in self.molids.get(molid, ())]


# fpts = FptDonor()
# print(len(fpts.molids))
# import numpy as np
# for molid in fpts.molids:
#     efp, ffp = fpts.get(molid)
#     efp = interpret_fpt(efp)
#     print(efp[4::3])
#     exit(22)
# exit(22)

# parse_new()

# t0 = time.time()
# for molnum, line in enumerate(line_iterator(up_to_n(to_new(), n=10000000000))):
#     if molnum > 0 and molnum % 10000 == 0:
#         taken = time.time() - t0
#         print('%d lines (%.2fs, %.2fmol/s)' % (molnum, taken, molnum / taken))
# The vanilla parser we wrote makes iteration go from 23000 to 2300 lines per second
# exit(22)


def feats(return_i=False):
    with open('/home/santi/testit/features.txt', 'rt') as reader:
        for i, cansmi in enumerate(reader):
            if i > 0 and i % 100000 == 0:
                print(i)
            yield cansmi.strip() if not return_i else (cansmi.strip(), (i,))


# from marisa_trie import Trie, RecordTrie
#
# trie = RecordTrie('I')
#
# # Generate the trie, save
# fmt = 'I'
# # trie = Trie(feats())
# trie = RecordTrie(fmt, feats(True))
# trie.save('/home/santi/testit/features-int.trie')
# # N.B. in order to be able to really append more, we will need
# # to store some more keys; the order given by the trie won't
# # be enough.
#
# # Reload the trie
# trie = Trie()
# trie.load('/home/santi/testit/features.trie')

#
# Generate an index; we probably will need to pass by some indirections
# index = array.array('I')
# for i, f in enumerate(feats()):
#     if i > 0 and i % 100000 == 0:
#         print(i)
#     index.append(trie.get(f))
#
#
# exit(22)


# Tries for the win:
#   http://kmike.ru/python-data-structures/
#   https://github.com/pytries
# SMILES uses ASCII as its alphabet
#   http://stackoverflow.com/questions/5891453/is-there-a-python-library-that-contains-a-list-of-all-the-ascii-characters
#   https://en.wikipedia.org/wiki/Simplified_molecular-input_line-entry_system
# import string
# import datrie
# trie = datrie.Trie(string.printable)
# trie[feat['cansmi']] = 1
# return trie


# Collisions for the win, injective or not...
# n_jobs = 6
# results = Parallel(n_jobs=n_jobs)(
#     delayed(parse_wfps)(
#         start=start,
#         step=n_jobs
#     )
#     for start in range(n_jobs)
# )
# import pickle
# with open('/home/santi/mola-hash.pkl', 'wb') as reader:
#     pickle.dump(results, reader)
#
# exit(22)


# import pandas as pd
# maybe use an specialised int set
# what is the sparsity?
# rdk_hashes = array.array('I')
# cansmi_hashes = array.array('I')
# for i, fpt in enumerate(parse_new()):
#     if i > 0 and i % 10000 == 0:
#         print(i)
#     for cansmi_hash, x in interpret_fpt(fpt):
#         rdk_hashes.extend(x[:, 2])
#         cansmi_hashes.extend([cansmi_hash] * len(x))
#
# df = pd.DataFrame({'rdk_hash': rdk_hashes, 'cansmi_hash': cansmi_hashes},
#                   dtype=np.uint32)
# df.to_pickle('/home/santi/hashes.pkl')

# with open('/home/santi/testit/rdk.array', 'wb') as writer:
#     rdk_hashes.tofile(writer)
#
# with open('/home/santi/testit/cansmi.array', 'wb') as writer:
#     cansmi_hashes.tofile(writer)
