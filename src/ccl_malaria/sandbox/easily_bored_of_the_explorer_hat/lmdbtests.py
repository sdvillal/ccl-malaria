# coding=utf-8
"""Playing with lightning MDB."""
import hashlib
from time import time
import lmdb

print('generating keys...')
start = time()
keys = [hashlib.sha256(str(i)).hexdigest() for i in xrange(50000000)]
print(time() - start)


# with open('/home/santi/shitza2.kk', 'w') as writer:
#         # txt.put(hashlib.sha256(str(i)).hexdigest(), str(i))
#         # writer.write('%s,%s\n' % (hashlib.sha256(str(i)).hexdigest(), str(i)))
#         txt.get().hexdigest())
# txt.commit()

# print 'querying...'
# env = lmdb.open('/home/santi/shitza.lmdb', map_size=1024*1024*1024*1024)
# txt = env.begin(write=False, buffers=True)
# start = time()
# print time() - start
#

def onthefly_s2i(keys_stream, dbdir='/home/santi/lmdb-tmp'):
    """Converting strings to indices, as they are seen, without keeping all in memory."""
    env = lmdb.open(dbdir, map_size=1024*1024*1024*1024)
    with env.begin(write=True, buffers=True) as txt:
        for key in keys_stream:
            if txt.get(key, default=None) is None:
                size = txt.stat()['entries']
                txt.put(key, str(size))
                if size % 1000 == 0:
                    print(size)


def onthefly_s2i2(keys_stream):
    d = {}
    for key in keys_stream:
        if not key in d:
            size = len(d)
            d[key] = size
            if size % 1000 == 0:
                print(size)
    return d

start = time()
onthefly_s2i2(keys)
print(time() - start)
