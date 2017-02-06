# coding=utf-8
"""A jumble of seemingly useful stuff."""
import collections
import datetime
import os
import inspect
import gc
import functools
from functools import partial
import re
import shutil
import unicodedata
import os.path as op
import cPickle as pickle
import itertools
import urllib
import numpy as np
from scipy.stats import nanmedian
from minioscail.integration.thirdparty import ntplib


def download(url, dest, overwrite=False, info=lambda msg: None):
    """Downloads a file and returns the path to the downloaded file."""
    if not op.isfile(dest) or overwrite:
        info('\tDownloading:\n\t  %s\n\tinto:\n\t  %s' % (url, dest))
        urllib.urlretrieve(url, dest)
        info('%s download succesful' % dest)
    return dest


def home():
    #What is the equivalent of user.home in python 3?
    return op.expanduser('~')


def ensure_writable_dir(path):
    """Ensures that a path is a writable directory."""
    if op.exists(path):
        if not op.isdir(path):
            raise Exception('%s exists but it is not a directory' % path)
        if not os.access(path, os.W_OK):
            raise Exception('%s is a directory but it is not writable' % path)
    else:
        try:
            os.makedirs(path)
        except:
            print 'Somebody else created the shit?'


def ensure_dir(path):
    ensure_writable_dir(path)


def make_temp_dir():
    import tempfile
    import os

    tempdir = tempfile.mktemp(prefix='vwrun__', dir='/tmp')
    os.makedirs(tempdir)

    return tempdir


def move_directories_with(
        root=op.join(op.expanduser('~'), '--kaggle', 'amazonsec', 'data', 'experiments', 'collected'),
        dest=op.join(op.expanduser('~'), '--kaggle', 'amazonsec', 'data', 'experiments', '__quarantine__', 'badfolds'),
        to_move=lambda dirpath, dirnames, filenames: 'BAD_FOLD_BAN.json' in filenames,
        symlink=False):

    def path2list(path):
        head, tail = op.split(path)
        if head and tail:
            return path2list(head) + [tail]
        if tail:
            return [tail]
        if head:
            return [head]
        return []

    def bottom_up_remove_empty_dirs(path):
        if op.isdir(path) and not os.listdir(path):
            shutil.rmtree(path)
            bottom_up_remove_empty_dirs(op.dirname(path))

    root = op.realpath(root)
    root_list = path2list(root)
    dest = op.realpath(dest)

    for dirpath, dirnames, filenames in os.walk(root):
        if to_move(dirpath, dirnames, filenames):
            #1- recreate hierarchy
            dirpath_list = path2list(dirpath)
            dest_root = op.join(dest, *dirpath_list[len(root_list):-1])
            ensure_writable_dir(dest_root)
            if symlink:
                os.symlink(op.realpath(dirpath), op.realpath(dest_root))
            else:
                #2- move directory
                shutil.move(dirpath, dest_root)
                #3- cleanup empty dirs
                bottom_up_remove_empty_dirs(op.dirname(dirpath))


def slugify(value, max_filename_length=200):
    """Create a valid filename from a bookmark title by:
      - Normalizing the string (see http://unicode.org/reports/tr15/)
      - Converting it to lowercase
      - Removing non-alpha characters
      - Converting spaces to hyphens
    Adapted from:
      - http://stackoverflow.com/questions/5574042/string-slugification-in-python
      - http://stackoverflow.com/questions/295135/turn-a-string-into-a-valid-filename-in-python
    See too: http://en.wikipedia.org/wiki/Comparison_of_file_systems#Limits.
    """
    value = unicode(value)
    value = unicodedata.normalize('NFKD', value).encode('ascii', 'ignore')
    value = unicode(re.sub('[^\w\s-]', '', value).strip().lower())
    value = unicode(re.sub('[-\s]+', '-', value))
    if max_filename_length is not None and len(value) > max_filename_length:
        return value[:max_filename_length]
    return value


class memoized(object):
    """ Decorator. Caches a function's return value each time it is called.
    If called later with the same arguments, the cached value is returned
    (not reevaluated).
    """
    def __init__(self, func):
        print 'Memo'
        self.func = func
        self.cache = {}

    def __call__(self, *args):
        print 'MemoCall'
        try:
            return self.cache[args]
        except KeyError:
            print 'Not memoized yet'
            value = self.func(*args)
            self.cache[args] = value
            return value
        except TypeError:
            print 'MemoCall uncacheable'
            # uncachable -- for instance, passing a list as an argument.
            # Better to not cache than to blow up entirely.
            return self.func(*args)

    def __repr__(self):
        """Return the function's docstring."""
        return self.func.__doc__

    def __get__(self, obj, objtype):
        """Support instance methods."""
        return functools.partial(self.__call__, obj)


def num(s):
    try:
        return int(s)
    except:
        try:
            return float(s)
        except:
            return s


def is_num(s):
    try:
        int(s)
        return True
    except:
        try:
            float(s)
            return True
        except:
            return False


def unpickle_or_none(path):
    try:
        with open(path) as reader:
            return pickle.load(reader)
    except Exception, _:
        return None


def one_line_pickle(obj, filepath):
    with open(filepath, 'w') as writer:
        pickle.dump(obj, writer, protocol=pickle.HIGHEST_PROTOCOL)


#######################
# Runtime introspection to get info on the calling functions
#######################

def giveupthefunc(frame=1):
    frame = inspect.currentframe(frame)
    code = frame.f_code
    globs = frame.f_globals
    functype = type(lambda: 0)
    funcs = []
    for func in gc.get_referrers(code):
        if type(func) is functype:
            if getattr(func, "func_code", None) is code:
                if getattr(func, "func_globals", None) is globs:
                    funcs.append(func)
                    if len(funcs) > 1:
                        return None
    return funcs[0] if funcs else None


def function_params(function):
    return inspect.getargspec(function)


def partial2call(p, positional=None, keywords=None):
    # FIXME: support built-in functions
    if not keywords:
        keywords = {}
    if not positional:
        positional = []
    if inspect.isfunction(p):
        args, _, _, defaults = inspect.getargspec(p)
        defaults = [] if not defaults else defaults
        args = [] if not args else args
        args_set = set(args)
        #Check that everything is fine...
        keywords = dict(zip(args[-len(defaults):], defaults) + keywords.items())  # N.B. order matters
        keywords_set = set(keywords.keys())
        if len(keywords_set - args_set) > 0:
            raise Exception('Some partial %r keywords are not parameters of the function %s' %
                            (keywords_set - args_set, p.__name__))
        if len(args_set) - len(keywords_set) < len(positional):
            raise Exception('There are too many positional arguments indicated '
                            'for the number of unbound positional parameters left.')
        return p.__name__, keywords
    if isinstance(p, partial):
        return partial2call(p.func,
                            positional=positional + list(p.args),                  # N.B. order matters
                            keywords=dict(p.keywords.items() + keywords.items()))  # N.B. order matters
    raise Exception('Only partials and functions are allowed, %r is none of them' % p)


###########
########### Iterables flatten
###########

def flatten(iterables):
    """Flattens an iterable of iterables, returning a generator."""
    return itertools.chain.from_iterable(iterables)


def lflatten(iterables):
    """Flattens an iterable of iterables, returning a list."""
    return list(flatten(iterables))


def flatten_multi(iterable):
    for element in iterable:
        if isinstance(element, collections.Iterable) and not isinstance(element, basestring):
            for sub in flatten_multi(element):
                yield sub
        else:
            yield element


def lflatten_multi(iterable):
    return list(flatten_multi(iterable))


def is_iterable(v):
    """Check whether an object is iterable or not."""
    try:
        iter(v)
    except:
        return False
    return True


def fill_missing_scores(scores):
    scores2 = scores.copy()
    scores2[~np.isfinite(scores2)] = nanmedian(scores2)
    return scores2


def internet_time(ntpservers=('ntp-0.imp.univie.ac.at', 'europe.pool.ntp.org')):
    """Makes a best effort to retrieve current UTC time from a reliable internet source.
    Returns a string like "Thu, 13 Mar 2014 11:35:41 UTC"
    """
    # Maybe also parse from, e.g., the webpage of the time service of the U.S. army
    try:
        for server in ntpservers:
            response = ntplib.NTPClient().request(server, version=3)
            dt = datetime.datetime.utcfromtimestamp(response.tx_time)
            return dt.strftime('%a, %d %b %Y %H:%M:%S UTC')
    except ImportError:
        return None