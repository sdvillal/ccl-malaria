# coding=utf-8
"""Some utilities for model evaluation."""
from itertools import combinations
from math import ceil
import numpy as np
from scipy.stats import kendalltau as scipy_kendalltau
try:
    from minioscail.common.faster import fast_auc as rocauc
except ImportError:
    from sklearn.metrics import roc_auc_score as rocauc


###############################################
# Evaluation metrics
###############################################

def enrichment_at(actual, scores, percentage=0.1, target=1, bigger_is_better=True):
    """Calculate the enrichment of the ranking induced by the given scores.
    Here enrichment is defined as percentage of actives recovered when screening a percentage of the compounds.

    Parameters:
      - actual: the class assignments (ground truth)
      - scores: the scores given by the predictor to the examples
      - percentage: the "screening" percentage in [0,1] (defaults to 0.1, i.e., a 10%)
      - target: the target class to compute the enrichment on
      - bigger_is_better: whether higher scores rank first or not

    Examples:
    >>> scores = np.array([1, 2, 3, 4, 5, 6, 7, 8])
    >>> labels = np.array([1, 1, 1, 0, 1, 0, 1, 0])
    >>> enrichment_at(labels, scores)
    0.0
    >>> enrichment_at(labels, scores, bigger_is_better=False)
    0.2
    >>> enrichment_at(labels, scores, percentage=0.5, bigger_is_better=False)
    0.6
    """
    num_actives = np.sum(actual == target)
    if 0 == num_actives:
        raise Exception('There are no actives in the dataset, cannot compute enrichment')
    num_at_percent = int(ceil(percentage * len(scores)))
    sorted_actuals = actual[np.argsort(-scores if bigger_is_better else scores)]
    num_actives_at_percent = np.sum(sorted_actuals[:num_at_percent] == target)
    tp = float(num_actives_at_percent)
    enrichment = tp / num_actives
    return enrichment
    # TODO: look at croc and yard, implement them in numpy-land
    # TODO: use fast roc to compute all enrichments fast


def max_enrichment_at(actual, percentage=0.1, target=1, bigger_is_better=True):
    return enrichment_at(actual, actual, percentage=percentage, target=target, bigger_is_better=bigger_is_better)


def kendalltau(a, b):
    """Computes the kendall-tau rank correlation between two arrays of scores."""
    return scipy_kendalltau(a, b)[0]


def kendalltau_all(scores, log=True):
    """
    Computes the kendall-tau statistic for all pairs {score_name, scores}.
    Scores must be a list of pairs (id, scores).
    """
    kts = {}
    for a, b in combinations(scores, 2):
        kt = kendalltau(a[1], b[1])
        kts[(a[0], b[0])] = kt
        if log:
            print a[0], b[0], '%.4f' % kt
    return kts

###############################################
# Rankings manipulation
###############################################


def scores2rankings(X, smaller_score_ranks_higher=True, eps=1E-9):
    """Translate scores to rankings, averaging the ranking of tied groups.

    Examples:
    >>> a = np.array([0.3, 0.4, 0.2, 0.5])
    >>> scores2rankings(a)
    array([ 1.,  2.,  0.,  3.])
    >>> scores2rankings(a, smaller_score_ranks_higher=False)
    array([ 2.,  1.,  3.,  0.])

    If X is a matrix, returns the rankings *by columns*, as in:
    >>> b = np.array([[0.3, 0.4, 0.2, 0.5], [0.2, 0.5, 0.1, 0.5]])
    >>> scores2rankings(b)
    array([[ 1. ,  0. ,  1. ,  0.5],
           [ 0. ,  1. ,  0. ,  0.5]])
    >>> scores2rankings(b, smaller_score_ranks_higher=False)
    array([[ 0. ,  1. ,  0. ,  0.5],
           [ 1. ,  0. ,  1. ,  0.5]])

    See also scipy.stats.rankdata:
      http://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.rankdata.html#scipy.stats.rankdata
    """
    def solve_ties(x, order, ranks):
        """Averages the rankings of scores that are considered the same."""
        i = 0
        while i < len(x):  # Terrible looking loop, check efficiency...
            #Find out how many group under this score...
            group_size = 1
            for j in xrange(i + 1, len(x)):
                if abs(x[order[j]] - x[order[j-1]]) < eps:  # N.B. we *do chain*
                                                            # (i.e., two elements in a group can be farther
                                                            # than eps away as long as there is a "sequence"
                                                            # of close-enough scores linking them).
                    group_size += 1
                else:
                    break
            #Re-rank according to the mean rank for all the group
            base_rank = ranks[order[i]]
            ranksum = (group_size * base_rank + (group_size * (group_size - 1) / 2.))
            rankmean = ranksum / group_size
            for j in xrange(i, i + group_size):
                ranks[order[j]] = rankmean
            i += group_size
        return ranks

    try:
        from minioscail.common.faster import fast_solve_ties
        solve_ties = fast_solve_ties
    except Exception:
        pass

    def s2r(x):
        """Converts scores in a vector to rankings, solving ties."""
        sorted_indices = np.argsort(x if smaller_score_ranks_higher else -x).astype(np.uint32)
        ranks = np.empty(len(x))
        ranks[sorted_indices] = np.arange(len(x))
        #Solve ties...
        return solve_ties(x, sorted_indices, ranks)

    if len(X.shape) == 1:
        return s2r(X).astype(np.int)
    else:
        # Probably we could use argsort axis here too, but lets go for loop ATM
        ranks = [s2r(X[:, col]) for col in xrange(X.shape[1])]
        return np.array(ranks).T


def rank_sort(scores, to_sort=(), reverse=False, select_top=None):
    """Gives rankings that we can use for reordering, but do not take care of ties."""
    if reverse:
        scores = -scores
    ranks = np.argsort(scores).argsort()

    def apply_ranking(a):
        ordered = np.empty_like(a)
        ordered[ranks] = a
        return ordered[:select_top] if select_top is not None else ordered
    return ranks, map(apply_ranking, to_sort)


###############################################
# Cross-validation
###############################################

# TODO: Stream cross validation (once/if ever the online framework is on).


def cv_splits(num_points, Y, num_folds, rng, stratify=True, banned_train=None, banned_test=None):
    """
    (Stratified) cross-validation.

    Parameters:
      - num_points: the number of elements to split
      - Y: the group for each point (e.g. the class, the score, the source...)
      - num_folds: the number of splits
      - rng: an instance of a python/numpy random number generator
      - stratify: if True, a best effort is carried to keep consistent the Y proportions in each split
      - banned_train: a list of indices to not include in train (e.g. non-target class for OCC) or None
      - banned_test: a list of indices to not include in test or None

    Returns a function that maps a fold index (from 0 to num_folds-1) to the indices of its train/test instances.
    """
    permutation = rng.permutation(num_points)
    if stratify:
        permutation = permutation[np.argsort(Y[permutation])]
    folds = [permutation[base::num_folds] for base in range(num_folds)]
    seed = rng.randint(1024*1024*1024)
    banned_train = set() if not banned_train else set(banned_train)
    banned_test = set() if not banned_test else set(banned_test)

    def cver(fold):
        if fold >= num_folds:
            raise Exception('There are not so many folds (requested fold %d for a cross-val of %d folds)' %
                            (fold, num_folds))
        rng = np.random.RandomState(seed)
        # Now select and permute test indices, removing the banned items.
        # N.B. Order of operations matter...
        test_indices = folds[fold][rng.permutation(len(folds[fold]))]
        ban_in_train = set(test_indices) | banned_train
        train_indices = [train_i for train_i in rng.permutation(num_points)
                         if train_i not in ban_in_train]
        test_indices = [test_i for test_i in test_indices if test_i not in banned_test]
        return train_indices, test_indices

    return cver