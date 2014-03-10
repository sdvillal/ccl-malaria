# coding=utf-8
#cython: boundscheck=False, wraparound=False, cdivision=True
#cython: profile=False
"""Cythonizing some code that needs to be faster..."""
cimport cython
import numpy as np
cimport numpy as np


@cython.embedsignature(True)
def fast_roc(double[:] scores,
             int[:] labels,
             int P=-1,
             int N=-1,
             int target_class=1,
             double eps=1E-9):
    """
    Computes roc-curve points as in Fawcett.
    In my benchmarks this is fast enough, and 40 times faster than sklearn's roc_curve.

    Parameters:
      - scores: a numpy double array with the scores
      - labels: a numpy int array with the classes
      - P: the number of positives (<=0 to recompute)
      - N: the number of negatives (<=0 to recompute)
      - target_class: what is the class for which we are computing the ROC?
      - eps: smallest difference to consider two scores are different

    Returns a (num__roc_points x 2) numpy array with the fpr in column 0 and tpr in column 1
    """
    #TODO: generic float types
    #TODO: return thresholds
    #TODO: compute AUC at the same time (useful?, it is not that costly anyway...)
    cdef double tp = 0
    cdef double fp = 0
    cdef double[:, :] curve
    cdef unsigned int num_points = 0
    cdef double last_score
    cdef unsigned int i
    cdef unsigned int[:] sort_order = np.argsort(scores).astype(np.uint32)[::-1]

    #Same number of scores and labels...
    if len(scores) != len(labels):
        raise Exception('Labels and scores lengths disagree')

    #Number of positives, number of negatives
    if 0 >= P or 0 >= N:
        P=0; N=0
        for i in range(len(labels)):
            if labels[i] == target_class:
                P += 1
            else:
                N += 1
    if 0 == P:
        raise Exception('There are not positives')
    if 0 == N:
        raise Exception('There are not negatives')

    #The curve (fpr, tpr, threshold)
    curve_np = np.empty((len(scores) + 1, 2), dtype=np.float)
    curve = curve_np

    #Make sure (0,0) is in the curve
    last_score = scores[sort_order[0]] + 1

    for i in sort_order:
        if abs(scores[i] - last_score) > eps:
            curve[num_points, 0] = fp / N
            curve[num_points, 1] = tp / P
            num_points += 1
            last_score = scores[i]

        if labels[i] == target_class:
            tp += 1
        else:
            fp += 1

    #Make sure 1, 1 is in the curve
    curve[num_points, 0] = fp / N
    curve[num_points, 1] = tp / P
    num_points += 1

    return curve_np[:num_points]


def fast_solve_ties(double[:] x,
                    unsigned int[:] order,
                    double[:] ranks,
                    double eps=1E-9):
    """Averages the rankings of scores that are considered the same."""
    # TODO: avoid double indirection would be faster?
    cdef unsigned int i, j
    cdef unsigned int group_size
    cdef double baserank
    cdef double ranksum
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

# TODO: use more generic types to play well with whatever numpy might be using, read cython doc