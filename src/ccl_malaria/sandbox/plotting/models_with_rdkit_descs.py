import h5py
import numpy as np
import argh
import os.path as op
try:
    import cPickle as pickle
except ImportError:
    import pickle
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_auc_score
from ccl_malaria.molscatalog import MalariaCatalog
from ccl_malaria import MALARIA_EXPS_ROOT
from minioscail.common.misc import home


# --- 1/ Read the saved hd5
# contains a dataset 'rdkdescs', a dataset 'fnames', and a dataset '


def get_X_y(hd5_file=op.join(home(), 'labrdkf.h5')):
    with h5py.File(hd5_file, mode='r') as h5:
        X = h5['rdkdescs'][:]
        molids = h5['molids'][:]
        y = MalariaCatalog().labels(molids, as01=True, asnp=True)
        rows_without_nan = ~np.isnan(X).any(axis=1)
        rows_with_label = ~np.isnan(y)
        rows_to_keep = rows_without_nan & rows_with_label
        print('Removed with missing:', np.sum(~rows_without_nan))
        print('Removed without label:', np.sum(~rows_with_label))
        # noinspection PyStringFormat
        print('Used to train %d (%d positives)' % (np.sum(rows_to_keep), np.sum(y == 1.)))
        return X[rows_to_keep], y[rows_to_keep], molids[rows_to_keep]


# --- 2/ Train cross-validated RF of different numbers of trees, compare them
def train_rf_models(hd5=op.join(home(), 'labrdkf.h5'),
                    num_folds=10, seed=0,
                    num_trees=(1000, 2000, 4000),
                    n_jobs_rf=40):
    X, y, _ = get_X_y(hd5)
    cv_scores = []
    for ntree in num_trees:
        clf = RandomForestClassifier(n_estimators=ntree, max_depth=None, min_samples_split=1,
                                     random_state=seed, n_jobs=n_jobs_rf)
        scores = cross_val_score(clf, X, y, scoring='roc_auc', cv=num_folds, n_jobs=1)
        hauc = np.mean(scores)
        cv_scores.append(hauc)
        # noinspection PyStringFormat
        print('Cross-validation AUC for %i trees: %.2f' % (ntree, hauc))
    return cv_scores


def exp_2(hd5=op.join(home(), 'labrdkf.h5'), n_jobs_rf=40, seed=0, num_folds=10, n_trees=10000):
    X, y, _ = get_X_y(hd5)
    # model1 = RandomForestClassifier(n_estimators=n_trees, max_depth=None, min_samples_split=1,
    #                                 random_state=seed, n_jobs=n_jobs_rf, compute_importances=True)
    # print 'Cross-validating the RF model....'
    # scores1 = cross_val_score(model1, X, y, scoring='roc_auc', cv=num_folds, n_jobs=1)
    # hauc1 = np.mean(scores1)
    # print 'AUC RF model: %.2f'%hauc1
    # train the full model
    # feat_imp1 = model1.fit(X, y).feature_importances_
    # with (open(op.join(home(), 'RF_4000trees.pkl'), 'w') as writer1,
    #       open(op.join(home(), 'RF_4000trees_feat_imp.pkl'), 'w') as writer2):
    #    pickle.dump(model1, writer1)
    #    pickle.dump(feat_imp1, writer2)

    model2 = ExtraTreesClassifier(n_estimators=n_trees, max_depth=None, min_samples_split=1, random_state=seed,
                                  n_jobs=n_jobs_rf)
    print('Cross-validating the ExtremelyRandomizedTrees model....')
    scores2 = cross_val_score(model2, X, y, scoring='roc_auc', cv=num_folds, n_jobs=1)
    hauc2 = np.mean(scores2)
    print('AUC ERT model: %.2f' % hauc2)
    # train the full model
    # noinspection PyUnresolvedReferences
    feat_imp2 = model2.fit(X, y).feature_importances_
    with open(op.join(home(), 'ERT_10000trees.pkl'), 'w') as writer1, \
            open(op.join(home(), 'ERT_10000trees_feat_imp.pkl'), 'w') as writer2:
        pickle.dump(model2, writer1)
        pickle.dump(feat_imp2, writer2)
    # model3 = GradientBoostingClassifier(n_estimators=500, learning_rate=1.0, max_depth=1, random_state=seed)
    # print 'Cross-validating the GradientBoostingClassifier model....'
    # scores3 = cross_val_score(model3, X, y, scoring='roc_auc', cv=num_folds, n_jobs=n_jobs_rf)
    # hauc3 = np.mean(scores3)
    # print 'AUC GBC model with 500 estimators: %.2f'%hauc3
    # # train the full model
    # feat_imp3 = model3.fit(X, y).feature_importances_
    # with (open(op.join(home(), 'GBC_500trees.pkl'), 'w') as writer1,
    #       open(op.join(home(), 'GBC_500trees_feat_imp.pkl'), 'w') as writer2):
    #     pickle.dump(model3, writer1)
    #     pickle.dump(feat_imp3, writer2)


def xys(fold):
    Xall, yall, molids_all = get_X_y()
    Xtrain = np.array([Xall[i] for i in range(len(yall)) if i not in fold])
    ytrain = np.array([yall[i] for i in range(len(yall)) if i not in fold])
    Xtest = np.array([Xall[i] for i in fold])
    ytest = np.array([yall[i] for i in fold])
    return Xtrain, ytrain, Xtest, ytest


def give_cross_val_folds(y, num_folds, seed=0, stratified=True):
    if num_folds < 2:
        raise Exception('The number of folds must be greater than 1')
    if num_folds > len(y):
        raise Exception('The number of folds (%d) is greater than the number of elements (%d)' % (num_folds, len(y)))
    rng = np.random.RandomState(seed)
    indices = rng.permutation(len(y))
    if stratified:
        indices = np.array(sorted(indices, key=lambda i: y[i]))
    return tuple([indices[fold::num_folds] for fold in range(num_folds)])


def sklearn_wrapper_cv(folds, classifier, cl_id, dest_dir):
    for i, fold in enumerate(folds):
        print('Fold %i' % i)
        Xtr, ytr, Xte, yte = xys(fold)
        classifier.fit(Xtr, ytr)
        with open(op.join(dest_dir, cl_id + '_' + str(i) + '.pkl'), 'w') as writer:
            pickle.dump(classifier, writer)
        scores = classifier.predict(Xte)
        auc = roc_auc_score(yte, scores)
        print('AUC for fold %i: %.2f' % (i, auc))
        print('********************')


def exp3(cv_folds=10, cv_seed=0, classifier_seed=0, n_estimators=10000, n_jobs=40):
    _, y, _ = get_X_y()
    folds = give_cross_val_folds(y, cv_folds, seed=cv_seed)
    classifier = RandomForestClassifier(n_estimators=n_estimators, max_depth=None, min_samples_split=1,
                                        random_state=classifier_seed, n_jobs=n_jobs)
    sklearn_wrapper_cv(folds, classifier, 'RF_10000trees', op.join(MALARIA_EXPS_ROOT, 'rf_exp3'))


if __name__ == '__main__':
    parser = argh.ArghParser()
    parser.add_commands([train_rf_models, exp_2, exp3])
    parser.dispatch()
