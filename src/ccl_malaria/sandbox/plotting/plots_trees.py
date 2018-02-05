# coding=utf-8
from __future__ import print_function, division
from collections import defaultdict
import os.path as op

import numpy as np
from ccl_malaria.trees_analysis import trees_results_to_pandas

from ccl_malaria.trees_fit import MALARIA_TREES_EXPERIMENT_ROOT
from minioscail.common.eval import kendalltau_all
from minioscail.common.misc import ensure_dir


def summary():
    """An example on how to manage OOB results."""
    # for result in results:
    #     print result.model_setup_id(), result.oob_auc()
    #     molids = result.ids('lab') + result.ids('amb')
    #     scores = np.vstack((result.scores('lab'), result.scores('amb')))
    #     print len(molids), len(scores)

    df = trees_results_to_pandas()
    directory = op.join(MALARIA_TREES_EXPERIMENT_ROOT, 'analysis')
    pics_dir = op.join(directory, 'figures')
    ensure_dir(directory)
    ensure_dir(pics_dir)

    print(df.columns)

    def aucs(df):
        aucss = []
        models = []
        stds = []
        for numtrees, gdf in df.groupby(['model_num_trees']):
            auc = gdf.oob_auc.mean()
            std = gdf.oob_auc.std()
            print('numtrees=%d, AUC=%.3f +/- %.3f' % (int(numtrees), auc, std))
            models.append(numtrees)
            aucss.append(auc)
            stds.append(std)
        return np.array(models), np.array(aucss), np.array(stds)

    def enrichments(df):
        enrichs = []
        models = []
        stds = []
        for numtrees, gdf in df.groupby(['model_num_trees']):
            enrich = gdf.oob_enrichment5.mean()
            std = gdf.oob_enrichment5.std()
            print('numtrees=%d, Enrichment=%.3f +/- %.3f' % (int(numtrees), enrich, std))
            models.append(numtrees)
            enrichs.append(enrich)
            stds.append(std)
        return np.array(models), np.array(enrichs), np.array(stds)

    def importances(df):
        f_names = df.result[0].f_names()
        f_importances = [res.f_importances() for res in df.result]
        return f_names, f_importances

    # noinspection PyUnusedLocal
    def f_importances_variability():
        # Do the f_importances change a lot in different seeds?
        f_names, f_importances = importances(df[df.model_num_trees == 6000])
        kendalltau_all(scores=list(enumerate(f_importances)))
        # What about the ranking of the molecules?
        kendalltau_all(scores=list(enumerate(res.scores(dset='lab')[:, 1] for res in
                                             df[((df.model_num_trees == 6000) & (df.model_seed < 2)) |
                                                ((df.model_num_trees == 100) & (df.model_seed < 2))].result)))

    # noinspection PyUnusedLocal
    def plot_auc_f_num_trees(df, show=True):
        # How does the AUC varies when we increase the number of trees?
        # How does it varies accross the different seeds?
        num_trees, aucss, stds = aucs(df)
        import matplotlib.pyplot as plt
        plt.errorbar(num_trees, aucss, yerr=stds)
        plt.ylim((0.6, 1))
        plt.xlabel('Number of trees')
        plt.ylabel('Average AUC for several random seeds')
        # Now let's add a little zoom to check what happens between AUC=0.9 and 1
        a = plt.axes([0.35, .25, .5, .3], axisbg='w')
        plt.errorbar(num_trees[aucss >= 0.92], aucss[aucss >= 0.92], yerr=stds[aucss >= 0.9])
        plt.setp(a, xticks=np.arange(0, np.max(num_trees[aucss >= 0.92])+100, 1000),
                 yticks=np.arange(0.92, np.max(aucss[aucss >= 0.92]) + 0.01, 0.02))
        if show:
            plt.show()
        plt.savefig(op.join(pics_dir, 'AUC_f_numtrees.png'), bbox_inches='tight')
        plt.savefig(op.join(pics_dir, 'AUC_f_numtrees.svg'), bbox_inches='tight')

    # noinspection PyUnusedLocal
    def plot_auc_enrichment_f_num_trees(df, show=True):
        num_trees, aucss, stds = aucs(df)
        _, enrichs, stds_enrich = enrichments(df)
        import matplotlib.pyplot as plt
        plt.errorbar(num_trees, aucss, yerr=stds)
        plt.errorbar(num_trees, enrichs, yerr=stds_enrich)
        plt.xlabel('Number of trees')
        plt.legend(['AUC', 'Enrichment'], loc='lower right')
        plt.savefig(op.join(pics_dir, 'AUC_and_enrichment_f_numtrees.png'), bbox_inches='tight')
        plt.savefig(op.join(pics_dir, 'AUC_and_enrichment_f_numtrees.svg'), bbox_inches='tight')
        if show:
            plt.show()

    # What will be the top molecules?
    # We will use the mean of uncalibrated scores for num_trees = 6000
    # noinspection PyUnusedLocal
    def final_scores(dset):
        results = df.result[df.model_num_trees == 6000]
        scores = np.mean([res.scores(dset) for res in results], axis=0)
        return scores

    def top_n_important_feats(df, num_trees=6000, n=10):
        f_names, f_importances = importances(df[df.model_num_trees == num_trees])
        # Average over the different seeds:
        f_importances = np.mean(f_importances, axis=0)
        # Little normalization to better see the differences in importances
        f_importances = (f_importances - np.min(f_importances)) / (np.max(f_importances) - np.min(f_importances))
        order = np.argsort(f_importances)
        f_names = np.array(f_names)
        f_names = f_names[order]
        f_importances = f_importances[order]
        return f_names[-n:], f_importances[-n:]

    # noinspection PyUnusedLocal
    def plot_how_many_times_in_top_n(df, n=10, show=True):
        num_experiments = 0
        occurrences_in_top_n = defaultdict(int)
        for numtrees, gdf in df.groupby(['model_num_trees']):
            num_experiments += 1
            f_names, _ = top_n_important_feats(df, num_trees=numtrees, n=n)
            for fn in f_names:
                occurrences_in_top_n[fn] += 1
        occurring_features = occurrences_in_top_n.keys()
        from matplotlib import pyplot as plt
        plt.plot(np.arange(1, len(occurring_features) + 1),
                 [occurrences_in_top_n[of]/float(num_experiments) for of in occurring_features], 'o')
        plt.ylim((0, 1.1))
        plt.xticks(np.arange(1, len(occurring_features) + 1), [of[6:] for of in occurring_features], rotation=25)
        plt.ylabel('Percentage of presence among the top %i features' % n)
        if show:
            plt.show()
        figure = plt.gcf()  # get current figure
        figure.set_size_inches(16, 6)
        plt.savefig(op.join(pics_dir, 'occurrences_features_top%i.png' % n), bbox_inches='tight', dpi=100)
        plt.savefig(op.join(pics_dir, 'occurrences_features_top%i.svg' % n), bbox_inches='tight', dpi=100)

    def plot_average_feat_importances(df, show=True):
        importancess = []
        f_names = None
        for numtrees, gdf in df.groupby(['model_num_trees']):
            f_names, f_importances = importances(df[df.model_num_trees == numtrees])
            # Average over the different seeds:
            f_importances = np.mean(f_importances, axis=0)
            # Little normalization to better see the differences in importances
            f_importances = (f_importances - np.min(f_importances)) / (np.max(f_importances) - np.min(f_importances))
            importancess.append(f_importances)
        av_imps = np.mean(np.array(importancess), axis=0)
        stds = np.std(np.array(importancess), axis=0)
        # Now we sort the features by importances, to get a nicer plot
        order = np.argsort(av_imps)
        av_imps = av_imps[order]
        stds = stds[order]
        f_names = f_names[order]
        import matplotlib.pyplot as plt
        plt.errorbar(np.arange(len(av_imps)), av_imps, yerr=stds, fmt='o')
        plt.xticks(np.arange(len(av_imps)), [f_name[6:] for f_name in f_names], rotation=90)
        plt.ylabel('Average normalized importance score')
        if show:
            plt.show()
        figure = plt.gcf()  # get current figure
        figure.set_size_inches(25, 17)
        plt.savefig(op.join(pics_dir, 'mean_feat_importances.png'))
        plt.savefig(op.join(pics_dir, 'mean_feat_importances.svg'))

    plot_average_feat_importances(df, show=True)
    # molids_lab = df.result[0].ids(dset='lab')
    # molids_amb = df.result[0].ids(dset='amb')
    #
    # scores_lab = final_scores(dset='lab')[:, 1]
    # scores_amb = final_scores(dset='amb')[:, 1]
    # scores = np.hstack((scores_lab, scores_amb))
    # print scores.shape
    # print np.sum(np.isfinite(scores))
    # print len(molids_lab + molids_amb)
    # arg_ranks = np.argsort(scores)[::-1]
    # my_rankis = scores2rankings(scores)
    #
    # molids = np.array(molids_lab + molids_amb)
    #
    # import matplotlib.pyplot as plt
    # plt.hist(scores[scores > 0.9], bins=200)
    # plt.show()
    #
    # mc = MalariaCatalog()
    # good_mols = molids[scores > 0.9]
    # for molid in good_mols:
    #     print molid, AllChem.MolToSmiles(mc.molid2mol(molid)), mc.label(molid)
    # img = MolsToGridImage(mc.molids2mols(good_mols), legends=mc.labels(good_mols))
    # img.save('/home/santi/shitza_rankings.png')

    # print rankings
    # ranked_mols = molids[rankings]
    # mc = MalariaCatalog()
    # mols = mc.molids2mols(ranked_mols[:50])
    # 1. in lab + amb
    # 2. in unl
    # 3. in scr


if __name__ == '__main__':
    summary()
