# coding=utf-8
"""Centralized access to command line applications."""
import argh


def main():
    parser = argh.ArghParser()

    # Initialize the molecules catalog
    def add_molscatalog_commands():
        from ccl_malaria.molscatalog import init
        parser.add_commands([init], namespace='catalog')
    add_molscatalog_commands()

    # Generate features
    def add_featurizer_commands():
        from ccl_malaria.features import cl, morgan, morgan_mp, munge_morgan, rdkfs, rdkfs_mp
        parser.add_commands([morgan,
                             morgan_mp,
                             munge_morgan,
                             rdkfs,
                             rdkfs_mp,
                             cl],
                            namespace='features')
    add_featurizer_commands()

    # Fitting logistic regressions
    def add_logreg_commands():
        from ccl_malaria.logregs_fit import cl, fit
        from ccl_malaria.logregs_analysis import submit
        parser.add_commands([fit, submit, cl],
                            namespace='logregs')
    add_logreg_commands()

    # Fitting trees
    def add_tree_commands():
        from ccl_malaria.trees_fit import fit
        from ccl_malaria.trees_analysis import submit
        parser.add_commands([fit, submit],
                            namespace='trees')
    add_tree_commands()

    # Final dumb blending
    def add_blending_commands():
        from ccl_malaria.results import merge_submissions
        parser.add_commands([merge_submissions], namespace='blending')
    add_blending_commands()

    parser.dispatch()


if __name__ == '__main__':
    main()
