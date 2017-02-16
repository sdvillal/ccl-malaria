# coding=utf-8
"""Centralized access to command line applications."""
import argh


def main():
    parser = argh.ArghParser()

    # Initialize the molecules catalog
    from ccl_malaria.molscatalog import init
    parser.add_commands([
        init,
    ])

    # Generate features
    def add_featurizer_commands(parser, namespace='features'):
        from ccl_malaria.features import cl, ecfps, ecfps_mp, munge_ecfps, rdkfs
        parser.add_commands([cl,
                             ecfps,
                             ecfps_mp,
                             munge_ecfps,
                             rdkfs],
                            namespace=namespace)
    add_featurizer_commands(parser)

    # Fitting logistic regressions
    def add_logreg_commands(parser, namespace='logregs'):
        from ccl_malaria.logregs_fit import cl, fit
        from ccl_malaria.logregs_analysis import submit
        parser.add_commands([cl,
                             fit,
                             submit],
                            namespace=namespace)
    add_logreg_commands(parser)

    # Fitting trees
    def add_tree_commands(parser, namespace='trees'):
        from ccl_malaria.trees_fit import fit
        from ccl_malaria.trees_analysis import submit
        parser.add_commands([fit,
                             submit],
                            namespace=namespace)
    add_tree_commands(parser)

    # Final dumb blending
    from ccl_malaria.results import merge_submissions
    parser.add_commands([
        merge_submissions
    ])
    parser.dispatch()


if __name__ == '__main__':
    main()
