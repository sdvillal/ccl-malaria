# coding=utf-8
"""Centralized access to command line applications."""
import argh

from ccl_malaria.logregs_analysis import do_logreg_submissions
from ccl_malaria.molscatalog import init
from ccl_malaria.results import final_merged_submissions


def main():
    parser = argh.ArghParser()
    parser.add_commands([
        init,
        do_logreg_submissions,
        final_merged_submissions
    ])
    parser.dispatch()


if __name__ == '__main__':
    main()
