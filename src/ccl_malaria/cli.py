# coding=utf-8
"""Centralized access to command line applications."""
import argh
from ccl_malaria.molscatalog import init


def main():
    parser = argh.ArghParser()
    parser.add_commands([
        init,
    ])
    parser.dispatch()


if __name__ == '__main__':
    main()
