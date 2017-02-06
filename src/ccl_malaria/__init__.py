# coding=utf-8
"""
Teach-discovery-treat challenge1: Malaria High-throughput Screen
  See: http://www.tdtproject.org/challenge-1---malaria-hts.html
"""
import logging
import os.path as op
from minioscail.common.misc import ensure_dir

__version__ = '0.2-dev0'

# --- Paths and other constants.

# Make everything relative to the source location...
_THIS_PATH = op.abspath(op.dirname(__file__))  # maybe jump to pkgutils?
# Where the data resides
MALARIA_DATA_ROOT = op.abspath(op.join(_THIS_PATH, '..', '..', 'data'))
# The original downloaded files will come here
MALARIA_ORIGINAL_DATA_ROOT = op.join(MALARIA_DATA_ROOT, 'original')
ensure_dir(MALARIA_ORIGINAL_DATA_ROOT)
# Different indices (like molid -> smiles) come here
MALARIA_INDICES_ROOT = op.join(MALARIA_DATA_ROOT, 'indices')
ensure_dir(MALARIA_INDICES_ROOT)
# Experiment results come here
MALARIA_EXPS_ROOT = op.join(MALARIA_DATA_ROOT, 'experiments')
ensure_dir(MALARIA_EXPS_ROOT)

# --- Common logger for the malaria code.

_logger = logging.getLogger('malaria')
_logger.setLevel(logging.DEBUG)
debug = _logger.debug
info = _logger.info
warning = _logger.warning
error = _logger.error
logging.basicConfig()
