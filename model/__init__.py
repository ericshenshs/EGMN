"""model/__init__.py.

This file is part of the watch-time prediction codebase.
Primary role: model.
Defines one or more PyTorch modules that map feature dicts to predictions.
"""

from __future__ import absolute_import
from model.wd import WideAndDeep
from model.cread import Cread
from model.d2q import D2Q
from model.tpm import TPM
from model.egmn import EGMN
