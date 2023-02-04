"""
This is the docs for the osbm package.
"""

from .utils import *
from . import imaging

module_path = __file__.split("__init__.py", maxsplit=1)[0]
__version__ = open(module_path + "VERSION", encoding="utf-8").read().strip()
del module_path
