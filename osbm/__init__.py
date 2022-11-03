from .utils import *


module_path = __file__.split("__init__.py", maxsplit=1)[0]
__version__ = open(module_path + "VERSION", encoding="utf-8").read().strip()
