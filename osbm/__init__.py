from . import utils

module_path = __file__.split("__init__.py")[0]
__version__ = open(module_path + "VERSION").read()


def download_gist(url: str = None):

    assert url is not None, "Please provide an URL"
    raise NotImplementedError


def set_matplotlib_rc(setting: str ="notebook"):
    raise NotImplementedError
