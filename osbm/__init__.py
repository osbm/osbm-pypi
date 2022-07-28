
print(__file__)
print(__name__)
# print current path
import os
print(os.getcwd())
print()
__version__ = open("osbm/VERSION").read()


def poisson_disc_sampling(size=(1000, 1000), k=30, radius=10, seed=42):

    raise NotImplementedError


def download_gist(url: str = None):

    assert url is not None, "Please provide an URL"


def set_matplotlib_rc(setting: str ="notebook"):
    raise NotImplementedError