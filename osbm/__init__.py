from . import utils
import importlib

module_path = __file__.split("__init__.py", maxsplit=1)[0]
__version__ = open(module_path + "VERSION", encoding="utf-8").read().strip()


def download_gist(url: str = None):

    assert url is not None, "Please provide an URL"
    raise NotImplementedError


def set_matplotlib_rc(setting: str = "notebook"):
    raise NotImplementedError


def is_colab():
    raise NotImplementedError()


def is_kaggle():
    assert importlib.util.find_spec("kaggle")

    raise NotImplementedError()


def gpu_name():
    raise NotImplementedError()


def download_huggingface_repository():
    raise NotImplementedError()

def add_kaggle_token():
    assert is_colab(), "This function is only available in Colab"
    from google.colab import files

    uploaded = files.upload()
    # take uploaded file and and move it to ~/.kaggle/kaggle.json

    for file_name in uploaded.keys():
        if file_name.endswith(".json"):
            files.copy(file_name, "~/.kaggle/kaggle.json")
            break
