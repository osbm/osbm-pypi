"""
The module that contains the utility functions.
"""

import importlib
import os
import io
import random
import math
import numpy as np
import pandas as pd
import ipywidgets as widgets
from IPython.core.display import display


# display(HTML('<h1>Hello, world!</h1>'))


def download_gist(url: str = None):
    assert url is not None, "Please provide an URL"
    # TODO: implement this function
    raise NotImplementedError


def set_matplotlib_rc(setting: str = "notebook"):
    # TODO: implement this function

    raise NotImplementedError


def is_colab() -> bool:
    """
    Check if the code is running in Google Colab.
    """
    return bool(importlib.util.find_spec("google.colab"))


def is_kaggle() -> bool:
    """
    Check if the code is running in Kaggle.
    """
    # assert importlib.util.find_spec("kaggle")
    return os.getcwd() == "/kaggle/working"


def get_gpu_info():
    """
    Get the nvidia GPU information as pandas dataframe.
    """

    command = "nvidia-smi --query-gpu=index,name,uuid,memory.total,memory.free,memory.used,count,utilization.gpu,utilization.memory --format=csv"

    command_output = os.popen(command).read()

    # remove units from column names
    command_output = command_output.replace(" [MiB]", "")
    command_output = command_output.replace(" [%]", "")

    if command_output == "":
        raise EnvironmentError("nvidia-smi is not installed")
    elif command_output[:21] == "NVIDIA-SMI has failed":
        return None

    df = pd.read_csv(io.StringIO(command_output), sep=", ")
    return df


def gpu_name():
    df = get_gpu_info()

    if df is None:
        return

    return list(df["name"])


def download_huggingface_repository():

    raise NotImplementedError()


def auth_kaggle():
    # assert is_colab(), "This function is only available in Colab"
    # get file from the user

    main_display = widgets.Output()

    def show_it(inputs):
        with main_display:
            main_display.clear_output()
            display(list(inputs["new"].keys())[-1])

    uploader = widgets.FileUpload(
        accept=".json",  # Accepted file extension e.g. '.txt', '.pdf', 'image/*', 'image/*,.pdf'
        multiple=False,  # True to accept multiple files upload else False
    )

    uploader.observe(show_it, names="value")

    # make dir for kaggle
    os.makedirs("~/.kaggle", exist_ok=True)

    with open("~/.kaggle/kaggle.json", "a", encoding="utf-8") as output_file:
        filename = list(uploader.value.keys())[0]
        content = upload.value[filename]["content"]
        output_file.write(content)

    # chmod 600 kaggle.json
    os.chmod("~/.kaggle/kaggle.json", 600)


def auth_huggingface():
    # TODO: implement this function
    raise NotImplementedError()


def rle2mask(rle, mask_shape):
    ''' takes a space-delimited RLE string in column-first order
    and turns it into a 2d boolean numpy array of shape mask_shape '''

    mask = np.zeros(np.prod(mask_shape), dtype=bool) # 1d mask array
    rle = np.array(rle.split()).astype(int) # rle values to ints
    starts = rle[::2]
    lengths = rle[1::2]
    for s, l in zip(starts, lengths):
        mask[s:s+l] = True
    return mask.reshape(np.flip(mask_shape)).T # flip because of column-first order


def mask2rle(mask):
    ''' takes a 2d boolean numpy array and turns it into a space-delimited RLE string '''

    mask = mask.T.reshape(-1) # make 1D, column-first
    mask = np.pad(mask, 1) # make sure that the 1d mask starts and ends with a 0
    starts = np.nonzero((~mask[:-1]) & mask[1:])[0] # start points
    ends = np.nonzero(mask[:-1] & (~mask[1:]))[0] # end points
    rle = np.empty(2 * starts.size, dtype=int) # interlacing...
    rle[0::2] = starts # ...starts...
    rle[1::2] = ends - starts # ...and lengths
    rle = ' '.join([ str(elem) for elem in rle ]) # turn into space-separated string
    return rle


class PoissanDiscSampling:
    """
    Poisson disc sampling algorithm.
    """

    def __init__(self, radius=30, size=(1920, 1080), seed=42, number_of_trials=30):
        """
        :param radius: radius of the disc
        """
        self.samples = []
        self.active_list = []
        self.radius = radius
        self.cell_size = self.radius / np.sqrt(2)
        self.sample_region_size = np.array(size)
        self.grid = np.zeros(
            (
                math.ceil(self.sample_region_size[0] / self.cell_size),
                math.ceil(self.sample_region_size[1] / self.cell_size),
            )
        )
        print(self.grid.shape)

        self.total_number_of_trials = number_of_trials
        self.margin = 100
        self.rng = random.Random()
        self.rng.seed(seed)

    def generate(self):
        """
        Generate the samples.
        """
        initial_sample = np.array(
            (self.sample_region_size / 2).astype(int)
        )  # start point
        self.active_list = [initial_sample]

        while len(self.active_list) > 0:
            active_index = self.rng.randrange(len(self.active_list))
            spawn_centre = self.active_list[active_index]
            candidate_accepted = False

            for _ in range(self.total_number_of_trials):
                angle = self.rng.random() * math.pi * 2
                direction = np.array([math.sin(angle), math.cos(angle)])
                candidate = spawn_centre + np.floor(
                    direction * self.rng.randrange(self.radius, 2 * self.radius)
                ).astype(int)
                if self.is_valid(candidate):

                    self.samples.append(candidate)

                    self.grid[tuple((candidate / self.cell_size).astype(int))] = len(
                        self.samples
                    )
                    candidate_accepted = True
                    self.active_list.append(candidate)
                    break

            if candidate_accepted is False:
                self.active_list.pop(active_index)

        return np.array(self.samples)

    def is_valid(self, candidate):
        """
        Check if the candidate point is valid by checking if it is within the
        sample region and if it is not too close to any other point.
        """
        # check if the candidate is outside of border
        if (candidate < self.sample_region_size).all() and (candidate > 0).all():
            cell = (candidate / self.cell_size).astype(int)
            search_start = np.maximum(0, cell - 2)
            search_end = np.minimum(self.grid.shape, cell + 2)

            diff = search_end - search_start
            for current_cell in np.ndindex(tuple(diff)):
                current_cell += search_start

                sample_index = self.grid[tuple(current_cell)] - 1

                if sample_index != -1:
                    """
                    distance_square = self.get_distance_square(
                        self.samples[int(sample_index)], candidate
                    )

                    if distance_square < self.radius**2:
                        return False
                    """
                    return (
                        np.linalg.norm(self.samples[int(sample_index)] - candidate)
                        < self.radius
                    )

            return True

        else:
            # Out of borders
            return False

    def get_distance_square(self, first_sample, second_sample):
        """
        Calculate the distance between two points.
        """
        return np.sum((first_sample - second_sample) ** 2)
        # difference = abs(a - b)
        # return (difference**2).sum()
