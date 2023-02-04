"""
The module that contains the utility functions.
"""

import os
import io
import math
import time
import random
import importlib
from typing import Optional, Callable

import numpy as np
import pandas as pd


def is_kaggle() -> bool:
    """
    Check if the machine is running in Kaggle.
    """
    return os.path.exists("/kaggle/working")


def measure_time(func: Callable) -> Callable:
    """decorator to measure time of a function"""

    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(f" The function {func.__name__} took {end - start} seconds")
        return result

    return wrapper


def add_kaggle_token(token: Optional[str] = None, path: Optional[str] = None):
    """
    Authenticate to Kaggle using a token.

    Args:
        token: Entire kaggle.json file content

        path: Path to kaggle.json file

    """
    assert (
        token is not None or path is not None
    ), "Either token or path must be provided"

    os.makedirs("~/.kaggle", exist_ok=True)
    if token is None:
        with open(path, "r", encoding="utf-8") as file:
            token = file.read()

    with open("~/.kaggle/kaggle.json", "w", encoding="utf-8") as file:
        file.write(token)

    os.chmod("~/.kaggle/kaggle.json", 0o600)


def is_colab() -> bool:
    """
    Check if the code is running in Google Colab.
    """
    return bool(importlib.util.find_spec("google.colab"))


def get_gpu_info() -> pd.DataFrame:
    """
    Get the nvidia GPU information as pandas dataframe.
    """
    # pylint: disable=line-too-long
    command = "nvidia-smi --query-gpu=index,name,memory.total,memory.free,memory.used,count,utilization.gpu,utilization.memory --format=csv"

    command_output = os.popen(command).read()

    # remove units from column names
    command_output = command_output.replace(" [MiB]", "")
    command_output = command_output.replace(" [%]", "")

    if command_output == "":
        raise EnvironmentError("nvidia-smi is not installed")
    elif command_output[:21] == "NVIDIA-SMI has failed":
        return None

    return pd.read_csv(io.StringIO(command_output), sep=", ")


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
                    distance_square = self.get_distance_square(
                        self.samples[int(sample_index)], candidate
                    )

                    if distance_square < self.radius**2:
                        return False

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
