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


if __name__ == "__main__":
    poisssan_disc_sampling = PoissanDiscSampling(
        radius=30, size=(1920, 1080), number_of_trials=3
    )
    points = poisssan_disc_sampling.generate()
    print(points)
