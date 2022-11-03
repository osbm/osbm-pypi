from osbm import utils
import numpy as np
import itertools


class TestPoissonDiscSampling:
    def test_generate_points_2d(self):
        config = {
            "radius": 10,
            "size": [100, 100],
            "number_of_trials": 100,
            "seed": 42,
        }
        sampling = utils.PoissanDiscSampling(**config)
        points = sampling.generate()

        for a, b in itertools.combinations(points, 2):
            assert np.linalg.norm(a - b) >= config["radius"]

    def test_generate_points_3d(self):
        config = {
            "radius": 10,
            "size": [100, 100],
            "number_of_trials": 100,
            "seed": 42,
        }
        sampling = utils.PoissanDiscSampling(**config)
        points = sampling.generate()

        for a, b in itertools.combinations(points, 2):
            assert np.linalg.norm(a - b) > config["radius"]

    def test_generate_points_5d(self):
        raise NotImplementedError()

    def test_generate(self):
        raise NotImplementedError()
