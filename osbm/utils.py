import numpy as np
import random, math


class poission_disc_sampling:
    def __init__(self, radius=30, size=(1920, 1080), seed=42, number_of_trials=30):

        self.samples = []
        self.activeList = []
        self.radius = radius
        self.cell_size = self.radius/np.sqrt(2)
        self.sample_region_size = np.array(size)
        self.grid = np.zeros((math.ceil(self.sample_region_size[0]/self.cell_size), math.ceil(self.sample_region_size[1]/self.cell_size)))
        print(self.grid.shape)

        self.total_number_of_trials = number_of_trials
        self.margin = 100
        self.rng = random.Random()
        self.rng.seed(seed)


    def __call__(self):
        initial_sample = np.array((self.sample_region_size / 2).astype(int)) # start point
        self.activeList = [initial_sample]

        while len(self.activeList) > 0:
            activeIndex = self.rng.randrange(len(self.activeList))
            spawnCentre = self.activeList[activeIndex]
            candidateAccepted = False

            for _ in range(self.total_number_of_trials):
                angle = self.rng.random() * math.pi * 2
                direction = np.array([math.sin(angle), math.cos(angle)])
                candidate = spawnCentre + np.floor(direction * self.rng.randrange(self.radius, 2*self.radius)).astype(int)
                if self.isValid(candidate):

                    self.samples.append(candidate)

                    self.grid[tuple((candidate/self.cell_size).astype(int))] = len(self.samples)
                    candidateAccepted = True
                    self.activeList.append(candidate)
                    break



            if candidateAccepted is False:
                self.activeList.pop(activeIndex)

        return self.samples

    def isValid(self, candidate):
        # check if the candidate is outside of border
        if (candidate < self.sample_region_size).all() and (candidate > 0).all():
            cell = (candidate/self.cell_size).astype(int)
            search_start = np.maximum(0, cell-2)
            search_end = np.minimum(self.grid.shape, cell+2)

            
            diff = search_end - search_start
            for current_cell in np.ndindex(tuple(diff)):
                current_cell += search_start

                sample_index = self.grid[tuple(current_cell)]-1

                if sample_index != -1:
                    distanceSquare = self.get_distance_square(self.samples[int(sample_index)], candidate)

                    if distanceSquare < self.radius**2:
                        return False

            return True

        else:
            # Out of borders
            return False

    def get_distance_square(self, a, b):
        difference = abs(a - b)
        return (difference**2).sum()


if __name__ == "__main__":
    sample = poission_disc_sampling(radius=30, size=(1920, 1080), number_of_trials=3)
    points = sample()
    print(points)
