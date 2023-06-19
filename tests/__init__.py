from osbm import print_full_array_info
import numpy as np

if __name__ == '__main__':
    # random normal array of lenght 20
    array = np.random.normal(0, 1, 20)
    print('Array:', array)
    print_full_array_info(array)
    print()
    print_full_array_info(array, is_sample=True)
