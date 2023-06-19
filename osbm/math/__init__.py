import numpy as np
import math
from scipy import stats

def print_full_array_info(array: np.ndarray, is_sample: bool = False):
    '''
    Print information about a numpy array

    Args:
        array (np.ndarray): Numpy array

        is_sample (bool): Will be used when calculating the standard deviation and variance

    '''

    print('shape:', array.shape)
    print('dtype:', array.dtype)

    mode = stats.mode(array, keepdims=False)
    min_value, q1, median, q3, max_value = np.percentile(array, [0, 25, 50, 75, 100])
    iqr = q3 - q1

    if mode[1] == 1:
        print('mode:', 'no mode')
    else:
        print('mode:', mode[0], 'count:', mode[1])
    print('min, Q1, median, Q3, max:', min_value, q1, median, q3, max_value)


    print("IQR:", iqr)

    print('mean:', array.mean())
    if is_sample:
        print('standard deviation:', array.std(ddof=1))
        print('variance:', array.var(ddof=1))
    else:
        print('standard deviation:', array.std())
        print('variance:', array.var())

    print('skewness:', stats.skew(array))
    print('kurtosis:', stats.kurtosis(array))
    print('mean absolute deviation:', stats.median_abs_deviation(array))
    print('median absolute deviation:', stats.median_abs_deviation(array, scale='normal'))


    high_outliers = array[array > q3 + 1.5 * iqr]
    low_outliers = array[array < q1 - 1.5 * iqr]
    print('outlier detection by IQR and Q1, Q3')
    num_h_outliers = len(high_outliers)
    num_l_outliers = len(low_outliers)
    if num_h_outliers == 0 and num_l_outliers == 0:
        print('no outliers')
    else:
        if num_h_outliers > 10:
            print('# of high outliers:', num_h_outliers)
        else:
            print('high outliers:', high_outliers)

        if num_l_outliers > 10:
            print('# of low outliers:', num_l_outliers)
        else:
            print('low outliers:', low_outliers)


    print('outlier detection by Z-score')
    z_scores = stats.zscore(array)
    z_scores = np.abs(z_scores)
    z_scores = z_scores[z_scores > 3]
    num_z_outliers = len(z_scores)
    if num_z_outliers == 0:
        print('no outliers')
    else:
        if num_z_outliers > 10:
            print('# of outliers:', num_z_outliers)
        else:
            print('outliers:', z_scores)
