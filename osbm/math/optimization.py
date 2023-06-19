import numpy as np

def is_square(matrix: np.ndarray) -> bool:
    '''
    Check if a matrix is square

    Args:
        matrix (np.ndarray): Numpy array

    Returns:
        bool: True if the matrix is square, False otherwise

    '''
    return matrix.shape[0] == matrix.shape[1]

def is_orthagonal(matrix: np.ndarray) -> bool:
    '''
    Check if a matrix is orthagonal

    Args:
        matrix (np.ndarray): Numpy array

    Returns:
        bool: True if the matrix is orthagonal, False otherwise

    '''
    assert is_square(matrix), 'Matrix must be square'
    return np.allclose(matrix @ matrix.T, np.eye(matrix.shape[0]))



def is_symmetric(matrix: np.ndarray) -> bool:
    '''
    Check if a matrix is symmetric

    Args:
        matrix (np.ndarray): Numpy array

    Returns:
        bool: True if the matrix is symmetric, False otherwise

    '''
    assert is_square(matrix), 'Matrix must be square'
    return np.allclose(matrix, matrix.T)

def is_positive_definite(matrix: np.ndarray) -> bool:
    '''
    Check if a matrix is positive definite

    Args:
        matrix (np.ndarray): Numpy array

    Returns:
        bool: True if the matrix is positive definite, False otherwise

    '''
    assert is_square(matrix), 'Matrix must be square'
    return np.all(np.linalg.eigvals(matrix) > 0)

def is_positive_semidefinite(matrix: np.ndarray) -> bool:
    '''
    Check if a matrix is positive semidefinite

    Args:
        matrix (np.ndarray): Numpy array

    Returns:
        bool: True if the matrix is positive semidefinite, False otherwise

    '''
    assert is_square(matrix), 'Matrix must be square'
    return np.all(np.linalg.eigvals(matrix) >= 0)
