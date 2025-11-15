import numpy as np


def solve(input: np.ndarray) -> np.ndarray:
    output = np.zeros((9, 9), dtype=int)
    for i in range(3):
        for j in range(3):
            if input[i, j]:
                output[i * 3:(i + 1) * 3, j * 3:3 * (j + 1)] = input

    return output
    