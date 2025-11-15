import numpy as np
from itertools import count
from utils import base_path, show_task

def solve(input: np.ndarray) -> np.ndarray:
    bottom_right_corners = set()
    for i in range(input.shape[0]):
        for j in range(input.shape[1]):
            if input[i, j] == 0:
                continue
            colour = input[i, j]
            if input[(i - 1, j)] == colour and input[(i, j - 1)] == colour:
                bottom_right_corners.add((i, j))
    fixed = set()
    for corner in bottom_right_corners:
        i, j = corner
        colour = input[i, j]
        fixed.add((i, j))
        fixed.add((i - 1, j))
        for dj in count(1):
            if input[(i, j - dj)] != colour:
                break
            fixed.add((i, j - dj))

    output = np.zeros_like(input)
    for i in range(input.shape[0]):
        for j in range(input.shape[1]):
            if (i, j) in fixed:
                output[i, j] = input[i, j]
            elif j - 1 > 0:
                output[i, j] = input[i, j - 1]


    return output

if __name__ == "__main__":
    from utils import show_task
    task_name = __file__.replace("solutions/", base_path + "/").replace(".py", ".json")
    show_task(task_name, solve)
