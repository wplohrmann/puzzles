import numpy as np
from utils import base_path, show_task


def solve(input: np.ndarray) -> np.ndarray:
    colours = {}
    for i in range(input.shape[0]):
        for j in range(input.shape[1]):
            if input[i, j] == 0:
                continue
            line_num = (i + j) % 3
            colours[line_num] = input[i, j]

    output = np.zeros_like(input)
    for i in range(input.shape[0]):
        for j in range(input.shape[1]):
            line_num = (i + j) % 3
            if line_num in colours:
                output[i, j] = colours[line_num]

    return output


if __name__ == "__main__":
    from utils import show_task

    task_name = __file__.replace("solutions/", base_path + "/").replace(".py", ".json")
    show_task(task_name, solve)
