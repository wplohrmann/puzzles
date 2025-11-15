from itertools import count
import numpy as np
from utils import base_path, show_task


def solve(input: np.ndarray) -> np.ndarray:
    output = np.zeros((3, 3), dtype=int)
    for i in range(3):
        for j in range(3):
            if input[i, j] and input[i, j+4]:
                output[i, j] = 2
    return output


if __name__ == "__main__":
    task_name = __file__.replace("solutions/", base_path + "/").replace(".py", ".json")
    show_task(task_name, solve)
