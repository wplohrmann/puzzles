from importlib import import_module
import json
import os
import matplotlib.pyplot as plt
import numpy as np


colours = ["#000000", "#0074D9", "#FF4136", "#2ECC40", "#FFDC00", "#AAAAAA",  "#F012BE", "#FF851B", "#7FDBFF", "#870C25"]
def to_colour(grid: np.ndarray) -> np.ndarray:
    h, w = grid.shape
    colour_grid = np.zeros((h, w, 3), dtype=np.uint8)
    for i in range(h):
        for j in range(w):
            colour_grid[i, j] = tuple(int(colours[grid[i, j]][k:k+2], 16) for k in (1, 3, 5))
    return colour_grid

base_path = "ARC-AGI/data/training"

tasks = os.listdir(base_path)
for task_name in sorted(tasks):
    task_path = os.path.join(base_path, task_name)
    with open(task_path) as f:
        task = json.load(f)

    examples = task["train"]
    fig, axs = plt.subplots(len(examples), 2)
    for i, example in enumerate(examples):
        input_grid = to_colour(np.array(example["input"]))
        output_grid = to_colour(np.array(example["output"]))
        axs[i, 0].imshow(input_grid)
        axs[i, 0].set_title("Input")
        axs[i, 1].imshow(output_grid)
        axs[i, 1].set_title("Output")
    plt.show()
    solution_path = os.path.join(f"solutions/{task_name.replace('.json', '.py')}")
    if not os.path.exists(solution_path):
        print(f"Solution missing for task {task_name}")
        break
    solve = import_module(f"solutions.{task_name.replace('.json', '')}").solve
    fig, axs = plt.subplots(1, 2)
    for test in task["test"]:
        input_grid = np.array(test["input"])
        predicted_output = solve(input_grid)
        predicted_output_coloured = to_colour(predicted_output)
        axs[0].imshow(to_colour(input_grid))
        axs[0].set_title("Input")
        axs[1].imshow(predicted_output_coloured)
        is_correct = (predicted_output == np.array(test["output"])).all()
        axs[1].set_title(f"Predicted output: {'Correct' if is_correct else 'Incorrect'}")
    plt.show() 

