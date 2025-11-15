from itertools import count
import numpy as np
from utils import base_path, show_task


def detect_objects(input: np.ndarray) -> list[set[tuple[int, int]]]:
    visited = set()
    objects = []
    for i in range(input.shape[0]):
        for j in range(input.shape[1]):
            coord = (i, j)
            if input[coord] == 0 or coord in visited:
                continue
            obj = set()
            seeds = {coord}
            while len(seeds) > 0:
                seed = seeds.pop()
                if seed in visited:
                    continue
                visited.add(seed)
                obj.add(seed)
                for di in [-1, 0, 1]:
                    for dj in [-1, 0, 1]:
                        if abs(di) + abs(dj) != 1:
                            continue
                        ni, nj = seed[0] + di, seed[1] + dj
                        if ni < 0 or ni >= input.shape[0]:
                            continue
                        if nj < 0 or nj >= input.shape[1]:
                            continue
                        ncoord = (ni, nj)
                        if input[ncoord] != 0 and ncoord not in visited:
                            seeds.add(ncoord)
            objects.append(obj)
    return objects


def solve(input: np.ndarray) -> np.ndarray:
    objects = detect_objects(input)
    biggest_object = max(objects, key=lambda obj: len(obj))
    width = max(j for i, j in biggest_object) - min(j for i, j in biggest_object) + 1
    height = max(i for i, j in biggest_object) - min(i for i, j in biggest_object) + 1
    # For each direction, put copies of the object in the same colour
    # in the same direction with a gap of 1
    output = input.copy()
    for di in [-1, 0, 1]:
        for dj in [-1, 0, 1]:
            if di == 0 and dj == 0:
                continue
            colour_this_direction = 0
            for n in count(1):
                coords = set()
                for coord in biggest_object:
                    i, j = coord
                    ni = i + di * (n * (height + 1))
                    nj = j + dj * (n * (width + 1))
                    coords.add((ni, nj))
                if n == 1:
                    for ni, nj in coords:
                        if 0 <= ni < input.shape[0] and 0 <= nj < input.shape[1] and input[ni, nj] != 0:
                            colour_this_direction = input[ni, nj]
                            break
                pixel_placed = False
                for ni, nj in coords:
                    if 0 <= ni < input.shape[0] and 0 <= nj < input.shape[1]:
                        pixel_placed = True
                        output[ni, nj] = colour_this_direction
                if not pixel_placed:
                    break

    return output


if __name__ == "__main__":
    from utils import show_task

    task_name = __file__.replace("solutions/", base_path + "/").replace(".py", ".json")
    show_task(task_name, solve)
