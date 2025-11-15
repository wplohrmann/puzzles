import numpy as np


def solve(input: np.ndarray) -> np.ndarray:
    output = input.copy()
    seeds = set()
    outside = set()
    for i in range(input.shape[0]):
        for j in range(input.shape[1]):
            if i == 0 or i == input.shape[0] - 1 or j == 0 or j == input.shape[1] - 1:
                coord = (i, j)
                if input[i, j] == 0:
                    seeds.add(coord)
                    outside.add(coord)
    while len(seeds) > 0:
        seed = seeds.pop()
        for di in [-1, 0, 1]:
            for dj in [-1, 0, 1]:
                if abs(di) + abs(dj) != 1:
                    continue
                ni, nj = seed[0] + di, seed[1] + dj
                if ni < 0 or ni >= input.shape[0]:
                    continue
                if nj < 0 or nj >= input.shape[0]:
                    continue
                coord = (ni, nj)
                if input[coord] == 0:
                    if coord not in outside:
                        outside.add(coord)
                        seeds.add(coord)
    for i in range(input.shape[0]):
        for j in range(input.shape[1]):
            coord = (i, j)
            if coord not in outside and input[coord] == 0:
                output[coord] = 4


    return output
