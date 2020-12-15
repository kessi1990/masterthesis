import pathlib
import random
from env import rooms, rooms_new
import time
import sys


def read_map_file(path):
    file = pathlib.Path(path)
    assert file.is_file(), "{} couldn't be opened. Check filepath".format(file)
    with open(path) as f:
        content = f.readlines()
    obstacles = []  # [i for i, cell in enumerate(content.split()) if cell == '#']
    starting_pos = None
    goal_pos = None
    width = 0
    height = 0
    for y, line in enumerate(content):
        for x, cell in enumerate(line.strip().split()):
            if cell == '#':
                obstacles.append((x, y))
            elif cell == 'S':
                starting_pos = (x, y)
            else:
                goal_pos = (x, y)
            width = x + 1
        height = y + 1
    return width, height, obstacles, starting_pos, goal_pos


def load_rooms_env(size='small'):
    if size == 'small':
        path = "../env/maps/rooms_9_9.txt"
        time_limit = 500
    else:
        path = "../env/maps/rooms_25_13.txt"
        time_limit = 5000
    size_x, size_y, obs_coords, starting_pos, goal_pos = read_map_file(path)
    return rooms.Environment(size_x, size_y, obs_coords, starting_pos, goal_pos, time_limit)


def load_grid(size='small'):
    if size == 'small':
        path = '../env/maps/rooms_9_9.txt'
        x = 9
        y = 9
        time_limit = 200
    else:
        path = '../env/maps/rooms_25_13.txt'
        x = 25
        y = 13
        time_limit = 1000

    file = pathlib.Path(path)
    assert file.is_file(), "{} couldn't be opened. Check filepath".format(file)

    with open(path) as f:
        content = f.read()

    obs_idx = []
    sub_goals = []
    starting_id = None
    goal_id = None

    for i, cell in enumerate(content.split()):
        if cell == '#':
            obs_idx.append(i)
        elif cell == 'S':
            starting_id = i
        elif cell == 'G':
            goal_id = i
        elif cell == '-':
            # sub_goals.append(i)
            sub_goals.append((i // x, i % x))
        elif cell == '.':
            continue
        else:
            raise ValueError(f'what kind of sorcery is this? {cell}')

    return y, x, obs_idx, sub_goals, (starting_id // x, starting_id % x), (goal_id // x, goal_id % x), time_limit


def load(size, frame_stack, device, random_start, random_goal):
    height, width, obs_coord, sub_goals, start_coord, goal_coord, time_limit = load_grid(size)
    return rooms_new.Grid(frame_stack, height, width, obs_coord, sub_goals, start_coord, goal_coord, device, time_limit, random_start, random_goal)
