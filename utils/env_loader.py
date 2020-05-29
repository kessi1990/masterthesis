import pathlib
from env import rooms


def read_map_file(path):
    file = pathlib.Path(path)
    assert file.is_file(), "{} couldn't be opened. Check filepath".format(file)
    with open(path) as f:
        content = f.readlines()
    obstacles = []
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


def load_rooms_env(time_limit=100, path="../env/maps/rooms_25_13.txt"):
    size_x, size_y, obs_coords, starting_pos, goal_pos = read_map_file(path)
    return rooms.Environment(size_x, size_y, obs_coords, starting_pos, goal_pos, time_limit)
