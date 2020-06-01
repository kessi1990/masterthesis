import numpy as np
import gc
from enum import Enum


class Environment:
    def __init__(self, size_x, size_y, obs_coords, starting_pos, goal_pos, time_limit):
        self.size_x = size_x
        self.size_y = size_y
        self.size_z = 3
        self.action_space = [0, 1, 2, 3]  # up down left right
        self.agent = None
        self.starting_pos = starting_pos
        self.goal = None
        self.goal_pos = goal_pos
        self.obstacles = []
        self.obs_coords = obs_coords
        self.step_counter = 0
        self.time_limit = time_limit
        self.done = False
        self.info = None

    def reset(self):
        self.init_objects()
        self.step_counter = 0
        self.done = False
        gc.collect()
        return self.encode_state()

    def step(self, action):
        self.step_counter += 1
        reward = Rewards.STEP.value
        if not self.collision(action):
            self.move(action)
            if self.check_goal():
                reward += Rewards.GOAL.value
                self.done = True
        else:
            reward += Rewards.COLLISION.value
        state = self.encode_state()
        if self.step_counter >= self.time_limit:
            self.done = True
        return state, reward, self.done, self.info

    def move(self, direction):
        if direction == 0:
            self.agent.y += -1
        elif direction == 1:
            self.agent.y += 1
        elif direction == 2:
            self.agent.x += -1
        else:
            self.agent.x += 1

    def collision(self, direction):
        if direction == 0:
            if (self.agent.x, self.agent.y - 1) in self.obs_coords:
                return True
        elif direction == 1:
            if (self.agent.x, self.agent.y + 1) in self.obs_coords:
                return True
        elif direction == 2:
            if (self.agent.x - 1, self.agent.y) in self.obs_coords:
                return True
        else:
            if (self.agent.x + 1, self.agent.y) in self.obs_coords:
                return True
        return False

    def check_goal(self):
        return self.agent.x == self.goal.x and self.agent.y == self.goal.y

    def encode_state(self):
        state = np.zeros([1, self.size_x, self.size_y, self.size_z])
        state[0, self.agent.x, self.agent.y, self.agent.z] = 1
        state[0, self.goal.x, self.goal.y, self.goal.z] = 1
        for obs in self.obstacles:
            state[0, obs.x, obs.y, obs.z] = 1
        return state

    def init_objects(self):
        self.obstacles = [Obstacle(x, y) for x, y in self.obs_coords]
        self.agent = Agent(self.starting_pos[0], self.starting_pos[1])
        self.goal = Goal(self.goal_pos[0], self.goal_pos[1])

    @staticmethod
    def print_obs(obs):
        print('{} {} {} {}'.format(type(obs), obs.x, obs.y, obs.z))


class Object:
    def __init__(self, x, y):
        self.x = x
        self.y = y


class Agent(Object):
    def __init__(self, x, y):
        super().__init__(x, y)
        self.z = 0


class Obstacle(Object):
    def __init__(self, x, y):
        super().__init__(x, y)
        self.z = 1


class Goal(Object):
    def __init__(self, x, y):
        super().__init__(x, y)
        self.z = 2


class Rewards(Enum):
    COLLISION = 0
    STEP = 0
    GOAL = 1
