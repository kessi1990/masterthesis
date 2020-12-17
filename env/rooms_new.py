import torch
import random
import gc
import numpy as np
import torchvision.transforms as t
from PIL import Image


class Grid:
    """

    """
    def __init__(self, frame_stack, height, width, obs_coord, sub_goals, start_coord, goal_coord, device, time_limit=200, random_start=False,
                 random_goal=False):
        """

        :param frame_stack:
        :param height:
        :param width:
        :param obs_coord:
        :param sub_goals:
        :param start_coord:
        :param goal_coord:
        :param device:
        :param time_limit:
        :param random_start:
        :param random_goal:
        """
        self.device = device
        self.transform = t.Compose([t.Grayscale(num_output_channels=1), t.ToTensor()])
        self.size = 'small' if (height == 9 and width == 9) else 'large'
        self.width = width
        self.height = height
        self.channels = 3

        self.action_space = {0: np.array((-1, 0)),
                             1: np.array((1, 0)),
                             2: np.array((0, -1)),
                             3: np.array((0, 1))}  # [0, 1, 2, 3]  # up down left right
        self.actions = list(self.action_space)

        self.random_start = random_start
        self.random_goal = random_goal

        self.start_coord = np.array(start_coord)
        self.goal_coord = np.array(goal_coord)
        self.obs_coord = np.array(obs_coord)
        self.sub_goals = np.array(sub_goals)

        self.curr_coord = np.copy(self.start_coord)

        self.state = None
        self.init_objects()

        self.step_counter = 0
        self.time_limit = time_limit
        self.done = True
        self.info = 'useless'

        self.frame_stack = frame_stack
        if self.frame_stack > 1:
            self.buffer = torch.zeros((1, self.frame_stack, self.height, self.width), dtype=torch.float32, device=device)

    def reset(self):
        """
        reset environment --> reset pixel values, counter and done flag
        :return:
        """

        self.init_objects()
        self.curr_coord = np.copy(self.start_coord)
        self.step_counter = 0
        self.done = False
        observation = self.transform(Image.fromarray(self.state.transpose(1, 2, 0))).unsqueeze(dim=0)
        if self.frame_stack > 1:
            self.buffer = torch.zeros((1, self.frame_stack, self.height, self.width), dtype=torch.float32, device=self.device)
            observation = self.buffer.clone().detach()
        gc.collect()
        # state (c, h, w) -> Image (w, h, c)
        return observation

    def step(self, action):
        """

        :param action:
        :return:
        """

        reward = 0
        self.step_counter += 1

        # check for collision
        if not self.collision(action):
            # no collision, move according to corresponding action
            self.move(action)

            # check if agent_coord == goal_coord
            if self.check_goal():
                # goal reached, return reward and done flag
                reward = 1
                self.done = True

        # check time_limit
        if self.step_counter >= self.time_limit:
            # time_limit exceeded, terminal condition met
            self.done = True

        observation = self.transform(Image.fromarray(self.state.transpose(1, 2, 0))).unsqueeze(dim=0)
        if self.frame_stack > 1:
            self.buffer = torch.cat((self.buffer[:, 1:], observation), dim=1)
            observation = self.buffer.clone().detach()
            print(f'observation {observation.shape}')

        # return state, reward, done flag, info
        return observation, reward, self.done, self.info

    def move(self, direction):
        """

        :param direction:
        :return:
        """
        self.state = np.copy(self.state)
        # set pixel value to zero at old position
        self.state[0, self.curr_coord[0], self.curr_coord[1]] = 0
        # compute new index / agent_coord
        self.curr_coord += self.action_space[direction]
        # set new index / agent_coord to max
        self.state[0, self.curr_coord[0], self.curr_coord[1]] = 255

    def collision(self, direction):
        """

        :param direction:
        :return:
        """
        y, x = self.curr_coord + self.action_space[direction]
        return int(self.state[2, y, x]) == 255

    def check_goal(self):
        """

        :return:
        """
        return np.array_equiv(self.curr_coord, self.goal_coord)

    def init_objects(self):
        """

        :return:
        """
        self.state = np.zeros((self.channels, self.height, self.width), dtype=np.uint8)
        if self.random_start:
            self.start_coord = np.array(self.sample_random_coord('agent'))
        if self.random_goal:
            self.goal_coord = np.array(self.sample_random_coord('goal'))
        self.state[0, self.start_coord[0], self.start_coord[1]] = 255
        self.state[1, self.goal_coord[0], self.goal_coord[1]] = 255
        np.put(self.state[2], self.obs_coord, 255)

    def sample_random_coord(self, obj):
        """
        hard-coded coords ... nobody cares
        :param obj:
        :return:
        """
        if self.size == 'small':
            starts = [10, 11, 12,
                      19, 20, 21,
                      28, 29, 30]
            goals = [50, 51, 52,
                     59, 60, 61,
                     68, 69, 70]
        else:
            starts = [26, 27, 28, 29, 30,
                      51, 52, 53, 54, 55,
                      76, 77, 78, 79, 80,
                      101, 102, 103, 104, 105,
                      126, 127, 128, 129, 130]
            goals = [194, 195, 196, 197, 198,
                     219, 220, 221, 222, 223,
                     244, 245, 246, 247, 248,
                     269, 270, 271, 272, 273,
                     294, 295, 296, 297, 298]

        pos_idx = random.choice(starts if obj == 'agent' else goals)
        return pos_idx // self.width, pos_idx % self.width
