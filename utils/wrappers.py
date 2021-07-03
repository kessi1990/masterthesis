import gym
import torch
import numpy as np

from collections import deque
from utils import transformation


class NoopResetEnv(gym.Wrapper):
    """

    """
    def __init__(self, env, noop_max=30):
        """

        :param env:
        :param noop_max:
        """
        """
        Sample initial states by taking random number of no-ops on reset.
        No-op is assumed to be action 0.
        """
        gym.Wrapper.__init__(self, env)
        self.noop_max = noop_max
        self.override_num_noops = None
        self.noop_action = 0
        assert env.unwrapped.get_action_meanings()[0] == 'NOOP'

    def reset(self, **kwargs):
        """

        :param kwargs:
        :return:
        """
        """ 
        Do no-op action for a number of steps in [1, noop_max].
        """
        self.env.reset(**kwargs)
        if self.override_num_noops is not None:
            noops = self.override_num_noops
        else:
            noops = self.unwrapped.np_random.randint(1, self.noop_max + 1)
        assert noops > 0
        obs = None
        for _ in range(noops):
            obs, _, done, _ = self.env.step(self.noop_action)
            if done:
                obs = self.env.reset(**kwargs)
        return obs

    def step(self, action):
        """

        :param action:
        :return:
        """
        return self.env.step(action)


class FireResetEnv(gym.Wrapper):
    """

    """
    def __init__(self, env=None):
        """

        :param env:
        """
        super(FireResetEnv, self).__init__(env)
        assert env.unwrapped.get_action_meanings()[1] == 'FIRE'
        assert len(env.unwrapped.get_action_meanings()) >= 3

    def step(self, action):
        """

        :param action:
        :return:
        """
        return self.env.step(action)

    def reset(self):
        """

        :return:
        """
        self.env.reset()
        obs, _, done, _ = self.env.step(1)
        if done:
            self.env.reset()
        obs, _, done, _ = self.env.step(2)
        if done:
            self.env.reset()
        return obs


class MaxAndSkipEnv(gym.Wrapper):
    """

    """
    def __init__(self, env=None, skip=4):
        """

        :param env:
        :param skip:
        """
        super(MaxAndSkipEnv, self).__init__(env)
        self._obs_buffer = deque(maxlen=2)
        self._skip = skip

    def step(self, action):
        """

        :param action:
        :return:
        """
        total_reward = 0.0
        done = None
        info = None
        for _ in range(self._skip):
            obs, reward, done, info = self.env.step(action)
            self._obs_buffer.append(obs)
            total_reward += reward
            if done:
                break
        max_frame = np.max(np.stack(self._obs_buffer), axis=0)
        return max_frame, total_reward, done, info

    def reset(self):
        """

        :return:
        """
        self._obs_buffer.clear()
        obs = self.env.reset()
        self._obs_buffer.append(obs)
        return obs


class FrameStack(gym.ObservationWrapper):
    """

    """
    def __init__(self, env, k):
        """

        :param env:
        :param k:
        """
        super(FrameStack, self).__init__(env)
        self.k = k
        self.buffer = torch.zeros(1, k, 84, 84)

    def reset(self):
        """

        :return:
        """
        self.buffer = torch.zeros(1, self.k, 84, 84)
        return self.observation(self.env.reset())

    def observation(self, observation):
        """

        :param observation:
        :return:
        """
        self.buffer = torch.cat((self.buffer[:, 1:], observation), dim=1)
        return self.buffer


class EpisodicLifeEnv(gym.Wrapper):
    """

    """
    def __init__(self, env):
        """

        :param env:
        """
        """
        Make end-of-life == end-of-episode, but only reset on true game over.
        Done by DeepMind for the DQN and co. since it helps value estimation.
        """
        gym.Wrapper.__init__(self, env)
        self.lives = 0
        self.was_real_done = True

    def step(self, action):
        """

        :param action:
        :return:
        """
        obs, reward, done, info = self.env.step(action)
        self.was_real_done = done
        """
        check current lives, make loss of life terminal,
        then update lives to handle bonus lives
        """
        lives = self.env.unwrapped.ale.lives()
        if self.lives > lives > 0:
            # for Qbert sometimes we stay in lives == 0 condition for a few frames
            # so it's important to keep lives > 0, so that we only reset once
            # the environment advertises done.
            done = True
        self.lives = lives
        return obs, reward, done, info

    def reset(self, **kwargs):
        """

        :param kwargs:
        :return:
        """
        """
        Reset only when lives are exhausted.
        This way all states are still reachable even though lives are episodic,
        and the learner need not know about any of this behind-the-scenes.
        """
        if self.was_real_done:
            obs = self.env.reset(**kwargs)
        else:
            # no-op step to advance from terminal/lost life state
            obs, _, _, _ = self.env.step(0)
        self.lives = self.env.unwrapped.ale.lives()
        return obs


class TransformFrame(gym.ObservationWrapper):
    """

    """
    def __init__(self, env=None):
        """

        :param env:
        """
        super(TransformFrame, self).__init__(env)
        config = {'top': 18, 'left': 0, 'crop_height': 84, 'crop_width': 84}
        self.transformation = transformation.Transformation(config)

    def observation(self, obs):
        """

        :param obs:
        :return:
        """
        return self.transformation.transform(obs)


def make_env(env_name, fs=False, k=4):
    env = gym.make(env_name + 'NoFrameskip-v4')
    env = NoopResetEnv(env)
    env = MaxAndSkipEnv(env)
    env = FireResetEnv(env)
    env = TransformFrame(env)
    if fs:
        env = FrameStack(env, k=k)
    env = EpisodicLifeEnv(env)
    return env



