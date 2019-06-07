import os, subprocess, time, signal
import gym
from gym import error, spaces, utils
from gym.utils import seeding
from gym_snake.envs.snake import Controller, Discrete
import numpy as np

try:
    import matplotlib.pyplot as plt
    import matplotlib
except ImportError as e:
    raise error.DependencyNotInstalled("{}. (HINT: see matplotlib documentation for installation https://matplotlib.org/faq/installing_faq.html#installation".format(e))

class SnakeEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, grid_size=[30,30], unit_size=10, unit_gap=1, snake_size=2, n_snakes=1, n_foods=1, random_init=True):
        self.grid_size = grid_size
        self.unit_size = unit_size
        self.unit_gap = unit_gap
        self.snake_size = snake_size
        self.n_snakes = n_snakes
        self.n_foods = n_foods
        self.viewer = None
        self.action_space = Discrete(4)
        self.random_init = random_init
        #Defined observation space
        #self.observation_space = spaces.Box(low=np.array([0]*11), high=np.array([1]*11), dtype=np.bool)
        self.observation_space = spaces.Tuple((spaces.Discrete(2), spaces.Discrete(2), spaces.Discrete(2), spaces.Discrete(2), spaces.Discrete(2), spaces.Discrete(2), spaces.Discrete(2), spaces.Discrete(2), spaces.Discrete(2), spaces.Discrete(2), spaces.Discrete(2)))

    def step(self, action):
        #self.last_obs, rewards, done, info = self.controller.step(action)
        last_obs, rewards, done, info = self.controller.step(action)
        self.last_obs = self.controller.grid.grid.copy()
        #return self.last_obs, rewards, done, info
        return last_obs, rewards, done, info

    def reset(self):
        self.controller = Controller(self.grid_size, self.unit_size, self.unit_gap, self.snake_size, self.n_snakes, self.n_foods, random_init=self.random_init)
        self.last_obs = self.controller.grid.grid.copy()
        obs = [tuple(self.controller.snakes[0].head) if self.controller.snakes and self.controller.snakes[0] else None, self.controller.grid.foodLocations[0]]
        #return self.last_obs
        return tuple(self.controller.generateObservationTuple(int(self.controller.snakes[0].direction)))

    def render(self, mode='human', close=False, frame_speed=.1):
        if self.viewer is None:
            self.fig = plt.figure()
            self.viewer = self.fig.add_subplot(111)
            plt.ion()
            self.fig.show()
        else:
            self.viewer.clear()
            self.viewer.imshow(self.last_obs)
            plt.pause(frame_speed)
        self.fig.canvas.draw()

    def seed(self, x):
        pass
