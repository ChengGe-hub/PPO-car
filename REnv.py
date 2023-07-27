import gym
from gym import spaces
import numpy as np


# 创建移动机器人环境类
class RobotEnv(gym.Env):
    def __init__(self, width, height):
        super(RobotEnv, self).__init__()
        self.width = width
        self.height = height
        self.robot_pos = np.array([0, 0])

        self.observation_space = spaces.Box(low=0, high=self.width - 1, shape=(2,), dtype=np.float32)
        self.action_space = spaces.Discrete(4)

    def reset(self):
        self.robot_pos = np.array([0, 0])
        return self.robot_pos

    def step(self, action):
        if action == 0:  # Up
            self.robot_pos[1] += 1
        elif action == 1:  # Down
            self.robot_pos[1] -= 1
        elif action == 2:  # Left
            self.robot_pos[0] -= 1
        elif action == 3:  # Right
            self.robot_pos[0] += 1

        self.robot_pos[0] = np.clip(self.robot_pos[0], 0, self.width - 1)
        self.robot_pos[1] = np.clip(self.robot_pos[1], 0, self.height - 1)

        reward = self.calculate_reward()
        done = self.check_terminal_state()

        return self.robot_pos.copy(), reward, done, {}

    def calculate_reward(self):
        distance1 = np.linalg.norm(self.robot_pos - np.array([self.width - 1, self.height - 1]))
        distance2 = np.linalg.norm(self.robot_pos - np.array([0, 0]))
        return - 2 * distance1

    def check_terminal_state(self):
        return np.array_equal(self.robot_pos, [self.width - 1, self.height - 1])

    def get_position(self):
        return self.robot_pos.copy()
