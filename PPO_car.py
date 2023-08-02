import gym
from gym import spaces
from REnv import RobotEnv
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np


# 定义策略网络
class Policy(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim):
        super(Policy, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        x = self.softmax(x)
        return x


# 定义PPO算法训练函数
def train_ppo(env, num_episodes, hidden_dim, batch_size, eps_clip, gamma, lr_actor, lr_critic):
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    policy = Policy(state_dim, action_dim, hidden_dim)
    optimizer_actor = optim.Adam(policy.parameters(), lr=lr_actor)
    optimizer_critic = optim.Adam(policy.parameters(), lr=lr_critic)
    mse_loss = nn.MSELoss()

    rewards = []
    positions = []
    best_positions = []
    best_rewards = -500000

    for episode in range(num_episodes):
        state = env.reset()
        done = False
        episode_reward = 0
        episode_loss_actor = 0
        episode_loss_critic = 0
        position = []
        #pos = []

        while not done:
            state = torch.from_numpy(state).float()
            action_probs = policy(state)
            dist = Categorical(logits=action_probs)
            action = dist.sample().item()
            next_state, reward, done, _ = env.step(action)
            #pos.append(next_state)

            next_state = torch.from_numpy(next_state).float()
            reward = torch.tensor(reward)
            done = torch.tensor(done)

            value = policy(state)
            next_value = policy(next_state).detach()
            advantage = reward + (1 - done.float()) * gamma * next_value - value.detach()

            action_probs = action_probs.squeeze(0)
            old_action_prob = action_probs[action]
            ratio = torch.exp(torch.log(action_probs[action]) - torch.log(old_action_prob))
            surrogate = torch.min(ratio * advantage, torch.clamp(ratio, 1 - eps_clip, 1 + eps_clip) * advantage)

            optimizer_actor.zero_grad()
            optimizer_critic.zero_grad()
            actor_loss = -surrogate.mean()
            critic_loss = mse_loss(value, reward + torch.where(done, torch.zeros_like(next_value), gamma * next_value))

            episode_loss_actor += actor_loss.item()
            episode_loss_critic += critic_loss.item()

            actor_loss.backward(retain_graph=True)
            critic_loss.backward()

            optimizer_actor.step()
            optimizer_critic.step()

            state = next_state.numpy()
            episode_reward += reward.item()
            position.append(env.get_position())

        episode_loss_actor /= len(position)
        episode_loss_critic /= len(position)
        print(f"Episode: {episode + 1}/{num_episodes}, Reward: {episode_reward}, Actor Loss: {episode_loss_actor}, Critic Loss: {episode_loss_critic}")

        if episode_reward > best_rewards:
            best_rewards = episode_reward
            best_positions = position

        rewards.append(episode_reward)
        positions.append(position)


    return rewards, positions, best_positions


# 设置训练参数
num_episodes = 300
hidden_dim = 256
batch_size = 32
eps_clip = 0.02
gamma = 0.9
lr_actor = 0.0001
lr_critic = 0.0001

# 创建移动机器人环境实例
env = RobotEnv(width=8, height=6)

# 进行训练
rewards, positions, best_positions = train_ppo(env, num_episodes, hidden_dim, batch_size, eps_clip, gamma, lr_actor, lr_critic)

# 绘制奖励曲线
plt.plot(rewards)
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.title("Reward Curve")
plt.show()

best_positions[:0] = [np.array([0, 0])]

# 绘制最佳轨迹
x = [pos[0] for pos in best_positions]
y = [pos[1] for pos in best_positions]
plt.plot(x, y, c='b', marker='o')
plt.xticks([-1, 0, 1, 2, 3, 4, 5, 6, 7, 8])
plt.yticks([-1, 0, 1, 2, 3, 4, 5, 6])
plt.xlabel("X")
plt.ylabel("Y")
plt.title("Best Trajectory")
plt.show()



