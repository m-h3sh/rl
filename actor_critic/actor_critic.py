import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import time
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch.nn.functional as F
from collections import deque

# defining some parameters
NUM_EPISODES = 1000
MAX_STEPS = 10000
GAMMA = 0.99
ACTOR_LR = 0.001
CRITIC_LR = 0.001
SOLVED_SCORE = 195

# defining the actor and critic networks
class Policy(nn.Module):
    def __init__(self, input_shape, output_dim):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_shape, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim),
            nn.Softmax(dim=-1)
        )

    def forward(self, x):
        return self.layers(x)

class Critic(nn.Module):
    def __init__(self, input_shape):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_shape, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        return self.layers(x)

def train(actor, critic):
    ep_rewards = []
    ep_range = tqdm(range(NUM_EPISODES))
    actor_optim = torch.optim.Adam(actor.parameters(), lr=ACTOR_LR)
    critic_optim = torch.optim.Adam(critic.parameters(), lr=CRITIC_LR)
    eps = np.finfo(np.float32).eps.item()

    for ep in ep_range:
        # initializing the state of the environment
        state, _ = env.reset()
        done = False
        reward = 0
        I = 1
        recent_rewards = deque(maxlen = 100)
        log_probs = []
        rewards = []
        val_states = []
        actor_optim.zero_grad()
        critic_optim.zero_grad()

        # running upto max steps in each episode
        for step in range(MAX_STEPS):
            # get action from actor
            state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            probs = actor(state)
            if probs.dim() > 2:
                probs = probs.squeeze(0)
            dist = torch.distributions.Categorical(probs) # sampling action
            action = dist.sample()
            log_prob = dist.log_prob(action)
            log_probs.append(log_prob)

            next_state, r, done, truncated, _ = env.step(action.item())
            next_state = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0)
            reward += r
            rewards.append(r)

            # finding val and next val from critic
            val = critic(state)
            val_states.append(val)

            state = next_state

            if done or truncated:
                break

        ep_rewards.append(reward)
        recent_rewards.append(reward)

        R = 0
        actor_loss_list = []
        critic_loss_list = []
        returns = []

        # calculating list of returns from the given states
        for r in rewards[::-1]:
            # Calculate the discounted value
            R = r + GAMMA * R
            returns.insert(0, R)
        # normalizing
        returns = torch.tensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + eps)

        for log_prob, state_val, R in zip(log_probs, val_states, returns):
            advantage = R - state_val.item()
            actor_loss = -log_prob * advantage
            actor_loss_list.append(actor_loss)
            critic_loss = F.smooth_l1_loss(state_val, R.unsqueeze(0)) #torch.tensor([R]))
            critic_loss_list.append(critic_loss)

        actor_loss = torch.stack(actor_loss_list).sum()
        critic_loss = torch.stack(critic_loss_list).sum()

        actor_loss.backward()
        critic_loss.backward()
        actor_optim.step()
        critic_optim.step()

        print(f"Episode {ep}: Reward = {sum(rewards)}")

    torch.save(actor.state_dict(), "actor.pth")

def test(env):
    actor = Policy(env.observation_space.shape[0], env.action_space.n).eval()
    actor.load_state_dict(torch.load("actor.pth"))
    for t in range(10):
        state, _ = env.reset()
        done = False
        reward = 0
        while not done:
            state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            probs = actor(state)
            dist = torch.distributions.Categorical(probs)
            action = torch.argmax(probs, dim=-1).item()
            state, r, done, truncated, _ = env.step(action)
            reward += r
            if done or truncated:
                break
        print(f"Test {t}, Reward: {reward}")

if __name__ == '__main__':
    env = gym.make('LunarLander-v3', render_mode='human')
    # env = gym.make('CartPole-v1', render_mode="human")
    (state, _) = env.reset()
    # env.render()

    # environment information
    inp_shape = env.observation_space.shape[0]
    out_dim = env.action_space.n
    # print(env.action_space)
    # out_dim = 1
    print(f"state space dimensions: {inp_shape}")
    print(f"action space dimensions: {out_dim}")

    actor = Policy(inp_shape, out_dim)
    critic = Critic(inp_shape)
    # train(actor, critic)
    test(env)
