import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import time
import matplotlib.pyplot as plt

discount_factor = 0.99
num_episodes = 350

class Policy(nn.Module):
    def __init__(self, input_shape, output_dim):
        super().__init__()
        self.inp_shape = input_shape
        self.out_dim = output_dim
        self.discount = 0.99
        self.num_episodes = num_episodes
        self.layers = nn.Sequential(
            nn.Linear(inp_shape, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim),
            nn.Softmax(dim=-1)
        )
        self.layers1 = nn.Sequential(
            nn.Linear(inp_shape, 32),
            nn.ReLU(),
            nn.Linear(32, output_dim),
            nn.Softmax(dim=-1)
        )

    def forward(self, x):
        return self.layers(x)

def compute_returns(rewards, disc_factor):
    t_steps = np.arange(len(rewards))
    r = rewards * disc_factor ** t_steps
    r = np.cumsum(r[::-1])[::-1] / disc_factor ** t_steps
    returns = r
    returns = (returns - returns.mean()) / (returns.std() + 1e-8)
    return returns

def compute_loss(log_probs, returns):
    loss = []
    # for log_prob, returns in zip(log_probs, returns):
    #     loss.append(-log_prob * returns)

    for log_prob, R in zip(log_probs, returns):
        loss.append(-log_prob * R)

    return torch.stack(loss).sum()

def plot_rewards(rewards):
    for i, reward_list in enumerate(rewards):
        plt.figure(figsize=(6, 4))
        plt.plot(reward_list)
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        plt.title(f'Episode Rewards (List {i+1})')
        plt.tight_layout()
        plt.savefig(f"rewards_list_{i+1}.png")  # Save each plot
        plt.close()  # Close the figure to free memory

def train(input_shape, output_dim, discount_factor, lr):
    policy = Policy(input_shape, output_dim)
    optimizer = optim.Adam(policy.parameters(), lr=lr)
    ep_rewards = []
    for episode in range(1, policy.num_episodes + 1):
        state, _ = env.reset()
        log_probs = []
        rewards = []
        episode_ended = False

        while not episode_ended:
            state = torch.as_tensor(state, dtype=torch.float32)
            # env.render()
            action_probs = policy(state) # getting action probabilities

            dist = torch.distributions.Categorical(action_probs) # sampling action
            action = dist.sample()
            log_prob = dist.log_prob(action) # adding log probability to array
            log_probs.append(log_prob)

            # executing the action and storing the next state, rewar
            next_state, reward, done, truncated, _ = env.step(action.item())
            rewards.append(reward)
            state = next_state

            if done or truncated:
                episode_ended = True
                returns = compute_returns(rewards, discount_factor)
                loss = compute_loss(log_probs, returns)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                ep_rewards.append(sum(rewards))
                print(f"Episode num {episode}, reward : {sum(rewards)}")

    torch.save(policy.state_dict(), "reinforce_policy.pth")
    return ep_rewards

def test(input_shape, output_dim):
    policy = Policy(input_shape, output_dim).eval()
    policy.load_state_dict(torch.load("reinforce_policy.pth"))

    # Loop over each episode for testing
    for episode in range(1, 20):
        state, _ = env.reset() # Reset the environment and get the initial state
        env.render()
        done = False
        truncation = False
        episode_reward = 0

        # Loop until the episode is done or truncated
        while not done and not truncation:
            state = torch.as_tensor(state, dtype=torch.float32)
            action_probs = policy(state) # Get action probabilities from the policy network
            action = torch.argmax(action_probs, dim=0).item() # Select the action with the highest probability (argmax)
            state, reward, done, truncation,_ = env.step(action) # Take a step in the environment
            episode_reward += float(reward) # Accumulate the reward for the episode

        print(f"Episode {episode}, Reward: {episode_reward}")

    env.close() # Close the environment (for closing the Pygame rendering window)

if __name__ == '__main__':
    # env = gym.make('LunarLander-v3', render_mode='human')
    env = gym.make('CartPole-v1', render_mode="human")
    (state, _) = env.reset()
    # env.render()

    # environment information
    inp_shape = env.observation_space.shape[0]
    out_dim = env.action_space.n
    print(env.action_space)
    # out_dim = 1
    print(f"state space dimensions: {inp_shape}")
    print(f"action space dimensions: {out_dim}")
    hyp = [0.99, 9.8e-4]
    disc = [0.99, 0.89, 0.79]
    lr = [9.8e-4, 0.005, 0.01]
    d_rewards = []
    for d in disc:
        d_rewards.append(train(inp_shape, out_dim, d, hyp[1]))
    # train(inp_shape, out_dim, hyp[0], hyp[1])
    # test(inp_shape, out_dim)

    plot_rewards(d_rewards)
