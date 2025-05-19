import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import gymnasium as gym
from tqdm import tqdm
import os
import matplotlib.pyplot as plt
from utils import ReplayBuffer, GaussianPolicy, QNetwork, ValueNetwork

def make_env():
    # Create Pendulum-v1 environment
    env = gym.make("Pendulum-v1", render_mode='rgb_array')
    return env

# Soft Actor-Critic agent
class SACAgent:
    def __init__(self, state_dim, action_dim, device, lr=3e-4):
        self.device = device
        self.gamma = 0.99
        self.tau = 0.00005
        self.alpha = 0.2  # entropy weight

        # Actor
        self.policy = GaussianPolicy(state_dim, action_dim).to(device)
        self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=lr)

        # Critics Q1 and Q2
        self.q1 = QNetwork(state_dim, action_dim).to(device)
        self.q2 = QNetwork(state_dim, action_dim).to(device)
        self.q1_optimizer = optim.Adam(self.q1.parameters(), lr=lr)
        self.q2_optimizer = optim.Adam(self.q2.parameters(), lr=lr)

        # Target value networks
        self.value = ValueNetwork(state_dim).to(device)  # V(s)
        self.value_target = ValueNetwork(state_dim).to(device)
        self.value_optimizer = optim.Adam(self.value.parameters(), lr=lr)
        self.value_target.load_state_dict(self.value.state_dict())

    def select_action(self, state, eval=False):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        if eval:
            mu, _ = self.policy(state)
            return torch.tanh(mu).cpu().data.numpy().flatten() * 2
        action, _ = self.policy.sample(state)
        return action.cpu().data.numpy().flatten()

    def update(self, replay_buffer, batch_size=256):
        states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)

        # Sample actions from policy for next_states
        next_actions, next_log_probs = self.policy.sample(next_states)

        # Compute target value: Q - alpha * log_prob
        with torch.no_grad():
            q1_next = self.q1.forward(next_states, next_actions)
            q2_next = self.q2.forward(next_states, next_actions)
        q_next = torch.min(q1_next, q2_next) - self.alpha * next_log_probs
        target_value = rewards + (1 - dones) * self.gamma * q_next.detach()
        # Update Q networks
        q1_pred = self.q1(states, actions)
        q2_pred = self.q2(states, actions)
        q1_loss = F.smooth_l1_loss(q1_pred, target_value)
        q2_loss = F.smooth_l1_loss(q2_pred, target_value)

        self.q1_optimizer.zero_grad()
        q1_loss.backward()
        self.q1_optimizer.step()

        self.q2_optimizer.zero_grad()
        q2_loss.backward()
        self.q2_optimizer.step()

        # Update Value network
        sampled_actions, log_probs = self.policy.sample(states)
        q1_val = self.q1(states, sampled_actions)
        q2_val = self.q2(states, sampled_actions)
        q_min = torch.min(q1_val, q2_val)
        value_pred = self.value(states)
        with torch.no_grad():
            value_target = (q_min - self.alpha * log_probs).detach()

        value_loss = F.smooth_l1_loss(value_pred, value_target)
        # value_loss = F.mse_loss(value_pred, value_target)
        self.value_optimizer.zero_grad()
        value_loss.backward()
        self.value_optimizer.step()

        # Update policy
        action_pi, log_pi = self.policy.sample(states)
        q1_pi = self.q1(states, action_pi)
        q2_pi = self.q2(states, action_pi)
        q_pi = torch.min(q1_pi, q2_pi)
        policy_loss = (self.alpha * log_pi - q_pi).mean()

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        # Soft update value_target
        for target_param, param in zip(self.value_target.parameters(), self.value.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - self.tau) + param.data * self.tau)

# Training loop for Pendulum-v1
if __name__ == '__main__':
    env = make_env()
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    agent = SACAgent(state_dim=state_dim, action_dim=action_dim, device=device)
    replay_buffer = ReplayBuffer(capacity=100000)

    # if os.path.exists("checkpoints_sac/"):
    #     agent.policy.load_state_dict(torch.load("checkpoints_sac/sac_pendulum.pth"))
    #     agent.q1.load_state_dict(torch.load("checkpoints_sac/sac_q1_pendulum.pth"))
    #     agent.q2.load_state_dict(torch.load("checkpoints_sac/sac_q2_pendulum.pth"))
    #     agent.value.load_state_dict(torch.load("checkpoints_sac/sac_value_pendulum.pth"))
    #     agent.value_target.load_state_dict(agent.value.state_dict())


    episodes = 1000
    max_steps = 200
    batch_size = 256
    start_steps = 1000  # fill replay buffer

    total_steps = 0
    rewards_per_episode = []
    for ep in tqdm(range(episodes)):
        state, _ = env.reset()
        episode_reward = 0
        for step in range(max_steps):
            if total_steps < start_steps:
                action = env.action_space.sample()
            else:
                action = agent.select_action(state)

            next_state, reward, done, trunc, _ = env.step(action)
            replay_buffer.push(state, action, reward, next_state, float(done))
            state = next_state
            episode_reward += reward
            total_steps += 1

            if len(replay_buffer) > batch_size:
                agent.update(replay_buffer, batch_size)
            if done or trunc:
                break
        
        rewards_per_episode.append(episode_reward)            

        if (ep + 1) % 10 == 0:
            print(f"Episode {ep+1}, Reward: {episode_reward:.2f}")
        
        if (ep + 1) % 100 == 0:
            print(f"Episode {ep}, Avg Reward: {episode_reward:.2f}")
            if not os.path.exists("checkpoints_sac/"):
                os.makedirs("checkpoints_sac/")
            torch.save(agent.policy.state_dict(), f"checkpoints_sac/sac_policy_pendulum_{ep+1}.pth")
            torch.save(agent.q1.state_dict(), f"checkpoints_sac/sac_q1_pendulum_{ep+1}.pth")
            torch.save(agent.q2.state_dict(), f"checkpoints_sac/sac_q2_pendulum_{ep+1}.pth")
            torch.save(agent.value.state_dict(), f"checkpoints_sac/sac_value_pendulum_{ep+1}.pth")
        
        
    env.close()

    plt.plot(rewards_per_episode)
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('SAC Training on Pendulum-v1')
    plt.savefig("sac_pendulum_training.png")
