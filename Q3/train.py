import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import gymnasium as gym
from tqdm import tqdm
import os
import matplotlib.pyplot as plt
from utils import ReplayBuffer, GaussianPolicy, Critic
from dmc import make_dmc_env

def make_env():
	# Create environment with state observations
	env_name = "humanoid-walk"
	env = make_dmc_env(env_name, np.random.randint(0, 1_000_000), flatten=True, use_pixels=False)
	return env


# Soft Actor-Critic agent
class SACAgent:
    def __init__(self, state_dim, action_dim, device, lr=3e-4):
        self.device = device
        self.gamma = 0.99
        self.tau = 0.005
        # self.alpha = 0.2
        
        self.target_entropy = torch.tensor(-action_dim, dtype=torch.float32).to(device)
        self.log_alpha = torch.tensor(-1.0, requires_grad=True, device=device)
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=1e-4)

        # Actor
        self.policy = GaussianPolicy(state_dim, action_dim, hidden_dim=512).to(device)
        self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=lr)

        # Critics
        self.critic = Critic(state_dim, action_dim, hidden_dim=512).to(device)
        self.critic_target = Critic(state_dim, action_dim, hidden_dim=512).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_target.eval()
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr)

    @property
    def alpha(self):
        return self.log_alpha.exp()

    def select_action(self, state, eval=False):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            if eval:
                mu, _ = self.policy(state)
                action = torch.tanh(mu)
            else:
                action, _ = self.policy.sample(state)
        return action.cpu().data.numpy().flatten()

    def update(self, replay_buffer, batch_size=256):
        states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)

        # Compute target value: Q - alpha * log_prob
        with torch.no_grad():
            next_actions, next_log_probs = self.policy.sample(next_states) # Sample actions from policy for next_states
            q1_next, q2_next = self.critic(next_states, next_actions)
            q_next = torch.min(q1_next, q2_next) - self.alpha * next_log_probs
            target_value = rewards + (1 - dones) * self.gamma * q_next.squeeze(-1).detach()
        
        # Update Q networks
        q1_pred, q2_pred = self.critic(states, actions)
        q1_pred = q1_pred.squeeze(-1)
        q2_pred = q2_pred.squeeze(-1)
        critic_loss = F.mse_loss(q1_pred, target_value) + F.mse_loss(q2_pred, target_value)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Update policy
        action_pi, log_pi = self.policy.sample(states)
        q1_pi, q2_pi = self.critic(states, action_pi)
        q_pi = torch.min(q1_pi, q2_pi)
        policy_loss = (self.alpha * log_pi - q_pi).mean()

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()

        # Soft update value_target
        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - self.tau) + param.data * self.tau)

# Training loop for humanoid
if __name__ == '__main__':
    env = make_env()
    start_episode = 0

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    agent = SACAgent(state_dim=state_dim, action_dim=action_dim, device=device)
    replay_buffer = ReplayBuffer(capacity=1_000_000)
    if os.path.exists("checkpoints/") and start_episode != 0:
        print("Loading pre-trained model...")
        agent.policy.load_state_dict(torch.load(f"checkpoints/sac_policy_humanoid_{start_episode}.pth"))
        agent.critic.load_state_dict(torch.load(f"checkpoints/sac_critic_humanoid_{start_episode}.pth"))
        agent.critic_target.load_state_dict(torch.load(f"checkpoints/sac_critic_target_humanoid_{start_episode}.pth"))
        agent.log_alpha = torch.load(f"checkpoints/sac_log_alpha_humanoid_{start_episode}.pth")


    episodes = 5000
    batch_size = 256
    random_eps = 50
    rewards_per_episode = []
    for ep in tqdm(range(episodes)):
        state, _ = env.reset()
        episode_reward = 0
        while True:
            if ep < random_eps:
                action = np.random.uniform(-1.0, 1.0, size=21)
            else:
                action = agent.select_action(state)
            next_state, reward, done, trunc, _ = env.step(action)
            replay_buffer.push(state, action, reward, next_state, float(done))
            state = next_state
            episode_reward += reward

            if len(replay_buffer) > batch_size:
                agent.update(replay_buffer, batch_size)
            if done or trunc:
                break
        
        rewards_per_episode.append(episode_reward)            

        if (ep + 1) % 10 == 0:
            print(f"Episode {ep+1}, Reward: {episode_reward:.2f}")
        
        if (ep + 1) % 100 == 0:
            print(f"Episode {ep}, Avg Reward: {episode_reward:.2f}")
            if not os.path.exists("checkpoints/"):
                os.makedirs("checkpoints/")
            torch.save(agent.policy.state_dict(), f"checkpoints/sac_policy_humanoid_{start_episode+ep+1}.pth")
            torch.save(agent.critic.state_dict(), f"checkpoints/sac_critic_humanoid_{start_episode+ep+1}.pth")
            torch.save(agent.critic_target.state_dict(), f"checkpoints/sac_critic_target_humanoid_{start_episode+ep+1}.pth")
            torch.save(agent.log_alpha, f"checkpoints/sac_log_alpha_humanoid_{start_episode+ep+1}.pth")
        
        
    env.close()

    plt.plot(rewards_per_episode)
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('SAC Training Curve')
    plt.savefig("sac_humanoid_training.png")
