import gymnasium as gym
import numpy as np
import torch
from utils import GaussianPolicy

# Do not modify the input of the 'act' function and the '__init__' function. 
class Agent(object):
    """Agent that acts randomly."""
    def __init__(self):
        self.action_space = gym.spaces.Box(-1.0, 1.0, (1,), np.float32)
        self.policy = GaussianPolicy(67, 21, 512)  # state_dim=67, action_dim=21
        self.policy.load_state_dict(torch.load("checkpoints/sac_humanoid_policy.pth"))
        self.policy.eval()  # Set the policy to evaluation mode
    
    def act(self, observation):
        """Returns an action given an observation."""
        # Convert observation to tensor
        observation = torch.FloatTensor(observation)
        # Get action from policy
        action, _ = self.policy(observation)
        return torch.tanh(action).cpu().data.numpy().flatten()
