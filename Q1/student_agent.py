import gymnasium as gym
import numpy as np
from utils import GaussianPolicy
import torch

# Do not modify the input of the 'act' function and the '__init__' function. 
class Agent(object):
    """Agent that acts randomly."""
    def __init__(self):
        # Pendulum-v1 has a Box action space with shape (1,)
        # Actions are in the range [-2.0, 2.0]
        self.action_space = gym.spaces.Box(-2.0, 2.0, (1,), np.float32)
        self.policy = GaussianPolicy(3, 1)  # state_dim=3, action_dim=1
        self.policy.load_state_dict(torch.load("checkpoints/sac_pendulum.pth"))
        self.policy.eval()  # Set the policy to evaluation mode
    
    def act(self, observation):
        """Returns an action given an observation."""
        # Convert observation to tensor
        observation = torch.FloatTensor(observation).unsqueeze(0)
        # Get action from policy
        action, _ = self.policy.sample(observation)
        return torch.tanh(action).cpu().data.numpy().flatten() * 2