import numpy as np
import gymnasium as gym
from gymnasium.utils import seeding
from gymnasium import spaces, logger
import time
import yaml
import os

from coppeliasim_zmqremoteapi_client import RemoteAPIClient

from scb_bot_model import ScbBotModel

class ScbBotEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self) -> None:
        super().__init__()

        # Get params
        print("Loading Parameters......")
        yaml_path = os.path.join(os.getcwd(), "config/env_params.yaml")
        with open(yaml_path, 'r') as file:
            params = yaml.safe_load(file)
        self.model_params = params['model_params']
        self.env_params = params['env_params']

        # Set env params
        self.action_space = spaces.Box(low=np.asarray(self.env_params['ac_low']), 
                                       high=np.asarray(self.env_params['ac_high']), 
                                       dtype=np.float32)
        print(f'Action Space:{self.action_space}')
        self.observation_space = spaces.Box(low=np.asarray(self.env_params['ob_low'], dtype=np.float32), 
                                       high=np.asarray(self.env_params['ob_high'],dtype=np.float32), 
                                       dtype=np.float32)
        print(f'Observation Space:{self.observation_space}')
        self.state = np.zeros(self.env_params['ob_dim'])
        self.seed(self.env_params['seed'])
        self.counts = 0

        # Get coppeliasim remote API
        print("Connecting to Coppeliasim......")
        self.client = RemoteAPIClient()
        self.sim = self.client.getObject('sim')
        self.sim.setStepping(True)

        # Init scb model
        print("Initializing SBC Robot Model......")
        self.scb_bot = ScbBotModel(self.sim, self.model_params)
        self.sim.startSimulation()
        
    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)

    def reset(self, seed=None):
        print("Reset simulation")
        self.counts = 0
        self.sim.stopSimulation()
        self.state = np.zeros(self.env_params['ob_dim'])
        time.sleep(self.env_params['sleep_time'])
        self.sim.setStepping(True)
        self.scb_bot.reset_model()
        self.sim.startSimulation()

        return np.array(self.state, dtype=np.float32), {}

    def step(self, action:np.ndarray):
        # Set action and update state
        self.scb_bot.set_torques(action)
        self.sim.step()
        
        time.sleep(self.env_params['sleep_time'])
        self.scb_bot.update_state()
        self.counts += 1

        # Episode done checking
        # Run a distance beyond the given radius
        dist = np.sqrt(self.state[0]**2 + self.state[1]**2)
        done =  (dist >= self.env_params['dist_max'])

        # Get new state and reward
        self.state = np.concatenate([self.scb_bot.position, self.scb_bot.orientation, self.scb_bot.joint_torques])
        reward = dist

        return self.state, reward, done, False, {}

    def render(self):
        return None
    
    def close(self):
        self.sim.stopSimulation()
        print("Close the environment")
        return None


if __name__ == '__main__':
    env = ScbBotEnv()
    env.reset()
    print("Reset Finished")
    for _ in range(500):
        action = env.action_space.sample()
        env.step(action)
        print(env.counts)
        print(env.state)

    env.close()
        




    

