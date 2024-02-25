import numpy as np
from scipy.spatial.transform import Rotation as R
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
        self.action_space = spaces.Box(low= - np.asarray(self.env_params['ac_high']), 
                                       high=np.asarray(self.env_params['ac_high']), 
                                       dtype=np.float32)
        print(f'Action Space:{self.action_space}')
        self.observation_space = spaces.Box(low= - np.asarray(self.env_params['ob_high'], dtype=np.float32), 
                                       high=np.asarray(self.env_params['ob_high'],dtype=np.float32), 
                                       dtype=np.float32)
        print(f'Observation Space:{self.observation_space}')
        self.state = np.zeros(self.env_params['ob_dim'])
        self.seed(self.env_params['seed'])
        self.counts = 0
        self.target_position = np.asarray(self.env_params['target_pose'][:3])
        self.target_orientation = np.asarray(self.env_params['target_pose'][3:])


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
        self.sim.startSimulation()

        return np.array(self.state, dtype=np.float32), {}

    def step(self, action:np.ndarray):
        # Update state
        self.scb_bot.update_state()
        sim_time = self.sim.getSimulationTime()
        rel_xyz = self.target_position - self.scb_bot.position
        cm_R = R.from_quat(self.scb_bot.orientation)
        rel_R = cm_R.inv() * R.from_quat(self.target_orientation)
        rel_eul = rel_R.as_euler('xyz')
        error_xyz = np.linalg.norm(rel_xyz[0:2]) # Only care about xy error
        error_euler = np.linalg.norm(rel_eul)
        
        # Set action 
        if sim_time < self.env_params['start_time']:
            action = np.zeros(3)
        self.scb_bot.set_torques(action)
        self.sim.step()
        
        # Episode done checking
        done = (sim_time >= self.env_params['sim_time_th']) \
                or ((error_xyz <= self.env_params['error_xyz_lth']) 
                    and (error_euler <= self.env_params['error_euler_th']) 
                    and (np.linalg.norm(self.scb_bot.cm_vel_angular) <= self.env_params['error_v_ang_th']) 
                    and (np.linalg.norm(self.scb_bot.cm_vel) <= self.env_params['error_v_th'])) \
                or (error_xyz > self.env_params['error_xyz_max']) \
                or ((sim_time >= self.env_params['sim_time_mid'])  and (error_xyz > self.env_params['error_xyz_hth']))
        done = bool(done)

        # Get reward
        reward = 1

        # Sim
        self.counts += 1
        for i in range(self.env_params['sim_per_step']):
            self.sim.step()
        self.state = np.concatenate([self.scb_bot.joint_vels, 
                                     self.scb_bot.cm_vel, 
                                     self.scb_bot.cm_vel_angular, 
                                     rel_xyz,
                                     rel_eul,
                                     np.asarray([sim_time])
                                    ])
                    
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
        print(action)

    env.close()
        




    

