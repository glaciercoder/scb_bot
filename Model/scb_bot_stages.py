import numpy as np
from scipy.spatial.transform import Rotation as R
import gymnasium as gym
from gymnasium.utils import seeding
from gymnasium import spaces, logger
import time
import yaml
import os
import math

from coppeliasim_zmqremoteapi_client import RemoteAPIClient

from scb_bot_model import ScbBotModel

class ScbBotEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, port=23000) -> None:
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
        self.total_energy = 0 
        self.target_position = np.asarray(self.env_params['target_pose'][:3])
        self.target_orientation = np.asarray(self.env_params['target_pose'][3:])
        self.leave_ground = False


        # Get coppeliasim remote API
        print("Connecting to Coppeliasim......")
        self.client = RemoteAPIClient(port=port)
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
        self.total_energy = 0
        self.leave_ground = False
        self.scb_bot.reset_model()
        self.sim.stopSimulation()
        self.state = np.zeros(self.env_params['ob_dim'])
        self.state[10] = self.target_position[1]
        time.sleep(self.env_params['sleep_time'])
        self.sim.setStepping(True)
        self.sim.startSimulation()

        return np.array(self.state, dtype=np.float32), {}

    def step(self, action:np.ndarray):
        #  print(f"action:{action}" )
        # Set action 
        # wait until the robot lands
        sim_time = self.sim.getSimulationTime()
        if sim_time < self.env_params['start_time']:
            self.scb_bot.set_torques(np.zeros(3))
        while sim_time < self.env_params['start_time']:
            self.sim.step()
            sim_time = self.sim.getSimulationTime()

        result, dist, coll = self.sim.checkDistance(self.scb_bot.robot_collection, self.scb_bot.floorhd, 0.05) 
        if not self.leave_ground and result == 1:
            self.scb_bot.set_torques(action)
            self.sim.step()
        else:
            self.leave_ground = True
        # Episode done checking

        # Update state
        self.scb_bot.update_state()
        sim_time = self.sim.getSimulationTime()
        rel_xyz = self.target_position - self.scb_bot.position
        rel_d = np.linalg.norm(rel_xyz[:2])
        v_parallel = np.linalg.norm(self.scb_bot.cm_vel[:2])
        v_vertical = self.scb_bot.cm_vel[2]
        cm_R = R.from_quat(self.scb_bot.orientation)
        rel_R = cm_R.inv() * R.from_quat(self.target_orientation)
        rel_eul = rel_R.as_euler('xyz')

        # Energy
        energy = 0
        for i in range(3):
            energy += 0.5 * 6e-4 * (self.scb_bot.joint_vels[i] ** 2 - self.scb_bot.joint_vels_last[i] ** 2)
        self.total_energy += energy


        # Get reward
        done = self.leave_ground
        reward = 0.0
        if done:
            reward = 500 * np.exp( - 10 *  np.absolute(self.scb_bot.cm_vel[0] * rel_xyz[1] - self.scb_bot.cm_vel[1] * rel_xyz[0])) + 500 * np.exp(- 10 * np.absolute(2*v_vertical * v_parallel - self.scb_bot.g * rel_d))
            reward -= 0.1 * self.total_energy 
        
        # Sim
        self.counts += 1
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
        state, reward, done, _, _ = env.step(action)
        print(reward)
        if done:
            env.reset()

    env.close()
        




    

