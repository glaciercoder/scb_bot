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

        robot_collection = self.sim.createCollection(0)
        self.sim.addItemToCollection(robot_collection, self.sim.handle_tree, self.scb_bot.bodyhd, 0)
        result, dist, coll = self.sim.checkDistance(robot_collection, self.scb_bot.floorhd, 0.05) 
        if not self.leave_ground and result == 1:
            self.scb_bot.set_torques(action)
            self.sim.step()
        else:
            self.leave_ground = True
        # Episode done checking
        # done = (sim_time >= self.env_params['sim_time_th']) \
        #         or ((error_xyz <= self.env_params['error_xyz_lth']) 
        #             and (error_euler <= self.env_params['error_euler_th']) 
        #             and (np.linalg.norm(self.scb_bot.cm_vel_angular) <= self.env_params['error_v_ang_th']) 
        #             and (np.linalg.norm(self.scb_bot.cm_vel) <= self.env_params['error_v_th'])) \
        #         or (error_xyz > self.env_params['error_xyz_max']) \
        #         or ((sim_time >= self.env_params['sim_time_mid'])  and (error_xyz > self.env_params['error_xyz_hth']))

        # Update state
        self.scb_bot.update_state()
        sim_time = self.sim.getSimulationTime()
        rel_xyz = self.target_position - self.scb_bot.position
        cm_R = R.from_quat(self.scb_bot.orientation)
        rel_R = cm_R.inv() * R.from_quat(self.target_orientation)
        rel_eul = rel_R.as_euler('xyz')
        error_xyz = np.linalg.norm(rel_xyz[0:2]) # Only care about xy error
        error_euler = np.linalg.norm(rel_eul)
        error_horizontal = np.linalg.norm(rel_xyz[0:2])

        # Get reward
        done = False
        reward = 0.0
        if self.leave_ground:
            result, dist, coll = self.sim.checkDistance(robot_collection, self.scb_bot.floorhd, 0.005)
            while result==0:
                for i in range(self.env_params['sim_per_step']):
                    self.sim.step()
                sim_time = self.sim.getSimulationTime()
                self.scb_bot.update_state()
                rel_xyz = self.target_position - self.scb_bot.position
                error_horizontal = np.linalg.norm(rel_xyz[0:2])

                if (sim_time >= self.env_params['sim_time_th']) or (np.linalg.norm(rel_xyz) > 3) or ((sim_time >= 60)  and (error_horizontal > 0.95)):
                    done = True
                    break
                result, dist, coll = self.sim.checkDistance(robot_collection, self.scb_bot.floorhd, 0.005)
            if done:
                reward = -100.0
            else:
                done = True
                reward = 10000.0 * math.exp(-5 * error_horizontal)
        else:
            #Accelaration Penalty
            acc = self.scb_bot.cm_vel - self.scb_bot.cm_vel_last
            acc_h = acc[0:2]
            acc_hnorm = acc_h / np.linalg.norm(acc_h)
            rel_xyz_hnorm = rel_xyz[0:2] / np.linalg.norm(rel_xyz[0:2])

            reward = 10000 * np.dot(acc_h, rel_xyz_hnorm) - abs(np.arccos(np.dot(acc_hnorm, rel_xyz_hnorm)))

            # energy penalty
            energy = 0
            for i in range(3):
                if (self.scb_bot.joint_vels[i]* self.scb_bot.cm_vel_last[i]) > 0:
                    energy += 0.5 * 6e-4 * abs(self.scb_bot.joint_vels[i] ** 2 - self.scb_bot.joint_vels_last[i] ** 2)
                else:
                    energy += 0.5 * 6e-4 * abs(self.scb_bot.joint_vels[i] ** 2 + self.scb_bot.joint_vels_last[i] ** 2)
            reward -= energy

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

    env.close()
        




    

