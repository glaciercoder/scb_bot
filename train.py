import sys, os
import gymnasium as gym
from stable_baselines3.sac import SAC
from stable_baselines3.common.monitor import Monitor
from datetime import datetime
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor

sys.path.append("./Model")
from scb_bot_env import ScbBotEnv

log_path = "./data"
model_path = './Model/saved_models'
env = DummyVecEnv([
    lambda: ScbBotEnv(port=23000),
    lambda: ScbBotEnv(port=23001),
    # lambda: ScbBotEnv(port=23002),
    # lambda: ScbBotEnv(port=23003),
    # lambda: ScbBotEnv(port=23004),
    # lambda: ScbBotEnv(port=23005)
])
env = VecMonitor(env, log_path)

# Create a new model
model = SAC(policy='MlpPolicy', env=env, learning_rate=7e-4, verbose=True, tensorboard_log=log_path)

# Learning the model
print('Learning the model')
model.learn(total_timesteps=20000)
print('Finished')
model_name = str(datetime.now().strftime("%Y-%m-%d %H-%M-%S"))
model.save(os.path.join(model_path, model_name))
print("Model saved as " + model_name)
