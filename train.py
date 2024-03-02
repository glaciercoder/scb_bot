import sys, os
import gymnasium as gym
from stable_baselines3.sac import SAC
from stable_baselines3.common.monitor import Monitor
from datetime import datetime
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor, SubprocVecEnv
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.callbacks import CallbackList
from stable_baselines3.common.callbacks import CheckpointCallback

sys.path.append("./Model")
from scb_bot_env import ScbBotEnv

if __name__ == '__main__':
    log_path = "./data"
    model_path = './Model/saved_models'
    env = SubprocVecEnv([
        lambda: ScbBotEnv(port=23000),
        lambda: ScbBotEnv(port=24000),
        lambda: ScbBotEnv(port=25000),
        lambda: ScbBotEnv(port=26000),
        lambda: ScbBotEnv(port=27000),
        lambda: ScbBotEnv(port=28000),
        lambda: ScbBotEnv(port=29000),
        lambda: ScbBotEnv(port=30000),
        lambda: ScbBotEnv(port=31000),
        lambda: ScbBotEnv(port=32000)
    ])
    env = VecMonitor(env, log_path)

    callback_save_best_model = EvalCallback(
        env, 
        best_model_save_path=model_path, 
        log_path=log_path, 
        eval_freq=500, deterministic=True, render=False)

    checkpoint_callback = CheckpointCallback(
      save_freq=50000,
      save_path=log_path,
      name_prefix=str(datetime.now().strftime("%Y-%m-%d %H-%M-%S")) + "ckpt",
      save_replay_buffer=True,
      save_vecnormalize=True,
    )
    callback_list = CallbackList([callback_save_best_model, checkpoint_callback])

    # Create a new model
    model = SAC(policy='MlpPolicy', env=env, 
                learning_rate=1e-3, verbose=True, train_freq=120, 
                learning_starts=1000, ent_coef=0.1, target_update_interval=120, gradient_steps=-1, 
                tensorboard_log=log_path)


    # Learning the model
    print('Learning the model')
    model.learn(total_timesteps=1000000, callback=callback_list)
    print('Finished')
    model_name = str(datetime.now().strftime("%Y-%m-%d %H-%M-%S"))
    model.save(os.path.join(model_path, model_name))
    print("Model saved as " + model_name)
