#!/usr/bin/env python3
"""
Train Austria model using env_austria_gray_resize.py (sample2-style).
No CLI arguments needed.
"""

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.vec_env import SubprocVecEnv, VecFrameStack, VecMonitor

from racecar_gym.env_austria_gray2 import RaceEnv

SCENARIO = "austria_competition_collisionStop"
TOTAL_TIMESTEPS = 5_000_000
FRAME_STACK = 8
N_ENVS = 40

VERSION_NAME = "v1"
LOG_DIR = "runs/austria_env" + VERSION_NAME
CHECKPOINT_DIR = "checkpoints/austria_env" + VERSION_NAME
BEST_DIR = "checkpoints/best_austria_env" + VERSION_NAME
CHECKPOINT_DIR = "checkpoints/periodic_austria_env" + VERSION_NAME
CHECKPOINT_FREQ = 50_000


def make_env(rank: int):
    def _thunk():
        env = RaceEnv(scenario=SCENARIO)
        env.reset(seed=rank)
        return env

    return _thunk


def main():
    env = SubprocVecEnv([make_env(i) for i in range(N_ENVS)])
    env = VecFrameStack(env, n_stack=FRAME_STACK, channels_order="first")
    env = VecMonitor(env)

    eval_env = SubprocVecEnv([make_env(i) for i in range(4)])
    eval_env = VecFrameStack(eval_env, n_stack=FRAME_STACK, channels_order="first")
    eval_env = VecMonitor(eval_env)

    model = PPO(
        "CnnPolicy",
        env,
        verbose=1,
        tensorboard_log=LOG_DIR,
        device="auto",
        learning_rate=3e-4,
        use_sde=True,
        n_steps=1024,
        batch_size=64,
        n_epochs=10,
        clip_range=0.2,
    )

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=BEST_DIR,
        log_path=f"{LOG_DIR}/eval",
        eval_freq=20_000,
        n_eval_episodes=5,
        deterministic=True,
        render=False,
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=CHECKPOINT_FREQ,
        save_path=CHECKPOINT_DIR,
        name_prefix="model",
    )

    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        callback=[eval_callback, checkpoint_callback],
        progress_bar=True,
    )
    model.save(f"{CHECKPOINT_DIR}/final_model.zip")


if __name__ == "__main__":
    main()
