#!/usr/bin/env python3
"""
Train Austria with TD3 (sample2-style).
No CLI arguments needed.
"""

import numpy as np
from stable_baselines3 import TD3
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.vec_env import SubprocVecEnv, VecFrameStack, VecMonitor

from racecar_gym.env_austria_gray_resize import RaceEnv

SCENARIO = "austria_competition_collisionStop"
TOTAL_TIMESTEPS = 5000_000
FRAME_STACK = 8
N_ENVS = 40

LOG_DIR = "runs/td3_austria_env"
CHECKPOINT_DIR = "checkpoints/td3_austria_env"
BEST_DIR = "checkpoints/best_td3_austria_env"
CHECKPOINT_FREQ = 200_000


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

    action_noise = NormalActionNoise(mean=np.zeros(2), sigma=0.1 * np.ones(2))

    model = TD3(
        "CnnPolicy",
        env,
        verbose=1,
        tensorboard_log=LOG_DIR,
        device="auto",
        learning_rate=3e-4,
        buffer_size=500_000,
        learning_starts=50_000,
        batch_size=256,
        train_freq=1,
        gradient_steps=1,
        action_noise=action_noise,
        policy_delay=2,
        target_policy_noise=0.2,
        target_noise_clip=0.5,
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
        name_prefix="td3",
        save_replay_buffer=False,
        save_vecnormalize=False,
    )

    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        callback=[eval_callback, checkpoint_callback],
        progress_bar=True,
    )
    model.save(f"{CHECKPOINT_DIR}/final_model.zip")


if __name__ == "__main__":
    main()
