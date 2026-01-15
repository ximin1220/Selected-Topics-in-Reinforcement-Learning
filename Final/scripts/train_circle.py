#!/usr/bin/env python3
"""
Train circle model with C2-style settings using env_circle_gray_resize.py.
No CLI arguments needed.
"""

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecFrameStack, VecMonitor

from racecar_gym.env_circle2 import RaceEnv

SCENARIO = "circle_cw_competition_collisionStop"
TOTAL_TIMESTEPS = 3000000
FRAME_STACK = 4
N_ENVS = 40

VERSION_NAME = "v1"
LOG_DIR = "runs/circle_" + VERSION_NAME
MODEL_PATH = "checkpoints/circle_" + VERSION_NAME + ".zip"
BEST_DIR = "checkpoints/best_circle_" + VERSION_NAME
CHECKPOINT_DIR = "checkpoints/periodic_circle_" + VERSION_NAME
CHECKPOINT_FREQ = 50_000
EVAL_FREQ = 20_000
EVAL_EPISODES = 5


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

    eval_env = DummyVecEnv([make_env(0)])
    eval_env = VecFrameStack(eval_env, n_stack=FRAME_STACK, channels_order="first")
    eval_env = VecMonitor(eval_env)

    model = PPO(
        "CnnPolicy",
        env,
        learning_rate=3e-4,
        n_steps=1024,
        batch_size=256,
        n_epochs=4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        vf_coef=0.5,
        tensorboard_log=LOG_DIR,
        verbose=1,
        device="auto",
    )

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=BEST_DIR,
        log_path=f"{LOG_DIR}/eval",
        eval_freq=EVAL_FREQ,
        n_eval_episodes=EVAL_EPISODES,
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
    model.save(MODEL_PATH)
    print(f"Saved model to {MODEL_PATH}")


if __name__ == "__main__":
    main()
