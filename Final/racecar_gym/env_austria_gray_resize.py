from collections import OrderedDict

import cv2
import gymnasium as gym
import numpy as np
from numpy import array, float32

# noinspection PyUnresolvedReferences
import racecar_gym.envs.gym_api


class RaceEnv(gym.Env):
    camera_name = "camera_competition"
    motor_name = "motor_competition"
    steering_name = "steering_competition"

    def __init__(
        self,
        scenario: str = "austria_competition_collisionStop",
        render_mode: str = "rgb_array_birds_eye",
        reset_when_collision: bool = False,
        obs_size: int = 84,
        **kwargs,
    ):
        self.scenario = scenario.upper()[0] + scenario.lower()[1:]
        self.env_id = f"SingleAgent{self.scenario}-v0"
        self.env = gym.make(
            id=self.env_id,
            render_mode=render_mode,
            reset_when_collision=reset_when_collision,
            **kwargs,
        )
        self.render_mode = render_mode
        self.obs_size = obs_size

        observation_spaces = {k: v for k, v in self.env.observation_space.items()}
        assert (
            self.camera_name in observation_spaces
        ), f"One of the sensors must be {self.camera_name}. Check the scenario file."

        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=float32)
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(1, obs_size, obs_size), dtype=np.uint8
        )
        self.prev_info = {}

    def _observation_postprocess(self, obs):
        obs = obs[self.camera_name].astype(np.uint8)
        obs = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        obs = cv2.resize(obs, (self.obs_size, self.obs_size), interpolation=cv2.INTER_AREA)
        return obs[None, ...]

    def reset(self, *args, **kwargs: dict):
        options = dict(kwargs.get("options") or {})
        options.setdefault("mode", "random")
        kwargs["options"] = options

        obs, *others = self.env.reset(*args, **kwargs)
        obs = self._observation_postprocess(obs)
        self.prev_info["motor"] = 0.0
        self.prev_info["steering"] = 0.0
        self.prev_info["state"] = others[0].copy()
        return obs, *others

    def step(self, actions):
        motor_action, steering_action = actions

        motor_action = np.clip(motor_action + np.random.normal(scale=0.001), -1.0, 1.0)
        steering_action = np.clip(steering_action + np.random.normal(scale=0.01), -1.0, 1.0)

        dict_actions = OrderedDict(
            [
                (self.motor_name, array(motor_action, dtype=float32)),
                (self.steering_name, array(steering_action, dtype=float32)),
            ]
        )

        obs, reward, terminated, truncated, state = self.env.step(dict_actions)
        obs = self._observation_postprocess(obs)

        reward = 0.0
        truncated = bool(truncated) or (state.get("time", 0) >= 100)

        if state.get("checkpoint") != self.prev_info["state"].get("checkpoint"):
            reward += 10.0

        reward += 0.5 * float(motor_action)
        reward -= 0.2 * (
            abs(motor_action - self.prev_info["motor"])
            + abs(steering_action - self.prev_info["steering"])
        )
        reward -= 0.05 * abs(steering_action)

        if state.get("progress", 0.0) > self.prev_info["state"].get("progress", 0.0):
            reward += 400.0 * (
                state["progress"] - self.prev_info["state"]["progress"]
            )
        elif state.get("progress", 0.0) == self.prev_info["state"].get("progress", 0.0):
            reward -= 0.2

        if state.get("wall_collision"):
            reward = -250.0
            terminated = True

        self.prev_info["motor"] = float(motor_action)
        self.prev_info["steering"] = float(steering_action)
        self.prev_info["state"] = state.copy()

        return obs, reward, terminated, truncated, state

    def render(self):
        return self.env.render()
