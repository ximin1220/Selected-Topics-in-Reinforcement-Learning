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
        scenario: str = "circle_cw_competition_collisionStop",
        render_mode: str = "rgb_array_birds_eye",
        reset_when_collision: bool = False,
        obs_size: int = 84,
        collision_penalty: float = 3.0,
        action_smoothness_penalty: float = 0.05,
        steering_penalty_scale: float = 0.02,
        speed_reward_scale: float = 0.10,
        reward_clip: float = 8.0,
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

        observation_spaces = {k: v for k, v in self.env.observation_space.items()}
        assert (
            self.camera_name in observation_spaces
        ), f"One of the sensors must be {self.camera_name}. Check the scenario file."

        self.render_mode = render_mode
        self.obs_size = obs_size
        self.collision_penalty = collision_penalty
        self.action_smoothness_penalty = action_smoothness_penalty
        self.steering_penalty_scale = steering_penalty_scale
        self.speed_reward_scale = speed_reward_scale
        self.reward_clip = reward_clip
        self._last_action = None

        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=float32)
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(1, obs_size, obs_size), dtype=np.uint8
        )

    def _observation_postprocess(self, obs):
        obs = obs[self.camera_name].astype(np.uint8)
        obs = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        obs = cv2.resize(obs, (self.obs_size, self.obs_size), interpolation=cv2.INTER_AREA)
        return obs[None, ...]

    def reset(self, *args, **kwargs: dict):
        self._last_action = None
        options = dict(kwargs.get("options") or {})
        options.setdefault("mode", "random")
        kwargs["options"] = options

        obs, *others = self.env.reset(*args, **kwargs)
        obs = self._observation_postprocess(obs)
        return obs, *others

    def step(self, actions):
        motor_action, steering_action = actions

        motor_action = float(np.clip(motor_action + np.random.normal(scale=0.001), -1.0, 1.0))
        steering_action = float(
            np.clip(steering_action + np.random.normal(scale=0.01), -1.0, 1.0)
        )

        dict_actions = OrderedDict(
            [
                (self.motor_name, array(motor_action, dtype=float32)),
                (self.steering_name, array(steering_action, dtype=float32)),
            ]
        )

        obs, reward, terminated, truncated, info = self.env.step(dict_actions)
        obs = self._observation_postprocess(obs)

        shaped = float(reward)
        if info.get("wall_collision"):
            shaped -= self.collision_penalty
        if info.get("collision_penalties"):
            shaped -= self.collision_penalty * len(info["collision_penalties"])
        if info.get("velocity") is not None:
            v = np.asarray(info["velocity"], dtype=np.float32)
            shaped += self.speed_reward_scale * float(np.linalg.norm(v[:2]))

        steering = float(np.asarray([motor_action, steering_action], dtype=np.float32)[1])
        shaped -= self.steering_penalty_scale * abs(steering)

        if self.action_smoothness_penalty:
            a = np.asarray([motor_action, steering_action], dtype=np.float32)
            if self._last_action is not None:
                shaped -= self.action_smoothness_penalty * float(
                    np.linalg.norm(a - self._last_action)
                )
            self._last_action = a

        if self.reward_clip > 0:
            shaped = float(np.clip(shaped, -self.reward_clip, self.reward_clip))

        return obs, shaped, terminated, truncated, info

    def render(self):
        return self.env.render()
