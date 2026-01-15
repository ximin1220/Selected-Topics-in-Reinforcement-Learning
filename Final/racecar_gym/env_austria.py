from collections import OrderedDict

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

        observation_spaces = {k: v for k, v in self.env.observation_space.items()}
        assert (
            self.camera_name in observation_spaces
        ), f"One of the sensors must be {self.camera_name}. Check the scenario file."

        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=float32)
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(3, 128, 128), dtype=np.uint8
        )

    def _observation_postprocess(self, obs):
        return obs[self.camera_name].astype(np.uint8).transpose(2, 0, 1)

    def reset(self, *args, **kwargs: dict):
        options = dict(kwargs.get("options") or {})
        options.setdefault("mode", "random")
        kwargs["options"] = options

        obs, *others = self.env.reset(*args, **kwargs)
        obs = self._observation_postprocess(obs)
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
        obs, *others = self.env.step(dict_actions)
        obs = self._observation_postprocess(obs)
        return obs, *others

    def render(self):
        return self.env.render()
