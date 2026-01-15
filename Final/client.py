import argparse
import json
from collections import deque
from typing import Sequence, Tuple

import cv2
import numpy as np
import requests

try:
    from stable_baselines3 import PPO
except ImportError:
    PPO = None


def connect(agent, url: str = 'http://localhost:5000'):
    while True:
        # Get the observation
        response = requests.get(f'{url}')
        if response.status_code != 200:
            print(f'[GET] HTTP {response.status_code}, body={response.text}')
            break
        try:
            payload = json.loads(response.text)
        except json.JSONDecodeError:
            print(f'[GET] Non-JSON response, body={response.text}')
            break
        if payload.get('error'):
            print(payload['error'])
            break
        if payload.get('terminal'):
            print('Episode finished (server says terminal).')
            return
        obs = payload['observation']
        obs = np.array(obs).astype(np.uint8)

        # Decide an action based on the observation (Replace this with your RL agent logic)
        action_to_take = agent.act(obs)  # Replace with actual action

        # Send an action and receive new observation, reward, and done status
        response = requests.post(f'{url}', json={'action': action_to_take.tolist()})
        if response.status_code != 200:
            print(f'[POST] HTTP {response.status_code}, body={response.text}')
            break
        try:
            result = json.loads(response.text)
        except json.JSONDecodeError:
            print(f'[POST] Non-JSON response, body={response.text}')
            break
        if result.get('error'):
            print(result['error'])
            break

        terminal = result['terminal']

        if terminal:
            print('Episode finished.')
            return


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--url', type=str, default='http://localhost:5000', help='The url of the server.')
    parser.add_argument('--model', type=str, default=None, help='Path to PPO checkpoint to run instead of random.')
    parser.add_argument('--frame-stack', type=int, default=4, help='Frame stack used during training.')
    parser.add_argument('--discrete-actions', action='store_true', help='Use predefined discrete actions (match training).')
    args = parser.parse_args()


    class RandomAgent:
        def __init__(self, action_space):
            self.action_space = action_space

        def act(self, observation):
            return self.action_space.sample()


    class PPOAgent:
        def __init__(self, model_path: str, frame_stack: int, discrete: bool, actions: Sequence[Tuple[float, float]]):
            assert PPO is not None, "stable_baselines3 is required for PPOAgent"
            self.model = PPO.load(model_path, device="auto")
            self.frame_stack = deque(maxlen=frame_stack)
            self.discrete = discrete
            self.actions = [np.array(a, dtype=np.float32) for a in actions]

        def _preprocess(self, obs: np.ndarray) -> np.ndarray:
            # obs is CHW (3,128,128); convert to HWC
            if obs.ndim == 3 and obs.shape[0] in (1, 3):
                obs = np.transpose(obs, (1, 2, 0))
            gray = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
            resized = cv2.resize(gray, (84, 84), interpolation=cv2.INTER_AREA)
            self.frame_stack.append(resized)
            while len(self.frame_stack) < self.frame_stack.maxlen:
                self.frame_stack.append(resized)
            stack = np.stack(self.frame_stack, axis=-1)  # HWC where C=stack
            # SB3 model expects channel-first
            stack = np.transpose(stack, (2, 0, 1)).astype(np.uint8)
            return stack

        def act(self, observation):
            proc = self._preprocess(observation)
            action, _ = self.model.predict(proc, deterministic=True)
            if self.discrete:
                action = self.actions[int(action)]
            return action

    # Initialize the RL Agent
    if args.model:
        actions = [
            (1.0, 0.0),    # straight
            (0.8, 0.5),    # gentle right
            (0.8, -0.5),   # gentle left
            (0.4, 1.0),    # sharp right
            (0.4, -1.0),   # sharp left
        ]
        agent = PPOAgent(
            model_path=args.model,
            frame_stack=args.frame_stack,
            discrete=args.discrete_actions,
            actions=actions,
        )
    else:
        import gymnasium as gym
        agent = RandomAgent(
            action_space=gym.spaces.Box(low=np.array([-1, -1]), high=np.array([1, 1]), dtype=np.float32))

    connect(agent, url=args.url)
