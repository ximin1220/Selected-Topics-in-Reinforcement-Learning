# dqn_agent_atari.py

import torch
import torch.nn as nn
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from base_agent import DQNBaseAgent, DQNParBaseAgent
from models.atari_model import AtariNetDQN, AtariNetDuelingDQN
import random


# --- 主要修改處 ---
import gymnasium as gym
import ale_py
gym.register_envs(ale_py)
from gymnasium.wrappers import AtariPreprocessing
from gymnasium.wrappers import FrameStackObservation  # <-- 從我們的新檔案導入 FrameStack
from gymnasium.vector import SyncVectorEnv

# --- 主要 Agent 程式碼 ---
class AtariDQNAgent(DQNBaseAgent):
  def __init__(self, config):
    super(AtariDQNAgent, self).__init__(config)

    # 疊圖，建立state
    env = gym.make(config["env_id"], frameskip=1)
    env = AtariPreprocessing(env, screen_size=84, grayscale_obs=True)
    self.env = FrameStackObservation(env, 4)

    test_env = gym.make(config["env_id"], frameskip=1)
    test_env = AtariPreprocessing(test_env, screen_size=84, grayscale_obs=True)
    self.test_env = FrameStackObservation(test_env, 4)

    # Online Network（ Behavior Network）.
    self.behavior_net = AtariNetDQN(self.env.action_space.n)
    self.behavior_net.to(self.device)
    # Target Network
    self.target_net = AtariNetDQN(self.env.action_space.n)
    self.target_net.to(self.device)
    self.target_net.load_state_dict(self.behavior_net.state_dict())
    self.lr = config["learning_rate"]
    self.optim = torch.optim.Adam(self.behavior_net.parameters(), lr=self.lr, eps=1.5e-4)

  def _obs_to_tensor(self, observation):
    obs_array = np.array(observation)
    return torch.tensor(obs_array, dtype=torch.float32, device=self.device).unsqueeze(0)

  def decide_agent_actions(self, observation, epsilon=0.0, action_space=None):
    if random.random() < epsilon:
      action = action_space.sample()
    else:
      with torch.no_grad(): #用 behavior_net 來選動作
        obs_tensor = self._obs_to_tensor(observation)
        q_values = self.behavior_net(obs_tensor)
        action = q_values.argmax().item()
    return action
  
  def update_behavior_network(self):
    state, action, reward, next_state, done = self.replay_buffer.sample(self.batch_size, self.device)
    q_values = self.behavior_net(state)
    q_value = q_values.gather(1, action.long())

    with torch.no_grad():
      q_next_values = self.target_net(next_state)
      q_next = q_next_values.max(1)[0].unsqueeze(1) #.max(1)[0]取最大值 .unsqueeze對其維度
      q_target = reward + self.gamma * q_next * (1 - done) #終止狀態（done=True）時不加 γ 部分

    #Loss and backward
    criterion = nn.SmoothL1Loss()
    loss = criterion(q_value, q_target)
    self.writer.add_scalar('DQN/Loss', loss.item(), self.total_time_step)
    self.optim.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(self.behavior_net.parameters(), 1.0)
    self.optim.step()

class AtariDDQNAgent(DQNBaseAgent):
  def __init__(self, config):
    super(AtariDDQNAgent, self).__init__(config)
    
    env = gym.make(config["env_id"], frameskip=1)
    env = AtariPreprocessing(env, screen_size=84, grayscale_obs=True)
    self.env = FrameStackObservation(env, 4)

    test_env = gym.make(config["env_id"], frameskip=1)
    test_env = AtariPreprocessing(test_env, screen_size=84, grayscale_obs=True)
    self.test_env = FrameStackObservation(test_env, 4)

    self.behavior_net = AtariNetDQN(self.env.action_space.n)
    self.behavior_net.to(self.device)
    self.target_net = AtariNetDQN(self.env.action_space.n)
    self.target_net.to(self.device)
    self.target_net.load_state_dict(self.behavior_net.state_dict())
    self.lr = config["learning_rate"]
    self.optim = torch.optim.Adam(self.behavior_net.parameters(), lr=self.lr, eps=1.5e-4)

  def _obs_to_tensor(self, observation):
    obs_array = np.array(observation)
    return torch.tensor(obs_array, dtype=torch.float32, device=self.device).unsqueeze(0)

  def decide_agent_actions(self, observation, epsilon=0.0, action_space=None):
    if random.random() < epsilon:
      action = action_space.sample()
    else:
      with torch.no_grad():
        obs_tensor = self._obs_to_tensor(observation)
        q_values = self.behavior_net(obs_tensor)
        action = q_values.argmax().item()
    return action
  
  def update_behavior_network(self):
    state, action, reward, next_state, done = self.replay_buffer.sample(self.batch_size, self.device)
    
    q_values = self.behavior_net(state)
    q_value = q_values.gather(1, action.long())
    
    with torch.no_grad():
      # --- ✅ DDQN 修改處 Start ---
      
      # 1. 使用 Behavior Network "選擇" 最佳的下一步動作 (取得動作的 index)
      # .max(1) 回傳 (values, indices)，我們取 [1] 也就是 indices
      best_next_actions = self.behavior_net(next_state).max(1)[1].unsqueeze(1)
      
      # 2. 使用 Target Network "評估" 這些被選中動作的 Q 值
      # 我們用 gather() 來根據 best_next_actions 的 index，從 target_net 的輸出中取出對應的 Q 值
      q_next = self.target_net(next_state).gather(1, best_next_actions)
      
      # --- DDQN 修改處 End ---
      
      # 3. 計算目標 Q 值 (這行不變)
      q_target = reward + self.gamma * q_next * (1 - done)

    criterion = nn.SmoothL1Loss()
    loss = criterion(q_value, q_target)
    
    self.writer.add_scalar('DQN/Loss', loss.item(), self.total_time_step)
    
    self.optim.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(self.behavior_net.parameters(), 1.0)
    self.optim.step()


class AtariDQNDuelAgent(DQNBaseAgent):
  def __init__(self, config):
    super(AtariDQNDuelAgent, self).__init__(config)
    
    env = gym.make(config["env_id"], frameskip=1)
    env = AtariPreprocessing(env, screen_size=84, grayscale_obs=True)
    self.env = FrameStackObservation(env, 4)

    test_env = gym.make(config["env_id"], frameskip=1)
    test_env = AtariPreprocessing(test_env, screen_size=84, grayscale_obs=True)
    self.test_env = FrameStackObservation(test_env, 4)

    self.behavior_net = AtariNetDuelingDQN(self.env.action_space.n)
    self.behavior_net.to(self.device)
    self.target_net = AtariNetDuelingDQN(self.env.action_space.n)
    self.target_net.to(self.device)
    self.target_net.load_state_dict(self.behavior_net.state_dict())
    self.lr = config["learning_rate"]
    self.optim = torch.optim.Adam(self.behavior_net.parameters(), lr=self.lr, eps=1.5e-4)

  def _obs_to_tensor(self, observation):
    obs_array = np.array(observation)
    return torch.tensor(obs_array, dtype=torch.float32, device=self.device).unsqueeze(0)

  def decide_agent_actions(self, observation, epsilon=0.0, action_space=None):
    if random.random() < epsilon:
      action = action_space.sample()
    else:
      with torch.no_grad():
        obs_tensor = self._obs_to_tensor(observation)
        q_values = self.behavior_net(obs_tensor)
        action = q_values.argmax().item()
    return action
  
  def update_behavior_network(self):
    state, action, reward, next_state, done = self.replay_buffer.sample(self.batch_size, self.device)
    
    q_values = self.behavior_net(state)
    q_value = q_values.gather(1, action.long())
    
    with torch.no_grad():
      # --- ✅ DDQN 修改處 Start ---
      
      # 1. 使用 Behavior Network "選擇" 最佳的下一步動作 (取得動作的 index)
      # .max(1) 回傳 (values, indices)，我們取 [1] 也就是 indices
      best_next_actions = self.behavior_net(next_state).max(1)[1].unsqueeze(1)
      
      # 2. 使用 Target Network "評估" 這些被選中動作的 Q 值
      # 我們用 gather() 來根據 best_next_actions 的 index，從 target_net 的輸出中取出對應的 Q 值
      q_next = self.target_net(next_state).gather(1, best_next_actions)
      
      # --- DDQN 修改處 End ---
      
      # 3. 計算目標 Q 值 (這行不變)
      q_target = reward + self.gamma * q_next * (1 - done)

    criterion = nn.SmoothL1Loss()
    loss = criterion(q_value, q_target)
    
    self.writer.add_scalar('DQN/Loss', loss.item(), self.total_time_step)
    
    self.optim.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(self.behavior_net.parameters(), 1.0)
    self.optim.step()


class AtariDQNParAgent(DQNParBaseAgent):
  def __init__(self, config):
    super().__init__(config)
    
    # --- ✅ 使用舊版的語法來建立平行環境 ---

    # 1. 定義一個輔助函式 (helper function)，用來建立單一的、包裹好的環境
    #    這個函式會被重複呼叫以建立多個獨立的環境
    def make_env():
        env = gym.make(config["env_id"], frameskip=1)
        env = AtariPreprocessing(env, screen_size=84, grayscale_obs=True)
        env = FrameStackObservation(env, 4)
        return env

    # 2. 建立一個包含多個「環境建立函式」的列表
    env_fns = [make_env for _ in range(config["num_envs"])]

    # 3. 使用 SyncVectorEnv 來將它們打包成一個平行的向量化環境
   
    self.env = SyncVectorEnv(env_fns)

    # 測試環境的建立方式維持不變，因為它只需要單一環境
    test_env = gym.make(config["env_id"], frameskip=1)
    test_env = AtariPreprocessing(test_env, screen_size=84, grayscale_obs=True)
    self.test_env = FrameStackObservation(test_env, 4)

    # 網路初始化不變，SyncVectorEnv 同樣有 .single_action_space 屬性
    self.behavior_net = AtariNetDQN(self.env.single_action_space.n)
    self.behavior_net.to(self.device)
    self.target_net = AtariNetDQN(self.env.single_action_space.n)
    self.target_net.to(self.device)
    self.target_net.load_state_dict(self.behavior_net.state_dict())
    self.lr = config["learning_rate"]
    self.optim = torch.optim.Adam(self.behavior_net.parameters(), lr=self.lr, eps=1.5e-4)

  def _obs_to_tensor(self, observation):
    obs_array = np.array(observation)
    return torch.tensor(obs_array, dtype=torch.float32, device=self.device)

  def decide_agent_actions(self, observation, epsilon=0.0):
    if random.random() < epsilon:
      return self.env.action_space.sample()
    else:
      with torch.no_grad():
        obs_tensor = self._obs_to_tensor(observation)
        q_values = self.behavior_net(obs_tensor)
        return q_values.argmax(dim=1).cpu().numpy()
  
  def update_behavior_network(self):
    state, action, reward, next_state, done = self.replay_buffer.sample(self.batch_size, self.device)
    q_values = self.behavior_net(state)
    q_value = q_values.gather(1, action.long())
    with torch.no_grad():
      q_next_values = self.target_net(next_state)
      q_next = q_next_values.max(1)[0].unsqueeze(1)
      q_target = reward + self.gamma * q_next * (1 - done)
    criterion = nn.SmoothL1Loss()
    loss = criterion(q_value, q_target)
    self.writer.add_scalar('DQN/Loss', loss.item(), self.total_time_step)
    self.optim.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(self.behavior_net.parameters(), 1.0)
    self.optim.step()