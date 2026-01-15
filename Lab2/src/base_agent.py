import torch
import torch.nn as nn
import numpy as np
import os
import time
from collections import deque
from torch.utils.tensorboard import SummaryWriter
from replay_buffer.replay_buffer import ReplayMemory
from abc import ABC, abstractmethod


class DQNBaseAgent(ABC):
	def __init__(self, config):
		self.gpu = config["gpu"]
		self.device = torch.device("cuda" if self.gpu and torch.cuda.is_available() else "cpu")
		self.total_time_step = 0
		self.training_steps = int(config["training_steps"])
		self.batch_size = int(config["batch_size"])
		self.epsilon = 1.0
		self.eps_min = config["eps_min"]
		self.eps_decay = config["eps_decay"]
		self.eval_epsilon = config["eval_epsilon"]
		self.warmup_steps = config["warmup_steps"]
		self.eval_interval = config["eval_interval"]
		self.eval_episode = config["eval_episode"]
		self.gamma = config["gamma"]
		self.update_freq = config["update_freq"]
		self.update_target_freq = config["update_target_freq"]
	
		self.replay_buffer = ReplayMemory(int(config["replay_buffer_capacity"]))
		self.writer = SummaryWriter(config["logdir"])

	@abstractmethod
	def decide_agent_actions(self, observation, epsilon=0.0, action_space=None):
		### TODO ###
		# get action from behavior net, with epsilon-greedy selection
		action = None
		return action
	
	def update(self):
		if self.total_time_step % self.update_freq == 0:
			self.update_behavior_network()
		if self.total_time_step % self.update_target_freq == 0:
			self.update_target_network()

	@abstractmethod
	def update_behavior_network(self):
		# sample a minibatch of transitions
		state, action, reward, next_state, done = self.replay_buffer.sample(self.batch_size, self.device)
		### TODO ###
		# calculate the loss and update the behavior network
		

	def update_target_network(self):
		self.target_net.load_state_dict(self.behavior_net.state_dict())
	
	def epsilon_decay(self):
		self.epsilon -= (1 - self.eps_min) / self.eps_decay
		self.epsilon = max(self.epsilon, self.eps_min)

	def train(self):
		episode_idx = 0
		while self.total_time_step <= self.training_steps:
			observation, info = self.env.reset()
			episode_reward = 0
			episode_len = 0
			episode_idx += 1
			while True:
				if self.total_time_step < self.warmup_steps:
					action = self.decide_agent_actions(observation, 1.0, self.env.action_space)
				else:
					action = self.decide_agent_actions(observation, self.epsilon, self.env.action_space)
					self.epsilon_decay()

				next_observation, reward, terminate, truncate, info = self.env.step(action)
				self.replay_buffer.append(observation, [action], [reward], next_observation, [int(terminate)])

				if self.total_time_step >= self.warmup_steps:
					self.update()

				episode_reward += reward
				episode_len += 1
				
				if terminate or truncate:
					self.writer.add_scalar('Train/Episode Reward', episode_reward, self.total_time_step)
					self.writer.add_scalar('Train/Episode Len', episode_len, self.total_time_step)
					print(f"[{self.total_time_step}/{self.training_steps}]  episode: {episode_idx}  episode reward: {episode_reward}  episode len: {episode_len}  epsilon: {self.epsilon}")
					break
					
				observation = next_observation
				self.total_time_step += 1
				
			if episode_idx % self.eval_interval == 0:
				# save model checkpoint
				avg_score = self.evaluate()
				self.save(os.path.join(self.writer.log_dir, f"model_{self.total_time_step}_{int(avg_score)}.pth"))
				self.writer.add_scalar('Evaluate/Episode Reward', avg_score, self.total_time_step)

	def evaluate(self):
		print("==============================================")
		print("Evaluating...")
		all_rewards = []
		for i in range(self.eval_episode):
			observation, info = self.test_env.reset()
			total_reward = 0
			while True:
				# self.test_env.render()
				action = self.decide_agent_actions(observation, self.eval_epsilon, self.test_env.action_space)
				next_observation, reward, terminate, truncate, info = self.test_env.step(action)
				total_reward += reward
				if terminate or truncate:
					print(f"episode {i+1} reward: {total_reward}")
					all_rewards.append(total_reward)
					break

				observation = next_observation
			

		avg = sum(all_rewards) / self.eval_episode
		print(f"average score: {avg}")
		print("==============================================")
		return avg
	
	# save model
	def save(self, save_path):
		torch.save(self.behavior_net.state_dict(), save_path)

	# load model
	def load(self, load_path):
		self.behavior_net.load_state_dict(torch.load(load_path))

	# load model weights and evaluate
	def load_and_evaluate(self, load_path):
		self.load(load_path)
		self.evaluate()


class DQNParBaseAgent(ABC):
    def __init__(self, config):
        self.num_envs = config.get("num_envs", 1) # 預設為 1 以相容舊版
        self.gpu = config["gpu"]
        self.device = torch.device("cuda" if self.gpu and torch.cuda.is_available() else "cpu")
        self.total_time_step = 0
        self.training_steps = int(config["training_steps"])
        self.batch_size = int(config["batch_size"])
        self.epsilon = 1.0
        self.eps_min = config["eps_min"]
        self.eps_decay = config["eps_decay"]
        self.eval_epsilon = config["eval_epsilon"]
        self.warmup_steps = config["warmup_steps"]
        self.eval_interval = config["eval_interval"]
        self.eval_episode = config["eval_episode"]
        self.gamma = config["gamma"]
        self.update_freq = config["update_freq"]
        self.update_target_freq = config["update_target_freq"]
        self.replay_buffer = ReplayMemory(int(config["replay_buffer_capacity"]))
        self.writer = SummaryWriter(config["logdir"])

    # decide_agent_actions 和 update_behavior_network 保持為 abstract
    @abstractmethod
    def decide_agent_actions(self, observation, epsilon=0.0):
        pass
    
    @abstractmethod
    def update_behavior_network(self):
        pass

    def update(self):
        # 更新頻率的邏輯需要調整
        if self.total_time_step // self.num_envs % self.update_freq == 0:
            self.update_behavior_network()
        if self.total_time_step // self.num_envs % self.update_target_freq == 0:
            self.update_target_network()

    def update_target_network(self):
        self.target_net.load_state_dict(self.behavior_net.state_dict())
    
    def epsilon_decay(self):
        # Epsilon 衰減速度不變
        self.epsilon -= (1 - self.eps_min) / self.eps_decay
        self.epsilon = max(self.epsilon, self.eps_min)

    # --- ✅ 重構後的訓練主迴圈 ---
    def train(self):
        # 初始化
        observation, info = self.env.reset()
        episode_rewards = np.zeros(self.num_envs, dtype=np.float32)
        episode_lens = np.zeros(self.num_envs, dtype=np.int32)
        total_episodes_done = 0

        while self.total_time_step < self.training_steps:
            # 決定動作
            if self.total_time_step < self.warmup_steps:
                action = self.env.action_space.sample()
            else:
                action = self.decide_agent_actions(observation, self.epsilon)
                self.epsilon_decay()

            # 與環境互動
            next_observation, reward, terminate, truncate, info = self.env.step(action)
            
            # 將經驗存入 Replay Buffer 需要一個 for 迴圈來遍歷該批次的 N 組結果
            for i in range(self.num_envs):
                # 注意： vectorized env 的 done 是 terminate OR truncate
                done = terminate[i] or truncate[i]
                # 當一個 episode 結束時，info 會包含 "final_observation"
                real_next_obs = info.get("final_observation", [None]*self.num_envs)[i]
                if real_next_obs is None:
                    real_next_obs = next_observation[i] # 如果 episode 未結束，就用正常的 next_obs
                
                self.replay_buffer.append(observation[i], [action[i]], [reward[i]], real_next_obs, [int(done)])

            # 更新當前 episode 的數據
            episode_rewards += reward
            episode_lens += 1
            
            # 總步數增加
            self.total_time_step += self.num_envs

            # 訓練網路
            if self.total_time_step >= self.warmup_steps:
                self.update()

            # 處理結束的 episode
            for i in range(self.num_envs):
                done = terminate[i] or truncate[i]
                if done:
                    total_episodes_done += 1
                    self.writer.add_scalar('Train/Episode Reward', episode_rewards[i], self.total_time_step)
                    self.writer.add_scalar('Train/Episode Len', episode_lens[i], self.total_time_step)
                    print(f"[{self.total_time_step}/{self.training_steps}] episode: {total_episodes_done} reward: {episode_rewards[i]} len: {episode_lens[i]} epsilon: {self.epsilon:.4f}")
                    
                    # 重置該環境的計數器
                    episode_rewards[i] = 0
                    episode_lens[i] = 0

                    # 進行評估
                    if total_episodes_done % self.eval_interval == 0:
                        avg_score = self.evaluate()
                        self.save(os.path.join(self.writer.log_dir, f"model_{self.total_time_step}_{int(avg_score)}.pth"))
                        self.writer.add_scalar('Evaluate/Episode Reward', avg_score, self.total_time_step)
            
            # 更新觀測值
            observation = next_observation

    # evaluate, save, load 函式維持不變
    def evaluate(self):
        print("==============================================")
        print("Evaluating...")
        all_rewards = []
        for i in range(self.eval_episode):
            observation, info = self.test_env.reset()
            total_reward = 0
            while True:
                action = self.decide_agent_actions(np.array([observation]), self.eval_epsilon)[0]
                next_observation, reward, terminate, truncate, info = self.test_env.step(action)
                total_reward += reward
                if terminate or truncate:
                    print(f"episode {i+1} reward: {total_reward}")
                    all_rewards.append(total_reward)
                    break
                observation = next_observation
        avg = sum(all_rewards) / self.eval_episode
        print(f"average score: {avg}")
        print("==============================================")
        return avg
    
    def save(self, save_path):
        torch.save(self.behavior_net.state_dict(), save_path)

    def load(self, load_path):
        self.behavior_net.load_state_dict(torch.load(load_path))

    def load_and_evaluate(self, load_path):
        self.load(load_path)
        self.evaluate()


