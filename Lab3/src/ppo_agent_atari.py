import torch
import torch.nn as nn
import numpy as np
import os
import time
from collections import deque
from torch.utils.tensorboard import SummaryWriter
from replay_buffer.gae_replay_buffer import GaeSampleMemory
from base_agent import PPOBaseAgent
from models.atari_model import AtariNet
import gymnasium as gym
import ale_py
gym.register_envs(ale_py)
from gymnasium.wrappers import AtariPreprocessing
from gymnasium.wrappers import FrameStackObservation


class AtariPPOAgent(PPOBaseAgent):
	def __init__(self, config):
		super(AtariPPOAgent, self).__init__(config)
		
		### TODO ###
		# initialize env
		# 我們需要 wrappers 來處理 Atari 環境：
		# 1. gym.make() 創建基本環境
		# 2. AtariPreprocessing 包含：灰階、縮放至 84x84、FrameSkip(4)
		# 3. FrameStack 將 4 幀畫面堆疊起來 (channel dimension)
		env = gym.make(config["env_id"], render_mode="rgb_array")
		env = AtariPreprocessing(env, screen_size=84, grayscale_obs=True, frame_skip=1)
		self.env = FrameStackObservation(env, 4)
		
		### TODO###
		# initialize test_env
		test_env = gym.make(config["env_id"], render_mode='rgb_array')
		test_env = AtariPreprocessing(test_env, screen_size=84, grayscale_obs=True, frame_skip=1)
		self.test_env = FrameStackObservation(test_env, 4)
		### END TODO ###

		# 現在 self.env.observation_space.shape 會是 (4, 84, 84)，符合模型輸入
		self.net = AtariNet(self.env.action_space.n)
		self.net.to(self.device)
		self.lr = config["learning_rate"]
		self.update_count = config["update_ppo_epoch"]
		self.optim = torch.optim.Adam(self.net.parameters(), lr=self.lr)
		
	def decide_agent_actions(self, observation, eval=False):
		### TODO ###
		# add batch dimension in observation
		# (4, 84, 84) -> (1, 4, 84, 84)
		# 由於 observation 可能是 lazy frame，使用 np.array() 確保轉換
		observation_tensor = torch.from_numpy(np.array(observation)).unsqueeze(0).to(self.device, dtype=torch.float32)
		
		# get action, value, logp from net
		with torch.no_grad():
			action, logp, value, _ = self.net(observation_tensor, eval=eval)
		
		return action, value, logp # 回傳 Tensors
		### END TODO ###

	
	def update(self):
		loss_counter = 0.0001
		total_surrogate_loss = 0
		total_v_loss = 0
		total_entropy = 0
		total_loss = 0
		frac = 1.0 - (self.total_time_step / self.training_steps)
		if frac < 0:
			frac = 0
		new_lr = self.lr * frac # self.lr 是 config 中的 2.5e-4
		self.optim.param_groups[0]['lr'] = new_lr

		batches = self.gae_replay_buffer.extract_batch(self.discount_factor_gamma, self.discount_factor_lambda)
		sample_count = len(batches["action"])
		batch_index = np.random.permutation(sample_count)
		
		observation_batch = {}
		for key in batches["observation"]:
			observation_batch[key] = batches["observation"][key][batch_index]
		action_batch = batches["action"][batch_index]
		return_batch = batches["return"][batch_index]
		adv_batch = batches["adv"][batch_index]
		v_batch = batches["value"][batch_index]
		logp_pi_batch = batches["logp_pi"][batch_index]

		for _ in range(self.update_count):
			for start in range(0, sample_count, self.batch_size):
				ob_train_batch = {}
				for key in observation_batch:
					ob_train_batch[key] = observation_batch[key][start:start + self.batch_size]
				ac_train_batch = action_batch[start:start + self.batch_size]
				return_train_batch = return_batch[start:start + self.batch_size]
				adv_train_batch = adv_batch[start:start + self.batch_size]
				v_train_batch = v_batch[start:start + self.batch_size]
				logp_pi_train_batch = logp_pi_batch[start:start + self.batch_size]

				ob_train_batch = torch.from_numpy(ob_train_batch["observation_2d"])
				ob_train_batch = ob_train_batch.to(self.device, dtype=torch.float32)
				ac_train_batch = torch.from_numpy(ac_train_batch)
				ac_train_batch = ac_train_batch.to(self.device, dtype=torch.long)
				adv_train_batch = torch.from_numpy(adv_train_batch)
				adv_train_batch = adv_train_batch.to(self.device, dtype=torch.float32)
				logp_pi_train_batch = torch.from_numpy(logp_pi_train_batch)
				logp_pi_train_batch = logp_pi_train_batch.to(self.device, dtype=torch.float32)
				return_train_batch = torch.from_numpy(return_train_batch)
				return_train_batch = return_train_batch.to(self.device, dtype=torch.float32)

				### TODO ###
				# calculate loss and update network
				# 傳入 buffer 中的 state 和 action，取得新的 logp, value, entropy
				_, new_logp, new_value, entropy_tensor = self.net(ob_train_batch, a=ac_train_batch)
				entropy = entropy_tensor.mean()

				# Policy Loss (L_CLIP)
				# L_CLIP(θ)=E[min(rt*​At​,clip(rt​,1−ϵ,1+ϵ)At​)]
				ratio = torch.exp(new_logp - logp_pi_train_batch) #r(t) = π_new / π_old
				surr1 = ratio * adv_train_batch
				surr2 = torch.clamp(ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon) * adv_train_batch
				surrogate_loss = -torch.min(surr1, surr2).mean()

				# calculate value loss
				# LV​(ϕ)=E[(Vϕ​(st​)−Rt​)^2]
				value_criterion = nn.MSELoss()
				v_loss = value_criterion(new_value, return_train_batch)
				
				# calculate total loss
				# L_total​=−LCLIP+cv​L_V​−ce​L_H
				loss = surrogate_loss + self.value_coefficient * v_loss - self.entropy_coefficient * entropy

				# update network
				self.optim.zero_grad()
				loss.backward()
				nn.utils.clip_grad_norm_(self.net.parameters(), self.max_gradient_norm)
				self.optim.step()

				total_surrogate_loss += surrogate_loss.item()
				total_v_loss += v_loss.item()
				total_entropy += entropy.item()
				total_loss += loss.item()
				loss_counter += 1
				### END TODO ###

		self.writer.add_scalar('PPO/Loss', total_loss / loss_counter, self.total_time_step)
		self.writer.add_scalar('PPO/Surrogate Loss', total_surrogate_loss / loss_counter, self.total_time_step)
		self.writer.add_scalar('PPO/Value Loss', total_v_loss / loss_counter, self.total_time_step)
		self.writer.add_scalar('PPO/Entropy', total_entropy / loss_counter, self.total_time_step)
		print(f"Loss: {total_loss / loss_counter}\
			\tSurrogate Loss: {total_surrogate_loss / loss_counter}\
			\tValue Loss: {total_v_loss / loss_counter}\
			\tEntropy: {total_entropy / loss_counter}\
			")