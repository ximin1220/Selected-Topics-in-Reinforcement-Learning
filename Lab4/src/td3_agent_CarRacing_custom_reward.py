import torch
import torch.nn as nn
import numpy as np
from base_agent_custom_reward import TD3BaseAgent
from models.CarRacing_model import ActorNetSimple, CriticNetSimple
from environment_wrapper.CarRacingEnv import CarRacingEnvironment
import random
from base_agent import OUNoiseGenerator, GaussianNoise

class CarRacingTD3Agent(TD3BaseAgent):
	def __init__(self, config):
		super(CarRacingTD3Agent, self).__init__(config)
		# initialize environment
		self.env = CarRacingEnvironment(N_frame=4, test=False)
		self.test_env = CarRacingEnvironment(N_frame=4, test=True)
		
		# behavior network
		self.actor_net = ActorNetSimple(self.env.observation_space.shape[0], self.env.action_space.shape[0], 4)
		self.critic_net1 = CriticNetSimple(self.env.observation_space.shape[0], self.env.action_space.shape[0], 4)
		self.critic_net2 = CriticNetSimple(self.env.observation_space.shape[0], self.env.action_space.shape[0], 4)
		self.actor_net.to(self.device)
		self.critic_net1.to(self.device)
		self.critic_net2.to(self.device)
		# target network
		self.target_actor_net = ActorNetSimple(self.env.observation_space.shape[0], self.env.action_space.shape[0], 4)
		self.target_critic_net1 = CriticNetSimple(self.env.observation_space.shape[0], self.env.action_space.shape[0], 4)
		self.target_critic_net2 = CriticNetSimple(self.env.observation_space.shape[0], self.env.action_space.shape[0], 4)
		self.target_actor_net.to(self.device)
		self.target_critic_net1.to(self.device)
		self.target_critic_net2.to(self.device)
		self.target_actor_net.load_state_dict(self.actor_net.state_dict())
		self.target_critic_net1.load_state_dict(self.critic_net1.state_dict())
		self.target_critic_net2.load_state_dict(self.critic_net2.state_dict())
		
		# set optimizer
		self.lra = config["lra"]
		self.lrc = config["lrc"]
		
		self.actor_opt = torch.optim.Adam(self.actor_net.parameters(), lr=self.lra)
		self.critic_opt1 = torch.optim.Adam(self.critic_net1.parameters(), lr=self.lrc)
		self.critic_opt2 = torch.optim.Adam(self.critic_net2.parameters(), lr=self.lrc)

		# choose Gaussian noise or OU noise

		# noise_mean = np.full(self.env.action_space.shape[0], 0.0, np.float32)
		# noise_std = np.full(self.env.action_space.shape[0], 1.0, np.float32)
		# self.noise = OUNoiseGenerator(noise_mean, noise_std)

		self.noise = GaussianNoise(self.env.action_space.shape[0], 0.0, 1.0)
		
	
	def decide_agent_actions(self, state, sigma=0.0, brake_rate=0.015):
		with torch.no_grad():
			# 1. 將 state (numpy array) 轉換為 torch tensor, 增加 batch 維度, 並移至 device
			state_tensor = torch.tensor(state, dtype=torch.float).unsqueeze(0).to(self.device)
			
			# 2. 從 actor_net 獲取動作 (只傳入 state 和 brake_rate)
			action = self.actor_net(state_tensor, brake_rate=brake_rate).cpu().numpy()[0]
		
		# 3. 加上探索噪音 (只在這裡加一次)
		noise = self.noise.generate() * sigma
		action = action + noise
		
		# 4. (重要) 裁剪動作到有效範圍
		action[0] = np.clip(action[0], -1.0, 1.0) # steer
		action[1] = np.clip(action[1], 0.0, 1.0)  # gas
		action[2] = np.clip(action[2], 0.0, 1.0)  # brake
		
		return action

	def update_behavior_network(self):
		# sample a minibatch of transitions
		state, action, reward, next_state, done = self.replay_buffer.sample(self.batch_size, self.device)

		## Update Critic (Clipped Double Q-Learning) ##
		with torch.no_grad():
			# 3. Target Policy Smoothing Regularization (目標策略平滑正則化)
			# 從 target_actor_net 選擇下一個動作 a_next
			a_next = self.target_actor_net(next_state)
			
			# (重要) 產生正確維度 [32, 3] 且被裁剪的噪音
			# (你可以使用我們之前討論的「非對稱噪音」版本，或這個標準版本)
			policy_noise = (torch.randn_like(action) * 0.2).clamp_(-0.5, 0.5).to(self.device)
			a_next = a_next + policy_noise
			
			# 將加了噪音的動作裁剪到環境的有效範圍
			a_next_clipped = torch.clone(a_next)
			a_next_clipped[:, 0] = a_next_clipped[:, 0].clamp_(-1.0, 1.0) # steer
			a_next_clipped[:, 1] = a_next_clipped[:, 1].clamp_(0.0, 1.0)  # gas
			a_next_clipped[:, 2] = a_next_clipped[:, 2].clamp_(0.0, 1.0)  # brake

			# 1. Clipped Double Q-Learning (裁剪雙Q學習)
			q_next1 = self.target_critic_net1(next_state, a_next_clipped)
			q_next2 = self.target_critic_net2(next_state, a_next_clipped)
			
			q_next = torch.min(q_next1, q_next2)
			q_target = reward + (1 - done) * self.gamma * q_next
		
		# 獲取當前的 Q values
		q_value1 = self.critic_net1(state, action)
		q_value2 = self.critic_net2(state, action)
		
		# critic loss function
		criterion = nn.MSELoss()
		critic_loss1 = criterion(q_value1, q_target)
		critic_loss2 = criterion(q_value2, q_target)

		# 優化 critic
		self.critic_opt1.zero_grad()
		critic_loss1.backward()
		self.critic_opt1.step()

		self.critic_opt2.zero_grad()
		critic_loss2.backward()
		self.critic_opt2.step()

		## 2. Delayed Actor(Policy) Updates (延遲策略更新) ##
		if self.total_time_step % self.update_freq == 0:
			## update actor ##
			action_pred = self.actor_net(state)
			actor_loss = -self.critic_net1(state, action_pred).mean()
			
			# 優化 actor
			self.actor_opt.zero_grad()
			actor_loss.backward()
			self.actor_opt.step()