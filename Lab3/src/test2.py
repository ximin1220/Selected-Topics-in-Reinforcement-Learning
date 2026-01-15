import os
import random
import numpy as np
import torch
import time

# 導入您自己的 PPO Agent
from ppo_agent_atari import AtariPPOAgent
import gymnasium as gym
import ale_py
gym.register_envs(ale_py)
from gymnasium.wrappers import AtariPreprocessing
from gymnasium.wrappers import FrameStackObservation
# 導入與您 PPO 訓練時相符的 gym 和 wrappers
# import gym
# from gym.wrappers import AtariPreprocessing, FrameStack

def set_seed(seed):
    """設定所有相關的隨機種子以確保可重現性"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)

# --- 主要執行區 ---
if __name__ == '__main__':

    # 1. 設定一個固定的隨機種子，您可以換成任何數字
    SEED = 87
    set_seed(SEED)
    print(f"Using fixed random seed: {SEED}")

    # --- ‼️請在這裡填入您要評估的模型檔名‼️ ---
    # 例如: "log/Enduro_release_decay/model_80000000_1700.pth"
    MODEL_PATH = "log/Enduro_release/model_44497796_1982.pth" 
    
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"找不到模型檔案: {MODEL_PATH}。請確認路徑是否正確。")

    # 2. 設定 Config (基於您的 main.py)
    # 這裡的大多數參數僅為初始化 Agent 所需
    config = {
        "seed": SEED,
        "gpu": torch.cuda.is_available(),
        "training_steps": 1e8,
        "update_sample_count": 10000,
        "discount_factor_gamma": 0.99,
        "discount_factor_lambda": 0.95,
        "clip_epsilon": 0.2,
        "max_gradient_norm": 0.5,
        "batch_size": 128,
        "logdir": 'log/test2/', # 指定一個日誌目錄
        "update_ppo_epoch": 3,
        "learning_rate": 2.5e-4,
        "value_coefficient": 0.5,
        "entropy_coefficient": 0.01,
        "horizon": 128,
        "env_id": 'ALE/Enduro-v5', # ‼️ 確保这與您訓練的模型一致
        "eval_interval": 100,
        "eval_episode": 1, # ‼️ 在此設定您想跑幾場遊戲
    }

    # 3. 建立 Agent 並載入模型
    print("Initializing PPO Agent...")
    agent = AtariPPOAgent(config)
    
    print(f"Loading model: {MODEL_PATH}...")
    device = torch.device("cuda" if config["gpu"] and torch.cuda.is_available() else "cpu")
    # agent.load() 是 base_agent.py 中的方法
    agent.load(MODEL_PATH)
    print("Model loaded successfully.")

    # 4. 建立一個「確定性」且「可視化」的環境
    
    # 關鍵：
    # render_mode='human' -> 直接在視窗中顯示遊戲畫面
    # frameskip=1 -> 讓 AtariPreprocessing wrapper 來控制 frameskip
    # repeat_action_probability=0.0 -> 關閉「黏滯動作」，確保環境行為固定
    env = gym.make(
        config["env_id"],
        render_mode='human',
        frameskip=1,
        repeat_action_probability=0.0 
    )

    # 套用與訓練時*完全相同*的預處理 Wrapper
    env = AtariPreprocessing(env)
    env = FrameStackObservation(env, 4)

    # 5. 執行多場可重現的遊戲
    
    all_rewards = []
    print(f"Starting {config['eval_episode']} deterministic evaluation games...")
    
    for i in range(config["eval_episode"]):
        # 關鍵：在 reset 時也傳入種子，確保遊戲的起始狀態完全一樣
        # 我們使用 SEED + i 確保每一場遊戲都是固定的，但彼此之間不同
        observation, info = env.reset(seed=SEED + i)
        
        # 舊版 gym (<0.21) 可能需要:
        # env.seed(SEED + i)
        # observation = env.reset()

        total_reward = 0
        done = False
        
        print(f"\n--- Starting Episode {i+1}/{config['eval_episode']} (Seed: {SEED + i}) ---")
        while not done:
            # 從 PPO Agent 獲取確定性動作
            # eval=True 會返回 greedy 動作
            action_tensor, _, _ = agent.decide_agent_actions(observation, eval=True)
            action = action_tensor.cpu().numpy()[0]
            
            observation, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            done = terminated or truncated
            time.sleep(0.01) # 稍微放慢速度，方便觀看

        print(f"Episode {i+1} finished. Score: {total_reward}")
        all_rewards.append(total_reward)

    # 關閉環境
    env.close()

    print("\n==============================================")
    print(f"All deterministic evaluations finished.")
    print(f"Scores: {all_rewards}")
    print(f"Average Score: {np.mean(all_rewards)}")
    print("==============================================")