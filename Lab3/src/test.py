import argparse
from ppo_agent_atari import AtariPPOAgent

def evaluate_agent(model_path):
    """
    載入指定的模型並執行評估。
    """
    
    # 這個 config 必須和訓練時的 config 保持一致
    # 尤其是 "env_id" 和 "horizon" 等
    config = {
        "gpu": True,
        "training_steps": 1e8,
        "update_sample_count": 10000,
        "discount_factor_gamma": 0.99,
        "discount_factor_lambda": 0.95,
        "clip_epsilon": 0.2,
        "max_gradient_norm": 0.5,
        "batch_size": 128,
        "logdir": 'log/test/', # Logdir 不太重要，但 env_id 很重要
        "update_ppo_epoch": 3,
        "learning_rate": 2.5e-4, # LR 在評估時不重要
        "value_coefficient": 0.5,
        "entropy_coefficient": 0.01,
        "horizon": 128,
        "env_id": 'ALE/Enduro-v5', # 必須和訓練時相同
        "eval_interval": 100,
        "eval_episode": 5, # 你可以增加評估的 episode 數量
    }

    print("初始化 Agent...")
    agent = AtariPPOAgent(config)
    
    print(f"正在載入模型: {model_path}")
    
    # 呼叫 base_agent.py 中定義好的 load_and_evaluate 函數
    agent.load_and_evaluate(model_path)

if __name__ == '__main__':
    # 設置參數解析器，讓我們可以從命令列傳入模型路徑
    
    
    model_path_default = 'log/Enduro_release/model_44497796_1982.pth'
    
    evaluate_agent(model_path_default)