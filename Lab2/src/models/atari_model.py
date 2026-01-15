import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class AtariNetDQN(nn.Module):
    def __init__(self, num_classes=4, init_weights=True):
        super(AtariNetDQN, self).__init__()
        self.cnn = nn.Sequential(nn.Conv2d(4, 32, kernel_size=8, stride=4),
                                        nn.ReLU(True),
                                        nn.Conv2d(32, 64, kernel_size=4, stride=2),
                                        nn.ReLU(True),
                                        nn.Conv2d(64, 64, kernel_size=3, stride=1),
                                        nn.ReLU(True)
                                        )
        self.classifier = nn.Sequential(nn.Linear(7*7*64, 512),
                                        nn.ReLU(True),
                                        nn.Linear(512, num_classes)
                                        )

        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = x.float() / 255.
        x = self.cnn(x)
        x = torch.flatten(x, start_dim=1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(m.bias, 0.0)

class AtariNetDuelingDQN(nn.Module):
    def __init__(self, num_classes=4, init_weights=True):
        super(AtariNetDuelingDQN, self).__init__()
        
        # --- 共享的 CNN 層 (與原版相同) ---
        self.cnn = nn.Sequential(nn.Conv2d(4, 32, kernel_size=8, stride=4),
                                        nn.ReLU(True),
                                        nn.Conv2d(32, 64, kernel_size=4, stride=2),
                                        nn.ReLU(True),
                                        nn.Conv2d(64, 64, kernel_size=3, stride=1),
                                        nn.ReLU(True)
                                        )
        
        # --- 原本的 self.classifier 被拆分成以下兩個分支 ---

        # 1. 狀態價值 (Value) 分支
        self.value_stream = nn.Sequential(
            nn.Linear(7*7*64, 512),
            nn.ReLU(True),
            nn.Linear(512, 1)  # 最終輸出一維，代表 V(s)
        )

        # 2. 動作優勢 (Advantage) 分支
        self.advantage_stream = nn.Sequential(
            nn.Linear(7*7*64, 512),
            nn.ReLU(True),
            nn.Linear(512, num_classes) # 最終輸出維度為動作數量，代表 A(s, a)
        )

        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = x.float() / 255.
        
        # --- 計算共享特徵 (與原版相同) ---
        features = self.cnn(x)
        features = torch.flatten(features, start_dim=1)
        
        # --- 分別通過兩個分支 ---
        value = self.value_stream(features)
        advantage = self.advantage_stream(features)
        
        # --- 組合 V 和 A 來計算最終的 Q 值 ---
        # 公式: Q(s,a) = V(s) + (A(s,a) - mean(A(s,a')))
        # 減去 Advantage 的平均值是為了增加訓練的穩定性
        q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))
        
        return q_values

    def _initialize_weights(self):
        # --- 權重初始化 (與原版幾乎相同) ---
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            # 移除了 BatchNorm2d 的部分，因為這個架構中沒有使用
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(m.bias, 0.0)