import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

class AtariNet(nn.Module):
    def __init__(self, num_classes=4, init_weights=True):
        super(AtariNet, self).__init__()

        self.cnn = nn.Sequential(nn.Conv2d(4, 32, kernel_size=8, stride=4),
                                        nn.ReLU(True),
                                        nn.Conv2d(32, 64, kernel_size=4, stride=2),
                                        nn.ReLU(True),
                                        nn.Conv2d(64, 64, kernel_size=3, stride=1),
                                        nn.ReLU(True)
                                        )
        self.action_logits = nn.Sequential(nn.Linear(7*7*64, 512),
                                        nn.ReLU(True),
                                        nn.Linear(512, num_classes)
                                        )
        self.value = nn.Sequential(nn.Linear(7*7*64, 512),
                                        nn.ReLU(True),
                                        nn.Linear(512, 1)
                                        )

        if init_weights:
            self._initialize_weights()

    def forward(self, x, eval=False, a=[]):
        x = x.float() / 255.
        x = self.cnn(x)
        x = torch.flatten(x, start_dim=1)
        value = self.value(x)
        value = torch.squeeze(value)

        logits = self.action_logits(x)
        
        dist = Categorical(logits=logits)
        
        ### TODO ###
        # Finish the forward function
        # Return action, action probability, value, entropy

        entropy = dist.entropy().mean()

        if eval:
            action = torch.argmax(logits, dim=-1) # 評估時取 greedy action
        elif len(a) > 0: #計算 Policy Loss 時使用 buffer 中的 action
            action = a # 更新時使用 buffer 中的 action
        else:
            action = dist.sample() # 訓練時random採樣 action

        logp = dist.log_prob(action)

        return action, logp, value, entropy
        ### END TODO ###


    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.orthogonal_(m.weight, np.sqrt(2))
                nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, np.sqrt(2))
                nn.init.constant_(m.bias, 0.0)