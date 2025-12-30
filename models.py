import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(SimpleCNN, self).__init__()
        # 卷积层
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.bn1 = nn.BatchNorm2d(32)  # 添加 BN，参数是通道数
        
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.bn2 = nn.BatchNorm2d(64)
        
        # 全连接层
        self.fc1 = nn.Linear(9216, 128)
        self.bn3 = nn.BatchNorm1d(128) # 注意全连接层用 BatchNorm1d
        
        self.fc2 = nn.Linear(128, 64)
        self.bn4 = nn.BatchNorm1d(64)
        
        self.fc3 = nn.Linear(64, num_classes)
        
        # 如果加了BN，通常可以减小Dropout甚至去掉，这里保留但需注意顺序
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)

    def forward(self, x, return_features=False):
        # Block 1
        x = self.conv1(x)
        x = self.bn1(x)       # Conv -> BN -> ReLU
        x = F.relu(x)
        
        # Block 2
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)  # Dropout 通常放在最后
        
        x = torch.flatten(x, 1)
        
        # FC Block 1
        x = self.fc1(x)
        x = self.bn3(x)       # Linear -> BN -> ReLU
        x = F.relu(x)
        x = self.dropout2(x)
        
        # FC Block 2
        features = self.fc2(x)
        # 注意：最后一个特征层是否加BN取决于用途。
        # 如果用于做Graph特征，通常不加激活或BN，保持线性特性；
        # 但原代码加了ReLU，这里保持一致也可以加BN。
        features = self.bn4(features) 
        features_return = F.normalize(features, p=2, dim=1)  # L2归一化
        x = F.relu(features)
        
        output = self.fc3(x)
        
        if return_features:
            return output, features_return
        return output
