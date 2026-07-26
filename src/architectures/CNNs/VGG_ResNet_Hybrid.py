import torch
import torch.nn as nn

class PinteeCNN(nn.Module):
    def __init__(self, n_classes: int=10, use_batch_norm: bool=False):
        super(PinteeCNN, self).__init__()

        self.use_batch_norm = use_batch_norm
        
        # Block 1: (batch_size, 3, 32, 32) -> (batch_size, 64, 16, 16)
        self.conv_layer1 = nn.Sequential(
            nn.Conv2d(3, 64, 3, stride=1, padding='same'), 
            nn.ReLU()
        )
        self.conv_layer2 = nn.Sequential(
            nn.Conv2d(64, 64, 3, stride=1, padding='same')
        )
        self.conv_layer3 = nn.Sequential(
            nn.Conv2d(64, 64, 3, stride=1, padding='same'), 
            nn.ReLU(), 
            nn.MaxPool2d(2)
        )
        
        # Block 2: (batch_size, 64, 16, 16) -> (batch_size, 128, 8, 8)
        self.conv_layer4 = nn.Sequential(
            nn.Conv2d(64, 128, 3, stride=1, padding='same'), 
            nn.ReLU()
        )
        self.conv_layer5 = nn.Sequential(
            nn.Conv2d(128, 128, 3, stride=1, padding='same')
        )
        self.conv_layer6 = nn.Sequential(
            nn.Conv2d(128, 128, 3, stride=1, padding='same'), 
            nn.ReLU(), 
            nn.MaxPool2d(2)
        )
        
        # Block 3: (batch_size, 128, 8, 8) -> (batch_size, 256, 4, 4)
        self.conv_layer7 = nn.Sequential(
            nn.Conv2d(128, 256, 3, stride=1, padding='same'), 
            nn.ReLU()
        )
        self.conv_layer8 = nn.Sequential(
            nn.Conv2d(256, 256, 3, stride=1, padding='same')
        )
        self.conv_layer9 = nn.Sequential(
            nn.Conv2d(256, 256, 3, stride=1, padding='same'), 
            nn.ReLU(), 
            nn.MaxPool2d(2)
        )
        
        # Block 4: (batch_size, 256, 4, 4) -> (batch_size, 512, 2, 2)
        self.conv_layer10 = nn.Sequential(
            nn.Conv2d(256, 512, 3, stride=1, padding='same'), 
            nn.ReLU()
        )
        self.conv_layer11 = nn.Sequential(
            nn.Conv2d(512, 512, 3, stride=1, padding='same')
        )
        self.conv_layer12 = nn.Sequential(
            nn.Conv2d(512, 512, 3, stride=1, padding='same'), 
            nn.ReLU(), 
            nn.MaxPool2d(2)
        )
        
        self.flatten = nn.Flatten()
        self.fc_layer1 = nn.Sequential(
            nn.Linear(512 * 2 * 2, 512), 
            nn.ReLU()
        )
        self.fc_layer2 = nn.Sequential(
            nn.Linear(512, n_classes)
        )

    def _batch_norm(self, n_features):
        if self.use_batch_norm:
            return nn.BatchNorm2d(n_features)
        return nn.Identity()

    def forward(self, x):
        # Block 1
        x = self.conv_layer1(x)
        identity = x
        x = self.conv_layer2(x)
        x = x + identity
        x = torch.relu(x)
        x = self.conv_layer3(x)

        # Block 2
        x = self.conv_layer4(x)
        identity = x
        x = self.conv_layer5(x)
        x = x + identity
        x = torch.relu(x)
        x = self.conv_layer6(x)

        # Block 3
        x = self.conv_layer7(x)
        identity = x
        x = self.conv_layer8(x)
        x = x + identity
        x = torch.relu(x)
        x = self.conv_layer9(x)

        # Block 4
        x = self.conv_layer10(x)
        identity = x
        x = self.conv_layer11(x)
        x = x + identity
        x = torch.relu(x)
        x = self.conv_layer12(x)

        x = self.flatten(x)
        x = self.fc_layer1(x)
        x = self.fc_layer2(x)   

        return x