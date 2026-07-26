import torch.nn as nn

class Lenet(nn.Module):
    def __init__(self, n_classes=10):
        super(Lenet, self).__init__()
        
        # Layer 1: (batch_size, 3, 32, 32) -> (batch_size, 6, 16, 16)
        self.conv_layer1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=6, kernel_size=3, stride=1, padding='same'), 
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        
        # Layer 2: (batch_size, 6, 16, 16) -> (batch_size, 16, 6, 6)
        self.conv_layer2 = nn.Sequential(
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )

        self.flatten = nn.Flatten()
        
        # Fully Connected Layers
        self.fc_layer1 = nn.Sequential(
            nn.Linear(16 * 6 * 6, 120), 
            nn.ReLU()
        )
        
        self.fc_layer2 = nn.Sequential(
            nn.Linear(120, 84), 
            nn.ReLU()
        )
        
        self.fc_layer3 = nn.Sequential(
            nn.Linear(84, n_classes)
        )

    def forward(self, x):
        x = self.conv_layer1(x)
        x = self.conv_layer2(x)
        x = self.flatten(x)
        x = self.fc_layer1(x)
        x = self.fc_layer2(x)   
        x = self.fc_layer3(x)   

        return x