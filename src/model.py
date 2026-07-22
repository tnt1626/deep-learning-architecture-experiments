import yaml
import torch.nn as nn

from config import PARAMS

with open(PARAMS, "r") as f:
    params = yaml.safe_load(f)

class PinteeCNN(nn.Module):
    def __init__(self, n_classes=10):
        super(PinteeCNN, self).__init__()
        # Improvement Activation
        if params['improvements']['use_advanced_activation']:
            self.activation = nn.SiLU()
        elif params['improvements']['use_relu']:
            self.activation = nn.ReLU()
        else:
            self.activation = nn.Sigmoid()

        self.use_skip_connection = params['improvements']['use_skip_connection']

        # Block 1: (batch_size, 3, 32, 32) -> (batch_size, 64, 16, 16)
        self.conv_layer1 = nn.Sequential(
            nn.Conv2d(3, 64, 3, stride=1, padding='same'), self.activation,
            self._batch_norm(64)
        )
        self.conv_layer2 = nn.Sequential(
            nn.Conv2d(64, 64, 3, stride=1, padding='same'),
            self._batch_norm(64)
        )
        self.conv_layer3 = nn.Sequential(
            nn.Conv2d(64, 64, 3, stride=1, padding='same'), self.activation, 
            self._batch_norm(64),
            nn.MaxPool2d(2)
        )
        # Block 2: (batch_size, 64, 16, 16) -> (batch_size, 128, 8, 8)
        self.conv_layer4 = nn.Sequential(
            nn.Conv2d(64, 128, 3, stride=1, padding='same'), self.activation,
            self._batch_norm(128)
        )
        self.conv_layer5 = nn.Sequential(
            nn.Conv2d(128, 128, 3, stride=1, padding='same'),
            self._batch_norm(128)
        )
        self.conv_layer6 = nn.Sequential(
            nn.Conv2d(128, 128, 3, stride=1, padding='same'), self.activation, 
            self._batch_norm(128),
            nn.MaxPool2d(2)
        )
        # Block 3: (batch_size, 128, 8, 8) -> (batch_size, 256, 4, 4)
        self.conv_layer7 = nn.Sequential(
            nn.Conv2d(128, 256, 3, stride=1, padding='same'), self.activation,
            self._batch_norm(256)
        )
        self.conv_layer8 = nn.Sequential(
            nn.Conv2d(256, 256, 3, stride=1, padding='same'),
            self._batch_norm(256)
        )
        self.conv_layer9 = nn.Sequential(
            nn.Conv2d(256, 256, 3, stride=1, padding='same'), self.activation, 
            self._batch_norm(256),
            nn.MaxPool2d(2)
        )
        # Block 3.5: (batch_size, 256, 4, 4) -> (batch_size, 512, 2, 2)
        self.conv_layer10 = nn.Sequential(
            nn.Conv2d(256, 512, 3, stride=1, padding='same'), self.activation,
            self._batch_norm(512)
        )
        self.conv_layer11 = nn.Sequential(
            nn.Conv2d(512, 512, 3, stride=1, padding='same'),
            self._batch_norm(512)
        )
        self.conv_layer12 = nn.Sequential(
            nn.Conv2d(512, 512, 3, stride=1, padding='same'), self.activation, 
            self._batch_norm(512),
            nn.MaxPool2d(2)
        )
        # Block 4: (batch_size, 512, 2, 2) -> (batch_size, 512 * 2 * 2) -> (batch_size, 512) -> (batch_size, 10)
        self.flatten = nn.Flatten()
        self.fc_layer1 = nn.Sequential(nn.Linear(512*2*2, 512), self.activation)
        self.fc_layer2 = nn.Sequential(nn.Linear(512, n_classes))

        self._initialize_weights()

    def _batch_norm(self, n_features):
        self.use_batch_norm = params['improvements']['use_batch_norm']
        if self.use_batch_norm:
            return nn.BatchNorm2d(n_features)
        return nn.Identity()

    def _initialize_weights(self):
        use_he = params['improvements']['use_he_initialization']
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                if use_he:
                    nn.init.kaiming_normal_(m.weight, mode='fan_out')
                else:
                    nn.init.xavier_normal_(m.weight)

    def forward(self, x):
        # Block 1
        x = self.conv_layer1(x)
        identity = x
        x = self.conv_layer2(x)
        if self.use_skip_connection:
            x = x + identity
        x = self.activation(x)
        x = self.conv_layer3(x)

        # Block 2
        x = self.conv_layer4(x)
        identity = x
        x = self.conv_layer5(x)
        if self.use_skip_connection:
            x = x + identity
        x = self.activation(x)
        x = self.conv_layer6(x)

        # Block 3
        x = self.conv_layer7(x)
        identity = x
        x = self.conv_layer8(x)
        if self.use_skip_connection:
            x = x + identity
        x = self.activation(x)
        x = self.conv_layer9(x)

        # Block 4
        x = self.conv_layer10(x)
        identity = x
        x = self.conv_layer11(x)
        if self.use_skip_connection:
            x = x + identity
        x = self.activation(x)
        x = self.conv_layer12(x)

        x = self.flatten(x)
        x = self.fc_layer1(x)
        x = self.fc_layer2(x)   

        return x
