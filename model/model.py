import torch.nn as nn
import torch.nn.functional as F
from base import BaseModel
import segmentation_models_pytorch as smp


class MnistModel(BaseModel):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, num_classes)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


class Unet_resnet34(BaseModel):
    def __init__(self, nclass=5):
        super().__init__()
        self.model = smp.Unet('resnet34', classes=nclass, activation=None)
        # model = smp.Unet('resnet34', classes=3, activation='softmax')

    def forward(self, x):
        return self.model(x)


class Unet_resnet50(BaseModel):
    def __init__(self, nclass=5):
        super().__init__()
        self.model = smp.Unet('resnet50', classes=nclass, activation=None)
        # model = smp.Unet('resnet34', classes=3, activation='softmax')

    def forward(self, x):
        return self.model(x)


