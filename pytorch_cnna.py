import torch
import torch.nn as nn


class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=(1,17)),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.model = nn.Sequential(
            BasicBlock(in_channels, out_channels),
        )

    def forward(self, x):
        x = self.model(x)
        return x


class ISMER_Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            ConvBlock(1, 16),
            ConvBlock(16, 32),
            nn.AvgPool2d(17),
            ConvBlock(32, 64),
            ConvBlock(64, 64),
            nn.Flatten(1,256),
            nn.Linear(256,17),
        )
    def forward(self,x):
        x=self.model(x)
        return x

if __name__ == "__main__":
    print('??????????????')
    model = ISMER_Model()
    print(model.eval())
    total_params=sum(p.numel()for p in model.parameters())
    print(total_params)
    data=torch.randn(17,87,1)
    print(data)
    a=model(data)
    print(a)
