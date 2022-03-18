import torch
import torch.nn as nn

__all__ = ['Vox_RNet', 'vox_rnet']


class Vox_RNet(nn.Module):
    def __init__(self, num_classes=10):
        super(Vox_RNet, self).__init__()
        self.down1 = nn.Sequential(
            nn.Conv3d(2, 8, kernel_size=5, stride=(2, 2, 2), padding=(2, 2, 2)),
            nn.BatchNorm3d(8),
            nn.ReLU(inplace=True)
        )
        self.down2 = nn.Sequential(
            nn.Conv3d(8, 16, kernel_size=3, stride=(2, 2, 2), padding=(1, 1, 1)),
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True)
        )

        self.down3 = nn.Sequential(
            nn.Conv3d(16, 32, kernel_size=3, stride=(2, 2, 2), padding=(1, 1, 1)),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True)
        )

        self.fc1 = nn.Sequential(
            nn.Linear(23328, 1024),
            nn.ReLU(inplace=True)
        )
        self.out = nn.Linear(1024,10*3)

    def forward(self, x):
        x1 = self.down1(x)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x3 = x3.view(-1, x3.shape[1]*x3.shape[2]*x3.shape[3]*x3.shape[4])
        x4 = self.fc1(x3)
        x = self.out(x4)
        return x


def vox_rnet(**kwargs):
    model = Vox_RNet(num_classes=kwargs['num_classes'])
    return model
