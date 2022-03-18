import torch
import torch.nn as nn

__all__ = ['Vox_UNet', 'vox_unet']


class Vox_UNet(nn.Module):
    def __init__(self, num_classes=10):
        super(Vox_UNet, self).__init__()
        self.down1 = nn.Sequential(
            nn.Conv3d(2, 8, 3, padding=1),
            nn.BatchNorm3d(8),
            nn.ReLU(inplace=True),
            nn.Conv3d(8, 8, 3, padding=1),
            nn.BatchNorm3d(8),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=2, stride=2)

        )
        self.down2 = nn.Sequential(
            nn.Conv3d(8, 16, 3, padding=1),
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True),
            nn.Conv3d(16, 16, 3, padding=1),
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=2, stride=2)
        )

        self.down3 = nn.Sequential(
            nn.Conv3d(16, 32, 3, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            nn.Conv3d(32, 32, 3, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
        )

        self.up1 = nn.Sequential(
            #nn.ConvTranspose3d(32, 32, kernel_size=2, stride=2),
            nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True),
            nn.ConvTranspose3d(32, 16, 3, padding=1),
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True),
            nn.ConvTranspose3d(16, 16, 3, padding=1),
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True)
        )

        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True),
            nn.ConvTranspose3d(16, 8, 3, padding=1),
            nn.BatchNorm3d(8),
            nn.ReLU(inplace=True),
            nn.ConvTranspose3d(8, 8, 3, padding=1),
            nn.BatchNorm3d(8),
            nn.ReLU(inplace=True)
        )

        self.out = nn.Sequential(
            nn.Conv3d(8, 10, kernel_size=1)
        )

    def forward(self, x):
        x1 = self.down1(x)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x4 = self.up1(x3)
        #x4 = torch.cat([x1, x3], dim=1)
        x5 = self.up2(x4)
        x = self.out(x5)
        return x


def vox_unet(**kwargs):
    model = Vox_UNet(num_classes=kwargs['num_classes'])
    return model
