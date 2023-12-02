import torch.nn as nn 
import torch.nn.functional as F
from utils import WNConv2d

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding, stride, dilation):
        super(ResidualBlock, self).__init__()
        self.in_norm = nn.BatchNorm2d(in_channels)
        self.in_conv = WNConv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride, dilation=dilation ,bias=False)

        self.out_norm = nn.BatchNorm2d(out_channels)
        self.out_conv = WNConv2d(out_channels, in_channels, kernel_size=kernel_size, padding=padding, stride=stride, dilation=dilation, bias=True)
    
    def forward(self, x):
        skip = x

        x = self.in_norm(x)
        x = F.relu(x)
        x = self.in_conv(x)

        x = self.out_norm(x)
        x = F.relu(x)
        x = self.out_conv(x)
        x = x+skip

        return x
    
if __name__ == "__main__":
    import torch
    rb = ResidualBlock(2, 5, kernel_size=(3,1), padding=(1,0), stride=(1,1))
    x = torch.rand((1, 2, 5,5))
    #rb(x)