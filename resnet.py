import torch
import torch.nn as nn
import torch.nn.functional as F
from residual_block import ResidualBlock
from utils import WNConv2d

class ResNet(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels, num_blocks,
                 kernel_size, padding, stride, double_after_norm):
        super(ResNet, self).__init__()
        #self.in_norm = nn.BatchNorm2d(in_channels)
        self.double_after_norm = double_after_norm
        self.in_conv = WNConv2d(2 * in_channels, mid_channels, kernel_size=kernel_size, padding=padding, stride=stride, bias=True)
        self.in_skip = WNConv2d(mid_channels, mid_channels, kernel_size=1, padding=0, stride=1, bias=True)

        self.blocks = nn.ModuleList([ResidualBlock(mid_channels, mid_channels, kernel_size=kernel_size, padding=padding, stride=stride, dilation=(i_+1)*2-1)
                                      for i_ in range(num_blocks)])
        self.skips = nn.ModuleList([WNConv2d(mid_channels, mid_channels, kernel_size=1, padding=0 ,stride=1, bias=True)])

        self.out_norm = nn.BatchNorm2d(mid_channels)
        self.out_conv = WNConv2d(mid_channels, out_channels, kernel_size=1, padding=0, stride=1, bias=True)

    
    def forward(self, x):
        #x = self.in_norm(x)
        #print(torch.max(x), torch.min(x), 'fuckkkkkk')
        #assert 0
        if self.double_after_norm:
            x *= 2.0
        x = torch.cat([x,-x], dim=1)
        x = F.relu(x)
        x = self.in_conv(x)
        x_skip = self.in_skip(x)
        for block, skip in zip(self.blocks, self.skips):
            x = block(x)
            x_skip += skip(x)
        
        x = self.out_norm(x_skip)
        x = F.relu(x)
        x = self.out_conv(x)

        return x
    

if __name__ == "__main__":
    rn = ResNet(1, 10, 5, 8, (3,1), (1,0), (1,1), False)
    x = torch.randn((5, 1, 4,3))
    print(rn(x).shape)

