import torch
import torch.nn as nn 
import torch.nn.functional as F
from resnet import ResNet


class InitialPosterior(nn.Module):
    def __init__(self, in_channels, mid_channels, num_blocks,
                 kernel_size, padding, stride, double_after_norm):
        super().__init__()
        self.net = ResNet(in_channels=in_channels, mid_channels=mid_channels, out_channels=2, num_blocks=num_blocks, kernel_size=kernel_size, padding=padding, stride=stride, double_after_norm=double_after_norm)
    
    def forward(self, x):
        mu, logvar = self.net(x).chunk(2, dim=1)
        mu = mu.squeeze()
        logvar = logvar.squeeze()
        samples = mu + torch.randn(logvar.shape, device=x.device) * torch.exp(0.5*logvar)
        return samples, mu, logvar

if __name__ == "__main__":
    x = torch.randn((2, 160, 1, 160))
    ip = InitialPosterior(160, 2, 8, (3,1), (1,0), (1,1), False)
    ip(x)