import torch
import torch.nn as nn 
from resnet import ResNet
from utils import mask

class CouplingLayer(nn.Module):
    def __init__(self, in_channels, mid_channels, num_blocks, reverse_mask):
        super(CouplingLayer, self).__init__()
        self.reverse_mask =reverse_mask
        self.st_net = ResNet(in_channels, mid_channels, 2, num_blocks=num_blocks,
                             kernel_size=(3,1), padding=(1,0), stride=(1,1), double_after_norm=False)
        
    def forward(self, x, logdet = 0):
        m = mask(x, reverse=self.reverse_mask)
        
        x_masked = x * m
        st = self.st_net(x_masked)
        s,t = st.chunk(2, dim=1)
        s = torch.tanh(s)
        m_top = m[:, 0:1, :, :]
        
        
        s = s * (1-m_top)
        t = t * (1-m_top)
        exp_s = s.exp()
        x_flow = x[:,0:1,:,:]
        x_flow = x_flow * exp_s + t
        
        #logdet += s.view(s.size(0), -1).sum(dim=-1)
        return x_flow, s

if __name__ == "__main__":
    cl = CouplingLayer(2, 5, 2, False)
    x = torch.randn((1, 2, 4, 4))
    cl(x)
