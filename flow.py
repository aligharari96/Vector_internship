import torch
import torch.nn as nn 
import torch.nn.functional as F
from coupling_layer import CouplingLayer
from utils import prepare_future

class RealNVP(nn.Module):
    def __init__(self, num_coupling, in_channels, mid_channels, num_blocks):
        super(RealNVP, self).__init__()
        self.couplings = nn.ModuleList([
            CouplingLayer(in_channels=in_channels, mid_channels=mid_channels, num_blocks=num_blocks, reverse_mask=i%2) for i in range(num_coupling)
        ])
        self.future = None

        
    def forward(self, x):
        logdet_tot = 0.0
        for coupling in self.couplings:
            x_prev = torch.roll(x, 1, dims=3)
            x_prev[:, :, :, 0] = 0.0
            
            x = torch.cat([x, x_prev, self.future], dim=1)
            
            x,lodget  = coupling(x)
            logdet_tot += lodget
        return x, logdet_tot
            

    
if __name__ == "__main__":
    from data import PolyphonicDataset
    train_set = PolyphonicDataset('/Users/alihosseinghararifoomani/Desktop/data/polyphonic/test.pkl')
    train_loader = torch.utils.data.DataLoader(train_set,batch_size = 1)
    rnvp = RealNVP(2, 162, 2, 8)
    for seq, rev_seq, seq_len in train_loader:
        rnvp.future = prepare_future(seq, seq_len)
        rnvp(torch.randn(seq.size(0), 1, 88, 160))
        
