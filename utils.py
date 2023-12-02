import torch 


class WNConv2d(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding, stride, dilation=1, bias=True):
        super(WNConv2d, self).__init__()
        self.conv = torch.nn.utils.weight_norm(
            torch.nn.Conv2d(in_channels, out_channels, kernel_size = kernel_size, padding=padding,stride=stride, bias=bias, dilation=dilation)
            )
        torch.nn.init.uniform_(self.conv.weight, -0.01,0.01)
    
    def forward(self, x):
        return self.conv(x)
    

def mask(x, reverse=False):
    mask = torch.ones_like(x)
    mask_top = torch.tensor([[(i+j)%2 for i in range(x.size(3))] for j in range(x.size(2))], requires_grad=False)
    if reverse:
        mask_top = 1.0 - mask_top
    mask_top = torch.zeros_like(mask_top)
    mask[:, 0:1, :, :] = mask_top.unsqueeze(0).repeat(x.size(0),1,1).unsqueeze(1)
    
    return mask

def prepare_future(x, seq_len, device):
    future = torch.zeros((x.size(0), x.size(1),x.size(2), x.size(1)))
    x_permuted = torch.permute(x, (0, 2,1)).unsqueeze(1)
    for i in range(x.size(1)):
        tmp = torch.zeros_like(x_permuted)
        tmp[:, :,:,:tmp.size(3)-i] = x_permuted[:,:,:,i:]
        future[:,i:i+1,:,:] = tmp
    for i in range(x.size(0)):
        future[i, seq_len[i]:,:,:] = 0.0
    return future.to(device)

def log_likelihood(z, mu, logvar):
    std = torch.exp(0.5 * logvar)
    #print(logvar[torch.isinf(torch.exp(0.5*logvar))])
    #print(torch.any(torch.isnan(logvar)), torch.any(torch.isinf(logvar)))
    normal = torch.distributions.normal.Normal(mu, std)
    return normal.log_prob(z)


if __name__ == "__main__":
    x = torch.randn((3,4))
    
    mu = torch.zeros((3,4))
    logvar = torch.zeros_like(mu)
    print(log_likelihood(x, mu, logvar).shape)

    

