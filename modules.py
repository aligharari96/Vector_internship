import torch 
import torch.nn as nn 

class GatedTransition(nn.Module):
    def __init__(self, z_dim, trans_dim):
        super(GatedTransition, self).__init__()
        self.gate = nn.Sequential( 
            nn.Linear(z_dim, trans_dim),
            nn.ReLU(),
            nn.Linear(trans_dim, z_dim),
            nn.Sigmoid()
        )
        self.proposed_mean = nn.Sequential(
            nn.Linear(z_dim, trans_dim),
            nn.ReLU(),
            nn.Linear(trans_dim, z_dim)
        )           
        self.z_to_mu = nn.Linear(z_dim, z_dim)
        nn.init.eye_(self.z_to_mu.weight)
        
        #self.z_to_mu.weight.data = torch.eye(z_dim)
        #self.z_to_mu.bias.data = torch.zeros(z_dim)
        self.z_to_logvar = nn.Linear(z_dim, z_dim, bias=False)
        nn.init.normal_(self.z_to_logvar.weight, 0.0, 0.1)
        #self.z_to_logvar.bias.data.fill_(0.01)
        self.relu = nn.ReLU()
    
    def forward(self, z_t_1):#z_t_1 --> [N * z_dim *T_max]
        n_batch, z_dim, T_ = z_t_1.shape
        z_t_1 = z_t_1.permute(0, 2,1).reshape(-1, z_dim)
        gate = self.gate(z_t_1) # compute the gating function
        proposed_mean = self.proposed_mean(z_t_1) # compute the 'proposed mean'
        mu = (1 - gate) * self.z_to_mu(z_t_1) + gate * proposed_mean # compute the scale used to sample z_t, using the proposed mean from
        logvar = self.z_to_logvar(self.relu(proposed_mean))
        epsilon = torch.randn(z_t_1.size(), device=z_t_1.device) # sampling z by re-parameterization
        z_t = mu + epsilon * torch.exp(0.5*logvar)   # [batch_sz x z_sz]
        z_t = z_t.reshape(n_batch, T_, z_dim).permute(0,2,1)
        mu = mu.reshape(n_batch, T_, z_dim).permute(0,2,1)
        logvar = logvar.reshape(n_batch, T_, z_dim).permute(0,2,1)
        return z_t, mu, logvar#.exp()

class Emitter(nn.Module):
    def __init__(self, z_dim, emission_dim, out_dim):
        super(Emitter, self).__init__() 
        self.z_dim = z_dim
        self.emission_dim = emission_dim
        self.out_dim = out_dim
        self.net = nn.Sequential( #Parameterizes the bernoulli observation likelihood `p(x_t|z_t)`
            nn.Linear(self.z_dim, self.emission_dim),
            nn.ReLU(),
            nn.Linear(self.emission_dim, self.emission_dim),
            nn.ReLU(),
            nn.Linear(self.emission_dim, self.out_dim)
        )
        #for m in self.net:
        #    if isinstance(m, nn.Linear):
        #        nn.init.xavier_uniform_(m.weight)
        #        m.bias.data.fill_(0.01)
        
    
    def forward(self, x):#x --> [N, z_dim, T_max]
        #print(x.shape)
        n_batch, z_dim, T_ = x.shape
        x = x.permute(0, 2,1).reshape(-1, z_dim)
        out = self.net(x)
        
        return out.reshape(n_batch, T_,self.out_dim)#.permute(0,2,1)
        
if __name__ == "__main__":
    gt = Emitter(3, 5, 2)
    x = torch.randn((2, 3, 5))
    gt(x)
