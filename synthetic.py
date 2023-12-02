import torch 
import matplotlib.pyplot as plt
from posterior import InitialPosterior
from flow import RealNVP
from modules import Emitter, GatedTransition
from utils import prepare_future, log_likelihood



def synthetic_data(z_dim, T, n_samples):
    Z = torch.zeros((n_samples, T, z_dim))#[n_samples, T_max, Z_dim]
    X = torch.zeros((n_samples, T, z_dim))
    Z_std = torch.ones((n_samples, T, z_dim))*0.1
    X_std = torch.ones((n_samples, T, z_dim))*0.5
    #z_current = torch.randn((n_samples, 1, z_dim))
    z_current = torch.randn((n_samples, 1, z_dim)) * 0.0 
    for i in range(T):
        z_mean_current = z_current
        Z[:, i:i+1, :] = z_mean_current
        z_current = z_mean_current + 0.5 +torch.randn(z_mean_current.shape)
        X[:, i:i+1, :] = 2 * z_current 

    return X + torch.randn(X.shape) * 0.1, Z

if __name__ == "__main__":
    X, Z = synthetic_data(z_dim=1, T=30, n_samples=1000)
    Z = Z.permute(0,2,1)
    
    initialposterior = InitialPosterior(in_channels=30, mid_channels=128, num_blocks=4, kernel_size=(3,1), padding=(1,0), stride=(1,1), double_after_norm=False)
    fututre = prepare_future(X, torch.ones((1000,), dtype=torch.int)*30)
    
    flow = RealNVP(num_coupling=2, in_channels=32, mid_channels=5, num_blocks=2)
    emitter = Emitter(z_dim=1, emission_dim=1, out_dim=1)
    gated_transition = GatedTransition(z_dim=1, trans_dim=1)
    z_0 = torch.nn.Parameter(torch.zeros(1))
    optimizer = torch.optim.Adam(list(initialposterior.parameters()) + 
                                 list(emitter.parameters())+list(gated_transition.parameters()), lr=1e-3)
    for i in range(3001):
        #fututre = prepare_future(data, torch.ones((10,), dtype=torch.int)*30)
        q_z_0, mu, logvar = initialposterior(fututre)
        logprob_q = log_likelihood(q_z_0[:,0,:,:], mu[:,0,:,:], logvar[:,0,:,:])
        qz_ = torch.cat([z_0.expand((X.size(0), z_0.size(0))).unsqueeze(2), q_z_0[:,0,:,:]], dim=2)[:,:,:-1]
        _, mu_prior, _ = gated_transition(qz_)
        
        #logprob_p = log_likelihood(q_z_0[:,0,:, :], Z, torch.ones_like(Z) * 0.1)
        logprob_p = log_likelihood(q_z_0[:,0,:, :], mu_prior, torch.ones_like(Z) * 0.1)
        kl = (logprob_q - logprob_p).sum()/logprob_p.shape[0]
        #print(X.permute(0,2,1).shape)
        
        y_hat = emitter(q_z_0[:, 0, :, :])
        
        #print(y_hat[0,:,0])
        
        mse = torch.nn.MSELoss()(X, y_hat)
        loss = 1.0 * kl + mse * 4
        loss.backward()
        optimizer.step()
        print(loss)
        if i %100==0:
            plt.cla()
            
            for idx in range(1, 17):
                plt.subplot(4,4,idx).plot(y_hat[idx,:,0].detach().numpy())
                plt.subplot(4,4,idx).plot(X[idx, :, 0])
            plt.tight_layout()
            plt.savefig('results/'+str(i)+'.png')
            plt.close()


        