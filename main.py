import torch 
import torch.nn as nn
from torch.utils.data import DataLoader
from data import PolyphonicDataset
from posterior import InitialPosterior
from flow import RealNVP
from modules import Emitter, GatedTransition
from utils import prepare_future, log_likelihood
import numpy as np
import sys

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def kl_div(mu1, logvar1, mu2=None, logvar2=None):#KL(P1||P2)
        one = torch.ones(1)
        if mu2 is None: mu2=torch.zeros(1)
        if logvar2 is None: logvar2=torch.zeros(1)
        return 0.5*(logvar2-logvar1+(torch.exp(logvar1)+(mu1-mu2).pow(2))/torch.exp(logvar2)-one)

if __name__ == "__main__":
    #torch.manual_seed(0)
    #np.random.seed(0)
    T_max = 160
    train_path = "polyphonic/train.pkl"
    test_path = "polyphonic/test.pkl"
    val_path = "polyphonic/valid.pkl"
    train_loader = DataLoader(PolyphonicDataset(train_path), batch_size=20)
    val_loader = DataLoader(PolyphonicDataset(val_path), batch_size=20)
    test_loader = DataLoader(PolyphonicDataset(test_path), batch_size=20)
    training_seq_lengths = PolyphonicDataset(train_path).seqlens.sum()
    val_seq_lengths = PolyphonicDataset(val_path).seqlens.sum()
    test_seq_lengths = PolyphonicDataset(test_path).seqlens.sum()
    
    initial_posterior = InitialPosterior(160, 32, 8, kernel_size=(3,1), padding=(1,0), stride=(1,1), double_after_norm=False)
    initial_posterior.to(DEVICE)
    #flow = RealNVP(2, T_max +2, 32, 2)
    emitter = Emitter(z_dim=88, emission_dim=64, out_dim=88)
    emitter.to(DEVICE)
    gated_transition = GatedTransition(88, 32)
    gated_transition.to(DEVICE)
    bce_loss = nn.BCEWithLogitsLoss(reduction='none')
    z_0 = nn.Parameter(torch.zeros(88).cuda())
    
    
    
    params = list(initial_posterior.parameters()) + \
        list(emitter.parameters()) + list(gated_transition.parameters()) + [z_0]
    optimizer = torch.optim.Adam(params=params, lr=3e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.9)
    for epoch in range(2000):
        #print(i)
        coeff = 0.0 + 1.0 * (epoch/500)
        if coeff >= 1.0:
             coeff = 1.0
        epoch_nll = 0.0
        epoch_kl = 0.0
        epoch_rec = 0.0
        for seq, rev_seq, seq_len in train_loader:
            seq = seq.to(DEVICE)
            rev_seq = rev_seq.to(DEVICE)
            seq_len = seq_len.to(DEVICE)
            
            optimizer.zero_grad()
            mask = torch.ones_like(seq, device = DEVICE).permute(0,2,1)#(N, dim, seq_len)
            for i in range(seq.shape[0]):
                mask[i, :, seq_len[i]:] = 0.0
            
            future = prepare_future(seq, seq_len, DEVICE)#(N, seq_len, dim, seq_len)
            
            q_z_0, mu_0, logvar_0 = initial_posterior(future)#(N, 1, dim, seq_len)
            #print(q_z_0.shape)
            
            
            qz_ = torch.cat([z_0.expand((seq.size(0), z_0.size(0))).unsqueeze(2), q_z_0.squeeze()], dim=2)[:,:,:-1]#(N, dim, seq_len)
            z_prior, mu_prior, logvar_prior = gated_transition(qz_)
            
            
            #q_z_0 = mu_0
            #mu_prior = mu_0
            #logvar_prior = logvar_0
            #kl_new = kl_div(mu_0, logvar_0, mu_prior, logvar_prior) * mask
            #kl = kl_new.sum()
            #print(kl)
            ll_qz = log_likelihood(q_z_0, mu_0, logvar_0)
            
            #z_prior, mu_prior, logvar_prior = gated_transition(qz_)
            
            ll_pz = log_likelihood(q_z_0, mu_prior, logvar_prior)
            
            kl = (ll_qz - ll_pz)*mask
            #print(kl.sum())
            #sys.exit()
            y_hat = emitter(q_z_0)
            
            
            recon_loss = bce_loss(y_hat, seq) * mask.permute(0,2,1)
            recon_loss = recon_loss.sum()
            kl_loss = kl.sum()
            loss = coeff *kl_loss + recon_loss
            loss.backward()
            optimizer.step()
            #scheduler.step()
            epoch_nll += loss.detach().clone()
            epoch_kl += kl_loss.detach().clone()
            epoch_rec += recon_loss.detach().clone()
        
        with torch.no_grad():
            val_nll = 0.0
            for seq, rev_seq, seq_len in val_loader: 
                seq = seq.to(DEVICE)
                rev_seq = rev_seq.to(DEVICE)
                seq_len = seq_len.to(DEVICE)
                mask = torch.ones_like(seq, device = DEVICE).permute(0,2,1)
                for i in range(seq.shape[0]):
                    mask[i, :, seq_len[i]:] = 0.0            
                future = prepare_future(seq, seq_len, DEVICE)
                q_z_0, mu_0, logvar_0 = initial_posterior(future)
                qz_ = torch.cat([z_0.expand((seq.size(0), z_0.size(0))).unsqueeze(2), q_z_0.squeeze()], dim=2)[:,:,:-1]
                z_prior, mu_prior, logvar_prior = gated_transition(qz_)
                ll_qz = log_likelihood(q_z_0, mu_0, logvar_0)
                ll_pz = log_likelihood(q_z_0, mu_prior, logvar_prior)
                kl = (ll_qz - ll_pz)*mask
                y_hat = emitter(q_z_0)
                recon_loss = bce_loss(y_hat, seq) * mask.permute(0,2,1)
                recon_loss = recon_loss.sum()
                kl_loss = kl.sum()
                loss = kl_loss + recon_loss
                val_nll += loss.detach().clone()


        print(epoch, epoch_nll/training_seq_lengths.sum(),"KL: ",epoch_kl/training_seq_lengths.sum(),epoch_rec/training_seq_lengths.sum(), val_nll/val_seq_lengths.sum())

        
        
        #z_prior, mu_prior, logvar_prior = gated_transition(qz_)
        #flow.future = future
        #q_z, log_det = flow(q_z_0)
        #q_z = q_z.squeeze()
        #nll_qz = log_det - log_likelihood(q_z_0, mu_0, logvar_0)
        
        #nll_qz = nll_qz.squeeze()
        #qz_ = torch.cat([z_0.expand((seq.size(0), z_0.size(0))).unsqueeze(2), q_z], dim=2)[:,:,:-1]
        
        #z_prior, mu_prior, logvar_prior = gated_transition(qz_)
        #nll_prior = -1.0 * log_likelihood(q_z, mu_prior, logvar_prior)
        #print(nll_prior)
        #assert 0
        #kl = (nll_qz - nll_prior) * mask
        #print(torch.mean(kl))
        #assert 0
        
        #y_hat = emitter(q_z)
        #recon_loss = bce_loss(y_hat, future[:,0,:,:]) * mask
        
        #kl_weight = 1.0
        #loss = (kl.sum() * kl_weight + recon_loss.sum())/mask.sum()
        #loss.backward()
        #optimizer.step()
        #print(loss)
        #recon_loss = bce_loss(y_hat.reshape(-1,), future[:, 0, :, :].squeeze().reshape(-1,))

        #recon_loss = recon_loss.reshape(y_hat.shape)
        
        #print(recon_loss[0:2, 0, 0:5])
        #print(bce_loss(y_hat[0:2,0,0:5], future[0:2,0,0,0:5]))


        
        



    
    