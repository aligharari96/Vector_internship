import torch
import torch.nn as nn
from utils import pad_and_reverse, kl_div
from torch.distributions import Normal

class Emitter(nn.Module):
    """
    Parameterizes the bernoulli observation likelihood `p(x_t | z_t)`
    """

    def __init__(self, input_dim, z_dim, emission_dim):
        super().__init__()
        # initialize the three linear transformations used in the neural network
        self.lin_z_to_hidden = nn.Linear(z_dim, emission_dim)
        self.lin_hidden_to_hidden = nn.Linear(emission_dim, emission_dim)
        self.lin_hidden_to_input = nn.Linear(emission_dim, input_dim)
        # initialize the two non-linearities used in the neural network
        self.relu = nn.ReLU()

    def forward(self, z_t):
        """
        Given the latent z at a particular time step t we return the vector of
        probabilities `ps` that parameterizes the bernoulli distribution `p(x_t|z_t)`
        """
        h1 = self.relu(self.lin_z_to_hidden(z_t))
        h2 = self.relu(self.lin_hidden_to_hidden(h1))
        ps = torch.sigmoid(self.lin_hidden_to_input(h2))
        return ps

class GatedTransition(nn.Module):
    """
    Parameterizes the gaussian latent transition probability `p(z_t | z_{t-1})`
    See section 5 in the reference for comparison.
    """

    def __init__(self, z_dim, transition_dim):
        super().__init__()
        # initialize the six linear transformations used in the neural network
        self.lin_gate_z_to_hidden = nn.Linear(z_dim, transition_dim)
        self.lin_gate_hidden_to_z = nn.Linear(transition_dim, z_dim)
        self.lin_proposed_mean_z_to_hidden = nn.Linear(z_dim, transition_dim)
        self.lin_proposed_mean_hidden_to_z = nn.Linear(transition_dim, z_dim)
        self.lin_sig = nn.Linear(z_dim, z_dim)
        self.lin_z_to_loc = nn.Linear(z_dim, z_dim)
        # modify the default initialization of lin_z_to_loc
        # so that it's starts out as the identity function
        self.lin_z_to_loc.weight.data = torch.eye(z_dim)
        self.lin_z_to_loc.bias.data = torch.zeros(z_dim)
        # initialize the three non-linearities used in the neural network
        self.relu = nn.ReLU()
        self.softplus = nn.Softplus()

    def forward(self, z_t_1):
        """
        Given the latent `z_{t-1}` corresponding to the time step t-1
        we return the mean and scale vectors that parameterize the
        (diagonal) gaussian distribution `p(z_t | z_{t-1})`
        """
        # compute the gating function
        _gate = self.relu(self.lin_gate_z_to_hidden(z_t_1))
        gate = torch.sigmoid(self.lin_gate_hidden_to_z(_gate))
        # compute the 'proposed mean'
        _proposed_mean = self.relu(self.lin_proposed_mean_z_to_hidden(z_t_1))
        proposed_mean = self.lin_proposed_mean_hidden_to_z(_proposed_mean)
        # assemble the actual mean used to sample z_t, which mixes a linear transformation
        # of z_{t-1} with the proposed mean modulated by the gating function
        loc = (1 - gate) * self.lin_z_to_loc(z_t_1) + gate * proposed_mean
        # compute the scale used to sample z_t, using the proposed mean from
        # above as input the softplus ensures that scale is positive
        scale = self.softplus(self.lin_sig(self.relu(proposed_mean)))
        # return loc, scale which can be fed into Normal
        return loc, scale


class Combiner(nn.Module):
    """
    Parameterizes `q(z_t | z_{t-1}, x_{t:T})`, which is the basic building block
    of the guide (i.e. the variational distribution). The dependence on `x_{t:T}` is
    through the hidden state of the RNN (see the PyTorch module `rnn` below)
    """

    def __init__(self, z_dim, rnn_dim):
        super().__init__()
        # initialize the three linear transformations used in the neural network
        self.lin_z_to_hidden = nn.Linear(z_dim, rnn_dim)
        self.lin_hidden_to_loc = nn.Linear(rnn_dim, z_dim)
        self.lin_hidden_to_scale = nn.Linear(rnn_dim, z_dim)
        # initialize the two non-linearities used in the neural network
        self.tanh = nn.Tanh()
        self.softplus = nn.Softplus()

    def forward(self, z_t_1, h_rnn):
        """
        Given the latent z at at a particular time step t-1 as well as the hidden
        state of the RNN `h(x_{t:T})` we return the mean and scale vectors that
        parameterize the (diagonal) gaussian distribution `q(z_t | z_{t-1}, x_{t:T})`
        """
        # combine the rnn hidden state with a transformed version of z_t_1
        h_combined = 0.5 * (self.tanh(self.lin_z_to_hidden(z_t_1)) + h_rnn)
        # use the combined hidden state to compute the mean used to sample z_t
        loc = self.lin_hidden_to_loc(h_combined)
        # use the combined hidden state to compute the scale used to sample z_t
        scale = self.softplus(self.lin_hidden_to_scale(h_combined))
        # return loc, scale which can be fed into Normal
        return loc, scale


class DMM(nn.Module):
    """
    This PyTorch Module encapsulates the model as well as the
    variational distribution (the guide) for the Deep Markov Model
    """

    def __init__(
        self,
        input_dim=88,
        z_dim=100,
        emission_dim=100,
        transition_dim=200,
        rnn_dim=600,
        num_layers=1,
        rnn_dropout_rate=0.0,
        num_iafs=0,
        iaf_dim=50,
        use_cuda=False,
    ):
        super().__init__()
        # instantiate PyTorch modules used in the model and guide below
        self.emitter = Emitter(input_dim, z_dim, emission_dim)
        self.trans = GatedTransition(z_dim, transition_dim)
        self.combiner = Combiner(z_dim, rnn_dim)
        # dropout just takes effect on inner layers of rnn
        rnn_dropout_rate = 0.0 if num_layers == 1 else rnn_dropout_rate
        self.rnn = nn.RNN(
            input_size=input_dim,
            hidden_size=rnn_dim,
            nonlinearity="relu",
            batch_first=True,
            bidirectional=False,
            num_layers=num_layers,
            dropout=rnn_dropout_rate,
        )

        

        # define a (trainable) parameters z_0 and z_q_0 that help define the probability
        # distributions p(z_1) and q(z_1)
        # (since for t = 1 there are no previous latents to condition on)
        self.z_0 = nn.Parameter(torch.zeros(z_dim))
        self.z_q_0 = nn.Parameter(torch.zeros(z_dim))
        # define a (trainable) parameter for the initial hidden state of the rnn
        self.h_0 = nn.Parameter(torch.zeros(1, 1, rnn_dim))

        self.use_cuda = use_cuda
        # if on gpu cuda-ize all PyTorch (sub)modules
        if use_cuda:
            self.cuda()
    

def loss_function(dmm, mini_batch,
        mini_batch_reversed,
        mini_batch_mask,
        mini_batch_seq_lengths,
        annealing_factor=1.0,):
    T_max = mini_batch.size(1)
    batch_size = mini_batch.size(0)
    z_prev = dmm.z_q_0.expand(mini_batch.size(0), dmm.z_0.size(0))
    h_0_contig = dmm.h_0.expand(1, mini_batch.size(0), dmm.rnn.hidden_size).contiguous()
    rnn_output, _ = dmm.rnn(mini_batch_reversed, h_0_contig)
        # reverse the time-ordering in the hidden state and un-pack it
    rnn_output = pad_and_reverse(rnn_output, mini_batch_seq_lengths)
    rec_losses = torch.zeros((mini_batch.size(0), T_max), device= mini_batch.device) 
    kl_states = torch.zeros((mini_batch.size(0), T_max), device= mini_batch.device)  
    
    for t in range(T_max):
        z_loc, z_scale = dmm.combiner(z_prev, rnn_output[:, t , :])
        #z_scale = torch.ones_like(z_scale)
        #z_loc = torch.ones_like(z_scale)

        #print(z_loc[0][0])
        #print(z_scale[0][0])
        #print(z_prev)
        #print(z_loc)
        #print(z_scale)
        #print(dmm.combiner.lin_z_to_hidden.weight)
        #assert 0
        if t ==0:
            z_prior_loc, z_prior_scale = dmm.trans(dmm.z_0.expand(mini_batch.size(0), dmm.z_0.size(0)))
        else:
            z_prior_loc, z_prior_scale = dmm.trans(z_prev)
        #z_prior_scale = torch.ones_like(z_scale)
        #z_prior_loc = torch.ones_like(z_scale)
        #print(z_prior_loc[0][0])
        
        
        #z = torch.randn(z_scale.shape) * torch.sqrt(z_scale) + z_loc
        z = Normal(z_loc, z_scale).sample((1,))[0]
        x_hat = dmm.emitter(z)
        #print(z.std(0))
        #assert 0
        #z_post = Normal(z_loc, z_scale).sample((200,)).mean(dim=0)
        
        
        kl = Normal(z_loc, z_scale).log_prob(z) - Normal(z_prior_loc, z_prior_scale).log_prob(z)

        kl = kl.sum(dim=1)
        #print(kl.shape)
        #assert 0

        
        #assert 0
        
        #kl = torch.sum(kl_div(z_loc, 2*torch.log(z_scale), z_prior_loc, 2*torch.log(z_prior_scale)), dim=1)
        #kl = torch.sum(kl_div(z_loc, z_scale, z_prior_loc, z_prior_scale), dim=1)
        kl_states[:,t] = kl
        
        
        #print(x_hat[0][0])
        
        rec_loss_ = nn.BCELoss(reduction='none')(x_hat.view(-1),   mini_batch[:,t,:].contiguous().view(-1)).view(batch_size,-1).sum(-1)
        
        
        
        rec_losses[:,t] = rec_loss_  
        #print(z[0][0])  
        z_prev = z
    x_mask = mini_batch_mask.gt(0).view(-1)
    rec_loss = rec_losses.view(-1).masked_select(x_mask).sum()#/torch.sum(mini_batch_seq_lengths)
    kl_loss = kl_states.view(-1).masked_select(x_mask).sum()#/torch.sum(mini_batch_seq_lengths)
    #print(rec_loss, kl_loss)
    return rec_loss, kl_loss




