from collections import namedtuple
from utils import *
from modules import DMM, loss_function

if __name__ == "__main__":
    #torch.manual_seed(0)
    #np.random.seed(0)
    dset = namedtuple("dset", ["name", "url", "filename"])

    JSB_CHORALES = dset(
    "jsb_chorales",
    "https://d2hg8soec8ck9v.cloudfront.net/datasets/polyphonic/jsb_chorales.pickle",
    "jsb_chorales.pkl",
)

    data = load_data(JSB_CHORALES ,'data/jsb_chorales')
    mini_batch_size = 40
    training_seq_lengths = data["train"]["sequence_lengths"]
    training_data_sequences = data["train"]["sequences"]
    test_seq_lengths = data["test"]["sequence_lengths"]
    test_data_sequences = data["test"]["sequences"]
    val_seq_lengths = data["valid"]["sequence_lengths"]
    val_data_sequences = data["valid"]["sequences"]
    N_train_data = len(training_seq_lengths)
    N_train_time_slices = float(torch.sum(training_seq_lengths))
    N_mini_batches = int(
        N_train_data / mini_batch_size
        + int(N_train_data % mini_batch_size > 0)
    )
    dmm = DMM()
    #with torch.no_grad():
    #    dmm.load_state_dict(torch.load('dmm.pt'))
    #print(torch.load('dmm.pt')['z_0'])
    #assert 0
    
    optimizer = torch.optim.Adam(dmm.parameters(), lr=0.0001, betas=(0.96, 0.996), weight_decay=0.001)

    shuffled_indices = torch.arange(N_train_data)

    for epoch in range(10000):
        
        epoch_nll = 0
        #print('--------------', epoch, '------------------')
        # process each mini-batch; this is where we take gradient steps
        for which_mini_batch in range(N_mini_batches):
            mini_batch,mini_batch_reversed,mini_batch_mask,mini_batch_seq_lengths = process_minibatch(epoch,training_data_sequences,training_seq_lengths,mini_batch_size,N_train_data, which_mini_batch, shuffled_indices)
            optimizer.zero_grad()
            rec_loss, kl_loss = loss_function(dmm, mini_batch, mini_batch_reversed, mini_batch_mask, mini_batch_seq_lengths)
            if 1000 > 0 and epoch < 1000: # compute the KL annealing factor            
                min_af = 0.2
                annealing_factor = min_af + (1.0 - min_af) * (
                float(which_mini_batch + epoch * N_mini_batches + 1)
                / float(1000 * N_mini_batches)
            )
            else:            
                annealing_factor = 1.0 # by default the KL annealing factor is unity
            loss = rec_loss + annealing_factor   *  kl_loss
            epoch_nll += loss.detach().clone()
            
            #print(annealing_factor,(loss)/torch.sum(mini_batch_seq_lengths))
            #assert 0
            #if which_mini_batch == 0:
            #    print(loss)
            
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(dmm.parameters(), 1)
            optimizer.step()
        print(epoch, epoch_nll/torch.sum(training_seq_lengths))
        #assert 0
            
