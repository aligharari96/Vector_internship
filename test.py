from data import PolyphonicDataset
from torch.utils.data import DataLoader
import torch.nn as nn

import torch

if __name__ == "__main__":
    m = nn.Sigmoid()
    loss = nn.BCELoss()
    input = torch.randn(3, 2, requires_grad=True)
    target = torch.rand(3, 2, requires_grad=False)
    output = loss(m(input), target)
    output.backward()
    print(output)
    """train_path = "/Users/alihosseinghararifoomani/Desktop/data/polyphonic/train.pkl"
    train_dataset = PolyphonicDataset(train_path)
    train_loader = DataLoader(train_dataset, batch_size=32)
    for seq, rev_seq, seq_len in train_loader:
        print(torch.sum(seq[0], dims=1), seq_len)
        assert 0"""