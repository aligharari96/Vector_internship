import numpy as np
import six.moves.cPickle as pickle
#from observations import jsb_chorales#, musedata, piano_midi_de, nottingham
import torch
import torch.nn as nn
import torch.utils.data as data
import os


def prepare_polyphonic(base_path, name='jsb_chorales', T_max=160, min_note=21, note_range=88):
    print("processing raw polyphonic music data...")
    data = eval(name)(base_path)
    processed = {}
    for split, data_split in zip(['train', 'test', 'valid'], data):
        processed = {}
        n_seqs = len(data_split)
        processed['seq_lens'] = np.zeros((n_seqs), dtype=np.int32)
        processed['sequences'] = np.zeros((n_seqs, T_max, note_range))
        for i in range(n_seqs):            
            seq_len = len(data_split[i])
            processed['seq_lens'][i] = seq_len
            for t in range(seq_len):                
                note_slice = np.array(list(data_split[i][t])) - min_note
                slice_len = len(note_slice)
                if slice_len > 0:
                    processed['sequences'][i, t, note_slice] = np.ones((slice_len))
        f_out = os.path.join(base_path, split+'.pkl')
        pickle.dump(processed, open(f_out, "wb"), pickle.HIGHEST_PROTOCOL)
        print("dumped processed data to %s" % f_out)

class PolyphonicDataset(data.Dataset):
    def __init__(self, filepath):
        # 1. Initialize file path or list of file names.
        """read training sequences(list of int array) from a pickle file"""
        print("loading data...")
        f= open(filepath, "rb")
        data = pickle.load(f)
        self.seqs = data['sequences']
        self.seqlens = data['seq_lens']
        self.data_len = len(self.seqs)
        print("{} entries".format(self.data_len))

    def __getitem__(self, offset):
        seq=self.seqs[offset].astype('float32')
        rev_seq= seq.copy()
        rev_seq[0:len(seq), :] = seq[(len(seq)-1)::-1, :]
        seq_len=self.seqlens[offset].astype('int64')
        return seq, rev_seq, seq_len

    def __len__(self):
        return self.data_len
    
if __name__ == "__main__":
    prepare_polyphonic(base_path='polyphonic', T_max=30)
    assert 0
    from torch.utils.data import DataLoader
    train_set = PolyphonicDataset('/Users/alihosseinghararifoomani/Desktop/data/polyphonic/train.pkl')
    train_set = PolyphonicDataset('/Users/alihosseinghararifoomani/Desktop/data/polyphonic/valid.pkl')
    train_set = PolyphonicDataset('/Users/alihosseinghararifoomani/Desktop/data/polyphonic/test.pkl')
    tr_loader = DataLoader(train_set)
    for seq, rev_seq, seq_len in tr_loader:
        print(seq.shape)
        print(rev_seq.shape)
        print(seq_len)
        assert 0