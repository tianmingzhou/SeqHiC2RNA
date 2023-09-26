import os
import torch
import numpy as np

from torch.utils.data import Dataset, DataLoader
from utils.utils import read_DNAseq_tsv, read_Expre_tsv

class mBC(Dataset):
    def __init__(self, seq, exp):
        self.seq = seq
        self.exp = exp

    def __getitem__(self, index):
        seq = torch.tensor(self.seq[index])
        exp = torch.tensor(self.exp[index])

        return seq.long(), exp.float(), index
    
    def __len__(self):
        return self.seq.shape[0]

def load_data(path, seed, batch_size, num_workers, target_len):
    
    total_sequences = read_DNAseq_tsv(os.path.join(path, 'sequence_1024_200.tsv'))
    total_expressions = read_Expre_tsv(os.path.join(path, 'expression_cov_1024_200_bulk.tsv'))

    # manually convert 1024-resolution to 128-resolution
    total_expressions = np.repeat(total_expressions, 8).reshape(total_expressions.shape[0], -1)

    # crop the DNA-sequence from two sides
    trim = (target_len - total_expressions.shape[1]) // 2
    total_expressions = total_expressions[:, -trim:trim]

    # generate random indice
    k = int(total_sequences.shape[0]/20)
    indice = np.zeros((total_sequences.shape[0], 1))
    indice[:k] = 1
    indice[k:2*k] = 2
    np.random.seed(seed)
    indice = np.random.permutation(indice)
    train_indice = np.where(indice==0)[0]
    valid_indice = np.where(indice==2)[0]
    test_indice = np.where(indice==1)[0]

    # split the data
    train_seq = total_sequences[train_indice]
    train_exp = total_expressions[train_indice]

    valid_seq = total_sequences[valid_indice]
    valid_exp = total_expressions[valid_indice]

    test_seq = total_sequences[test_indice]
    test_exp = total_expressions[test_indice]

    train_dataset = mBC(train_seq, train_exp)
    valid_dataset = mBC(valid_seq, valid_exp)
    test_dataset = mBC(test_seq, test_exp)

    train_loader = DataLoader(
        dataset = train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    valid_loader = DataLoader(
        dataset = valid_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(
        dataset = test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, valid_loader, test_loader











