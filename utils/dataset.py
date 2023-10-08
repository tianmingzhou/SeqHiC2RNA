import os
import torch
import numpy as np

from torch.utils.data import Dataset, DataLoader
from utils.utils import read_DNAseq_tsv, read_Expre_tsv, read_Expre_mtx, read_1D_HiC

class bulk_mBC(Dataset):
    def __init__(self, seq, exp, hic_1d):
        self.seq = seq
        self.exp = exp
        self.hic_1d = hic_1d

    def __getitem__(self, index):
        seq = torch.tensor(self.seq[index])
        exp = torch.tensor(self.exp[index])
        hic_1d = torch.tensor(self.hic_1d[index])

        return seq.long(), exp.float(), hic_1d.float()
    
    def __len__(self):
        return self.seq.shape[0]

def load_data_bulk(path, seed, batch_size, num_workers, target_len, algo):
    
    total_sequences = read_DNAseq_tsv(os.path.join(path, 'sequence_1024_200.tsv'))
    total_expressions = read_Expre_tsv(os.path.join(path, 'expression_cov_1024_200_bulk.tsv'))
    total_1D_HiC = read_1D_HiC(os.path.join(path, f'1d-score-bulk-10kb-{algo}_1024_200.pkl'))

    # # manually convert 1024-resolution to 128-resolution
    # total_expressions = np.repeat(total_expressions, 8).reshape(total_expressions.shape[0], -1)

    # crop the DNA-sequence from two sides
    trim = (target_len - total_expressions.shape[1]) // 2
    total_expressions = total_expressions[:, -trim:trim]

    # Normalize the Expression data
    row_min = np.min(total_expressions, axis=1, keepdims=True)
    row_max = np.max(total_expressions, axis=1, keepdims=True)
    total_expressions = (total_expressions - row_min) / (row_max - row_min)
    total_expressions = np.log1p(total_expressions * 1e4)

    # transform the 1D HiC data
    total_1D_HiC = total_1D_HiC.reshape(-1, 400)

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
    train_1d_hic = total_1D_HiC[train_indice]

    valid_seq = total_sequences[valid_indice]
    valid_exp = total_expressions[valid_indice]
    valid_1d_hic = total_1D_HiC[valid_indice]

    test_seq = total_sequences[test_indice]
    test_exp = total_expressions[test_indice]
    test_1d_hic = total_1D_HiC[test_indice]

    train_dataset = bulk_mBC(train_seq, train_exp, train_1d_hic)
    valid_dataset = bulk_mBC(valid_seq, valid_exp, valid_1d_hic)
    test_dataset = bulk_mBC(test_seq, test_exp, test_1d_hic)

    train_loader = DataLoader(
        dataset = train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    valid_loader = DataLoader(
        dataset = valid_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(
        dataset = test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, valid_loader, test_loader

def load_data_sc(path, seed, batch_size, num_workers, target_len):
    total_sequences = read_DNAseq_tsv(os.path.join(path, 'sequence_1024_200.tsv'))
    total_expressions = read_Expre_mtx(os.path.join(path, 'expression_cov_1024_200.mtx'))

    


