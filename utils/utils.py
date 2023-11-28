import random
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import scanpy as sc
import pickle
import h5py

from tqdm import tqdm
from typing import List
from utils.data import str_to_seq_indices
from scipy.sparse import coo_matrix

def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.random.manual_seed(seed)
    torch.cuda.manual_seed(seed)

def read_DNAseq_tsv(path):
    total_sequences = []
    with open(path, mode='r') as f:
        lines = f.readlines()
        for line in tqdm(lines, desc='Processing'):
            line = line.strip()
            sequence = (np.frombuffer(line.upper().encode(), dtype=np.uint8) + 1) % 5
            total_sequences.append(sequence.reshape(1, -1))
    
    total_sequences = np.concatenate(total_sequences, axis=0)

    return total_sequences

def read_DNAseq_tsv_enf(path):
    total_sequences = []
    with open(path, mode='r') as f:
        lines = f.readlines()
        for line in tqdm(lines, desc='Processing'):
            line = line.strip()
            sequence = str_to_seq_indices(line)
            total_sequences.append(sequence.reshape(1, -1))
    total_sequences = np.concatenate(total_sequences, axis=0)

    return total_sequences


def read_Expre_tsv(path):
    total_expressions = pd.read_csv(path, sep='\t', header=None)
    return total_expressions.values

def read_Expre_mtx(path):
    total_expressions = sc.read_mtx(path)
    return total_expressions

def read_1D_HiC(path):
    with open(path, 'rb') as f:
        data = pickle.load(f)
    return data

def read_pbulk_exp(path):
    with open(path, 'rb') as f:
        data = pickle.load(f).squeeze()
    for i in range(len(data)):
        data[i] = data[i].toarray()
    data = np.concatenate(data, axis=0)
    return data
    
def hic_h5_coo(path):
    total_hic = []
    with h5py.File(path, 'r') as f:
        cell_type = list(f.keys())
        gene_name = list(f[cell_type[0]])
        for c_type in tqdm(cell_type, desc='Processing'):
            type_hic = f[c_type]
            for g_name in gene_name:
                hic = type_hic[g_name]
                row = hic['row']
                col = hic['col']
                data = hic['data']

                coo_mat = coo_matrix((data, (row, col)), shape=(400, 400))
                coo_mat.sum_duplicates()
                coo_mat.data = np.log1p(coo_mat.data)
                total_hic.append(coo_mat)
    return total_hic