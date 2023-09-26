import random
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd

from tqdm import tqdm

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

def read_Expre_tsv(path):
    total_expressions = pd.read_csv(path, sep='\t', header=None)
    return total_expressions.values

def pearson_corr_coef(x, y, dim = 1, reduce_dims = (-1,)):
    x_centered = x - x.mean(dim = dim, keepdim = True)
    y_centered = y - y.mean(dim = dim, keepdim = True)
    return F.cosine_similarity(x_centered, y_centered, dim = dim).mean(dim = reduce_dims)
    
