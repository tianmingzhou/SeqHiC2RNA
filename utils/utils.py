import random
import math
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import scanpy as sc
import pickle

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

def read_Expre_mtx(path):
    total_expressions = sc.read_mtx(path)
    return total_expressions

def read_1D_HiC(path):
    with open(path, 'rb') as f:
        data = pickle.load(f)
    return data