import torch
import torch.nn as nn
import os

from torch.utils.data import Dataset, DataLoader

from einops import rearrange

from algo.module import AttentionPool, Residual
from algo.module import ConvBlock, exponential_linspace_int

from utils.data import seq_indices_to_one_hot
from utils.utils import read_DNAseq_tsv_enf

from tqdm import tqdm

class CNN_Extractor(nn.Module):
    def __init__(self):
        super(CNN_Extractor, self).__init__()
        dim = 1536
        dim_divisible_by = 1536 / 12
        self.dim = dim
        half_dim = dim // 2
        num_downsample = 7


        # create stem

        self.stem = nn.Sequential(
            nn.Conv1d(4, half_dim, 15, padding = 7),
            Residual(ConvBlock(half_dim)),
            AttentionPool(half_dim, pool_size = 2)
        )       

        # create conv tower

        filter_list = exponential_linspace_int(half_dim, dim, num = (num_downsample - 1), divisible_by = dim_divisible_by)
        filter_list = [half_dim, *filter_list]  

        conv_layers = []
        for dim_in, dim_out in zip(filter_list[:-1], filter_list[1:]):
            conv_layers.append(nn.Sequential(
                ConvBlock(dim_in, dim_out, kernel_size = 5),
                Residual(ConvBlock(dim_out, dim_out, 1)),
                AttentionPool(dim_out, pool_size = 2)
            ))

        self.conv_tower = nn.Sequential(*conv_layers)     

    def forward(
        self,
        x,
    ):
        if x.dtype == torch.long:
            x = seq_indices_to_one_hot(x)
        
        x = rearrange(x, 'b n d -> b d n')
        x = self.stem(x)
        x = self.conv_tower(x)
        x = rearrange(x, 'b d n -> b n d')

        return x

class bulk_mBC(Dataset):
    def __init__(self, seq):
        self.seq = seq

    def __getitem__(self, index):
        seq = torch.tensor(self.seq[index])

        return seq.long()
    
    def __len__(self):
        return self.seq.shape[0]

# prepare the device
device = torch.device("cuda:7")

# load data
total_sequences = read_DNAseq_tsv_enf('./data/sequence_1024_200.tsv')
dataset = bulk_mBC(total_sequences)
data_loader = DataLoader(dataset=dataset, batch_size=2, shuffle=False, num_workers=8)

# load model
model = CNN_Extractor().to(device)
model.stem.load_state_dict(torch.load('./algo/pretrain/stem.pt'))
model.conv_tower.load_state_dict(torch.load('./algo/pretrain/conv_tower.pt'))
model.eval()
with torch.no_grad():
    total_token = []
    with tqdm(total=len(data_loader), dynamic_ncols=True) as t:
        t.set_description('Generate Sequence Vector')
        for seq in data_loader:
            seq = seq.to(device)
            token = model(seq)
            total_token.append(token.detach().cpu())
            t.update()
    total_token = torch.concat(total_token, dim=0)

torch.save(total_token, './data/pretrain/sequence_vector.pt')





