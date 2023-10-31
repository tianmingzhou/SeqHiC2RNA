import torch
import os

path = './data'
pretrain_tensor = torch.load(os.path.join(path, 'sequence_vector.pt'))
pretrain_tensor = pretrain_tensor.half()
torch.save(pretrain_tensor, os.path.join(path, 'sequence_vector.pt'))
