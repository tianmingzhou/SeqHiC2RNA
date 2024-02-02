import torch
import argparse
import os

from torch import nn
from algo.Hcformer_pretrain import Hcformer, CNN_Extractor
from utils.data import str_to_seq_indices

parser = argparse.ArgumentParser()

parser.add_argument('--data_path', default='./data', help='Path of the dataset')
parser.add_argument('--pretrain_para_path', default='./algo/pretrain', help='Path to store the pretrain model')
parser.add_argument('--hcformer_path', default='./output/model/hcformer_pbulk/hic1d2d/d1xcmvsr')
parser.add_argument('--gpu', nargs='*', type=int, default='0', help='Set GPU Ids : Eg: For CPU = -1, For Single GPU = 0')

args = parser.parse_args()
print(args)

# prepare the device
if len(args.gpu) > 0:
    device = torch.device(f"cuda:{args.gpu[0]}")
    print(f"Device is {args.gpu}")
else:
    device = torch.device(f"cuda:{args.gpu}" if (torch.cuda.is_available() and args.gpu >= 0) else "cpu")
    print(f"Device is {device}.")

# prepare the input sample
seq = torch.randint(0, 5, (1, 409600))
# if your input sample is str type, you can use this function to convert your string
# we convert A->0, C->1, G->2, T->3, N->4
# seq = str_to_seq_indices(seq_str)

hic_1d = torch.rand(1, 400, 5)
hic_2d = torch.rand(1, 400, 400)

# Define the CNN Extractor
# We need to first use this CNN Extractor to convert one hot DNA sequence 
cnn = CNN_Extractor().to(device)
cnn.stem.load_state_dict(torch.load(os.path.join(args.pretrain_para_path, 'stem.pt')))
cnn.conv_tower.load_state_dict(torch.load(os.path.join(args.pretrain_para_path, 'conv_tower.pt')))

# Define the Hcformer
# Do not need to change the hyperparameter in the following model definition
model = Hcformer.from_hparams(
    dim = 768,
    seq_dim = 768,
    depth = 11,
    heads = 8,
    output_heads = dict(human=1),
    target_length = 240,
    dim_divisible_by = 128,
    hic_1d = True,
    hic_1d_feat_num = 5, 
    hic_1d_feat_dim = 768,
    hic_2d = True,
).to(device)


# You may face some problem loading the parameter here, that's because I haven't saved the 
# model's parameter in a general way(My Fault), you can manually convert the model parameter
# yourself, such as load the parameter and save it in general way(do not have "module" in key).
# Or you could just first let the model to be paralleled then load the parameter.
# You can use the following code to manually transforme the model parameter to be general version,
# note that you need to select at least two gpu "--gpu 0 1" to transform the parameter.
# if len(args.gpu) > 1:
#     model = nn.DataParallel(model, device_ids=args.gpu)
# model.load_state_dict(torch.load(args.hcformer_path))
# torch.save(model.module.state_dict(), args.hcformer_path)

model.load_state_dict(torch.load(args.hcformer_path))
if len(args.gpu) > 1:
    model = nn.DataParallel(model, device_ids=args.gpu)

cnn.eval()
model.eval()

with torch.no_grad():
    seq, hic_1d, hic_2d = seq.to(device), hic_1d.to(device), hic_2d.to(device)
    seq_dense = cnn(seq)
    pred = model(seq_dense, head='human', hic_1d=hic_1d, hic_2d=hic_2d)


