import torch
import torch.optim as optim
import argparse
import numpy as np

from utils.utils import seed_all
from utils.dataset import load_data
from algo.enformer import Enformer
from tqdm import tqdm

def train():
    train_loader, valid_loader, test_loader = load_data(
        path = args.data_path, 
        seed = args.seed, 
        batch_size = args.batch_size, 
        num_workers = args.num_workers, 
        target_len = args.target_length)

    model = Enformer.from_hparams(
        dim = args.dim,
        depth = args.depth,
        heads = args.heads,
        output_heads = dict(human=args.output_heads),
        target_length = args.target_length,
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)

    # start training
    print('Start training')
    for epoch in range(args.epochs):
        train_loss = []
        model.train()
        with tqdm(total=len(train_loader), dynamic_ncols=True) as t:
            t.set_description(f'Epoch: {epoch}/{args.epochs}')
            for seq, exp, index in train_loader:
                seq, exp, index = seq.to(device), exp.to(device), index.to(device)
                loss = model(seq, head='human', target=exp, index=index)

                train_loss.append(loss.item())

                # backward
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                t.update()
                t.set_postfix({
                    'train_loss': f'{loss.item():.4f}',
                })
        train_loss = np.mean(train_loss)
        print("In epoch {}, Train Loss: {:.5}\n".format(epoch+1, train_loss))


if __name__=='__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_path', default='./data', help='Path of the dataset')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--batch_size', default=2, type=int)
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--lr', default=1e-4, type=float, help='Learning Rate')
    parser.add_argument('--wd', default=0.0, help='L2 Regularization for Optimizer')
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument("--gpu", type=int, default="0", help="Set GPU Ids : Eg: For CPU = -1, For Single GPU = 0")

    # Enformer hyperparameters
    parser.add_argument('--dim', default=1536, type=int)
    parser.add_argument('--depth', default=11, type=int, help='Number of transformer blocks')
    parser.add_argument('--heads', default=8, type=int, help='Attention Heads')
    parser.add_argument('--output_heads', default=3740, type=int)
    parser.add_argument('--target_length', default=1920, type=int)

    args = parser.parse_args()
    print(args)

    device = torch.device(f"cuda:{args.gpu}" if (torch.cuda.is_available() and args.gpu >= 0) else "cpu")
    print(f"Device is {device}.")

    seed_all(args.seed)

    train()







