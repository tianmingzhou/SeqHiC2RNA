import torch
import torch.optim as optim
import torch.nn.functional as F
import argparse
import numpy as np
import os
import wandb
import shutil

from utils.utils import seed_all
from utils.dataset import load_data_pbulk
from algo.Hcformer_pretrain import Hcformer
from algo.module import pearson_corr_coef, poisson_loss
from tqdm import tqdm
from torch import nn
from typing import List
from scipy.sparse import coo_matrix

def sparse_to_torch(coo_matrix: List[coo_matrix]):
    dense_matrix = []
    for m in coo_matrix:
        m = m.toarray()
        m = m + m.T
        m /= torch.ones(400) + torch.eye(400)
        dense_matrix.append(m)
    return torch.stack(dense_matrix, dim=0)

def evaluation(model, data_loader, device):
    model.eval()
    with torch.no_grad():
        with tqdm(total=len(data_loader), dynamic_ncols=True) as t:
            t.set_description('Evaluation: ')
            total_pred = []
            total_exp = []
            for item in data_loader:
                if args.hic_1d and args.hic_2d:
                    seq, exp, hic_1d, hic_2d = item[0].to(args.device), item[1], item[2].to(args.device), item[3]
                    hic_2d = sparse_to_torch(hic_2d).to(args.device)
                elif args.hic_1d:
                    seq, exp, hic_1d = item[0].to(args.device), item[1], item[2].to(args.device)
                    hic_2d = None
                elif args.hic_2d:
                    seq, exp, hic_2d = item[0].to(args.device), item[1], item[2]
                    hic_1d = None
                    hic_2d = sparse_to_torch(hic_2d).to(args.device)
                pred = model(seq, head='human', hic_1d=hic_1d, hic_2d=hic_2d)

                total_pred.append(pred.detach().cpu())
                total_exp.append(exp.unsqueeze(-1))
                t.update()
            total_pred = torch.concat(total_pred, dim=0)
            total_exp  = torch.concat(total_exp,  dim=0)

    return pearson_corr_coef(total_pred, total_exp)[0]

def train():
    if args.use_wandb:
        run = wandb.init()
        lr = wandb.config.lr
        wd = wandb.config.wd
        depth = wandb.config.depth
    else:
        lr = args.lr
        wd = args.wd
        depth = args.depth

    train_loader, valid_loader, test_loader = load_data_pbulk(
        path = args.data_path, 
        seed = args.seed, 
        batch_size = args.batch_size, 
        num_workers = args.num_workers, 
        target_len = args.target_length,
        hic_1d=args.hic_1d,
        hic_2d=args.hic_2d)

    model = Hcformer.from_hparams(
        dim = args.dim,
        seq_dim = args.dim,
        depth = depth,
        heads = args.heads,
        output_heads = dict(human=args.output_heads),
        target_length = args.target_length,
        dim_divisible_by = args.dim / 12,
        hic_1d = args.hic_1d,
        hic_1d_feat_num = args.hic_1d_feat_num,       
        hic_1d_feat_dim = args.dim,
        hic_2d = args.hic_2d,
    ).to(args.device)

    if len(args.gpu) > 1:
        model = nn.DataParallel(model, device_ids=args.gpu)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=wd)

    # start training
    print('Start training')
    best_pearson_corr_coef = -1
    best_epoch = 0
    kill_cnt = 0
    for epoch in range(args.epochs):
        train_loss = []
        train_pred = []
        train_exp = []
        model.train()
        with tqdm(total=len(train_loader), dynamic_ncols=True) as t:
            t.set_description(f'Epoch: {epoch+1}/{args.epochs}')
            for item in train_loader:
                if args.hic_1d and args.hic_2d:
                    seq, exp, hic_1d, hic_2d = item[0].to(args.device), item[1].to(args.device), item[2].to(args.device), item[3]
                    hic_2d = sparse_to_torch(hic_2d).to(args.device)
                elif args.hic_1d:
                    seq, exp, hic_1d = item[0].to(args.device), item[1].to(args.device), item[2].to(args.device)
                    hic_2d = None
                elif args.hic_2d:
                    seq, exp, hic_2d = item[0].to(args.device), item[1].to(args.device), item[2]
                    hic_1d = None
                    hic_2d = sparse_to_torch(hic_2d).to(args.device)
                pred = model(seq, head='human', hic_1d=hic_1d, hic_2d=hic_2d)

                # compute loss and metric
                tr_loss = poisson_loss(pred, exp.unsqueeze(-1))
                train_loss.append(tr_loss.item())

                train_pred.append(pred.detach().cpu())
                train_exp.append(exp.unsqueeze(-1).detach().cpu())

                # backward
                optimizer.zero_grad()
                tr_loss.backward()
                optimizer.step()

                t.update()
                t.set_postfix({
                    'train_loss': f'{tr_loss.item():.4f}',
                })
        train_loss = np.mean(train_loss)
        train_pred = torch.concat(train_pred, dim=0)
        train_exp = torch.concat(train_exp, dim=0)
        train_pearson_corr_coef = pearson_corr_coef(train_pred, train_exp)[0]

        # validate
        mean_valid_pearson_corr_coef = evaluation(model, valid_loader, args.device)

        print("In epoch {}, Train Loss: {:.5}, Train Pearson_Corr_Coef: {:.5}, Valid Pearson_Corr_Coef: {:.5}\n".format(epoch+1, train_loss, train_pearson_corr_coef, mean_valid_pearson_corr_coef))
        if args.use_wandb:
            wandb.log({
                'epoch': epoch+1,
                'Train Loss': train_loss,
                'Train Pearson_Corr_Coef': train_pearson_corr_coef,
                'Valid Pearson_Corr_Coef': mean_valid_pearson_corr_coef,
            })

        if mean_valid_pearson_corr_coef > best_pearson_corr_coef:
            best_pearson_corr_coef = mean_valid_pearson_corr_coef
            best_epoch = epoch + 1
            kill_cnt = 0
            if args.use_wandb:
                torch.save(model.state_dict(), os.path.join(args.model_save_path, run.id))
            else:
                torch.save(model.state_dict(), os.path.join(args.model_save_path, 'best_model'+str(args.gpu)))
            print("saving model...")
        else:
            kill_cnt += 1
            if kill_cnt >= args.early_stop:
                print('early stop.')
                break
        
    # Use the best model to test
    model.eval()
    if args.use_wandb:
        model.load_state_dict(torch.load(os.path.join(args.model_save_path, run.id)))
    else:
        model.load_state_dict(torch.load(os.path.join(args.model_save_path, 'best_model'+str(args.gpu))))
    mean_test_pearson_corr_coef = evaluation(model, test_loader, args.device)
    print("Best epoch: {}, Best Valid Pearson_Corr_Coef: {:.5}, Test Pearson_Corr_Coef: {:.5}\n".format(best_epoch, best_pearson_corr_coef, mean_test_pearson_corr_coef)) # We Already plus 1 for best epoch in previous code
    if args.use_wandb:
        wandb.log({
            'Best Epoch': best_epoch,
            'Best Valid Pearson_Corr_Coef': best_pearson_corr_coef,
            'Test Pearson_Corr_Coef': mean_test_pearson_corr_coef,
        })

if __name__=='__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_path', default='./data', help='Path of the dataset')
    parser.add_argument('--out_path', default='./output', help='Path to save the output')   
    parser.add_argument('--use_wandb', action='store_true')
    parser.add_argument('--use_sweep', action='store_true')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--batch_size', default=2, type=int)
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--lr', default=1e-4, type=float, help='Learning Rate')
    parser.add_argument('--wd', default=0.0, type=float, help='L2 Regularization for Optimizer')
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--gpu', nargs='*', type=int, default='0', help='Set GPU Ids : Eg: For CPU = -1, For Single GPU = 0')
    parser.add_argument('--num', type=int, default=0, help='To distinguish different sweep')
    parser.add_argument('--early_stop', default=10, type=int, help='Patience for early stop.')

    # Hcformer hyperparameters
    parser.add_argument('--dim', default=1536, type=int)
    parser.add_argument('--seq_dim', default=1536, type=int)
    parser.add_argument('--depth', default=11, type=int, help='Number of transformer blocks')
    parser.add_argument('--heads', default=8, type=int, help='Attention Heads')
    parser.add_argument('--output_heads', default=1, type=int)
    parser.add_argument('--target_length', default=240, type=int)
    parser.add_argument('--hic_1d', action='store_true')
    parser.add_argument('--hic_1d_feat_num', default=5, type=int)
    parser.add_argument('--hic_1d_feat_dim', default=1536, type=int)
    parser.add_argument('--hic_2d', action='store_true')

    # parallelize sweep
    parser.add_argument('--parallelize', action='store_true')
    parser.add_argument('--sweep_id', type=str)

    args = parser.parse_args()
    print(args)

    # prepare the device
    if len(args.gpu) > 0:
        device = torch.device(f"cuda:{args.gpu[0]}")
        print(f"Device is {args.gpu}")
    else:
        device = torch.device(f"cuda:{args.gpu}" if (torch.cuda.is_available() and args.gpu >= 0) else "cpu")
        print(f"Device is {device}.")
    args.device = device
    
    # prepare the output
    args.out_path = os.path.join(args.out_path, 'hcformer_pbulk')
    os.makedirs(args.out_path, exist_ok=True)
    os.makedirs(os.path.join(args.out_path, 'model'), exist_ok=True)
    args.model_save_path = os.path.join(args.out_path, 'model')

    seed_all(args.seed)

    if args.use_wandb:
        if args.use_sweep:
            sweep_name = 'hcformer_pbulk'+str(args.num)
            sweep_configuration = {
                'project': 'hcformer_pbulk',
                'method': 'random',
                'name': sweep_name,
                'parameters':{
                    'lr':{
                        'values': [5e-4, 1e-4, 5e-5],
                    },
                    'wd':{
                        'values': [1e-4, 5e-5, 1e-5, 5e-6, 1e-6],
                    },
                    'depth':{
                        'values': [11],
                    }
                }
            }
            args.model_save_path = os.path.join(args.model_save_path, sweep_name)
            if os.path.exists(args.model_save_path):
                shutil.rmtree(args.model_save_path)
            os.mkdir(args.model_save_path)
            sweep_id = wandb.sweep(sweep=sweep_configuration, project='hcformer_pbulk')
            wandb.agent(sweep_id, function=train)
        elif args.parallelize:
            args.model_save_path = os.path.join(args.model_save_path, 'hcformer_pbulk'+str(args.num))
            if not os.path.exists(args.model_save_path):
                os.mkdir(args.model_save_path)
            wandb.agent(sweep_id=args.sweep_id, function=train)
    else:
        train()