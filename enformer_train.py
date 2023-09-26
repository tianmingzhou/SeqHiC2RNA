import torch
import torch.optim as optim
import torch.nn.functional as F
import argparse
import numpy as np
import os
import wandb

from utils.utils import seed_all, pearson_corr_coef
from utils.dataset import load_data
from algo.enformer import Enformer
from tqdm import tqdm


def evaluation(model, data_loader, device):
    model.eval()
    with torch.no_grad():
        with tqdm(total=len(data_loader), dynamic_ncols=True) as t:
            t.set_description('Evaluation: ')
            total_pred = []
            total_exp = []
            for seq, exp, index in data_loader:
                seq, index = seq.to(device), index.to(device)
                pred = model(seq, head='human', index=index, return_fetch_pred=True)

                total_pred.append(pred.detach().cpu())
                total_exp.append(exp)
                t.update()
            total_pred = torch.concat(total_pred, dim=0)
            total_exp  = torch.concat(total_exp,  dim=0)

    return pearson_corr_coef(total_pred, total_exp)

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

    train_loader, valid_loader, test_loader = load_data(
        path = args.data_path, 
        seed = args.seed, 
        batch_size = args.batch_size, 
        num_workers = args.num_workers, 
        target_len = args.target_length)

    model = Enformer.from_hparams(
        dim = args.dim,
        depth = depth,
        heads = args.heads,
        output_heads = dict(human=args.output_heads),
        target_length = args.target_length,
    ).to(args.device)

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=wd)

    # start training
    print('Start training')
    best_pearson_corr_coef = -1
    best_epoch = 0
    for epoch in range(args.epochs):
        train_loss = []
        model.train()
        with tqdm(total=len(train_loader), dynamic_ncols=True) as t:
            t.set_description(f'Epoch: {epoch+1}/{args.epochs}')
            for seq, exp, index in train_loader:
                seq, exp, index = seq.to(args.device), exp.to(args.device), index.to(args.device)
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

        # validate
        mean_valid_pearson_corr_coef = evaluation(model, valid_loader, args.device)

        print("In epoch {}, Train Loss: {:.5}, Valid Pearson_Corr_Coef: {:.5}\n".format(epoch+1, train_loss, mean_valid_pearson_corr_coef))
        if args.use_wandb:
            wandb.log({
                'epoch': epoch+1,
                'Train Loss': train_loss,
                'Valid Pearson_Corr_Coef': mean_valid_pearson_corr_coef,
            })

        if mean_valid_pearson_corr_coef > best_pearson_corr_coef:
            best_pearson_corr_coef = mean_valid_pearson_corr_coef
            best_epoch = epoch + 1
            if args.use_wandb:
                torch.save(model.state_dict(), os.path.join(args.model_save_path, run.id))
            else:
                torch.save(model.state_dict(), os.path.join(args.model_save_path, 'best_model'+str(args.gpu)))
            print("saving model...")
        
    # Use the best model to test
    model.eval()
    if args.use_wandb:
        model.load_state_dict(torch.load(os.path.join(args.model_save_path, run.id)))
    else:
        model.load_state_dict(torch.load(os.path.join(args.model_save_path, 'best_model'+str(args.gpu))))
    mean_test_pearson_corr_coef = evaluation(model, test_loader, args.device)
    print("Best epoch: {}, Test Pearson_Corr_Coef: {:.5}\n".format(best_epoch, mean_test_pearson_corr_coef)) # We Already plus 1 for best epoch in previous code
    if args.use_wandb:
        wandb.log({
            'Best Epoch': best_epoch,
            'Test Pearson_Corr_Coef': mean_test_pearson_corr_coef,
        })

if __name__=='__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_path', default='./data', help='Path of the dataset')
    parser.add_argument("--out_path", default="./output", help="Path to save the output")   
    parser.add_argument("--use_wandb", action='store_true')
    parser.add_argument("--use_sweep", action='store_true')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--batch_size', default=2, type=int)
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--lr', default=1e-4, type=float, help='Learning Rate')
    parser.add_argument('--wd', default=0.0, help='L2 Regularization for Optimizer')
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument("--gpu", type=int, default="0", help="Set GPU Ids : Eg: For CPU = -1, For Single GPU = 0")
    parser.add_argument("--num", type=int, default=0, help='To distinguish different project')

    # Enformer hyperparameters
    parser.add_argument('--dim', default=1536, type=int)
    parser.add_argument('--depth', default=11, type=int, help='Number of transformer blocks')
    parser.add_argument('--heads', default=8, type=int, help='Attention Heads')
    parser.add_argument('--output_heads', default=3740, type=int)
    parser.add_argument('--target_length', default=1920, type=int)

    args = parser.parse_args()
    print(args)

    # prepare the device
    device = torch.device(f"cuda:{args.gpu}" if (torch.cuda.is_available() and args.gpu >= 0) else "cpu")
    print(f"Device is {device}.")
    args.device = device
    
    # prepare the output
    args.out_path = os.path.join(args.out_path, 'enformer')
    os.makedirs(args.out_path, exist_ok=True)
    os.makedirs(os.path.join(args.out_path, 'model'), exist_ok=True)
    args.model_save_path = os.path.join(args.out_path, 'model')

    seed_all(args.seed)

    if args.use_wandb:
        if args.use_sweep:
            project_name = 'enformer'+str(args.num)
            sweep_configuration = {
                'project': project_name,
                'method': 'grid',
                'name': 'enformer_sweep_try',
                'parameters':{
                    'lr':{
                        'values': [1e-3, 5e-4, 1e-4, 5e-5, 1e-5, 5e-6],
                    },
                    'wd':{
                        'values': [1e-3, 5e-4, 1e-4, 5e-5, 1e-5, 5e-6, 1e-6, 5e-7, 1e-7],
                    },
                    'depth':{
                        'distribution': 'int_uniform',
                        'min': 2,
                        'max': 11,
                    }
                }
            }
        args.model_save_path = os.path.join(args.model_save_path, project_name)
        sweep_id = wandb.sweep(sweep=sweep_configuration, project=project_name)
        wandb.agent(sweep_id, function=train)
    else:
        train()







