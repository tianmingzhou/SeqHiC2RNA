import torch
import torch.optim as optim
import torch.nn.functional as F
import argparse
import numpy as np
import os
import wandb
import shutil

from utils.utils import seed_all
from utils.dataset import load_data_bulk_pretrain
from algo.Enformer_pretrain import Enformer
from algo.module import pearson_corr_coef, poisson_loss
from tqdm import tqdm


def evaluation(model, data_loader, device):
    model.eval()
    with torch.no_grad():
        with tqdm(total=len(data_loader), dynamic_ncols=True) as t:
            t.set_description('Evaluation: ')
            total_pred = []
            total_exp = []
            for seq, exp in data_loader:
                seq = seq.to(device)
                pred = model(seq, head='human')

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

    train_loader, valid_loader, test_loader = load_data_bulk_pretrain(
        path = args.data_path, 
        pretrain_vec_path=args.pretrain_vec_path,
        seed = args.seed, 
        batch_size = args.batch_size, 
        num_workers = args.num_workers, 
        target_len = args.target_length,)

    model = Enformer.from_hparams(
        dim = 1536,
        depth = depth,
        heads = 8,
        output_heads = dict(human=args.output_heads),
        target_length = args.target_length,
        dim_divisible_by = 1536 / 12,
        pool_after_transformer = False,
        use_checkpointing = False,
    ).to(args.device)

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
            for seq, exp in train_loader:
                seq, exp = seq.to(args.device), exp.to(args.device)
                pred = model(seq, head='human')

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
    parser.add_argument('--pretrain_vec_path', default='./data', help='Path of the pretrain vector')
    parser.add_argument('--pretrain_path', default='./algo/pretrain', help='Path to save the pretrain parameter')
    parser.add_argument('--use_wandb', action='store_true')
    parser.add_argument('--use_sweep', action='store_true')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--batch_size', default=2, type=int)
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--lr', default=1e-4, type=float, help='Learning Rate')
    parser.add_argument('--wd', default=0.0, help='L2 Regularization for Optimizer')
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--gpu', type=int, default='0', help='Set GPU Ids : Eg: For CPU = -1, For Single GPU = 0')
    parser.add_argument('--num', type=int, default=0, help='To distinguish different sweep')
    parser.add_argument('--early_stop', default=10, type=int, help='Patience for early stop.')

    # Enformer hyperparameters
    parser.add_argument('--depth', default=11, type=int, help='Number of transformer blocks')
    parser.add_argument('--output_heads', default=1, type=int)
    parser.add_argument('--target_length', default=240, type=int)

    # parallelize sweep
    parser.add_argument('--parallelize', action='store_true')
    parser.add_argument('--sweep_id', type=str)

    args = parser.parse_args()
    print(args)

    # prepare the device
    device = torch.device(f"cuda:{args.gpu}" if (torch.cuda.is_available() and args.gpu >= 0) else "cpu")
    print(f"Device is {device}.")
    args.device = device
    
    # prepare the output
    args.out_path = os.path.join(args.out_path, 'enformer_pretrain')
    os.makedirs(args.out_path, exist_ok=True)
    os.makedirs(os.path.join(args.out_path, 'model'), exist_ok=True)
    args.model_save_path = os.path.join(args.out_path, 'model')

    seed_all(args.seed)

    if args.use_wandb:
        if args.use_sweep:
            sweep_name = 'enformer_pretrain_pool_before_transformer'+str(args.num)
            sweep_configuration = {
                'project': 'enformer_pretrain',
                'method': 'random',
                'name': sweep_name,
                'parameters':{
                    'lr':{
                        'values': [1e-3, 5e-4, 1e-4, 5e-5, 1e-5, 5e-6],
                    },
                    'wd':{
                        'values': [1e-3, 5e-4, 1e-4, 5e-5, 1e-5, 5e-6, 1e-6, 5e-7, 1e-7],
                    },
                    'depth':{
                        'distribution': 'int_uniform',
                        'min': 5,
                        'max': 11,
                    }
                }
            }
            args.model_save_path = os.path.join(args.model_save_path, sweep_name)
            if os.path.exists(args.model_save_path):
                shutil.rmtree(args.model_save_path)
            os.mkdir(args.model_save_path)
            sweep_id = wandb.sweep(sweep=sweep_configuration, project='enformer_pretrain')
            wandb.agent(sweep_id, function=train)
        elif args.parallelize:
            args.model_save_path = os.path.join(args.model_save_path, 'enformer_pretrain_pool_before_transformer'+str(args.num))
            if not os.path.exists(args.model_save_path):
                os.mkdir(args.model_save_path)
            wandb.agent(sweep_id=args.sweep_id, function=train)
    else:
        train()







