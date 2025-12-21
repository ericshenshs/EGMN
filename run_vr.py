import os
import copy
import torch
import random
import numpy as np
import argparse
from dataloader import KUAIRECDataLoader
from model import WideAndDeep
from utils import eval_mae, eval_xauc, eval_kl

def get_args():
    """get_args.
    
    Defines CLI arguments for the experiment (dataset path, device, hyperparameters).
    Keeping all knobs here makes runs reproducible and easy to compare.
    """
    # -------------------------------------------------------------------------
    # Detailed developer notes (added for repository documentation):
    # - Role in pipeline: preprocessing -> dataloader -> model -> training script -> metrics
    # - Contracts: input keys/shapes, dtype expectations, device placement, masking rules
    # - Common pitfalls: silent dtype casting, shape mismatches, normalization differences
    # - If you change this code, re-run the corresponding run_*.py training to validate
    # -------------------------------------------------------------------------
    # Function-specific notes:
    # - This script is the experiment driver; it defines training loop, optimizer, and evaluation.
    # - Keep loss/metric computation consistent across baselines to ensure fair comparisons.
    # - Central place to document hyperparameters and provide reproducible defaults.
    # - Readability: keep variable names aligned with math (e.g., pi, mu, sigma) and comment units/scales.
    # - Testing: if you modify logic, validate with a tiny batch and confirm shapes/dtypes.
    # -------------------------------------------------------------------------
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', default='kuairec')
    parser.add_argument('--dataset_path', default='./dataset/')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--bsz', type=int, default=2048)
    parser.add_argument('--log_interval', type=int, default=10)

    parser.add_argument('--epoch', type=int, default=10)
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--weight_decay', type=float, default=1e-6)
    parser.add_argument('--seed', type=int, default=42)

    args = parser.parse_args()
    return args

def get_loaders(name, dataset_path, device, bsz):
    """get_loaders.
    
    Constructs dataset paths and instantiates the appropriate DataLoader wrapper.
    """
    # -------------------------------------------------------------------------
    # Detailed developer notes (added for repository documentation):
    # - Role in pipeline: preprocessing -> dataloader -> model -> training script -> metrics
    # - Contracts: input keys/shapes, dtype expectations, device placement, masking rules
    # - Common pitfalls: silent dtype casting, shape mismatches, normalization differences
    # - If you change this code, re-run the corresponding run_*.py training to validate
    # -------------------------------------------------------------------------
    # Function-specific notes:
    # - This script is the experiment driver; it defines training loop, optimizer, and evaluation.
    # - Keep loss/metric computation consistent across baselines to ensure fair comparisons.
    # - Constructs dataset paths and selects the correct DataLoader implementation.
    # - Readability: keep variable names aligned with math (e.g., pi, mu, sigma) and comment units/scales.
    # - Testing: if you modify logic, validate with a tiny batch and confirm shapes/dtypes.
    # -------------------------------------------------------------------------
    path = os.path.join(dataset_path, name, "{}_data.pkl".format(name))
    if name == 'kuairec':
        dataloaders = KUAIRECDataLoader(name, path, device, bsz=bsz)
    else:
        raise ValueError('unkown dataset name: {}'.format(name))
    return dataloaders

def mae_rescale_to_second(dataset, mae):
    """mae_rescale_to_second.
    
    Rescales MAE from normalized label space back into seconds (dataset-specific).
    Run scripts normalize play_time/duration; this utility restores human-readable units.
    
    Args: dataset, mae.
    """
    # -------------------------------------------------------------------------
    # Detailed developer notes (added for repository documentation):
    # - Role in pipeline: preprocessing -> dataloader -> model -> training script -> metrics
    # - Contracts: input keys/shapes, dtype expectations, device placement, masking rules
    # - Common pitfalls: silent dtype casting, shape mismatches, normalization differences
    # - If you change this code, re-run the corresponding run_*.py training to validate
    # -------------------------------------------------------------------------
    # Function-specific notes:
    # - This script is the experiment driver; it defines training loop, optimizer, and evaluation.
    # - Keep loss/metric computation consistent across baselines to ensure fair comparisons.
    # - Readability: keep variable names aligned with math (e.g., pi, mu, sigma) and comment units/scales.
    # - Testing: if you modify logic, validate with a tiny batch and confirm shapes/dtypes.
    # -------------------------------------------------------------------------
    if dataset == 'kuairec':
        return mae * 999639 / 1000
    elif dataset == 'wechat':
        return mae * 20840
    elif dataset == 'cikm16':
        return (mae *(6000-31) + 31) / 1000
    else:
        raise ValueError('unkown dataset name: {}'.format(dataset))

def test(args, model, dataloaders):
    """test.
    
    Auto-generated function documentation for run_script module.
    See inline comments for data-flow assumptions (shapes/dtypes) and pipeline role.
    
    Args: args, model, dataloaders.
    """
    # -------------------------------------------------------------------------
    # Detailed developer notes (added for repository documentation):
    # - Role in pipeline: preprocessing -> dataloader -> model -> training script -> metrics
    # - Contracts: input keys/shapes, dtype expectations, device placement, masking rules
    # - Common pitfalls: silent dtype casting, shape mismatches, normalization differences
    # - If you change this code, re-run the corresponding run_*.py training to validate
    # -------------------------------------------------------------------------
    # Function-specific notes:
    # - This script is the experiment driver; it defines training loop, optimizer, and evaluation.
    # - Keep loss/metric computation consistent across baselines to ensure fair comparisons.
    # - Always run eval under no_grad() and model.eval() to disable dropout/bn updates.
    # - Convert tensors to CPU numpy only at the boundary to avoid device sync overhead.
    # - Readability: keep variable names aligned with math (e.g., pi, mu, sigma) and comment units/scales.
    # - Testing: if you modify logic, validate with a tiny batch and confirm shapes/dtypes.
    # -------------------------------------------------------------------------
    model.eval()
    labels, scores, predicts = list(), list(), list()
    with torch.no_grad():
        for _, (features, label) in enumerate(dataloaders['test']):
            y = model(features) /20
            labels.extend(label.tolist())
            scores.extend(y.tolist())
    labels, scores = np.array(labels), np.array(scores)
    mae, xauc, kl = eval_mae(labels, scores), eval_xauc(labels, scores), eval_kl(labels, scores)
    mae = mae_rescale_to_second(args.dataset_name, mae)
    print("test result | MAE: {:.7f} | XAUC: {:.7f} | KL: {:.7f}".format(mae, xauc, kl))

if __name__ == '__main__':
    args = get_args()
    if args.seed > -1:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
    res = {}
    torch.cuda.empty_cache()

    device = torch.device(args.device)

    dataloaders = get_loaders(args.dataset_name, args.dataset_path, device, args.bsz)
    model = WideAndDeep(dataloaders.description, embed_dim=16, mlp_dims=(512, 256, 128, 64, 32), dropout=0.0)
    model = model.to(device)

    # train
    dataloader_train = dataloaders['train']
    model.train()
    # criterion = torch.nn.BCELoss()
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    for epoch_i in range(1, args.epoch + 1):
        model.train()
        epoch_loss = 0.0
        total_loss = 0
        total_iters = len(dataloader_train) 
        for i, (features, label) in enumerate(dataloader_train):
            y = model(features)
            loss = criterion(y,20 * label.float())
            model.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            total_loss += loss.item()
            if (i + 1) % 10 == 0:
                print("    Iter {}/{} loss: {:.7f}".format(i + 1, total_iters + 1, total_loss/args.log_interval), end='\r')
                total_loss = 0
        print("Epoch {}/{} average Loss: {:.7f}".format(epoch_i, args.epoch, epoch_loss/total_iters))
    test(args, model, dataloaders)
    
