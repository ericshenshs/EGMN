"""run_egmn.py.

This file is part of the watch-time prediction codebase.
Primary role: run_script.
Executable experiment driver: loads data, trains a model, and reports metrics.
"""

import os
import copy
import torch
import random
import numpy as np
import argparse
from dataloader import KUAIRECDataLoader
from model import EGMN
from utils import eval_mae, eval_xauc, eval_kl
from sklearn.metrics import roc_auc_score

def get_args():
    """get_args.
    
    Parses CLI arguments for this experiment entrypoint.
    The defaults define the baseline reproduction setting for this method.
    
    Args: (none).
    Returns: argparse.Namespace.
    """
    # Notes:
    # - Keep defaults stable for reproducibility (hyperparameters, device, batch size).
    # - If you add new args, propagate them to model construction / loss computation consistently.
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', default='kuairec')
    parser.add_argument('--dataset_path', default='./dataset/')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--bsz', type=int, default=2048)
    parser.add_argument('--log_interval', type=int, default=10)

    parser.add_argument('--alpha', type=float, default=0.1)
    parser.add_argument('--beta', type=float, default=1.0)
    parser.add_argument('--epoch', type=int, default=10)
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--weight_decay', type=float, default=1e-6)
    parser.add_argument('--runs', type=int, default=1, help = 'number of executions to compute the average metrics')
    parser.add_argument('--seed', type=int, default=42)

    args = parser.parse_args()
    return args

def get_loaders(name, dataset_path, device, bsz):
    """get_loaders.
    
    Builds the dataset artifact path and returns a dataset-specific DataLoader wrapper.
    
    Args: name, dataset_path, device, bsz.
    Returns: KUAIRECDataLoader-like object with train/test loaders and .description.
    """
    # Notes:
    # - Encodes the on-disk dataset layout assumption: dataset_path/dataset_name/{name}_data.pkl.
    # - Returns a loader exposing .description schema, used to size embeddings and choose feature types.
    path = os.path.join(dataset_path, name, "{}_data.pkl".format(name))
    if name == 'kuairec':
        dataloaders = KUAIRECDataLoader(name, path, device, bsz=bsz)
    else:
        raise ValueError('unkown dataset name: {}'.format(name))
    return dataloaders

def mae_rescale_to_second(dataset, mae):
    """mae_rescale_to_second.
    
    Converts MAE from normalized play_time units back to seconds for reporting.
    The normalization constants mirror preprocessing conventions for each dataset.
    """
    # Notes:
    # - Labels are normalized in preprocessing; this rescales MAE back into seconds for reporting.
    # - Constants are dataset-specific and should match the normalization used in preprocessing.
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
    
    Evaluation loop for the trained model on the test split.
    Collects predictions and computes MAE/XAUC/KL via utils.py.
    
    Args: args, model, dataloaders.
    Returns: None (prints metrics).
    """
    # Notes:
    # - Uses model.eval() + torch.no_grad() to produce deterministic predictions and reduce memory.
    # - Always compute metrics via utils.py to keep cross-method comparisons consistent.
    # - If the model predicts in a transformed space (D2Q/D2CO/TPM), decode back to play_time before scoring.
    model.eval()
    labels, scores, predicts = list(), list(), list()
    with torch.no_grad():
        for _, (features, label) in enumerate(dataloaders['test']):
            y = model.predict(features) 
            duration = features['duration'].squeeze()
            pred = y.squeeze()
            labels.extend(label.tolist())
            scores.extend(pred.tolist())
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
    model = EGMN(dataloaders.description, embed_dim=16, share_mlp_dims=(256, 128, 64), output_mlp_dims=(32, 16), dropout=0.2)
    model = model.to(device)
    model.train()

    # train
    dataloader_train = dataloaders['train']
    # optimizer = torch.optim.Adam(params=model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    optimizer = torch.optim.Adagrad(params=model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    for epoch_i in range(1, args.epoch + 1):
        model.train()
        epoch_loss, epoch_nll, epoch_reg, epoch_entropy = 0.0, 0.0, 0.0, 0.0
        total_loss, total_nll, total_reg, total_entropy = 0.0, 0.0, 0.0, 0.0
        total_iters = len(dataloader_train) 
        for i, (features, label) in enumerate(dataloader_train):
            pi, lambda_, mu, sigma = model(features)
            duration = features['duration'].squeeze()
            nll_loss, reg_loss, entropy_loss = model.loss(label.float(), pi, lambda_, mu, sigma, features['duration'].view(-1, 1))
            loss = nll_loss + args.alpha * entropy_loss + args.beta * reg_loss 

            model.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item(); epoch_nll += nll_loss.item(); epoch_reg += reg_loss.item(); epoch_entropy += entropy_loss.item()
            total_loss += loss.item(); total_nll += nll_loss.item(); total_reg += reg_loss.item(); total_entropy += entropy_loss.item()
            if (i + 1) % 10 == 0:
                print("    Iter {}/{} loss: {:.7f}, epoch nll: {:.7f}, epoch reg: {:.7f}, epoch entropy: {:.7f}".format(i + 1, total_iters + 1, total_loss/args.log_interval, total_nll/args.log_interval, total_reg/args.log_interval, total_entropy/args.log_interval), end='\r')
                total_loss, total_nll, total_reg = 0, 0, 0
        print("Epoch {}/{} average Loss: {:.7f}, nll: {:.7f}, reg: {:.7f}, entropy: {:.7f}".format(epoch_i, args.epoch, epoch_loss/total_iters, epoch_nll/total_iters, epoch_reg/total_iters, epoch_entropy/total_iters))
    test(args, model, dataloaders)
    
