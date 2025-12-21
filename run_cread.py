"""run_cread.py.

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
from model import Cread
from utils import eval_mae, eval_xauc, eval_kl

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

    parser.add_argument('--epoch', type=int, default=10)
    parser.add_argument('--restore_w', type=float, default=1.0)
    parser.add_argument('--ord_w', type=float, default=0.00002)
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--weight_decay', type=float, default=1e-6)
    parser.add_argument('--bkt_num', type=float, default=50)
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

def discretize_time_label(playtime, split_nodes):
    """discretize_time_label.
    
    CREAD label transform helper.
    discretize_time_label() creates M binary threshold labels; restore_time_label() reconstructs a scalar.
    """
    # Notes:
    # - Converts scalar play_time into M binary threshold labels: play_time > split_nodes[m].
    # - Provides ordinal supervision for each head of the Cread model.
    playtime = playtime.reshape([-1, 1, 1])
    split_nodes = split_nodes.reshape([1, 1, -1])
    cmp_tensor = playtime > split_nodes
    binary_labels = torch.where(cmp_tensor, torch.ones_like(cmp_tensor), torch.zeros_like(cmp_tensor)) # [bsz, 1, M]
    return torch.squeeze(binary_labels)

def restore_time_label(preds, split_nodes):
    """restore_time_label.
    
    CREAD label transform helper.
    discretize_time_label() creates M binary threshold labels; restore_time_label() reconstructs a scalar.
    """
    # Notes:
    # - Converts threshold probabilities into an expected play_time by weighting bucket sizes.
    # - This produces a scalar prediction comparable to other baselines.
    append_split_nodes = torch.concat((torch.tensor([0]).to(torch.float32).to(split_nodes.device), split_nodes)) 
    left_split_nodes, right_split_nodes = append_split_nodes[:-1], append_split_nodes[1:]
    bkt_size_list = right_split_nodes - left_split_nodes
    return torch.sum(preds * bkt_size_list.view([1, -1]), dim=1) # [bsz]

def get_ord_criterion(preds):
    """get_ord_criterion.
    
    Ordinal consistency penalty for threshold heads.
    Encourages predicted threshold probabilities to be non-increasing as the threshold grows.
    """
    # Notes:
    # - Penalizes violations of monotonicity across threshold heads (encourages ordered outputs).
    # - Implements sum(max(pred[j+1] - pred[j], 0)) as in the original code.
    left_preds, right_preds = preds[:,:-1], preds[:,1:]
    return torch.sum(torch.clamp(right_preds - left_preds, min=0.0))

def get_split_nodes(all_labels, M, alpha):
    """get_split_nodes.
    
    Computes discretization thresholds (split nodes) for CREAD.
    The thresholds determine how the continuous label is converted into ordinal supervision.
    """
    # Notes:
    # - Computes split nodes (thresholds) for discretizing labels into ordinal bins.
    # - Grid search selects alpha to balance within-bin and between-bin objectives used by CREAD.
    split_nodes = []
    cdf_list = []
    for m in range(1, M+1):
        z = m / M
        gamma = (1 - np.exp(-alpha*z)) / (1 - np.exp(-alpha))
        split_nodes.append(torch.quantile(all_labels, gamma))
        cdf_list.append(gamma)
    return torch.tensor(split_nodes), torch.tensor(cdf_list)

def cread_grid_search(dataloader_train, M):
    """cread_grid_search.
    
    Computes discretization thresholds (split nodes) for CREAD.
    The thresholds determine how the continuous label is converted into ordinal supervision.
    """
    # Notes:
    # - Computes split nodes (thresholds) for discretizing labels into ordinal bins.
    # - Grid search selects alpha to balance within-bin and between-bin objectives used by CREAD.
    all_labels = []
    for (_, label) in dataloader_train:
        all_labels.append(label) 
    all_labels = torch.concat(all_labels).to(torch.float32)
    alpha_search_space = list(np.arange(0.001, 5.0, 0.1))
    beta_search_space = [50] 
    best_loss, best_alpha, best_beta, best_split = None, None, None, None
    print("Strat Cread Split Nodes Search....l")
    for alpha in alpha_search_space:
        for beta in beta_search_space:
            split_nodes, cdf_list = get_split_nodes(all_labels, M, alpha)
            split_nodes_left, split_nodes_right = torch.cat([torch.tensor([0]), split_nodes[:-1]]), split_nodes
            cdf_list_left, cdf_list_right = torch.cat([torch.tensor([0]), cdf_list[:-1]]), cdf_list
            A_w = torch.sum(torch.pow(cdf_list_right - cdf_list_left, 2)) * torch.sum(torch.pow(split_nodes_right - split_nodes_left, 2)/(cdf_list_right - cdf_list_left))
            A_b = torch.sum(torch.pow(cdf_list_right - cdf_list_left, 2)) * torch.sum(torch.pow(split_nodes_right - split_nodes_left, 2))
            A_loss = A_w + beta * A_b
            print("Searching | alpha={:.7f}, beta={:.7f}: A_loss={:.7f},A_w={:.7f},A_b={:.7f} ".format(alpha, beta, A_loss, A_w, A_b))
            if (best_loss == None) or (best_loss > A_loss):
                best_loss, best_alpha, best_beta, best_split = A_loss, alpha, beta, split_nodes
    print("Cread Search Complete! Best Loss is {:.7f}, Best Alpha is {:.7f}, Best Beta is {:.7f}.".format(best_loss, best_alpha, best_beta))
    return best_split

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
            preds = model(features)
            y = restore_time_label(preds, split_nodes)
            labels.extend(label.tolist())
            scores.extend(y.tolist())
    labels, scores = np.array(labels), np.array(scores)
    mae, xauc, kl = eval_mae(labels, scores), eval_xauc(labels, scores), eval_kl(labels, scores)
    mae = mae_rescale_to_second(args.dataset_name, mae)
    print("test result | MAE: {:.7f} | XAUC: {:.7f} | KL: {:.7f}".format(mae, xauc, kl))

if __name__ == '__main__':
    args = get_args()
    # Set random seeds to make runs reproducible (within the limits of GPU nondeterminism).
    if args.seed > -1:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
    res = {}
    # Clear cached GPU memory (useful when running multiple scripts sequentially).
    torch.cuda.empty_cache()

    device = torch.device(args.device)

    # consturct DataLoader
    dataloaders = get_loaders(args.dataset_name, args.dataset_path, device, args.bsz)

    # discretization strategy
    split_nodes = cread_grid_search(dataloaders['train'], args.bkt_num).to(device)
    print("split node list: ", split_nodes)
    M = split_nodes.shape[0]

    # construct model
    model = Cread(dataloaders.description, embed_dim=16, share_mlp_dims=(512, 256, 128),  output_mlp_dims=(64, 32), head_num=M, dropout=0.0)
    model = model.to(device)

    # train
    dataloader_train = dataloaders['train']
    model.train()
    bce_criterion = torch.nn.BCELoss()
    huber_criterion  = torch.nn.HuberLoss()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    for epoch_i in range(1, args.epoch + 1):
        model.train()
        epoch_loss, epoch_loss_bce, epoch_loss_restore, epoch_loss_ord = 0.0, 0.0, 0.0, 0.0
        total_loss, total_loss_bce, total_loss_restore, total_loss_ord = 0.0, 0.0, 0.0, 0.0
        total_iters = len(dataloader_train) 
        for i, (features, label) in enumerate(dataloader_train):
            preds = model(features) # [bsz, M]
            binary_labels = discretize_time_label(label, split_nodes)
            restore_pred = restore_time_label(preds, split_nodes)
            loss_bce = bce_criterion(preds, binary_labels.float())
            loss_restore = huber_criterion(restore_pred, label.float())
            loss_ord = get_ord_criterion(preds)
            loss = loss_bce + args.restore_w * loss_restore + args.ord_w * loss_ord
            model.zero_grad()
            # Backpropagate through the model to accumulate gradients in parameters.
            loss.backward()
            # Apply one optimization step (parameter update).
            optimizer.step()
            epoch_loss += loss.item(); epoch_loss_bce += loss_bce.item(); epoch_loss_restore += args.restore_w * loss_restore.item(); epoch_loss_ord += args.ord_w * loss_ord.item()
            total_loss += loss.item(); total_loss_bce += loss_bce.item(); total_loss_restore += args.restore_w * loss_restore.item(); total_loss_ord += args.ord_w * loss_ord.item()
            if (i + 1) % 10 == 0:
                print("    Iter {}/{} loss: {:.7f}, loss_bce: {:.7f}, loss_restore: {:.7f}, loss_ord: {:.7f}  ".format(i + 1, total_iters + 1, total_loss/args.log_interval, total_loss_bce/args.log_interval, total_loss_restore/args.log_interval, total_loss_ord/args.log_interval ), end='\r')
                total_loss = 0
        print("Epoch {}/{} average Loss: {:.7f}, loss_bce: {:.7f}, loss_restore: {:.7f}, loss_ord: {:.7f}".format(epoch_i, args.epoch, epoch_loss/total_iters, epoch_loss_bce/total_iters, epoch_loss_restore/total_iters, epoch_loss_ord/total_iters))
    test(args, model,dataloaders)
