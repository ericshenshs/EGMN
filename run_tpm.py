import os
import copy
import torch
import random
import numpy as np
import argparse
from dataloader import KUAIRECDataLoader
from model import TPM
from utils import eval_mae, eval_xauc, eval_kl, get_playtime_percentiles_range, get_tree_encoded_value, get_tree_encoded_label, get_tree_classify_loss 


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', default='wechat')
    parser.add_argument('--dataset_path', default='./dataset/')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--bsz', type=int, default=2048)
    parser.add_argument('--log_interval', type=int, default=10)

    parser.add_argument('--wr_bucknum', type=int, default=32)
    parser.add_argument('--epoch', type=int, default=10)
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--weight_decay', type=float, default=1e-6)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--variance_weight', type=float, default=0.0001)
    parser.add_argument('--mse_weight', type=float, default=1)
    parser.add_argument('--tree_cla_weight', type=float, default=1)
    args = parser.parse_args()
    return args


def get_loaders(name, dataset_path, device, bsz):
    path = os.path.join(dataset_path, name, "{}_data.pkl".format(name))
    if name == 'kuairec':
        dataloaders = KUAIRECDataLoader(name, path, device, bsz=bsz)
    else:
        raise ValueError('unkown dataset name: {}'.format(name))
    return dataloaders

def mae_rescale_to_second(dataset, mae):
    if dataset == 'kuairec':
        return mae * 999639 / 1000
    elif dataset == 'wechat':
        return mae * 20840
    elif dataset == 'cikm16':
        return (mae *(6000-31) + 31) / 1000
    else:
        raise ValueError('unkown dataset name: {}'.format(dataset))

def test(args, model, dataloaders):
    model.eval()
    labels, scores, predicts = list(), list(), list()
    with torch.no_grad():
        for _, (features, label) in enumerate(dataloaders['test']):
            y = model(features)
            encoded_y, variance = get_tree_encoded_value(y, args.wr_bucknum, bucket_begins, bucket_ends)
            labels.extend(label.tolist())
            scores.extend(encoded_y.flatten().tolist())
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
    model = TPM(dataloaders.description, class_num=args.wr_bucknum-1 ,embed_dim=16, mlp_dims=(128, 64, 32), dropout=0.0)
    model = model.to(device)

    # get the palytime bucket ranges
    dataloader_train = dataloaders['train']
    bucket_begins, bucket_ends = get_playtime_percentiles_range(dataloader_train, args.wr_bucknum ,args.device)

    # train  
    model.train()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    for epoch_i in range(1, args.epoch + 1):
        model.train()
        epoch_loss = 0.0
        total_loss = 0
        print_tc_loss = 0
        print_mse_loss = 0
        print_var_loss = 0
        total_iters = len(dataloader_train) 
        for i, (features, label) in enumerate(dataloader_train):
            y = model(features)
            encoded_y, variance = get_tree_encoded_value(y, args.wr_bucknum, bucket_begins, bucket_ends)
            encoded_label, bucket_weights = get_tree_encoded_label(label, args.wr_bucknum, bucket_begins, bucket_ends)
            tree_classify_loss = get_tree_classify_loss(encoded_label, bucket_weights, y, args.wr_bucknum)
            mse_loss_fn = torch.nn.MSELoss(reduction='mean')
            mse_loss = mse_loss_fn(encoded_y, label.view(-1,1).float()) 
            loss =  tree_classify_loss * args.tree_cla_weight  + mse_loss * args.mse_weight + variance * args.variance_weight 
            model.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            total_loss += loss.item()
            print_tc_loss += tree_classify_loss.item()
            print_mse_loss += mse_loss.item()
            print_var_loss += variance.item()
            if (i + 1) % 10 == 0:
                print("    Iter {}/{} total_loss: {:.7f}, tc_loss: {:.7f}, mse_loss: {:.7f},var_loss: {:.7f},".format(i + 1, total_iters + 1, total_loss/args.log_interval, print_tc_loss/args.log_interval, print_mse_loss/args.log_interval, print_var_loss/args.log_interval), end='\r')
                total_loss = 0
                print_tc_loss = 0
                print_mse_loss = 0
                print_var_loss = 0
        print("Epoch {}/{} average Loss: {:.7f}".format(epoch_i, args.epoch, epoch_loss/total_iters))
    test(args, model, dataloaders)
    
