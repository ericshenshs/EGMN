"""utils.py.

This file is part of the watch-time prediction codebase.
Primary role: utils.
Shared metrics and helper functions used by multiple run scripts/models.
"""

import math
import numpy as np
import matplotlib.pyplot as plt
import os
import math
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
 
import torch
import numpy as np
 
def get_playtime_percentiles_range(dataloader, wr_bucknum, _device):
    """get_playtime_percentiles_range.
    
    Computes percentile-based bucket boundaries from training labels.
    Used by TPM to define bucket ranges (begins/ends) for encoding and decoding.
    """
    # Notes:
    # - Aggregates all labels from a dataloader and computes percentile bucket boundaries.
    # - Bucket boundaries are used by TPM to define leaf bucket midpoints and encoding ranges.
    all_play_time = []
    for _, (_, label) in enumerate(dataloader):
        play_time = label
        all_play_time.append(play_time)
    all_play_time = torch.cat(all_play_time, dim=0) 
    play_time_np = all_play_time.cpu().numpy()
    percen_value = np.percentile(play_time_np, np.linspace(0.0, 100.0, num=wr_bucknum + 1).astype(np.float32)).tolist()
    bucket_begins = torch.tensor(percen_value[:-1], dtype=torch.float32, device=_device).unsqueeze(0)
    bucket_ends = torch.tensor(percen_value[1:], dtype=torch.float32, device=_device).unsqueeze(0)
    return bucket_begins, bucket_ends
 
 
def get_tree_classify_loss(label_dict, weight_dict, label_encoding_predict, tree_num_intervals=32):
    """get_tree_classify_loss.
    
    Computes average BCE-with-logits loss over all internal tree nodes for TPM.
    Each node corresponds to a binary decision; weights mask samples not applicable to that node.
    """
    # Notes:
    # - Computes average BCE-with-logits loss over all internal tree nodes.
    # - Weights allow masking out labels that are outside an interval's validity range.
    auxiliary_loss_ = 0.0
    height = int(math.log2(tree_num_intervals)) 
    for i in range(height):
        for j in range(2**i):
            interval_label = label_dict[1000*i + j].reshape(-1, 1)  
            interval_weight = weight_dict[1000*i + j].reshape(-1, 1) 
            interval_preds = label_encoding_predict[:, 2**i - 1 + j].view(-1,1)
            interval_loss = F.binary_cross_entropy_with_logits(interval_preds, interval_label, weight=interval_weight)
            auxiliary_loss_ += interval_loss  
    final_loss = auxiliary_loss_ / (tree_num_intervals - 1.0)
    return final_loss.float()
 
def get_tree_encoded_label(label,tree_num_intervals, begins, ends, name="label_encoding"):
    """get_tree_encoded_label.
    
    Encodes a scalar label into binary decisions at each internal node of a complete binary tree.
    Also computes weights that restrict supervision to the node's interval.
    """
    # Notes:
    # - Builds a dict of binary labels (left/right decisions) for each internal node in the tree.
    # - Also builds weights to ignore samples outside the node's interval during training.
    label_dict = {}
    weight_dict = {}
    height = int(math.log2(tree_num_intervals))
    for i in range(height):
        for j in range(2**i):
            temp_ind = max(int(tree_num_intervals * 1.0 / (2**i) * j) - 1, 0)
            
            if j == 0:
                weight_temp = torch.where(label < begins[:, temp_ind].reshape(-1, 1), torch.zeros_like(label), torch.ones_like(label))
            else:
                weight_temp = torch.where(label < ends[:, temp_ind].reshape(-1, 1), torch.zeros_like(label), torch.ones_like(label))
            
            temp_ind = max(int(tree_num_intervals * 1.0 / (2**i) * (j + 1)) - 1, 0)
            weight_temp = torch.where(label < ends[:, temp_ind].reshape(-1, 1), weight_temp, torch.zeros_like(label))
            
            temp_ind = max(int(tree_num_intervals * (1.0 / (2**i) * j + 1.0 / (2**(i + 1)))) - 1, 0)
            label_temp = torch.where(label < ends[:, temp_ind].reshape(-1, 1), torch.zeros_like(label), torch.ones_like(label))
            
            label_dict[1000 * i + j] = label_temp
            weight_dict[1000 * i + j] = weight_temp
 
    return label_dict, weight_dict
 
 
def get_tree_encoded_value(label_encoding_predict, tree_num_intervals, begins, ends, name="encoded_playtime"):
    """get_tree_encoded_value.
    
    Decodes TPM node probabilities into an expected play_time bucket midpoint.
    Also returns a variance-like summary for uncertainty regularization.
    """
    # Notes:
    # - Decodes node probabilities into a distribution over leaves and returns expected bucket midpoint.
    # - Also returns a variance-like summary to penalize overly uncertain predictions during training.
    height = int(math.log2(tree_num_intervals))
    encoded_prob_list = []
    
    temp_encoded_playtime = (begins + ends) / 2.0  
    encoded_playtime = temp_encoded_playtime  
    
    batch_size = label_encoding_predict.size(0)
    device = label_encoding_predict.device
 
    for i in range(tree_num_intervals):
        temp = torch.zeros(batch_size, dtype=torch.float32, device=device)
        cur_code = 2 ** height - 1 + i
        
        for j in range(1, height + 1):
            classifier_branch = cur_code % 2     
            classifier_idx = (cur_code - 1) // 2  
            
            probs = label_encoding_predict[:, classifier_idx]
            condition = torch.tensor(classifier_branch == 1, dtype=torch.bool, device=device)
            log_p = torch.where(condition, torch.log(1.0 - probs + 0.00001), torch.log(probs + 0.00001))
            temp += log_p
            
            cur_code = classifier_idx
        encoded_prob_list.append(temp)
    encoded_prob = torch.exp(torch.stack(encoded_prob_list, dim=1)) 
    encoded_playtime = torch.sum(temp_encoded_playtime * encoded_prob, dim=-1, keepdim=True)
    
    e_x2 = torch.sum((encoded_playtime ** 2) * encoded_prob, dim=-1, keepdim=True)
    square_of_e_x = encoded_playtime ** 2
    var = torch.sqrt(torch.abs(e_x2 - square_of_e_x) + 1e-8) 
    
    return encoded_playtime.float(), torch.sum(var).float()
 
 
class InversePairsCalc:
    """InversePairsCalc.
    
    Counts inversions in a list (used to compute XAUC efficiently).
    The inversion count equals the number of out-of-order label pairs after sorting by score.
    """
    # Notes:
    # - Used by eval_xauc(): counts inversions in the label sequence after sorting by prediction.
    # - Inversion count provides an O(n log n) alternative to enumerating all pairs for ranking evaluation.
    def InversePairs(self, data):
        """InversePairs.
        
        Counts inversions (out-of-order pairs) in a list using a merge-sort style algorithm.
        Used by eval_xauc() to score ranking consistency efficiently.
        """
        # Notes:
        # - Merge-sort based inversion counting (O(n log n)); used to compute XAUC efficiently.
        if not data :
            return False
        if len(data)==1 :
            return 0
        def merge(tuple_fir,tuple_sec):
            """merge.
            
            Internal helper for inversion-counting merge sort.
            Returns a sorted list and the inversion count accumulated during merging.
            """
            # Notes:
            # - Merge-sort based inversion counting (O(n log n)); used to compute XAUC efficiently.
            array_before = tuple_fir[0]
            cnt_before = tuple_fir[1]
            array_after = tuple_sec[0]
            cnt_after = tuple_sec[1]
            cnt = cnt_before+cnt_after
            flag = len(array_after)-1
            array_merge = []
            for i in range(len(array_before)-1,-1,-1):
                while array_before[i]<=array_after[flag] and flag>=0 :
                    array_merge.append(array_after[flag])
                    flag -= 1
                if flag == -1 :
                    break
                else:
                    array_merge.append(array_before[i])
                    cnt += (flag+1)
            if flag == -1 :
                for j in range(i,-1,-1):
                    array_merge.append(array_before[j])
            else:
                for j in range(flag ,-1,-1):
                    array_merge.append(array_after[j])
            return array_merge[::-1],cnt
 
        def mergesort(array):
            """mergesort.
            
            Internal helper for inversion-counting merge sort.
            Returns a sorted list and the inversion count accumulated during merging.
            """
            # Notes:
            # - Merge-sort based inversion counting (O(n log n)); used to compute XAUC efficiently.
            if len(array)==1:
                return (array,0)
            cut = math.floor(len(array)/2)
            tuple_fir=mergesort(array[:cut])
            tuple_sec=mergesort(array[cut:])
            return merge(tuple_fir, tuple_sec)
        return mergesort(data)[1]
 
def eval_xauc(labels, pres):
    """eval_xauc.
    
    Computes XAUC: fraction of correctly ordered label pairs after sorting by prediction.
    Implementation: sort by prediction, then count inversions in the label sequence.
    """
    # Notes:
    # - Sort predictions descending and count inversions in labels to compute pairwise ordering accuracy.
    # - Equivalent to the fraction of correctly ordered pairs among all possible pairs.
    label_preds = zip(labels.reshape(-1), pres.reshape(-1))
    # Sort by prediction and count label inversions for XAUC (ranking consistency).
    sorted_label_preds = sorted(
        label_preds, key=lambda lc: lc[1], reverse=True)
    label_preds_len = len(sorted_label_preds)
    pairs_cnt = label_preds_len * (label_preds_len-1) / 2
    
    labels_sort = [ele[0] for ele in sorted_label_preds]
    S=InversePairsCalc()
    total_positive = S.InversePairs(labels_sort)
    xauc = total_positive / pairs_cnt
    return xauc

def eval_auc(labels, pres):
    """eval_auc.
    
    ROC-AUC via sklearn. Only meaningful for binary labels.
    """
    # Notes:
    # - Standard ROC-AUC via sklearn; only meaningful if labels are binary/treated as binary.
    auc = roc_auc_score(labels, pres)
    return  auc

def eval_mae(labels, scores):
    """eval_mae.
    
    Mean absolute error on numpy arrays.
    """
    # Notes:
    # - Simple mean(|label - pred|) on numpy arrays; used for primary regression reporting.
    return np.mean(np.abs(labels - scores))

def eval_kl(samples_p, samples_q, bins=100, epsilon=1e-10):
    # 计算直方图分箱概率
    """eval_kl.
    
    Approximates KL divergence between two sample distributions using histogram binning.
    Clips probabilities by epsilon to avoid log(0) and division by zero.
    """
    # Notes:
    # - Histogram-based KL approximation; uses shared bins and epsilon clipping for stability.
    hist_p, bin_edges = np.histogram(samples_p, bins=bins, density=True)
    hist_q, _ = np.histogram(samples_q, bins=bin_edges, density=True)
    
    # 计算每个分箱的宽度（用于归一化）
    bin_width = np.diff(bin_edges)
    hist_p = hist_p * bin_width  # 转为概率质量
    hist_q = hist_q * bin_width
    
    # 防止零概率
    hist_p = np.clip(hist_p, epsilon, None)
    hist_q = np.clip(hist_q, epsilon, None)
    
    # 计算KL散度
    kl = np.sum(hist_p * np.log(hist_p / hist_q))
    return kl 
 