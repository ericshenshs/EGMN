"""model/egmn.py.

This file is part of the watch-time prediction codebase.
Primary role: model.
Defines one or more PyTorch modules that map feature dicts to predictions.
"""

import torch
import torch.nn as nn
import torch.distributions as D
import torch.nn.functional as F
import numpy as np

from model.layers import FactorizationMachine, MultiLayerPerceptron, DurationMultiLayerPerceptron

class EGMN(torch.nn.Module):

    """EGMN (Exponential-Gaussian Mixture Network).
    
    Distributional watch-time predictor with a mixture head:
    - Component 0: Exponential (captures heavy mass near 0 / quick-skip).
    - Components 1..K: (truncated-at-0) Gaussians (capture long tail / multiple modes).
    
    Forward returns mixture parameters; predict() returns mixture-mean scalar.
    """
    # Notes:
    # - Mixture head is tailored to watch-time: Exponential captures quick-skip spike; Gaussians capture long tail.
    # - forward() emits mixture parameters; loss() computes NLL + auxiliary terms; predict() returns mixture mean used for MAE.
    # - Assumes normalized labels (play_time) consistent with preprocessing; MAE is rescaled in run scripts.
    def __init__(self, description, embed_dim, share_mlp_dims, output_mlp_dims, dropout):
        """__init__.
        
        Stores schema-derived feature metadata and builds submodules.
        description determines which features are embedded (spr/seq) vs treated as continuous (ctn).
        """
        # Notes:
        # - Parses description schema to decide which features are embedded vs treated as continuous.
        # - Calls build() to allocate submodules sized to vocab sizes and embedding dimension.
        super().__init__()
        self.features = {name: (size, type) for name, size, type in description if (type in ["ctn", 'seq', 'spr'])}
        self.build(embed_dim, share_mlp_dims, output_mlp_dims, dropout)
    
    def build(self, embed_dim, share_mlp_dims, output_mlp_dims, dropout):
        """build.
        
        Constructs submodules/parameters based on the dataset description (vocab sizes, feature types).
        """
        # Notes:
        # - Creates per-feature embedding tables for spr/seq and per-feature linear layers for ctn.
        # - Defines the MLP tower/head that converts concatenated feature representations into outputs.
        self.emb_layer = torch.nn.ModuleDict()
        self.ctn_emb_layer = torch.nn.ParameterDict()
        self.ctn_linear_layer = torch.nn.ModuleDict()
        embed_output_dim = 0
        for name, (size, type) in self.features.items():
            if type == 'spr':
                self.emb_layer[name] = torch.nn.Embedding(size, embed_dim)
                embed_output_dim += embed_dim
            elif type == 'ctn':
                self.ctn_linear_layer[name] = torch.nn.Linear(1, 1, bias=False)
                embed_output_dim += 1
            elif type == 'seq':
                self.emb_layer[name] = torch.nn.Embedding(size, embed_dim)
                embed_output_dim += embed_dim
            else:
                raise ValueError('unkown feature type: {}'.format(type))


        self.share_mlp = MultiLayerPerceptron(embed_output_dim, share_mlp_dims, dropout, output_layer=False)

        hidden_dim = share_mlp_dims[-1]# + 1

        # 指数分布参数分支（快滑峰）
        self.lambda_layer = nn.Sequential(
            nn.Linear(hidden_dim, 1),
            nn.Softplus(beta=0.5)
        )

        comp_num = 10
        self.mixture_logits = nn.Linear(hidden_dim, comp_num+1)
        self.gauss_mu = nn.Sequential(
            nn.Linear(hidden_dim, comp_num),
            # MultiLayerPerceptron(hidden_dim, output_mlp_dims, dropout, output_layer=True),
            nn.Softplus()
        )
        self.gauss_sigma = nn.Sequential(
            nn.Linear(hidden_dim, comp_num),
            # MultiLayerPerceptron(hidden_dim, output_mlp_dims, dropout, output_layer=True),
            nn.Softplus()
        )
        return

    def init(self):
        """init.
        
        Optional parameter initialization helper (uniform in this repo).
        Not all run scripts call this explicitly.
        """
        # Notes:
        # - Optional uniform initialization helper used by some experiments; not always called.
        for param in self.parameters():
            torch.nn.init.uniform_(param, -0.01, 0.01)

    def forward(self, x_dict):
        """forward.
        
        Computes model outputs from a feature dict.
        The exact output shape is model-specific and must match the run script loss.
        
        Args: self, x_dict.
        """
        # Notes:
        # - Builds per-feature representations then concatenates them into a single vector per sample.
        # - Sequence features are pooled by masked mean: sum(emb*mask)/sum(mask).
        # - Final output is typically sigmoid-transformed in these baselines (matching their run scripts).
        linears = []
        embs = []
        # Iterate over schema-declared features and build per-feature representations.
        for name, (_, type) in self.features.items():
            x = x_dict[name]
            if type == 'spr':
                embs.append(self.emb_layer[name](x).squeeze(1))
            elif type == 'ctn':
                linears.append(self.ctn_linear_layer[name](x))
            elif type == 'seq':
                seq_emb = self.emb_layer[name](x)
                seq_mask = torch.unsqueeze(x_dict["{}mask".format(name)], dim=2)
                # Masked mean pooling over sequence length (avoid attending to padding).
                embs.append(torch.sum(seq_emb * seq_mask, dim=1) / torch.sum(seq_mask, dim=1))
            else:
                raise ValueError('unkwon feature: {}'.format(name))
        emb = torch.concat(embs + linears, dim=1)

        hidden = self.share_mlp(emb)
        # hidden = torch.concat([hidden, engage_pred.view(-1, 1)], dim=1)

        lambda_ = self.lambda_layer(hidden) + 1e-6
        pi = self.mixture_logits(hidden)  # [batch, componet]
        
        mu = self.gauss_mu(hidden) + 1/lambda_# [batch, component]
        # mu = torch.cumsum(mu, dim=1) + 1/lambda_
        sigma = self.gauss_sigma(hidden) + 1e-6  # [batch, component]
        
        return pi, lambda_, mu, sigma

    def loss(self, y_true, pi, lambda_, mu, sigma, duration):
        """loss.
        
        Computes three losses for EGMN training:
        - nll_loss: negative log-likelihood under the mixture distribution (main objective).
        - reg_loss: L1 between mixture-mean prediction and y_true (stabilizes and improves MAE).
        - entropy_loss: sum(p * log p) (negative entropy); adding it encourages higher entropy.
        """
        # Notes:
        # - This block explains intent, invariants, and key assumptions specific to this function.
        batch_size = y_true.shape[0]
        y_true = y_true.view(-1, 1)

        # 指数分布（快滑峰）
        exp_dist = D.Exponential(rate=lambda_.view(-1))
        log_prob_short = exp_dist.log_prob(y_true.view(-1)).view(batch_size, 1)

        #高斯分布
        log_prob_all = []
        for comp_idx in range(mu.shape[1]):
            normal_dist = D.Normal(loc=mu[:, comp_idx], scale=sigma[:, comp_idx])
            trunc_min = torch.zeros_like(mu[:, comp_idx])
            prob_long = 1.0 - normal_dist.cdf(trunc_min) # 左侧 <0截断
            log_prob = normal_dist.log_prob(y_true.view(-1)) - torch.log(prob_long + 1e-6) # [batch, comp_nm]
            log_prob_all.append(log_prob.view(-1, 1))
        log_prob_all = torch.concat([log_prob_short] + log_prob_all, dim=1)

        # 混合概率
        mix_probs = torch.softmax(pi, dim=1)

        # sample_w = (1 + y_true * duration)
        # sample_w = torch.where(y_true * video_durations.view(-1, 1) < 0.005, 0.5 * torch.ones_like(y_true), torch.ones_like(y_true) )
        # nll loss
        log_mix_probs = torch.log_softmax(pi, dim=1)
        total_log_prob = torch.logsumexp(
            log_mix_probs + log_prob_all, 
            dim=1, keepdim=True
        )
        nll_loss = -torch.mean(total_log_prob)
            
        # reconstruction loss
        pi = torch.softmax(pi, dim=1)
        pred =  torch.sum(pi * torch.concat([1/lambda_, mu], dim=1), dim=1, keepdim=True)
        reg_loss = F.l1_loss(pred, y_true.float())

        # mixture entropy loss
        entropy_loss = torch.sum(mix_probs * torch.log(mix_probs + 1e-6), dim=1).mean()

        return nll_loss, reg_loss, entropy_loss

    def get_quantile(self, pi, lambda_, mu, sigma, tau=0.5):
        """get_quantile.
        
        Brute-force quantile approximation by scanning a dense grid and matching target CDF.
        Expensive; typically used for analysis rather than training.
        """
        # Notes:
        # - This block explains intent, invariants, and key assumptions specific to this function.
        exp_dist = D.Exponential(rate=lambda_.view(-1, 1))
        norm_dist_list = []
        for comp_idx in range(mu.shape[1]):
            normal_dist = D.Normal(loc=mu[:, comp_idx:comp_idx+1], scale=sigma[:, comp_idx:comp_idx+1])
            norm_dist_list.append(normal_dist)
        try_list = torch.arange(0, 1, 0.0001).view(1, -1).to(pi.device)
        cdf = exp_dist.cdf(try_list.view(1, -1))
        for norm_dist in norm_dist_list:
            cdf += norm_dist.cdf(try_list.view(1, -1))
        try_list = try_list.view(-1) 
        idx = (cdf < tau).to(torch.int8).sum(dim=1)
        return try_list[idx]

    def predict(self, x):
        """predict.
        
        Inference helper returning the scalar mixture mean E[y | x].
        Wraps computation in torch.no_grad() to reduce memory usage.
        """
        # Notes:
        # - This block explains intent, invariants, and key assumptions specific to this function.
        with torch.no_grad():
            pi, lambda_, mu, sigma = self.forward(x)
            pi = torch.softmax(pi, dim=1)
            return torch.sum(pi * torch.concat([1/lambda_, mu], dim=1), dim=1)



