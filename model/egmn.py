import torch
import torch.nn as nn
import torch.distributions as D
import torch.nn.functional as F
import numpy as np

from model.layers import FactorizationMachine, MultiLayerPerceptron, DurationMultiLayerPerceptron

class EGMN(torch.nn.Module):

    """EGMN (Exponential-Gaussian Mixture Network).
    
    Implements a distributional watch-time predictor:
    - Encodes sparse IDs, sequences, and continuous features into a shared representation.
    - Predicts mixture parameters for 1 Exponential (short/quick-skip) + K Gaussians (long-tail modes).
    - Provides (a) a likelihood-based training loss and (b) a scalar prediction via mixture mean.
    
    Used by: run_egmn.py (training loop + evaluation).
    """
    # -------------------------------------------------------------------------
    # Detailed developer notes (added for repository documentation):
    # - Role in pipeline: preprocessing -> dataloader -> model -> training script -> metrics
    # - Contracts: input keys/shapes, dtype expectations, device placement, masking rules
    # - Common pitfalls: silent dtype casting, shape mismatches, normalization differences
    # - If you change this code, re-run the corresponding run_*.py training to validate
    # -------------------------------------------------------------------------
    # Function-specific notes:
    # - Expectation: input is a feature dict produced by the dataloader; keys must match the dataset description.
    # - Ensure embedding-index tensors are dtype long and on the same device as the model.
    # - Keep output shapes stable because run scripts and losses depend on them.
    # - Class invariants: document expected member attributes and their shapes.
    # - Initialization: if adding parameters, consider initialization to avoid training instability.
    # -------------------------------------------------------------------------
    def __init__(self, description, embed_dim, share_mlp_dims, output_mlp_dims, dropout):
        """__init__.
        
        Args:
        - description: feature schema list (name, size, type).
        - embed_dim: embedding dimension for sparse/sequence features.
        - share_mlp_dims: hidden sizes for shared MLP tower.
        - output_mlp_dims: (unused in this implementation; reserved for extensions).
        - dropout: dropout rate used in MLP.
        """
        # -------------------------------------------------------------------------
        # Detailed developer notes (added for repository documentation):
        # - Role in pipeline: preprocessing -> dataloader -> model -> training script -> metrics
        # - Contracts: input keys/shapes, dtype expectations, device placement, masking rules
        # - Common pitfalls: silent dtype casting, shape mismatches, normalization differences
        # - If you change this code, re-run the corresponding run_*.py training to validate
        # -------------------------------------------------------------------------
        # Function-specific notes:
        # - Expectation: input is a feature dict produced by the dataloader; keys must match the dataset description.
        # - Ensure embedding-index tensors are dtype long and on the same device as the model.
        # - Keep output shapes stable because run scripts and losses depend on them.
        # - Readability: keep variable names aligned with math (e.g., pi, mu, sigma) and comment units/scales.
        # - Testing: if you modify logic, validate with a tiny batch and confirm shapes/dtypes.
        # -------------------------------------------------------------------------
        super().__init__()
        self.features = {name: (size, type) for name, size, type in description if (type in ["ctn", 'seq', 'spr'])}
        self.build(embed_dim, share_mlp_dims, output_mlp_dims, dropout)
    
    def build(self, embed_dim, share_mlp_dims, output_mlp_dims, dropout):
        """build.
        
        Constructs embedding/linear layers for each feature and the shared MLP tower.
        Also constructs output heads that parameterize the mixture distribution:
        - lambda_layer: Exponential rate (positive via Softplus).
        - mixture_logits: unnormalized logits for mixture weights.
        - gauss_mu / gauss_sigma: parameters for Gaussian components (positive via Softplus).
        """
        # -------------------------------------------------------------------------
        # Detailed developer notes (added for repository documentation):
        # - Role in pipeline: preprocessing -> dataloader -> model -> training script -> metrics
        # - Contracts: input keys/shapes, dtype expectations, device placement, masking rules
        # - Common pitfalls: silent dtype casting, shape mismatches, normalization differences
        # - If you change this code, re-run the corresponding run_*.py training to validate
        # -------------------------------------------------------------------------
        # Function-specific notes:
        # - Expectation: input is a feature dict produced by the dataloader; keys must match the dataset description.
        # - Ensure embedding-index tensors are dtype long and on the same device as the model.
        # - Keep output shapes stable because run scripts and losses depend on them.
        # - Readability: keep variable names aligned with math (e.g., pi, mu, sigma) and comment units/scales.
        # - Testing: if you modify logic, validate with a tiny batch and confirm shapes/dtypes.
        # -------------------------------------------------------------------------
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
        
        Auto-generated function documentation for model module.
        See inline comments for data-flow assumptions (shapes/dtypes) and pipeline role.
        
        Args: self.
        """
        # -------------------------------------------------------------------------
        # Detailed developer notes (added for repository documentation):
        # - Role in pipeline: preprocessing -> dataloader -> model -> training script -> metrics
        # - Contracts: input keys/shapes, dtype expectations, device placement, masking rules
        # - Common pitfalls: silent dtype casting, shape mismatches, normalization differences
        # - If you change this code, re-run the corresponding run_*.py training to validate
        # -------------------------------------------------------------------------
        # Function-specific notes:
        # - Expectation: input is a feature dict produced by the dataloader; keys must match the dataset description.
        # - Ensure embedding-index tensors are dtype long and on the same device as the model.
        # - Keep output shapes stable because run scripts and losses depend on them.
        # - Readability: keep variable names aligned with math (e.g., pi, mu, sigma) and comment units/scales.
        # - Testing: if you modify logic, validate with a tiny batch and confirm shapes/dtypes.
        # -------------------------------------------------------------------------
        for param in self.parameters():
            torch.nn.init.uniform_(param, -0.01, 0.01)

    def forward(self, x_dict):
        """forward.
        
        Encodes a feature dict into mixture parameters.
        
        Args:
        - x_dict: dict[str, Tensor] produced by the dataloader.
        
        Returns:
        - pi: (B, K+1) logits for mixture weights.
        - lambda_: (B, 1) exponential rate (>0).
        - mu: (B, K) Gaussian means (shifted to be > 1/lambda_ in this implementation).
        - sigma: (B, K) Gaussian stddev (>0).
        """
        # -------------------------------------------------------------------------
        # Detailed developer notes (added for repository documentation):
        # - Role in pipeline: preprocessing -> dataloader -> model -> training script -> metrics
        # - Contracts: input keys/shapes, dtype expectations, device placement, masking rules
        # - Common pitfalls: silent dtype casting, shape mismatches, normalization differences
        # - If you change this code, re-run the corresponding run_*.py training to validate
        # -------------------------------------------------------------------------
        # Function-specific notes:
        # - Expectation: input is a feature dict produced by the dataloader; keys must match the dataset description.
        # - Ensure embedding-index tensors are dtype long and on the same device as the model.
        # - Keep output shapes stable because run scripts and losses depend on them.
        # - Shape note: sparse features are typically (B, 1) longs; sequence features are (B, L) with a corresponding mask (B, L).
        # - Pooling note: sequence embeddings are masked and then averaged; be careful about division by zero if mask sums can be 0.
        # - Readability: keep variable names aligned with math (e.g., pi, mu, sigma) and comment units/scales.
        # - Testing: if you modify logic, validate with a tiny batch and confirm shapes/dtypes.
        # -------------------------------------------------------------------------
        linears = []
        embs = []
        for name, (_, type) in self.features.items():
            x = x_dict[name]
            if type == 'spr':
                embs.append(self.emb_layer[name](x).squeeze(1))
            elif type == 'ctn':
                linears.append(self.ctn_linear_layer[name](x))
            elif type == 'seq':
                seq_emb = self.emb_layer[name](x)
                seq_mask = torch.unsqueeze(x_dict["{}mask".format(name)], dim=2)
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
        
        Computes training losses for EGMN.
        - nll_loss: negative log-likelihood under the mixture distribution.
        - reg_loss: L1 loss between mixture-mean prediction and y_true (stabilizes training).
        - entropy_loss: mixture entropy regularizer (encourages/controls component usage).
        """
        # -------------------------------------------------------------------------
        # Detailed developer notes (added for repository documentation):
        # - Role in pipeline: preprocessing -> dataloader -> model -> training script -> metrics
        # - Contracts: input keys/shapes, dtype expectations, device placement, masking rules
        # - Common pitfalls: silent dtype casting, shape mismatches, normalization differences
        # - If you change this code, re-run the corresponding run_*.py training to validate
        # -------------------------------------------------------------------------
        # Function-specific notes:
        # - Expectation: input is a feature dict produced by the dataloader; keys must match the dataset description.
        # - Ensure embedding-index tensors are dtype long and on the same device as the model.
        # - Keep output shapes stable because run scripts and losses depend on them.
        # - Loss decomposition: primary likelihood/fit term + auxiliary regularizers (e.g., L1 reconstruction, entropy).
        # - Numerical stability: use epsilons when taking logs/dividing to avoid NaNs.
        # - Readability: keep variable names aligned with math (e.g., pi, mu, sigma) and comment units/scales.
        # - Testing: if you modify logic, validate with a tiny batch and confirm shapes/dtypes.
        # -------------------------------------------------------------------------
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
        
        Approximates a mixture quantile by brute-force scanning a dense grid of candidate values.
        Note: this is expensive (grid search) and used mainly for analysis/diagnostics.
        """
        # -------------------------------------------------------------------------
        # Detailed developer notes (added for repository documentation):
        # - Role in pipeline: preprocessing -> dataloader -> model -> training script -> metrics
        # - Contracts: input keys/shapes, dtype expectations, device placement, masking rules
        # - Common pitfalls: silent dtype casting, shape mismatches, normalization differences
        # - If you change this code, re-run the corresponding run_*.py training to validate
        # -------------------------------------------------------------------------
        # Function-specific notes:
        # - Expectation: input is a feature dict produced by the dataloader; keys must match the dataset description.
        # - Ensure embedding-index tensors are dtype long and on the same device as the model.
        # - Keep output shapes stable because run scripts and losses depend on them.
        # - Readability: keep variable names aligned with math (e.g., pi, mu, sigma) and comment units/scales.
        # - Testing: if you modify logic, validate with a tiny batch and confirm shapes/dtypes.
        # -------------------------------------------------------------------------
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
        
        Inference helper: returns the scalar mixture-mean prediction.
        Uses no_grad() to avoid autograd overhead.
        """
        # -------------------------------------------------------------------------
        # Detailed developer notes (added for repository documentation):
        # - Role in pipeline: preprocessing -> dataloader -> model -> training script -> metrics
        # - Contracts: input keys/shapes, dtype expectations, device placement, masking rules
        # - Common pitfalls: silent dtype casting, shape mismatches, normalization differences
        # - If you change this code, re-run the corresponding run_*.py training to validate
        # -------------------------------------------------------------------------
        # Function-specific notes:
        # - Expectation: input is a feature dict produced by the dataloader; keys must match the dataset description.
        # - Ensure embedding-index tensors are dtype long and on the same device as the model.
        # - Keep output shapes stable because run scripts and losses depend on them.
        # - Inference path should be wrapped in no_grad() to reduce memory and avoid autograd overhead.
        # - Output should be a 1D tensor of scalar predictions aligned with labels.
        # - Readability: keep variable names aligned with math (e.g., pi, mu, sigma) and comment units/scales.
        # - Testing: if you modify logic, validate with a tiny batch and confirm shapes/dtypes.
        # -------------------------------------------------------------------------
        with torch.no_grad():
            pi, lambda_, mu, sigma = self.forward(x)
            pi = torch.softmax(pi, dim=1)
            return torch.sum(pi * torch.concat([1/lambda_, mu], dim=1), dim=1)



