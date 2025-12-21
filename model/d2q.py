"""model/d2q.py.

This file is part of the watch-time prediction codebase.
Primary role: model.
Defines one or more PyTorch modules that map feature dicts to predictions.
"""

import torch

from model.layers import MultiLayerPerceptronD2Q

class D2Q(torch.nn.Module):

    """D2Q.
    
    Baseline predicting a within-duration-bucket quantile coordinate for play_time.
    The run script (run_d2q.py) maps between raw play_time values and this target using CSV quantiles.
    Model output is sigmoid-bounded to [0,1].
    """
    # Notes:
    # - Input contract: features are dict[str, Tensor] from dataloader; sparse/seq indices must be torch.long.
    # - Sequence pooling uses a mask (name+"mask"); mask should be float (0/1) and sum(mask) should be > 0.
    # - Output contract: shape and semantics must match the consuming run_*.py loss and evaluation logic.
    def __init__(self, description, embed_dim, mlp_dims, dropout):
        """__init__.
        
        Stores schema-derived feature metadata and builds submodules.
        description determines which features are embedded (spr/seq) vs treated as continuous (ctn).
        """
        # Notes:
        # - Parses description schema to decide which features are embedded vs treated as continuous.
        # - Calls build() to allocate submodules sized to vocab sizes and embedding dimension.
        super().__init__()
        self.features = {name: (size, type) for name, size, type in description if (type in ["ctn", 'seq', 'spr'])}
        self.build(embed_dim, mlp_dims, dropout)
    
    def build(self, embed_dim, mlp_dims, dropout):
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
        self.mlp = MultiLayerPerceptronD2Q(embed_output_dim, mlp_dims, dropout)
        return

    # def init(self):
    #     for param in self.parameters():
    #         torch.nn.init.uniform_(param, -0.01, 0.01)

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
        res = self.mlp(emb)
        res = res.squeeze(1)
        return torch.sigmoid(res)