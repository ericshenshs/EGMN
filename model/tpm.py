"""model/tpm.py.

This file is part of the watch-time prediction codebase.
Primary role: model.
Defines one or more PyTorch modules that map feature dicts to predictions.
"""

import torch

from model.layers import  MultiLayerPerceptronTPM

class TPM(torch.nn.Module):

    """TPM.
    
    Tree-structured probability baseline.
    Outputs logits/probabilities for internal tree nodes; utils.py decodes these into an expected play_time.
    run_tpm.py trains using node-wise BCE + MSE on decoded expectation + variance regularization.
    """
    # Notes:
    # - Input contract: features are dict[str, Tensor] from dataloader; sparse/seq indices must be torch.long.
    # - Sequence pooling uses a mask (name+"mask"); mask should be float (0/1) and sum(mask) should be > 0.
    # - Output contract: shape and semantics must match the consuming run_*.py loss and evaluation logic.
    def __init__(self, description, class_num, embed_dim, mlp_dims, dropout):
        """__init__.
        
        Stores schema-derived feature metadata and builds submodules.
        description determines which features are embedded (spr/seq) vs treated as continuous (ctn).
        """
        # Notes:
        # - Parses description schema to decide which features are embedded vs treated as continuous.
        # - Calls build() to allocate submodules sized to vocab sizes and embedding dimension.
        super().__init__()
        self.features = {name: (size, type) for name, size, type in description if (type in ["ctn", 'seq', 'spr'])}
        self.build(embed_dim, mlp_dims, dropout,class_num)
    
    def build(self, embed_dim, mlp_dims, dropout,class_num):
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
            elif type == 'seq':
                self.emb_layer[name] = torch.nn.Embedding(size, embed_dim)
                embed_output_dim += embed_dim
            else:
                raise ValueError('unkown feature type: {}'.format(type))
        self.mlp = MultiLayerPerceptronTPM(embed_output_dim, mlp_dims, dropout, class_num)
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
        emb = torch.cat(embs, dim=1)
        res = self.mlp(emb)
        return torch.sigmoid(res)