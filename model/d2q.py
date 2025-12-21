import torch

from model.layers import MultiLayerPerceptronD2Q

class D2Q(torch.nn.Module):

    """D2Q baseline model.
    
    Predicts a (normalized) quantile position of play_time within a duration bucket.
    The run script maps between raw play_time values and quantile targets using precomputed CSVs.
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
    def __init__(self, description, embed_dim, mlp_dims, dropout):
        """__init__.
        
        Auto-generated function documentation for model module.
        See inline comments for data-flow assumptions (shapes/dtypes) and pipeline role.
        
        Args: self, description, embed_dim, mlp_dims, dropout.
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
        self.build(embed_dim, mlp_dims, dropout)
    
    def build(self, embed_dim, mlp_dims, dropout):
        """build.
        
        Auto-generated function documentation for model module.
        See inline comments for data-flow assumptions (shapes/dtypes) and pipeline role.
        
        Args: self, embed_dim, mlp_dims, dropout.
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
        self.mlp = MultiLayerPerceptronD2Q(embed_output_dim, mlp_dims, dropout)
        return

    # def init(self):
    #     for param in self.parameters():
    #         torch.nn.init.uniform_(param, -0.01, 0.01)

    def forward(self, x_dict):
        """forward.
        
        Auto-generated function documentation for model module.
        See inline comments for data-flow assumptions (shapes/dtypes) and pipeline role.
        
        Args: self, x_dict.
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
        res = self.mlp(emb)
        res = res.squeeze(1)
        return torch.sigmoid(res)