"""model/layers.py.

This file is part of the watch-time prediction codebase.
Primary role: model.
Defines one or more PyTorch modules that map feature dicts to predictions.
"""

import numpy as np
import torch
import torch.nn.functional as F

class FeaturesLinear(torch.nn.Module):

    """FeaturesLinear.
    
    Linear (wide) term for multi-field categorical inputs using a single embedding table with offsets.
    Given x of shape (B, num_fields) with per-field indices, it offsets indices into a shared table and sums.
    """
    # Notes:
    # - Implements a “wide” linear term for multi-field categorical inputs using embedding lookups.
    # - Offsets map per-field indices into a shared embedding table (sum(field_dims) rows).
    def __init__(self, field_dims, output_dim=1):
        """__init__.
        
        Args:
        - field_dims: list/array of vocab sizes per categorical field.
        - output_dim: output dimension (usually 1 for a wide term).
        
        Implementation detail: offsets map per-field indices into a shared embedding table.
        """
        # Notes:
        # - Creates embedding table of size sum(field_dims) representing per-field linear weights.
        # - Stores offsets so each field uses its own slice of the shared embedding table.
        super().__init__()
        self.fc = torch.nn.Embedding(sum(field_dims), output_dim)
        self.bias = torch.nn.Parameter(torch.zeros((output_dim,)))
        self.offsets = np.array((0, *np.cumsum(field_dims)[:-1]), dtype=np.long)

    def forward(self, x):
        """forward.
        
        Args:
        - x: Long tensor (B, num_fields) of per-field categorical indices.
        
        Returns:
        - (B, output_dim) linear term computed by summing per-field embeddings + bias.
        """
        # Notes:
        # - Adds offsets then sums embeddings across fields to form a linear score + bias.
        # - Input x must be integer indices (torch.long).
        x = x + x.new_tensor(self.offsets).unsqueeze(0)
        return torch.sum(self.fc(x), dim=1) + self.bias


class FeaturesEmbedding(torch.nn.Module):

    """FeaturesEmbedding.
    
    Embedding layer for multi-field categorical inputs using a single embedding table with field offsets.
    Input: (B, num_fields) long indices. Output: (B, num_fields, embed_dim) float embeddings.
    """
    # Notes:
    # - Input contract: features are dict[str, Tensor] from dataloader; sparse/seq indices must be torch.long.
    # - Sequence pooling uses a mask (name+"mask"); mask should be float (0/1) and sum(mask) should be > 0.
    # - Output contract: shape and semantics must match the consuming run_*.py loss and evaluation logic.
    def __init__(self, field_dims, embed_dim):
        """__init__.
        
        Args:
        - field_dims: list/array of vocab sizes per field.
        - embed_dim: embedding dimension for each field.
        """
        # Notes:
        # - Parses description schema to decide which features are embedded vs treated as continuous.
        # - Calls build() to allocate submodules sized to vocab sizes and embedding dimension.
        super().__init__()
        self.embedding = torch.nn.Embedding(sum(field_dims), embed_dim)
        self.offsets = np.array((0, *np.cumsum(field_dims)[:-1]), dtype=np.long)
        torch.nn.init.xavier_uniform_(self.embedding.weight.data)

    def forward(self, x):
        """forward.
        
        Args:
        - x: Long tensor (B, num_fields).
        
        Returns:
        - Float tensor (B, num_fields, embed_dim).
        """
        # Notes:
        # - Builds per-feature representations then concatenates them into a single vector per sample.
        # - Sequence features are pooled by masked mean: sum(emb*mask)/sum(mask).
        # - Final output is typically sigmoid-transformed in these baselines (matching their run scripts).
        x = x + x.new_tensor(self.offsets).unsqueeze(0)
        return self.embedding(x)


class SeqFeatureEmbedding(torch.nn.Module):

    """SeqFeatureEmbedding.
    
    Embedding helper for a list of sequence-like categorical tensors with per-sequence offsets.
    Each sequence is embedded and pooled (sum) then concatenated.
    """
    # Notes:
    # - Input contract: features are dict[str, Tensor] from dataloader; sparse/seq indices must be torch.long.
    # - Sequence pooling uses a mask (name+"mask"); mask should be float (0/1) and sum(mask) should be > 0.
    # - Output contract: shape and semantics must match the consuming run_*.py loss and evaluation logic.
    def __init__(self, vocab_sizes, embed_dim):
        """__init__.
        
        Args:
        - vocab_sizes: list of vocab sizes for each sequence feature.
        - embed_dim: embedding dimension.
        """
        # Notes:
        # - Parses description schema to decide which features are embedded vs treated as continuous.
        # - Calls build() to allocate submodules sized to vocab sizes and embedding dimension.
        super().__init__()
        self.embedding = torch.nn.Embedding(sum(vocab_sizes), embed_dim)
        self.offsets = np.array((0, *np.cumsum(vocab_sizes)[:-1]), dtype=np.long)
        torch.nn.init.xavier_uniform_(self.embedding.weight.data)

    def forward(self, x_list):
        """forward.
        
        Args:
        - x_list: list of Long tensors, one per sequence feature, typically shaped (B, L).
        
        Returns:
        - Float tensor (B, num_seq_features, embed_dim) after summing over sequence length.
        """
        # Notes:
        # - Builds per-feature representations then concatenates them into a single vector per sample.
        # - Sequence features are pooled by masked mean: sum(emb*mask)/sum(mask).
        # - Final output is typically sigmoid-transformed in these baselines (matching their run scripts).
        embs = list()
        for i, x in enumerate(x_list):
            embs.append(self.embedding(x + self.offsets[i]).sum(dim=1, keepdims=True))
        emb = torch.concat(embs, dim=1)
        return emb


class FactorizationMachine(torch.nn.Module):

    """FactorizationMachine.
    
    Second-order interaction layer: 0.5 * ( (sum x)^2 - sum(x^2) ).
    Commonly used as the interaction term in FM/FFM-style models.
    """
    # Notes:
    # - Input contract: features are dict[str, Tensor] from dataloader; sparse/seq indices must be torch.long.
    # - Sequence pooling uses a mask (name+"mask"); mask should be float (0/1) and sum(mask) should be > 0.
    # - Output contract: shape and semantics must match the consuming run_*.py loss and evaluation logic.
    def __init__(self, reduce_sum=True):
        """__init__.
        
        Args:
        - reduce_sum: if True, sums interaction vector over embed_dim to a scalar per sample.
        """
        # Notes:
        # - Parses description schema to decide which features are embedded vs treated as continuous.
        # - Calls build() to allocate submodules sized to vocab sizes and embedding dimension.
        super().__init__()
        self.reduce_sum = reduce_sum

    def forward(self, x):
        """forward.
        
        Args:
        - x: Float tensor (B, num_fields, embed_dim).
        
        Returns:
        - If reduce_sum: (B, 1) interaction term; else: (B, embed_dim).
        """
        # Notes:
        # - Builds per-feature representations then concatenates them into a single vector per sample.
        # - Sequence features are pooled by masked mean: sum(emb*mask)/sum(mask).
        # - Final output is typically sigmoid-transformed in these baselines (matching their run scripts).
        square_of_sum = torch.sum(x, dim=1) ** 2
        sum_of_square = torch.sum(x ** 2, dim=1)
        ix = square_of_sum - sum_of_square
        if self.reduce_sum:
            ix = torch.sum(ix, dim=1, keepdim=True)
        return 0.5 * ix


class MultiLayerPerceptron(torch.nn.Module):

    """MultiLayerPerceptron.
    
    Generic MLP block used across baselines (Linear -> BN -> ReLU -> Dropout repeated).
    Optionally appends a final Linear output head.
    """
    # Notes:
    # - Input contract: features are dict[str, Tensor] from dataloader; sparse/seq indices must be torch.long.
    # - Sequence pooling uses a mask (name+"mask"); mask should be float (0/1) and sum(mask) should be > 0.
    # - Output contract: shape and semantics must match the consuming run_*.py loss and evaluation logic.
    def __init__(self, input_dim, embed_dims, dropout, output_layer=True):
        """__init__.
        
        Builds an MLP with BatchNorm/ReLU/Dropout for each hidden layer.
        Optionally adds a final Linear layer to produce a scalar output.
        """
        # Notes:
        # - Parses description schema to decide which features are embedded vs treated as continuous.
        # - Calls build() to allocate submodules sized to vocab sizes and embedding dimension.
        super().__init__()
        layers = list()
        for embed_dim in embed_dims:
            layers.append(torch.nn.Linear(input_dim, embed_dim))
            layers.append(torch.nn.BatchNorm1d(embed_dim))
            layers.append(torch.nn.ReLU())
            layers.append(torch.nn.Dropout(p=dropout))
            input_dim = embed_dim
        if output_layer:
            layers.append(torch.nn.Linear(input_dim, 1))
        self.mlp = torch.nn.Sequential(*layers)

    def forward(self, x):
        """forward.
        
        Args:
        - x: Float tensor (B, input_dim).
        
        Returns:
        - Output of the MLP (shape depends on whether output_layer=True).
        """
        # Notes:
        # - Builds per-feature representations then concatenates them into a single vector per sample.
        # - Sequence features are pooled by masked mean: sum(emb*mask)/sum(mask).
        # - Final output is typically sigmoid-transformed in these baselines (matching their run scripts).
        return self.mlp(x)



class DurationMultiLayerPerceptron(torch.nn.Module):

    """DurationMultiLayerPerceptron.
    
    MLP variant that concatenates a duration feature at each layer input.
    This conditions hidden transformations on video duration throughout the network depth.
    """
    # Notes:
    # - Input contract: features are dict[str, Tensor] from dataloader; sparse/seq indices must be torch.long.
    # - Sequence pooling uses a mask (name+"mask"); mask should be float (0/1) and sum(mask) should be > 0.
    # - Output contract: shape and semantics must match the consuming run_*.py loss and evaluation logic.
    def __init__(self, input_dim, embed_dims, dropout, output_layer=True):
        """__init__.
        
        MLP where each layer takes concat([x, duration]) as input.
        This injects duration context at every depth.
        """
        # Notes:
        # - Parses description schema to decide which features are embedded vs treated as continuous.
        # - Calls build() to allocate submodules sized to vocab sizes and embedding dimension.
        super().__init__()
        self.mlps = torch.nn.ModuleList()
        for embed_dim in embed_dims:
            layers = list()
            layers.append(torch.nn.Linear(input_dim + 1, embed_dim))
            layers.append(torch.nn.BatchNorm1d(embed_dim))
            layers.append(torch.nn.ReLU())
            layers.append(torch.nn.Dropout(p=dropout))
            self.mlps.append(torch.nn.Sequential(*layers))
            input_dim = embed_dim

    def forward(self, x, duration):
        """forward.
        
        Args:
        - x: Float tensor (B, d).
        - duration: Float tensor (B, 1).
        
        Returns:
        - Updated representation after all layers.
        """
        # Notes:
        # - Builds per-feature representations then concatenates them into a single vector per sample.
        # - Sequence features are pooled by masked mean: sum(emb*mask)/sum(mask).
        # - Final output is typically sigmoid-transformed in these baselines (matching their run scripts).
        for mlp in self.mlps:
            x = mlp(torch.concat([x, duration], dim=1))
        return x


class Swish(torch.nn.Module):
    """Swish.
    
    Swish activation: x * sigmoid(x).
    Used in some baselines (e.g., D2Q/TPM variants) as a smoother alternative to ReLU.
    """
    # Notes:
    # - Input contract: features are dict[str, Tensor] from dataloader; sparse/seq indices must be torch.long.
    # - Sequence pooling uses a mask (name+"mask"); mask should be float (0/1) and sum(mask) should be > 0.
    # - Output contract: shape and semantics must match the consuming run_*.py loss and evaluation logic.
    def forward(self, x):
        """forward.
        
        Applies Swish nonlinearity: x * sigmoid(x).
        """
        # Notes:
        # - Builds per-feature representations then concatenates them into a single vector per sample.
        # - Sequence features are pooled by masked mean: sum(emb*mask)/sum(mask).
        # - Final output is typically sigmoid-transformed in these baselines (matching their run scripts).
        return x * torch.sigmoid(x)

class MultiLayerPerceptronD2Q(torch.nn.Module):

    """MultiLayerPerceptronD2Q.
    
    MLP variant using Swish activation; used by the D2Q baseline.
    Optionally appends a final scalar output head.
    """
    # Notes:
    # - Input contract: features are dict[str, Tensor] from dataloader; sparse/seq indices must be torch.long.
    # - Sequence pooling uses a mask (name+"mask"); mask should be float (0/1) and sum(mask) should be > 0.
    # - Output contract: shape and semantics must match the consuming run_*.py loss and evaluation logic.
    def __init__(self, input_dim, embed_dims, dropout, output_layer=True):
        """__init__.
        
        Same structure as MultiLayerPerceptron but uses Swish activation.
        """
        # Notes:
        # - Parses description schema to decide which features are embedded vs treated as continuous.
        # - Calls build() to allocate submodules sized to vocab sizes and embedding dimension.
        super().__init__()
        layers = list()
        for embed_dim in embed_dims:
            layers.append(torch.nn.Linear(input_dim, embed_dim))
            layers.append(torch.nn.BatchNorm1d(embed_dim))
            layers.append(Swish())
            layers.append(torch.nn.Dropout(p=dropout))
            input_dim = embed_dim
        if output_layer:
            layers.append(torch.nn.Linear(input_dim, 1))
        self.mlp = torch.nn.Sequential(*layers)

    def forward(self, x):
        """forward.
        
        Args:
        - x: Float tensor (B, d).
        
        Returns:
        - MLP output.
        """
        # Notes:
        # - Builds per-feature representations then concatenates them into a single vector per sample.
        # - Sequence features are pooled by masked mean: sum(emb*mask)/sum(mask).
        # - Final output is typically sigmoid-transformed in these baselines (matching their run scripts).
        return self.mlp(x)

class MultiLayerPerceptronTPM(torch.nn.Module):

    """MultiLayerPerceptronTPM.
    
    MLP variant producing class_num outputs (tree node logits/probabilities) for TPM.
    """
    # Notes:
    # - Input contract: features are dict[str, Tensor] from dataloader; sparse/seq indices must be torch.long.
    # - Sequence pooling uses a mask (name+"mask"); mask should be float (0/1) and sum(mask) should be > 0.
    # - Output contract: shape and semantics must match the consuming run_*.py loss and evaluation logic.
    def __init__(self, input_dim, embed_dims, dropout, class_num, output_layer=True):
        """__init__.
        
        MLP ending in a Linear layer with class_num outputs (tree nodes).
        """
        # Notes:
        # - Parses description schema to decide which features are embedded vs treated as continuous.
        # - Calls build() to allocate submodules sized to vocab sizes and embedding dimension.
        super().__init__()
        layers = list()
        for embed_dim in embed_dims:
            layers.append(torch.nn.Linear(input_dim, embed_dim))
            layers.append(torch.nn.BatchNorm1d(embed_dim))
            layers.append(Swish())
            layers.append(torch.nn.Dropout(p=dropout))
            input_dim = embed_dim
        if output_layer:
            layers.append(torch.nn.Linear(input_dim, class_num))
        self.mlp = torch.nn.Sequential(*layers)

    def forward(self, x):
        """forward.
        
        Args:
        - x: Float tensor (B, d).
        
        Returns:
        - Tensor (B, class_num).
        """
        # Notes:
        # - Builds per-feature representations then concatenates them into a single vector per sample.
        # - Sequence features are pooled by masked mean: sum(emb*mask)/sum(mask).
        # - Final output is typically sigmoid-transformed in these baselines (matching their run scripts).
        return self.mlp(x)

class AttentionalFactorizationMachine(torch.nn.Module):

    """AttentionalFactorizationMachine.
    
    AFM layer: applies attention over pairwise feature interactions and aggregates them.
    Useful when you want interaction terms but not all pairs should contribute equally.
    """
    # Notes:
    # - Input contract: features are dict[str, Tensor] from dataloader; sparse/seq indices must be torch.long.
    # - Sequence pooling uses a mask (name+"mask"); mask should be float (0/1) and sum(mask) should be > 0.
    # - Output contract: shape and semantics must match the consuming run_*.py loss and evaluation logic.
    def __init__(self, embed_dim, attn_size, dropouts):
        """__init__.
        
        Args:
        - embed_dim: dimension of field embeddings.
        - attn_size: hidden size for attention network.
        - dropouts: tuple/list with dropout rates (on attention scores and on aggregated output).
        """
        # Notes:
        # - Parses description schema to decide which features are embedded vs treated as continuous.
        # - Calls build() to allocate submodules sized to vocab sizes and embedding dimension.
        super().__init__()
        self.attention = torch.nn.Linear(embed_dim, attn_size)
        self.projection = torch.nn.Linear(attn_size, 1)
        self.fc = torch.nn.Linear(embed_dim, 1)
        self.dropouts = dropouts

    def forward(self, x):
        """forward.
        
        Computes attention-weighted sum over pairwise interactions.
        Args: x is float tensor (B, num_fields, embed_dim).
        Returns: (B, 1).
        """
        # Notes:
        # - Builds per-feature representations then concatenates them into a single vector per sample.
        # - Sequence features are pooled by masked mean: sum(emb*mask)/sum(mask).
        # - Final output is typically sigmoid-transformed in these baselines (matching their run scripts).
        num_fields = x.shape[1]
        row, col = list(), list()
        for i in range(num_fields - 1):
            for j in range(i + 1, num_fields):
                row.append(i), col.append(j)
        p, q = x[:, row], x[:, col]
        inner_product = p * q
        attn_scores = F.relu(self.attention(inner_product))
        attn_scores = F.softmax(self.projection(attn_scores), dim=1)
        attn_scores = F.dropout(attn_scores, p=self.dropouts[0], training=self.training)
        attn_output = torch.sum(attn_scores * inner_product, dim=1)
        attn_output = F.dropout(attn_output, p=self.dropouts[1], training=self.training)
        return self.fc(attn_output)

class CrossNetwork(torch.nn.Module):

    """CrossNetwork.
    
    DCN-style cross network: iteratively builds explicit feature crosses.
    Update: x_{l+1} = x0 * (w_l^T x_l) + b_l + x_l.
    """
    # Notes:
    # - Input contract: features are dict[str, Tensor] from dataloader; sparse/seq indices must be torch.long.
    # - Sequence pooling uses a mask (name+"mask"); mask should be float (0/1) and sum(mask) should be > 0.
    # - Output contract: shape and semantics must match the consuming run_*.py loss and evaluation logic.
    def __init__(self, input_dim, num_layers):
        """__init__.
        
        Args:
        - input_dim: dimensionality of x.
        - num_layers: number of cross layers.
        """
        # Notes:
        # - Parses description schema to decide which features are embedded vs treated as continuous.
        # - Calls build() to allocate submodules sized to vocab sizes and embedding dimension.
        super().__init__()
        self.num_layers = num_layers
        self.w = torch.nn.ModuleList([
            torch.nn.Linear(input_dim, 1, bias=False) for _ in range(num_layers)
        ])
        self.b = torch.nn.ParameterList([
            torch.nn.Parameter(torch.zeros((input_dim,))) for _ in range(num_layers)
        ])

    def forward(self, x):
        """forward.
        
        Args:
        - x: Float tensor (B, input_dim).
        
        Returns:
        - Float tensor (B, input_dim) after explicit cross feature transformations.
        """
        # Notes:
        # - Builds per-feature representations then concatenates them into a single vector per sample.
        # - Sequence features are pooled by masked mean: sum(emb*mask)/sum(mask).
        # - Final output is typically sigmoid-transformed in these baselines (matching their run scripts).
        x0 = x
        for i in range(self.num_layers):
            xw = self.w[i](x)
            x = x0 * xw + self.b[i] + x
        return x