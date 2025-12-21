import numpy as np
import torch
import torch.nn.functional as F

class FeaturesLinear(torch.nn.Module):

    """FeaturesLinear.
    
    Implements the linear (wide) part for field-wise categorical features using an offset embedding table.
    Input: (B, num_fields) long indices; Output: (B, output_dim).
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
    def __init__(self, field_dims, output_dim=1):
        """__init__.
        
        Auto-generated function documentation for model module.
        See inline comments for data-flow assumptions (shapes/dtypes) and pipeline role.
        
        Args: self, field_dims, output_dim.
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
        self.fc = torch.nn.Embedding(sum(field_dims), output_dim)
        self.bias = torch.nn.Parameter(torch.zeros((output_dim,)))
        self.offsets = np.array((0, *np.cumsum(field_dims)[:-1]), dtype=np.long)

    def forward(self, x):
        """forward.
        
        Auto-generated function documentation for model module.
        See inline comments for data-flow assumptions (shapes/dtypes) and pipeline role.
        
        Args: self, x.
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
        x = x + x.new_tensor(self.offsets).unsqueeze(0)
        return torch.sum(self.fc(x), dim=1) + self.bias


class FeaturesEmbedding(torch.nn.Module):

    """FeaturesEmbedding.
    
    Embeds multi-field categorical features using a single embedding table with field offsets.
    Input: (B, num_fields) long indices; Output: (B, num_fields, embed_dim).
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
    def __init__(self, field_dims, embed_dim):
        """
        :param x: Long tensor of size ``(batch_size, num_fields)``
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
        self.embedding = torch.nn.Embedding(sum(field_dims), embed_dim)
        self.offsets = np.array((0, *np.cumsum(field_dims)[:-1]), dtype=np.long)
        torch.nn.init.xavier_uniform_(self.embedding.weight.data)

    def forward(self, x):
        """forward.
        
        Auto-generated function documentation for model module.
        See inline comments for data-flow assumptions (shapes/dtypes) and pipeline role.
        
        Args: self, x.
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
        x = x + x.new_tensor(self.offsets).unsqueeze(0)
        return self.embedding(x)


class SeqFeatureEmbedding(torch.nn.Module):

    """SeqFeatureEmbedding.
    
    Embeds a list of sequence-like feature tensors and concatenates pooled embeddings.
    Each sequence is offset into a shared embedding table.
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
    def __init__(self, vocab_sizes, embed_dim):
        """__init__.
        
        Auto-generated function documentation for model module.
        See inline comments for data-flow assumptions (shapes/dtypes) and pipeline role.
        
        Args: self, vocab_sizes, embed_dim.
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
        self.embedding = torch.nn.Embedding(sum(vocab_sizes), embed_dim)
        self.offsets = np.array((0, *np.cumsum(vocab_sizes)[:-1]), dtype=np.long)
        torch.nn.init.xavier_uniform_(self.embedding.weight.data)

    def forward(self, x_list):
        """forward.
        
        Auto-generated function documentation for model module.
        See inline comments for data-flow assumptions (shapes/dtypes) and pipeline role.
        
        Args: self, x_list.
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
        embs = list()
        for i, x in enumerate(x_list):
            embs.append(self.embedding(x + self.offsets[i]).sum(dim=1, keepdims=True))
        emb = torch.concat(embs, dim=1)
        return emb


class FactorizationMachine(torch.nn.Module):

    """FactorizationMachine.
    
    Computes second-order feature interactions: 0.5 * ( (sum x)^2 - sum(x^2) ).
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
    def __init__(self, reduce_sum=True):
        """__init__.
        
        Auto-generated function documentation for model module.
        See inline comments for data-flow assumptions (shapes/dtypes) and pipeline role.
        
        Args: self, reduce_sum.
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
        self.reduce_sum = reduce_sum

    def forward(self, x):
        """
        :param x: Float tensor of size ``(batch_size, num_fields, embed_dim)``
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
        square_of_sum = torch.sum(x, dim=1) ** 2
        sum_of_square = torch.sum(x ** 2, dim=1)
        ix = square_of_sum - sum_of_square
        if self.reduce_sum:
            ix = torch.sum(ix, dim=1, keepdim=True)
        return 0.5 * ix


class MultiLayerPerceptron(torch.nn.Module):

    """MultiLayerPerceptron.
    
    Standard MLP block: Linear -> BatchNorm -> ReLU -> Dropout repeated, optional final output layer.
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
    def __init__(self, input_dim, embed_dims, dropout, output_layer=True):
        """__init__.
        
        Auto-generated function documentation for model module.
        See inline comments for data-flow assumptions (shapes/dtypes) and pipeline role.
        
        Args: self, input_dim, embed_dims, dropout, output_layer.
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
        """
        :param x: Float tensor of size ``(batch_size, embed_dim)``
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
        return self.mlp(x)



class DurationMultiLayerPerceptron(torch.nn.Module):

    """DurationMultiLayerPerceptron.
    
    MLP variant that concatenates a duration feature at every layer input.
    Used to condition representations on video duration at multiple depths.
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
    def __init__(self, input_dim, embed_dims, dropout, output_layer=True):
        """__init__.
        
        Auto-generated function documentation for model module.
        See inline comments for data-flow assumptions (shapes/dtypes) and pipeline role.
        
        Args: self, input_dim, embed_dims, dropout, output_layer.
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
        """
        :param x: Float tensor of size ``(batch_size, embed_dim)``
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
        for mlp in self.mlps:
            x = mlp(torch.concat([x, duration], dim=1))
        return x


class Swish(torch.nn.Module):
    """Swish activation (x * sigmoid(x)).
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
    def forward(self, x):
        """forward.
        
        Auto-generated function documentation for model module.
        See inline comments for data-flow assumptions (shapes/dtypes) and pipeline role.
        
        Args: self, x.
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
        return x * torch.sigmoid(x)

class MultiLayerPerceptronD2Q(torch.nn.Module):

    """MultiLayerPerceptronD2Q.
    
    MLP variant using Swish activation; used by D2Q.
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
    def __init__(self, input_dim, embed_dims, dropout, output_layer=True):
        """__init__.
        
        Auto-generated function documentation for model module.
        See inline comments for data-flow assumptions (shapes/dtypes) and pipeline role.
        
        Args: self, input_dim, embed_dims, dropout, output_layer.
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
        """
        :param x: Float tensor of size ``(batch_size, embed_dim)``
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
        return self.mlp(x)

class MultiLayerPerceptronTPM(torch.nn.Module):

    """MultiLayerPerceptronTPM.
    
    MLP variant producing class_num outputs (tree node probabilities/logits) for TPM.
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
    def __init__(self, input_dim, embed_dims, dropout, class_num, output_layer=True):
        """__init__.
        
        Auto-generated function documentation for model module.
        See inline comments for data-flow assumptions (shapes/dtypes) and pipeline role.
        
        Args: self, input_dim, embed_dims, dropout, class_num, output_layer.
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
        """
        :param x: Float tensor of size ``(batch_size, embed_dim)``
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
        return self.mlp(x)

class AttentionalFactorizationMachine(torch.nn.Module):

    """AttentionalFactorizationMachine.
    
    AFM: applies attention over pairwise interactions to weight and aggregate them.
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
    def __init__(self, embed_dim, attn_size, dropouts):
        """__init__.
        
        Auto-generated function documentation for model module.
        See inline comments for data-flow assumptions (shapes/dtypes) and pipeline role.
        
        Args: self, embed_dim, attn_size, dropouts.
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
        self.attention = torch.nn.Linear(embed_dim, attn_size)
        self.projection = torch.nn.Linear(attn_size, 1)
        self.fc = torch.nn.Linear(embed_dim, 1)
        self.dropouts = dropouts

    def forward(self, x):
        """
        :param x: Float tensor of size ``(batch_size, num_fields, embed_dim)``
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
    
    DCN cross layers: x_{l+1} = x0 * (w_l^T x_l) + b_l + x_l.
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
    def __init__(self, input_dim, num_layers):
        """__init__.
        
        Auto-generated function documentation for model module.
        See inline comments for data-flow assumptions (shapes/dtypes) and pipeline role.
        
        Args: self, input_dim, num_layers.
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
        self.num_layers = num_layers
        self.w = torch.nn.ModuleList([
            torch.nn.Linear(input_dim, 1, bias=False) for _ in range(num_layers)
        ])
        self.b = torch.nn.ParameterList([
            torch.nn.Parameter(torch.zeros((input_dim,))) for _ in range(num_layers)
        ])

    def forward(self, x):
        """
        :param x: Float tensor of size ``(batch_size, num_fields, embed_dim)``
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
        x0 = x
        for i in range(self.num_layers):
            xw = self.w[i](x)
            x = x0 * xw + self.b[i] + x
        return x