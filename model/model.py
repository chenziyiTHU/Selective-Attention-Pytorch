from dataclasses import dataclass, field
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional
import math


@dataclass
class ModelArgs:
    d: int = 8
    d_model: int = 64 * d
    n_layers: int = d
    n_heads: int = d
    vocab_size: int = 8000
    max_batch_size: int = 256
    budgets: List[int] = field(default_factory=lambda: [64] * 8)  # Only for inference

class MultiheadSelectiveAttention(nn.Module):
    """Multi-head selective attention module."""
    def __init__(self, d_model, num_heads):
        """
        Initializes the MultiheadSelectiveAttention module.

        Args:
            d_model (int): The input and output feature dimension.
            num_heads (int): The number of attention heads.

        Attributes:
            d_model (int): The input and output feature dimension.
            num_heads (int): The number of attention heads.
            d_head (int): The dimension of each head.
            W_q (nn.Linear): The linear layer for the query projection.
            W_k (nn.Linear): The linear layer for the key projection.
            W_v (nn.Linear): The linear layer for the value projection.
            W_o (nn.Linear): The linear layer for the output projection.
            norm_q (nn.LayerNorm): The layer normalization for the query.
            norm_k (nn.LayerNorm): The layer normalization for the key.
        """
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_head = d_model // num_heads

        self.W_q = nn.Linear(d_model, d_model, bias=False)  # Remove the biases
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)

        self.norm_q = nn.LayerNorm(d_model)
        self.norm_k = nn.LayerNorm(d_model)

    def forward(self, X):
        """
        Forward pass of the MultiheadSelectiveAttention module.

        Args:
            X (torch.Tensor): The input tensor of shape (batch, N, d_model).

        Returns:
            (output, F_mask) (Tuple[torch.Tensor, torch.Tensor]): The output tensor of shape (batch, N, d_model) and the masking score tensor of shape (batch, N, N).
        """
        batch, N, d_model = X.size()
        
        # Normalize the Q and K projections
        # Shape: (batch, N, num_heads * d_head)
        Q = self.norm_q(self.W_q(X))  
        K = self.norm_k(self.W_k(X))
        V = self.W_v(X) 
        
        # Reshape and permute for multi-head attention
        # Shape: (batch, num_heads, N, d_head)
        Q = Q.view(batch, N, self.num_heads, self.d_head).transpose(1, 2)
        K = K.view(batch, N, self.num_heads, self.d_head).transpose(1, 2)
        V = V.view(batch, N, self.num_heads, self.d_head).transpose(1, 2)

        attn_logits = torch.einsum('b h n d, b h m d -> b h n m', Q, K) / math.sqrt(self.d_head)
        causal_mask = torch.tril(torch.ones(N, N, device=X.device, dtype=bool)).view(1, 1, N, N)
        attn_logits = torch.where(causal_mask, attn_logits, float('-inf'))

        # Selective Attention Implementation
        S = attn_logits[:, 0]           # Select head 0
        S = F.relu(S)                   # Only positive selection
        S[..., 0] = 0                   # Do not mask <BOS>
        S = (1 - torch.eye(N, device=X.device)) * S      # Do not mask self
        S = torch.roll(S, 1, -2)
        S[..., 0, :] = 0                # Mask strictly in the future
        F_mask = torch.cumsum(S, axis=-2)  # Accumulate, shape: (batch, N, N)
        attn_logits -= F_mask[:, None] 

        attn_weights = F.softmax(attn_logits, dim=-1)  # (batch, num_heads, N, N)
        O = torch.matmul(attn_weights, V)  # (batch, num_heads, N, d_head)
        O = O.transpose(1, 2).reshape(batch, N, self.d_model)  # (batch, N, num_heads * d_head)
        output = self.W_o(O)  # (batch, N, d_model)

        return output, F_mask
    

class MultiheadSelectiveAttentionWithTokenPruning(nn.Module):
    def __init__(self, d_model, num_heads, max_batch_size, budget):
        super().__init__()
        self.base_attention = MultiheadSelectiveAttention(d_model, num_heads)
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_head = d_model // num_heads
        self.max_batch_size = max_batch_size
        self.budget = budget

        self.cache_k = torch.zeros((max_batch_size, num_heads, budget, self.d_head)).cuda()
        self.cache_v = torch.zeros((max_batch_size, num_heads, budget, self.d_head)).cuda()

    def forward(self, X, start_pos, use_pruning=False):
        if not use_pruning:
            # Training mode
            return self.base_attention(X)

        # Inference mode
        batch, N, d_model = X.size()
        
        # Normalize the Q and K projections
        # Shape: (batch, N, num_heads * d_head)
        XQ = self.base_attention.norm_q(self.base_attention.W_q(X))
        XK = self.base_attention.norm_k(self.base_attention.W_k(X))
        XV = self.base_attention.W_v(X)
        
        # Reshape and permute for multi-head attention
        # Shape: (batch, num_heads, N, d_head)
        XQ = XQ.view(batch, N, self.num_heads, self.d_head).transpose(1, 2)
        XK = XK.view(batch, N, self.num_heads, self.d_head).transpose(1, 2)
        XV = XV.view(batch, N, self.num_heads, self.d_head).transpose(1, 2)

        # Previous cached K and V
        K_cached = self.cache_k[:batch, :, :start_pos]
        V_cached = self.cache_v[:batch, :, :start_pos]

        K = torch.cat([K_cached, XK], dim=2)  # (batch, num_heads, cache_len + N, d_head)
        V = torch.cat([V_cached, XV], dim=2)

        attn_logits = torch.einsum('b h n d, b h m d -> b h n m', XQ, K) / math.sqrt(self.d_head)  # (batch, num_heads, N, cached + N)
        causal_mask = torch.tril(torch.ones(N, N, device=X.device, dtype=bool)).view(1, 1, N, N)
        attn_logits[:, :, :, start_pos:] = torch.where(causal_mask, attn_logits[:, :, :, start_pos:], float('-inf'))

        # Selective Attention Implementation
        S = attn_logits[:, 0]                           # Select head 0, size: (batch, N, cached_len + N)
        S = F.relu(S)                                   # Only positive selection
        S[..., start_pos] = 0                           # Do not mask <BOS>
        S[:, :, start_pos:] = (1 - torch.eye(N, device=X.device)) * S[:, :, start_pos:]    # Do not mask self
        S = torch.roll(S, 1, -2)
        S[..., 0, :] = 0                                # Mask strictly in the future
        F_mask = torch.cumsum(S, axis=-2)               # Accumulate, size: (batch, N, cached_len + N)

        # Token Pruning
        pruning_mask = torch.ones_like(F_mask, dtype=torch.bool)
        num_unmasked = min(self.budget, start_pos + N)
        unmasked_indices = torch.arange(num_unmasked, dtype=int, device=X.device).unsqueeze(0).repeat(batch, 1)  # (batch, budget) or (batch, cached_len + N)
        for i in range(start_pos, start_pos + N):
            if i < self.budget:
                continue
            else:
                # print(unmasked_indices)
                new_indices = torch.cat([unmasked_indices, i * torch.ones((batch, 1), dtype=int, device=X.device)], dim=1).to(X.device)
                index = torch.argmax(torch.gather(F_mask[:, i-start_pos, :], 1, new_indices), dim=-1)

                # Prune the token with the largest accumulated score
                for j in range(i - start_pos, N):
                    pruning_mask[:, j, :].scatter_(1, new_indices.gather(1, index.unsqueeze(1)), False)
                
                unmasked_indices = new_indices[new_indices != new_indices.gather(1, index.unsqueeze(1)).repeat(1, self.budget + 1)].reshape(batch, -1)
        
        assert unmasked_indices.shape[1] <= self.budget, "Incorrect number of tokens retained"

        # Update the KV cache with the unmasked key and value for the next iteration
        # Shape: (batch, num_heads, budget, d_head)
        self.cache_k[:batch, :, :num_unmasked] = torch.gather(K, 2, unmasked_indices.view(batch, 1, -1, 1).expand(-1, K.size(1), -1, K.size(-1)))
        self.cache_v[:batch, :, :num_unmasked] = torch.gather(V, 2, unmasked_indices.view(batch, 1, -1, 1).expand(-1, V.size(1), -1, V.size(-1)))

        attn_logits = torch.where(pruning_mask.unsqueeze(1).repeat(1, self.num_heads, 1, 1), attn_logits, float('-inf'))
        attn_weights = F.softmax(attn_logits, dim=-1)  # (batch, num_heads, N, M)
        O = torch.matmul(attn_weights, V)  # (batch, num_heads, N, d_head)
        O = O.transpose(1, 2).reshape(batch, N, self.d_model)  # (batch, N, num_heads * d_head)

        output = self.base_attention.W_o(O)  # (batch, N, d_model)
        return output, F_mask


class SwiGLU(nn.Module):
    """Swish-Gated Linear Unit (SwiGLU) module."""
    def __init__(self, d_model, hidden_dim):
        """
        Initializes the SwiGLU module.

        Args:
            d_model (int): The input and output feature dimension.
            hidden_dim (int): The hidden layer dimension.

        Attributes:
            fc1 (nn.Linear): The first linear layer.
            fc2 (nn.Linear): The second linear layer.
            gate (nn.Linear): The gate linear layer.
        """
        super().__init__()
        self.fc1 = nn.Linear(d_model, hidden_dim, bias=False)
        self.fc2 = nn.Linear(hidden_dim, d_model, bias=False)
        self.gate = nn.Linear(d_model, hidden_dim, bias=False)

    def forward(self, X):
        return self.fc2(F.silu(self.gate(X)) * self.fc1(X))
    

class SelectiveTransformerBlock(nn.Module):
    """Transformer block with selective attention."""
    def __init__(self, layer_id, d_model, num_heads, max_batch_size, budget):
        """
        Initializes the SelectiveTransformerBlock module.
        
        Args:
            layer_id (int): Identifier for the layer.
            d_model (int): The input and output feature dimension.
            num_heads (int): The number of attention heads.
            max_batch_size (int): The maximum batch size for inference.
            budget (int): The budget for token pruning.

        Attributes:
            layer_id (int): Identifier for the layer.
            d_model (int): The input and output feature dimension.
            num_heads (int): The number of attention heads.
            d_head (int): The dimension of each head.
            budget (int): The budget for token pruning.
            attention (MultiheadSelectiveAttentionWithTokenPruning): The multi-head selective attention layer.
            feedforward (SwiGLU): The SwiGLU layer.
            attention_norm (nn.RMSNorm): The RMSNorm layer for the attention output.
            ffn_norm (nn.RMSNorm): The RMSNorm layer for the feedforward output.
        """
        super().__init__()
        self.layer_id = layer_id
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_head = d_model // num_heads
        self.budget = budget
        self.attention = MultiheadSelectiveAttentionWithTokenPruning(d_model, num_heads, max_batch_size, budget)
        self.feedforward = SwiGLU(d_model, d_model * 4)
        self.attention_norm = nn.RMSNorm(d_model)
        self.ffn_norm = nn.RMSNorm(d_model)

    def forward(self, X, start_pos=None, use_pruning=False):
        """
        Forward pass of the SelectiveTransformerBlock module.

        Args:
            X (torch.Tensor): The input tensor of shape (batch, N, d_model).
            start_pos (int): The starting position for KV cache.
            use_pruning (bool): Whether to use token pruning.

        Returns:
            (output, F_mask) (Tuple[torch.Tensor, torch.Tensor]): The output tensor of shape (batch, N, d_model) and the masking score tensor of shape (batch, N, N).
        """
        H, F_mask = self.attention(self.attention_norm(X), start_pos, use_pruning)
        H = H + X
        out = H + self.feedforward(self.ffn_norm(H))
        return out, F_mask
    

class SelectiveTransformer(nn.Module):
    """Transformer model with selective attention."""
    def __init__(self, params: ModelArgs):
        """
        Initializes the SelectiveTransformer module.

        Args:
            params (ModelArgs): The model arguments.

        Attributes:
            params (ModelArgs): The model arguments.
            vocab_size (int): The size of the vocabulary.
            n_layers (int): The number of layers in the model.
            token_embedding (nn.Embedding): The token embedding layer.
            layers (nn.ModuleList): The list of SelectiveTransformerBlock layers.
            norm (nn.RMSNorm): The RMSNorm layer.
            output_layer (nn.Linear): The output linear layer.
        """
        super().__init__()
        assert len(params.budgets) == params.n_layers, "Budgets must be specified for each layer"
        self.params = params
        self.vocab_size = params.vocab_size
        self.n_layers = params.n_layers

        self.token_embedding = nn.Embedding(params.vocab_size, params.d_model)

        self.layers = nn.ModuleList()
        for layer_id in range(params.n_layers):
            self.layers.append(SelectiveTransformerBlock(layer_id, params.d_model, params.n_heads, params.max_batch_size, params.budgets[layer_id]))

        self.norm = nn.RMSNorm(params.d_model)
        self.output_layer = nn.Linear(params.d_model, params.vocab_size, bias=False)

    def forward(self, tokens: torch.Tensor, start_pos: int = None, use_pruning: bool = False):
        """
        Forward pass of the SelectiveTransformer module.
        
        Args:
            tokens (torch.Tensor): The input tensor of shape (batch, N).

        Returns:
            (output, F_masks) (Tuple[torch.Tensor, List[torch.Tensor]]): The output tensor of shape (batch, N, vocab_size) and the list of masking score tensors of each layer.
        """
        h = self.token_embedding(tokens)
        F_masks = []
        for layer in self.layers:
            h, F_mask = layer(h, start_pos, use_pruning)
            F_masks.append(F_mask)
        h = self.norm(h)
        output = self.output_layer(h).float()
        return output, F_masks
    
    @torch.inference_mode()
    def generate(self, tokens, max_new_tokens):
        batch_size = tokens.size(0)
        generated = tokens
        for i in range(max_new_tokens):
            output, _ = self.forward(generated, i, use_pruning=True)
            next_token = torch.argmax(output[:, -1], dim=-1).unsqueeze(1)
            generated = torch.cat([generated, next_token], dim=-1)
        return generated


if __name__ == "__main__":
    # Test MultiheadSelectiveAttention
    d_model = 64
    num_heads = 8
    # Example input
    input_tensor = torch.randn((2, 16, d_model))
    model = MultiheadSelectiveAttention(d_model, num_heads)
    output, F_mask = model(input_tensor)
    print(output.shape)  # Expected output: torch.Size([2, 16, 64])
    print(F_mask.shape)  # Expected output: torch.Size([2, 16, 16])

    # Test SelectiveTransformer
    model_args = ModelArgs()
    model = SelectiveTransformer(model_args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    # Example input
    batch_size = 2
    seq_len = 128
    input_ids = torch.randint(0, model_args.vocab_size, (batch_size, seq_len)).to(device)
    output, F_masks = model(input_ids)
    print(output.shape)  # Expected output: torch.Size([2, 128, 8000])
    print(len(F_masks))  # Expected output: 8
    print(F_masks[0].shape)  # Expected output: torch.Size([2, 128, 128])

    # Test SelectiveTransformer generate
    output_ids = model.generate(input_ids, 32)
    print(output_ids.shape)  # Expected output: torch.Size([2, 160])
