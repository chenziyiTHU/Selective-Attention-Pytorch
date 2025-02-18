
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
import math


@dataclass
class ModelArgs:
    d: int = 8
    d_model: int = 64 * 8
    n_layers: int = 8
    n_heads: int = 8
    vocab_size: int = -1 # Defined later by tokenizer
    
    max_batch_size: int = 256
    budget: int = 512


class MultiheadSelectiveAttention(nn.Module):
    ####### TODO: Check the MultiheadSelectiveAttention module
    def __init__(self, d_model, num_heads, max_batch_size, max_seq_len):
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

        self.cache_k = torch.zeros((max_batch_size, num_heads, max_seq_len, self.d_head)).cuda()
        self.cache_v = torch.zeros((max_batch_size, num_heads, max_seq_len, self.d_head)).cuda()

    def forward(self, X, start_pos):
        batch, N, d_model = X.size()
        
        # Normalize the Q and K projections
        # Shape: (batch, N, num_heads * d_head)
        XQ = self.norm_q(self.W_q(X))  
        XK = self.norm_k(self.W_k(X))
        XV = self.W_v(X) 
        
        # Reshape and permute for multi-head attention
        # Shape: (batch, num_heads, N, d_head)
        XQ = XQ.view(batch, N, self.num_heads, self.d_head).transpose(1, 2)
        XK = XK.view(batch, N, self.num_heads, self.d_head).transpose(1, 2)
        XV = XV.view(batch, N, self.num_heads, self.d_head).transpose(1, 2)

        # Cache the K and V for the next iteration
        self.cache_k[:batch, :, start_pos: start_pos + N] = XK
        self.cache_v[:batch, :, start_pos: start_pos + N] = XV

        K = self.cache_k[:batch, :, : start_pos + N]
        V = self.cache_v[:batch, :, : start_pos + N]

        attn_logits = torch.einsum('b h n d, b h m d -> b h n m', XQ, K) / math.sqrt(self.d_head)
        causal_mask = torch.tril(torch.ones(N, N, device=X.device, dtype=bool)).view(1, 1, N, N)
        attn_logits[:, :, :, start_pos:] = torch.where(causal_mask, attn_logits[:, :, :, start_pos:], float('-inf'))

        # Selective Attention Implementation
        S = attn_logits[:, 0]           # Select head 0
        S = F.relu(S)                   # Only positive selection
        S[..., 0] = 0                   # Do not mask <BOS>
        S = (1 - torch.eye(N)) * S      # Do not mask self
        S = torch.roll(S, 1, -2)
        S[..., 0, :] = 0                # Mask strictly in the future
        F_mask = torch.cumsum(S, axis=-2)  # Accumulate
        attn_logits -= F_mask[:, None] 

        attn_weights = F.softmax(attn_logits, dim=-1)  # (batch, num_heads, N, N)
        O = torch.matmul(attn_weights, V)  # (batch, num_heads, N, d_head)
        O = O.transpose(1, 2).reshape(batch, N, self.d_model)  # (batch, N, num_heads * d_head)
        output = self.W_o(O)  # (batch, N, d_model)

        return output
    

class MultiheadSelectiveAttentionWithTokenPruning(nn.Module):
    def __init__(self, d_model, num_heads, max_batch_size, budget):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_head = d_model // num_heads
        self.budget = budget

        self.W_q = nn.Linear(d_model, d_model, bias=False)  # Remove the biases
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)

        self.norm_q = nn.LayerNorm(d_model)
        self.norm_k = nn.LayerNorm(d_model)

        self.cache_k = torch.zeros((max_batch_size, num_heads, budget, self.d_head))
        self.cache_v = torch.zeros((max_batch_size, num_heads, budget, self.d_head))

    def forward(self, X, start_pos):
        batch, N, d_model = X.size()
        
        # Normalize the Q and K projections
        # Shape: (batch, N, num_heads * d_head)
        XQ = self.norm_q(self.W_q(X))
        XK = self.norm_k(self.W_k(X))
        XV = self.W_v(X)
        
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
        S[:, :, start_pos:] = (1 - torch.eye(N)) * S[:, :, start_pos:]    # Do not mask self
        S = torch.roll(S, 1, -2)
        S[..., 0, :] = 0                                # Mask strictly in the future
        F_mask = torch.cumsum(S, axis=-2)               # Accumulate, size: (batch, N, cached_len + N)

        # Token Pruning
        pruning_mask = torch.ones_like(F_mask, dtype=torch.bool)
        num_unmasked = min(self.budget, start_pos + N)
        unmasked_indices = torch.arange(num_unmasked, dtype=int).unsqueeze(0).repeat(batch, 1)  # (batch, budget) or (batch, cached_len + N)
        for i in range(start_pos, start_pos + N):
            if i < self.budget:
                continue
            else:
                print(unmasked_indices)
                new_indices = torch.cat([unmasked_indices, i * torch.ones((batch, 1), dtype=int)], dim=1)
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

        attn_logits = torch.where(pruning_mask.unsqueeze(1).repeat(1, num_heads, 1, 1), attn_logits, float('-inf'))
        attn_weights = F.softmax(attn_logits, dim=-1)  # (batch, num_heads, N, M)
        O = torch.matmul(attn_weights, V)  # (batch, num_heads, N, d_head)
        O = O.transpose(1, 2).reshape(batch, N, self.d_model)  # (batch, N, num_heads * d_head)

        output = self.W_o(O)  # (batch, N, d_model)
        return output


class SwiGLU(nn.Module):
    def __init__(self, d_model, hidden_dim):
        super().__init__()
        self.fc1 = nn.Linear(d_model, hidden_dim, bias=False)
        self.fc2 = nn.Linear(hidden_dim, d_model, bias=False)
        self.gate = nn.Linear(d_model, hidden_dim, bias=False)

    def forward(self, X):
        return self.fc2(F.silu(self.gate(X)) * self.fc1(X))
    

class SelectiveTransformerBlock(nn.Module):
    def __init__(self, layer_id, d_model, num_heads, max_batch_size, budget):
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

    def forward(self, X, start_pos):
        H = X +self.attention(self.attention_norm(X), start_pos)
        out = H + self.feedforward(self.ffn_norm(H))
        return out
    

class SelectiveTransformer(nn.Module):
    def __init__(self, params: ModelArgs):
        super().__init__()
        self.params = params
        self.vocab_size = params.vocab_size
        self.n_layers = params.n_layers

        self.token_embedding = nn.Embedding(params.vocab_size, params.d_model)

        self.layers = nn.ModuleList()
        for layer_id in range(params.n_layers):
            self.layers.append(SelectiveTransformerBlock(layer_id, params.d_model, params.n_heads, params.max_batch_size, params.budget))

        self.norm = nn.RMSNorm(params.d_model)
        self.output_layer = nn.Linear(params.d_model, params.vocab_size, bias=False)

    @torch.inference_mode()
    def forward(self, tokens: torch.Tensor, start_pos: int):
        batch_size, seq_len = tokens.shape
        h = self.token_embedding(tokens)

        for layer in self.layers:
            h = layer(h, start_pos)
        h = self.norm(h)
        output = self.output_layer(h).float()
        return output


if __name__ == "__main__":
    # Test
    d_model = 64
    num_heads = 8
    max_batch_size = 32
    max_seq_len = 64
    budget = 8

    input_tensor = torch.randn((2, 16, d_model))
    input_tensor_2 = torch.randn((2, 8, d_model))

    # Test MultiheadSelectiveAttentionWithTokenPruning
    model = MultiheadSelectiveAttentionWithTokenPruning(d_model, num_heads, max_batch_size, budget)
    # First input
    output = model(input_tensor, 0)
    print(output.shape)  # Expected output: torch.Size([2, 16, 64])
    # Second input
    output_2 = model(input_tensor_2, budget)
    print(output_2.shape)  # Expected output: torch.Size([2, 8, 64])

    # Test SelectiveTransformer
    model_args = ModelArgs()
    model_args.vocab_size = 3200
    model = SelectiveTransformer(model_args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    # Example input
    batch_size = 2
    seq_len = 128
    input_ids = torch.randint(0, model_args.vocab_size, (batch_size, seq_len)).to(device)
    start_pos = 0
    output = model(input_ids, start_pos)
    print(output.shape)  # Expected output: torch.Size([2, 128, 3200])


