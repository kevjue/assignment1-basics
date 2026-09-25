import torch
from einops import einsum, reduce, rearrange
import math
from cs336_basics.utils import softmax


class Linear(torch.nn.Module):
    def __init__(
        self, in_features: int, out_features: int, device: torch.device | None = None, dtype: torch.dtype | None = None
    ):
        super().__init__()
        W = torch.empty(out_features, in_features, device=device, dtype=dtype)
        std = (2.0 / (in_features + out_features)) ** 0.5
        torch.nn.init.trunc_normal_(W, 0, std, -3 * std, 3 * std)
        self.W = torch.nn.Parameter(W)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return einsum(x, self.W, "... d_in, d_out d_in -> ... d_out")


class Embeddings(torch.nn.Module):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        embeddings = torch.empty(num_embeddings, embedding_dim, device=device, dtype=dtype)
        torch.nn.init.trunc_normal_(embeddings, 0, 1, -3, 3)
        self.embeddings = torch.nn.Parameter(embeddings)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.embeddings[token_ids]


class RMSNorm(torch.nn.Module):
    def __init__(
        self, d_model: int, eps: float = 1e-5, device: torch.device | None = None, dtype: torch.dtype | None = None
    ):
        super().__init__()
        self.eps = eps
        gains = torch.ones(d_model, device=device, dtype=dtype)
        self.gains = torch.nn.Parameter(gains)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype

        x = x.to(torch.float32)
        x_2 = x.pow(2)
        rms = x_2.mean(dim=-1, keepdim=True)
        rms = rms + self.eps
        rms = rms.sqrt()
        x = x / rms
        x = x * self.gains

        return x.to(in_dtype)


class SwiGLU(torch.nn.Module):
    def __init__(self, d_model: int, d_ff: int, device: torch.device | None = None, dtype: torch.dtype | None = None):
        super().__init__()
        self.w1 = Linear(d_model, d_ff, device=device, dtype=dtype)
        self.w2 = Linear(d_ff, d_model, device=device, dtype=dtype)
        self.w3 = Linear(d_model, d_ff, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = self.w1.forward(x)
        gate = gate * torch.sigmoid(gate)
        return self.w2.forward(gate * self.w3.forward(x))


class RotaryPositionalEmbedding(torch.nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device: torch.device | None = None):
        super().__init__()

        token_pos_indices = torch.arange(max_seq_len, device=device)[:, None]

        d_pair_pos = torch.arange(1, d_k // 2 + 1, device=device, dtype=torch.float32)[None, :]
        d_pair_pos *= 2
        d_pair_pos -= 2
        d_pair_pos /= d_k
        d_pair_pos = theta**d_pair_pos

        angle = token_pos_indices / d_pair_pos

        rotation_matrix = torch.ones(max_seq_len, d_k // 2, 2, 2, device=device, dtype=torch.float32)
        angle = rearrange(angle, "s k_pair -> s k_pair 1 1")

        rotation_matrix *= angle
        torch.cos_(rotation_matrix[..., 0, 0])
        torch.sin_(rotation_matrix[..., 1, 0])
        torch.sin_(rotation_matrix[..., 0, 1]).neg_()
        torch.cos_(rotation_matrix[..., 1, 1])

        print("rotation matrix is ", rotation_matrix)

        self.register_buffer("rotation_matrix", rotation_matrix, persistent=False)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        x = rearrange(x, "... seq_len (d_k_pair pair) -> ... seq_len d_k_pair pair", pair=2)
        rot = self.rotation_matrix[token_positions]
        ret = einsum(
            x, rot, "... seq_len d_k_pair pair, ... seq_len d_k_pair pair_out pair -> ... seq_len d_k_pair pair_out"
        )
        return rearrange(ret, "... seq_len d_k_pair pair_out -> ... seq_len (d_k_pair pair_out)")


class MultiHeadSelfAttention(torch.nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        rope: RotaryPositionalEmbedding | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        d_k = d_model // num_heads
        d_v = d_k

        self.num_heads = num_heads
        self.rope = rope

        self.w_q = Linear(d_k * num_heads, d_model, device=device, dtype=dtype)
        self.w_k = Linear(d_k * num_heads, d_model, device=device, dtype=dtype)
        self.w_v = Linear(d_v * num_heads, d_model, device=device, dtype=dtype)
        self.w_o = Linear(d_model, d_v * num_heads, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q = self.w_q.forward(x)
        k = self.w_k.forward(x)
        v = self.w_v.forward(x)

        seq_len = x.shape[-2]
        mask = torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool), diagonal=1)

        q = rearrange(q, "... seq_len (num_heads d_q) -> ... num_heads seq_len d_q", num_heads=self.num_heads)
        k = rearrange(k, "... seq_len (num_heads d_k) -> ... num_heads seq_len d_k", num_heads=self.num_heads)
        v = rearrange(v, "... seq_len (num_heads d_v) -> ... num_heads seq_len d_v", num_heads=self.num_heads)

        if self.rope is not None:
            pos = torch.arange(seq_len)
            q = self.rope.forward(q, pos)
            k = self.rope.forward(k, pos)

        att = scaled_dot_product_attention(q, k, v, mask)
        att = rearrange(att, "... num_heads seq_len v_d -> ... seq_len (num_heads v_d)")
        return self.w_o.forward(att)


def scaled_dot_product_attention(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor | None = None
) -> torch.Tensor:
    dot_products = einsum(q, k, "... q_seq_len d_q, ... k_seq_len d_q -> ... q_seq_len k_seq_len")
    dot_products /= math.sqrt(q.shape[-1])
    if mask is not None:
        dot_products = dot_products.masked_fill(mask, float("-inf"))
    soft_maxes = softmax(dot_products, -1)

    return einsum(soft_maxes, v, "... q_seq_len k_seq_len, ... k_seq_len v_d -> ... q_seq_len v_d")


class TransformerBlock(torch.nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        theta: float,
        max_seq_len: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()

        self.attn_pre_norm = RMSNorm(d_model, device=device, dtype=dtype)

        rope = RotaryPositionalEmbedding(theta, d_model // num_heads, max_seq_len)
        self.attn = MultiHeadSelfAttention(d_model, num_heads, rope=rope, device=device, dtype=dtype)

        self.mlp_pre_norm = RMSNorm(d_model, device=device, dtype=dtype)
        self.mlp = SwiGLU(d_model, d_ff, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_res = x + self.attn(self.attn_pre_norm(x))
        mlp_res = attn_res + self.mlp(self.mlp_pre_norm(attn_res))

        return mlp_res


class TransformerLM(torch.nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        num_heads: int,
        d_ff: int,
        theta: float,
        max_seq_len: int,
        num_layers: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()

        self.embeddings = Embeddings(vocab_size, d_model, device=device, dtype=dtype)

        self.layers = torch.nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(
                TransformerBlock(d_model, num_heads, d_ff, theta, max_seq_len, device=device, dtype=dtype)
            )

        self.final_norm = RMSNorm(d_model, device=device, dtype=dtype)
        self.final_linear = Linear(d_model, vocab_size, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input = self.embeddings.forward(x)
        for tfer in self.layers:
            input = tfer.forward(input)
        return self.final_linear.forward(self.final_norm.forward(input))
