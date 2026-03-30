import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple

import config as config
import transformer as transformer
import arch_util as archutil

class RMSNorm(nn.Module):
    """RMSNorm: Root Mean Square Layer Normalization (LLaMA 스타일)"""
    def __init__(self, d_model: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch_size, seq_len, d_model)
        rms = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * rms * self.weight

class SwiGLU(nn.Module):
    """SwiGLU 활성화 함수 (LLaMA에서 사용)"""
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff, bias=False)
        self.w2 = nn.Linear(d_model, d_ff, bias=False)
        self.w3 = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # SwiGLU: x * Swish(xW1 + b1) * (xW2 + b2)
        # 여기서는 Swish 대신 SiLU 사용 (PyTorch의 silu)
        return self.w3(F.silu(self.w1(x)) * self.w2(x))

def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """텐서의 절반을 회전 (RoPE 구현용)"""
    x1, x2 = x[..., :x.shape[-1]//2], x[..., x.shape[-1]//2:]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """RoPE (Rotary Position Embeddings) 적용"""
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

class LLaMAAttention(nn.Module):
    """LLaMA 스타일 어텐션 (RoPE 포함) - 기존 MHA 대신 사용"""
    def __init__(self, d_model: int, n_heads: int, max_position_embeddings: int = 2048):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.scaling = self.head_dim ** -0.5

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)

        # RoPE용 cos, sin 캐시
        self.register_buffer(
            "cos_cached",
            self._precompute_cos_sin(max_position_embeddings, self.head_dim)[0],
            persistent=False
        )
        self.register_buffer(
            "sin_cached",
            self._precompute_cos_sin(max_position_embeddings, self.head_dim)[1],
            persistent=False
        )

    def _precompute_cos_sin(self, max_position_embeddings: int, head_dim: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """RoPE용 cos, sin 값들을 미리 계산"""
        inv_freq = 1.0 / (10000 ** (torch.arange(0, head_dim, 2).float() / head_dim))
        t = torch.arange(max_position_embeddings).type_as(inv_freq)
        freqs = torch.outer(t, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos(), emb.sin()

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        
        # Q, K, V 프로젝션
        q = self.q_proj(x).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)

        # RoPE 적용 - 차원 수정
        cos = self.cos_cached[:seq_len].unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, head_dim)
        sin = self.sin_cached[:seq_len].unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, head_dim)
        # print(f"q shape: {q.shape}, cos shape: {cos.shape}")
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        # 어텐션 계산
        attn_weights = torch.matmul(q, k.transpose(2, 3)) * self.scaling
        
        if mask is not None:
            mask = mask.unsqueeze(1)  # (B, 1, seq_len, seq_len)
            # mask = mask.unsqueeze(0).unsqueeze(0)
            attn_weights = attn_weights.masked_fill(mask, float('-inf'))
        
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_output = torch.matmul(attn_weights, v)
        
        # 출력 재구성
        attn_output = attn_output.permute(0, 2, 1, 3).contiguous().reshape(batch_size, seq_len, self.n_heads * self.head_dim)
        return self.o_proj(attn_output)

class LLaMAFeedForward(nn.Module):
    """LLaMA 스타일 Feed Forward (SwiGLU 사용) - 기존 FF 대신 사용"""
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.swiglu = SwiGLU(d_model, d_ff)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.swiglu(x)

class LLaMASublayerConnection(nn.Module):
    """LLaMA 스타일 Sublayer Connection (RMSNorm 사용) - 기존 SublayerConnection 대신 사용"""
    def __init__(self, d_emb: int, dropout_rate: float):
        super().__init__()
        self.norm = RMSNorm(d_emb)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x: torch.Tensor, sublayer) -> torch.Tensor:
        # LLaMA 스타일: residual connection이 norm 밖에 있음
        residual = x
        x = self.norm(x)
        sub_out = sublayer(x)
        sub_out = self.dropout(sub_out)
        return residual + sub_out

class LLaMADecoderLayer(nn.Module):
    """LLaMA 디코더 레이어 (기존 DecoderLayer 구조 재사용하되 LLaMA 특화 컴포넌트 사용)"""
    def __init__(self, config: config.Config):
        super().__init__()
        self.d_model = config.d_emb
        self.n_heads = config.n_heads_dec_sa
        
        # LLaMA 특화 컴포넌트 사용
        self.input_norm = RMSNorm(config.d_emb)
        self.post_attention_norm = RMSNorm(config.d_emb)
        self.attention = LLaMAAttention(config.d_emb, config.n_heads_dec_sa)
        self.feed_forward = LLaMAFeedForward(config.d_emb, config.d_ff)
        self.dropout = nn.Dropout(config.dropout_rate_dec)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Pre-norm 어텐션 (LLaMA 스타일)
        residual = x
        x = self.input_norm(x)
        x = self.attention(x, mask)
        x = self.dropout(x)
        x = residual + x

        # Pre-norm FFN (LLaMA 스타일)
        residual = x
        x = self.post_attention_norm(x)
        x = self.feed_forward(x)
        x = self.dropout(x)
        x = residual + x

        return x

class LLaMA1(nn.Module):
    """LLaMA 모델 (기존 구조 재활용하되 LLaMA 특화 컴포넌트 사용)"""
    def __init__(self, config: config.Config):
        super().__init__()
        self.ctx_window_dec = config.ctx_window_dec

        self.config = config
        
        # 기존 ELUT 사용 (임베딩)
        self.embed_tokens = transformer.ELUT(config.vocab_size, config.d_emb)
        
        # LLaMA 특화 디코더 레이어들
        self.layers = nn.ModuleList([
            LLaMADecoderLayer(config) for _ in range(config.n_layers_dec)
        ])
        
        # 출력 레이어 (LLaMA 특화)
        self.norm = RMSNorm(config.d_emb)
        self.lm_head = nn.Linear(config.d_emb, config.vocab_size, bias=False)
        
        # 가중치 공유 (기존 ELUT의 lut 사용)
        self.lm_head.weight = self.embed_tokens.lut.weight

    def forward(self, input_ids: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        input_ids = archutil.crop_data_to_ctx_window(input_ids, self.ctx_window_dec)

        # 기존 ELUT 사용
        x = self.embed_tokens(input_ids)
        
        # LLaMA 특화 레이어들 통과
        for layer in self.layers:
            x = layer(x, mask)
        
        # 최종 정규화 및 출력
        x = self.norm(x)
        logits = self.lm_head(x)
        
        return logits 
