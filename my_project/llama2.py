import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple

import config as config
import transformer as transformer
import llama1 as llama1
import arch_util as archutil

class GroupedQueryAttention(nn.Module):
    """Grouped Query Attention (GQA) - Llama 2 스타일"""
    def __init__(self, d_model: int, n_heads: int, n_groups: int = 8, max_position_embeddings: int = 2048):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_groups = n_groups
        self.head_dim = d_model // n_heads
        self.scaling = self.head_dim ** -0.5
        
        # GQA: n_groups는 KV 헤드의 수, n_heads는 Q 헤드의 수
        self.n_kv_heads = n_groups
        self.n_q_heads = n_heads
        
        # 각 그룹당 쿼리 헤드 수
        self.num_queries_per_kv = self.n_q_heads // self.n_kv_heads
        
        # 프로젝션 레이어들
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, self.n_kv_heads * self.head_dim, bias=False)  # KV는 그룹 수만큼만
        self.v_proj = nn.Linear(d_model, self.n_kv_heads * self.head_dim, bias=False)
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
        q = self.q_proj(x).view(batch_size, seq_len, self.n_q_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.n_kv_heads, self.head_dim).transpose(1, 2)

        # RoPE 적용
        cos = self.cos_cached[:seq_len].unsqueeze(0).unsqueeze(0)
        sin = self.sin_cached[:seq_len].unsqueeze(0).unsqueeze(0)
        q, k = llama1.apply_rotary_pos_emb(q, k, cos, sin)

        # GQA: K, V를 Q 헤드 수에 맞게 복제
        # k: (batch_size, n_kv_heads, seq_len, head_dim)
        # v: (batch_size, n_kv_heads, seq_len, head_dim)
        # -> (batch_size, n_q_heads, seq_len, head_dim)
        
        # K, V를 그룹별로 복제
        k = k.repeat_interleave(self.num_queries_per_kv, dim=1)
        v = v.repeat_interleave(self.num_queries_per_kv, dim=1)

        # 어텐션 계산
        attn_weights = torch.matmul(q, k.transpose(2, 3)) * self.scaling
        
        if mask is not None:
            mask = mask.unsqueeze(1)  # (B, 1, seq_len, seq_len)
            # mask = mask.unsqueeze(0).unsqueeze(0)
            attn_weights = attn_weights.masked_fill(mask, float('-inf'))
        
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_output = torch.matmul(attn_weights, v)
        
        # 출력 재구성
        attn_output = attn_output.permute(0, 2, 1, 3).contiguous().reshape(batch_size, seq_len, self.d_model)
        return self.o_proj(attn_output)

class LLaMA2DecoderLayer(nn.Module):
    """LLaMA 2 디코더 레이어 (GQA 사용)"""
    def __init__(self, config: config.Config, n_groups: int = 8):
        super().__init__()
        self.d_model = config.d_emb
        self.n_heads = config.n_heads_dec_sa
        
        # LLaMA 2 특화 컴포넌트 사용
        self.input_norm = llama1.RMSNorm(config.d_emb)
        self.post_attention_norm = llama1.RMSNorm(config.d_emb)
        self.attention = GroupedQueryAttention(config.d_emb, config.n_heads_dec_sa, n_groups, config.ctx_window_dec)
        self.feed_forward = llama1.LLaMAFeedForward(config.d_emb, config.d_ff)
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

class LLaMA2(nn.Module):
    """LLaMA 2 모델 (GQA 사용)"""
    def __init__(self, config: config.Config, n_groups: int = 8):
        super().__init__()

        self.ctx_window_dec = config.ctx_window_dec

        self.config = config

        if config.n_groups is None:
            n_groups = config.n_heads_dec_sa
        else:
            # 설정 파일에 n_groups 값이 있으면 그 값을 사용합니다.
            n_groups = config.n_groups
        
        # 기존 ELUT 사용 (임베딩)
        self.embed_tokens = transformer.ELUT(config.vocab_size, config.d_emb)
        
        # LLaMA 2 특화 디코더 레이어들 (GQA 사용)
        self.layers = nn.ModuleList([
            LLaMA2DecoderLayer(config, n_groups) for _ in range(config.n_layers_dec)
        ])
        
        # 출력 레이어 (LLaMA 특화)
        self.norm = llama1.RMSNorm(config.d_emb)
        self.lm_head = nn.Linear(config.d_emb, config.vocab_size, bias=False)
        
        # 가중치 공유 (기존 ELUT의 lut 사용)
        self.lm_head.weight = self.embed_tokens.lut.weight

    def forward(self, input_ids: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        input_ids = archutil.crop_data_to_ctx_window(input_ids, self.ctx_window_dec)

        # 기존 ELUT 사용
        x = self.embed_tokens(input_ids)
        
        # LLaMA 2 특화 레이어들 통과 (GQA 사용)
        for layer in self.layers:
            x = layer(x, mask)
        
        # 최종 정규화 및 출력
        x = self.norm(x)
        logits = self.lm_head(x)
        
        return logits 
