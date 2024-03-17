'''
Full definition of a decoder-only transformer language model.

Reference:
1) https://github.com/mistralai/mistral-src
'''
import math
from einops import rearrange, repeat
import logging
from munch import Munch, munchify
from pathlib import Path
import tomllib
from typing import Optional, List

import torch
import torch.nn as nn
from torch.nn import functional as F
from xformers.ops.fmha import memory_efficient_attention

from cache import CacheView, RotatingBufferCache
from utils import apply_rotary_emb, repeat_kv, precompute_freqs_cis
from utils import SimpleInputMetadata


# Attention layer
class Attention(nn.Module):
    def __init__(self, config: Munch) -> None:
        super().__init__()
        assert config.n_embed % config.n_head == 0
        self.config = config

        self.n_head: int = config.n_head
        self.n_kv_head: int = config.n_kv_head
        self.repeats = self.n_head // self.n_kv_head

        self.head_dim: int = config.head_dim
        self.scale = self.head_dim ** -0.5

        self.q = nn.Linear(
            config.block_size, config.n_head * self.head_dim, bias=False
        )
        self.k = nn.Linear(
            config.block_size, config.n_kv_heads * self.head_dim, bias=False
        )
        self.v = nn.Linear(
            config.block_size, config.n_kv_heads * self.head_dim, bias=False
        )
        self.o = nn.Linear(
            config.n_head * self.head_dim, config.block_size, bias=False
        )

    def forward(
        self,
        x: torch.Tensor,
        freq_cis: torch.Tensor,
        cache: Optional[CacheView],
    ) -> torch.Tensor:
        B, T = x.shape

        xq, xk, xv = self.q(x), self.k(x), self.v(x)
        xq = xq.view(B, self.n_head, self.head_dim)
        xk = xk.view(B, self.n_kv_head, self.head_dim)
        xv = xv.view(B, self.n_kv_head, self.head_dim)
        xq, xk = apply_rotary_emb(xq, xk, freq_cis)

        # TODO: read about the KV cache
        if cache is None:
            k, v = xk, xv
        elif cache.prefill:
            k, v = cache.interleave_kv(xk, xv)
            cache.update(xk, xv)
        else:
            cache.update(xk, xv)
            k, v = cache.key, cache.value
            key = key.view(
                B * cache.sliding_window, self.n_kv_head, self.head_dim
            )
            val = val.view(
                B * cache.sliding_window, self.n_kv_head, self.head_dim
            )

        k, v = repeat_kv(k, v, self.repeats, dim=1)

        xq, k, v = xq[None, ...], k[None, ...], v[None, ...]
        output = memory_efficient_attention(
            xq, k, v, None if cache is None else cache.mask
        )

        return self.o(output.view(B, self.n_head * self.head_dim))


# Feed-forward layer
class FeedForward(nn.Module):
    def __init__(self, config: Munch):
        super().__init__()

        self.w1 = nn.Linear(config.block_size, config.hidden_dim, bias=False)
        self.w2 = nn.Linear(config.hidden_dim, config.block_size, bias=False)
        self.w3 = nn.Linear(config.block_size, config.hidden_dim, bias=False)

    def forward(self, x) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


# RMSNorm layer
class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


# MoE layer
class MoeLayer(nn.Module):
    def __init__(
        self,
        experts: List[nn.Module],
        gate: nn.Module,
        n_expert: int = 8,
        n_expert_per_token: int = 2,
    ):
        super().__init__()
        assert len(experts) > 0
        self.experts = nn.ModuleList(experts)
        self.gate = gate
        self.n_experts = n_expert
        self.n_experts_per_tok = n_expert_per_token

    def forward(self, inputs: torch.Tensor):
        gate_logits = self.gate(inputs)
        weights, selected_experts = torch.topk(
            gate_logits, self.n_experts_per_tok
        )
        weights = F.softmax(weights, dim=1, dtype=torch.float).to(inputs.dtype)
        results = torch.zeros_like(inputs)

        for i, expert in enumerate(self.experts):
            batch_idx, nth_expert = torch.where(selected_experts == i)
            results[batch_idx] += weights[batch_idx, nth_expert, None] * expert(
                inputs[batch_idx]
            )
        return results


# Transformer block
class TransformerBlock(nn.Module):
    def __init__(self, config:Munch) -> None:
        super().__init__()
        self.config = config

        self.block_size = config.block_size
        self.n_head = config.n_head

        self.attention = Attention(config)
        self.attention_norm = RMSNorm(config.block_size, eps=config.rmsnorm_eps)
        self.ffn_norm = RMSNorm(config.block_size, eps=config.rmsnorm_eps)

        self.ffn = FeedForward(config) if not config.use_moe else MoeLayer(
            experts = [FeedForward(config) for _ in range(config.moe.n_expert)],
            gate = nn.Linear(config.block_size, config.moe.n_expert),
            n_expert = config.moe.n_expert,
            n_expert_per_token = config.moe.n_expert_per_token,
        )

    def forward(
        self,
        x: torch.Tensor,
        freq_cis: torch.Tensor,
        cache: Optional[CacheView],
    ) -> torch.Tensor:
        r = self.attention(self.attention_norm(x), freq_cis, cache)
        h = x + r
        r = self.ffc(self.ffn_norm(h))
        o = h + r
        return o


# Decoder-only Transformer model
class Transformer(nn.Module):
    def __init__(
        self,
        config: Munch,
        pipeline_rank: int = 0,
        n_pipeline_rank: int = 1,
    ) -> None:
        super().__init__()
        self.config = config
        self.vocab_size = config.vocab_size
        self.n_layer = config.n_layer
        self._percomputed_freqs_cis: Optional[torch.Tensor] = None

        assert pipeline_rank < n_pipeline_rank, (
            f'pipeline_rank {pipeline_rank} ≮ n_pipeline_rank {n_pipeline_rank}'
        )
        self.pipeline_rank = pipeline_rank
        self.n_pipeline_rank = n_pipeline_rank

        # Initialise general modules as None
        self.vocab_embed: Optional[nn.Embedding] = None
        self.norm: Optional[RMSNorm] = None
        self.output: Optional[nn.Linear] = None

        # Update modules specific to some ranks
        if pipeline_rank == 0:
            self.vocab_embed = nn.Embedding(
                config.vocab_size, config.block_size
            )
        if pipeline_rank == n_pipeline_rank - 1:
            self.norm = RMSNorm(config.block_size, eps=config.rmsnorm_eps)
            self.output = nn.Linear(
                config.block_size, config.vocab_size, bias = False
            )
    
        # Initialise all layers but slice off those not of this rank
        layers = [TransformerBlock(config) for _ in range(config.n_layer)]
        n_layers_per_rank = math.ceil(config.n_layer / n_pipeline_rank)
        offset = pipeline_rank * n_layers_per_rank
        end = min(self.n_layer, offset + n_layers_per_rank)
        self.layers = nn.ModuleDict({
            str(i): layer for i, layer in enumerate(layers[offset:end])
        })
        self.n_local_layers = len(self.layers)

    @property
    def dtype(self) -> torch.dtype:
        return next(self.parameters()).dtype

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    @property
    def freqs_cis(self) -> torch.Tensor:
        # We cache freqs_cis but need to take care that it is on the right 
        # device and has the right dtype (complex64). The fact that the dtype 
        # is different from the module's  dtype means we cannot register it as 
        # a buffer
        if self._percomputed_freqs_cis is None:
            # If no sliding window, assume a larger sequence length
            theta = self.config.rope.theta
            if theta is None:
                theta = 1e6 if self.config.sliding_window > 0 else 1e4
            self._percomputed_freqs_cis = precompute_freqs_cis(
                self.config.head_dim, 128_000, theta
            )

        if self._percomputed_freqs_cis != self.device:
            self._percomputed_freqs_cis = self._percomputed_freqs_cis.to(
                device = self.device, dtype = self.dtype
            )
        return self._percomputed_freqs_cis

    def forward_partial(
        self,
        input_ids: torch.Tensor,
        seq_lens: List[int],
        cache: Optional[RotatingBufferCache] = None,
    ) -> torch.Tensor:
        """Local forward pass.

        If doing pipeline parallelism, this will return the activations of the
        last layer. For the last stage, this will return the normalised final
        embeddings.
        """
        assert len(seq_lens) < self.config.max_batch_size, (
            f'Max batch size is {self.config.max_batch_size}, ' + \
            f'got {len(seq_lens)}'
        )

        (n_tokens,) = input_ids.shape
        assert sum(seq_lens) == n_tokens, (
            f'Sum of seq_lens {sum(seq_lens)} != n_tokens {n_tokens}'
        )

        if cache is not None:
            input_metadata = cache.get_input_metadata(seq_lens)
        else:
            input_metadata = SimpleInputMetadata.from_seqlens(
                seq_lens, self.device
            )

        if self.pipeline_rank == 0:
            assert self.vocab_embed is not None
            h = self.vocab_embed(input_ids)
        else:
            h = torch.empty(
                n_tokens, self.config.block_size, device = self.device,
                dtype=self.dtype
            )
            torch.distributed.recv(h, src = self.pipeline_rank - 1)
        freqs_cis = self.freqs_cis[input_metadata.positions]

        for local_layer_id, layer in enumerate(self.layers.values()):
            if cache is not None:
                assert input_metadata is not None
                cache_view = cache.get_view(local_layer_id, input_metadata)
            else:
                cache_view = None
            h = layer(h, freqs_cis, cache_view)

        if cache is not None:
            cache.update_seqlens(seq_lens)
        if self.pipeline_rank < self.n_pipeline_rank - 1:
            torch.distributed.send(h, dst = self.pipeline_rank + 1)
            return h
        else:
            # Lask rank has a final normalisation step
            assert self.norm is not None
            return self.norm(h)

    def forward(
        self,
        input_ids: torch.Tensor,
        seq_lens: List[int],
        cache: Optional[RotatingBufferCache] = None,
    ) -> torch.Tensor:
        h = self.forward_partial(input_ids, seq_lens, cache)

        if self.pipeline_rank < self.n_pipeline_rank - 1:
            # ignore the intermediate activations as we'll get the final output 
            # from the last stage
            outs = torch.empty(
                h.shape[0], self.vocab_size, device=h.device, dtype=h.dtype
            )
        else:
            assert self.output is not None
            outs = self.output(h)

        if self.n_pipeline_rank > 1:
            torch.distributed.broadcast(outs, src=self.n_pipeline_rank - 1)

        return outs.float()

    def load_state_dict(self, state_dict, *args, **kwargs):
        state_to_load = {}
        skipped = set([])
        for k, v in state_dict.items():
            if k.startswith("vocab_embed"):
                if self.pipeline_rank == 0:
                    state_to_load[k] = v
                else:
                    logging.debug(
                        'Skipping parameter %s at pipeline rank %d',
                        k,
                        self.pipeline_rank,
                    )
                    skipped.add(k)
            elif k.startswith("norm") or k.startswith("output"):
                if self.pipeline_rank == self.num_pipeline_ranks - 1:
                    state_to_load[k] = v
                else:
                    logging.debug(
                        'Skipping parameter %s at pipeline rank %d',
                        k,
                        self.pipeline_rank,
                    )
                    skipped.add(k)
            elif k.startswith("layers"):
                layer_id = k.split(".")[1]
                if layer_id in self.layers:
                    state_to_load[k] = v
                else:
                    logging.debug(
                        'Skipping parameter %s at pipeline rank %d',
                        k,
                        self.pipeline_rank,
                    )
                    skipped.add(k)
            else:
                raise ValueError(f'Unexpected key {k}')
        assert set(state_dict.keys()) == skipped.union(
            set(state_to_load.keys())
        )
        super().load_state_dict(state_to_load, *args, **kwargs)

    @staticmethod
    def from_folder(
        folder: Path,
        max_batch_size: int = 1,
        n_pipeline_rank: int = 1,
        device="cuda",
        dtype=torch.float16,
    ) -> "Transformer":
        with open(folder / "model_arch.toml", "r") as f:
            model_config = munchify(tomllib.load(f))
        model_config.max_batch_size = max_batch_size
        if n_pipeline_rank > 1:
            pipeline_rank = torch.distributed.get_rank()
        else:
            pipeline_rank = 0
        with torch.device("meta"):
            model = Transformer(
                model_config,
                pipeline_rank=pipeline_rank,
                n_pipeline_rank=n_pipeline_rank,
            )
        loaded = torch.load(str(folder / "consolidated.00.pth"), mmap=True)
        model.load_state_dict(loaded, assign=True)
        return model.to(device=device, dtype=dtype)
