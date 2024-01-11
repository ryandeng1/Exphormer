import numpy as np
import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter
from torch.nn.init import constant_, xavier_normal_, xavier_uniform_

from torch_geometric.graphgym.models.layer import LayerConfig
from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.register import register_layer

from einops import rearrange
from torch import Tensor, nn
# import xformers.ops as xops
from typing import Callable, Optional, Sequence, Tuple, Union

from math import log

# Efficient implementation equivalent to the following:
def scaled_dot_product_attention(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None) -> torch.Tensor:
    # Efficient implementation equivalent to the following:
    L, S = query.size(-2), key.size(-2)
    scale_factor = 1 / math.sqrt(query.size(-1)) if scale is None else scale
    attn_bias = torch.zeros(L, S, dtype=query.dtype, device=query.device)
    if is_causal:
        assert attn_mask is None
        temp_mask = torch.ones(L, S, dtype=torch.bool).tril(diagonal=0)
        attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))
        attn_bias.to(query.dtype)

    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            attn_mask.masked_fill_(attn_mask.logical_not(), float("-inf"))
        else:
            attn_bias += attn_mask
    attn_weight = query @ key.transpose(-2, -1) * scale_factor
    attn_weight += attn_bias
    attn_weight = torch.softmax(attn_weight, dim=-1)
    attn_weight = torch.dropout(attn_weight, dropout_p, train=True)
    return attn_weight @ value


class DilatedAttention(nn.Module):
    """Implement dilated, scaled dot product attention with softmax.
    Arguments
    ---------
        softmax_scale: The temperature to use for the softmax attention.
                      (default: 1/sqrt(d_keys) where d_keys is computed at
                      runtime)
        attention_dropout: The dropout rate to apply to the attention
                           (default: 0.0)
    """

    def __init__(
        self,
        segment_lengths: Sequence[int],
        dilation_rates: Sequence[int],
        softmax_scale: Optional[float] = None,
        attention_dropout: float = 0.0,
    ):
        super().__init__()
        if len(segment_lengths) != len(dilation_rates):
            raise ValueError(
                "segment_lengths and dilation_rates must have the same length"
            )

        self.segment_lengths = segment_lengths
        self.dilation_rates = dilation_rates
        self.softmax_scale = softmax_scale
        self.dropout_p = attention_dropout
        self.op = None

    def forward(
        self, query: Tensor, key: Tensor, value: Tensor, is_causal: bool = False
    ) -> Tensor:
        # Notation:
        #   b - batch size
        #   n - sequence length
        #   h - number of heads
        #   d - embedding dimension
        #   s - segment length
        #   r - dilation rate
        #   g - group size (i.e. number of heads per segment length)
        #
        # Input shape of query, key, value: (b, n, h, d)

        # apply some permutation to q, k, v
        b, _, h, _ = query.shape
        out = torch.zeros_like(query)

        # *** NOTE ***
        # The original paper does not describe how to handle the case where
        #   h % len(self.segment_lengths) != 0
        #
        # In my first implementation, I naively assumed (and asserted) that
        # 'h % len(self.segment_lengths) == 0', so that I could evenly distribute
        # the heads between the different segment lengths. However, it was not
        # possible to reproduce the LongNet hyperparameters with that restriction:
        #   h=12, segment_lengths=[2048, 4096, 8192, 16384, 32768]
        #   h % len(segment_lengths) == 2
        #
        # For that reason, I have removed the assertion, and instead grouped the heads
        # into (potentially) unequally sized groups.  If not perfectly divisible, then
        # the first few groups will have an extraattention head.
        num_groups = len(self.dilation_rates)
        group_sizes = [h // num_groups] * num_groups
        for i in range(h % num_groups):
            group_sizes[i] += 1

        for i, (g, r, s) in enumerate(
            zip(group_sizes, self.dilation_rates, self.segment_lengths)
        ):
            # Split the input sequences into segments of length 'self.segment_length'
            q = rearrange(query, "b (n s) h d -> b n s h d", s=s)
            k = rearrange(key, "b (n s) h d -> b n s h d", s=s)
            v = rearrange(value, "b (n s) h d -> b n s h d", s=s)
            # Apply dilation and segment offset
            offset = i % r
            hmin = i * g
            hmax = (i + 1) * g
            q = q[:, :, offset::r, hmin:hmax, :]
            k = k[:, :, offset::r, hmin:hmax, :]
            v = v[:, :, offset::r, hmin:hmax, :]
            # Fold all 'n' segments into the batch dimension
            q = rearrange(q, "b n s h d -> (b n) s h d")
            k = rearrange(k, "b n s h d -> (b n) s h d")
            v = rearrange(v, "b n s h d -> (b n) s h d")

            # Apply memory efficient attention
            # NOTE: If flash attention is correctly installed, then this will also
            # automatically use the flash attention implementation.
            # attn_bias = xops.LowerTriangularMask() if is_causal else None
            # x = torch.nn.functional.scaled_dot_product_attention(query=q, key=k, value=v, dropout_p=self.dropout_p)
            x = scaled_dot_product_attention(query=q, key=k, value=v, dropout_p=self.dropout_p)
            """
            x = xops.memory_efficient_attention(
                query=q, key=k, value=v, op=self.op, attn_bias=attn_bias
            )
            """
            # Unfold 'n' segments back out of the batch dimension.
            x = rearrange(x, "(b n) s h d -> b n s h d", b=b)
            # Normalize attention outputs across the sequence length dimension. This
            # is necessary because the attention outputs from each dilation rate /
            # segment length are summed together.
            x = x / x.sum(dim=(1, 2), keepdim=True)

            # Gather the attention outputs from each dilation rate / segment length.
            out = rearrange(out, "b (n s) h d -> b n s h d", s=s)
            out[:, :, offset::r, hmin:hmax, :] += x
            out = rearrange(out, "b n s h d -> b (n s) h d", s=s)

        # We have already normalized each attention output across the sequence length.
        # Now, normalize across all attention outputs by dividing by the number of
        # attention groups.  See: https://arxiv.org/pdf/2307.02486.pdf, Eq. 10
        return out / num_groups


class MultiheadDilatedAttention(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dilation_rates: Sequence[int],
        segment_lengths: Sequence[int],
        dropout: float = 0.0,
        bias: bool = True,
        layer_norm: bool = True,
        layer_norm_eps: float = 1e-5,
        gamma_init: float = 1.0,
        device: Optional[Union[torch.device, str]] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.layer_norm = layer_norm
        self.gamma_init = gamma_init

        if not embed_dim % self.num_heads == 0:
            raise ValueError(
                f"embed_dim ({embed_dim}) must be divisible by "
                f"num_heads ({num_heads})"
            )
        num_dilations = len(dilation_rates)
        num_segments = len(segment_lengths)
        if num_dilations != num_segments:
            raise ValueError(
                f"len(dilation_rates) ({num_dilations}) must be equal to "
                f"len(segment_lengths) ({num_segments})"
            )
        head_dim = embed_dim // num_heads
        if not head_dim % 8 == 0:
            raise ValueError(
                f"head_dim (embed_dim / num_heads = {head_dim}) must be divisible by 8"
            )
        if not head_dim <= 128:
            raise ValueError(
                f"head_dim (embed_dim / num_heads = {head_dim}) must be <= 128"
            )

        self.q_proj = nn.Linear(
            embed_dim, embed_dim, bias=bias, device=device, dtype=dtype
        )
        self.k_proj = nn.Linear(
            embed_dim, embed_dim, bias=bias, device=device, dtype=dtype
        )
        self.v_proj = nn.Linear(
            embed_dim, embed_dim, bias=bias, device=device, dtype=dtype
        )
        self.attention = DilatedAttention(
            segment_lengths=segment_lengths,
            dilation_rates=dilation_rates,
            attention_dropout=dropout,
        )
        self.norm: Optional[nn.LayerNorm] = None
        if layer_norm:
            self.norm = nn.LayerNorm(
                embed_dim, eps=layer_norm_eps, device=device, dtype=dtype
            )
        self.out_proj = nn.Linear(
            embed_dim, embed_dim, bias=bias, device=device, dtype=dtype
        )

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.xavier_normal_(self.q_proj.weight)
        if self.q_proj.bias is not None:
            nn.init.constant_(self.q_proj.bias, 0)
        nn.init.xavier_normal_(self.k_proj.weight)
        if self.k_proj.bias is not None:
            nn.init.constant_(self.k_proj.bias, 0)

        # NOTE: We follow the initialization strategy from MAGNETO.  See:
        # https://arxiv.org/pdf/2210.06423.pdf, Fig. 2
        # Gain (self.gamma_init) should be provided as a keyword argument when
        # initializing the larger Transformer model, since it requires knowledge
        # of the number of encoder/decoder layers in the model.

        nn.init.xavier_normal_(self.v_proj.weight, gain=self.gamma_init)
        if self.v_proj.bias is not None:
            nn.init.constant_(self.v_proj.bias, 0)
        nn.init.xavier_normal_(self.out_proj.weight, gain=self.gamma_init)
        if self.out_proj.bias is not None:
            nn.init.constant_(self.out_proj.bias, 0)

    def forward(
        self, query: Tensor, key: Tensor, value: Tensor, is_causal: bool = False
    ) -> Tuple[Tensor, None]:
        # Notation:
        #   b - batch size
        #   n - sequence length
        #   h - number of heads
        #   d - embedding dimension
        #
        # Input shape: (b, n, d)
        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)

        # Unfold 'd' dimension into 'h' separate attention heads.
        q = rearrange(q, "b n (h d) -> b n h d", h=self.num_heads)
        k = rearrange(k, "b n (h d) -> b n h d", h=self.num_heads)
        v = rearrange(v, "b n (h d) -> b n h d", h=self.num_heads)
        # Apply attention, then fold 'h' attention heads back into 'd'.
        x = self.attention(q, k, v, is_causal=is_causal)
        x = rearrange(x, "b n h d -> b n (h d)")

        # NOTE: This is different from 'nn.MultiheadAttention'! The LongNet paper
        # follows the MAGNETO architecture, which applies an extra layer norm
        # before the linear output projection.  The cross-attention layer in the
        # MAGNETO decoder does not include this layer norm, so users have the option
        # to disable it (layer_norm=False).
        if self.layer_norm:
            assert self.norm is not None
            x = self.norm(x)
        # Linear projection on attention outputs.
        x = self.out_proj(x)

        return x

class DilatedTransformerEncoderLayer(nn.Module):
    # NOTE: Mostly pulled from 'nn.TransformerEncoderLayer', but with changes:
    #   - use sub-LayerNorm like in MAGNETO. See: https://arxiv.org/abs/2210.06423
    #   - use MultiheadDilatedAttention instead of MultiheadAttention

    def __init__(
        self,
        d_model: int,
        nhead: int,
        segment_lengths: Sequence[int],
        dilation_rates: Sequence[int],
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation = F.relu,
        layer_norm_eps: float = 1e-5,
        gamma_init: float = 1.0,
        device: Optional[Union[torch.device, str]] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        super().__init__()
        # Legacy string support for activation function.
        if isinstance(activation, str):
            activation = _get_activation_fn(activation)

        self.activation = activation
        self.gamma_init = gamma_init

        self.dropout = nn.Dropout(dropout)
        # Self-attention block
        self.norm1 = nn.LayerNorm(
            d_model, eps=layer_norm_eps, device=device, dtype=dtype
        )
        self.self_attn = MultiheadDilatedAttention(  # type: ignore
            embed_dim=d_model,
            num_heads=nhead,
            dilation_rates=dilation_rates,
            segment_lengths=segment_lengths,
            dropout=dropout,
            layer_norm=True,
            layer_norm_eps=layer_norm_eps,
            gamma_init=gamma_init,
            device=device,
            dtype=dtype,
        )
        # Feedforward block
        self.norm2 = nn.LayerNorm(
            d_model, eps=layer_norm_eps, device=device, dtype=dtype
        )
        self.linear1 = nn.Linear(d_model, dim_feedforward, device=device, dtype=dtype)
        self.norm3 = nn.LayerNorm(
            dim_feedforward, eps=layer_norm_eps, device=device, dtype=dtype
        )
        self.linear2 = nn.Linear(dim_feedforward, d_model, device=device, dtype=dtype)

        self._reset_parameters()

    def _reset_parameters(self):
        # NOTE: We follow the initialization strategy from MAGNETO.  See:
        # https://arxiv.org/pdf/2210.06423.pdf, Fig. 2
        # The 'MultiheadDilatedAttention' module uses ths same initialization,
        # so we just need to worry about the 'Linear' modules here.
        nn.init.xavier_normal_(self.linear1.weight, gain=self.gamma_init)
        nn.init.constant_(self.linear1.bias, 0)
        nn.init.xavier_normal_(self.linear2.weight, gain=self.gamma_init)
        nn.init.constant_(self.linear2.bias, 0)

    def _self_attention_block(self, x: Tensor, is_causal: bool = False) -> Tensor:
        x = self.norm1(x)
        x = self.self_attn(x, x, x, is_causal=is_causal)
        x = self.dropout(x)
        return x

    def _feedforward_block(self, x: Tensor) -> Tensor:
        x = self.norm2(x)
        x = self.activation(self.linear1(x))
        x = self.dropout(x)
        x = self.norm3(x)
        x = self.linear2(x)
        x = self.dropout(x)
        return x

    def forward(self, src: Tensor, is_causal: bool = False) -> Tensor:
        x = src
        x = x + self._self_attention_block(x, is_causal=is_causal)
        x = x + self._feedforward_block(x)
        return x

class DilatedTransformerDecoderLayer(nn.Module):
    # NOTE: Mostly pulled from 'nn.TransformerDecoderLayer', but with changes:
    #   - use sub-LayerNorm like in MAGNETO. See: https://arxiv.org/abs/2210.06423
    #   - use MultiheadDilatedAttention instead of MultiheadAttention

    def __init__(
        self,
        d_model: int,
        nhead: int,
        segment_lengths: Sequence[int],
        dilation_rates: Sequence[int],
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation = F.relu,
        layer_norm_eps: float = 1e-5,
        gamma_init: float = 1.0,
        device = None,
        dtype = None,
    ) -> None:
        super().__init__()
        # Legacy string support for activation function.
        if isinstance(activation, str):
            activation = _get_activation_fn(activation)

        self.activation = activation
        self.gamma_init = gamma_init

        self.dropout = nn.Dropout(dropout)
        # Self-attention block
        self.norm1 = nn.LayerNorm(
            d_model, eps=layer_norm_eps, device=device, dtype=dtype
        )
        self.self_attn = MultiheadDilatedAttention(  # type: ignore
            embed_dim=d_model,
            num_heads=nhead,
            dilation_rates=dilation_rates,
            segment_lengths=segment_lengths,
            dropout=dropout,
            layer_norm=False,
            gamma_init=gamma_init,
            device=device,
            dtype=dtype,
        )
        # Multi-head attention block
        self.norm2 = nn.LayerNorm(
            d_model, eps=layer_norm_eps, device=device, dtype=dtype
        )
        self.multihead_attn = MultiheadDilatedAttention(  # type: ignore
            embed_dim=d_model,
            num_heads=nhead,
            dilation_rates=dilation_rates,
            segment_lengths=segment_lengths,
            dropout=dropout,
            layer_norm_eps=layer_norm_eps,
            gamma_init=gamma_init,
            device=device,
            dtype=dtype,
        )
        # Feedforward block
        self.norm3 = nn.LayerNorm(
            d_model, eps=layer_norm_eps, device=device, dtype=dtype
        )
        self.linear1 = nn.Linear(d_model, dim_feedforward, device=device, dtype=dtype)
        self.norm4 = nn.LayerNorm(
            dim_feedforward, eps=layer_norm_eps, device=device, dtype=dtype
        )
        self.linear2 = nn.Linear(dim_feedforward, d_model, device=device, dtype=dtype)

        self._reset_parameters()

    def _reset_parameters(self):
        # NOTE: We follow the initialization strategy from MAGNETO.  See:
        # https://arxiv.org/pdf/2210.06423.pdf, Fig. 2
        # The 'MultiheadDilatedAttention' module uses ths same initialization,
        # so we just need to worry about the 'Linear' modules here.
        nn.init.xavier_normal_(self.linear1.weight, gain=self.gamma_init)
        nn.init.constant_(self.linear1.bias, 0)
        nn.init.xavier_normal_(self.linear2.weight, gain=self.gamma_init)
        nn.init.constant_(self.linear2.bias, 0)

    def _self_attention_block(self, x: Tensor, is_causal: bool = False) -> Tensor:
        x = self.norm1(x)
        x = self.self_attn(x, x, x, is_causal=is_causal)
        x = self.dropout(x)
        return x

    def _multihead_attention_block(
        self, x: Tensor, memory: Tensor, is_causal: bool = False
    ) -> Tensor:
        x = self.norm2(x)
        x = self.multihead_attn(x, memory, memory, is_causal=is_causal)
        x = self.dropout(x)
        return x

    def _feedforward_block(self, x: Tensor) -> Tensor:
        x = self.norm3(x)
        x = self.activation(self.linear1(x))
        x = self.dropout(x)
        x = self.norm4(x)
        x = self.linear2(x)
        x = self.dropout(x)
        return x

    def forward(
        self,
        tgt: Tensor,
        memory: Tensor,
        tgt_is_causal: bool = False,
        memory_is_causal: bool = False,
    ) -> Tensor:
        x = tgt
        x = x + self._self_attention_block(x, is_causal=tgt_is_causal)
        x = x + self._multihead_attention_block(x, memory, is_causal=memory_is_causal)
        x = x + self._feedforward_block(x)
        return x

class LongNet(nn.Module):
    """These are the *base* LongNet hyperparameters taken from the paper.  See:
    https://arxiv.org/pdf/2307.02486.pdf, Section 4.1 & Appendix A
    """

    def __init__(
        self,
        d_model: int = 768,
        nhead: int = 12,
        num_encoder_layers: int = 12,
        num_decoder_layers: int = 12,
        dim_feedforward: int = 3072,
        segment_lengths: Sequence[int] = [2048, 4096, 8192, 16384, 32768],
        dilation_rates: Sequence[int] = [1, 2, 4, 6, 12],
        dropout: float = 0.0,
        activation = F.relu,
        layer_norm_eps: float = 1e-5,
        device = None,
        dtype = None,
    ):
        super().__init__()
        # The 'gamma_init' parameters are different for the encoder and decoder,
        # and depend on the number of encoder/decoder layers.  See MAGNETO paper:
        # https://arxiv.org/pdf/2210.06423.pdf, Figure 2
        encoder_gamma_init = (
            log(3 * num_decoder_layers) * log(2 * num_encoder_layers) / 3
        ) ** 0.5
        decoder_gamma_init = log(3 * num_decoder_layers) ** 0.5

        self.encoder = nn.TransformerEncoder(
            encoder_layer=DilatedTransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                segment_lengths=segment_lengths,
                dilation_rates=dilation_rates,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                activation=activation,
                layer_norm_eps=layer_norm_eps,
                gamma_init=encoder_gamma_init,
                device=device,
                dtype=dtype,
            ),
            num_layers=num_encoder_layers,
            # mask_check=False,
            # enable_nested_tensor=False,
        )
        self.decoder = nn.TransformerDecoder(
            decoder_layer=DilatedTransformerDecoderLayer(
                d_model=d_model,
                nhead=nhead,
                segment_lengths=segment_lengths,
                dilation_rates=dilation_rates,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                activation=activation,
                layer_norm_eps=layer_norm_eps,
                gamma_init=decoder_gamma_init,
                device=device,
                dtype=dtype,
            ),
            num_layers=num_decoder_layers,
        )

    def forward(self, x: Tensor, is_causal: bool = True) -> Tensor:
        """
        Input shape: (batch_size, seq_len, d_model)
        Output shape: (batch_size, seq_len, d_model)

        NOTE: Assume that 'is_causal' applies to both the encoder and decoder.
        We're primarily interested in causal attention for language modeling, which is
        what was discussed in the LongNet paper.  But in principle, leave the option
        open for other applications.
        """
        tgt = x
        for layer in self.encoder.layers:
            x = layer(x, is_causal=is_causal)
        if self.encoder.norm is not None:
            x = self.encoder.norm(x)

        mem = x
        for layer in self.decoder.layers:
            tgt = layer(tgt, mem, memory_is_causal=is_causal, tgt_is_causal=is_causal)
        if self.decoder.norm is not None:
            tgt = self.decoder.norm(tgt)

        return tgt

class LongNetLM(nn.Module):
    def __init__(
        self,
        num_tokens: int,
        d_model: int = 768,
        nhead: int = 12,
        num_encoder_layers: int = 12,
        num_decoder_layers: int = 12,
        dim_feedforward: int = 3072,
        segment_lengths: Sequence[int] = [2048, 4096, 8192, 16384, 32768],
        dilation_rates: Sequence[int] = [1, 2, 4, 6, 12],
        dropout: float = 0.0,
        activation = F.relu,
        layer_norm_eps: float = 1e-5,
        device = None,
        dtype = None,
    ):
        super().__init__()
        self.token_embedding = nn.Embedding(
            num_tokens, d_model, device=device, dtype=dtype
        )
        self.pos_embedding = XPOS(d_model).to(device=device, dtype=dtype)
        self.long_net = LongNet(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            segment_lengths=segment_lengths,
            dilation_rates=dilation_rates,
            dropout=dropout,
            activation=activation,
            layer_norm_eps=layer_norm_eps,
            device=device,
            dtype=dtype,
        )
        self.norm = nn.LayerNorm(
            d_model, eps=layer_norm_eps, device=device, dtype=dtype
        )
        self.out = nn.Linear(d_model, num_tokens, device=device, dtype=dtype)

    def _reset_parameters(self):
        nn.init.kaiming_normal_(self.out.weight)
        nn.init.constant_(self.out.bias, 0)

    def forward(self, x: Tensor, is_causal: bool = True) -> Tensor:
        x = self.token_embedding(x)
        x = x + self.pos_embedding(x)
        x = self.long_net(x, is_causal=is_causal)
        x = self.norm(x)
        return self.out(x)

register_layer('Dilated', MultiheadDilatedAttention)
register_layer('DilatedTransformer', DilatedTransformerEncoderLayer)
