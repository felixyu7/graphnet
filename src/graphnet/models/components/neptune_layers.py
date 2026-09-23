"""Layer classes for the Neptune point-transformer backbone."""

from typing import Any, Callable, List, Optional, Sequence, Tuple, cast

import torch
import torch.nn as nn
from torch.functional import Tensor

from pytorch_lightning import LightningModule

from graphnet.models.components.attention_blocks import DropPath
from graphnet.models.utils import flex_attention


class RMSNorm(LightningModule):
    """Root-mean-square layer normalization.

    Equivalent to `torch.nn.RMSNorm`, but the weight is cast to the input
    dtype before the call. The standard-library module keeps its weight in
    float32, which forces an unfused path under autocast; casting lets the
    fused bfloat16/float16 kernel fire instead.
    """

    def __init__(self, dim: int, eps: float = 1e-5):
        """Construct `RMSNorm`.

        Args:
            dim: Size of the normalized (last) dimension.
            eps: Term added to the denominator for numerical stability.
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps
        self.normalized_shape = (dim,)

    def forward(self, x: Tensor) -> Tensor:
        """Normalize `x` over its last dimension."""
        return torch.nn.functional.rms_norm(
            x, self.normalized_shape, self.weight.to(x.dtype), self.eps
        )


class SwiGLU(LightningModule):
    """Feed-forward block with a SwiGLU gate.

    The two input projections are fused into a single `Linear`, which is
    both faster and keeps the parameter count identical to the unfused form.
    """

    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        dropout: float = 0.1,
        bias: bool = True,
    ):
        """Construct `SwiGLU`.

        Args:
            dim: Input and output dimension.
            hidden_dim: Inner dimension of the gated projection.
            dropout: Dropout applied to the block output.
            bias: Whether the projections carry a bias.
        """
        super().__init__()
        self.w13 = nn.Linear(dim, 2 * hidden_dim, bias=bias)
        self.w2 = nn.Linear(hidden_dim, dim, bias=bias)
        self.dropout = nn.Dropout(dropout)
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        """Apply Xavier-uniform weights and zero biases."""
        for module in (self.w13, self.w2):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, x: Tensor) -> Tensor:
        """Apply the gated feed-forward transform."""
        a, b = self.w13(x).chunk(2, dim=-1)
        return self.dropout(self.w2(torch.nn.functional.silu(a) * b))


class RoPE4D(LightningModule):
    """Rotary position embedding over four coordinates `(x, y, z, t)`.

    Implements the standard axis-aligned 4D construction: each coordinate
    axis is given its own rotation planes, so the four generators stay
    linearly independent. Planes are allocated one per axis first and then
    round-robin, which is what guarantees that independence for any even
    `dim >= 8`.

    Pair `j` rotates dimensions `(j, dim / 2 + j)` via a contiguous
    `chunk`/`cat` on the last axis; each side of the rotation is then a
    contiguous half, which is markedly faster than a strided
    `(2j, 2j + 1)` layout.

    See https://arxiv.org/abs/2504.06308.
    """

    freqs: Tensor
    coord_select: Tensor

    def __init__(
        self,
        dim: int,
        scales: Sequence[float] = (1.0, 1.0, 1.0, 1.0),
        base: int = 10000,
    ):
        """Construct `RoPE4D`.

        Args:
            dim: Head dimension. Must be even and at least 8.
            scales: Per-axis frequency scale for `(x, y, z, t)`, in radians
                per coordinate unit.
            base: Ratio between the lowest and highest frequency in each
                axis' band, i.e. frequencies span `scale / base` to `scale`.
        """
        super().__init__()
        if dim % 2 != 0:
            raise ValueError(f"RoPE4D dim must be even (got {dim})")
        if dim < 8:
            raise ValueError(
                f"RoPE4D requires dim >= 8 for 4 axes (got {dim})"
            )
        if len(scales) != 4:
            raise ValueError(f"RoPE4D expects 4 scales (got {len(scales)})")
        self.dim = dim
        self.scales = tuple(scales)
        self.base = base

        num_planes = dim // 2

        # One plane per axis first (ensures linear independence), then
        # distribute what is left round-robin.
        allocation = [1, 1, 1, 1]
        for i in range(num_planes - 4):
            allocation[i % 4] += 1

        all_freqs = [
            self._build_freqs(n_planes, scale)
            for n_planes, scale in zip(allocation, self.scales)
        ]
        self.register_buffer("freqs", torch.cat(all_freqs))  # (dim / 2,)

        coord_select: List[int] = []
        for axis_idx, n_planes in enumerate(allocation):
            coord_select.extend([axis_idx] * n_planes)
        self.register_buffer(
            "coord_select", torch.tensor(coord_select, dtype=torch.long)
        )

    def _build_freqs(self, num_bands: int, scale: float) -> Tensor:
        """Build log-spaced frequency bands for one axis."""
        if num_bands == 1:
            return torch.tensor([1.0 / self.base]) * scale
        exponents = torch.arange(num_bands, dtype=torch.float32) / (
            num_bands - 1
        )
        freqs = (1.0 / self.base) * (self.base**exponents)
        return freqs * scale

    def compute_tables(
        self, coords: Tensor, dtype: Optional[torch.dtype] = None
    ) -> Tuple[Tensor, Tensor]:
        """Precompute the `(cos, sin)` rotation tables for `coords`.

        Every layer of an encoder stack sees the same coordinates and the
        same RoPE configuration, so the tables are computed once and shared.

        Args:
            coords: `[B, S, 4]` coordinates.
            dtype: Optional dtype to cast the tables to. Trig is always
                evaluated in float32 for accuracy; casting once here lets the
                per-layer rotation stay in the model dtype.

        Returns:
            A `(cos, sin)` pair, each of shape `[B, 1, S, dim / 2]`.
        """
        coords_f = coords if coords.dtype == torch.float32 else coords.float()
        coord_per_plane = coords_f.index_select(
            dim=-1, index=self.coord_select
        )
        angles = (coord_per_plane * self.freqs).unsqueeze(1)
        cos_a = torch.cos(angles)
        sin_a = torch.sin(angles)
        if dtype is not None and dtype != cos_a.dtype:
            cos_a = cos_a.to(dtype)
            sin_a = sin_a.to(dtype)
        return cos_a, sin_a

    def forward(
        self,
        x: Tensor,
        coords: Tensor,
        tables: Optional[Tuple[Tensor, Tensor]] = None,
    ) -> Tensor:
        """Rotate queries or keys according to their coordinates.

        The rotation is `(a, b) -> (a cos t - b sin t, a sin t + b cos t)`
        for pair `j = (j, dim / 2 + j)`.

        Args:
            x: `[B, H, S, dim]` queries or keys.
            coords: `[B, S, 4]` coordinates `(x, y, z, t)`.
            tables: Optional precomputed `(cos, sin)` pair from
                :meth:`compute_tables`, avoiding recomputation of the
                transcendentals for every layer and for both Q and K.

        Returns:
            The rotated tensor, shaped like `x`.
        """
        if tables is None:
            tables = self.compute_tables(coords, dtype=x.dtype)
        cos_a, sin_a = tables

        # Fast path: same-dtype tables, so stay entirely in x's dtype.
        if cos_a.dtype == x.dtype:
            x1, x2 = x.chunk(2, dim=-1)
            return torch.cat(
                (x1 * cos_a - x2 * sin_a, x1 * sin_a + x2 * cos_a), dim=-1
            )

        # Mixed-dtype fallback: do the maths in float32 and cast back.
        x_f = x if x.dtype == torch.float32 else x.float()
        x1, x2 = x_f.chunk(2, dim=-1)
        out = torch.cat(
            (x1 * cos_a - x2 * sin_a, x1 * sin_a + x2 * cos_a), dim=-1
        )
        return out if out.dtype == x.dtype else out.to(x.dtype)


class AttentionPool(LightningModule):
    """Pool a sequence to one vector, attending from a learned query."""

    def __init__(self, dim: int):
        """Construct `AttentionPool`.

        Args:
            dim: Feature dimension of the sequence and of the output.
        """
        super().__init__()
        self.q = nn.Parameter(torch.randn(1, 1, dim) * (dim**-0.5))
        self.kv = nn.Linear(dim, 2 * dim, bias=False)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        """Pool `x` to a single vector per batch element.

        Args:
            x: `[B, S, D]` sequence.
            mask: Optional bool `[B, S]`; True marks a valid element.

        Returns:
            `[B, D]` pooled features, zero for fully-masked rows.
        """
        batch_size, _, n_features = x.shape
        q = self.q.expand(batch_size, -1, -1)
        k, v = self.kv(x).chunk(2, dim=-1)

        attn = torch.bmm(q, k.transpose(1, 2)) * (n_features**-0.5)
        if mask is not None:
            # A finite fill rather than -inf: a fully-masked row (a zero-hit
            # event) must soft-max to finite garbage, which is zeroed below.
            # -inf would yield NaN and poison the whole batch.
            attn = attn.masked_fill(
                ~mask.unsqueeze(1), torch.finfo(attn.dtype).min
            )
        attn = torch.nn.functional.softmax(attn, dim=-1)
        out = torch.bmm(attn, v).squeeze(1)
        out = self.proj(out)
        if mask is not None:
            out = out * mask.any(dim=1, keepdim=True).to(out.dtype)
        return out


class NeptuneTransformerEncoderLayer(LightningModule):
    """Pre-norm transformer encoder layer with 4D rotary attention.

    Used by :class:`~graphnet.models.transformer.neptune.Neptune`. Compared
    with a vanilla encoder layer it uses RMSNorm instead of LayerNorm, a
    fused QKV projection, per-head query/key normalization before the rotary
    embedding, LayerScale on both residual branches, and a SwiGLU
    feed-forward network.

    Attention-matrix dropout is deliberately unused: `flex_attention` cannot
    express it, so omitting it keeps the packed and padded attention paths
    regularized identically. Residual/feed-forward dropout and stochastic
    depth still apply.
    """

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int,
        dropout: float = 0.1,
        layer_norm_eps: float = 1e-5,
        bias: bool = True,
        ff_bias: bool = False,
        qk_norm: bool = True,
        rope_scales: Sequence[float] = (180.0, 180.0, 180.0, 40.0),
        rope_base: int = 60,
        drop_path_rate: float = 0.0,
        layerscale_init: float = 1e-5,
    ):
        """Construct `NeptuneTransformerEncoderLayer`.

        Args:
            d_model: Token dimension.
            nhead: Number of attention heads. `d_model / nhead` must be even
                and at least 8, as required by :class:`RoPE4D`.
            dim_feedforward: Inner dimension of the SwiGLU network.
            dropout: Dropout on the residual and feed-forward branches.
            layer_norm_eps: Epsilon of the RMSNorm layers.
            bias: Whether the attention projections carry a bias.
            ff_bias: Whether the feed-forward projections carry a bias.
            qk_norm: Whether to RMS-normalize queries and keys per head
                before applying the rotary embedding.
            rope_scales: Per-axis rotary frequency scales for `(x, y, z, t)`.
            rope_base: Rotary frequency span, see :class:`RoPE4D`.
            drop_path_rate: Stochastic-depth rate for this layer.
            layerscale_init: Initial value of the LayerScale parameters.
        """
        super().__init__()
        if d_model % nhead != 0:
            raise ValueError("d_model must be divisible by nhead")
        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead
        if self.head_dim % 2 != 0:
            raise ValueError(
                f"head_dim must be even for RoPE4D (got {self.head_dim})"
            )

        self.qkv_proj = nn.Linear(d_model, 3 * d_model, bias=bias)
        self.out_proj = nn.Linear(d_model, d_model, bias=bias)
        self._initialize_weights()

        self.norm1 = RMSNorm(d_model, eps=layer_norm_eps)
        self.norm2 = RMSNorm(d_model, eps=layer_norm_eps)

        self.qk_norm = qk_norm
        if qk_norm:
            self.q_norm = RMSNorm(self.head_dim, eps=layer_norm_eps)
            self.k_norm = RMSNorm(self.head_dim, eps=layer_norm_eps)

        self.ffn = SwiGLU(d_model, dim_feedforward, dropout, bias=ff_bias)
        self.rope = RoPE4D(
            dim=self.head_dim, scales=rope_scales, base=rope_base
        )

        # LayerScale: learnable per-channel gain on each residual branch.
        self.gamma_1 = nn.Parameter(layerscale_init * torch.ones(d_model))
        self.gamma_2 = nn.Parameter(layerscale_init * torch.ones(d_model))

        self.dropout = nn.Dropout(dropout)
        self.drop_path1 = DropPath(drop_path_rate)
        self.drop_path2 = DropPath(drop_path_rate)

    def _initialize_weights(self) -> None:
        """Apply Xavier-uniform weights and zero biases."""
        nn.init.xavier_uniform_(self.qkv_proj.weight)
        if self.qkv_proj.bias is not None:
            nn.init.zeros_(self.qkv_proj.bias)
        nn.init.xavier_uniform_(self.out_proj.weight)
        if self.out_proj.bias is not None:
            nn.init.zeros_(self.out_proj.bias)

    def _attn(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        attn_mask: Optional[Tensor],
        block_mask: Any,
    ) -> Tensor:
        """Dispatch to packed block-diagonal flex attention or padded SDPA."""
        if block_mask is not None:
            return flex_attention(q, k, v, block_mask=block_mask)
        return torch.nn.functional.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, dropout_p=0.0, is_causal=False
        )

    def prepare_attention_mask(
        self, key_padding_mask: Optional[Tensor], device: torch.device
    ) -> Optional[Tensor]:
        """Convert a key-padding mask to the boolean SDPA convention.

        Args:
            key_padding_mask: Bool `[B, S]` where True marks *padding*, as in
                `torch.nn.MultiheadAttention`.
            device: Device to build the mask on.

        Returns:
            Bool `[B, 1, 1, S]` where True marks positions that *may* be
            attended, or None if no mask was given.
        """
        if key_padding_mask is None:
            return None
        allow = ~key_padding_mask.to(torch.bool).to(device)
        # A fully-padded row (a zero-hit event) would soft-max over all -inf
        # and NaN-poison the batch; let it attend everywhere instead. Pooling
        # zeroes those rows afterwards, matching the packed path.
        allow = allow | ~allow.any(dim=-1, keepdim=True)
        return allow.unsqueeze(1).unsqueeze(2)

    def forward(
        self,
        src: Tensor,
        centroids: Tensor,
        src_key_padding_mask: Optional[Tensor] = None,
        rope_tables: Optional[Tuple[Tensor, Tensor]] = None,
        attn_mask: Optional[Tensor] = None,
        block_mask: Any = None,
        doc_id: Optional[Tensor] = None,
        num_docs: Optional[int] = None,
    ) -> Tensor:
        """Apply one encoder layer.

        Args:
            src: `[B, S, d_model]` token features.
            centroids: `[B, S, 4]` token coordinates driving the rotary
                embedding.
            src_key_padding_mask: Bool `[B, S]` where True marks padding.
                Only used when neither `attn_mask` nor `block_mask` is given.
            rope_tables: Optional shared `(cos, sin)` tables.
            attn_mask: Optional prebuilt boolean SDPA mask.
            block_mask: Optional `flex_attention` `BlockMask` selecting the
                packed path.
            doc_id: Optional `[N]` event index per token, on the packed path.
            num_docs: Number of events in `doc_id`.

        Returns:
            `[B, S, d_model]` updated token features.
        """
        batch_size, seq_length, _ = src.shape

        x = src
        x_norm = self.norm1(x)

        qkv = self.qkv_proj(x_norm)
        qkv = qkv.reshape(batch_size, seq_length, 3, self.nhead, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, H, S, D)
        q, k, v = qkv[0], qkv[1], qkv[2]

        if self.qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)

        q = self.rope(q, centroids, tables=rope_tables)
        k = self.rope(k, centroids, tables=rope_tables)

        # Use the mask prebuilt by the encoder when available; otherwise
        # build one here for single-layer callers. Skipped entirely on the
        # packed path, which masks via `block_mask`.
        if attn_mask is None and block_mask is None:
            attn_mask = self.prepare_attention_mask(
                src_key_padding_mask, q.device
            )

        attn_output = self._attn(q, k, v, attn_mask, block_mask)
        attn_output = attn_output.transpose(1, 2)  # (B, S, H, D)
        attn_output = attn_output.contiguous().view(
            batch_size, seq_length, self.d_model
        )
        attn_output = self.out_proj(attn_output)

        x = x + self.drop_path1(
            self.dropout(self.gamma_1 * attn_output), doc_id, num_docs
        )

        x_norm = self.norm2(x)
        ff_output = self.ffn(x_norm)
        x = x + self.drop_path2(self.gamma_2 * ff_output, doc_id, num_docs)

        return x


class NeptuneTransformerEncoder(LightningModule):
    """Stack of :class:`NeptuneTransformerEncoderLayer`.

    Layers are built by a factory so each depth gets its own stochastic-depth
    rate, ramped linearly from 0 to `drop_path_rate`. Because every layer
    shares the same coordinates and rotary configuration, the `(cos, sin)`
    tables and the attention mask are built once here and threaded through
    the stack.
    """

    def __init__(
        self,
        layer_factory: Callable[[float], NeptuneTransformerEncoderLayer],
        num_layers: int,
        norm: Optional[LightningModule] = None,
        drop_path_rate: float = 0.0,
    ):
        """Construct `NeptuneTransformerEncoder`.

        Args:
            layer_factory: Callable mapping a stochastic-depth rate to a
                fresh :class:`NeptuneTransformerEncoderLayer`.
            num_layers: Number of layers in the stack.
            norm: Optional module applied to the stack output.
            drop_path_rate: Stochastic-depth rate of the final layer; earlier
                layers are scaled linearly towards zero.
        """
        super().__init__()
        if num_layers <= 0:
            raise ValueError("num_layers must be positive")
        if num_layers == 1:
            drop_rates = [drop_path_rate]
        else:
            drop_rates = [
                drop_path_rate * float(i) / (num_layers - 1)
                for i in range(num_layers)
            ]
        self.layers = nn.ModuleList(
            [layer_factory(rate) for rate in drop_rates]
        )
        self.num_layers = num_layers
        self.norm = norm

    def forward(
        self,
        src: Tensor,
        centroids: Tensor,
        src_key_padding_mask: Optional[Tensor] = None,
        block_mask: Any = None,
        doc_id: Optional[Tensor] = None,
        num_docs: Optional[int] = None,
    ) -> Tensor:
        """Run the encoder stack.

        Args:
            src: `[B, S, d_model]` token features.
            centroids: `[B, S, 4]` token coordinates.
            src_key_padding_mask: Bool `[B, S]` where True marks padding.
            block_mask: Optional `flex_attention` `BlockMask` for the packed
                path.
            doc_id: Optional `[N]` event index per token, on the packed path.
            num_docs: Number of events in `doc_id`.

        Returns:
            `[B, S, d_model]` encoded tokens.
        """
        first = cast(NeptuneTransformerEncoderLayer, self.layers[0])
        # Pre-cast the tables to src.dtype so the per-layer rotation skips a
        # float32 round-trip on bfloat16/float16 paths.
        rope_tables = first.rope.compute_tables(centroids, dtype=src.dtype)
        attn_mask = (
            first.prepare_attention_mask(src_key_padding_mask, src.device)
            if block_mask is None
            else None
        )

        output = src
        for layer in self.layers:
            output = layer(
                output,
                centroids,
                rope_tables=rope_tables,
                attn_mask=attn_mask,
                block_mask=block_mask,
                doc_id=doc_id,
                num_docs=num_docs,
            )

        if self.norm is not None:
            output = self.norm(output)

        return output
