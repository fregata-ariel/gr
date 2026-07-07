from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
from flax import linen as nn

from poc0.constants import (
    MAX_SEQ_LEN,
    MODEL_D_MODEL,
    MODEL_DROPOUT_RATE,
    MODEL_HEAD_DIM,
    MODEL_MLP_DIM,
    MODEL_N_HEADS,
    MODEL_N_LAYERS,
    VOCAB_SIZE,
)


@dataclass(frozen=True)
class TransformerConfig:
    vocab_size: int = VOCAB_SIZE
    max_seq_len: int = MAX_SEQ_LEN
    n_layers: int = MODEL_N_LAYERS
    d_model: int = MODEL_D_MODEL
    n_heads: int = MODEL_N_HEADS
    head_dim: int = MODEL_HEAD_DIM
    mlp_dim: int = MODEL_MLP_DIM
    dropout_rate: float = MODEL_DROPOUT_RATE


def make_causal_attention_mask(seq_len: int) -> jax.Array:
    positions = jnp.arange(seq_len)
    return positions[None, None, :, None] >= positions[None, None, None, :]


def make_decoder_attention_mask(input_ids: jax.Array) -> jax.Array:
    return make_causal_attention_mask(int(input_ids.shape[1]))


def count_parameters(params: object) -> int:
    return sum(leaf.size for leaf in jax.tree_util.tree_leaves(params))


class TransformerBlock(nn.Module):
    config: TransformerConfig

    @nn.compact
    def __call__(self, hidden_states: jax.Array, *, train: bool) -> jax.Array:
        config = self.config

        residual = hidden_states
        hidden_states = nn.LayerNorm(
            dtype=jnp.float32,
            param_dtype=jnp.float32,
        )(hidden_states)
        hidden_states = nn.MultiHeadDotProductAttention(
            num_heads=config.n_heads,
            qkv_features=config.n_heads * config.head_dim,
            out_features=config.d_model,
            dropout_rate=config.dropout_rate,
            use_bias=True,
            dtype=jnp.float32,
            param_dtype=jnp.float32,
        )(
            hidden_states,
            mask=make_causal_attention_mask(hidden_states.shape[1]),
            deterministic=not train,
        )
        hidden_states = nn.Dropout(rate=config.dropout_rate)(
            hidden_states,
            deterministic=not train,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = nn.LayerNorm(
            dtype=jnp.float32,
            param_dtype=jnp.float32,
        )(hidden_states)
        hidden_states = nn.Dense(
            config.mlp_dim,
            use_bias=True,
            dtype=jnp.float32,
            param_dtype=jnp.float32,
        )(hidden_states)
        hidden_states = nn.gelu(hidden_states)
        hidden_states = nn.Dense(
            config.d_model,
            use_bias=True,
            dtype=jnp.float32,
            param_dtype=jnp.float32,
        )(hidden_states)
        hidden_states = nn.Dropout(rate=config.dropout_rate)(
            hidden_states,
            deterministic=not train,
        )
        return residual + hidden_states


class CausalTransformerLM(nn.Module):
    config: TransformerConfig

    @nn.compact
    def __call__(self, input_ids: jax.Array, *, train: bool) -> jax.Array:
        config = self.config

        if input_ids.ndim != 2:
            raise ValueError(
                f"input_ids must have shape [batch, seq_len], got {input_ids.shape}"
            )
        if input_ids.shape[1] > config.max_seq_len:
            raise ValueError(
                f"input_ids seq_len exceeds max_seq_len: {input_ids.shape[1]} > {config.max_seq_len}"
            )

        token_embedding = self.param(
            "token_embedding",
            nn.initializers.normal(stddev=0.02),
            (config.vocab_size, config.d_model),
            jnp.float32,
        )
        position_embedding = self.param(
            "position_embedding",
            nn.initializers.normal(stddev=0.02),
            (config.max_seq_len, config.d_model),
            jnp.float32,
        )

        hidden_states = jnp.take(token_embedding, input_ids, axis=0)
        hidden_states = hidden_states + position_embedding[None, : input_ids.shape[1], :]

        for layer_index in range(config.n_layers):
            hidden_states = TransformerBlock(
                config=config,
                name=f"layers_{layer_index}",
            )(hidden_states, train=train)

        hidden_states = nn.LayerNorm(
            dtype=jnp.float32,
            param_dtype=jnp.float32,
            name="final_layer_norm",
        )(hidden_states)
        logits = jnp.einsum("bsd,vd->bsv", hidden_states, token_embedding)
        return logits.astype(jnp.float32)
