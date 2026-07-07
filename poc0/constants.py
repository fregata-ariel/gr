from __future__ import annotations

from typing import Final

MAX_SEQ_LEN: Final[int] = 80
PTR_COUNT: Final[int] = 32
TRAIN_SPLIT_FRACTION: Final[float] = 0.95

PAD_TOKEN: Final[str] = "PAD"
BOS_TOKEN: Final[str] = "BOS"

_FIXED_TOKENS: Final[tuple[str, ...]] = (
    PAD_TOKEN,
    BOS_TOKEN,
    "ADD_ENTRY",
    "ADD_LINEAR",
    "ADD_MERGE",
    "ADD_LOOP",
    "OPEN",
    "CLOSE",
    "STOP",
)
_POINTER_TOKENS: Final[tuple[str, ...]] = tuple(
    f"ptr_{index}" for index in range(PTR_COUNT)
)
VOCAB: Final[tuple[str, ...]] = _FIXED_TOKENS + _POINTER_TOKENS
VOCAB_SIZE: Final[int] = len(VOCAB)

TOKEN_TO_ID: Final[dict[str, int]] = {
    token: token_id for token_id, token in enumerate(VOCAB)
}
ID_TO_TOKEN: Final[dict[int, str]] = {
    token_id: token for token, token_id in TOKEN_TO_ID.items()
}

PAD_ID: Final[int] = TOKEN_TO_ID[PAD_TOKEN]
BOS_ID: Final[int] = TOKEN_TO_ID[BOS_TOKEN]

SPLIT_SEED: Final[int] = 20260707
TRAIN_SHUFFLE_SEED: Final[int] = 20260708
INIT_SEED: Final[int] = 20260709
SAMPLING_SEED: Final[int] = 20260710

MODEL_N_LAYERS: Final[int] = 4
MODEL_D_MODEL: Final[int] = 256
MODEL_N_HEADS: Final[int] = 4
MODEL_HEAD_DIM: Final[int] = 64
MODEL_MLP_DIM: Final[int] = 1024
MODEL_ACTIVATION: Final[str] = "gelu"
MODEL_DROPOUT_RATE: Final[float] = 0.0
MODEL_TRAIN_DTYPE: Final[str] = "float32"

OPTIMIZER_NAME: Final[str] = "adamw"
ADAM_BETA1: Final[float] = 0.9
ADAM_BETA2: Final[float] = 0.95
ADAM_EPS: Final[float] = 1e-8
WEIGHT_DECAY: Final[float] = 0.01
GLOBAL_GRAD_CLIP_NORM: Final[float] = 1.0
PEAK_LEARNING_RATE: Final[float] = 3e-4
END_LEARNING_RATE: Final[float] = 3e-5
WARMUP_STEPS: Final[int] = 200
TRAIN_STEPS: Final[int] = 3000
BATCH_SIZE: Final[int] = 256
EVAL_EVERY_STEPS: Final[int] = 100
CHECKPOINT_EVERY_STEPS: Final[int] = 500
CHECKPOINT_KEEP_LAST: Final[int] = 3
DEFAULT_SAMPLING_TEMPERATURE: Final[float] = 1.0
