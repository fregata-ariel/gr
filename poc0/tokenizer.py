from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from poc0.constants import BOS_ID, BOS_TOKEN, ID_TO_TOKEN, MAX_SEQ_LEN, PAD_ID, PAD_TOKEN, TOKEN_TO_ID


@dataclass(frozen=True)
class TokenExample:
    input_ids: np.ndarray
    target_ids: np.ndarray
    loss_mask: np.ndarray


def token_to_id(token: str) -> int:
    if token in TOKEN_TO_ID:
        return TOKEN_TO_ID[token]

    if token.startswith("ptr_"):
        suffix = token.removeprefix("ptr_")
        if suffix.isdigit():
            raise ValueError(f"Pointer token out of range: {token}")

    raise ValueError(f"Unknown token: {token}")


def id_to_token(token_id: int) -> str:
    try:
        return ID_TO_TOKEN[token_id]
    except KeyError as exc:
        raise ValueError(f"Unknown token id: {token_id}") from exc


def tokens_to_ids(tokens: Iterable[str]) -> list[int]:
    return [token_to_id(token) for token in tokens]


def ids_to_tokens(token_ids: Iterable[int]) -> list[str]:
    return [id_to_token(token_id) for token_id in token_ids]


def record_to_example(raw_tokens: list[str]) -> TokenExample:
    if not raw_tokens:
        raise ValueError("Raw record must not be empty")

    for token in raw_tokens:
        if token == BOS_TOKEN or token == PAD_TOKEN:
            raise ValueError(f"Raw record must not include special token: {token}")

    raw_token_ids = tokens_to_ids(raw_tokens)
    full_token_ids = [BOS_ID, *raw_token_ids]

    if len(full_token_ids) > MAX_SEQ_LEN:
        raise ValueError(
            "Record exceeds MAX_SEQ_LEN after BOS prepend: "
            f"{len(full_token_ids)} > {MAX_SEQ_LEN}"
        )

    input_ids = np.full(MAX_SEQ_LEN, PAD_ID, dtype=np.int32)
    target_ids = np.full(MAX_SEQ_LEN, PAD_ID, dtype=np.int32)
    loss_mask = np.zeros(MAX_SEQ_LEN, dtype=np.bool_)

    if len(full_token_ids) > 1:
        active_length = len(full_token_ids) - 1
        input_ids[:active_length] = full_token_ids[:-1]
        target_ids[:active_length] = full_token_ids[1:]
        loss_mask[:active_length] = True

    return TokenExample(
        input_ids=input_ids,
        target_ids=target_ids,
        loss_mask=loss_mask,
    )
