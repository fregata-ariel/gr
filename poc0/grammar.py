from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np

from poc0.constants import PTR_COUNT, TOKEN_TO_ID, VOCAB, VOCAB_SIZE


class _Phase(str, Enum):
    ITEM_TOP = "ITEM_TOP"
    ITEM_CHILD = "ITEM_CHILD"
    PARENTS_NONLOOP = "PARENTS_NONLOOP"
    PARENTS_LOOP = "PARENTS_LOOP"
    EXPECT_CLOSE = "EXPECT_CLOSE"
    DONE = "DONE"


_ADD_NONLOOP_TOKENS = ("ADD_ENTRY", "ADD_LINEAR", "ADD_MERGE")
_ITEM_BOUNDARY_TOKENS = (*_ADD_NONLOOP_TOKENS, "ADD_LOOP", "STOP")
_ITEM_BOUNDARY_IDS = tuple(TOKEN_TO_ID[token] for token in _ITEM_BOUNDARY_TOKENS)
_OPEN_ID = TOKEN_TO_ID["OPEN"]
_CLOSE_ID = TOKEN_TO_ID["CLOSE"]


@dataclass(frozen=True)
class GrammarTracker:
    stack: tuple[int, ...]
    phase: _Phase
    pending_n_items: int | None = None
    previous_ptr: int = -1

    @classmethod
    def initial(cls) -> GrammarTracker:
        return cls(stack=(0,), phase=_Phase.ITEM_TOP)

    @property
    def is_terminal(self) -> bool:
        return self.phase == _Phase.DONE

    def legal_token_ids(self) -> tuple[int, ...]:
        return tuple(
            token_id
            for token_id in range(VOCAB_SIZE)
            if self.is_legal_token_id(token_id)
        )

    def legal_mask(self) -> np.ndarray:
        return np.asarray(
            [self.is_legal_token_id(token_id) for token_id in range(VOCAB_SIZE)],
            dtype=np.bool_,
        )

    def is_legal_token(self, token: str) -> bool:
        token_id = TOKEN_TO_ID.get(token)
        if token_id is None:
            return False
        return self.is_legal_token_id(token_id)

    def is_legal_token_id(self, token_id: int) -> bool:
        if token_id < 0 or token_id >= VOCAB_SIZE:
            return False
        return self._is_legal_vocab_token(VOCAB[token_id])

    def advance_token(self, token: str) -> GrammarTracker:
        if not self.is_legal_token(token):
            raise ValueError(f"illegal token {token!r} for phase {self.phase.value}")
        return self._advance_legal_token(token)

    def advance_token_id(self, token_id: int) -> GrammarTracker:
        if token_id < 0 or token_id >= VOCAB_SIZE:
            raise ValueError(f"illegal token id {token_id} for phase {self.phase.value}")
        return self.advance_token(VOCAB[token_id])

    def _is_legal_vocab_token(self, token: str) -> bool:
        ptr_value = _pointer_value(token)

        if self.phase in (_Phase.ITEM_TOP, _Phase.ITEM_CHILD):
            return token in _ITEM_BOUNDARY_TOKENS
        if self.phase == _Phase.PARENTS_NONLOOP:
            if ptr_value is not None:
                return self._ptr_is_legal(ptr_value)
            return token in _ITEM_BOUNDARY_TOKENS
        if self.phase == _Phase.PARENTS_LOOP:
            if ptr_value is not None:
                return self._ptr_is_legal(ptr_value)
            return token == "OPEN"
        if self.phase == _Phase.EXPECT_CLOSE:
            return token == "CLOSE"
        return False

    def _ptr_is_legal(self, ptr_value: int) -> bool:
        if self.pending_n_items is None:
            return False
        return self.previous_ptr < ptr_value < self.pending_n_items

    def _advance_legal_token(self, token: str) -> GrammarTracker:
        if self.phase == _Phase.ITEM_TOP:
            return self._advance_item(token, item_phase=_Phase.ITEM_TOP)
        if self.phase == _Phase.ITEM_CHILD:
            return self._advance_item(token, item_phase=_Phase.ITEM_CHILD)
        if self.phase == _Phase.PARENTS_NONLOOP:
            return self._advance_parents_nonloop(token)
        if self.phase == _Phase.PARENTS_LOOP:
            return self._advance_parents_loop(token)
        if self.phase == _Phase.EXPECT_CLOSE:
            return self._item_phase_for_stack(self.stack)
        raise ValueError("tracker is terminal")

    def _advance_item(self, token: str, *, item_phase: _Phase) -> GrammarTracker:
        current_n_items = self.stack[-1]

        if token in _ADD_NONLOOP_TOKENS:
            return GrammarTracker(
                stack=self.stack,
                phase=_Phase.PARENTS_NONLOOP,
                pending_n_items=current_n_items,
                previous_ptr=-1,
            )
        if token == "ADD_LOOP":
            return GrammarTracker(
                stack=self.stack,
                phase=_Phase.PARENTS_LOOP,
                pending_n_items=current_n_items,
                previous_ptr=-1,
            )
        if item_phase == _Phase.ITEM_TOP:
            return GrammarTracker(
                stack=self.stack,
                phase=_Phase.DONE,
            )

        return GrammarTracker(
            stack=self.stack[:-1],
            phase=_Phase.EXPECT_CLOSE,
        )

    def _advance_parents_nonloop(self, token: str) -> GrammarTracker:
        ptr_value = _pointer_value(token)
        if ptr_value is not None:
            return GrammarTracker(
                stack=self.stack,
                phase=self.phase,
                pending_n_items=self.pending_n_items,
                previous_ptr=ptr_value,
            )

        completed_stack = self._completed_pending_item_stack()
        completed_n_items = completed_stack[-1]

        if token in _ADD_NONLOOP_TOKENS:
            return GrammarTracker(
                stack=completed_stack,
                phase=_Phase.PARENTS_NONLOOP,
                pending_n_items=completed_n_items,
                previous_ptr=-1,
            )
        if token == "ADD_LOOP":
            return GrammarTracker(
                stack=completed_stack,
                phase=_Phase.PARENTS_LOOP,
                pending_n_items=completed_n_items,
                previous_ptr=-1,
            )
        if len(completed_stack) == 1:
            return GrammarTracker(
                stack=completed_stack,
                phase=_Phase.DONE,
            )
        return GrammarTracker(
            stack=completed_stack[:-1],
            phase=_Phase.EXPECT_CLOSE,
        )

    def _advance_parents_loop(self, token: str) -> GrammarTracker:
        ptr_value = _pointer_value(token)
        if ptr_value is not None:
            return GrammarTracker(
                stack=self.stack,
                phase=self.phase,
                pending_n_items=self.pending_n_items,
                previous_ptr=ptr_value,
            )

        completed_stack = self._completed_pending_item_stack()
        return GrammarTracker(
            stack=(*completed_stack, 0),
            phase=_Phase.ITEM_CHILD,
        )

    def _completed_pending_item_stack(self) -> tuple[int, ...]:
        if self.pending_n_items is None:
            raise ValueError("pending_n_items must be set in a parents phase")
        return (*self.stack[:-1], self.pending_n_items + 1)

    @staticmethod
    def _item_phase_for_stack(stack: tuple[int, ...]) -> GrammarTracker:
        return GrammarTracker(
            stack=stack,
            phase=_Phase.ITEM_TOP if len(stack) == 1 else _Phase.ITEM_CHILD,
        )


def _pointer_value(token: str) -> int | None:
    if not token.startswith("ptr_"):
        return None

    suffix = token[4:]
    if not suffix.isdigit():
        return None

    value = int(suffix)
    if value < 0 or value >= PTR_COUNT:
        return None
    return value
