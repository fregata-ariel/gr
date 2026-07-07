from __future__ import annotations

import pytest

from poc0.constants import MAX_SEQ_LEN, TOKEN_TO_ID, VOCAB
from poc0.tokenizer import record_to_example


def test_vocab_ids_are_exact():
    expected_vocab = [
        "PAD",
        "BOS",
        "ADD_ENTRY",
        "ADD_LINEAR",
        "ADD_MERGE",
        "ADD_LOOP",
        "OPEN",
        "CLOSE",
        "STOP",
        "ptr_0",
        "ptr_1",
        "ptr_2",
        "ptr_3",
        "ptr_4",
        "ptr_5",
        "ptr_6",
        "ptr_7",
        "ptr_8",
        "ptr_9",
        "ptr_10",
        "ptr_11",
        "ptr_12",
        "ptr_13",
        "ptr_14",
        "ptr_15",
        "ptr_16",
        "ptr_17",
        "ptr_18",
        "ptr_19",
        "ptr_20",
        "ptr_21",
        "ptr_22",
        "ptr_23",
        "ptr_24",
        "ptr_25",
        "ptr_26",
        "ptr_27",
        "ptr_28",
        "ptr_29",
        "ptr_30",
        "ptr_31",
    ]

    assert VOCAB == tuple(expected_vocab)
    assert TOKEN_TO_ID == {
        token: token_id for token_id, token in enumerate(expected_vocab)
    }


def test_record_to_example_adds_bos_and_masks_to_stop():
    example = record_to_example(
        [
            "ADD_ENTRY",
            "ADD_LOOP",
            "ptr_0",
            "OPEN",
            "ADD_LINEAR",
            "ptr_1",
            "STOP",
            "CLOSE",
            "STOP",
        ]
    )

    expected_input_prefix = [
        TOKEN_TO_ID["BOS"],
        TOKEN_TO_ID["ADD_ENTRY"],
        TOKEN_TO_ID["ADD_LOOP"],
        TOKEN_TO_ID["ptr_0"],
        TOKEN_TO_ID["OPEN"],
        TOKEN_TO_ID["ADD_LINEAR"],
        TOKEN_TO_ID["ptr_1"],
        TOKEN_TO_ID["STOP"],
        TOKEN_TO_ID["CLOSE"],
    ]
    expected_target_prefix = [
        TOKEN_TO_ID["ADD_ENTRY"],
        TOKEN_TO_ID["ADD_LOOP"],
        TOKEN_TO_ID["ptr_0"],
        TOKEN_TO_ID["OPEN"],
        TOKEN_TO_ID["ADD_LINEAR"],
        TOKEN_TO_ID["ptr_1"],
        TOKEN_TO_ID["STOP"],
        TOKEN_TO_ID["CLOSE"],
        TOKEN_TO_ID["STOP"],
    ]

    assert example.input_ids.tolist()[: len(expected_input_prefix)] == (
        expected_input_prefix
    )
    assert example.target_ids.tolist()[: len(expected_target_prefix)] == (
        expected_target_prefix
    )
    assert example.loss_mask.tolist()[: len(expected_target_prefix)] == (
        [True] * len(expected_target_prefix)
    )

    assert example.input_ids.shape == (MAX_SEQ_LEN,)
    assert example.target_ids.shape == (MAX_SEQ_LEN,)
    assert example.loss_mask.shape == (MAX_SEQ_LEN,)
    assert set(example.input_ids.tolist()[len(expected_input_prefix):]) == {
        TOKEN_TO_ID["PAD"]
    }
    assert set(example.target_ids.tolist()[len(expected_target_prefix):]) == {
        TOKEN_TO_ID["PAD"]
    }
    assert set(example.loss_mask.tolist()[len(expected_target_prefix):]) == {
        False
    }


def test_tokenizer_rejects_unknown_token_and_pointer_overflow():
    with pytest.raises(ValueError, match="UNKNOWN_TOKEN"):
        record_to_example(["UNKNOWN_TOKEN", "STOP"])

    with pytest.raises(ValueError, match="ptr_32"):
        record_to_example(["ADD_ENTRY", "ptr_32", "STOP"])
