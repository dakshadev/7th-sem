import pytest
import torch

from src.config import ModelConfig
from src.model import SmallEncoder


def tiny_model() -> SmallEncoder:
    config = ModelConfig(
        vocab_size=31,
        max_length=8,
        width=16,
        layers=2,
        heads=4,
        ffn_width=32,
        dropout=0.0,
        special_token_ids=(0, 1, 2, 3),
    )
    return SmallEncoder(config)


def test_encoder_shapes_hidden_states_and_padding_invariance() -> None:
    model = tiny_model().eval()
    valid = torch.tensor([[1, 5, 6, 2, 0, 0]])
    changed_padding = torch.tensor([[1, 5, 6, 2, 7, 8]])
    mask = valid.ne(0)
    first = model(valid, mask, return_hidden_states=True)
    second = model(changed_padding, mask, return_hidden_states=True)
    assert first.last_hidden_state.shape == (1, 6, 16)
    assert len(first.hidden_states) == 2
    torch.testing.assert_close(first.last_hidden_state[:, :4], second.last_hidden_state[:, :4])


def test_encoder_rejects_bad_or_empty_inputs() -> None:
    model = tiny_model()
    with pytest.raises(ValueError, match="int64"):
        model(torch.ones(1, 3), torch.ones(1, 3, dtype=torch.bool))
    with pytest.raises(ValueError, match="valid token"):
        model(torch.zeros(1, 3, dtype=torch.long), torch.zeros(1, 3, dtype=torch.bool))
    with pytest.raises(ValueError, match="batch"):
        model(torch.zeros(0, 3, dtype=torch.long), torch.zeros(0, 3, dtype=torch.bool))
