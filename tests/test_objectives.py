import torch

from src.objectives import final_target, latent_loss, normalized_top2_target, selected_mlm_loss


def test_mlm_only_selected_labels_and_empty_selection() -> None:
    logits = torch.randn(1, 3, 11, requires_grad=True)
    ids = torch.tensor([[2, 3, 4]])
    selected = torch.tensor([[False, True, False]])
    first = selected_mlm_loss(logits, ids, selected)
    changed = selected_mlm_loss(logits, torch.tensor([[9, 3, 8]]), selected)
    torch.testing.assert_close(first.loss, changed.loss)
    assert selected_mlm_loss(logits, ids, torch.zeros_like(selected)).loss is None


def test_top2_teacher_target_is_detached_and_correct() -> None:
    layer_one = torch.tensor([[[1.0, 3.0], [2.0, 4.0]]], requires_grad=True)
    layer_two = torch.tensor([[[3.0, 1.0], [4.0, 2.0]]], requires_grad=True)
    mask = torch.tensor([[True, False]])
    target = normalized_top2_target((layer_one, layer_two), mask)
    assert target.shape == (1, 2)
    assert not target.requires_grad
    assert torch.allclose(target, torch.zeros_like(target), atol=1e-5)


def test_top2_ignores_earlier_layers_and_normalizes_before_averaging() -> None:
    earlier = torch.tensor([[[100.0, -100.0, 7.0]]])
    penultimate = torch.tensor([[[1.0, 2.0, 10.0]]])
    final = torch.tensor([[[3.0, -2.0, 1.0]]])
    selected = torch.tensor([[True]])
    target = normalized_top2_target((earlier, penultimate, final), selected)
    expected = (
        torch.nn.functional.layer_norm(penultimate, (3,))
        + torch.nn.functional.layer_norm(final, (3,))
    ) / 2
    torch.testing.assert_close(target, expected[selected])
    assert not torch.allclose(
        target, torch.nn.functional.layer_norm((penultimate + final) / 2, (3,))[selected]
    )


def test_masked_latent_loss_has_finite_gradients() -> None:
    values = torch.randn(1, 3, 4, requires_grad=True)
    target = final_target(torch.zeros_like(values), torch.tensor([[False, True, False]]))
    result = latent_loss(values, target, torch.tensor([[False, True, False]]), "mse")
    assert result.loss is not None
    result.loss.backward()
    assert torch.isfinite(values.grad).all()
