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


def test_vicreg_variance_term_reacts_to_collapse() -> None:
    from src.objectives import vicreg_variance_term

    generator = torch.Generator().manual_seed(0)
    # Every dimension is constant across samples, so its per-column std is exactly zero.
    collapsed = torch.randn(8, generator=generator).unsqueeze(0).repeat(64, 1)
    assert collapsed.std(dim=0, unbiased=False).abs().max().item() == 0.0
    # eps sits under the sqrt, so sqrt(0) is lifted to sqrt(eps) and the default eps lifts the
    # floor term to 1 - 0.01. A tight floor check needs a tiny eps.
    assert abs(float(vicreg_variance_term(collapsed)) - 1.0) < 1.1e-2
    floor_term = vicreg_variance_term(collapsed, floor=1.0, eps=1e-8)
    assert abs(float(floor_term) - 1.0) < 1e-3

    # Exactly unit per-column std: nothing is below the floor, so the hinge is zero.
    noise = torch.randn(4096, 8, generator=generator)
    noise = noise / noise.std(dim=0, unbiased=False)
    spread = vicreg_variance_term(noise, floor=1.0)
    assert float(spread) < 1e-2


def test_vicreg_covariance_term_is_zero_for_independent_columns() -> None:
    from src.objectives import vicreg_covariance_term

    generator = torch.Generator().manual_seed(0)
    independent = torch.randn(2000, 8, generator=generator)
    baseline = float(vicreg_covariance_term(independent))
    assert baseline < 0.02
    assert torch.isfinite(torch.tensor(baseline)) and baseline >= 0

    duplicated = independent.clone()
    duplicated[:, -1] = duplicated[:, 0]
    redundant = float(vicreg_covariance_term(duplicated))
    assert redundant > 10 * baseline
    assert redundant > 0.1


def test_barlow_off_diagonal_penalises_duplicated_dimensions() -> None:
    from src.objectives import barlow_redundancy_term

    generator = torch.Generator().manual_seed(0)
    independent = torch.randn(2000, 8, generator=generator)
    baseline = barlow_redundancy_term(independent)
    # Eight perfect copies of one signal: every off-diagonal correlation is 1, so the term is
    # 8 * 7 / 8 = 7 before the on-diagonal correction, which is already satisfied.
    duplicated = barlow_redundancy_term(
        torch.randn(2000, generator=generator).unsqueeze(1).repeat(1, 8)
    )
    for value in (baseline, duplicated):
        assert value.dim() == 0 and torch.isfinite(value) and value >= 0
    assert float(duplicated) > 100 * float(baseline)
    assert abs(float(duplicated) - 7.0) < 1e-2


def test_collapse_regulariser_dispatches_and_validates() -> None:
    import pytest

    from src.objectives import collapse_regulariser

    embeddings = torch.randn(64, 8, generator=torch.Generator().manual_seed(0))
    assert collapse_regulariser(embeddings, "none") is None
    for kind in ("vicreg", "barlow"):
        term = collapse_regulariser(embeddings, kind)
        assert term is not None and term.dim() == 0
        assert torch.isfinite(term) and term >= 0
    with pytest.raises(ValueError, match="unknown regulariser"):
        collapse_regulariser(embeddings, "bogus")
    with pytest.raises(ValueError, match="matrix"):
        collapse_regulariser(torch.randn(64), "vicreg")
    for kind in ("vicreg", "barlow"):
        with pytest.raises(ValueError, match="two samples"):
            collapse_regulariser(torch.randn(1, 8), kind)
