def test_default_config_and_invalid_dimensions() -> None:
    import pytest

    from src.config import ExperimentConfig, MaskConfig, ModelConfig, ObjectiveConfig, TeacherConfig

    config = ExperimentConfig()
    assert config.model.layers == 4
    with pytest.raises(ValueError, match="divisible"):
        ModelConfig(width=255, heads=4)
    with pytest.raises(ValueError, match="span"):
        ExperimentConfig(objective=ObjectiveConfig(variant="latent_a"))
    valid = ExperimentConfig(
        objective=ObjectiveConfig(variant="latent_a"),
        mask=MaskConfig(policy="span"),
        teacher=TeacherConfig(target_norm="none"),
    )
    assert valid.mask.policy == "span"


def test_config_rejects_unknown_variants_nonfinite_values_and_bad_special_ids() -> None:
    import math

    import pytest

    from src.config import ModelConfig, ObjectiveConfig, OptimizerConfig, TeacherConfig

    with pytest.raises(ValueError, match="variant"):
        ObjectiveConfig(variant="other")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="normalization"):
        TeacherConfig(target_norm="other")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="finite"):
        OptimizerConfig(learning_rate=math.nan)
    with pytest.raises(ValueError, match="betas"):
        OptimizerConfig(betas=(0.9, 1.0))
    with pytest.raises(ValueError, match="heads"):
        ModelConfig(heads=0)
    with pytest.raises(ValueError, match="special"):
        ModelConfig(vocab_size=8, special_token_ids=(0, 8))


def test_config_json_roundtrip_is_canonical_and_rejects_nested_unknown_keys() -> None:
    import pytest

    from src.config import ExperimentConfig

    value = ExperimentConfig.from_dict(
        {
            "model": {"vocab_size": 16, "special_token_ids": [0, 1, 2, 3]},
            "optimizer": {"betas": [0.8, 0.95]},
        }
    )
    assert value == ExperimentConfig.from_dict(value.to_dict())
    assert isinstance(value.optimizer.betas, tuple)
    assert isinstance(value.model.special_token_ids, tuple)
    with pytest.raises(ValueError, match="unknown model"):
        ExperimentConfig.from_dict({"model": {"unknown": 1}})


def test_recipe_coupling_requires_two_layers_for_latent_b() -> None:
    import pytest

    from src.config import ExperimentConfig, ModelConfig, ObjectiveConfig

    with pytest.raises(ValueError, match="two encoder layers"):
        ExperimentConfig(
            model=ModelConfig(vocab_size=16, layers=1, special_token_ids=(0, 1, 2, 3)),
            objective=ObjectiveConfig(variant="latent_b"),
        )


def test_mlm_requires_independent_metadata() -> None:
    import pytest

    from src.config import ExperimentConfig, MaskConfig

    with pytest.raises(ValueError, match="independent"):
        ExperimentConfig(mask=MaskConfig(policy="span"))
