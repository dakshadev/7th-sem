def test_default_config_and_invalid_dimensions() -> None:
    import pytest

    from src.config import ExperimentConfig, MaskConfig, ModelConfig, ObjectiveConfig, TeacherConfig

    config = ExperimentConfig()
    assert config.model.layers == 4
    with pytest.raises(ValueError, match="divisible"):
        ModelConfig(width=255, heads=4)
    # Masking is a free axis now: the latent recipe does not force a mask policy, because the
    # D-G ablations hold masking fixed while changing one objective component.
    spans = ExperimentConfig(
        objective=ObjectiveConfig(variant="latent_a"),
        mask=MaskConfig(policy="span"),
        teacher=TeacherConfig(target_norm="none"),
    )
    assert spans.mask.policy == "span"
    independent = ExperimentConfig(
        objective=ObjectiveConfig(variant="latent_a"),
        mask=MaskConfig(policy="independent"),
        teacher=TeacherConfig(target_norm="none"),
    )
    assert independent.mask.policy == "independent"
    with pytest.raises(ValueError, match="two encoder layers"):
        ExperimentConfig(
            model=ModelConfig(vocab_size=16, layers=1, special_token_ids=(0, 1, 2, 3)),
            objective=ObjectiveConfig(kind="latent", target="top2_normalized"),
        )


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


def test_masking_is_a_free_axis_for_every_objective() -> None:
    """The D-G ablations vary one objective component at a time while masking is held in
    MaskConfig, so no objective may pin the mask policy."""
    from src.config import ExperimentConfig, MaskConfig, ObjectiveConfig

    for policy in ("span", "independent"):
        for objective in (
            ObjectiveConfig(variant="mlm"),
            ObjectiveConfig(variant="latent_a", target="final"),
            ObjectiveConfig(variant="latent_b"),
        ):
            config = ExperimentConfig(objective=objective, mask=MaskConfig(policy=policy))
            assert config.mask.policy == policy


def test_legacy_variants_expand_into_components() -> None:
    from src.config import ObjectiveConfig

    a = ObjectiveConfig(variant="latent_a")
    assert (a.kind, a.target, a.predictor, a.loss, a.label) == (
        "latent", "final", "attention", "mse", "A",
    )
    b = ObjectiveConfig(variant="latent_b")
    assert (b.kind, b.target, b.predictor, b.loss, b.label) == (
        "latent", "top2_normalized", "linear", "smooth_l1", "B",
    )
    c = ObjectiveConfig(variant="mlm")
    assert (c.kind, c.label) == ("mlm", "C")


def test_component_ablations_differ_from_a_in_exactly_one_field() -> None:
    from src.config import ExperimentConfig

    base = ExperimentConfig.from_dict(
        {"objective": {"kind": "latent", "target": "final", "predictor": "attention",
                       "loss": "mse", "label": "A"}, "mask": {"policy": "span"},
         "teacher": {"target_norm": "none"}}
    )
    specs = {
        "D": {"target": "top2_normalized"},
        "E": {},
        "F": {"predictor": "linear"},
        "G": {"loss": "smooth_l1"},
    }
    base_components = (base.objective.target, base.objective.predictor, base.objective.loss)
    for label, override in specs.items():
        payload = {"kind": "latent", "target": "final", "predictor": "attention",
                   "loss": "mse", "label": label, **override}
        mask = {"policy": "span"} if label != "E" else {"policy": "independent"}
        teacher = {"target_norm": "layernorm" if label == "D" else "none"}
        config = ExperimentConfig.from_dict(
            {"objective": payload, "mask": mask, "teacher": teacher}
        )
        components = (config.objective.target, config.objective.predictor, config.objective.loss)
        if label == "E":
            assert config.mask.policy == "independent"
            assert components == base_components
        else:
            differing = [
                name for name, a, b in zip(
                    ("target", "predictor", "loss"), base_components, components, strict=True
                ) if a != b
            ]
            assert differing == [override.popitem()[0]] or len(differing) == 1


def test_config_ignores_documentation_note() -> None:
    import pytest

    from src.config import ExperimentConfig

    config = ExperimentConfig.from_dict(
        {"_note": "human-readable description", "mask": {"policy": "span"}}
    )
    assert config.mask.policy == "span"
    with pytest.raises(ValueError, match="unknown config keys"):
        ExperimentConfig.from_dict({"unexpected": 1})


def test_regulariser_config_validation() -> None:
    import pytest

    from src.config import ExperimentConfig, RegulariserConfig

    assert RegulariserConfig().kind == "none" and RegulariserConfig().weight == 0.0
    with pytest.raises(ValueError, match="positive weight"):
        RegulariserConfig(kind="vicreg", weight=0.0)
    with pytest.raises(ValueError, match="must be 0"):
        RegulariserConfig(kind="none", weight=0.1)
    with pytest.raises(ValueError, match="nonnegative"):
        RegulariserConfig(kind="vicreg", weight=-0.1)
    with pytest.raises(ValueError, match="variance_floor"):
        RegulariserConfig(kind="vicreg", weight=0.1, variance_floor=0.0)
    with pytest.raises(ValueError, match="covariance_weight"):
        RegulariserConfig(kind="vicreg", weight=0.1, covariance_weight=-0.1)
    with pytest.raises(ValueError, match="eps"):
        RegulariserConfig(kind="vicreg", weight=0.1, eps=0.0)
    with pytest.raises(ValueError, match="kind"):
        RegulariserConfig(kind="other", weight=0.1)  # type: ignore[arg-type]

    block = {
        "kind": "vicreg",
        "weight": 0.3,
        "variance_floor": 0.5,
        "variance_weight": 2.0,
        "covariance_weight": 0.08,
        "diagonal_weight": 0.01,
        "eps": 1e-5,
    }
    settings = ExperimentConfig.from_dict({"regulariser": block}).regulariser
    assert settings.kind == "vicreg"
    assert settings.weight == 0.3
    assert settings.variance_floor == 0.5
    assert settings.variance_weight == 2.0
    assert settings.covariance_weight == 0.08
    assert settings.diagonal_weight == 0.01
    assert settings.eps == 1e-5
    # The block must survive a full round-trip, including through to_dict().
    config = ExperimentConfig.from_dict({"regulariser": block})
    assert ExperimentConfig.from_dict(config.to_dict()).regulariser == settings


def test_mask_config_validates_span_lengths() -> None:
    import pytest

    from src.config import MaskConfig

    assert MaskConfig().max_span_length == 5
    assert MaskConfig(span_length=4, max_span_length=6).max_span_length == 6
    with pytest.raises(ValueError, match="span_length"):
        MaskConfig(span_length=0)
    with pytest.raises(ValueError, match="max_span_length"):
        MaskConfig(max_span_length=0)


def test_mask_and_ema_extensions_validate() -> None:
    import pytest

    from src.config import MaskConfig, TeacherConfig

    assert MaskConfig().corruption == "mask"
    assert MaskConfig(corruption="bert_80_10_10").corruption == "bert_80_10_10"
    assert MaskConfig(resample_in_loop=True).resample_in_loop is True
    with pytest.raises(ValueError, match="corruption"):
        MaskConfig(corruption="other")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="mask_token_id"):
        MaskConfig(mask_token_id=-1)
    with pytest.raises(ValueError, match="resample_in_loop"):
        MaskConfig(resample_in_loop="yes")  # type: ignore[arg-type]

    assert TeacherConfig(ema_decay=0.9, ema_decay_end=0.99).ema_decay_end == 0.99
    with pytest.raises(ValueError, match="ema_decay_end"):
        TeacherConfig(ema_decay=0.9, ema_decay_end=0.8)
    with pytest.raises(ValueError, match="ema_decay_end"):
        TeacherConfig(ema_decay=0.9, ema_decay_end=1.0)


def _flatten(value: object, prefix: str = "") -> dict[str, object]:
    items: dict[str, object] = {}
    if isinstance(value, dict):
        for key, item in value.items():
            items.update(_flatten(item, f"{prefix}.{key}" if prefix else str(key)))
    else:
        items[prefix] = value
    return items


def test_v2_configs_are_the_recorded_configs_plus_the_training_path() -> None:
    """A2/B2/C2 must differ from A/B/C only by the path changes, or the comparison is confounded."""
    from src.config import load_config

    pairs = {
        "A2": ("configs/latent_a.json", "configs/latent_a2_v2.json"),
        "B2": ("configs/latent_b.json", "configs/latent_b2_v2.json"),
        "C2": ("configs/mlm_c.json", "configs/mlm_c2_v2.json"),
    }
    for label, (recorded_path, v2_path) in pairs.items():
        recorded = _flatten(load_config(recorded_path).to_dict())
        v2 = _flatten(load_config(v2_path).to_dict())
        assert set(recorded) == set(v2)
        differing = {key for key in recorded if recorded[key] != v2[key]}
        # max_span_length is written explicitly in the v2 files but equals the dataclass
        # default, so the only behavioural deltas are the path switches themselves.
        expected = {"objective.label", "mask.resample_in_loop"}
        if label != "C2":
            expected.add("teacher.ema_decay_end")
        assert differing == expected, (label, sorted(differing))
        assert v2["objective.label"] == label


def test_teacher_mode_validation() -> None:
    import pytest

    from src.config import ExperimentConfig, TeacherConfig

    assert TeacherConfig().mode == "ema"
    assert TeacherConfig(mode="self").mode == "self"
    assert ExperimentConfig.from_dict({"teacher": {"mode": "self"}}).teacher.mode == "self"
    with pytest.raises(ValueError, match="teacher mode"):
        TeacherConfig(mode="other")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="teacher mode"):
        ExperimentConfig.from_dict({"teacher": {"mode": "student"}})
