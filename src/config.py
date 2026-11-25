"""Validated, explicit experiment settings for the Person 1 training path."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, Literal


def _finite_number(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite")
    return number


def _positive_int(value: object, name: str, minimum: int = 1) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")


def _allowed_mapping(value: object, cls: type[object], name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be an object")
    unknown = set(value) - {item.name for item in fields(cls)}
    if unknown:
        raise ValueError(f"unknown {name} keys: {sorted(unknown)}")
    return dict(value)


@dataclass(frozen=True)
class ModelConfig:
    vocab_size: int = 30_522
    max_length: int = 128
    width: int = 256
    layers: int = 4
    heads: int = 4
    ffn_width: int = 1024
    dropout: float = 0.1
    norm_first: bool = True
    pad_token_id: int = 0
    # BERT's full special set: [PAD], [UNK], [CLS], [SEP], [MASK]. [UNK] must be here:
    # predicting it is meaningless, and the resampler derives eligibility from this tuple.
    special_token_ids: tuple[int, ...] = (0, 100, 101, 102, 103)

    def __post_init__(self) -> None:
        _positive_int(self.vocab_size, "vocab_size", 2)
        _positive_int(self.max_length, "max_length")
        _positive_int(self.layers, "layers")
        _positive_int(self.width, "width", 4)
        _positive_int(self.heads, "heads")
        _positive_int(self.ffn_width, "ffn_width")
        if self.width % self.heads:
            raise ValueError("width must be divisible by heads")
        if self.ffn_width < self.width:
            raise ValueError("ffn_width must be at least width")
        if not 0 <= _finite_number(self.dropout, "dropout") < 1:
            raise ValueError("dropout must be in [0, 1)")
        if isinstance(self.pad_token_id, bool) or not isinstance(self.pad_token_id, int):
            raise ValueError("pad_token_id must be an integer inside vocabulary")
        if not 0 <= self.pad_token_id < self.vocab_size:
            raise ValueError("pad_token_id must be inside vocabulary")
        if not isinstance(self.special_token_ids, tuple) or not self.special_token_ids:
            raise ValueError("special_token_ids must be a nonempty tuple")
        if any(
            isinstance(item, bool) or not isinstance(item, int) for item in self.special_token_ids
        ):
            raise ValueError("special_token_ids must contain integers")
        if any(item < 0 or item >= self.vocab_size for item in self.special_token_ids):
            raise ValueError("special token IDs must be inside vocabulary")
        if self.pad_token_id not in self.special_token_ids:
            raise ValueError("special_token_ids must include pad_token_id")


@dataclass(frozen=True)
class OptimizerConfig:
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    betas: tuple[float, float] = (0.9, 0.999)
    warmup_updates: int = 100
    max_updates: int = 10_000
    grad_clip_norm: float = 1.0

    def __post_init__(self) -> None:
        if _finite_number(self.learning_rate, "learning_rate") <= 0:
            raise ValueError("optimizer learning rate and update counts must be valid")
        _positive_int(self.max_updates, "max_updates")
        _positive_int(self.warmup_updates, "warmup_updates", 0)
        if self.warmup_updates > self.max_updates:
            raise ValueError("optimizer learning rate and update counts must be valid")
        if not isinstance(self.betas, tuple) or len(self.betas) != 2:
            raise ValueError("betas must be a tuple of two finite values in [0, 1)")
        if any(not 0 <= _finite_number(beta, "betas") < 1 for beta in self.betas):
            raise ValueError("betas must be a tuple of two finite values in [0, 1)")
        if (
            _finite_number(self.weight_decay, "weight_decay") < 0
            or _finite_number(self.grad_clip_norm, "grad_clip_norm") <= 0
        ):
            raise ValueError("weight decay must be nonnegative and clip must be positive")


@dataclass(frozen=True)
class TeacherConfig:
    """Teacher used for the latent targets.

    mode='ema' keeps a separate exponentially moving average copy of the student, which is the
    recipe in the proposal. mode='self' is the ablation that removes it: targets come from the
    student's own forward pass, detached. That answers "is the EMA needed at all?" on the same
    objective, data and budget.
    """

    ema_decay: float = 0.996
    # Optional linear ramp to this decay across optimizer.max_updates. data2vec schedules
    # 0.999 -> 0.9999; leaving it None keeps the constant decay used by the recorded runs.
    ema_decay_end: float | None = None
    # 'ema' and 'self' are the real objectives. 'constant' (zero targets) and 'noise'
    # (scale-matched Gaussian) are controls: if they degrade the probe as much as the real
    # objective, the degradation is optimisation-induced, not caused by the target.
    mode: Literal["ema", "self", "constant", "noise"] = "ema"
    # Target construction lives in ObjectiveConfig.target so component-wise ablations (D-G) can
    # vary the target without also changing masking, predictor or loss. This flag is now only a
    # cross-check: 'layernorm' is legal solely with the top2_normalized target.
    target_norm: Literal["none", "layernorm"] = "none"

    def __post_init__(self) -> None:
        if self.target_norm not in {"none", "layernorm"}:
            raise ValueError("teacher normalization must be 'none' or 'layernorm'")
        if self.mode not in {"ema", "self", "constant", "noise"}:
            raise ValueError("teacher mode must be 'ema', 'self', 'constant' or 'noise'")
        if not 0 <= _finite_number(self.ema_decay, "ema_decay") < 1:
            raise ValueError("ema_decay must be in [0, 1)")
        if self.ema_decay_end is not None:
            end = _finite_number(self.ema_decay_end, "ema_decay_end")
            if not 0 <= end < 1 or end < self.ema_decay:
                raise ValueError("ema_decay_end must be in [0, 1) and at least ema_decay")


@dataclass(frozen=True)
class MaskConfig:
    """Masking recipe metadata; sampling itself remains Person 2's data responsibility."""

    policy: Literal["independent", "span"] = "independent"
    ratio: float = 0.15
    # Geometric mean span length, and the hard cap on any single span.
    span_length: int = 3
    max_span_length: int = 5
    mask_token_id: int = 103
    # 'mask' replaces every selected position with [MASK]. 'bert_80_10_10' uses BERT's
    # 80% mask / 10% random token / 10% unchanged corruption, which blunts the
    # train/evaluation mismatch; it has not been run at the recorded budgets.
    corruption: Literal["mask", "bert_80_10_10"] = "mask"
    # Rebuild masked_input_ids and prediction_mask per training batch instead of trusting
    # the single realisation baked into the artifact. Off by default so old artifacts keep
    # their recorded behaviour; new runs should enable it.
    resample_in_loop: bool = False

    def __post_init__(self) -> None:
        if self.policy not in {"independent", "span"}:
            raise ValueError("mask policy must be 'independent' or 'span'")
        if not 0 < _finite_number(self.ratio, "mask ratio") < 1:
            raise ValueError("mask ratio must be in (0, 1)")
        _positive_int(self.span_length, "span_length")
        _positive_int(self.max_span_length, "max_span_length")
        if (
            isinstance(self.mask_token_id, bool)
            or not isinstance(self.mask_token_id, int)
            or self.mask_token_id < 0
        ):
            raise ValueError("mask_token_id must be a nonnegative integer")
        if self.corruption not in {"mask", "bert_80_10_10"}:
            raise ValueError("corruption must be 'mask' or 'bert_80_10_10'")
        if not isinstance(self.resample_in_loop, bool):
            raise ValueError("resample_in_loop must be boolean")


@dataclass(frozen=True)
class RegulariserConfig:
    """Optional anti-collapse term added to the latent objective.

    This is the axis the mid-project review flagged as missing: the project measured collapse
    (effective rank) but never tried to prevent it. Both options keep the objective, encoder,
    data and budget fixed and add only a statistics term.

    kind='vicreg' is VICReg's variance and covariance terms: hold per-dimension standard
    deviation above a floor, and push off-diagonal covariance to zero.
    kind='barlow' is Barlow Twins' redundancy reduction: normalise each dimension, then push
    the off-diagonal of the correlation matrix to zero, with an optional on-diagonal term
    pulling self-correlation toward one.

    Neither is the full published method. VICReg and Barlow Twins are two-view objectives with
    an invariance term between views; here the invariance term is supplied by the existing
    masked-latent loss and only the statistics terms are added. That is the intended
    comparison (collapse guard as an auxiliary) and is reported as such, not as a
    reproduction of either paper.
    """

    kind: Literal["none", "vicreg", "barlow"] = "none"
    weight: float = 0.0
    variance_floor: float = 1.0
    variance_weight: float = 1.0
    covariance_weight: float = 0.04
    diagonal_weight: float = 0.005
    eps: float = 1e-4

    def __post_init__(self) -> None:
        if self.kind not in {"none", "vicreg", "barlow"}:
            raise ValueError("regulariser kind must be 'none', 'vicreg' or 'barlow'")
        weight = _finite_number(self.weight, "regulariser weight")
        if weight < 0:
            raise ValueError("regulariser weight must be nonnegative")
        if self.kind != "none" and weight <= 0:
            raise ValueError(f"regulariser {self.kind!r} needs a positive weight")
        if self.kind == "none" and weight != 0:
            raise ValueError("regulariser weight must be 0 when kind is 'none'")
        if _finite_number(self.variance_floor, "variance_floor") <= 0:
            raise ValueError("variance_floor must be positive")
        for name, value in (
            ("variance_weight", self.variance_weight),
            ("covariance_weight", self.covariance_weight),
            ("diagonal_weight", self.diagonal_weight),
            ("eps", self.eps),
        ):
            if _finite_number(value, name) < 0 or (name == "eps" and value == 0):
                raise ValueError(f"{name} must be positive")


@dataclass(frozen=True)
class ObjectiveConfig:
    """Objective defined by its individual components.

    Recipes A/B are presets of these fields, and the D-G ablations change exactly one field
    relative to A while masking stays in MaskConfig. `label` is the reported run name.
    """

    kind: Literal["mlm", "latent"] = "mlm"
    target: Literal["final", "top2_normalized"] = "final"
    predictor: Literal["attention", "linear"] = "attention"
    loss: Literal["mse", "smooth_l1"] = "mse"
    label: str = "C"
    smooth_l1_beta: float = 1.0
    predictor_dropout: float = 0.1
    # Optional bottleneck for the MLM decoder (width -> bottleneck -> vocab). The full head
    # is ~7.9M parameters against a 1.7M attention predictor and a 66k linear one, so a
    # reduced head is the control that separates objective effects from head capacity.
    mlm_head_width: int | None = None
    # Deprecated: a legacy recipe name that expands into the component fields below. Kept so
    # existing configs and callers keep working; new configs declare the components directly.
    variant: str = ""

    def __post_init__(self) -> None:
        if self.variant:
            if self.variant not in LEGACY_VARIANTS:
                raise ValueError(
                    f"legacy objective variant {self.variant!r} is not one of "
                    f"{sorted(LEGACY_VARIANTS)}; declare kind/target/predictor/loss instead"
                )
            for key, value in LEGACY_VARIANTS[self.variant].items():
                object.__setattr__(self, key, value)
            object.__setattr__(self, "variant", "")
        if self.kind not in {"mlm", "latent"}:
            raise ValueError("objective kind must be 'mlm' or 'latent'")
        if self.target not in {"final", "top2_normalized"}:
            raise ValueError("objective target must be 'final' or 'top2_normalized'")
        if self.predictor not in {"attention", "linear"}:
            raise ValueError("objective predictor must be 'attention' or 'linear'")
        if self.loss not in {"mse", "smooth_l1"}:
            raise ValueError("objective loss must be 'mse' or 'smooth_l1'")
        if not isinstance(self.label, str) or not self.label.strip():
            raise ValueError("objective label must be a nonempty string")
        if (
            _finite_number(self.smooth_l1_beta, "smooth_l1_beta") <= 0
            or not 0 <= _finite_number(self.predictor_dropout, "predictor_dropout") < 1
        ):
            raise ValueError("invalid objective settings")
        if self.mlm_head_width is not None:
            _positive_int(self.mlm_head_width, "mlm_head_width")
            if self.kind != "mlm":
                raise ValueError("mlm_head_width only applies to kind='mlm'")



LEGACY_VARIANTS: dict[str, dict[str, Any]] = {
    "mlm": {"kind": "mlm", "label": "C"},
    "latent_a": {
        "kind": "latent",
        "target": "final",
        "predictor": "attention",
        "loss": "mse",
        "label": "A",
    },
    "latent_b": {
        "kind": "latent",
        "target": "top2_normalized",
        "predictor": "linear",
        "loss": "smooth_l1",
        "label": "B",
    },
}


def _objective_from_dict(value: dict[str, Any]) -> ObjectiveConfig:
    """Accept either explicit components or the deprecated single `variant` key."""
    payload = dict(value)
    legacy = payload.pop("variant", None)
    if legacy is not None:
        if legacy not in LEGACY_VARIANTS:
            raise ValueError(
                f"legacy variant {legacy!r} is not one of {sorted(LEGACY_VARIANTS)}; "
                "declare kind/target/predictor/loss instead"
            )
        merged = {**LEGACY_VARIANTS[legacy], **payload}
        return ObjectiveConfig(**merged)
    return ObjectiveConfig(**payload)


@dataclass(frozen=True)
class RunConfig:
    seed: int = 17
    token_budget: int = 5_000_000
    device: str = "cpu"
    amp: bool = False
    output_dir: str = "results/person1"
    checkpoint_every_updates: int = 100

    def __post_init__(self) -> None:
        _positive_int(self.seed, "seed", 0)
        _positive_int(self.token_budget, "token_budget")
        _positive_int(self.checkpoint_every_updates, "checkpoint_every_updates")
        if not isinstance(self.device, str) or not self.device:
            raise ValueError("device must be a nonempty string")
        if not isinstance(self.output_dir, str) or not self.output_dir:
            raise ValueError("output_dir must be a nonempty string")
        if not isinstance(self.amp, bool):
            raise ValueError("amp must be boolean")
        if self.amp:
            raise ValueError("AMP is intentionally unsupported; this runner is explicit fp32")


@dataclass(frozen=True)
class ExperimentConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    teacher: TeacherConfig = field(default_factory=TeacherConfig)
    mask: MaskConfig = field(default_factory=MaskConfig)
    objective: ObjectiveConfig = field(default_factory=ObjectiveConfig)
    regulariser: RegulariserConfig = field(default_factory=RegulariserConfig)
    run: RunConfig = field(default_factory=RunConfig)
    schema_version: int = 1

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise ValueError("unsupported config schema")
        # Masking is a free axis so component-wise ablations can hold it fixed or vary it
        # independently of the objective. Structural constraints only:
        if self.objective.kind == "latent" and self.objective.target == "top2_normalized":
            if self.model.layers < 2:
                raise ValueError("top2_normalized target requires at least two encoder layers")
        if self.objective.kind == "mlm" and (
            self.objective.target != "final"
            or self.objective.predictor != "attention"
            or self.objective.loss != "mse"
        ):
            raise ValueError(
                "mlm has no teacher target, predictor or latent loss; "
                "use kind='latent' to select target/predictor/loss"
            )
        if (
            self.objective.kind == "latent"
            and self.teacher.target_norm == "layernorm"
            and self.objective.target != "top2_normalized"
        ):
            raise ValueError(
                "teacher.target_norm='layernorm' only applies to the top2_normalized target; "
                "set target_norm='none' for the final-layer target"
            )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> ExperimentConfig:
        allowed = {
            "model",
            "optimizer",
            "teacher",
            "mask",
            "objective",
            "regulariser",
            "run",
            "schema_version",
            "_note",  # free-form documentation carried in the config file; ignored
        }
        extra = set(value) - allowed
        if extra:
            raise ValueError(f"unknown config keys: {sorted(extra)}")
        model = _allowed_mapping(value.get("model", {}), ModelConfig, "model")
        optimizer = _allowed_mapping(value.get("optimizer", {}), OptimizerConfig, "optimizer")
        teacher = _allowed_mapping(value.get("teacher", {}), TeacherConfig, "teacher")
        mask = _allowed_mapping(value.get("mask", {}), MaskConfig, "mask")
        regulariser = _allowed_mapping(
            value.get("regulariser", {}), RegulariserConfig, "regulariser"
        )
        objective = dict(value.get("objective", {}))
        if not isinstance(objective, dict):
            raise ValueError("objective must be an object")
        legacy = objective.pop("variant", None)
        if legacy:
            if legacy not in LEGACY_VARIANTS:
                raise ValueError(
                    f"legacy objective variant {legacy!r} is not one of {sorted(LEGACY_VARIANTS)}; "
                    "declare kind/target/predictor/loss instead"
                )
            objective = {**LEGACY_VARIANTS[legacy], **objective}
        run = _allowed_mapping(value.get("run", {}), RunConfig, "run")
        if "special_token_ids" in model:
            model["special_token_ids"] = tuple(model["special_token_ids"])
        if "betas" in optimizer:
            optimizer["betas"] = tuple(optimizer["betas"])
        return cls(
            model=ModelConfig(**model),
            optimizer=OptimizerConfig(**optimizer),
            teacher=TeacherConfig(**teacher),
            mask=MaskConfig(**mask),
            objective=ObjectiveConfig(**objective),
            regulariser=RegulariserConfig(**regulariser),
            run=RunConfig(**run),
            schema_version=value.get("schema_version", 1),
        )


def load_config(path: str | Path) -> ExperimentConfig:
    with Path(path).open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError("configuration root must be an object")
    return ExperimentConfig.from_dict(value)
