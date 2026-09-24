"""Pure planning helpers for the Modal regression fan-out.

``plan_fanout`` turns the local checkout plus the container layout into one
:class:`FanoutUnit` per target. Each unit's ``argv`` is a complete
``main_regression_target.main(argv)`` invocation written with *container*
paths, so a unit can be handed to a Modal worker unchanged.

This module deliberately has no ``modal`` import: the plan is built and tested
locally, and only the argv strings know about the container layout.

The plan never pre-filters by completion. Completion markers live on the Modal
results volume, so ``--skip-completed`` is forwarded to the runner inside the
container instead. ``completed_targets`` is provided for callers that do have
the results directory on local media.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import NamedTuple

# Allow ``import regression_fanout_plan`` from the repo root or a test module.
_SRC_DIR = Path(__file__).resolve().parent
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from main_regression_target import (  # noqa: E402
    COMPLETED_DIRNAME,
    ConfigError,
    discover_feature_subsets,
    load_config,
)

DEFAULT_OUTPUT_PROFILE = "slim"


class RuntimeLayout(NamedTuple):
    """Paths as seen *inside* the Modal container."""

    genomic: str
    metadata: str
    index_column: str
    config: str
    feature_base: str
    results_dir: str


class FanoutOptions(NamedTuple):
    """What to run, independently of where it runs."""

    targets: tuple[str, ...] = ()
    experiments: tuple[str, ...] = ()
    feature_models: tuple[str, ...] = ()
    all_features_only: bool = False
    skip_completed: bool = False
    output_profile: str = DEFAULT_OUTPUT_PROFILE


class FanoutUnit(NamedTuple):
    """One target to run remotely: the label plus its full CLI invocation."""

    target: str
    argv: tuple[str, ...]


def build_target_argv(
    target: str, layout: RuntimeLayout, options: FanoutOptions
) -> tuple[str, ...]:
    """Build a full ``main_regression_target`` invocation for one target.

    Every path in the argv is a container path from ``layout``; no local path
    leaks in. ``--output-profile`` defaults to ``slim`` via ``FanoutOptions``.
    """
    argv: list[str] = [
        "--genomic",
        layout.genomic,
        "--metadata",
        layout.metadata,
        "--index-column",
        layout.index_column,
        "--config",
        layout.config,
        "--target",
        target,
        "--feature-base-dir",
        layout.feature_base,
        "--results-dir",
        layout.results_dir,
        "--output-profile",
        options.output_profile,
    ]
    if options.skip_completed:
        argv.append("--skip-completed")
    if options.all_features_only:
        argv.append("--all-features-only")
    for experiment in options.experiments:
        argv.extend(["--experiment", experiment])
    for feature_model in options.feature_models:
        argv.extend(["--feature-model", feature_model])
    return tuple(argv)


def feature_subset_labels(
    feature_base_dir: Path | str,
    experiments: tuple[str, ...] = (),
    feature_models: tuple[str, ...] = (),
) -> list[str]:
    """Return ``"EXPERIMENT/MODEL"`` labels for the feature subsets to run.

    Wraps :func:`main_regression_target.discover_feature_subsets` so the guard
    in :func:`plan_fanout` and the Modal entrypoint summary agree on what will
    actually run. Raises ``ConfigError`` when ``feature_base_dir`` is missing.
    """
    subsets = discover_feature_subsets(
        Path(feature_base_dir), list(experiments), list(feature_models)
    )
    return [f"{subset['experiment']}/{subset['model']}" for subset in subsets]


def plan_fanout(
    config_path: Path | str,
    feature_base_dir: Path | str,
    results_dir: Path | str,
    layout: RuntimeLayout,
    options: FanoutOptions = FanoutOptions(),
) -> list[FanoutUnit]:
    """Build one :class:`FanoutUnit` per target to run.

    ``config_path`` and ``feature_base_dir`` are local inputs and must exist:
    the config defines the target set, and the feature directory is what the
    Modal image mounts. ``results_dir`` is the local results root; it is not
    read here (completion markers live on the remote volume) but is part of
    the layout so callers can pair this plan with ``completed_targets``.

    Explicit ``options.targets`` must be a subset of the config's
    ``target_columns``; anything else raises
    ``main_regression_target.ConfigError``. Without explicit targets, every
    config target is planned, in config order.

    Unless ``options.all_features_only`` is set, at least one feature subset
    must be discovered under ``feature_base_dir`` (honoring the experiment and
    feature-model filters); a zero-subset plan raises
    ``main_regression_target.ConfigError`` so the run cannot silently degrade
    to the all-features baseline.
    """
    config = load_config(Path(config_path))

    feature_base = Path(feature_base_dir)
    if not feature_base.is_dir():
        raise ConfigError(f"Feature base dir not found: {feature_base}")

    if not options.all_features_only and not feature_subset_labels(
        feature_base, options.experiments, options.feature_models
    ):
        raise ConfigError(
            f"No feature subsets found under {feature_base}: expected at least "
            "one '<experiment>/<model>/<model>_top_features*.csv'. Refusing to "
            "run only the all-features baseline; pass all_features_only=True "
            "to allow that explicitly."
        )

    known = set(config["target_columns"])
    requested = options.targets or tuple(config["target_columns"])
    unknown = [target for target in requested if target not in known]
    if unknown:
        raise ConfigError(
            f"Target(s) {unknown} are not listed in the config's "
            "'target_columns'. Add them to the config so they are excluded "
            "from the feature matrix."
        )

    seen: set[str] = set()
    units: list[FanoutUnit] = []
    for target in requested:
        if target in seen:
            continue
        seen.add(target)
        units.append(FanoutUnit(target=target, argv=build_target_argv(target, layout, options)))
    return units


def completed_targets(results_dir: Path | str) -> set[str]:
    """Return the targets that already have a completion marker.

    Markers are the JSON files ``main_regression_target`` writes under
    ``{results_dir}/_completed/<sanitized_target>.json``; the original target
    name is read from the payload because the file name is sanitized. Markers
    that are unreadable or incomplete are skipped, so a partially written
    marker just means the target runs again.
    """
    completed_dir = Path(results_dir) / COMPLETED_DIRNAME
    if not completed_dir.is_dir():
        return set()

    targets: set[str] = set()
    for marker in sorted(completed_dir.glob("*.json")):
        try:
            payload = json.loads(marker.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        target = payload.get("target") if isinstance(payload, dict) else None
        if isinstance(target, str) and target:
            targets.add(target)
    return targets
