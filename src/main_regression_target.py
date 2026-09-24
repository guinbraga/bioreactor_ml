"""Non-interactive, config-driven regression batch runner.

This is the unattended counterpart of ``main_regression_batch.py``. It runs the
exact same workload -- every feature subset discovered under
``--feature-base-dir`` plus an all-features baseline -- for the target column(s)
given on the command line, but takes every input from arguments and a JSON
config instead of interactive prompts.

It is designed to be invoked once per target so independent targets can be
fanned out across parallel workers or machines:

    python src/main_regression_target.py \\
        --genomic  data/genomic/03_map_complete_absolute_n_hits_table.csv \\
        --metadata data/metadata/06_metadata_reactor_clean.csv \\
        --index-column "Cluster Sample ID" \\
        --target "pH" \\
        --config configs/regression_target.example.json \\
        --results-dir results/40_batch_regression

Important: the feature matrix excludes *every* column listed under
``target_columns`` in the config, not only the target being predicted. This
mirrors ``DataManager.get_X_y`` in the interactive batch runner, so the full
target set must live in the config even when you run a single target. A
per-target invocation that only knew its own target would silently use the
other targets as features and produce different numbers.

Output layout is identical to ``main_regression_batch.py`` so both can share
``results/40_batch_regression``.

Exit codes:
    0  every requested target finished successfully
    1  at least one target failed (see the log for the traceback)
    2  invalid arguments or configuration
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import platform
import re
import socket
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, TypedDict

# Allow ``python src/main_regression_target.py`` from the repo root: the
# pipeline modules import each other by bare name (``from controller import``).
sys.path.insert(0, str(Path(__file__).resolve().parent))

from controller import (  # noqa: E402
    evaluate_experiment_regression,
    fetch_metadata_columns,
    fetch_pipeline_components,
    setup_data,
)

LOGGER = logging.getLogger("regression_target")

DEFAULT_FEATURE_BASE_DIR = Path("results/38_cluster_importances")
DEFAULT_RESULTS_DIR = Path("results/40_batch_regression")
COMPLETED_DIRNAME = "_completed"
ALL_FEATURES_DIRNAME = "all_features"


class ConfigError(ValueError):
    """Raised when CLI arguments or the JSON config are invalid."""


class ModelConfig(TypedDict):
    selected_scaler_sequences: list[tuple[str, ...]]
    selected_selectors: list[str]
    cv: str
    groups: str | None


class ExperimentConfig(TypedDict):
    target_columns: list[str]
    models: dict[str, ModelConfig]


class FeatureSubset(TypedDict):
    experiment: str
    model: str
    csv_path: str


# ====== Argument parsing ====== #


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="main_regression_target.py",
        description=(
            "Run the batch regression pipeline for one target column (or a "
            "small set) without interactive prompts."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--genomic",
        required=True,
        type=Path,
        help="Path to the genomic data (samples as columns, features as rows).",
    )
    parser.add_argument(
        "--metadata",
        required=True,
        type=Path,
        help="Path to the experiment metadata (samples as rows).",
    )
    parser.add_argument(
        "--index-column",
        required=True,
        help="Metadata column used to join with the genomic samples.",
    )
    parser.add_argument(
        "--config",
        required=True,
        type=Path,
        help="JSON config with the target set and model definitions.",
    )
    parser.add_argument(
        "--target",
        action="append",
        default=[],
        dest="targets",
        metavar="COLUMN",
        help=(
            "Target column to run. Repeatable. Must be one of the config's "
            "'target_columns'. Defaults to every column in the config."
        ),
    )
    parser.add_argument(
        "--targets-file",
        type=Path,
        default=None,
        help="Optional file with one target column per line ('#' comments allowed).",
    )
    parser.add_argument(
        "--feature-base-dir",
        type=Path,
        default=DEFAULT_FEATURE_BASE_DIR,
        help="Directory holding <experiment>/<model>/<model>_top_features.csv.",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=DEFAULT_RESULTS_DIR,
        help="Root directory for results (same layout as the batch runner).",
    )
    parser.add_argument(
        "--experiment",
        action="append",
        default=[],
        dest="experiments",
        metavar="NAME",
        help="Only run feature subsets from this experiment dir. Repeatable.",
    )
    parser.add_argument(
        "--feature-model",
        action="append",
        default=[],
        dest="feature_models",
        metavar="NAME",
        help="Only run feature subsets from this model dir. Repeatable.",
    )
    parser.add_argument(
        "--all-features-only",
        action="store_true",
        help="Skip feature subsets and run only the all-features baseline.",
    )
    parser.add_argument(
        "--skip-completed",
        action="store_true",
        help="Skip targets that already have a completion marker.",
    )
    parser.add_argument(
        "--no-persist",
        action="store_true",
        help="Run the pipeline but do not write results (smoke test).",
    )
    parser.add_argument(
        "--output-profile",
        choices=["full", "slim"],
        default="full",
        help=(
            "'full' persists per-sample SHAP waterfall PNGs (current behavior); "
            "'slim' skips them to cut output size ~90%."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the execution plan and exit without running anything.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity.",
    )
    return parser


# ====== Config loading and validation ====== #


def load_config(config_path: Path) -> ExperimentConfig:
    """Load and structurally validate the JSON experiment config."""
    if not config_path.is_file():
        raise ConfigError(f"Config file not found: {config_path}")
    try:
        raw: Any = json.loads(config_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ConfigError(f"Config file {config_path} is not valid JSON: {exc}") from exc

    if not isinstance(raw, dict):
        raise ConfigError("Config must be a JSON object.")

    target_columns = raw.get("target_columns")
    if (
        not isinstance(target_columns, list)
        or not target_columns
        or not all(isinstance(column, str) and column for column in target_columns)
    ):
        raise ConfigError(
            "'target_columns' must be a non-empty list of column names. "
            "It defines the full set of targets excluded from the features."
        )

    raw_models = raw.get("models")
    if not isinstance(raw_models, dict) or not raw_models:
        raise ConfigError("'models' must be a non-empty object keyed by model name.")

    models: dict[str, ModelConfig] = {}
    for name, cfg in raw_models.items():
        if not isinstance(cfg, dict):
            raise ConfigError(f"[{name}] config must be an object.")

        sequences = cfg.get("selected_scaler_sequences")
        if not isinstance(sequences, list) or not sequences:
            raise ConfigError(
                f"[{name}] 'selected_scaler_sequences' must be a non-empty list."
            )
        normalized_sequences: list[tuple[str, ...]] = []
        for sequence in sequences:
            if not isinstance(sequence, list) or not all(
                isinstance(scaler, str) for scaler in sequence
            ):
                raise ConfigError(
                    f"[{name}] each scaler sequence must be a list of strings."
                )
            normalized_sequences.append(tuple(sequence))

        selectors = cfg.get("selected_selectors")
        if (
            not isinstance(selectors, list)
            or not selectors
            or not all(isinstance(selector, str) for selector in selectors)
        ):
            raise ConfigError(
                f"[{name}] 'selected_selectors' must be a non-empty list of strings."
            )

        cv = cfg.get("cv")
        if not isinstance(cv, str) or not cv:
            raise ConfigError(f"[{name}] 'cv' must be a non-empty string.")

        groups = cfg.get("groups")
        if groups is not None and not isinstance(groups, str):
            raise ConfigError(f"[{name}] 'groups' must be a string or null.")

        models[name] = {
            "selected_scaler_sequences": normalized_sequences,
            "selected_selectors": list(selectors),
            "cv": cv,
            "groups": groups,
        }

    return {"target_columns": list(target_columns), "models": models}


def validate_model_configs(
    models: dict[str, ModelConfig], components: dict[str, list[str]]
) -> None:
    """Check every model/scaler/selector/cv name against the pipeline registry."""
    regression_models = {m for m in components["models"] if "Regressor" in m}
    valid_scalers = set(components["scalers"])
    valid_selectors = set(components["selectors"])
    valid_cvs = set(components["cross_validators"])

    for name, cfg in models.items():
        if name not in regression_models:
            raise ConfigError(
                f"Model '{name}' is not an available regression model. "
                f"Available: {sorted(regression_models)}"
            )
        if cfg["cv"] not in valid_cvs:
            raise ConfigError(
                f"[{name}] Unknown cv '{cfg['cv']}'. Available: {sorted(valid_cvs)}"
            )
        if cfg["cv"] == "Leave One Group Out" and not cfg["groups"]:
            raise ConfigError(
                f"[{name}] 'groups' is required when cv is 'Leave One Group Out'."
            )
        for sequence in cfg["selected_scaler_sequences"]:
            for scaler in sequence:
                if scaler not in valid_scalers:
                    raise ConfigError(
                        f"[{name}] Unknown scaler '{scaler}'. "
                        f"Available: {sorted(valid_scalers)}"
                    )
        for selector in cfg["selected_selectors"]:
            if selector not in valid_selectors:
                raise ConfigError(
                    f"[{name}] Unknown selector '{selector}'. "
                    f"Available: {sorted(valid_selectors)}"
                )


def validate_columns(
    config: ExperimentConfig,
    targets: list[str],
    index_column: str,
    available_columns: list[str],
) -> None:
    """Check targets, index and group columns against the metadata header."""
    available = set(available_columns)
    unknown_targets = [t for t in targets if t not in available]
    if unknown_targets:
        raise ConfigError(
            f"Target column(s) not found in metadata: {unknown_targets}. "
            f"Available: {available_columns}"
        )

    missing_exclusions = [c for c in config["target_columns"] if c not in available]
    if missing_exclusions:
        raise ConfigError(
            "These config 'target_columns' are missing from the metadata and "
            f"would break feature exclusion: {missing_exclusions}"
        )

    if index_column not in available:
        raise ConfigError(
            f"Index column '{index_column}' not found in metadata. "
            f"Available: {available_columns}"
        )
    if index_column in config["target_columns"]:
        raise ConfigError(
            f"Index column '{index_column}' is also a target column; "
            "they must be disjoint."
        )

    for name, cfg in config["models"].items():
        groups = cfg["groups"]
        if groups is not None and groups not in available:
            raise ConfigError(
                f"[{name}] groups column '{groups}' not found in metadata. "
                f"Available: {available_columns}"
            )


# ====== Feature subset discovery ====== #


def discover_feature_subsets(
    base_dir: Path,
    experiments: list[str] | None = None,
    feature_models: list[str] | None = None,
) -> list[FeatureSubset]:
    """Find the first <model>_top_features[_suffix].csv under each model dir.

    A model directory may hold several feature-set variants (``_1``, ``_1T``,
    ``_Hindgut``, ...). Exactly one subset is returned per model: the
    lexicographically first variant, which keeps the plain
    ``<model>_top_features.csv`` layout working unchanged.
    """
    if not base_dir.is_dir():
        raise ConfigError(f"Feature base dir not found: {base_dir}")

    experiment_filter = set(experiments or [])
    model_filter = set(feature_models or [])
    subsets: list[FeatureSubset] = []

    for experiment_dir in sorted(base_dir.iterdir()):
        if not experiment_dir.is_dir():
            continue
        if experiment_filter and experiment_dir.name not in experiment_filter:
            continue
        for model_dir in sorted(experiment_dir.iterdir()):
            if not model_dir.is_dir():
                continue
            if model_filter and model_dir.name not in model_filter:
                continue
            # Model names contain spaces and may contain glob metacharacters,
            # so match file names directly instead of using ``Path.glob``.
            pattern = re.compile(
                rf"{re.escape(model_dir.name)}_top_features(_[^.]+)?\.csv$"
            )
            matches = sorted(
                entry.name
                for entry in model_dir.iterdir()
                if entry.is_file() and pattern.search(entry.name)
            )
            if matches:
                subsets.append(
                    {
                        "experiment": experiment_dir.name,
                        "model": model_dir.name,
                        "csv_path": str(model_dir / matches[0]),
                    }
                )
    return subsets


# ====== Targets ====== #


def resolve_targets(cli_targets: list[str], targets_file: Path | None) -> list[str]:
    """Combine --target and --targets-file, trimming and de-duplicating."""
    targets: list[str] = []
    for target in cli_targets:
        cleaned = target.strip()
        if cleaned:
            targets.append(cleaned)

    if targets_file is not None:
        if not targets_file.is_file():
            raise ConfigError(f"--targets-file not found: {targets_file}")
        for line in targets_file.read_text(encoding="utf-8").splitlines():
            cleaned = line.strip()
            if cleaned and not cleaned.startswith("#"):
                targets.append(cleaned)

    seen: set[str] = set()
    unique: list[str] = []
    for target in targets:
        if target not in seen:
            seen.add(target)
            unique.append(target)
    return unique


# ====== Completion markers ====== #


def completion_marker_path(results_dir: Path, target: str) -> Path:
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", target).strip("._-") or "target"
    return results_dir / COMPLETED_DIRNAME / f"{safe}.json"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def write_completion_marker(
    marker_path: Path,
    target: str,
    args: argparse.Namespace,
    config: ExperimentConfig,
    subsets: list[FeatureSubset],
) -> None:
    marker_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "target": target,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "results_dir": str(args.results_dir),
        "feature_base_dir": str(args.feature_base_dir),
        "config_path": str(args.config),
        "config_sha256": _sha256(args.config),
        "models": list(config["models"]),
        "feature_subsets": [f"{s['experiment']}/{s['model']}" for s in subsets],
        "all_features_only": args.all_features_only,
        "hostname": socket.gethostname(),
        "python": platform.python_version(),
    }
    marker_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


# ====== Progress callbacks ====== #


def on_model_begin(model_name: str) -> None:
    LOGGER.info("  starting evaluation for %s", model_name)


def on_split_begin(i: int, n_splits: int) -> None:
    LOGGER.debug("    split %d/%d", i + 1, n_splits)


def on_persist(model_name: str) -> None:
    LOGGER.info("  persisting results for %s", model_name)


def on_complete(model_name: str) -> None:
    LOGGER.info("  finished %s", model_name)


# ====== Execution ====== #


def run_target(
    data_manager: Any,
    target: str,
    config: ExperimentConfig,
    subsets: list[FeatureSubset],
    args: argparse.Namespace,
) -> None:
    """Run the all-features baseline and every feature subset for one target."""
    n_subsets = 0 if args.all_features_only else len(subsets)
    total_runs = 1 + n_subsets
    run_idx = 0

    run_idx += 1
    LOGGER.info("(%d/%d) target=%s -- all features", run_idx, total_runs, target)
    evaluate_experiment_regression(
        data_manager=data_manager,
        target_column=target,
        selected_models=config["models"],
        results_dir=str(Path(args.results_dir) / ALL_FEATURES_DIRNAME),
        selected_features_csv=None,
        on_model_begin=on_model_begin,
        on_split_begin=on_split_begin,
        on_complete=on_complete,
        on_persist=on_persist,
        persist_to_disk=not args.no_persist,
        output_profile=args.output_profile,
    )

    if args.all_features_only:
        return

    for subset in subsets:
        run_idx += 1
        label = f"{subset['experiment']} / {subset['model']}"
        LOGGER.info("(%d/%d) target=%s -- features from %s", run_idx, total_runs, target, label)
        evaluate_experiment_regression(
            data_manager=data_manager,
            target_column=target,
            selected_models=config["models"],
            results_dir=str(
                Path(args.results_dir) / subset["experiment"] / subset["model"]
            ),
            selected_features_csv=subset["csv_path"],
            on_model_begin=on_model_begin,
            on_split_begin=on_split_begin,
            on_complete=on_complete,
            on_persist=on_persist,
            persist_to_disk=not args.no_persist,
            output_profile=args.output_profile,
        )


def _configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stdout,
        force=True,
    )


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    _configure_logging(args.log_level)

    try:
        config = load_config(args.config)
        components = fetch_pipeline_components()
        validate_model_configs(config["models"], components)

        targets = resolve_targets(args.targets, args.targets_file) or list(
            config["target_columns"]
        )
        not_in_config = [t for t in targets if t not in set(config["target_columns"])]
        if not_in_config:
            raise ConfigError(
                f"Target(s) {not_in_config} are not listed in the config's "
                "'target_columns'. Add them to the config so they are excluded "
                "from the feature matrix."
            )

        available_columns = fetch_metadata_columns(args.metadata)
        validate_columns(config, targets, args.index_column, available_columns)

        subsets = (
            []
            if args.all_features_only
            else discover_feature_subsets(
                args.feature_base_dir, args.experiments, args.feature_models
            )
        )
    except ConfigError as exc:
        LOGGER.error("%s", exc)
        return 2

    if not args.all_features_only and not subsets:
        LOGGER.warning(
            "No feature subsets found under %s; only the all-features baseline "
            "will run.",
            args.feature_base_dir,
        )

    n_subsets = 0 if args.all_features_only else len(subsets)
    total_runs = len(targets) * (1 + n_subsets)
    LOGGER.info(
        "Plan: %d target(s) x (1 all-features + %d subset(s)) = %d run(s)",
        len(targets),
        n_subsets,
        total_runs,
    )
    for target in targets:
        LOGGER.info("  target: %s", target)
    for subset in subsets:
        LOGGER.info("  subset: %s / %s", subset["experiment"], subset["model"])
    LOGGER.info("Models: %s", ", ".join(config["models"]))
    LOGGER.info("Results dir: %s", args.results_dir)
    LOGGER.info(
        "Output profile: %s (%s)",
        args.output_profile,
        "persist per-sample SHAP waterfall PNGs"
        if args.output_profile == "full"
        else "skip per-sample SHAP waterfall PNGs",
    )

    if args.dry_run:
        LOGGER.info("Dry run: nothing executed.")
        return 0

    merge_results, data_manager = setup_data(
        genomic_file_path=args.genomic,
        metadata_file_path=args.metadata,
        target_columns=config["target_columns"],
        metadata_file_index=args.index_column,
    )
    LOGGER.info(
        "Merge summary: genomic=%s metadata=%s merged=%s excluded=%s",
        merge_results["genomic_n_samples"],
        merge_results["metadata_n_samples"],
        merge_results["merged_n_samples"],
        merge_results["excluded_samples"],
    )

    failures: list[str] = []
    for target in targets:
        marker = completion_marker_path(args.results_dir, target)
        if args.skip_completed and marker.exists():
            LOGGER.info("Skipping target=%s (marker exists: %s)", target, marker)
            continue
        try:
            run_target(data_manager, target, config, subsets, args)
        except Exception:
            LOGGER.exception("Target %s failed", target)
            failures.append(target)
            continue
        if not args.no_persist:
            write_completion_marker(marker, target, args, config, subsets)
        LOGGER.info("Target %s complete.", target)

    if failures:
        LOGGER.error(
            "Finished with %d failed target(s): %s", len(failures), ", ".join(failures)
        )
        return 1

    LOGGER.info("All targets complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
