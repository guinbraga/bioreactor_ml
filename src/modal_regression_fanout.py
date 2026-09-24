"""Modal fan-out for the per-target regression batch.

Every target runs ``main_regression_target.main(argv)`` in its own CPU-only
container, and the 25 targets fan out with ``Function.map`` (free plan allows
100 concurrent containers). No GPU is requested: the workload is plain
scikit-learn.

Run from the repo root with a local Python env that has ``modal`` plus the
regression dependencies installed:

    modal run src/modal_regression_fanout.py --dry-run
    modal run src/modal_regression_fanout.py
    modal run src/modal_regression_fanout.py --targets "pH,CH4 (%)"

Results and completion markers land on the ``bioreactor-results`` volume under
``/results/44_reg_no_corr``.
"""

from __future__ import annotations

from pathlib import Path
from typing import TypedDict

import modal


class TargetResult(TypedDict):
    """What a worker reports back for one target."""

    target: str
    exit_code: int


APP_NAME = "bioreactor-regression-batch"
VOLUME_NAME = "bioreactor-results"
VOLUME_MOUNT = "/results"

# Container layout. Keep these in sync with the image below and with the
# RuntimeLayout built in the local entrypoint.
SRC_DIR = "/root/src"
GENOMIC = "/root/data/genomic/03_map_complete_absolute_n_hits_table.csv"
METADATA = "/root/data/metadata/06_metadata_reactor_clean.csv"
INDEX_COLUMN = "Cluster Sample ID"
CONFIG = "/root/configs/regression_target.example.json"
FEATURE_BASE = "/root/features"
RESULTS_DIR = "/results/44_reg_no_corr"

# Local checkout root; only used to declare the image's local sources (Modal
# resolves them lazily at run time, never at import time).
REPO_ROOT = Path(__file__).resolve().parent.parent

app = modal.App(APP_NAME)

volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)

# Only the regression-path dependencies are installed; see requirements-modal.txt.
# The feature dir ships CSV subsets only: the ignore list keeps the 15
# <model>_top_features.csv inputs and drops the PNGs/parquet/joblib/tex/json.
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install_from_requirements(str(REPO_ROOT / "requirements-modal.txt"))
    .add_local_file(
        str(REPO_ROOT / "requirements-modal.txt"), "/root/requirements-modal.txt"
    )
    .add_local_dir(str(REPO_ROOT / "src"), SRC_DIR, ignore=["**/__pycache__"])
    .add_local_dir(str(REPO_ROOT / "data"), "/root/data")
    .add_local_dir(str(REPO_ROOT / "configs"), "/root/configs")
    .add_local_dir(
        str(REPO_ROOT / "results" / "42_no_corr"),
        FEATURE_BASE,
        ignore=["**/*.png", "**/*.parquet", "**/*.joblib", "**/*.tex", "**/*.json"],
    )
)


@app.function(
    image=image,
    volumes={VOLUME_MOUNT: volume},
    cpu=4,
    memory=8192,
    timeout=14400,
    max_containers=25,
    retries=1,
)
def run_target(unit: tuple[str, tuple[str, ...]]) -> TargetResult:
    """Run one target in its own container and commit the results volume."""
    import sys

    # /root/src is mounted but not on PYTHONPATH; make the pipeline importable.
    sys.path.insert(0, SRC_DIR)
    from main_regression_target import main

    target, argv = unit
    exit_code = main(list(argv))
    # Commit inside the function so finished targets survive even if the
    # container is later preempted.
    volume.commit()
    return {"target": target, "exit_code": exit_code}


@app.local_entrypoint()
def fanout(
    dry_run: bool = False,
    all_features_only: bool = False,
    output_profile: str = "slim",
    max_containers: int = 25,
    targets: str = "",
) -> None:
    """Build the plan locally, then fan it out over Modal containers."""
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from regression_fanout_plan import (
        FanoutOptions,
        RuntimeLayout,
        feature_subset_labels,
        plan_fanout,
    )

    layout = RuntimeLayout(
        genomic=GENOMIC,
        metadata=METADATA,
        index_column=INDEX_COLUMN,
        config=CONFIG,
        feature_base=FEATURE_BASE,
        results_dir=RESULTS_DIR,
    )
    options = FanoutOptions(
        targets=tuple(part.strip() for part in targets.split(",") if part.strip()),
        all_features_only=all_features_only,
        output_profile=output_profile,
    )
    units = plan_fanout(
        config_path=REPO_ROOT / "configs" / "regression_target.example.json",
        feature_base_dir=REPO_ROOT / "results" / "42_no_corr",
        results_dir=REPO_ROOT / "results" / "44_reg_no_corr",
        layout=layout,
        options=options,
    )

    print(f"Plan: {len(units)} target(s) -> {RESULTS_DIR}")
    if options.all_features_only:
        print("Feature subsets: skipped (all_features_only)")
    else:
        feature_labels = feature_subset_labels(
            REPO_ROOT / "results" / "42_no_corr",
            options.experiments,
            options.feature_models,
        )
        print(f"Feature subsets: {len(feature_labels)} (first variant per model)")
    if dry_run:
        for unit in units:
            print(f"  target={unit.target}")
            print(f"    main_regression_target {' '.join(unit.argv)}")
        print("Dry run: nothing executed.")
        return

    # FanoutUnit lives in regression_fanout_plan, which is not on the worker's
    # PYTHONPATH; plain (target, argv) tuples unpickle without importing it.
    inputs = [(unit.target, unit.argv) for unit in units]
    results = list(
        run_target.with_options(max_containers=max_containers).map(
            inputs, return_exceptions=True
        )
    )

    succeeded = 0
    failures: list[str] = []
    for unit, result in zip(units, results):
        if isinstance(result, BaseException):
            failures.append(f"{unit.target}: exception {result!r}")
        elif result["exit_code"] == 0:
            succeeded += 1
        else:
            failures.append(f"{unit.target}: exit code {result['exit_code']}")

    print(f"Fan-out finished: {succeeded} ok, {len(failures)} failed of {len(units)}")
    for failure in failures:
        print(f"  FAILED {failure}")
    if failures:
        raise SystemExit(1)
