"""Offline end-to-end replay of the bundled public paper evidence."""
from __future__ import annotations

import json
import shutil
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ..provenance import sha256_file
from .evidence import (
    PAPER_RESOURCES,
    load_resource,
    metric_agreement,
    plot_agreement,
    verify_all,
)

EXPERIMENTS = {
    "main-table": "Main rebenchmark table and per-dataset scores",
    "metric-agreement": "Direction-aware rank correlations and heatmaps",
    "sampling-sweeps": "Recorded NFE and naive-sampler sweeps (historical protocol)",
    "mauve-sensitivity": "MAUVE scored-count curves and measurements",
    "context-horizons": "Gen-PPL context-horizon measurements",
    "jcd-directions": "Directional coding divergences and diagnostic plot",
    "entropy": "Entropy census, controls, and diagnostic plot",
    "encoder-sensitivity": "Encoder scores, ranks, and Vega-Lite specifications",
}

METRICS = ["gen_ppl", "h_emp", "self_bleu", "mauve", "gm", "jcd", "ed", "fbd", "mmd"]
NUMERIC_EXPORTS = {
    "mauve_sample_count.json": "data/mauve_sample_count.json",
    "gen_ppl_context_horizons.json": "data/gen_ppl_context_horizons.json",
    "jcd_directional.json": "data/jcd_directional.json",
    "entropy_census.json": "data/entropy_census.json",
    "entropy_controls.json": "data/entropy_controls.json",
    "encoder_rank_sensitivity.json": "specs/encoder_rank_sensitivity.vega.json",
    "encoder_rank_movement.json": "data/encoder_rank_movement.json",
    "encoder_pairwise_agreement.json": "specs/encoder_pairwise_agreement.vega.json",
    "encoder_pair_leaderboard.json": "data/encoder_pair_leaderboard.json",
    "encoder_scores_ranks.json": "data/encoder_scores_ranks.json",
    "archived_sweep.json": "data/sampling_sweeps.json",
}


def _cohort(views: Any) -> str:
    values = views if isinstance(views, list) else str(views or "").split(",")
    names = {value.strip() for value in values}
    if "lm1b" in names:
        return "LM1B (L=128)"
    if "short-list" in names:
        return "OWT short list"
    return "OWT all models"


def _central_and_interval(value: Any) -> tuple[float, float | None, float | None] | None:
    """Parse a recorded number, optionally with an interval or uncertainty."""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        central = float(value)
        return central, None, None
    text = str(value).strip()
    if not text:
        return None
    try:
        if "±" in text:
            central_text, uncertainty_text = text.split("±", 1)
            central = float(central_text.strip())
            uncertainty = float(uncertainty_text.strip().split()[0])
            return central, central - uncertainty, central + uncertainty
        if text.startswith("[") and text.endswith("]"):
            lower_text, upper_text = text[1:-1].split(",", 1)
            lower = float(lower_text)
            upper = float(upper_text)
            return (lower + upper) / 2, lower, upper
        central = float(text.split()[0])
        return central, None, None
    except (ValueError, TypeError) as error:
        raise ValueError(f"unparseable numeric evidence value {value!r}") from error


def _save_figure(figure: Any, destination: Path) -> None:
    if destination.exists():
        raise ValueError(f"refusing to overwrite existing output {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=200, bbox_inches="tight")


def _utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
                    encoding="utf-8")


def _export_main_table(destination: Path) -> None:
    from .evidence import export_rebenchmark
    export_rebenchmark(load_resource("rebenchmark"), destination / "tables/main.csv")
    export_rebenchmark(load_resource("rebenchmark"), destination / "tables/main.json")
    for dataset in ("owt", "lm1b"):
        export_rebenchmark(load_resource("rebenchmark"),
                           destination / "tables" / f"{dataset}-generated.csv",
                           dataset=dataset, include_reference=False)


def _export_agreement(destination: Path) -> None:
    payload = load_resource("rebenchmark")
    agreements = {
        dataset: metric_agreement(payload, dataset, METRICS)
        for dataset in ("owt", "lm1b")
    }
    _write_json(destination / "tables/metric_agreement.json", agreements)
    for dataset in ("owt", "lm1b"):
        plot_agreement(payload, destination / "figures" / f"agreement-{dataset}.png",
                       dataset=dataset, metrics=METRICS)


def _export_numeric_sources(destination: Path, sources: list[str]) -> None:
    for name in sources:
        relative = NUMERIC_EXPORTS[name + ".json"]
        name += ".json"
        if name not in PAPER_RESOURCES.values():
            raise ValueError(f"replay source is not a packaged resource: {name}")
        payload = load_resource(next(key for key, filename in PAPER_RESOURCES.items()
                                     if filename == name))
        _write_json(destination / relative, payload)


def _plot_mauve_sample_count(destination: Path) -> None:
    """Render all recorded MAUVE scored-count series with recorded intervals."""
    try:
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise ImportError("plotting requires matplotlib from the paper extra") from error
    payload = load_resource("mauve_sample_count")
    counts = [int(str(label).split("=")[-1].strip()) for label in payload["labels"]]
    if counts != sorted(counts) or len(set(counts)) != len(counts):
        raise ValueError("MAUVE sample-count labels must be unique and increasing")
    cohorts = ["OWT short list", "OWT all models", "LM1B (L=128)"]
    figure, axes = plt.subplots(3, 1, figsize=(16, 14), sharex=True)
    for axis, cohort in zip(axes, cohorts, strict=True):
        plotted = 0
        for row in payload["rows"]:
            if _cohort(row["views"]) != cohort:
                continue
            points = [_central_and_interval(row.get(f"value_{index}"))
                      for index in range(1, len(counts) + 1)]
            valid = [(count, point) for count, point in zip(counts, points, strict=True)
                     if point is not None]
            if not valid:
                continue
            x_values = [item[0] for item in valid]
            y_values = [item[1][0] for item in valid]
            lower = [item[1][1] for item in valid if item[1][1] is not None]
            upper = [item[1][2] for item in valid if item[1][2] is not None]
            axis.plot(x_values, y_values, linewidth=1.2, alpha=0.85,
                      label=row["label"], marker=("o", "s", "^", "v", "D")[plotted % 5],
                      markersize=3, color=plt.get_cmap("turbo")((plotted % 40) / 39))
            if lower and len(lower) == len(valid):
                axis.fill_between(x_values, lower, upper, linewidth=0, alpha=0.08)
            plotted += 1
        axis.set_title(f"{cohort} ({plotted} series)")
        axis.legend(loc="upper left", bbox_to_anchor=(1.01, 1), fontsize=6, ncol=2)
        axis.set_ylabel("MAUVE")
        axis.grid(True, alpha=0.2)
    axes[-1].set_xlabel("Scored samples")
    axes[-1].set_xticks(counts, [str(count) for count in counts])
    figure.suptitle("MAUVE scored-count sensitivity (recorded central estimates and intervals)")
    figure.tight_layout()
    _save_figure(figure, destination)
    plt.close(figure)


def _plot_jcd_directional(destination: Path) -> None:
    """Render the recorded directional excess-codelength matrix."""
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError as error:
        raise ImportError("plotting requires matplotlib and numpy from the paper extra") from error
    payload = load_resource("jcd_directional")
    columns = [column["label"] for column in payload["columns"]]
    rows = [row for row in payload["rows"] if not row.get("section")]
    matrix = np.full((len(rows), len(columns)), np.nan)
    for row_index, row in enumerate(rows):
        for column_index, value in enumerate(row["values"]):
            point = _central_and_interval(value)
            if point is not None:
                matrix[row_index, column_index] = point[0]
    figure, axis = plt.subplots(figsize=(11, 16))
    image = axis.imshow(matrix, aspect="auto", cmap="viridis")
    axis.set_xticks(range(len(columns)), columns, rotation=35, ha="right")
    axis.set_yticks(range(len(rows)), [row["label"] for row in rows], fontsize=5)
    for row_index in range(len(rows)):
        for column_index in range(len(columns)):
            value = matrix[row_index, column_index]
            if np.isfinite(value):
                axis.text(column_index, row_index, f"{value:g}", ha="center", va="center",
                          fontsize=4.5)
    axis.set_title("Recorded directional excess codelength (central estimates)")
    figure.colorbar(image, ax=axis, label="Excess codelength")
    figure.tight_layout()
    _save_figure(figure, destination)
    plt.close(figure)


def _plot_entropy_diagnostics(destination: Path) -> None:
    """Render entropy census values and the eight exact controls."""
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError as error:
        raise ImportError("plotting requires matplotlib and numpy from the paper extra") from error
    census = load_resource("entropy_census")
    controls = load_resource("entropy_controls")
    census_rows = [row for row in census["rows"] if not row.get("section")]
    census_columns = [column["label"] for column in census["columns"]]
    control_columns = [column["label"] for column in controls["columns"]]
    census_matrix = np.asarray([[float(value) for value in row["values"]]
                                 for row in census_rows], dtype=float)
    control_matrix = np.asarray([[float(value) for value in row["values"]]
                                 for row in controls["rows"]], dtype=float)
    figure, (census_axis, control_axis) = plt.subplots(
        2, 1, figsize=(11, 14),
        gridspec_kw={"height_ratios": [len(census_rows), len(controls["rows"])]})
    census_image = census_axis.imshow(census_matrix, aspect="auto", cmap="viridis")
    census_axis.set_xticks(range(len(census_columns)), census_columns, rotation=25, ha="right")
    census_axis.set_yticks(range(len(census_rows)), [row["label"] for row in census_rows], fontsize=5)
    census_axis.set_title("Entropy census (recorded values)")
    figure.colorbar(census_image, ax=census_axis, label="Nats")
    control_image = control_axis.imshow(control_matrix, aspect="auto", cmap="viridis")
    control_axis.set_xticks(range(len(control_columns)), control_columns, rotation=25, ha="right")
    control_axis.set_yticks(range(len(controls["rows"])), [row["label"] for row in controls["rows"]], fontsize=7)
    control_axis.set_title("Eight exact entropy controls")
    figure.colorbar(control_image, ax=control_axis, label="Nats")
    figure.tight_layout()
    _save_figure(figure, destination)
    plt.close(figure)


def _run_experiments(destination: Path, selected: list[str]) -> None:
    for name in selected:
        if name == "main-table":
            _export_main_table(destination)
        elif name == "metric-agreement":
            _export_agreement(destination)
        elif name == "sampling-sweeps":
            _export_numeric_sources(destination, ["archived_sweep"])
        elif name == "mauve-sensitivity":
            _export_numeric_sources(destination, ["mauve_sample_count"])
            _plot_mauve_sample_count(destination / "figures/mauve-sample-count.png")
        elif name == "context-horizons":
            _export_numeric_sources(destination, ["gen_ppl_context_horizons"])
        elif name == "jcd-directions":
            _export_numeric_sources(destination, ["jcd_directional"])
            _plot_jcd_directional(destination / "figures/jcd-directional.png")
        elif name == "entropy":
            _export_numeric_sources(destination, ["entropy_census", "entropy_controls"])
            _plot_entropy_diagnostics(destination / "figures/entropy-diagnostics.png")
        elif name == "encoder-sensitivity":
            _export_numeric_sources(destination, [
                "encoder_rank_sensitivity", "encoder_rank_movement",
                "encoder_pairwise_agreement", "encoder_pair_leaderboard", "encoder_scores_ranks",
            ])


def reproduce_run(destination: Path, *, experiment: str = "all") -> dict[str, Any]:
    """Reconstruct one named analysis, or all analyses, from released measurements."""
    if experiment != "all" and experiment not in EXPERIMENTS:
        raise ValueError(f"unknown experiment {experiment!r}; choose from {list(EXPERIMENTS)}")
    selected = list(EXPERIMENTS) if experiment == "all" else [experiment]
    destination = Path(destination).expanduser()
    if destination.exists():
        if destination.is_dir() and not any(destination.iterdir()):
            shutil.rmtree(destination)
        else:
            raise ValueError(f"refusing to overwrite existing output {destination}")
    parent = destination.parent
    parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{destination.name}.", dir=parent))
    try:
        verification = verify_all()
        if verification["status"] != "verified":
            raise ValueError(f"bundled evidence verification failed: {verification}")
        _run_experiments(temporary, selected)
        outputs = []
        for path in sorted(item for item in temporary.rglob("*") if item.is_file()):
            outputs.append({"path": path.relative_to(temporary).as_posix(),
                            "bytes": path.stat().st_size, "sha256": sha256_file(path)})
        manifest = {
            "schema_version": 1,
            "generated_at_utc": _utc_now(),
            "source_resources": sorted(PAPER_RESOURCES),
            "outputs": outputs,
            "experiments": selected,
            "verification": verification,
            "limitations": {
                "replay": "Uses released measurements; does not rerun metric inference.",
                "missing_exact_references": ["owt_data_train", "lm1b_data_train"],
                "raw_judgments_bundled": False,
                "judgment_replay": "Excluded from this release.",
                "encoder_figures": "Vega-Lite specification exports, not rendered figures.",
                "sampling_sweeps": "Older recorded protocol, separate from the main table.",
            },
        }
        _write_json(temporary / "replay-manifest.json", manifest)
        temporary.replace(destination)
    finally:
        shutil.rmtree(temporary, ignore_errors=True)
    loaded = json.loads((destination / "replay-manifest.json").read_text(encoding="utf-8"))
    return {"status": "written", "path": str(destination), "outputs": loaded["outputs"]}


__all__ = ["EXPERIMENTS", "reproduce_run"]
