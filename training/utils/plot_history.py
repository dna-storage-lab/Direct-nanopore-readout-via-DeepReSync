from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


DEFAULT_METRICS = [
    ("val.upper188_err", "Upper188 error"),
    ("val.lower235_err", "Lower235 error"),
]


def load_history(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        history = json.load(f)
    if not isinstance(history, list):
        raise ValueError(f"{path} must contain a JSON list")
    if not history:
        raise ValueError(f"{path} is empty")
    return history


def metric_value(row: dict, metric: str) -> float | None:
    parts = metric.split(".")
    value: object = row
    for part in parts:
        if not isinstance(value, dict) or part not in value:
            return None
        value = value[part]
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def metric_series(history: list[dict], metric: str) -> tuple[list[int], list[float]]:
    epochs: list[int] = []
    values: list[float] = []
    for index, row in enumerate(history, start=1):
        value = metric_value(row, metric)
        if value is None:
            continue
        epochs.append(int(row.get("epoch", index)))
        values.append(value)
    return epochs, values


def run_label(path: Path) -> str:
    parent = path.parent.name
    grandparent = path.parent.parent.name
    if grandparent.startswith("final"):
        return parent
    return f"{grandparent}/{parent}"


def run_labels(paths: list[Path]) -> dict[Path, str]:
    labels = {path: run_label(path) for path in paths}
    counts: dict[str, int] = {}
    for label in labels.values():
        counts[label] = counts.get(label, 0) + 1
    for path, label in list(labels.items()):
        if counts[label] > 1:
            labels[path] = f"{path.parent.parent.name}/{path.parent.name}"
    return labels


def quantile(values: list[float], q: float) -> float:
    if not values:
        raise ValueError("quantile requires non-empty values")
    ordered = sorted(values)
    index = (len(ordered) - 1) * q
    lower = math.floor(index)
    upper = math.ceil(index)
    if lower == upper:
        return ordered[lower]
    fraction = index - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def zoom_error_axis(ax: plt.Axes, values: list[float], target_lines: list[float]) -> None:
    """Zoom low-error plots so late training trends are visible."""

    finite = [value for value in values if math.isfinite(value)]
    if not finite:
        return

    low_region = [value for value in finite if value <= 0.12]
    zoom_values = low_region if len(low_region) >= 10 else finite
    ymin = max(0.0, min(zoom_values) * 0.85)
    ymax = quantile(zoom_values, 0.90) * 1.2
    visible_targets = [line for line in target_lines if line >= ymin]
    if visible_targets:
        ymax = max(ymax, max(visible_targets) * 1.15)

    if min(zoom_values) < 0.06:
        ymax = max(ymax, 0.06)
    if ymax <= ymin:
        ymax = ymin + 0.01
    ax.set_ylim(ymin, ymax)


def discover_histories(root: Path) -> list[Path]:
    if root.is_file():
        return [root]
    return sorted(path for path in root.rglob("history.json") if path.is_file())


def parse_metric_specs(specs: list[str]) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    for spec in specs:
        if ":" in spec:
            metric, title = spec.split(":", 1)
        else:
            metric = spec
            title = spec.replace("train.", "Train ").replace("val.", "Val ").replace("_", " ")
        out.append((metric, title))
    return out


def plot_histories(
    history_paths: list[Path],
    output_path: Path,
    metrics: list[tuple[str, str]],
    dpi: int,
    target_lines: list[float],
) -> None:
    histories = [(path, load_history(path)) for path in history_paths]
    if not histories:
        raise ValueError("no history files found")
    labels = run_labels(history_paths)

    available_metrics: list[tuple[str, str]] = []
    for metric, title in metrics:
        if any(metric_series(history, metric)[1] for _, history in histories):
            available_metrics.append((metric, title))
    if not available_metrics:
        raise ValueError("none of the requested metrics exist in the selected histories")

    cols = 2 if len(available_metrics) > 1 else 1
    rows = math.ceil(len(available_metrics) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(9.5 * cols, 5.8 * rows), dpi=dpi, squeeze=False)

    for axis_index, (metric, title) in enumerate(available_metrics):
        ax = axes[axis_index // cols][axis_index % cols]
        plotted_values: list[float] = []
        for path, history in histories:
            epochs, values = metric_series(history, metric)
            if not values:
                continue
            plotted_values.extend(values)
            ax.plot(epochs, values, linewidth=1.6, label=labels[path])
        if metric.endswith("_err"):
            for target in target_lines:
                ax.axhline(target, linestyle="--", linewidth=0.9, color="0.5", alpha=0.7)
            zoom_error_axis(ax, plotted_values, target_lines)
        ax.set_title(title)
        ax.set_xlabel("epoch")
        ax.grid(True, alpha=0.28)
        ax.legend(fontsize=8)

    for axis_index in range(len(available_metrics), rows * cols):
        axes[axis_index // cols][axis_index % cols].axis("off")

    fig.suptitle("DeepResync training histories", fontsize=12)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)

    print(f"saved {output_path}")
    for path, history in histories:
        last = history[-1]
        pieces = [labels[path], f"epoch={last.get('epoch', len(history))}"]
        for metric, _ in available_metrics[:4]:
            value = metric_value(last, metric)
            if value is not None:
                pieces.append(f"{metric}={value:.4f}")
        print("  " + "  ".join(pieces))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot DeepResync training curves from one or more history.json files.")
    parser.add_argument(
        "--history",
        nargs="+",
        default=None,
        help="One or more history.json files. Also accepts a run directory containing history.json files.",
    )
    parser.add_argument("--output", default=None, help="Output PNG path.")
    parser.add_argument(
        "--metric",
        nargs="*",
        default=None,
        help=(
            "Metrics to plot, e.g. val.loss val.info_base_err. "
            "Use metric:title to override subplot title."
        ),
    )
    parser.add_argument("--target-err", type=float, nargs="*", default=[0.03, 0.02, 0.01])
    parser.add_argument("--dpi", type=int, default=160)
    parser.add_argument("--watch", action="store_true", help="Keep refreshing the plot.")
    parser.add_argument("--interval", type=float, default=60.0, help="Refresh interval in seconds.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    inputs = args.history or ["runs/final7_fast_basehead_from_sup_cos2e4_min3e5"]

    history_paths: list[Path] = []
    for item in inputs:
        history_paths.extend(discover_histories(Path(item)))
    history_paths = sorted(dict.fromkeys(history_paths))

    if not history_paths:
        print("no history.json files found", file=sys.stderr)
        return 1

    if args.output:
        output_path = Path(args.output)
    elif len(history_paths) == 1:
        output_path = history_paths[0].with_name("history_curves.png")
    else:
        common = history_paths[0].parent.parent
        output_path = common / "history_compare.png"

    metrics = parse_metric_specs(args.metric) if args.metric else DEFAULT_METRICS

    while True:
        try:
            plot_histories(history_paths, output_path, metrics, args.dpi, args.target_err)
        except (json.JSONDecodeError, OSError) as exc:
            print(f"history is not ready yet: {exc}", file=sys.stderr)
        except Exception as exc:
            print(f"failed to plot history: {exc}", file=sys.stderr)
            return 1

        if not args.watch:
            return 0
        time.sleep(args.interval)


if __name__ == "__main__":
    raise SystemExit(main())
