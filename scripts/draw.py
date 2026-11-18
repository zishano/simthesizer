"""Plot complete serving results. The first run is the scatter reference."""

import argparse
import json
import math
from pathlib import Path

from results import (
    LATENCIES,
    load_runs,
    measurements,
    metric_unit,
    positive_float,
    protect_output,
    summarize_run,
    throughput,
)


def draw(runs, output, metrics, view, window, title, columns, size):
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    from matplotlib.ticker import MaxNLocator

    samples = [measurements(run) for run in runs]
    end = max(sample["makespan"][()] for sample in samples)
    columns = min(columns, len(metrics))
    rows = math.ceil(len(metrics) / columns)
    figure, axes = plt.subplots(
        rows,
        columns,
        figsize=size or (5 * columns, 3.5 * rows),
        squeeze=False,
        layout="constrained",
    )
    colors = plt.get_cmap("tab10").colors
    for ax, metric in zip(axes.flat, metrics):
        ax.grid(True, alpha=0.2)
        ax.set_axisbelow(True)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
        if metric == "throughput":
            for i, run in enumerate(runs):
                edges, values = throughput(run, window, end)
                ax.stairs(
                    values,
                    edges,
                    label=run.label,
                    color=colors[i % len(colors)],
                    linewidth=1.5,
                )
            ax.set(
                title=f"Output throughput ({window:g} s bins)",
                xlabel="Time since first arrival (s)",
                ylabel="Output tokens / s",
                xlim=(0, (math.floor(end / window) + 1) * window),
                ylim=(0, None),
            )
        elif metric == "cumulative":
            for i, run in enumerate(runs):
                origin = run.origin
                times = sorted(time - origin for time in run.token_times())
                ax.step(
                    [0, *times, end],
                    [0, *range(1, len(times) + 1), len(times)],
                    where="post",
                    label=run.label,
                    color=colors[i % len(colors)],
                )
            ax.set(
                title="Cumulative output",
                xlabel="Time since first arrival (s)",
                ylabel="Output tokens",
                xlim=(0, max(end, window)),
                ylim=(0, None),
            )
        elif view == "cdf":
            for i, (run, sample) in enumerate(zip(runs, samples)):
                values = list(sample[metric].values())
                if values:
                    ax.ecdf(
                        values,
                        label=run.label,
                        color=colors[i % len(colors)],
                    )
            ax.set(
                title=LATENCIES[metric],
                xlabel=f"Latency ({metric_unit(metric)})",
                ylabel="CDF",
                xlim=(0, None),
                ylim=(0, 1),
            )
        else:
            reference = samples[0][metric]
            maximum = max(reference.values(), default=0)
            for i, (run, sample) in enumerate(zip(runs[1:], samples[1:]), 1):
                candidate = sample[metric]
                if reference:
                    ax.scatter(
                        list(reference.values()),
                        [candidate[key] for key in reference],
                        s=12,
                        alpha=0.45,
                        label=run.label,
                        color=colors[i % len(colors)],
                        rasterized=metric == "itl",
                    )
                    maximum = max(maximum, max(candidate.values()))
            bound = maximum * 1.05 or 1
            ax.plot([0, bound], [0, bound], color="0.4", linestyle="--", linewidth=1)
            ax.set(
                title=LATENCIES[metric],
                xlabel=f"{runs[0].label} ({metric_unit(metric)})",
                ylabel=f"Prediction ({metric_unit(metric)})",
                xlim=(0, bound),
                ylim=(0, bound),
            )
            ax.set_aspect("equal", adjustable="box")
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(handles, labels, fontsize="small", frameon=False)
        else:
            ax.text(
                0.5,
                0.5,
                "No eligible samples",
                ha="center",
                transform=ax.transAxes,
            )
    for ax in list(axes.flat)[len(metrics) :]:
        ax.remove()
    if title:
        figure.suptitle(title)
    try:
        output.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output, dpi=180)
    finally:
        plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "runs", nargs="+", help="[LABEL=]PATH to result JSONLs; first is the reference"
    )
    parser.add_argument(
        "--output", type=Path, required=True, help="PNG, SVG, or PDF path"
    )
    parser.add_argument(
        "--metrics-json", type=Path, help="also refresh metric summaries (one run only)"
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        choices=[*LATENCIES, "throughput", "cumulative"],
        default=[
            "ttft",
            "tpot",
            "itl",
            "e2e",
            "request_e2e",
            "throughput",
        ],
    )
    parser.add_argument(
        "--view",
        choices=["cdf", "scatter"],
        default="cdf",
        help="latency view; throughput and cumulative always use time",
    )
    parser.add_argument(
        "--window-sec",
        type=positive_float,
        default=1,
        help="throughput bin width; does not affect comparison metrics",
    )
    parser.add_argument("--title")
    parser.add_argument("--columns", type=int, default=3)
    parser.add_argument(
        "--size",
        nargs=2,
        type=positive_float,
        metavar=("WIDTH", "HEIGHT"),
        help="figure size in inches",
    )
    args = parser.parse_args()
    try:
        runs = load_runs(args.runs)
        if args.columns < 1:
            raise ValueError("columns must be positive")
        if args.view == "scatter" and len(runs) < 2:
            raise ValueError("scatter requires a reference and at least one candidate")
        protect_output(args.output, runs)
        if args.metrics_json:
            if len(runs) != 1:
                raise ValueError("--metrics-json requires exactly one run")
            protect_output(args.metrics_json, runs)
            if args.metrics_json.resolve() == args.output.resolve():
                raise ValueError("metrics JSON and figure must use different paths")
        draw(
            runs,
            args.output,
            list(dict.fromkeys(args.metrics)),
            args.view,
            args.window_sec,
            args.title,
            args.columns,
            args.size,
        )
        if args.metrics_json:
            args.metrics_json.parent.mkdir(parents=True, exist_ok=True)
            args.metrics_json.write_text(
                json.dumps(summarize_run(runs[0]), indent=2, allow_nan=False) + "\n"
            )
        print(args.output)
    except (ValueError, OSError) as exc:
        parser.exit(1, f"error: {exc}\n")


if __name__ == "__main__":
    main()
