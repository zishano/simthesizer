"""Plot one deployment point per run: tokens/s/user versus tokens/s/GPU."""

import argparse
from pathlib import Path

from results import load_run, positive_float, protect_output, summarize_run


def draw(runs, output, title=None, size=None):
    points = []
    unavailable = []
    for run in runs:
        metrics = summarize_run(run)
        x = metrics["tokens_per_user"]["mean"]
        y = metrics["tokens_per_gpu"]["mean"]
        if x is None or y is None:
            unavailable.append(run.label)
            continue
        points.append((run, x, y))

    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    from matplotlib.ticker import MaxNLocator

    figure, ax = plt.subplots(figsize=size or (7.5, 5.5), layout="constrained")
    colors = plt.get_cmap("tab10").colors
    for index, (run, x, y) in enumerate(points):
        color = colors[index % len(colors)]
        ax.scatter(
            [x],
            [y],
            s=75,
            color=color,
            zorder=3,
            label=f"{run.label} ({len(run.requests)} requests, {run.gpu_count} GPUs)",
        )
        ax.vlines(x, 0, y, colors=color, linestyles="--", alpha=0.3)
        ax.hlines(y, 0, x, colors=color, linestyles="--", alpha=0.3)
        ax.annotate(
            f"({x:.2f}, {y:.2f})",
            (x, y),
            xytext=(-12, -14),
            textcoords="offset points",
            ha="right",
            va="top",
            color=color,
        )
    ax.set(
        title=title or "Service throughput",
        xlabel="Tokens/s/user",
        ylabel="Tokens/s/GPU",
        xlim=(0, max((x for _, x, _ in points), default=0) * 1.2 or 1),
        ylim=(0, max((y for _, _, y in points), default=0) * 1.2 or 1),
    )
    ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
    ax.grid(True, alpha=0.2)
    ax.set_axisbelow(True)
    if points:
        ax.legend(loc="upper left", frameon=False, fontsize="small")
    else:
        ax.text(
            0.5,
            0.5,
            "No eligible deployment point",
            ha="center",
            transform=ax.transAxes,
        )
    missing = (
        "\nNo point for "
        + ", ".join(unavailable)
        + ": decode rate or GPU allocation unavailable."
        if unavailable
        else ""
    )
    figure.supxlabel(
        "User: mean decode rate per LLM call (excluding TTFT).\n"
        "GPU: whole-service output throughput / all allocated GPUs." + missing,
        fontsize=9,
        color="0.35",
    )
    try:
        output.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output, dpi=180)
    finally:
        plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="+", help="[LABEL=]PATH; one point per run")
    parser.add_argument("--output", type=Path, required=True, help="PNG, SVG or PDF")
    parser.add_argument("--title")
    parser.add_argument(
        "--size", nargs=2, type=positive_float, metavar=("WIDTH", "HEIGHT")
    )
    args = parser.parse_args()
    try:
        runs = [load_run(spec) for spec in args.runs]
        if len({run.label for run in runs}) != len(runs):
            raise ValueError("duplicate run labels; use LABEL=PATH to distinguish them")
        protect_output(args.output, runs)
        draw(runs, args.output, args.title, args.size)
        print(args.output)
    except (ValueError, OSError) as exc:
        parser.exit(1, f"error: {exc}\n")


if __name__ == "__main__":
    main()
