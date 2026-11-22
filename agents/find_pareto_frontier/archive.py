"""Derive provisional per-case Pareto sets from completed evaluation receipts."""

import argparse
from pathlib import Path

import launcher


def objectives(row):
    point = row["objectives"]
    return (
        point["ttft_p99"],
        *([point["tpot_p99"]] if row["tpot_applicable"] else []),
        -point["goodput_per_gpu"],
    )


def dominates(left, right):
    a, b = objectives(left), objectives(right)
    if len(a) != len(b):
        raise ValueError("inconsistent TPOT applicability within a case group")
    return all(x <= y for x, y in zip(a, b)) and any(x < y for x, y in zip(a, b))


def pareto(rows):
    groups = {}
    for row in rows:
        if row.get("event") == "finished" and row.get("status") == "VALID_FEASIBLE":
            groups.setdefault(row["group_id"], {})[row["candidate_id"]] = row
    result = {}
    for group, candidates in groups.items():
        frontier = []
        for row in sorted(candidates.values(), key=lambda item: item["candidate_id"]):
            if any(dominates(other, row) for other in candidates.values()):
                continue
            equivalent = next(
                (old for old in frontier if objectives(old) == objectives(row)), None
            )
            if equivalent:
                equivalent["equivalent_candidates"].append(row["candidate_id"])
            else:
                frontier.append(
                    {**row, "equivalent_candidates": [], "verification": "provisional"}
                )
        result[group] = frontier
    return result


def update_archive(artifacts):
    artifacts = Path(artifacts)
    from evaluate import Experiment

    experiment = Experiment(artifacts.parent / "task/manifest.json")
    if experiment.artifacts.resolve() != artifacts.resolve():
        raise ValueError("archive must use the frozen run's artifacts directory")
    groups = pareto(experiment.records())
    launcher.save(
        artifacts / "frontier.json", {"verification": "provisional", "groups": groups}
    )
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(9, 6))
    for points in groups.values():
        label = points[0]["case"]["id"]
        axis.scatter(
            [p["objectives"]["ttft_p99"] for p in points],
            [p["objectives"]["goodput_per_gpu"] for p in points],
            label=label,
        )
    axis.set(
        xlabel="TTFT p99 (s)",
        ylabel="SLO goodput (output token/s/GPU)",
        title="Provisional Pareto candidates; each workload/load is a separate group",
    )
    if groups:
        axis.legend(fontsize=8)
    axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(artifacts / "frontier.png", dpi=140)
    plt.close(figure)
    return groups


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", required=True, type=Path)
    args = parser.parse_args()
    update_archive(args.artifacts)
