"""Plot the accuracy-vs-latency Pareto frontier from results/results.json.

One line per strategy; each marker is a knob value (e.g. k=1,3,5,9). Reading
vertically at any latency budget tells you which strategy wins that budget.
"""
from __future__ import annotations

import json
import os

RESULTS_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "results", "results.json")
PLOT_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "results", "pareto.png")

_LABELS = {"input_output": "Input–output (direct)",
           "self_consistency": "Self-consistency",
           "react": "ReAct",
           "agentic": "Agent (tool-use)",
           "iterative": "Self-Refine",
           "tree_of_thoughts": "Tree of Thoughts",
           "fleet_of_agents": "Fleet of Agents"}


def plot(results_path: str = RESULTS_PATH, out_path: str = PLOT_PATH) -> str:
    import matplotlib.pyplot as plt

    with open(os.path.normpath(results_path)) as f:
        data = json.load(f)

    by_strategy: dict[str, list[dict]] = {}
    for r in data["results"]:
        by_strategy.setdefault(r["strategy"], []).append(r)

    fig, ax = plt.subplots(figsize=(7, 5))
    for strat, rows in by_strategy.items():
        rows.sort(key=lambda r: r["mean_latency_s"])
        xs = [r["mean_latency_s"] for r in rows]
        ys = [100 * r["accuracy"] for r in rows]
        ax.plot(xs, ys, "o-", label=_LABELS.get(strat, strat), linewidth=2,
                markersize=7)
        for r in rows:  # annotate each point with its knob value
            ax.annotate(f"{r['knob']}={r['knob_value']}",
                        (r["mean_latency_s"], 100 * r["accuracy"]),
                        textcoords="offset points", xytext=(6, 6), fontsize=8)

    ax.set_xlabel("Mean latency per problem (s)  ->  larger budget")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title(f"Accuracy vs. latency on GSM8K  ({data['model']}, n={data['n']})")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.normpath(out_path), dpi=150)
    print(f"Saved -> {os.path.normpath(out_path)}")
    return out_path
