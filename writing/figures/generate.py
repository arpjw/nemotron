"""Regenerate the article's measured-result SVG figures.

Requires matplotlib. Values come from results/BANKING77.md,
results/BANKING77_PER_CLASS.json, and results/SYNTHETIC_FINANCE.md;
this script does not score predictions.
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

OUT = Path(__file__).resolve().parent
PER_CLASS = OUT.parent.parent / "results" / "BANKING77_PER_CLASS.json"
plt.rcParams.update({"font.family": "DejaVu Sans", "svg.fonttype": "none", "svg.hashsalt": "nemotron-article"})
THEMES = {
    "light": {"bg": "#fbfaf7", "ink": "#172126", "muted": "#68767d", "grid": "#d8dfdf", "base": "#83939b", "tuned": "#a4523e"},
    "dark": {"bg": "#11191b", "ink": "#f3f3ed", "muted": "#b4c0c3", "grid": "#3c4b50", "base": "#9fb1b8", "tuned": "#e7a58d"},
}


def finish(fig: plt.Figure, name: str, theme: str) -> None:
    path = OUT / f"{name}-{theme}.svg"
    fig.savefig(path, format="svg", bbox_inches="tight", pad_inches=0.22, metadata={"Date": None})
    path.write_text("\n".join(line.rstrip() for line in path.read_text().splitlines()) + "\n")
    plt.close(fig)


def macro_f1(theme: str, c: dict[str, str]) -> None:
    fig, ax = plt.subplots(figsize=(9, 3.7), dpi=160)
    fig.patch.set_facecolor(c["bg"])
    ax.set_facecolor(c["bg"])
    rows = [("Development", 1993, 0.3991469667, 0.8297779466), ("Official test", 3080, 0.4133733218, 0.8444279358)]
    for y, (name, n, base, tuned) in zip((1, 0), rows):
        ax.plot([base, tuned], [y, y], color=c["grid"], linewidth=7, solid_capstyle="round", zorder=1)
        ax.scatter([base], [y], s=180, color=c["base"], zorder=3)
        ax.scatter([tuned], [y], s=180, color=c["tuned"], zorder=3)
        ax.text(base - 0.025, y + 0.19, f"Base {base:.3f}", ha="right", color=c["muted"], fontsize=11)
        ax.text(tuned + 0.025, y + 0.19, f"Adapter {tuned:.3f}", ha="left", color=c["ink"], fontsize=11, weight="bold")
    ax.set_yticks([1, 0], ["Development\n1,993 queries", "Official test\n3,080 queries"])
    ax.set_xlim(0.25, 0.98)
    ax.set_ylim(-0.55, 1.55)
    ax.set_xticks(np.arange(0.3, 1.0, 0.1))
    ax.set_xlabel("Macro-F1 · same taxonomy-constrained decoder", color=c["muted"], labelpad=12)
    ax.grid(axis="x", color=c["grid"], alpha=0.75)
    for spine in ax.spines.values(): spine.set_visible(False)
    ax.tick_params(axis="both", colors=c["muted"], length=0, labelsize=10)
    ax.tick_params(axis="y", pad=18)
    fig.tight_layout()
    finish(fig, "macro-f1", theme)


def synthetic_limits(theme: str, c: dict[str, str]) -> None:
    fig, ax = plt.subplots(figsize=(9, 3.8), dpi=160)
    fig.patch.set_facecolor(c["bg"])
    ax.set_facecolor(c["bg"])
    names = ["Valid JSON", "Exact answers", "Correct numeric value"]
    counts = ["400 / 400", "168 / 400", "105 / 320"]
    fractions = [1.0, 168 / 400, 105 / 320]
    ys = [2, 1, 0]
    ax.barh(ys, [100] * 3, height=0.44, color=c["grid"], alpha=0.42)
    ax.barh(ys, [100 * x for x in fractions], height=0.44, color=[c["base"], c["tuned"], c["tuned"]])
    for y, fraction, count in zip(ys, fractions, counts):
        ax.text(min(100 * fraction + 2.4, 102), y, f"{count}  ({fraction:.1%})", va="center", color=c["ink"], fontsize=11, weight="bold")
    ax.set_yticks(ys, names)
    ax.set_xlim(0, 127)
    ax.set_xticks([0, 25, 50, 75, 100], ["0%", "25%", "50%", "75%", "100%"])
    ax.set_xlabel("Correct on corrected synthetic development data", color=c["muted"], labelpad=12)
    ax.grid(axis="x", color=c["grid"], alpha=0.75)
    ax.set_axisbelow(True)
    for spine in ax.spines.values(): spine.set_visible(False)
    ax.tick_params(axis="both", colors=c["muted"], length=0, labelsize=10)
    ax.tick_params(axis="y", pad=18)
    fig.tight_layout()
    finish(fig, "synthetic-limits", theme)


def intent_coverage(theme: str, c: dict[str, str]) -> None:
    """Plot the empirical survival curve of official per-intent F1 scores."""
    rows = json.loads(PER_CLASS.read_text())["classes"]
    assert len(rows) == 77 and sum(row["support"] for row in rows) == 3080
    thresholds = np.linspace(0, 1, 501)
    base = np.array([sum(row["base_f1"] >= t for row in rows) for t in thresholds])
    tuned = np.array([sum(row["adapter_f1"] >= t for row in rows) for t in thresholds])
    assert sum(row["base_f1"] >= 0.8 for row in rows) == 6
    assert sum(row["adapter_f1"] >= 0.8 for row in rows) == 55

    fig, ax = plt.subplots(figsize=(9, 4.1), dpi=160)
    fig.patch.set_facecolor(c["bg"])
    ax.set_facecolor(c["bg"])
    ax.fill_between(thresholds, base, tuned, color=c["tuned"], alpha=0.12, step="post")
    ax.step(thresholds, base, where="post", color=c["base"], linewidth=2.7, label="Untouched Nemotron")
    ax.step(thresholds, tuned, where="post", color=c["tuned"], linewidth=3.2, label="Selected adapter")
    ax.axvline(0.8, color=c["grid"], linestyle="--", linewidth=1.4)
    ax.scatter([0.8, 0.8], [6, 55], color=[c["base"], c["tuned"]], s=80, zorder=4)
    ax.annotate("6 intents", (0.8, 6), xytext=(10, 6), textcoords="offset points", color=c["base"], fontsize=10)
    ax.annotate("55 intents", (0.8, 55), xytext=(10, 6), textcoords="offset points", color=c["tuned"], fontsize=10, weight="bold")
    ax.set(xlim=(0, 1), ylim=(0, 79), xticks=np.arange(0, 1.01, 0.2), yticks=[0, 20, 40, 60, 77])
    ax.set_xlabel("Minimum per-intent F1 on the official test", color=c["muted"], labelpad=11)
    ax.set_ylabel("Number of intents at or above threshold", color=c["muted"], labelpad=11)
    ax.grid(color=c["grid"], alpha=0.65)
    ax.set_axisbelow(True)
    for spine in ax.spines.values(): spine.set_visible(False)
    ax.tick_params(axis="both", colors=c["muted"], length=0, labelsize=10)
    legend = ax.legend(frameon=False, loc="upper right", fontsize=10)
    for item in legend.get_texts(): item.set_color(c["ink"])
    fig.tight_layout()
    finish(fig, "intent-coverage", theme)


if __name__ == "__main__":
    for theme_name, colors in THEMES.items():
        macro_f1(theme_name, colors)
        synthetic_limits(theme_name, colors)
        intent_coverage(theme_name, colors)
