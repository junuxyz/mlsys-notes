"""Compare benchmark times with an H100 peak roofline estimate.

Run: python analyze.py results/modal
"""

import argparse
import csv
import json
import statistics
from pathlib import Path

import theory


def summarize(data):
    rows = []
    for q in data["q_lens"]:
        baseline = statistics.median(
            s["time_us"]
            for s in data["samples"]
            if s["q_len"] == q and s["method"] == "standard"
        )
        for method in data["methods"]:
            times = [
                s["time_us"] / 1000
                for s in data["samples"]
                if s["q_len"] == q and s["method"] == method
            ]
            estimate = theory.estimate_time(q, data["kv_len"], method)
            ideal_baseline = theory.estimate_time(q, data["kv_len"], "standard")
            median = statistics.median(times)
            rows.append(
                {
                    "q_len": q,
                    "kv_len": data["kv_len"],
                    "method": method,
                    **estimate,
                    "measured_ms": median,
                    "min_ms": min(times),
                    "max_ms": max(times),
                    "speedup": baseline / 1000 / median,
                    "ideal_speedup": ideal_baseline["estimate_ms"] / estimate["estimate_ms"],
                    "measured_over_ideal": median / estimate["estimate_ms"],
                }
            )
    return rows


def plot(rows, data, output):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import to_rgb
    from matplotlib.patches import Patch
    from matplotlib.transforms import offset_copy

    bg, ink, ink_2, muted, rule = "#fbfaf7", "#1c1c20", "#5b5a60", "#a09e96", "#e4e1d9"
    colors = {
        "standard": "#12b886",
        "unmerged_absorb": "#3b5bdb",
        "merged_absorb": "#e8590c",
    }
    labels = {
        "standard": "Standard",
        "unmerged_absorb": "Absorbed (unmerged)",
        "merged_absorb": "Absorbed (merged)",
    }
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Inter", "Avenir Next", "Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
            "font.size": 9,
            "text.color": ink,
        }
    )

    def tint(color, amount=0.24):
        return tuple(a * amount + b * (1 - amount) for a, b in zip(to_rgb(color), to_rgb(bg), strict=True))

    def change(r):
        s = r["speedup"]
        return f"{s:.1f}\u00d7 faster" if s >= 1 else f"{1 / s:.1f}\u00d7 slower"

    def headline(q, cases):
        phase = {1: "decode", data["kv_len"]: "prefill"}.get(q)
        name = f"q_len = {q} ({phase})" if phase else f"q_len = {q}"
        others = [r for r in cases if r["method"] != "standard"]
        best = max(others, key=lambda r: r["speedup"])
        if best["speedup"] >= 1:
            return f"{name}: {labels[best['method']]} is {change(best)}"
        slow = sorted(1 / r["speedup"] for r in others)
        return f"{name}: absorption is {slow[0]:.1f}\u2013{slow[-1]:.1f}\u00d7 slower"

    n = len(data["q_lens"])
    fig, axes = plt.subplots(n, 1, figsize=(9, 1.4 + 2.05 * n), squeeze=False, facecolor=bg)
    for ax, q in zip(axes[:, 0], data["q_lens"], strict=True):
        cases = [row for row in rows if row["q_len"] == q]
        ax.set_facecolor(bg)
        top = max(r["max_ms"] for r in cases)
        winner = min(cases, key=lambda r: r["measured_ms"])
        for y, r in enumerate(cases):
            c = colors[r["method"]]
            win = r is winner
            ax.barh(y, r["measured_ms"], 0.6, color=tint(c, 0.24 if win else 0.12), lw=0)
            ax.barh(y, r["estimate_ms"], 0.6, color=c if win else tint(c, 0.45), lw=0)
            ax.hlines(y, r["min_ms"], r["max_ms"], color=c if win else tint(c, 0.45), lw=1.1)
            ax.text(
                r["max_ms"] + top * 0.02,
                y,
                f"{r['measured_ms']:.2f} ms",
                va="center",
                fontsize=9.5,
                fontweight="semibold" if r is winner else "normal",
                color=ink if r is winner else ink_2,
            )
        ax.set_yticks(
            range(len(cases)), [labels[r["method"]] for r in cases], fontsize=9.5, color=ink_2
        )
        for tick, r in zip(ax.get_yticklabels(), cases, strict=True):
            tick.set_fontweight("semibold" if r is winner else "normal")
            tick.set_color(ink if r is winner else ink_2)
        ax.invert_yaxis()
        ax.set_xlim(0, top * 1.2)
        ax.set_ylim(len(cases) - 0.45, -0.55)
        ax.xaxis.set_major_formatter(lambda v, _: "0" if v == 0 else f"{v:g} ms")
        ax.xaxis.set_major_locator(plt.MaxNLocator(4))
        ax.tick_params(axis="x", colors=ink_2, labelsize=8.5, length=0, pad=5)
        ax.tick_params(axis="y", length=0, pad=10)
        ax.spines[:].set_visible(False)
        ax.spines["bottom"].set_visible(True)
        ax.spines["bottom"].set_color(rule)
        bounds = sorted({r["bound"] for r in cases})
        ax.set_title(
            headline(q, cases),
            loc="left",
            fontsize=10.5,
            fontweight="semibold",
            color=ink,
            pad=26,
            va="baseline",
        )
        ax.text(
            0.0,
            1.0,
            f"all {bounds[0]} bound" if len(bounds) == 1 else " / ".join(bounds) + " bound",
            transform=offset_copy(ax.transAxes, fig=fig, y=10, units="points"),
            ha="left",
            va="baseline",
            fontsize=9,
            color=ink,
        )

    fig.text(0.02, 0.965, "MLA comparison", fontsize=15, fontweight="semibold", color=ink, va="top",
             fontfamily=["Inter Display", "Inter", "sans-serif"])
    fig.text(
        0.02,
        0.915,
        "Ideal roofline vs. measured, bf16",
        fontsize=9.5,
        color=muted,
        va="top",
    )
    fig.legend(
        handles=[
            Patch(color=ink_2, label="ideal (roofline)"),
            Patch(color=tint(ink_2), label="measured"),
        ],
        loc="upper right",
        bbox_to_anchor=(0.98, 0.968),
        ncol=2,
        frameon=False,
        fontsize=8.5,
        handlelength=1.2,
        handleheight=0.8,
        columnspacing=1.4,
        labelcolor=ink_2,
    )
    fig.tight_layout(rect=(0.0, 0.0, 1, 0.86), h_pad=2.2)
    fig.savefig(output / "comparison.png", dpi=200, facecolor=bg)
    fig.savefig(output / "comparison.svg", facecolor=bg)
    plt.close(fig)


def plot_speedup(rows, data, output):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    bg, ink, muted, rule = "#fbfaf7", "#16161a", "#8a887f", "#e6e3db"
    theory_c, measured_c = "#cfd6f6", "#3b5bdb"
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Inter", "Avenir Next", "Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
            "font.size": 9,
            "text.color": ink,
        }
    )
    labels = {"unmerged_absorb": "Unmerged", "merged_absorb": "Merged"}

    def phase(q):
        return "Decode" if q == 1 else "Prefill" if q == data["kv_len"] else f"q_len={q}"

    cases = [r for r in rows if r["method"] in labels]
    x = range(len(cases))
    width = 0.36

    fig, ax = plt.subplots(figsize=(7, 4), facecolor=bg)
    ax.set_facecolor(bg)
    for offset, key, color, label in (
        (-width / 2, "ideal_speedup", theory_c, "Theory"),
        (width / 2, "speedup", measured_c, "Measured"),
    ):
        bars = ax.bar([i + offset for i in x], [r[key] for r in cases], width * 0.92,
                      color=color, label=label, zorder=2)
        for text in ax.bar_label(bars, fmt="%.2f\u00d7", padding=3, fontsize=8.5, color=ink):
            text.set_bbox({"boxstyle": "square,pad=0.1", "fc": bg, "ec": "none"})
    ax.axhline(1, color=ink, lw=0.8, ls=(0, (3, 3)), zorder=1)
    ax.text(len(cases) - 0.42, 1.03, "Standard = 1\u00d7", va="bottom", ha="right", fontsize=8,
            color=muted)

    ax.set_xticks(list(x), [f"{labels[r['method']]}\n{phase(r['q_len'])}" for r in cases],
                  fontsize=9.5)
    ax.set_xlim(-0.6, len(cases) - 0.4)
    ax.set_ylim(0, max(max(r["speedup"], r["ideal_speedup"]) for r in cases) * 1.15)
    ax.set_ylabel("Speedup vs. Standard", color=muted, fontsize=8.5)
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:g}\u00d7")
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.spines["bottom"].set_color(rule)
    ax.tick_params(length=0)
    ax.tick_params(axis="y", colors=muted, labelsize=8)
    ax.grid(axis="y", color=rule, lw=0.7)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=8.5, loc="upper right", handlelength=1, handleheight=1)
    ax.set_title("MLA absorption speedup",
                 loc="left", fontsize=12, fontweight="semibold", pad=14,
                 fontfamily=["Inter Display", "Inter", "sans-serif"])
    fig.tight_layout()
    fig.savefig(output / "speedup.png", dpi=200, facecolor=bg)
    fig.savefig(output / "speedup.svg", facecolor=bg)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path)
    args = parser.parse_args()
    path = args.results
    if path.is_dir():
        (path,) = path.rglob("benchmark.json")
    data = json.loads(path.read_text())
    rows = summarize(data)
    output = path.parent / "analysis"
    output.mkdir(exist_ok=True)
    with (output / "comparison.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(
        f"{'q_len':>5} {'Method':<18} {'Model bound':>12} {'Ideal ms':>10} {'Measured ms':>12} {'Ideal speedup':>14} {'Speedup':>8} {'Measured/ideal':>15}"
    )
    for row in rows:
        print(
            f"{row['q_len']:>5} {row['method']:<18} {row['bound']:>12} {row['estimate_ms']:>10.3f} {row['measured_ms']:>12.3f} {row['ideal_speedup']:>13.2f}x {row['speedup']:>7.2f}x {row['measured_over_ideal']:>14.2f}x"
        )
    plot(rows, data, output)
    plot_speedup(rows, data, output)
    print(f"Saved comparison.csv, comparison.png/svg, speedup.png/svg in {output}")


if __name__ == "__main__":
    main()
