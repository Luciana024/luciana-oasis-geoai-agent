"""Paper figures for rolling graph-fusion alpha. Does not overwrite website_article_v1."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from common.utils import project_root

EDI_ALPHA = Path("data/results/exports/website_article_v1/website/rolling_alpha.csv")
GLA_ALPHA = Path("data/results/regions/S12000049/rolling/final_test/W730/rolling_alpha.csv")
OUT_DIRS = (
    Path("docs/figures"),
    Path("data/results/exports/paper_tables_combined_v1/figures"),
)

SERIES = (
    ("alpha_geo", r"$\alpha_{\mathrm{geo}}$", "#1f77b4", "o"),
    ("alpha_transport", r"$\alpha_{\mathrm{transport}}$", "#ff7f0e", "s"),
    ("alpha_mobility", r"$\alpha_{\mathrm{mobility}}$", "#2ca02c", "^"),
)


def _load(rel: Path) -> pd.DataFrame:
    frame = pd.read_csv(project_root() / rel)
    frame["update_id"] = frame["update_id"].astype(str)
    return frame.sort_values("update_id")


def _draw(ax, frame: pd.DataFrame, title: str, *, show_ylabel: bool) -> None:
    xs = list(range(len(frame)))
    labels = frame["update_id"].tolist()
    for col, label, color, marker in SERIES:
        ys = frame[col].astype(float).tolist()
        ax.plot(xs, ys, color=color, marker=marker, markersize=5.5, linewidth=1.6, label=label)
        ax.scatter([xs[-1]], [ys[-1]], s=42, facecolors="none", edgecolors=color, linewidths=1.4, zorder=3)
    ax.axvline(xs[-1], color="0.65", linestyle=":", linewidth=0.9)
    ax.set_xticks(xs, labels)
    ax.set_ylim(0.20, 0.52)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("Rolling update")
    if show_ylabel:
        ax.set_ylabel(r"Graph-fusion weight $\alpha$")
    ax.grid(True, axis="y", linestyle="--", linewidth=0.4, alpha=0.7)


def plot_alpha_trajectories() -> dict[str, str]:
    edi = _load(EDI_ALPHA)
    gla = _load(GLA_ALPHA)
    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.85), sharey=True)
    _draw(axes[0], edi, "Edinburgh (111 IZs)", show_ylabel=True)
    _draw(axes[1], gla, "Glasgow (136 IZs)", show_ylabel=False)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.04))
    fig.tight_layout(rect=(0, 0, 1, 0.92))

    singles = {
        "edinburgh": (edi, "Edinburgh", "fig_alpha_edinburgh"),
        "glasgow": (gla, "Glasgow", "fig_alpha_glasgow"),
    }
    written: dict[str, str] = {}
    for dest_rel in OUT_DIRS:
        dest = project_root() / dest_rel
        dest.mkdir(parents=True, exist_ok=True)
        both_pdf = dest / "fig_alpha_both_cities.pdf"
        both_png = dest / "fig_alpha_both_cities.png"
        fig.savefig(both_pdf, bbox_inches="tight")
        fig.savefig(both_png, dpi=300, bbox_inches="tight")
        written[str(both_pdf)] = "both"
        for _key, (frame, title, stem) in singles.items():
            one, ax = plt.subplots(figsize=(3.4, 2.55))
            _draw(ax, frame, f"{title} graph-fusion $\\alpha$", show_ylabel=True)
            ax.legend(frameon=False, loc="upper right", fontsize=7.5)
            one.tight_layout()
            one.savefig(dest / f"{stem}.pdf", bbox_inches="tight")
            one.savefig(dest / f"{stem}.png", dpi=300, bbox_inches="tight")
            plt.close(one)
            written[str(dest / f"{stem}.pdf")] = title
    plt.close(fig)
    return written


if __name__ == "__main__":
    paths = plot_alpha_trajectories()
    for path in paths:
        print(path)
