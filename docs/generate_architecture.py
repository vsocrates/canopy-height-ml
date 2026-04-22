"""
Generate docs/figures/architecture.png — visual architecture diagram for the README.

Layout:
  - Orchestrator (top, orange) with role description
  - Three sub-agents (middle, teal) with descriptions — no tool names
  - Foundational stack bar (bottom, slate) with logo-tagged Python tools
  - Thin grey dispatch arrows orchestrator -> agents
  - Big curved orange arrows on both sides to emphasize the iterative replan loop
"""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

OUT_PNG = Path(__file__).parent / "figures" / "architecture.png"
OUT_SVG = Path(__file__).parent / "figures" / "architecture.svg"

# Palette — dark, modern, accessible
BG = "#0B1220"
ORCH = "#FB923C"
AGENT = "#34D399"
FOUND = "#334155"
ARROW = "#FB923C"
DISPATCH = "#64748B"
TEXT_DARK = "#0B1220"
TEXT_LIGHT = "#F8FAFC"
TEXT_MUTED = "#94A3B8"


def main() -> None:
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 10)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.patch.set_facecolor(BG)

    # Title
    ax.text(
        7, 9.5, "Agentic Canopy Height Pipeline",
        ha="center", va="center",
        fontsize=22, fontweight="bold", color=TEXT_LIGHT,
    )

    # --- Orchestrator ---
    orch_x, orch_y, orch_w, orch_h = 3.0, 7.15, 8.0, 1.75
    orch = FancyBboxPatch(
        (orch_x, orch_y), orch_w, orch_h,
        boxstyle="round,pad=0.08,rounding_size=0.2",
        facecolor=ORCH, edgecolor="none",
    )
    ax.add_patch(orch)
    ax.text(
        orch_x + orch_w / 2, orch_y + orch_h - 0.4, "ORCHESTRATOR",
        ha="center", va="center",
        fontsize=22, fontweight="bold", color=TEXT_DARK,
    )
    ax.text(
        orch_x + orch_w / 2, orch_y + orch_h - 1.0,
        "Drives the pipeline end-to-end. Inspects typed agent decisions,",
        ha="center", va="center", fontsize=13, color=TEXT_DARK,
    )
    ax.text(
        orch_x + orch_w / 2, orch_y + orch_h - 1.38,
        "replans on failure, aborts when unresolvable.",
        ha="center", va="center", fontsize=13, color=TEXT_DARK,
    )

    # --- Sub-agents ---
    agent_y, agent_h, agent_w = 4.15, 1.95, 3.85
    agents = [
        (0.3, "INGESTOR", [
            "Queries GEDI L2A and",
            "Sentinel-2 scene availability.",
            "Filters shots by quality flag,",
            "sensitivity, slope.",
        ]),
        (5.08, "TRANSFORMER", [
            "Samples S2 bands at shots.",
            "Estimates spatial",
            "autocorrelation, assigns",
            "block-based CV folds.",
        ]),
        (9.85, "QA", [
            "Validates feature",
            "completeness, EVI/NDVI",
            "ranges, fold balance before",
            "model training.",
        ]),
    ]

    for x, name, desc in agents:
        box = FancyBboxPatch(
            (x, agent_y), agent_w, agent_h,
            boxstyle="round,pad=0.07,rounding_size=0.15",
            facecolor=AGENT, edgecolor="none",
        )
        ax.add_patch(box)
        ax.text(
            x + agent_w / 2, agent_y + agent_h - 0.37, name,
            ha="center", va="center",
            fontsize=17, fontweight="bold", color=TEXT_DARK,
        )
        for i, line in enumerate(desc):
            ax.text(
                x + agent_w / 2, agent_y + agent_h - 0.85 - i * 0.33, line,
                ha="center", va="center",
                fontsize=12, color=TEXT_DARK,
            )

    # --- Dispatch arrows (orchestrator -> each agent) ---
    for x, _, _ in agents:
        arrow = FancyArrowPatch(
            (x + agent_w / 2, orch_y - 0.02),
            (x + agent_w / 2, agent_y + agent_h + 0.02),
            arrowstyle="-|>", mutation_scale=18,
            linewidth=1.8, color=DISPATCH,
        )
        ax.add_patch(arrow)

    # --- Big curved replan-loop arrows (right side + left side) ---
    right_arrow = FancyArrowPatch(
        (13.55, 5.0), (13.55, 8.1),
        connectionstyle="arc3,rad=-0.45",
        arrowstyle="-|>", mutation_scale=30,
        linewidth=4.5, color=ARROW,
    )
    ax.add_patch(right_arrow)
    ax.text(
        13.7, 6.55, "replan /\nabort",
        ha="center", va="center",
        fontsize=13, fontweight="bold", color=ARROW,
    )

    left_arrow = FancyArrowPatch(
        (0.45, 5.0), (0.45, 8.1),
        connectionstyle="arc3,rad=0.45",
        arrowstyle="-|>", mutation_scale=30,
        linewidth=4.5, color=ARROW,
    )
    ax.add_patch(left_arrow)
    ax.text(
        0.3, 6.55, "typed\ndecisions",
        ha="center", va="center",
        fontsize=13, fontweight="bold", color=ARROW,
    )

    # --- Foundational stack bar ---
    found_x, found_y, found_w, found_h = 0.3, 1.1, 13.4, 1.85
    found = FancyBboxPatch(
        (found_x, found_y), found_w, found_h,
        boxstyle="round,pad=0.08,rounding_size=0.2",
        facecolor=FOUND, edgecolor="none",
    )
    ax.add_patch(found)
    ax.text(
        found_x + found_w / 2, found_y + found_h - 0.32,
        "FOUNDATIONAL STACK",
        ha="center", va="center",
        fontsize=12, fontweight="bold",
        color=TEXT_LIGHT, alpha=0.75,
    )

    tools = [
        ("pydantic-ai", "#E92063"),
        ("Earth Engine", "#4285F4"),
        ("SQLAlchemy", "#D71F00"),
        ("Logfire", "#FF6B35"),
    ]
    slot_w = found_w / len(tools)
    for i, (name, color) in enumerate(tools):
        cx = found_x + slot_w * (i + 0.5)
        cy = found_y + 0.75
        circle = plt.Circle((cx - 0.95, cy), 0.22, color=color, zorder=5)
        ax.add_patch(circle)
        ax.text(
            cx - 0.6, cy, name,
            ha="left", va="center",
            fontsize=14, fontweight="bold", color=TEXT_LIGHT,
        )

    # Foundation caption
    ax.text(
        found_x + found_w / 2, found_y - 0.35,
        "Shared by every agent — typed deps, GEE tools, ORM persistence, end-to-end tracing.",
        ha="center", va="center",
        fontsize=11, color=TEXT_MUTED, style="italic",
    )

    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUT_PNG, dpi=200, facecolor=BG, bbox_inches="tight")
    plt.savefig(OUT_SVG, facecolor=BG, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUT_PNG}")
    print(f"wrote {OUT_SVG}")


if __name__ == "__main__":
    main()
