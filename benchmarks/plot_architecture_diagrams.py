"""AutoPARK Architectural Diagrams Generator (Clean, Non-Overlapping, Publication-Quality).

Generates the macro architecture diagram used in the paper. Every label describes behavior
implemented in this repository (multi-node DDP training with a rank-0 coordinator, synchronous
incident handling).
"""
from __future__ import annotations

from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

# Publication styling
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 8.0,
    "axes.labelsize": 8.5,
    "axes.titlesize": 9.0,
    "figure.titlesize": 10.0,
})

PALETTE = {
    # Neutral slate (Dev / User surface)
    "slate_bg": "#F8FAFC",
    "slate_border": "#94A3B8",
    "slate_banner": "#E2E8F0",
    "slate_header": "#0F172A",
    "card_white": "#FFFFFF",
    
    # In-Cluster Execution (Ice Blue)
    "exec_bg": "#F0F7FF",
    "exec_border": "#3B82F6",
    "exec_banner": "#DBEAFE",
    "exec_header": "#1E3A8A",
    "exec_card": "#FFFFFF",
    
    # Shared-RAM Bridge (Emerald / Teal)
    "bridge_bg": "#F0FDF4",
    "bridge_border": "#10B981",
    "bridge_banner": "#D1FAE5",
    "bridge_header": "#065F46",
    "bridge_card": "#FFFFFF",
    
    # Cognitive Plane (Soft Lavender)
    "cog_bg": "#FAF5FF",
    "cog_border": "#8B5CF6",
    "cog_banner": "#EDE9FE",
    "cog_header": "#581C87",
    "cog_card": "#FFFFFF",
    
    # Alerts / Triggers (Rose)
    "alert_bg": "#FFF1F2",
    "alert_border": "#F43F5E",
    "alert_banner": "#FFE4E6",
    "alert_header": "#9F1239",
    
    # Safety / Gate (Amber)
    "safety_bg": "#FFFBEB",
    "safety_border": "#F59E0B",
    "safety_banner": "#FEF3C7",
    "safety_header": "#78350F",

    # Text colors
    "text_dark": "#0F172A",
    "text_muted": "#475569",
    "text_blue": "#1D4ED8",
    "text_green": "#047857",
    "text_purple": "#6D28D9",
    "text_code": "#0369A1",
}


def draw_card(ax, x, y, w, h, bg_color, border_color, border_width=1.0, corner_radius=0.010, linestyle="-", zorder=2):
    """Draw a clean rounded card."""
    card = FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad=0.003,rounding_size={corner_radius*1.2}",
        facecolor=bg_color,
        edgecolor=border_color,
        linewidth=border_width,
        linestyle=linestyle,
        zorder=zorder
    )
    ax.add_patch(card)
    return card


def plot_fig_arch_overview(
    output_pdf: Path | str = "paper/figures/fig_arch_overview.pdf",
    output_png: Path | str = "benchmarks/figures/fig_arch_overview.png",
) -> None:
    """Diagram 1: Macro-Level System Architecture (Bird's-Eye View of System Planes)."""
    fig, ax = plt.subplots(figsize=(11.0, 5.0), dpi=300)
    ax.set_xlim(0, 1.0)
    ax.set_ylim(0, 1.0)
    ax.axis("off")

    # -------------------------------------------------------------------------
    # 1. TOP: DEVELOPER ADOPTION LAYER (Full Width)
    # -------------------------------------------------------------------------
    top_y = 0.875
    top_h = 0.115
    draw_card(ax, 0.015, top_y, 0.970, top_h, PALETTE["slate_bg"], PALETTE["slate_border"], border_width=1.1, zorder=1)
    
    # Banner title
    draw_card(ax, 0.025, top_y + top_h - 0.034, 0.950, 0.026, PALETTE["slate_banner"], PALETTE["slate_border"], border_width=0.7, zorder=2)
    ax.text(0.500, top_y + top_h - 0.021, "DEVELOPER ADOPTION LAYER",
            fontsize=7.2, fontweight="bold", color=PALETTE["slate_header"], ha="center", va="center", zorder=3)

    dev_cards = [
        ("Python SDK (Standard PyTorch wrapper)", "runner = AutoParkRunner.from_loaders(model, ld)", 0.025, 0.280),
        ("Task & Data Protocols", "@register_model  |  BaseDataModule", 0.315, 0.245),
        ("CLI (validate -> load -> train -> [tune] -> evaluate)", "run_pipeline.py --agentic --model tiny", 0.570, 0.405),
    ]
    for title, code_txt, cx, cw in dev_cards:
        draw_card(ax, cx, top_y + 0.008, cw, 0.065, PALETTE["card_white"], PALETTE["slate_border"], border_width=0.8, zorder=2)
        ax.text(cx + 0.010, top_y + 0.055, title, fontsize=6.3, fontweight="bold", color=PALETTE["slate_header"], va="center", zorder=3)
        ax.text(cx + 0.010, top_y + 0.034, code_txt, fontsize=5.7, fontfamily="monospace", color=PALETTE["text_code"], va="center", zorder=3)

    # Connecting arrow down to cluster
    ax.annotate("", xy=(0.180, 0.770), xytext=(0.180, 0.875),
                arrowprops=dict(arrowstyle="-|>", color=PALETTE["exec_border"], lw=1.5))
    ax.text(0.172, 0.825, "Kubernetes Indexed Job (EKS)", fontsize=5.6, fontweight="bold", color=PALETTE["exec_header"], va="center", ha="right")

    # -------------------------------------------------------------------------
    # 2. HIGHWAY GUTTER (y: 0.770 to 0.865): ASYNC TELEMETRY ARROW
    # -------------------------------------------------------------------------
    # Telemetry flow: Col A (x: 0.190, y: 0.755) -> Col C (x: 0.810, y: 0.755)
    ax.annotate("", xy=(0.810, 0.760), xytext=(0.190, 0.760),
                arrowprops=dict(arrowstyle="-|>", color=PALETTE["alert_border"], lw=1.6, ls="--",
                                connectionstyle="arc3,rad=-0.12"), zorder=2)
    ax.text(0.500, 0.815, "Stage exception & epoch metrics", fontsize=5.8, fontweight="bold",
            color=PALETTE["alert_header"], ha="center", va="center", zorder=3,
            bbox=dict(boxstyle="round,pad=0.25", facecolor="#FFF1F2", edgecolor="#FECDD3", lw=0.9))

    # -------------------------------------------------------------------------
    # 3. THREE OPERATIONAL COLUMNS (x: 0.015 to 0.985, y: 0.015 to 0.755)
    # -------------------------------------------------------------------------
    bot_y = 0.015
    bot_h = 0.740

    # === COLUMN A: EXECUTION PLANE (DDP, one rank per node) ===
    col_a_x = 0.015
    col_a_w = 0.355
    draw_card(ax, col_a_x, bot_y, col_a_w, bot_h, PALETTE["exec_bg"], PALETTE["exec_border"], border_width=1.2, zorder=1)
    
    # Col A Banner
    draw_card(ax, col_a_x + 0.008, bot_y + bot_h - 0.038, col_a_w - 0.016, 0.030, PALETTE["exec_banner"], PALETTE["exec_border"], border_width=0.8, zorder=2)
    ax.text(col_a_x + col_a_w/2, bot_y + bot_h - 0.023, "EXECUTION PLANE (DDP, ONE RANK PER NODE)",
            fontsize=7.2, fontweight="bold", color=PALETTE["exec_header"], ha="center", va="center", zorder=3)

    a_cards = [
        ("Rank 0 Pod (node 1): Coordinator",
         "pipeline runner + agents + TCPStore",
         "• Publishes each training attempt's config to all ranks\n• Trains as DDP rank 0; runs data stages and evaluation\n• Tracks worker heartbeats (every 0.5 s)",
         0.495, 0.190),
        ("Worker Ranks 1-3 (nodes 2-4)",
         "python -m pipeline.distributed",
         "• Join a fresh gloo process group per attempt\n• All-reduce gradients every step (global batch split)\n• Killed process restarted in pod; rejoins next attempt",
         0.310, 0.165),
        ("Stage Runner with Bounded Retries",
         "PipelineRunner._run_stage_with_retry",
         "• Catches stage exceptions (process keeps running)\n• Re-runs the failed stage with the current config",
         0.125, 0.165),
    ]
    for title, sub, bullets, cy, ch in a_cards:
        draw_card(ax, col_a_x + 0.008, cy, col_a_w - 0.016, ch, PALETTE["exec_card"], PALETTE["exec_border"], border_width=0.9, zorder=2)
        ax.text(col_a_x + 0.016, cy + ch - 0.022, title, fontsize=6.7, fontweight="bold", color=PALETTE["exec_header"], va="center", zorder=3)
        ax.text(col_a_x + 0.016, cy + ch - 0.046, sub, fontsize=5.6, fontfamily="monospace", color=PALETTE["text_blue"], va="center", zorder=3)
        ax.text(col_a_x + 0.016, cy + (ch - 0.046)/2 - 0.008, bullets, fontsize=5.3, color=PALETTE["text_muted"], va="center", zorder=3)

    # Col A Bottom Badge
    draw_card(ax, col_a_x + 0.008, bot_y + 0.012, col_a_w - 0.016, 0.080, "#EFF6FF", "#93C5FD", border_width=0.8, zorder=2)
    ax.text(col_a_x + col_a_w/2, bot_y + 0.060, "SYNCHRONOUS INCIDENT PATH",
            fontsize=5.7, fontweight="bold", color="#1E40AF", ha="center", va="center", zorder=3)
    ax.text(col_a_x + col_a_w/2, bot_y + 0.034, "All ranks idle while agents handle an incident.",
            fontsize=5.1, color="#3B82F6", ha="center", va="center", zorder=3)

    # === COLUMN B: SHARED-RAM COMMUNICATION BRIDGE ===
    col_b_x = 0.395
    col_b_w = 0.185
    draw_card(ax, col_b_x, bot_y, col_b_w, bot_h, PALETTE["bridge_bg"], PALETTE["bridge_border"], border_width=1.2, zorder=1)
    
    # Col B Banner
    draw_card(ax, col_b_x + 0.008, bot_y + bot_h - 0.038, col_b_w - 0.016, 0.030, PALETTE["bridge_banner"], PALETTE["bridge_border"], border_width=0.8, zorder=2)
    ax.text(col_b_x + col_b_w/2, bot_y + bot_h - 0.023, "AGENTIC INTERFACE",
            fontsize=7.2, fontweight="bold", color=PALETTE["bridge_header"], ha="center", va="center", zorder=3)

    # Card 1: PipelineConfig
    draw_card(ax, col_b_x + 0.008, 0.425, col_b_w - 0.016, 0.270, PALETTE["bridge_card"], PALETTE["bridge_border"], border_width=1.1, zorder=2)
    ax.text(col_b_x + col_b_w/2, 0.665, "RANK-0 STATE", fontsize=7.2, fontweight="bold", color=PALETTE["bridge_header"], ha="center", va="center", zorder=3)
    ax.text(col_b_x + col_b_w/2, 0.638, "PipelineConfig", fontsize=6.8, fontfamily="monospace", fontweight="bold", color="#047857", ha="center", va="center", zorder=3)
    ax.text(col_b_x + col_b_w/2, 0.578, "State Vector $\\mathbf{\\Theta}_{\\text{runtime}}$:\n⟨B, $\\eta$, accum, clip⟩", fontsize=5.8, color=PALETTE["text_dark"], ha="center", va="center", zorder=3)
    ax.text(col_b_x + col_b_w/2, 0.485, "• Mutated in place by Remediator\n• Sent to all ranks at next attempt", fontsize=5.3, color=PALETTE["text_muted"], ha="center", va="center", zorder=3)

    # Card 2: Reversion Store
    draw_card(ax, col_b_x + 0.008, 0.125, col_b_w - 0.016, 0.280, "#ECFDF5", "#A7F3D0", border_width=0.9, zorder=2)
    ax.text(col_b_x + col_b_w/2, 0.375, "Reversion Store", fontsize=6.5, fontweight="bold", color=PALETTE["bridge_header"], ha="center", va="center", zorder=3)
    ax.text(col_b_x + col_b_w/2, 0.347, "`reversion_manifest`", fontsize=5.5, fontfamily="monospace", color="#047857", ha="center", va="center", zorder=3)
    ax.text(col_b_x + col_b_w/2, 0.270, "• Records every config diff\n• One-command rollback re-run\n• Audit trail (action_log.json)", fontsize=5.3, color=PALETTE["text_muted"], ha="center", va="center", zorder=3)
    ax.text(col_b_x + col_b_w/2, 0.165, "Manual rollback", fontsize=5.2, fontweight="bold", color="#059669", ha="center", va="center", zorder=3)

    # Col B Bottom Badge
    draw_card(ax, col_b_x + 0.008, bot_y + 0.012, col_b_w - 0.016, 0.080, "#D1FAE5", "#6EE7B7", border_width=0.8, zorder=2)
    ax.text(col_b_x + col_b_w/2, bot_y + 0.060, "RANK-0 STATE", fontsize=5.7, fontweight="bold", color="#065F46", ha="center", va="center", zorder=3)
    ax.text(col_b_x + col_b_w/2, bot_y + 0.034, "No pod restart", fontsize=5.1, color="#047857", ha="center", va="center", zorder=3)

    # Gutter 1 (x: 0.370 to 0.395): Worker query to Shared-RAM
    ax.annotate("", xy=(col_b_x, 0.540), xytext=(col_a_x + col_a_w, 0.540),
                arrowprops=dict(arrowstyle="<->", color=PALETTE["exec_header"], lw=1.5, ls="--"))

    # === COLUMN C: OUT-OF-BAND AGENT CONTROL PLANE ===
    col_c_x = 0.645
    col_c_w = 0.340
    draw_card(ax, col_c_x, bot_y, col_c_w, bot_h, PALETTE["cog_bg"], PALETTE["cog_border"], border_width=1.2, zorder=1)
    
    # Col C Banner
    draw_card(ax, col_c_x + 0.008, bot_y + bot_h - 0.038, col_c_w - 0.016, 0.030, PALETTE["cog_banner"], PALETTE["cog_border"], border_width=0.8, zorder=2)
    ax.text(col_c_x + col_c_w/2, bot_y + bot_h - 0.023, "AGENT CONTROL PLANE",
            fontsize=7.2, fontweight="bold", color=PALETTE["cog_header"], ha="center", va="center", zorder=3)

    c_cards = [
        ("1. Local Detectors",
         "• Typed stage exceptions (NaN, OOM, grad norm)\n• Failed collective + heartbeat liveness (worker loss)\n• Per-epoch checks: accuracy floor, plateau, grad norm",
         PALETTE["alert_bg"], PALETTE["alert_border"], 0.555, 0.140),
        ("2. Multi-Agent Supervisory Fleet",
         "• Supervisor routes to TrainingWatch, ClusterHealth, DataQuality\n• Claude Haiku 4.5 per incident, plus one call per epoch",
         PALETTE["cog_card"], PALETTE["cog_border"], 0.415, 0.125),
        ("3. Action Allow-List",
         "• Claude's reply must name an approved action\n• Per-stage retry budget (max_retries = 3)",
         PALETTE["safety_bg"], PALETTE["safety_border"], 0.265, 0.125),
        ("4. Remediation Actions",
         "• LR x0.1, batch /2 (accum x2), enable clipping,\n  restart workers (wait for all heartbeats), retry stage",
         PALETTE["bridge_bg"], PALETTE["bridge_border"], 0.130, 0.115),
    ]
    for title, bullets, bg, bd, cy, ch in c_cards:
        draw_card(ax, col_c_x + 0.008, cy, col_c_w - 0.016, ch, bg, bd, border_width=0.9, zorder=2)
        ax.text(col_c_x + 0.016, cy + ch - 0.018, title, fontsize=6.6, fontweight="bold",
                color=PALETTE["slate_header"], va="top", zorder=3)
        ax.text(col_c_x + 0.016, cy + ch - 0.046, bullets, fontsize=5.3,
                color=PALETTE["text_dark"], linespacing=1.35, va="top", zorder=3)

    # Col C Bottom Badge
    draw_card(ax, col_c_x + 0.008, bot_y + 0.012, col_c_w - 0.016, 0.080, "#F5F3FF", "#DDD6FE", border_width=0.8, zorder=2)
    ax.text(col_c_x + col_c_w/2, bot_y + 0.060, "INCIDENT HANDLING IS MEASURED (TABLE 2)",
            fontsize=5.7, fontweight="bold", color="#6D28D9", ha="center", va="center", zorder=3)
    ax.text(col_c_x + col_c_w/2, bot_y + 0.034, "Claude calls run inline with the pipeline.",
            fontsize=5.1, color="#7C3AED", ha="center", va="center", zorder=3)

    # Gutter 2 (x: 0.580 to 0.645): Actuation Plugin -> Shared-RAM PipelineConfig
    ax.annotate("", xy=(col_b_x + col_b_w, 0.495), xytext=(col_c_x, 0.188),
                arrowprops=dict(arrowstyle="-|>", color="#059669", lw=1.6,
                                connectionstyle="arc3,rad=-0.16"), zorder=2)
    ax.text(0.6125, 0.335, "In-place config\nmutation", fontsize=5.3, fontweight="bold",
            color="#047857", ha="center", va="center", zorder=3,
            bbox=dict(boxstyle="round,pad=0.20", facecolor="#ECFDF5", edgecolor="#A7F3D0", lw=0.9))

    fig.savefig(output_pdf, bbox_inches="tight", dpi=300)
    fig.savefig(output_png, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"Refined Macro-Architecture saved to {output_pdf} and {output_png}")


if __name__ == "__main__":
    plot_fig_arch_overview()
