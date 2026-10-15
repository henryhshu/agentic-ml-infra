"""AutoPARK Architectural Diagrams Generator (Clean, Non-Overlapping, Publication-Quality).

Generates two complementary, non-overlapping architectural diagrams at distinct granularities:
1. Macro Architecture: Bird's-eye view of system planes, component boundaries, and decoupling bridge.
2. Split-Plane Runtime Mechanism: Detailed step-boundary execution, async incident triage, and hot actuation flow.
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
        ("Production CLI (Launches distributed Ray cluster jobs)", "run_pipeline.py --agentic --model vit_b", 0.570, 0.405),
    ]
    for title, code_txt, cx, cw in dev_cards:
        draw_card(ax, cx, top_y + 0.008, cw, 0.065, PALETTE["card_white"], PALETTE["slate_border"], border_width=0.8, zorder=2)
        ax.text(cx + 0.010, top_y + 0.055, title, fontsize=6.3, fontweight="bold", color=PALETTE["slate_header"], va="center", zorder=3)
        ax.text(cx + 0.010, top_y + 0.034, code_txt, fontsize=5.7, fontfamily="monospace", color=PALETTE["text_code"], va="center", zorder=3)

    # Connecting arrow down to cluster
    ax.annotate("", xy=(0.180, 0.770), xytext=(0.180, 0.875),
                arrowprops=dict(arrowstyle="-|>", color=PALETTE["exec_border"], lw=1.5))
    ax.text(0.188, 0.825, "Deploys Workload", fontsize=5.6, fontweight="bold", color=PALETTE["exec_header"], va="center")

    # -------------------------------------------------------------------------
    # 2. HIGHWAY GUTTER (y: 0.770 to 0.865): ASYNC TELEMETRY ARROW
    # -------------------------------------------------------------------------
    # Telemetry flow: Col A (x: 0.190, y: 0.755) -> Col C (x: 0.810, y: 0.755)
    ax.annotate("", xy=(0.810, 0.760), xytext=(0.190, 0.760),
                arrowprops=dict(arrowstyle="-|>", color=PALETTE["alert_border"], lw=1.6, ls="--",
                                connectionstyle="arc3,rad=-0.12"), zorder=2)
    ax.text(0.500, 0.815, "Async Incident Signal & Telemetry", fontsize=5.8, fontweight="bold",
            color=PALETTE["alert_header"], ha="center", va="center", zorder=3,
            bbox=dict(boxstyle="round,pad=0.25", facecolor="#FFF1F2", edgecolor="#FECDD3", lw=0.9))

    # -------------------------------------------------------------------------
    # 3. THREE OPERATIONAL COLUMNS (x: 0.015 to 0.985, y: 0.015 to 0.755)
    # -------------------------------------------------------------------------
    bot_y = 0.015
    bot_h = 0.740

    # === COLUMN A: IN-CLUSTER EXECUTION PLANE (Ray on AWS EKS) ===
    col_a_x = 0.015
    col_a_w = 0.355
    draw_card(ax, col_a_x, bot_y, col_a_w, bot_h, PALETTE["exec_bg"], PALETTE["exec_border"], border_width=1.2, zorder=1)
    
    # Col A Banner
    draw_card(ax, col_a_x + 0.008, bot_y + bot_h - 0.038, col_a_w - 0.016, 0.030, PALETTE["exec_banner"], PALETTE["exec_border"], border_width=0.8, zorder=2)
    ax.text(col_a_x + col_a_w/2, bot_y + bot_h - 0.023, "IN-CLUSTER EXECUTION PLANE",
            fontsize=7.2, fontweight="bold", color=PALETTE["exec_header"], ha="center", va="center", zorder=3)

    a_cards = [
        ("Distributed Training Workers",
         "PyTorch DDP on Ray Actors (Preemptible EC2)",
         "• High-throughput Ring AllReduce gradient sync\n• Micro-batch loop with `model.no_sync()` guard\n• Sustains >98% GPU hardware compute utilization",
         0.495, 0.190),
        ("Apache Arrow Zero-Copy IPC Store",
         "Shared-Memory Plasma Buffers",
         "• Multi-worker streaming partitions without serialization\n• Dynamic partition slice quarantine on data corruption",
         0.310, 0.165),
        ("Non-Blocking Telemetry Streamer",
         "Asynchronous Metrics & Heartbeat Queue",
         "• Real-time loss, grad norms, and memory stats\n• Overhead <0.1ms; workers never block on control plane",
         0.125, 0.165),
    ]
    for title, sub, bullets, cy, ch in a_cards:
        draw_card(ax, col_a_x + 0.008, cy, col_a_w - 0.016, ch, PALETTE["exec_card"], PALETTE["exec_border"], border_width=0.9, zorder=2)
        ax.text(col_a_x + 0.016, cy + ch - 0.022, title, fontsize=6.7, fontweight="bold", color=PALETTE["exec_header"], va="center", zorder=3)
        ax.text(col_a_x + 0.016, cy + ch - 0.046, sub, fontsize=5.6, fontfamily="monospace", color=PALETTE["text_blue"], va="center", zorder=3)
        ax.text(col_a_x + 0.016, cy + (ch - 0.046)/2 - 0.008, bullets, fontsize=5.3, color=PALETTE["text_muted"], va="center", zorder=3)

    # Col A Bottom Badge
    draw_card(ax, col_a_x + 0.008, bot_y + 0.012, col_a_w - 0.016, 0.080, "#EFF6FF", "#93C5FD", border_width=0.8, zorder=2)
    ax.text(col_a_x + col_a_w/2, bot_y + 0.060, "FAST CRITICAL PATH",
            fontsize=5.7, fontweight="bold", color="#1E40AF", ha="center", va="center", zorder=3)
    ax.text(col_a_x + col_a_w/2, bot_y + 0.034, "Workers execute tensor loops without LLM interference.",
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
    ax.text(col_b_x + col_b_w/2, 0.665, "SHARED-RAM", fontsize=7.2, fontweight="bold", color=PALETTE["bridge_header"], ha="center", va="center", zorder=3)
    ax.text(col_b_x + col_b_w/2, 0.638, "PipelineConfig", fontsize=6.8, fontfamily="monospace", fontweight="bold", color="#047857", ha="center", va="center", zorder=3)
    ax.text(col_b_x + col_b_w/2, 0.578, "State Vector $\\mathbf{\\Theta}_{\\text{runtime}}$:\n⟨B, $\\eta$, accum, clip⟩", fontsize=5.8, color=PALETTE["text_dark"], ha="center", va="center", zorder=3)
    ax.text(col_b_x + col_b_w/2, 0.485, "• Atomic in-place updates\n• Fast worker sync <0.1ms", fontsize=5.3, color=PALETTE["text_muted"], ha="center", va="center", zorder=3)

    # Card 2: Reversion Store
    draw_card(ax, col_b_x + 0.008, 0.125, col_b_w - 0.016, 0.280, "#ECFDF5", "#A7F3D0", border_width=0.9, zorder=2)
    ax.text(col_b_x + col_b_w/2, 0.375, "Reversion Store", fontsize=6.5, fontweight="bold", color=PALETTE["bridge_header"], ha="center", va="center", zorder=3)
    ax.text(col_b_x + col_b_w/2, 0.347, "`reversion_manifest`", fontsize=5.5, fontfamily="monospace", color="#047857", ha="center", va="center", zorder=3)
    ax.text(col_b_x + col_b_w/2, 0.270, "• Snapshots state $\\mathbf{\\Theta}_t$\n• Rollback on stall\n• Full audit trail logged", fontsize=5.3, color=PALETTE["text_muted"], ha="center", va="center", zorder=3)
    ax.text(col_b_x + col_b_w/2, 0.165, "Zero Data Loss", fontsize=5.2, fontweight="bold", color="#059669", ha="center", va="center", zorder=3)

    # Col B Bottom Badge
    draw_card(ax, col_b_x + 0.008, bot_y + 0.012, col_b_w - 0.016, 0.080, "#D1FAE5", "#6EE7B7", border_width=0.8, zorder=2)
    ax.text(col_b_x + col_b_w/2, bot_y + 0.060, "IPC BRIDGE", fontsize=5.7, fontweight="bold", color="#065F46", ha="center", va="center", zorder=3)
    ax.text(col_b_x + col_b_w/2, bot_y + 0.034, "Removes Cold Reboots", fontsize=5.1, color="#047857", ha="center", va="center", zorder=3)

    # Gutter 1 (x: 0.370 to 0.395): Worker query to Shared-RAM
    ax.annotate("", xy=(col_b_x, 0.540), xytext=(col_a_x + col_a_w, 0.540),
                arrowprops=dict(arrowstyle="<->", color=PALETTE["exec_header"], lw=1.5, ls="--"))

    # === COLUMN C: OUT-OF-BAND AGENT CONTROL PLANE ===
    col_c_x = 0.645
    col_c_w = 0.340
    draw_card(ax, col_c_x, bot_y, col_c_w, bot_h, PALETTE["cog_bg"], PALETTE["cog_border"], border_width=1.2, zorder=1)
    
    # Col C Banner
    draw_card(ax, col_c_x + 0.008, bot_y + bot_h - 0.038, col_c_w - 0.016, 0.030, PALETTE["cog_banner"], PALETTE["cog_border"], border_width=0.8, zorder=2)
    ax.text(col_c_x + col_c_w/2, bot_y + bot_h - 0.023, "AGENT CONTROL PLANE (OUT-OF-BAND)",
            fontsize=7.2, fontweight="bold", color=PALETTE["cog_header"], ha="center", va="center", zorder=3)

    c_cards = [
        ("1. Local Invariant Evaluators",
         "• Checks for NaN loss, OOM spike rates, and 2-min node preemption\n• Wakes LLM supervisor strictly upon anomaly",
         PALETTE["alert_bg"], PALETTE["alert_border"], 0.565, 0.125),
        ("2. Multi-Agent Supervisory Fleet",
         "• Supervisor coordinates TrainingWatch, ClusterHealth, & DataQuality\n• Queries Claude Haiku via HTTPS upon anomaly",
         PALETTE["cog_card"], PALETTE["cog_border"], 0.415, 0.125),
        ("3. Deterministic Safety Gate",
         "• Typed schemas strictly blocks raw bash/shell strings\n• Bounded knob guardrails (1<=B<=1024) & finite budget (K<=4)",
         PALETTE["safety_bg"], PALETTE["safety_border"], 0.265, 0.125),
        ("4. Modular Actuation Plugins",
         "• In-Memory Tuner, Preemption Migrator, Quarantine, AST Patcher",
         PALETTE["bridge_bg"], PALETTE["bridge_border"], 0.135, 0.105),
    ]
    for title, bullets, bg, bd, cy, ch in c_cards:
        draw_card(ax, col_c_x + 0.008, cy, col_c_w - 0.016, ch, bg, bd, border_width=0.9, zorder=2)
        ax.text(col_c_x + 0.016, cy + ch - 0.018, title, fontsize=6.6, fontweight="bold",
                color=PALETTE["slate_header"], va="top", zorder=3)
        ax.text(col_c_x + 0.016, cy + ch - 0.046, bullets, fontsize=5.3,
                color=PALETTE["text_dark"], linespacing=1.35, va="top", zorder=3)

    # Col C Bottom Badge
    draw_card(ax, col_c_x + 0.008, bot_y + 0.012, col_c_w - 0.016, 0.080, "#F5F3FF", "#DDD6FE", border_width=0.8, zorder=2)
    ax.text(col_c_x + col_c_w/2, bot_y + 0.060, "ASYNCHRONOUS AGENT PATH: 10 TO 30 SECONDS",
            fontsize=5.7, fontweight="bold", color="#6D28D9", ha="center", va="center", zorder=3)
    ax.text(col_c_x + col_c_w/2, bot_y + 0.034, "Runs strictly out-of-band; zero impact on worker training throughput.",
            fontsize=5.1, color="#7C3AED", ha="center", va="center", zorder=3)

    # Gutter 2 (x: 0.580 to 0.645): Actuation Plugin -> Shared-RAM PipelineConfig
    ax.annotate("", xy=(col_b_x + col_b_w, 0.495), xytext=(col_c_x, 0.188),
                arrowprops=dict(arrowstyle="-|>", color="#059669", lw=1.6,
                                connectionstyle="arc3,rad=-0.16"), zorder=2)
    ax.text(0.6125, 0.335, "Hot In-Memory\nMutation (1.1s)", fontsize=5.3, fontweight="bold",
            color="#047857", ha="center", va="center", zorder=3,
            bbox=dict(boxstyle="round,pad=0.20", facecolor="#ECFDF5", edgecolor="#A7F3D0", lw=0.9))

    fig.savefig(output_pdf, bbox_inches="tight", dpi=300)
    fig.savefig(output_png, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"Refined Macro-Architecture saved to {output_pdf} and {output_png}")


def plot_fig_split_plane(
    output_pdf: Path | str = "paper/figures/fig_split_plane.pdf",
    output_png: Path | str = "benchmarks/figures/fig_split_plane.png",
) -> None:
    """Diagram 2: Split-Plane Runtime Mechanism (Step-Boundary Actuation Flow)."""
    fig, ax = plt.subplots(figsize=(11.0, 4.6), dpi=300)
    ax.set_xlim(0, 1.0)
    ax.set_ylim(0, 1.0)
    ax.axis("off")

    # Outer container
    draw_card(ax, 0.015, 0.130, 0.970, 0.855, "#F8FAFC", "#64748B", border_width=1.2, zorder=1)
    
    # Outer Title Banner
    draw_card(ax, 0.025, 0.940, 0.950, 0.036, "#E2E8F0", "#94A3B8", border_width=0.8, zorder=2)
    ax.text(0.500, 0.958, "SPLIT-PLANE CLOSED-LOOP RUNTIME MECHANISM: STEP-BOUNDARY ACTUATION FLOW",
            fontsize=7.3, fontweight="bold", color="#0F172A", ha="center", va="center", zorder=3)

    phases = [
        ("PHASE 1: LINE-RATE TRAINING",
         "DDP Step (s) [Microseconds]",
         PALETTE["exec_bg"], PALETTE["exec_border"],
         [
             ("Micro-Batch Loop",
              "with model.no_sync():\n    loss = model(inputs)\n    loss.backward()",
              "Accumulates local gradients;\nsuppresses redundant NCCL AllReduce"),
             ("Step Boundary (s % k == 0)",
              "loss.backward()\noptimizer.step()\noptimizer.zero_grad()",
              "Synchronized Ring AllReduce;\ngradient parity mathematically preserved"),
         ],
         "Throughput: >98% GPU Util",
         0.025, 0.228),

        ("PHASE 2: INCIDENT DETECTION",
         "Sub-Second Trigger (T_det = 1.0s)",
         PALETTE["alert_bg"], PALETTE["alert_border"],
         [
             ("Local Invariants Evaluator",
              "dMemory/dt > tau (OOM Spike)\nLoss == NaN / Inf\nNode Preemption (2-min notice)",
              "Lightweight daemon rule checks;\nwakes LLM only when threshold breaches"),
             ("Non-Blocking Offload",
              "Enqueue incident envelope\nCheckpoint actor state locally",
              "Decoupled from training loop;\nworkers never block waiting on LLM"),
         ],
         "Zero Critical-Path Stalls",
         0.269, 0.228),

        ("PHASE 3: COGNITIVE TRIAGE",
         "Out-of-Band Reasoning (~12s)",
         PALETTE["cog_bg"], PALETTE["cog_border"],
         [
             ("Multi-Agent Diagnosis",
              "Supervisor + Expert Fleet\nClaude Haiku 4.5 via anthropic[aws]",
              "Parses traces & cluster topology;\nsynthesizes structured operational plan"),
             ("Deterministic Safety Gate",
              "Validate Pydantic Schema\nEnforce Bounds (1 <= B <= 1024)\nCap Budget (K <= 4 attempts)",
              "Non-negotiable policy filter;\nblocks hallucinated or unsafe actions"),
         ],
         "Safety: Non-Negotiable Gate",
         0.513, 0.228),

        ("PHASE 4: HOT ACTUATION",
         "Zero-Restart Resumption (s+1)",
         PALETTE["bridge_bg"], PALETTE["bridge_border"],
         [
             ("Shared-RAM Knob Mutation",
              "Plugin writes directly to RAM:\nPipelineConfig.mutate(B/2, lr=0.01)",
              "Physical actuation T_act = 1.1s;\n0 container reboots, 0 CUDA init waste"),
             ("Step-Boundary Sync (<0.1ms)",
              "Worker queries at step s+1:\nif config.has_mutation():\n    apply_runtime_knobs()",
              "Pulls new knobs from shared RAM;\nseamlessly resumes line-rate training"),
         ],
         "Recovery: 0 Pod Reboots",
         0.757, 0.228),
    ]

    for p_title, p_sub, p_bg, p_bd, p_cards, p_badge, px, pw in phases:
        py = 0.145
        ph = 0.780
        # Phase card
        draw_card(ax, px, py, pw, ph, p_bg, p_bd, border_width=1.1, zorder=2)
        
        # Phase header
        draw_card(ax, px + 0.006, py + ph - 0.058, pw - 0.012, 0.050, PALETTE["card_white"], p_bd, border_width=0.8, zorder=3)
        ax.text(px + pw/2, py + ph - 0.024, p_title, fontsize=6.0, fontweight="bold", color=p_bd, ha="center", va="center", zorder=4)
        ax.text(px + pw/2, py + ph - 0.044, p_sub, fontsize=5.1, fontfamily="monospace", color=PALETTE["text_muted"], ha="center", va="center", zorder=4)

        # 2 Internal Cards
        for c_idx, (c_name, c_code, c_note) in enumerate(p_cards):
            cy = py + 0.120 + (1 - c_idx) * 0.290
            ch = 0.270
            draw_card(ax, px + 0.008, cy, pw - 0.016, ch, PALETTE["card_white"], p_bd, border_width=0.7, zorder=3)
            ax.text(px + 0.014, cy + ch - 0.022, c_name, fontsize=6.3, fontweight="bold", color=PALETTE["slate_header"], va="center", zorder=4)
            ax.text(px + 0.014, cy + ch - 0.075, c_code, fontsize=5.3, fontfamily="monospace", color="#1E293B", va="center", zorder=4)
            ax.text(px + 0.014, cy + 0.038, c_note, fontsize=5.1, color=PALETTE["text_muted"], va="center", zorder=4)

        # Bottom badge
        draw_card(ax, px + 0.008, py + 0.015, pw - 0.016, 0.080, PALETTE["card_white"], p_bd, border_width=0.8, zorder=3)
        ax.text(px + pw/2, py + 0.055, p_badge, fontsize=5.7, fontweight="bold", color=p_bd, ha="center", va="center", zorder=4)

        # Arrow to next phase
        if px < 0.700:
            ax.annotate("", xy=(px + pw + 0.011, py + ph/2), xytext=(px + pw - 0.002, py + ph/2),
                        arrowprops=dict(arrowstyle="-|>", color=p_bd, lw=1.6))

    # -------------------------------------------------------------------------
    # BOTTOM: SUMMARY COMPARISON BANNER (Full Width)
    # -------------------------------------------------------------------------
    bar_y = 0.015
    bar_h = 0.098
    draw_card(ax, 0.015, bar_y, 0.970, bar_h, "#0F172A", "#334155", border_width=1.0)
    
    # Left AutoPARK
    draw_card(ax, 0.025, bar_y + 0.010, 0.465, bar_h - 0.020, "#1E293B", "#34D399", border_width=0.8, zorder=2)
    ax.text(0.035, bar_y + bar_h - 0.032, "AUTOPARK IN-MEMORY HEALING (HOT SYNC)", fontsize=6.3, fontweight="bold", color="#34D399", va="center", zorder=3)
    ax.text(0.035, bar_y + 0.030, "Shared-RAM mutation -> Step sync -> MTTR 14-31s -> 0 pod reboots -> 0 GPU idle waste",
            fontsize=5.2, color="#F8FAFC", va="center", zorder=3)

    # Right Classical Baseline
    draw_card(ax, 0.510, bar_y + 0.010, 0.465, bar_h - 0.020, "#1E293B", "#F87171", border_width=0.8, zorder=2)
    ax.text(0.520, bar_y + bar_h - 0.032, "CONVENTIONAL SRE RESTARTS (COLD SYNC)", fontsize=6.3, fontweight="bold", color="#F87171", va="center", zorder=3)
    ax.text(0.520, bar_y + 0.030, "Gang crash -> 15-45 min triage -> Cold pod init -> Loss of CUDA context -> Re-crashes on same config",
            fontsize=5.2, color="#F8FAFC", va="center", zorder=3)

    fig.savefig(output_pdf, bbox_inches="tight", dpi=300)
    fig.savefig(output_png, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"Refined Split-Plane Mechanism saved to {output_pdf} and {output_png}")


if __name__ == "__main__":
    plot_fig_arch_overview()
    plot_fig_split_plane()
