#!/usr/bin/env python3
"""Draw a high-resolution architecture figure for the local JSSP Decision Transformer."""

from __future__ import annotations

import os
import textwrap
from dataclasses import dataclass
from typing import Iterable

from PIL import Image, ImageDraw, ImageFont


OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "figures")
PNG_PATH = os.path.join(OUT_DIR, "jssp_decision_transformer_architecture.png")
SVG_PATH = os.path.join(OUT_DIR, "jssp_decision_transformer_architecture.svg")

W, H = 5600, 3400


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ]
    for path in candidates:
        if os.path.exists(path):
            return ImageFont.truetype(path, size=size)
    return ImageFont.load_default()


TITLE = font(72, True)
SUBTITLE = font(36)
HEAD = font(38, True)
BODY = font(29)
SMALL = font(25)
TINY = font(21)


@dataclass
class Box:
    key: str
    xy: tuple[int, int, int, int]
    title: str
    body: str
    fill: str
    outline: str
    title_fill: str = "#14213d"
    body_fill: str = "#1f2937"
    radius: int = 26


def wrap(text: str, width: int) -> list[str]:
    lines: list[str] = []
    for part in text.split("\n"):
        if not part:
            lines.append("")
        else:
            lines.extend(textwrap.wrap(part, width=width, break_long_words=False))
    return lines


def rounded(draw: ImageDraw.ImageDraw, xy: tuple[int, int, int, int], fill: str, outline: str, width: int = 5, radius: int = 26) -> None:
    draw.rounded_rectangle(xy, radius=radius, fill=fill, outline=outline, width=width)


def draw_box(draw: ImageDraw.ImageDraw, box: Box) -> None:
    x1, y1, x2, y2 = box.xy
    rounded(draw, box.xy, box.fill, box.outline, radius=box.radius)
    draw.text((x1 + 34, y1 + 26), box.title, font=HEAD, fill=box.title_fill)
    y = y1 + 86
    chars = max(22, int((x2 - x1 - 70) / 15))
    for line in wrap(box.body, chars):
        draw.text((x1 + 34, y), line, font=BODY, fill=box.body_fill)
        y += 40


def arrow(draw: ImageDraw.ImageDraw, start: tuple[int, int], end: tuple[int, int], color: str = "#334155", width: int = 8, label: str | None = None) -> None:
    draw.line([start, end], fill=color, width=width)
    sx, sy = start
    ex, ey = end
    import math

    ang = math.atan2(ey - sy, ex - sx)
    size = 24
    left = (ex - size * math.cos(ang - math.pi / 6), ey - size * math.sin(ang - math.pi / 6))
    right = (ex - size * math.cos(ang + math.pi / 6), ey - size * math.sin(ang + math.pi / 6))
    draw.polygon([end, left, right], fill=color)
    if label:
        mx, my = (sx + ex) / 2, (sy + ey) / 2
        tw = draw.textlength(label, font=SMALL)
        draw.rounded_rectangle((mx - tw / 2 - 16, my - 24, mx + tw / 2 + 16, my + 18), radius=12, fill="#ffffff", outline=color, width=2)
        draw.text((mx - tw / 2, my - 19), label, font=SMALL, fill=color)


def center_right(box: Box) -> tuple[int, int]:
    x1, y1, x2, y2 = box.xy
    return x2, (y1 + y2) // 2


def center_left(box: Box) -> tuple[int, int]:
    x1, y1, x2, y2 = box.xy
    return x1, (y1 + y2) // 2


def center_top(box: Box) -> tuple[int, int]:
    x1, y1, x2, y2 = box.xy
    return (x1 + x2) // 2, y1


def center_bottom(box: Box) -> tuple[int, int]:
    x1, y1, x2, y2 = box.xy
    return (x1 + x2) // 2, y2


def svg_escape(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def save_svg(boxes: Iterable[Box], edges: list[tuple[str, str, str, str]]) -> None:
    by_key = {b.key: b for b in boxes}

    def point(box: Box, side: str) -> tuple[int, int]:
        if side == "right":
            return center_right(box)
        if side == "left":
            return center_left(box)
        if side == "top":
            return center_top(box)
        return center_bottom(box)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">',
        '<rect width="100%" height="100%" fill="#f8fafc"/>',
        '<defs><marker id="arrow" markerWidth="16" markerHeight="16" refX="14" refY="8" orient="auto" markerUnits="strokeWidth"><path d="M0,0 L16,8 L0,16 z" fill="#334155"/></marker></defs>',
        '<text x="160" y="130" font-family="DejaVu Sans,Arial" font-size="72" font-weight="700" fill="#0f172a">JSSP Decision Transformer Architecture</text>',
        '<text x="160" y="190" font-family="DejaVu Sans,Arial" font-size="36" fill="#475569">Qwen2.5 causal backbone + LoRA, graph state encoder, RTG-conditioned job-level scheduling policy</text>',
    ]
    for src, dst, label, mode in edges:
        sb = by_key[src]
        db = by_key[dst]
        s_side, d_side = mode.split("-")
        sx, sy = point(sb, s_side)
        dx, dy = point(db, d_side)
        parts.append(f'<line x1="{sx}" y1="{sy}" x2="{dx}" y2="{dy}" stroke="#334155" stroke-width="8" marker-end="url(#arrow)"/>')
        if label:
            mx, my = (sx + dx) / 2, (sy + dy) / 2
            parts.append(f'<text x="{mx}" y="{my - 14}" font-family="DejaVu Sans,Arial" font-size="25" text-anchor="middle" fill="#334155">{svg_escape(label)}</text>')
    for b in boxes:
        x1, y1, x2, y2 = b.xy
        parts.append(f'<rect x="{x1}" y="{y1}" width="{x2-x1}" height="{y2-y1}" rx="{b.radius}" fill="{b.fill}" stroke="{b.outline}" stroke-width="5"/>')
        parts.append(f'<text x="{x1+34}" y="{y1+68}" font-family="DejaVu Sans,Arial" font-size="38" font-weight="700" fill="{b.title_fill}">{svg_escape(b.title)}</text>')
        y = y1 + 126
        chars = max(22, int((x2 - x1 - 70) / 15))
        for line in wrap(b.body, chars):
            parts.append(f'<text x="{x1+34}" y="{y}" font-family="DejaVu Sans,Arial" font-size="29" fill="{b.body_fill}">{svg_escape(line)}</text>')
            y += 40
    parts.append("</svg>")
    with open(SVG_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(parts))


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    image = Image.new("RGB", (W, H), "#f8fafc")
    draw = ImageDraw.Draw(image)

    draw.text((160, 70), "JSSP Decision Transformer Architecture", font=TITLE, fill="#0f172a")
    draw.text((160, 158), "Qwen2.5 causal backbone + LoRA, graph state encoder, RTG-conditioned job-level scheduling policy", font=SUBTITLE, fill="#475569")
    draw.line((160, 235, 5440, 235), fill="#cbd5e1", width=5)

    boxes = [
        Box("raw", (170, 340, 1010, 760), "Environment / Dataset Inputs", "states: dict or tensor [B,T,*]\nreturns_to_go: [B,T,1]\ntimesteps: [B,T]\nprev actions: [B,T] or [B,T-1]\naction_mask: [B,T,n_jobs]\nmask: [B,T] for padded steps", "#e0f2fe", "#0284c7"),
        Box("collate", (170, 900, 1010, 1240), "Trajectory Collation", "Pads variable-length trajectories.\nActions use -100 for ignored labels.\nRTG padding uses -1e4.\nState dict components are padded to common feature widths.", "#f0f9ff", "#38bdf8"),
        Box("graph_input", (1210, 300, 2060, 760), "Graph State Tensorization", "Required keys for graph encoder:\njob_next_op, machine_sequence,\nremaining_processing_times,\nnormalized_machine_availability,\noverall_info.\nOptional: machine_busy_mask,\nscheduled_machine_adj,\nproblem_n_jobs / machines.", "#dcfce7", "#16a34a"),
        Box("node_feat", (2260, 300, 3160, 760), "Operation Node Features", "Each operation node uses 18 features:\nprocessing time log/linear,\njob remaining load, machine/job/op ids,\ndone / next / future flags,\nremaining-op ratio, machine busy/available,\nnode mask, queue position, LB feature,\nplus 3 global overall features.", "#ecfccb", "#65a30d"),
        Box("gnn", (3360, 300, 4310, 760), "GraphJSSPStateEncoder", "node_in Linear -> ReLU.\n3 GNN layers over three edge types:\n1 precedence edges within each job,\n2 disjunctive machine-competition edges,\n3 scheduled machine-order edges.\nEach edge type has Linear message + vector gate.\nGRUCell + LayerNorm updates node states.", "#fef9c3", "#ca8a04"),
        Box("pool", (4520, 300, 5420, 760), "Job Attention Pooling", "Mean-pool operation nodes per job.\nConcatenate each job token with global pool.\njob_token_proj -> job tokens [B,T,J,Denc].\nLearned query MultiheadAttention over jobs.\nOutput state_vec [B,T,Denc] + job_valid_mask aux.", "#fff7ed", "#ea580c"),
        Box("proj", (1210, 900, 2060, 1240), "State Projection", "state_vec [B,T,Denc]\n-> state_proj Linear to Qwen hidden size D\n-> LayerNorm\n-> Dropout(0.05)\nGrad hook sanitizes NaN/Inf and clips input gradients.", "#ede9fe", "#7c3aed"),
        Box("rtg", (1210, 1370, 2060, 1640), "RTG Embedding", "returns_to_go [B,T,1]\nClamp to [-1e4,1e4]\nrtg_proj Linear(1,D)\nDropout + numeric clamp.", "#dbeafe", "#2563eb"),
        Box("time", (1210, 1760, 2060, 2030), "Time Embedding", "timesteps [B,T]\nClamp to [0,max_ep_len-1]\ntimestep_embedding -> [B,T,D].", "#dbeafe", "#2563eb"),
        Box("act", (1210, 2150, 2060, 2500), "Previous Action Embedding", "Job-level action ids.\nValid jobs: 0..n_jobs-1.\nPadding/dummy id: n_jobs.\naction_proj Embedding(n_jobs+1,D).\nIf [B,T-1], prepend dummy so token at t is A_{t-1}.", "#dbeafe", "#2563eb"),
        Box("interleave", (2360, 1320, 3370, 2030), "Decision Transformer Tokenization", "For every time step t, construct:\n[RTG_t, State_t, Time_t, A_{t-1}].\nFlatten to [B,4T,D].\nAdd type embeddings with ids [0,1,3,2].\ninput_fuse_norm then scale by 0.1.\nPadding mask [B,T] expands to [B,4T].", "#fae8ff", "#c026d3"),
        Box("qwen", (3660, 1200, 4660, 2120), "Qwen2.5 Causal Transformer Backbone", "inputs_embeds, not text tokens.\nCausal LM applies autoregressive attention.\nLoRA trains q/k/v/o projections and MLP gate/up/down projections.\nOptional 4-bit loading and gradient checkpointing.\nReturns hidden states [B,4T,D].", "#fee2e2", "#dc2626"),
        Box("extract", (4880, 1320, 5420, 2030), "Action Positions", "Take hidden states at positions 4t+3,\nthe A_{t-1} token positions.\nThese positions can attend to RTG_t, S_t, Time_t and previous history.\nOutput action_hidden [B,T,D].", "#ffe4e6", "#e11d48"),
        Box("heads", (3480, 2350, 4440, 2940), "Prediction Heads", "Action head: Linear(D,n_jobs) -> logits [B,T,n_jobs].\nOptional candidate-aware head:\ncontext_proj(action_hidden) + job_proj(job_tokens)\n-> LayerNorm -> tanh -> Linear(D,1), masked by job_valid_mask.\nValue head: LayerNorm -> Linear -> GELU -> Linear -> value_preds [B,T].", "#f1f5f9", "#475569"),
        Box("maskout", (4660, 2350, 5420, 2940), "Output / Decision", "Apply action_mask or env.get_action_mask().\nInvalid jobs get -1e9 logits.\nTraining losses use masked CE, label smoothing,\noptional gap-weighting, unlikelihood, and value auxiliary loss.\nInference selects argmax/softmax over last timestep logits.\nPredicted action: job id A_t.", "#e2e8f0", "#334155"),
        Box("fallback", (170, 1380, 1010, 1830), "Non-Graph State Fallback", "If graph encoder is disabled, state tensor [B,T,state_dim]\ncan be projected directly by state_proj.\nA simpler JSSPStateEncoder also exists for compact state dictionaries.", "#f8fafc", "#94a3b8"),
    ]

    edges = [
        ("raw", "collate", "batch padding", "bottom-top"),
        ("raw", "graph_input", "state dict", "right-left"),
        ("graph_input", "node_feat", "pad to max jobs/machines", "right-left"),
        ("node_feat", "gnn", "operation graph", "right-left"),
        ("gnn", "pool", "node states", "right-left"),
        ("pool", "proj", "state_vec", "bottom-top"),
        ("raw", "rtg", "RTG", "right-left"),
        ("raw", "time", "t", "right-left"),
        ("raw", "act", "A history", "right-left"),
        ("proj", "interleave", "State_t", "right-left"),
        ("rtg", "interleave", "RTG_t", "right-left"),
        ("time", "interleave", "Time_t", "right-left"),
        ("act", "interleave", "A_{t-1}", "right-left"),
        ("interleave", "qwen", "[B,4T,D]", "right-left"),
        ("qwen", "extract", "hidden states", "right-left"),
        ("extract", "heads", "action_hidden", "bottom-top"),
        ("pool", "heads", "job_tokens aux", "bottom-top"),
        ("heads", "maskout", "logits/value", "right-left"),
        ("fallback", "proj", "direct state path", "right-left"),
    ]

    # Soft background bands.
    draw.rounded_rectangle((120, 260, 5480, 820), radius=34, fill="#ffffff", outline="#e2e8f0", width=3)
    draw.rounded_rectangle((120, 850, 5480, 3060), radius=34, fill="#ffffff", outline="#e2e8f0", width=3)
    draw.text((160, 280), "State graph encoder path", font=SUBTITLE, fill="#166534")
    draw.text((160, 830), "Decision Transformer path", font=SUBTITLE, fill="#581c87")

    by_key = {b.key: b for b in boxes}
    for src, dst, label, mode in edges:
        s_side, d_side = mode.split("-")
        start = {"right": center_right, "left": center_left, "top": center_top, "bottom": center_bottom}[s_side](by_key[src])
        end = {"right": center_right, "left": center_left, "top": center_top, "bottom": center_bottom}[d_side](by_key[dst])
        arrow(draw, start, end, label=label)

    for b in boxes:
        draw_box(draw, b)

    # Legend and shape summary.
    legend_y = 3120
    draw.text((170, legend_y), "I/O summary:", font=HEAD, fill="#0f172a")
    summary = (
        "Input sequence length T -> internal token length 4T. "
        "Main output logits are [batch, T, n_jobs]; value auxiliary output is [batch, T]. "
        "At rollout, only the last timestep logits are used to choose the next schedulable job."
    )
    x = 500
    for line in wrap(summary, 150):
        draw.text((x, legend_y + 6), line, font=BODY, fill="#334155")
        legend_y += 38
    draw.text((170, 3300), f"Generated from local files: decision_transformer.py, dt_train.py, dt_data_processing.py. Saved as {os.path.basename(PNG_PATH)} and SVG.", font=TINY, fill="#64748b")

    image.save(PNG_PATH, dpi=(300, 300), quality=95)
    save_svg(boxes, edges)
    print(PNG_PATH)
    print(SVG_PATH)


if __name__ == "__main__":
    main()
