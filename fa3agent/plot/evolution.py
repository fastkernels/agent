"""AVO-style evolution plot, regenerated each step.

Replicates the AVO evolution-trajectory figure (see ``.research/avo.md`` §6) for our
FA3-distillation run:

  * X axis = committed kernel version / step index (the committed lineage).
  * Y axis = speedup RATIO (candidate latency vs baseline latency); 1.0 = parity.
  * Solid line  = running-best ``min_ratio``    (worst-case speedup; our fitness floor).
  * Solid line  = running-best ``geomean_ratio`` (the metric we optimize on).
  * Circle markers on steps that set a NEW running best (geomean).
  * Faint dashed lines = per-config ratios (one per Config).
  * Horizontal dashed reference line at y=1.0 = baseline parity.

The input is ``candidates.jsonl`` -- one JSON object per line, each an ``asdict``-serialized
``common.StepRecord``.  Incorrect / build-failed candidates have ratio 0 and are plotted at 0
(so regressions/failures are visible as drops to the floor).

Primary renderer is matplotlib (Agg backend, headless).  If matplotlib cannot be imported
*and* cannot be installed, we fall back to a clean ASCII/textual plot written to ``.txt``.

Public API (see fa3agent.interfaces):
    update_plot(run_dir, candidates_jsonl, out_png=None) -> str
"""
from __future__ import annotations

import json
import math
import os
from typing import Any, Optional

__all__ = ["update_plot"]


# --------------------------------------------------------------------------------------
# Parsing  (robust to missing / partial / malformed fields)
# --------------------------------------------------------------------------------------
def _safe_float(v: Any) -> float:
    """Coerce to a finite float; non-numeric / NaN / inf -> 0.0."""
    try:
        f = float(v)
    except (TypeError, ValueError):
        return 0.0
    if math.isnan(f) or math.isinf(f):
        return 0.0
    return f


def _parse_steps(candidates_jsonl: str) -> list[dict]:
    """Read candidates.jsonl -> list of distilled per-step dicts, sorted by step.

    Each returned dict has keys:
        step:int  cid:str  status:str  is_best:bool
        min_ratio:float  geomean_ratio:float
        per_config: dict[config_name -> ratio float]
    Lines that are blank or not valid JSON are skipped silently.
    """
    rows: list[dict] = []
    if not candidates_jsonl or not os.path.exists(candidates_jsonl):
        return rows
    with open(candidates_jsonl, "r") as fh:
        for i, line in enumerate(fh):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except (ValueError, TypeError):
                continue
            if not isinstance(rec, dict):
                continue
            ev = rec.get("eval") or {}
            if not isinstance(ev, dict):
                ev = {}
            per_config: dict[str, float] = {}
            pc = ev.get("per_config") or []
            if isinstance(pc, list):
                for cr in pc:
                    if not isinstance(cr, dict):
                        continue
                    name = cr.get("config")
                    if not name:
                        continue
                    per_config[str(name)] = _safe_float(cr.get("ratio"))
            step = rec.get("step")
            try:
                step = int(step)
            except (TypeError, ValueError):
                step = i
            rows.append(
                dict(
                    step=step,
                    cid=str(rec.get("cid", f"step{step}")),
                    status=str(rec.get("status", "")),
                    is_best=bool(rec.get("is_best", False)),
                    min_ratio=_safe_float(ev.get("min_ratio")),
                    geomean_ratio=_safe_float(ev.get("geomean_ratio")),
                    per_config=per_config,
                )
            )
    rows.sort(key=lambda r: r["step"])
    return rows


def _running_best(values: list[float]) -> list[float]:
    """Cumulative max, treating None/<=0 as 'no contribution' (carry prior best)."""
    best = 0.0
    out: list[float] = []
    for v in values:
        if v > best:
            best = v
        out.append(best)
    return out


# --------------------------------------------------------------------------------------
# matplotlib renderer
# --------------------------------------------------------------------------------------
def _render_matplotlib(rows: list[dict], out_png: str) -> str:
    import matplotlib

    matplotlib.use("Agg")  # headless / no display
    import matplotlib.pyplot as plt

    xs = [r["step"] for r in rows]
    min_r = [r["min_ratio"] for r in rows]
    geo_r = [r["geomean_ratio"] for r in rows]

    rb_min = _running_best(min_r)
    rb_geo = _running_best(geo_r)

    fig, ax = plt.subplots(figsize=(10, 6))

    # Per-config faint dashed lines (one per config that ever appears).
    config_names: list[str] = []
    for r in rows:
        for name in r["per_config"]:
            if name not in config_names:
                config_names.append(name)
    cmap = plt.get_cmap("tab10")
    for idx, name in enumerate(config_names):
        ys = [r["per_config"].get(name, 0.0) for r in rows]
        ax.plot(
            xs,
            ys,
            linestyle="--",
            linewidth=1.0,
            alpha=0.4,
            color=cmap(idx % 10),
            label=f"cfg: {name}",
        )

    # Running-best min_ratio (worst-case speedup) -- solid.
    ax.plot(
        xs,
        rb_min,
        linestyle="-",
        linewidth=2.0,
        color="#1f77b4",
        marker=".",
        label="Running best (min ratio)",
    )

    # Running-best geomean -- solid, with circle markers on NEW-best steps.
    ax.plot(
        xs,
        rb_geo,
        linestyle="-",
        linewidth=2.2,
        color="#2ca02c",
        label="Running best (geomean)",
    )
    new_best_x: list[float] = []
    new_best_y: list[float] = []
    prev = 0.0
    for r in rows:
        # A "new best" step is one whose own geomean strictly advanced the running best.
        if r["geomean_ratio"] > prev + 1e-12:
            new_best_x.append(r["step"])
            new_best_y.append(r["geomean_ratio"])
            prev = r["geomean_ratio"]
    if new_best_x:
        ax.scatter(
            new_best_x,
            new_best_y,
            s=90,
            facecolors="none",
            edgecolors="#2ca02c",
            linewidths=2.0,
            zorder=5,
            label="New best (geomean)",
        )

    # Baseline parity reference at y=1.0.
    ax.axhline(
        y=1.0,
        linestyle="--",
        linewidth=1.3,
        color="0.4",
        label="Baseline parity (1.0x)",
    )

    # Annotate the final committed point with its geomean ratio.
    if rows:
        last = rows[-1]
        fy = rb_geo[-1]
        ax.annotate(
            f"{last['cid']} ({fy:.3f}x)",
            xy=(last["step"], fy),
            xytext=(6, 8),
            textcoords="offset points",
            fontsize=9,
            color="#2ca02c",
        )

    ax.set_title("Evolution Trajectory: FA3 Distillation (speedup vs vLLM FA3 baseline)")
    ax.set_xlabel("Committed kernel version (step)")
    ax.set_ylabel("Speedup ratio  (baseline_latency / candidate_latency)")
    ax.grid(True, linestyle=":", alpha=0.4)

    # Integer x ticks when the lineage is short.
    if xs and (max(xs) - min(xs)) <= 30:
        ax.set_xticks(sorted(set(xs)))

    # Sensible y-limits: include parity line and a margin above the best.
    all_y = rb_min + rb_geo + [1.0]
    for r in rows:
        all_y.extend(r["per_config"].values())
    pos_y = [y for y in all_y if y > 0]
    ymax = max(pos_y) if pos_y else 1.2
    ax.set_ylim(0.0, max(1.2, ymax * 1.12))

    ax.legend(loc="best", fontsize=8, framealpha=0.85)
    fig.tight_layout()

    os.makedirs(os.path.dirname(out_png) or ".", exist_ok=True)
    fig.savefig(out_png, dpi=120)
    plt.close(fig)
    return out_png


# --------------------------------------------------------------------------------------
# ASCII fallback renderer
# --------------------------------------------------------------------------------------
def _render_ascii(rows: list[dict], out_txt: str, width: int = 60, height: int = 18) -> str:
    """Write a clean textual evolution plot to ``out_txt`` and return its path."""
    lines: list[str] = []
    lines.append("Evolution Trajectory: FA3 Distillation (speedup vs vLLM FA3 baseline)")
    lines.append("X = committed step,  Y = speedup ratio (baseline/candidate),  1.0 = parity")
    lines.append("")

    if not rows:
        lines.append("(no candidates yet)")
        body = "\n".join(lines) + "\n"
        os.makedirs(os.path.dirname(out_txt) or ".", exist_ok=True)
        with open(out_txt, "w") as fh:
            fh.write(body)
        return out_txt

    geo = [r["geomean_ratio"] for r in rows]
    mn = [r["min_ratio"] for r in rows]
    rb_geo = _running_best(geo)
    rb_min = _running_best(mn)

    all_vals = rb_geo + rb_min + geo + mn + [1.0]
    pos = [v for v in all_vals if v > 0]
    ymax = max(pos) if pos else 1.2
    ymax = max(1.2, ymax * 1.05)
    ymin = 0.0

    n = len(rows)
    grid = [[" "] * n for _ in range(height)]

    def yrow(val: float) -> int:
        frac = (val - ymin) / (ymax - ymin) if ymax > ymin else 0.0
        frac = min(max(frac, 0.0), 1.0)
        # top row = highest value
        return height - 1 - int(round(frac * (height - 1)))

    # baseline parity line
    parity_row = yrow(1.0)
    for c in range(n):
        grid[parity_row][c] = "-"

    # running-best min ('o') and geomean ('*'); geomean drawn last (priority)
    for c in range(n):
        if rb_min[c] > 0:
            grid[yrow(rb_min[c])][c] = "o"
    for c in range(n):
        if rb_geo[c] > 0:
            grid[yrow(rb_geo[c])][c] = "*"
    # mark new-best steps with '#'
    prev = 0.0
    for c, r in enumerate(rows):
        if r["geomean_ratio"] > prev + 1e-12:
            grid[yrow(rb_geo[c])][c] = "#"
            prev = r["geomean_ratio"]
        prev = max(prev, rb_geo[c])

    # render grid with y-axis labels
    for row_i in range(height):
        # label only top, parity, bottom rows
        frac = 1.0 - row_i / (height - 1) if height > 1 else 1.0
        yval = ymin + frac * (ymax - ymin)
        label = f"{yval:5.2f} |"
        lines.append(label + "".join(grid[row_i]))
    # x-axis
    lines.append(" " * 6 + "+" + "-" * n)
    xticks = " " * 7 + "".join(str(rows[c]["step"] % 10) for c in range(n))
    lines.append(xticks)
    lines.append("")
    lines.append("legend: * running-best geomean   o running-best min   # new best   - parity(1.0)")
    lines.append("")

    # tabular detail
    lines.append("step  cid                 status      min_ratio  geomean  best  per_config")
    for r in rows:
        pc = " ".join(f"{k}={v:.3f}" for k, v in r["per_config"].items())
        if len(pc) > 60:
            pc = pc[:57] + "..."
        lines.append(
            f"{r['step']:<5d} {r['cid']:<19.19s} {r['status']:<11.11s} "
            f"{r['min_ratio']:<9.3f} {r['geomean_ratio']:<8.3f} "
            f"{'Y' if r['is_best'] else ' ':<5s} {pc}"
        )

    body = "\n".join(lines) + "\n"
    os.makedirs(os.path.dirname(out_txt) or ".", exist_ok=True)
    with open(out_txt, "w") as fh:
        fh.write(body)
    return out_txt


# --------------------------------------------------------------------------------------
# Public entry point
# --------------------------------------------------------------------------------------
def update_plot(run_dir: str, candidates_jsonl: str, out_png: Optional[str] = None) -> str:
    """Regenerate the AVO-style evolution plot from ``candidates_jsonl``.

    Args:
        run_dir: run directory (default output goes under ``<run_dir>/plots/``).
        candidates_jsonl: path to candidates.jsonl (one StepRecord asdict per line).
        out_png: optional explicit output path. Defaults to
            ``<run_dir>/plots/evolution.png``.

    Returns:
        Absolute path to the rendered file (``.png`` on the matplotlib path; ``.txt``
        if it fell back to the ASCII renderer).
    """
    if out_png is None:
        out_png = os.path.join(run_dir, "plots", "evolution.png")
    out_png = os.path.abspath(out_png)

    rows = _parse_steps(candidates_jsonl)

    # Try matplotlib; install-on-demand is allowed for this viz dep.
    try:
        import matplotlib  # noqa: F401
        return os.path.abspath(_render_matplotlib(rows, out_png))
    except ImportError:
        pass
    try:
        import subprocess
        import sys

        subprocess.run(
            [sys.executable, "-m", "pip", "install", "matplotlib"],
            check=True,
            capture_output=True,
            timeout=300,
        )
        import matplotlib  # noqa: F401
        return os.path.abspath(_render_matplotlib(rows, out_png))
    except Exception:
        # Clean textual fallback.
        base, _ext = os.path.splitext(out_png)
        out_txt = base + ".txt"
        return os.path.abspath(_render_ascii(rows, out_txt))
