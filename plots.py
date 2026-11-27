"""Renders the two run plots to PNG bytes.

Why not pyplot / flet's MatplotlibChart: pyplot keeps one global "current
figure", which raced between the GP polling thread and UI callbacks. The
chart could then serialise a half-built figure (blank flash) or one before
`tight_layout` (shrunken plot). Here every call builds its own `Figure`
with fixed margins and a fixed pixel size, so consecutive frames are
pixel-aligned and can be swapped without any layout change.
"""

import contextlib
import io
import math
import threading
from collections import Counter

import matplotlib

matplotlib.use("Agg")

import matplotlib.colors  # noqa: E402
import matplotlib.lines  # noqa: E402
import matplotlib.patches  # noqa: E402

import numpy as np
from matplotlib.figure import Figure
from matplotlib.ticker import FuncFormatter, MaxNLocator
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers the 3d projection)

import problems
import style

# Plots are rendered at the exact size of their container (in UI pixels),
# at 2x resolution for sharp text, with fixed pixel margins around the axes.
DEFAULT_SIZE = (500, 360)
SCALE = 2
# Left margins only leave room for the tick labels (the error plot has no axis
# label; its card title says what is plotted).
MARGIN_LEFT_ERROR, MARGIN_LEFT_FIT = 40, 50
MARGIN_RIGHT, MARGIN_TOP, MARGIN_BOTTOM = 16, 10, 42

_lock = threading.RLock()

_RC = {
    "font.family": "sans-serif",
    "font.sans-serif": style.PLOT_SANS,
    "font.monospace": style.PLOT_MONO,
    "font.size": 9,
    "axes.edgecolor": style.RULE,
    "axes.labelcolor": style.MUTED,
    "axes.labelsize": 9,
    "axes.linewidth": 0.8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.color": style.RULE_SOFT,
    "grid.linewidth": 0.7,
    "xtick.color": style.FAINT,
    "ytick.color": style.FAINT,
    "xtick.labelcolor": style.MUTED,
    "ytick.labelcolor": style.MUTED,
    "xtick.major.size": 0,
    "ytick.major.size": 0,
    "xtick.major.pad": 6,
    "ytick.major.pad": 4,
    "legend.frameon": True,
    "legend.framealpha": 0.92,
    "legend.edgecolor": style.RULE,
    "legend.facecolor": style.PANEL,
    "legend.fontsize": 8,
    "legend.borderpad": 0.6,
    "legend.handlelength": 2.2,
    "lines.solid_capstyle": "round",
    "pdf.fonttype": 42,   # embed TrueType fonts: text stays editable in exported PDFs
}


def _figure(width, height, left):
    width, height = max(int(width), 240), max(int(height), 180)
    fig = Figure(figsize=(width / 100, height / 100), dpi=100, facecolor=_panel())
    # Fixed pixel margins: the axes box must not move between frames.
    fig.subplots_adjust(left=left / width, right=1 - MARGIN_RIGHT / width,
                        top=1 - MARGIN_TOP / height, bottom=MARGIN_BOTTOM / height)
    ax = fig.add_subplot()
    ax.set_facecolor(_panel())
    ax.set_axisbelow(True)
    return fig, ax


DEFAULT_VIEW = (24.0, -55.0)   # elevation, azimuth of the 3D plot


_output = threading.local()


@contextlib.contextmanager
def output_format(fmt):
    """Render to another format in this thread, e.g. `with output_format("pdf"): ...`."""
    _output.fmt = fmt
    try:
        yield
    finally:
        _output.fmt = "png"


def _png(fig, scale=SCALE):
    buf = io.BytesIO()
    fmt = getattr(_output, "fmt", "png")
    scale = getattr(_output, "scale", None) or scale
    fig.savefig(buf, format=fmt, dpi=100 * scale, facecolor=_panel())
    return buf.getvalue()


def _panel():
    """Plot background: the app's panel colour, white for printing (see print_style)."""
    return "#FFFFFF" if getattr(_output, "white", False) else style.PANEL


def _rc():
    return {**_RC, "legend.facecolor": _panel()}


@contextlib.contextmanager
def print_style(scale=5):
    """In this thread: white background and `scale` × 100 dpi (plots for the PDF report)."""
    _output.white, _output.scale = True, scale
    try:
        yield
    finally:
        _output.white, _output.scale = False, None


def _short_number(value, _pos=None):
    """3 significant digits, short exponent: 0.25, 12.5, 4e5, 3e-4."""
    if value == 0:
        return "0"
    text = f"{value:.3g}"
    return text.replace("e+0", "e").replace("e+", "e").replace("e-0", "e-")


def _clean(values):
    arr = np.asarray(values, dtype=float)
    arr[~np.isfinite(arr)] = np.nan
    return arr


def render_error_plot(snapshot, generations, adjust, size=DEFAULT_SIZE, holdout="validation", percent=False):
    # holdout=None: no separate test set (Boolean problems use the full truth table)
    """Error curves; with `percent` (program synthesis) the share of solved cases instead."""
    with _lock, matplotlib.rc_context(_rc()):
        fig, ax = _figure(*size, MARGIN_LEFT_ERROR)
        n = snapshot.generation if snapshot else 0
        if n:
            xs = np.arange(1, n + 1)
            series = [
                (f"Avg. {holdout}", snapshot.avg_validation, style.TEAL, "-", 1.6),
                ("Avg. training", snapshot.avg_training, style.TEAL_SOFT, (0, (4, 2.5)), 1.4),
                (f"Best {holdout}", snapshot.best_validation, style.ACCENT, "-", 1.8),
                ("Best training", snapshot.best_training, style.ACCENT, (0, (1.5, 2)), 1.6),
            ]
            if percent:
                series = [
                    ("Best test", snapshot.best_validation, style.GREEN, "-", 1.8),
                    ("Best training", snapshot.best_training, style.GREEN, (0, (1.5, 2)), 1.6),
                    ("Avg. test", snapshot.avg_validation, style.TEAL, "-", 1.4),
                    ("Avg. training", snapshot.avg_training, style.TEAL_SOFT, (0, (4, 2.5)), 1.4),
                ]
                if holdout is None:
                    series = [("Best program", snapshot.best_training, style.GREEN, "-", 1.8),
                              ("Population avg.", snapshot.avg_training, style.TEAL, "-", 1.4)]
            for label, values, color, ls, lw in series:
                last = values[-1]
                if percent:
                    value = f"{last:5.1f} %"
                else:
                    value = f"{last:.4f}" if math.isfinite(last) else "inf"
                ax.plot(xs, _clean(values), color=color, linestyle=ls, linewidth=lw,
                        label=f"{label}  {value}")
            leg = ax.legend(loc="lower right" if percent else "upper right")
            for text in leg.get_texts():
                text.set_color(style.INK)
                text.set_fontfamily("monospace")
                text.set_fontsize(7.5)
            if percent:
                ax.set_ylim(-2, 102)
            elif adjust:
                first = snapshot.avg_validation[0]
                top = min(first, 15.0) if math.isfinite(first) else 15.0
                ax.set_ylim(-0.04, top + 0.2)
        elif percent:
            ax.set_ylim(-2, 102)
        else:
            ax.set_ylim(0, 1)
            ax.set_yticklabels([])
        ax.set_xlim(1, max(generations, 2))
        ax.xaxis.set_major_locator(MaxNLocator(nbins=6, integer=True, min_n_ticks=2))
        # compact tick labels (max ~5 characters) so the narrow left margin always fits
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0f}%" if percent else _short_number(v)))
        ax.set_xlabel("Generation")
        return _png(fig)


def render_fit_plot(snapshot, problem, train_X, train_y, adjust, size=DEFAULT_SIZE,
                    view=DEFAULT_VIEW, scale=SCALE, test_size=0, test_y=None):
    """Best individual vs. training data: a curve for one input, a surface for two.
    `view` (elevation, azimuth) and a lower `scale` are used while the 3D plot is dragged."""
    n_edge = problem.edge_count(len(train_y)) if problem.kind == "ps" else 0
    with _lock, matplotlib.rc_context(_rc()):
        if problem.kind == "logic":
            return _render_kmap(snapshot, problem, train_y, size)
        if problem.kind == "ps":
            return _render_cases(snapshot, len(train_y), test_size, size, n_edge)
        if getattr(problem, "dataset", False):
            return _render_predicted(snapshot, train_y, test_y or [], adjust, size)
        if problem.inputs == 2:
            return _render_surface(snapshot, problem, train_X, train_y, adjust, size, view, scale)
        fig, ax = _figure(*size, MARGIN_LEFT_FIT)
        xs = np.array([row[0] for row in train_X])
        ys = np.asarray(train_y, dtype=float)
        curve_x = problems.prediction_axis(problem)
        grid = np.linspace(curve_x[0], curve_x[-1], 400)
        ax.plot(grid, problems.evaluate(problem.expression, grid[:, None]), color=style.FAINT,
                linewidth=1.0, linestyle=(0, (3, 3)), label="Target function", zorder=1)
        ds = _downsample(snapshot)
        ax.scatter(xs, ys, s=16, facecolors=_panel(), edgecolors=style.INK if ds is None else style.FAINT,
                   linewidths=0.9, label="Training points", zorder=3)
        if ds is not None:
            ax.scatter(xs[ds], ys[ds], s=30, facecolors=style.TEAL, edgecolors=_panel(),
                       linewidths=0.9, label="Current down-sample", zorder=4)
        out = _outliers(snapshot)
        if out is not None:
            ax.scatter(xs[out], ys[out], s=44, marker="x", color=style.OCHRE, linewidths=1.6,
                       label="Excluded as outlier", zorder=5)
        if snapshot is not None and snapshot.best_prediction is not None:
            ax.plot(curve_x, snapshot.best_prediction, color=style.ACCENT,
                    linewidth=2.0, label="Best individual", zorder=2)
        ax.set_xlabel("x")
        ax.set_ylabel("y", labelpad=2)
        legend_pos = "best"
        if adjust:
            ax.set_xticks(np.linspace(*problem.domain, 5))
            ax.set_xlim(*problem.validation_domain)
            if problem.limits is not None:
                y_ticks, y_lim, legend_pos = problem.limits
                ax.set_yticks(y_ticks)
                ax.set_ylim(*y_lim)
            else:
                ax.set_ylim(*problems.value_range(problem, pad=0.12))
        leg = ax.legend(loc=legend_pos)
        for text in leg.get_texts():
            text.set_color(style.INK)
        return _png(fig)


CASE_HEADER, CASE_LABEL = 30, 18


def _case_footer(width):
    """Legend rows depend on the width only, so renderer and hover/click agree."""
    return (28, 1) if width >= 640 else (46, 2)


MAX_CELL = 80


def case_layout(n_cases, n_edge, size):
    """Pixel rectangles (x, y, w, h) of the training-case cells in the case plot, plus
    the block labels and the left edge of the content. The grid grows with the plot
    and is centred in it. Shared by the renderer and the hover / click lookup."""
    width, height = max(int(size[0]), 240), max(int(size[1]), 180)
    blocks = [(0, n_edge, "Edge cases"), (n_edge, n_cases - n_edge, "Random cases")] if n_edge else \
        [(0, n_cases, "Training cases")]
    blocks = [blk for blk in blocks if blk[1] > 0]
    extra = CASE_LABEL * len(blocks) + 6 * (len(blocks) - 1)
    area_h = height - CASE_HEADER - _case_footer(width)[0]
    avail_w, avail_h = width - 4, area_h - extra
    best = (0, 1)
    for cols in range(1, max(n_cases, 1) + 1):
        rows = sum(math.ceil(count / cols) for _, count, _ in blocks)
        cell = min(avail_w / cols, avail_h / max(rows, 1), MAX_CELL)
        if cell > best[0]:
            best = (cell, cols)
    cell, cols = best
    rows = sum(math.ceil(count / cols) for _, count, _ in blocks)
    x0 = 2 + max(0.0, (avail_w - cols * cell) / 2)
    y = CASE_HEADER + max(0.0, (area_h - extra - rows * cell) / 2)
    rects, labels = [], []
    for start, count, label in blocks:
        labels.append((label, count, y))
        y += CASE_LABEL
        for k in range(count):
            r, c = divmod(k, cols)
            rects.append((x0 + c * cell, y + r * cell, cell, cell))
        y += math.ceil(count / cols) * cell + 6
    return rects, labels, x0


def _gray(bits):
    return [format(i ^ (i >> 1), f"0{bits}b") if bits else "" for i in range(2 ** bits)]


def kmap_layout(n_inputs, size):
    """Karnaugh map of a truth table: rows = Gray code of the first half of the inputs,
    columns = Gray code of the rest. Returns (rects indexed by truth-table row,
    row codes with y, column codes with x, cell size, grid origin)."""
    width, height = max(int(size[0]), 240), max(int(size[1]), 180)
    k = n_inputs // 2
    m = n_inputs - k
    rows, cols = _gray(k), _gray(m)
    left = 30 + 8 * max(k, 1)           # room for the row variable names and codes
    top = CASE_HEADER + 54              # variable names + column codes
    avail_w = width - left - 6
    avail_h = height - top - _case_footer(width)[0]
    cell = max(6.0, min(avail_w / len(cols), avail_h / len(rows), 90))
    x0 = left + max(0.0, (avail_w - cell * len(cols)) / 2)
    y0 = top + max(0.0, (avail_h - cell * len(rows)) / 2)
    rects = [None] * (2 ** n_inputs)
    for r, rc in enumerate(rows):
        for c, cc in enumerate(cols):
            rects[int(rc + cc, 2) if n_inputs else 0] = (x0 + c * cell, y0 + r * cell, cell, cell)
    return rects, [(code, y0 + (r + 0.5) * cell) for r, code in enumerate(rows)], \
        [(code, x0 + (c + 0.5) * cell) for c, code in enumerate(cols)], cell, (x0, y0)


def case_index_at(problem, n_cases, size, x, y):
    """Training case under the pixel position (x, y) of the case plot, or None."""
    if problem.kind == "logic":
        cells = kmap_layout(len(problem.arg_names), size)[0]
        rects = [cells[t] for t in _cell_index(problem, n_cases)]
    else:
        rects = case_layout(n_cases, problem.edge_count(n_cases), size)[0]
    for i, rect in enumerate(rects):
        if rect is None:
            continue
        rx, ry, w, h = rect
        if rx <= x < rx + w and ry <= y < ry + h:
            return i
    return None


def _cell_index(problem, n_cases):
    """Truth-table cell of every case (identity for complete tables)."""
    return getattr(problem, "cell_index", None) or list(range(n_cases))


def _render_kmap(snapshot, problem, train_y, size):
    """Boolean problems: Karnaugh map. Each cell is one truth-table row and shows the
    expected output; fill = share of the population that gets the row right,
    frame = the best program gets it right."""
    width, height = max(int(size[0]), 240), max(int(size[1]), 180)
    fig = Figure(figsize=(width / 100, height / 100), dpi=100, facecolor=_panel())
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_axis_off()
    ax.set_xlim(0, width)
    ax.set_ylim(height, 0)
    names = problem.arg_names
    n = len(names)
    k = n // 2
    rects, row_codes, col_codes, cell, (x0, y0) = kmap_layout(n, size)

    share = None if snapshot is None or snapshot.case_share is None else np.asarray(snapshot.case_share)
    solved = None if snapshot is None or snapshot.best_solved is None else np.asarray(snapshot.best_solved)
    ds = set(_downsample(snapshot).tolist()) if _downsample(snapshot) is not None else set()
    code_size = min(10.5, max(7.0, cell * 0.22))
    value_size = min(20.0, max(7.0, cell * 0.36))

    # Variable names sit at their axis: the first half of the inputs is coded along the
    # map's rows (left, vertical label), the rest along its columns (top). Codes are Gray
    # codes, so neighbouring cells differ in exactly one input bit.
    grid_w, grid_h = cell * len(col_codes), cell * len(row_codes)
    ax.text(x0 + grid_w / 2, y0 - 34, "  ".join(names[k:]), ha="center", va="center", color=style.TEAL,
            fontsize=10, fontfamily="monospace", fontweight="bold")
    if k:
        label_x = max(10.0, x0 - 6 - code_size * 0.75 * k - 12)   # just left of the row codes
        ax.text(label_x, y0 + grid_h / 2, "  ".join(names[:k]), ha="center", va="center", color=style.TEAL,
                fontsize=10, fontfamily="monospace", fontweight="bold", rotation=90)
    for code, y in row_codes:
        ax.text(x0 - 6, y, code, ha="right", va="center", color=style.INK, fontsize=code_size, fontfamily="monospace")
    for code, x in col_codes:
        ax.text(x, y0 - 8, code, ha="center", va="bottom", color=style.INK, fontsize=code_size,
                fontfamily="monospace")

    case_of = {t: i for i, t in enumerate(_cell_index(problem, len(train_y)))}
    for t, rect in enumerate(rects):
        x, y, w, h = rect
        if t not in case_of:   # input combination missing in the data set
            gap = max(1.5, 0.08 * w)
            ax.add_patch(matplotlib.patches.FancyBboxPatch(
                (x + gap / 2, y + gap / 2), w - gap, h - gap, boxstyle=f"round,pad=0,rounding_size={0.12 * w:.1f}",
                facecolor=_panel(), edgecolor=style.RULE, linewidth=1.0, linestyle=(0, (2, 2))))
            ax.text(x + w / 2, y + h / 2, "–", ha="center", va="center", color=style.FAINT,
                    fontsize=value_size * 0.8, fontfamily="monospace")
            continue
        i = case_of[t]
        gap = max(1.5, 0.08 * w)
        face = _SHARE_CMAP(float(share[i])) if share is not None and i < len(share) else style.GREEN_LIGHT
        best_ok = solved is not None and i < len(solved) and solved[i]
        ax.add_patch(matplotlib.patches.FancyBboxPatch(
            (x + gap / 2, y + gap / 2), w - gap, h - gap, boxstyle=f"round,pad=0,rounding_size={0.12 * w:.1f}",
            facecolor=face, edgecolor=style.INK if best_ok else face, linewidth=1.8 if best_ok else 0))
        dark = share is not None and i < len(share) and share[i] > 0.6
        ax.text(x + w / 2, y + h / 2, "1" if train_y[i] else "0", ha="center", va="center",
                color=_panel() if dark else style.INK, fontsize=value_size, fontfamily="monospace")
        if i in ds:
            ax.add_patch(matplotlib.patches.Circle((x + w - 0.2 * w, y + 0.2 * h), max(1.5, 0.06 * w),
                                                   facecolor=style.INK, edgecolor="none"))

    shape = f"{len(row_codes)} × {len(col_codes)}"
    n_rows = len(train_y)
    if solved is not None:
        summary = f"best program  {int(solved.sum())}/{n_rows} input combinations correct"
    elif n_rows < len(rects):
        summary = f"{n_rows} of {len(rects)} input combinations in the data  ·  {shape} cells"
    else:
        summary = f"{len(rects)} input combinations = {shape} cells  ·  cell = expected output"
    ax.text(4, 14, summary, ha="left", va="center", color=style.INK, fontsize=9.5, fontfamily="monospace")
    handles = [matplotlib.patches.Patch(facecolor=style.GREEN_LIGHT, label="no one gets it right"),
               matplotlib.patches.Patch(facecolor=style.GREEN, label="whole population"),
               matplotlib.patches.Patch(facecolor=_panel(), edgecolor=style.INK, linewidth=1.6,
                                        label="correct in best program")]
    if len(case_of) < len(rects):
        handles.append(matplotlib.patches.Patch(facecolor=_panel(), edgecolor=style.RULE, linestyle=(0, (2, 2)),
                                                label="not in the data"))
    if ds:
        handles.append(matplotlib.lines.Line2D([], [], linestyle="", marker="o", markersize=4,
                                               color=style.INK, label="down-sample"))
    rows = _case_footer(width)[1]
    leg = fig.legend(handles=handles, loc="lower left", bbox_to_anchor=(max(0.0, (x0 - 40) / width), 0),
                     ncol=len(handles) if rows == 1 else math.ceil(len(handles) / 2),
                     frameon=False, handlelength=1.2, columnspacing=1.0, labelspacing=0.4, fontsize=8)
    for text in leg.get_texts():
        text.set_color(style.INK)
    return _png(fig)


def case_at(n_cases, n_edge, size, x, y):
    """Index of the case cell under the pixel position (x, y), or None."""
    rects, _, _ = case_layout(n_cases, n_edge, size)
    for i, (rx, ry, w, h) in enumerate(rects):
        if rx <= x < rx + w and ry <= y < ry + h:
            return i
    return None


_SHARE_CMAP = matplotlib.colors.LinearSegmentedColormap.from_list(
    "share", [style.GREEN_LIGHT, "#A9CDB0", style.GREEN])


def _render_cases(snapshot, n_cases, n_test, size, n_edge=0):
    """Program synthesis: one cell per training case. Fill = share of the population
    that solves the case (case difficulty), frame = solved by the best program."""
    width, height = max(int(size[0]), 240), max(int(size[1]), 180)
    fig = Figure(figsize=(width / 100, height / 100), dpi=100, facecolor=_panel())
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_axis_off()
    ax.set_xlim(0, width)
    ax.set_ylim(height, 0)
    rects, labels, x0 = case_layout(n_cases, n_edge, size)

    share = None if snapshot is None or snapshot.case_share is None else np.asarray(snapshot.case_share)
    solved = None if snapshot is None or snapshot.best_solved is None else np.asarray(snapshot.best_solved)
    ds = set(_downsample(snapshot).tolist()) if _downsample(snapshot) is not None else set()
    out = set(_outliers(snapshot).tolist()) if _outliers(snapshot) is not None else set()

    for label, count, y in labels:
        ax.text(x0, y + CASE_LABEL / 2, f"{label} ({count})", ha="left", va="center", color=style.INK,
                fontsize=8.5)
    for i, (x, y, w, h) in enumerate(rects):
        gap = max(1.5, 0.12 * w)
        face = _SHARE_CMAP(float(share[i])) if share is not None and i < len(share) else style.GREEN_LIGHT
        best_ok = solved is not None and i < len(solved) and solved[i]
        ax.add_patch(matplotlib.patches.FancyBboxPatch(
            (x + gap / 2, y + gap / 2), w - gap, h - gap, boxstyle=f"round,pad=0,rounding_size={0.15 * w:.1f}",
            facecolor=face, edgecolor=style.INK if best_ok else face, linewidth=1.8 if best_ok else 0))
        if i in ds:
            ax.add_patch(matplotlib.patches.Circle((x + w / 2, y + h / 2), max(1.6, 0.12 * w),
                                                   facecolor=style.INK, edgecolor="none"))
        if i in out:
            k = 0.28 * w
            ax.plot([x + k, x + w - k], [y + k, y + h - k], color=style.OCHRE, linewidth=1.6)
            ax.plot([x + k, x + w - k], [y + h - k, y + k], color=style.OCHRE, linewidth=1.6)

    if solved is not None:
        summary = f"best program  {int(solved.sum())}/{n_cases} train  ·  {snapshot.best_test_solved}/{n_test} test"
    else:
        summary = f"{n_cases} training cases"
    ax.text(x0, 14, summary, ha="left", va="center", color=style.INK, fontsize=9.5, fontfamily="monospace")

    handles = [matplotlib.patches.Patch(facecolor=style.GREEN_LIGHT, label="no one solves it"),
               matplotlib.patches.Patch(facecolor=style.GREEN, label="whole population"),
               matplotlib.patches.Patch(facecolor=_panel(), edgecolor=style.INK, linewidth=1.6,
                                        label="solved by best program")]
    if ds:
        handles.append(matplotlib.lines.Line2D([], [], linestyle="", marker="o", markersize=4,
                                               color=style.INK, label="down-sample"))
    if out:
        handles.append(matplotlib.lines.Line2D([], [], linestyle="", marker="x", markersize=6,
                                               markeredgewidth=1.6, color=style.OCHRE, label="outlier"))
    rows = _case_footer(width)[1]
    leg = fig.legend(handles=handles, loc="lower left", bbox_to_anchor=(max(0.0, (x0 - 8) / width), 0),
                     ncol=len(handles) if rows == 1 else math.ceil(len(handles) / 2),
                     frameon=False, handlelength=1.2, columnspacing=1.0, labelspacing=0.4, fontsize=8)
    for text in leg.get_texts():
        text.set_color(style.INK)
    return _png(fig)


def _r2(actual, predicted):
    """Coefficient of determination; invalid predictions count as predicting the mean."""
    a, p = np.asarray(actual, dtype=float), np.asarray(predicted, dtype=float)
    if len(a) < 2:
        return float("nan")
    p = np.where(np.isfinite(p), p, a.mean())
    ss_tot = float(np.sum((a - a.mean()) ** 2)) or 1.0
    return 1.0 - float(np.sum((a - p) ** 2)) / ss_tot


def _render_predicted(snapshot, train_y, test_y, adjust, size):
    """Data sets with any number of inputs: prediction of the best individual against
    the measured target value; perfect predictions lie on the diagonal."""
    fig, ax = _figure(*size, MARGIN_LEFT_FIT)
    ys, yt = np.asarray(train_y, dtype=float), np.asarray(test_y, dtype=float)
    every = np.concatenate([ys, yt]) if len(yt) else ys
    lo, hi = float(every.min()), float(every.max())
    pad = 0.06 * (hi - lo or 1.0)
    lo, hi = lo - pad, hi + pad
    ax.plot([lo, hi], [lo, hi], color=style.FAINT, linewidth=1.0, linestyle=(0, (3, 3)), zorder=1,
            label="perfect prediction")
    if snapshot is not None and snapshot.best_prediction is not None:
        p_train, p_test = snapshot.best_prediction
        if adjust:
            p_train, p_test = np.clip(p_train, lo, hi), np.clip(p_test, lo, hi)
        ax.scatter(ys, p_train, s=12, facecolors=_panel(), edgecolors=style.INK, linewidths=0.7, zorder=3,
                   label=f"training   R² {_r2(ys, snapshot.best_prediction[0]):.3f}")
        if len(yt):
            ax.scatter(yt, p_test, s=14, color=style.ACCENT, edgecolors="none", alpha=0.75, zorder=4,
                       label=f"validation   R² {_r2(yt, snapshot.best_prediction[1]):.3f}")
    else:
        ax.scatter(ys, ys, s=0)   # keeps the axes for the empty plot
    ax.set_xlim(lo, hi)
    if adjust:
        ax.set_ylim(lo, hi)
    ax.set_xlabel("measured value")
    ax.set_ylabel("prediction", labelpad=2)
    leg = ax.legend(loc="upper left")
    for text in leg.get_texts():
        text.set_color(style.INK)
        text.set_fontfamily("monospace")
    return _png(fig)


def _outliers(snapshot):
    if snapshot is None or snapshot.outliers is None or len(snapshot.outliers) == 0:
        return None
    return np.asarray(snapshot.outliers, dtype=int)


def _downsample(snapshot):
    if snapshot is None or snapshot.downsample is None or len(snapshot.downsample) == 0:
        return None
    return np.asarray(snapshot.downsample, dtype=int)


def _render_surface(snapshot, problem, train_X, train_y, adjust, size, view, scale):
    width, height = max(int(size[0]), 240), max(int(size[1]), 180)
    fig = Figure(figsize=(width / 100, height / 100), dpi=100, facecolor=_panel())
    # The 3D axes fill the figure; its own box is fixed by the view, so frames stay aligned.
    ax = fig.add_axes((-0.04, -0.01, 0.94, 1.04), projection="3d", facecolor=_panel(),
                      computed_zorder=False)  # draw order below: wireframe, surface, points
    ax.view_init(elev=view[0], azim=view[1])
    ax.set_box_aspect(None, zoom=0.98)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.set_pane_color(matplotlib.colors.to_rgba(style.PAPER, 0.6))
        axis._axinfo["grid"].update(color=style.RULE_SOFT, linewidth=0.6)
        axis.line.set_color(style.RULE)
        axis.set_tick_params(colors=style.MUTED, labelsize=7.5, pad=1)
    ax.set_xlabel("x", labelpad=-2)
    ax.set_ylabel("y", labelpad=-2)
    ax.set_zlabel("f(x, y)", labelpad=0)

    X = np.asarray(train_X, dtype=float)
    z = np.asarray(train_y, dtype=float)
    SURFACE_X, SURFACE_Y = problems.surface_grid(problem)
    n = SURFACE_X.shape[0]
    target = problems.evaluate(problem.expression, np.column_stack([SURFACE_X.ravel(), SURFACE_Y.ravel()]))
    lo, hi = problems.value_range(problem, pad=0.12)

    ax.plot_wireframe(SURFACE_X, SURFACE_Y, target.reshape(SURFACE_X.shape), rstride=2, cstride=2,
                      color=style.FAINT, linewidth=0.5, alpha=0.9, zorder=1)
    handles = [matplotlib.lines.Line2D([], [], color=style.FAINT, linewidth=0.8, label="Target function")]

    if snapshot is not None and snapshot.best_prediction is not None:
        Z = np.array(snapshot.best_prediction, dtype=float)
        if adjust:
            Z[(Z < lo) | (Z > hi)] = np.nan   # keep the axes readable; clipped parts are not drawn
        ax.plot_surface(SURFACE_X, SURFACE_Y, Z, color=style.ACCENT, alpha=0.55, linewidth=0,
                        antialiased=True, shade=True, rcount=n, ccount=n, zorder=2)
        handles.append(matplotlib.patches.Patch(color=style.ACCENT, alpha=0.6, label="Best individual"))

    ds = _downsample(snapshot)
    ax.scatter(X[:, 0], X[:, 1], z, s=9, c=style.INK if ds is None else style.MUTED, depthshade=False, zorder=3)
    handles.append(matplotlib.lines.Line2D([], [], linestyle="", marker="o", markersize=4,
                                           color=style.INK, label="Training points"))
    if ds is not None:
        ax.scatter(X[ds, 0], X[ds, 1], z[ds], s=26, c=style.TEAL, edgecolors=_panel(),
                   linewidths=0.6, depthshade=False, zorder=4)
        handles.append(matplotlib.lines.Line2D([], [], linestyle="", marker="o", markersize=5,
                                               color=style.TEAL, label="Current down-sample"))
    out = _outliers(snapshot)
    if out is not None:
        ax.scatter(X[out, 0], X[out, 1], z[out], s=40, marker="x", color=style.OCHRE, linewidths=1.6,
                   depthshade=False, zorder=5)
        handles.append(matplotlib.lines.Line2D([], [], linestyle="", marker="x", markersize=6,
                                               markeredgewidth=1.6, color=style.OCHRE, label="Excluded as outlier"))

    ax.set_xlim(*problem.validation_domain)
    ax.set_ylim(*problem.validation_domain)
    if adjust:
        ax.set_zlim(lo, hi)
    for setter in (ax.xaxis, ax.yaxis, ax.zaxis):
        setter.set_major_locator(MaxNLocator(nbins=5))
    leg = fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.01, 0.99))
    for text in leg.get_texts():
        text.set_color(style.INK)
    return _png(fig, scale)


# --------------------------------------------------------------------------
# Population dynamics
# --------------------------------------------------------------------------

DYNAMICS_VIEWS = {
    "fitness": "Fitness distribution",
    "size": "Tree size & depth",
    "diversity": "Diversity",
    "selection": "Selection pressure",
    "variation": "Variation effects",
    "blocks": "Building-block usage",
    "params": "Parameters over generations",
}

# parameters a parameter control may change (operators.CONTROLLABLE), with short axis labels
PARAM_LABELS = {"selection_method": "selection", "crossover_type": "crossover op.",
                "mutation_type": "mutation op.", "bloat_control": "bloat control", "population_size": "population",
                "crossover_probability": "crossover", "mutation_probability": "mutation",
                "tourn_size": "tourn. size", "parsimony_size": "parsimony size",
                "double_tournament_order": "tourn. order", "lexicase_epsilon": "fixed ε", "downsampling": "down-sampling", "ds_rate": "down-sample", "parent_sampling_rate": "parent sampl.",
                "distance_interval": "dist. interval", "outlier_rate": "outlier γ",
                "max_tree_depth": "max. depth", "max_tree_size": "max. size", "elitism": "elitism",
                "inner_node_probability": "inner nodes", "subtree_min_depth": "subtree min.",
                "subtree_max_depth": "subtree max.", "subtree_method": "subtree method",
                "ephemeral_mode": "ephemeral"}
PARAM_INTEGER = {"population_size", "tourn_size", "distance_interval", "max_tree_depth", "max_tree_size", "elitism",
                 "subtree_min_depth", "subtree_max_depth"}
PARAM_CHOICES = {"selection_method", "crossover_type", "mutation_type", "bloat_control", "subtree_method",
                 "ephemeral_mode", "double_tournament_order", "downsampling"}   # names, not numbers
MAX_PARAM_PANELS = 5


def _stacked_axes(fig, n, size, left_margin=MARGIN_LEFT_FIT):
    """n panels above each other with a shared x axis (instead of several y axes)."""
    width, height = max(int(size[0]), 240), max(int(size[1]), 180)
    left, right = left_margin / width, 1 - MARGIN_RIGHT / width
    top, bottom = 1 - MARGIN_TOP / height, MARGIN_BOTTOM / height
    gap = 18 / height
    h = (top - bottom - gap * (n - 1)) / n
    axes = []
    for k in range(n):
        y = top - (k + 1) * h - k * gap
        ax = fig.add_axes((left, y, right - left, h), sharex=axes[0] if axes else None)
        ax.set_facecolor(_panel())
        ax.set_axisbelow(True)
        axes.append(ax)
    for ax in axes[:-1]:
        ax.tick_params(labelbottom=False)
    axes[-1].xaxis.set_major_locator(MaxNLocator(nbins=6, integer=True, min_n_ticks=2))
    axes[-1].set_xlabel("Generation")
    return axes


def _no_parameter_changed(fig, generations):
    ax = fig.axes[0] if fig.axes else fig.add_axes((0.1, 0.15, 0.85, 0.8))
    ax.set_xlim(1, max(generations, 2))
    ax.set_yticks([])
    ax.text(0.5, 0.5, "No parameter changed during the run\n(Advanced settings → Adaptive control)",
            transform=ax.transAxes, ha="center", va="center", fontsize=10, color=style.INK)


def _choice_values(values, categories):
    """Names as positions on the y axis (categories: the names in order of first use)."""
    for v in values:
        if v and v not in categories:
            categories.append(v)
    return np.array([categories.index(v) if v else np.nan for v in values], dtype=float)


CHOICE_MARGIN = 118   # left margin (pixels) when a panel shows operator names


def _choice_axis(ax, categories):
    ax.set_yticks(range(len(categories)))
    ax.set_yticklabels([_short_label(c, 18) for c in categories], fontsize=7.5)
    ax.set_ylim(-0.5, len(categories) - 0.5)
    ax.grid(axis="y", visible=False)


def _param_axis(ax, key):
    if key in PARAM_CHOICES:   # the names take the room of a y label: the label goes inside the panel
        ax.text(0.005, 0.97, PARAM_LABELS.get(key, key), transform=ax.transAxes, ha="left", va="top",
                fontsize=8, color=style.MUTED)
        return
    if key in PARAM_INTEGER:
        ax.yaxis.set_major_locator(MaxNLocator(nbins=4, integer=True))
    else:
        ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
    ax.set_ylabel(PARAM_LABELS.get(key, key), labelpad=2)


def _render_params(dyn, xs, generations, size):
    """Values of the parameters a parameter control changed, one panel per parameter."""
    series = {key: [(d.get("params") or {}).get(key) for d in dyn] for key in PARAM_LABELS}
    changed = [key for key, values in series.items() if len({v for v in values if v is not None}) > 1]
    width, height = max(int(size[0]), 240), max(int(size[1]), 180)
    fig = Figure(figsize=(width / 100, height / 100), dpi=100, facecolor=_panel())
    if not changed:
        axes = _stacked_axes(fig, 1, size)
        _no_parameter_changed(fig, generations)
        return _png(fig)
    keys = changed[:MAX_PARAM_PANELS]
    axes = _stacked_axes(fig, len(keys), size,
                         CHOICE_MARGIN if PARAM_CHOICES & set(keys) else MARGIN_LEFT_FIT)
    for ax, key in zip(axes, keys):
        if key in PARAM_CHOICES:
            categories = []
            values = _choice_values(series[key], categories)
            ax.plot(xs, values, color=style.ACCENT, linewidth=1.8, drawstyle="steps-post")
            _choice_axis(ax, categories)
        else:
            values = np.array([np.nan if v is None else v for v in series[key]], dtype=float)
            ax.plot(xs, values, color=style.ACCENT, linewidth=1.8, drawstyle="steps-post")
        _param_axis(ax, key)
    axes[0].set_xlim(1, max(generations, 2))
    return _png(fig)


def _percent_axis(ax):
    ax.set_ylim(-2, 102)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0f}%"))


def _style_legend(ax, loc="best"):
    leg = ax.legend(loc=loc, fontsize=7.5)
    for text in leg.get_texts():
        text.set_color(style.INK)
    return leg


def _band(ax, xs, rows, lo, hi, color, alpha, label):
    a = np.array([r[lo] for r in rows], dtype=float)
    b = np.array([r[hi] for r in rows], dtype=float)
    ax.fill_between(xs, a, b, color=color, alpha=alpha, linewidth=0, label=label)


def render_dynamics(view, snapshot, generations, size=DEFAULT_SIZE, kind="sr"):
    """Population-level views over the generations (all follow the generation slider)."""
    with _lock, matplotlib.rc_context(_rc()):
        dyn = (snapshot.dynamics or []) if snapshot is not None else []
        xs = np.arange(1, len(dyn) + 1)
        if view == "size":
            return _render_size(dyn, xs, generations, size)
        if view == "params":
            return _render_params(dyn, xs, generations, size)
        fig, ax = _figure(*size, MARGIN_LEFT_FIT)
        ax.set_xlim(1, max(generations, 2))
        ax.xaxis.set_major_locator(MaxNLocator(nbins=6, integer=True, min_n_ticks=2))
        ax.set_xlabel("Generation")
        if not dyn:
            if view != "fitness":
                _percent_axis(ax)
            return _png(fig)

        if view == "fitness":
            rows = [d["fitness"] for d in dyn]
            _band(ax, xs, rows, 0, 4, style.TEAL, 0.14, "10th–90th percentile")
            _band(ax, xs, rows, 1, 3, style.TEAL, 0.30, "IQR (25th–75th percentile)")
            ax.plot(xs, [r[2] for r in rows], color=style.TEAL, linewidth=1.8, label="median")
            best = snapshot.best_training[:len(dyn)]
            ax.plot(xs, best, color=style.ACCENT, linewidth=1.6, label="best")
            if kind == "sr":
                values = np.array([v for r in rows for v in r] + list(best), dtype=float)
                if np.all(values[np.isfinite(values)] > 0):
                    ax.set_yscale("log")
                ax.yaxis.set_major_formatter(FuncFormatter(_short_number))
            else:
                _percent_axis(ax)
            _style_legend(ax, "upper right" if kind == "sr" else "lower right")
        elif view == "diversity":
            ax.plot(xs, [d["genotypic"] for d in dyn], color=style.TEAL, linewidth=1.8,
                    label="genotypic: different trees")
            ax.plot(xs, [d["phenotypic"] for d in dyn], color=style.ACCENT, linewidth=1.8,
                    label="phenotypic: different error vectors")
            _percent_axis(ax)
            _style_legend(ax, "lower left")
        elif view == "selection":
            gx = [x for x, d in zip(xs, dyn) if d["parents"] is not None]
            ax.plot(gx, [d["parents"] for d in dyn if d["parents"] is not None], color=style.TEAL,
                    linewidth=1.8, label="individuals selected at least once")
            ax.plot(gx, [d["top_parent"] for d in dyn if d["top_parent"] is not None], color=style.ACCENT,
                    linewidth=1.8, label="selections won by the most selected one")
            _percent_axis(ax)
            _style_legend(ax, "upper right")
        elif view == "variation":
            gx = [x for x, d in zip(xs, dyn) if d["variation"]]
            vs = [d["variation"] for d in dyn if d["variation"]]

            def share(op, k):
                return [100.0 * v[op][k] / sum(v[op]) if sum(v[op]) else np.nan for v in vs]
            ax.plot(gx, share("cx", 0), color=style.TEAL, linewidth=1.8, label="crossover: better than parent")
            ax.plot(gx, share("cx", 2), color=style.TEAL, linewidth=1.4, linestyle=(0, (4, 2.5)),
                    label="crossover: worse")
            ax.plot(gx, share("mut", 0), color=style.ACCENT, linewidth=1.8, label="mutation: better")
            ax.plot(gx, share("mut", 2), color=style.ACCENT, linewidth=1.4, linestyle=(0, (4, 2.5)),
                    label="mutation: worse")
            _percent_axis(ax)
            _style_legend(ax, "upper right")
        elif view == "blocks":
            totals = Counter()
            for d in dyn:
                totals.update(d["blocks"])
            top = sorted(sorted(totals, key=totals.get, reverse=True)[:len(style.CATEGORICAL) - 1])
            series = [[100.0 * d["blocks"].get(name, 0.0) for d in dyn] for name in top]
            other = [100.0 - sum(col) for col in zip(*series)] if series else [100.0] * len(dyn)
            colors = style.CATEGORICAL[:len(top)] + [style.RULE]
            ax.stackplot(xs, *series, other, colors=colors, labels=top + ["other"],
                         edgecolor=_panel(), linewidth=0.6)
            ax.set_ylim(0, 100)
            ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0f}%"))
            leg = ax.legend(loc="upper left", fontsize=7, ncol=2, framealpha=0.95)
            for text in leg.get_texts():
                text.set_color(style.INK)
                text.set_fontfamily("monospace")
        return _png(fig)


def _render_size(dyn, xs, generations, size):
    """Two small panels instead of two y-axes: tree size (nodes) on top, depth below."""
    width, height = max(int(size[0]), 240), max(int(size[1]), 180)
    fig = Figure(figsize=(width / 100, height / 100), dpi=100, facecolor=_panel())
    left, right = MARGIN_LEFT_FIT / width, 1 - MARGIN_RIGHT / width
    top, bottom = 1 - MARGIN_TOP / height, MARGIN_BOTTOM / height
    gap = 18 / height
    mid = bottom + (top - bottom - gap) / 2
    ax1 = fig.add_axes((left, mid + gap, right - left, top - mid - gap))
    ax2 = fig.add_axes((left, bottom, right - left, mid - bottom), sharex=ax1)
    for ax, key, name, color in ((ax1, "size", "nodes", style.ACCENT), (ax2, "depth", "depth", style.TEAL)):
        ax.set_facecolor(_panel())
        ax.set_axisbelow(True)
        if dyn:
            rows = [d[key] for d in dyn]
            _band(ax, xs, rows, 1, 3, color, 0.25, "IQR")
            ax.plot(xs, [r[2] for r in rows], color=color, linewidth=1.8, label="median")
            ax.plot(xs, [r[5] for r in rows], color=color, linewidth=1.1, linestyle=(0, (3, 2)), label="maximum")
            leg = _style_legend(ax, "upper left")
            leg.set_title(name, prop={"size": 8, "weight": "bold"})
        ax.yaxis.set_major_locator(MaxNLocator(nbins=4, integer=True))
    ax1.set_xlim(1, max(generations, 2))
    ax1.tick_params(labelbottom=False)
    ax2.xaxis.set_major_locator(MaxNLocator(nbins=6, integer=True, min_n_ticks=2))
    ax2.set_xlabel("Generation")
    return _png(fig)


# --------------------------------------------------------------------------
# Experiments (repeated runs)
# --------------------------------------------------------------------------

def _short_label(text, n=16):
    return text if len(text) <= n else text[:n - 1] + "…"


ERROR_FLOOR = 1e-10   # errors below this count as exact; they would squash a log axis


def _floor_axis(ax, values):
    """Log axis for errors over several decades; exact solutions sit on the floor."""
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) and np.all(values > 0) and values.max() / values.min() > 50:
        ax.set_yscale("log")
        if values.min() <= ERROR_FLOOR * 1.0001:
            ax.set_ylim(bottom=ERROR_FLOOR / 3)
    ax.yaxis.set_major_formatter(FuncFormatter(
        lambda v, _: "≤1e-10" if abs(v - ERROR_FLOOR) < ERROR_FLOOR * 1e-6 else _short_number(v)))




def render_experiment_final(groups, labels, kind, size=DEFAULT_SIZE):
    """Final values of all runs per configuration: box plot plus the single runs;
    for program synthesis and logic additionally the success rate (95 % Wilson CI)."""
    import experiments
    with _lock, matplotlib.rc_context(_rc()):
        width, height = max(int(size[0]), 240), max(int(size[1]), 180)
        fig = Figure(figsize=(width / 100, height / 100), dpi=100, facecolor=_panel())
        left, right = MARGIN_LEFT_FIT / width, 1 - MARGIN_RIGHT / width
        top, bottom = 1 - MARGIN_TOP / height, MARGIN_BOTTOM / height
        n = max(len(labels), 1)
        xs = np.arange(1, n + 1)
        if kind == "sr":
            axes = [fig.add_axes((left, bottom, right - left, top - bottom))]
        else:
            gap = 18 / height
            mid = bottom + (top - bottom - gap) * 0.42
            ax_rate = fig.add_axes((left, mid + gap, right - left, top - mid - gap))
            axes = [fig.add_axes((left, bottom, right - left, mid - bottom), sharex=ax_rate), ax_rate]
        for ax in axes:
            ax.set_facecolor(_panel())
            ax.set_axisbelow(True)
            ax.set_xlim(0.4, n + 0.6)
            ax.grid(axis="x", visible=False)
        ax = axes[0]
        rng = np.random.default_rng(0)
        all_values = []
        for i, (g, label) in enumerate(zip(groups, labels)):
            color = style.CATEGORICAL[i % len(style.CATEGORICAL)]
            v = experiments.final_values(g, kind)
            v = v[np.isfinite(v)]
            if not len(v):
                continue
            if kind == "sr":
                v = np.maximum(v, ERROR_FLOOR)
            all_values.extend(v)
            ax.boxplot([v], positions=[i + 1], widths=0.5, showfliers=False, patch_artist=True,
                       boxprops=dict(facecolor=matplotlib.colors.to_rgba(color, 0.18), edgecolor=color, linewidth=1.2),
                       medianprops=dict(color=color, linewidth=2), whiskerprops=dict(color=color, linewidth=1),
                       capprops=dict(color=color, linewidth=1))
            jitter = rng.uniform(-0.16, 0.16, len(v))
            ax.scatter(i + 1 + jitter, v, s=11, color=color, alpha=0.75, linewidths=0.4, edgecolors=_panel(),
                       zorder=3)
        if kind == "sr":
            _floor_axis(ax, all_values)
        else:
            _percent_axis(ax)
            ax.yaxis.set_major_locator(MaxNLocator(nbins=3))
            ax.set_ylabel("solved", labelpad=2)
            rate_ax = axes[1]
            for i, g in enumerate(groups):
                if not g:
                    continue
                color = style.CATEGORICAL[i % len(style.CATEGORICAL)]
                s = sum(1 for r in g if r["success"])
                lo, hi = experiments.wilson(s, len(g))
                p = 100.0 * s / len(g)
                rate_ax.bar(i + 1, p, width=0.5, color=matplotlib.colors.to_rgba(color, 0.75), edgecolor=color,
                            linewidth=0)
                rate_ax.errorbar(i + 1, p, yerr=[[max(0.0, p - 100 * lo)], [max(0.0, 100 * hi - p)]], color=style.INK, linewidth=1,
                                 capsize=4)
            _percent_axis(rate_ax)
            rate_ax.set_ylim(0, 104)
            rate_ax.yaxis.set_major_locator(MaxNLocator(nbins=3))
            rate_ax.set_ylabel("success", labelpad=2)
            rate_ax.tick_params(labelbottom=False)
        ax.set_xticks(xs)
        ax.set_xticklabels([_short_label(l) for l in labels] if labels else [""])
        return _png(fig)


def render_experiment_convergence(groups, labels, kind, generations, size=DEFAULT_SIZE):
    """Median of the compared value over the generations, with the IQR band."""
    with _lock, matplotlib.rc_context(_rc()):
        fig, ax = _figure(*size, MARGIN_LEFT_FIT)
        ax.set_xlim(1, max(generations, 2))
        ax.xaxis.set_major_locator(MaxNLocator(nbins=6, integer=True, min_n_ticks=2))
        ax.set_xlabel("Generation")
        import experiments
        key = experiments.success_column(kind)
        all_rows = []
        drawn = False
        for i, (g, label) in enumerate(zip(groups, labels)):
            rows = experiments.series_matrix(g, key, generations)
            if not len(rows):
                continue
            if kind == "sr":
                rows = np.maximum(rows, ERROR_FLOOR)
                all_rows.append(rows.ravel())
            color = style.CATEGORICAL[i % len(style.CATEGORICAL)]
            xs = np.arange(1, rows.shape[1] + 1)
            with np.errstate(all="ignore"), warnings_ignored():
                q1, med, q3 = np.nanpercentile(rows, [25, 50, 75], axis=0)
            ax.fill_between(xs, q1, q3, color=color, alpha=0.16, linewidth=0)
            ax.plot(xs, med, color=color, linewidth=1.8, label=f"{_short_label(label, 28)}  (n = {len(rows)})")
            drawn = True
        if kind == "sr":
            _floor_axis(ax, np.concatenate(all_rows) if all_rows else [])
        else:
            _percent_axis(ax)
        if drawn:
            _style_legend(ax, "lower left" if kind == "sr" else "lower right")
        return _png(fig)


# Population dynamics over repeated runs: per run one value per generation (e.g. the
# median tree size of the population), then the median over the runs with the IQR band.
# Two quantities of one view get two stacked panels, never two y-axes.
EXPERIMENT_DYNAMICS = {
    "fitness": [("best_training", "best"), ("fitness_median", "pop. median")],
    "size": [("size_median", "nodes"), ("depth_median", "depth")],
    "diversity": [("genotypic_diversity", "genotypic"), ("phenotypic_diversity", "phenotypic")],
    "selection": [("selected_at_least_once", "selected"), ("most_selected_share", "top parent")],
    "variation": [("crossover_better", "crossover"), ("mutation_better", "mutation")],
}
EXPERIMENT_DYNAMICS_TITLES = {
    "fitness": "Fitness: best and population median",
    "size": "Tree size & depth (population median)",
    "diversity": "Diversity: genotypic and phenotypic",
    "selection": "Selection: selected at least once, top parent",
    "variation": "Variation: children better than their parent",
    "blocks": "Building-block usage (mean over generations and runs)",
    "params": "Parameters over generations (adaptive control)",
}


def render_experiment_dynamics(view, groups, labels, kind, generations, size=DEFAULT_SIZE):
    import experiments
    with _lock, matplotlib.rc_context(_rc()):
        if view == "blocks":
            return _render_experiment_blocks(groups, labels, size)
        width, height = max(int(size[0]), 240), max(int(size[1]), 180)
        fig = Figure(figsize=(width / 100, height / 100), dpi=100, facecolor=_panel())
        if view == "params":   # the parameters some run changed, one panel each
            keys = []
            for key in PARAM_LABELS:
                for g in groups:
                    for r in g:
                        values = experiments.run_series(r).get(f"param_{key}", [])
                        distinct = {v for v in values if v not in ("", None) and not (isinstance(v, float)
                                                                                      and math.isnan(v))}
                        if len(distinct) > 1 and key not in keys:
                            keys.append(key)
            if not keys:
                _stacked_axes(fig, 1, size)
                _no_parameter_changed(fig, generations)
                return _png(fig)
            panels = [(f"param_{key}", key) for key in keys[:MAX_PARAM_PANELS]]
        else:
            panels = EXPERIMENT_DYNAMICS[view]
        choice_panel = view == "params" and any(name in PARAM_CHOICES for _, name in panels)
        axes = _stacked_axes(fig, len(panels), size, CHOICE_MARGIN if choice_panel else MARGIN_LEFT_FIT)
        ax1 = axes[0]
        percent = view in ("diversity", "selection", "variation") or (view == "fitness" and kind != "sr")
        log = view == "fitness" and kind == "sr"
        for ax, (column, name) in zip(axes, panels):
            if view == "params" and name in PARAM_CHOICES:
                # names: per configuration the operator most runs use in each generation
                categories = []
                for i, (g, label) in enumerate(zip(groups, labels)):
                    runs = [experiments.run_series(r)[column] for r in g]
                    if not runs:
                        continue
                    mode = []
                    for gi in range(generations):
                        used = [run[gi] for run in runs if gi < len(run) and run[gi]]
                        mode.append(Counter(used).most_common(1)[0][0] if used else "")
                    ax.plot(np.arange(1, generations + 1), _choice_values(mode, categories),
                            color=style.CATEGORICAL[i % len(style.CATEGORICAL)], linewidth=1.8,
                            drawstyle="steps-post", label=_short_label(label, 28))
                _choice_axis(ax, categories or [""])
                _param_axis(ax, name)
                continue
            values = []
            for i, (g, label) in enumerate(zip(groups, labels)):
                rows = experiments.series_matrix(g, column, generations)
                if not len(rows) or not np.isfinite(rows).any():
                    continue
                if log:
                    rows = np.maximum(rows, ERROR_FLOOR)
                color = style.CATEGORICAL[i % len(style.CATEGORICAL)]
                xs = np.arange(1, rows.shape[1] + 1)
                with np.errstate(all="ignore"), warnings_ignored():
                    q1, med, q3 = np.nanpercentile(rows, [25, 50, 75], axis=0)
                values.extend(rows[np.isfinite(rows)])
                steps = "steps-post" if view == "params" else "default"   # parameters change in steps
                ax.fill_between(xs, q1, q3, color=color, alpha=0.16, linewidth=0, step="post" if view == "params" else None)
                ax.plot(xs, med, color=color, linewidth=1.8, label=_short_label(label, 28), drawstyle=steps)
            if view == "params":
                _param_axis(ax, name)
                continue
            if log:
                _floor_axis(ax, values)
            elif percent:
                _percent_axis(ax)
                ax.yaxis.set_major_locator(MaxNLocator(nbins=3))
            else:
                ax.yaxis.set_major_locator(MaxNLocator(nbins=4, integer=True))
            ax.set_ylabel(name, labelpad=2)
        if ax1.get_legend_handles_labels()[0]:
            _style_legend(ax1, "lower right" if view == "fitness" and kind != "sr" else "best")
        ax1.set_xlim(1, max(generations, 2))
        return _png(fig)


def _render_experiment_blocks(groups, labels, size):
    """Share of the nodes per building block, one bar per configuration (mean over generations and runs)."""
    import experiments
    fig, ax = _figure(*size, MARGIN_LEFT_FIT)
    ax.grid(axis="x", visible=False)
    names = experiments.block_names(groups)
    shares = []
    for g in groups:
        per_run = [[np.mean(experiments.run_series(r).get(f"block_{n}", [0.0]) or [0.0]) for n in names] for r in g]
        shares.append(np.mean(per_run, axis=0) if per_run else np.zeros(len(names)))
    if names and shares:
        totals = np.sum(shares, axis=0)
        top = sorted(np.argsort(totals)[::-1][:8], key=lambda k: names[k])
        k = len(groups)
        w = 0.8 / max(k, 1)
        xs = np.arange(len(top))
        for i, row in enumerate(shares):
            color = style.CATEGORICAL[i % len(style.CATEGORICAL)]
            ax.bar(xs - 0.4 + w * (i + 0.5), [row[j] for j in top], width=w * 0.88, color=color, linewidth=0,
                   label=_short_label(labels[i], 28))
        ax.set_xticks(xs)
        ax.set_xticklabels([_short_label(SYMBOLS.get(names[j], names[j]), 10) for j in top], fontfamily="monospace")
        ax.set_xlim(-0.6, len(top) - 0.4)
        ax.set_ylim(0, max(5.0, float(np.max([[row[j] for j in top] for row in shares])) * 1.18))
        _style_legend(ax, "upper right")
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0f}%"))
    return _png(fig)


# --------------------------------------------------------------------------
# Experiment report (PDF with all plots)
# --------------------------------------------------------------------------

EXPERIMENT_CAPTIONS = {
    "fitness": "Top: training value of the best individual, bottom: median of the population. Per run one value "
               "per generation; line: median over the runs, band: IQR over the runs.",
    "size": "Median number of nodes (top) and median depth (bottom) of the trees in the population; line: median "
            "over the runs, band: IQR over the runs.",
    "diversity": "Share of different trees (genotypic, top) and of different error vectors (phenotypic, bottom) "
                 "in the population; median over the runs, band: IQR.",
    "selection": "Share of the population selected at least once as a parent (top) and share of all selections won "
                 "by the most selected individual (bottom); median over the runs.",
    "variation": "Share of crossover children (top) and mutation children (bottom) that are better than their "
                 "parent; median over the runs, band: IQR.",
    "blocks": "Share of all nodes in the population per building block, averaged over all generations and runs.",
    "params": "Values of the parameters changed by the adaptive control in each generation; line: median over "
              "the runs, band: IQR (operators: the one most runs use).",
}
REPORT_SIZE = (11.69, 8.27)   # A4 landscape, inches
REPORT_PLOT = (514, 282)      # one plot, 1 px = 1/100 inch: its size on the page (fonts at their real size)


def experiment_report_plots(kind, groups, labels, generations):
    """(title, caption, png) of every plot of one problem, in report order (white, 500 dpi)."""
    with print_style():
        return _report_plots(kind, groups, labels, generations)


def _report_plots(kind, groups, labels, generations):
    import experiments
    metric, higher = experiments.primary_metric(kind)
    better = "higher is better" if higher else "lower is better"
    out = []
    if kind == "sr":
        out.append((f"Final {metric}",
                    f"Best individual of each run after the last generation (one dot per run); box: IQR of the "
                    f"runs, line: median; {better}, ≤1e-10 counts as exact.",
                    render_experiment_final(groups, labels, kind, REPORT_PLOT)))
    else:
        out.append((f"Success rate and {metric}",
                    f"Top: share of runs whose best program is perfect, with 95 % Wilson interval. Bottom: {metric} "
                    f"of each run's best program (one dot per run; box: IQR, line: median).",
                    render_experiment_final(groups, labels, kind, REPORT_PLOT)))
    out.append((f"Median {metric} over the generations",
                f"Best individual of each run; line: median over the runs, band: IQR over the runs; {better}.",
                render_experiment_convergence(groups, labels, kind, generations, REPORT_PLOT)))
    for view in DYNAMICS_VIEWS:
        out.append((EXPERIMENT_DYNAMICS_TITLES[view], EXPERIMENT_CAPTIONS[view],
                    render_experiment_dynamics(view, groups, labels, kind, generations, REPORT_PLOT)))
    return out


ZIP_PLOT = (800, 480)   # pixels at 100 dpi, i.e. 8 x 4.8 inches


def experiment_plots_zip(sets, labels, generations):
    """ZIP with every experiment plot of every problem as PNG (300 dpi) and PDF (vector), white
    background as in the report. sets: list of (problem name, kind, groups)."""
    import re
    import zipfile
    views = [("final", None), ("convergence", None)] + [(v, v) for v in DYNAMICS_VIEWS]
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for name, kind, groups in sets:
            folder = re.sub(r"[^A-Za-z0-9_-]+", "-", name).strip("-")[:40] or "problem"
            for k, (key, view) in enumerate(views, 1):
                for fmt in ("png", "pdf"):
                    with print_style(scale=3), output_format(fmt):
                        if key == "final":
                            data = render_experiment_final(groups, labels, kind, ZIP_PLOT)
                        elif key == "convergence":
                            data = render_experiment_convergence(groups, labels, kind, generations, ZIP_PLOT)
                        else:
                            data = render_experiment_dynamics(view, groups, labels, kind, generations, ZIP_PLOT)
                    zf.writestr(f"{folder}/{k:02d}_{key}.{fmt}", data)
    return buf.getvalue()


def render_experiment_report(overview, sections):
    """PDF: overview page(s), then per problem its plots, four per page.

    overview: list of (style, text[, color]) lines, style in h1, h2, text, mono, swatch
    sections: list of (heading, subheading, [(title, caption, png), ...])"""
    import textwrap
    import matplotlib.image as mpimg
    from matplotlib.backends.backend_pdf import PdfPages
    W, H = REPORT_SIZE
    buf = io.BytesIO()
    with _lock, matplotlib.rc_context(_rc()), PdfPages(buf, metadata={"Title": "GPExplore experiment",
                                                                     "Creator": "GPExplore"}) as pdf:
        def page():
            return Figure(figsize=(W, H), facecolor="white")

        fig, y = page(), 0.93
        steps = {"h1": 0.06, "h2": 0.045, "text": 0.03, "mono": 0.027, "swatch": 0.03}
        for line in overview:
            kind, text = line[0], line[1]
            if y - steps[kind] < 0.05:
                pdf.savefig(fig)
                fig, y = page(), 0.93
            if kind == "h1":
                fig.text(0.05, y, text, fontsize=18, fontweight="bold", color=style.INK, va="top")
            elif kind == "h2":
                y -= 0.012
                fig.text(0.05, y, text, fontsize=11.5, fontweight="bold", color=style.INK, va="top")
            elif kind == "swatch":
                fig.patches.append(matplotlib.patches.Rectangle((0.05, y - 0.017), 0.009, 0.013,
                                                                transform=fig.transFigure, color=line[2]))
                fig.text(0.065, y, text, fontsize=9, color=style.INK, va="top", fontfamily="monospace")
            else:
                fig.text(0.05, y, text, fontsize=9 if kind == "text" else 8.5, va="top",
                         color=line[2] if len(line) > 2 else style.INK,
                         fontfamily="monospace" if kind == "mono" else None)
            y -= steps[kind]
        pdf.savefig(fig)

        for heading, subheading, items in sections:
            pages = max(1, math.ceil(len(items) / 4))
            for pi in range(pages):
                fig = page()
                fig.text(0.04, 0.965, heading, fontsize=14, fontweight="bold", color=style.INK, va="top")
                fig.text(0.04, 0.932, subheading, fontsize=9, color=style.MUTED, va="top")
                fig.text(0.96, 0.965, f"{pi + 1} / {pages}", fontsize=9, color=style.MUTED, va="top", ha="right")
                for k, (title, caption, png) in enumerate(items[4 * pi:4 * pi + 4]):
                    x0 = 0.04 + (k % 2) * 0.48
                    top = 0.885 - (k // 2) * 0.445
                    fig.text(x0, top, title, fontsize=10.5, fontweight="bold", color=style.INK, va="top")
                    fig.text(x0, top - 0.027, textwrap.fill(caption, 98), fontsize=7.5, color=style.INK, va="top",
                             linespacing=1.3)
                    img = mpimg.imread(io.BytesIO(png))
                    w_frac = 0.44
                    h_frac = w_frac * W / H * img.shape[0] / img.shape[1]
                    ax = fig.add_axes((x0, top - 0.072 - h_frac, w_frac, h_frac))
                    ax.imshow(img, interpolation="none")   # embedded as is: no resampling
                    ax.set_axis_off()
                pdf.savefig(fig)
    return buf.getvalue()


@contextlib.contextmanager
def warnings_ignored():
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        yield


# --------------------------------------------------------------------------
# Expression tree
# --------------------------------------------------------------------------

# --------------------------------------------------------------------------
# Ancestry (genealogy of one individual)
# --------------------------------------------------------------------------

ANCESTRY_OPS = {   # operator -> (label, colour)
    "cx": ("crossover", style.TEAL),
    "mut": ("mutation", style.ACCENT),
    "cx+mut": ("crossover + mutation", style.CATEGORICAL[6]),
    "copy": ("selected unchanged", style.FAINT),
    "elite": ("elite", style.INK),
    "init": ("initial population", style.RULE),
}


def render_ancestry(nodes, root, kind="sr"):
    """Ancestry graph of one individual (as in Burlacu et al. 2013, McPhee et al. 2017): one column per
    generation, the individual on the right, its parents and their parents to the left; lines connect
    parents to their children. A node shows the individual's fitness (training error, or % of training
    cases solved); its colour says how it was created. nodes: {id: (generation, operator,
    parent ids, fitness, size, expression)}. Returns (png, width, height) like render_tree."""
    with _lock, matplotlib.rc_context(_rc()):
        gens = sorted({rec[0] for rec in nodes.values()})
        # column order: breadth-first from the root, so parents stay near their children
        columns, seen, frontier = {g: [] for g in gens}, set(), [root]
        while frontier:
            nxt = []
            for u in frontier:
                if u in seen or u not in nodes:
                    continue
                seen.add(u)
                columns[nodes[u][0]].append(u)
                nxt.extend(nodes[u][2])
            frontier = nxt
        # fewer crossing lines: from the newest generation backwards, order each column by the mean
        # position of its children in the column to its right (barycentre heuristic)
        children = {}
        for u, rec in nodes.items():
            for parent in rec[2]:
                children.setdefault(parent, []).append(u)
        rank = {u: i for i, u in enumerate(columns[gens[-1]])}
        for g in reversed(gens[:-1]):
            col = columns[g]
            key = {u: np.mean([rank[c] for c in children.get(u, []) if c in rank] or [len(col)]) for u in col}
            columns[g] = sorted(col, key=lambda u: key[u])
            rank.update({u: i * max(1, len(columns[gens[-1]])) / max(1, len(col)) for i, u in enumerate(columns[g])})
        rows = max([len(c) for c in columns.values()] + [1])
        dx, dy = 1.25, 0.62
        width_in = max(6.0, len(gens) * dx + 1.4)
        height_in = max(3.4, rows * dy + 1.6)
        fig = Figure(figsize=(width_in, height_in), dpi=100, facecolor=_panel())
        ax = fig.add_axes((0, 0, 1, 1))
        ax.set_axis_off()
        ax.set_xlim(-0.9, width_in - 0.9)
        ax.set_ylim(height_in - 1.05, -0.65)
        xy = {}
        for k, g in enumerate(gens):
            col = columns[g]
            top = (rows - len(col)) * dy / 2
            for j, u in enumerate(col):
                xy[u] = (k * dx, top + j * dy)
        for u, (x, y) in xy.items():   # lines first, under the nodes
            for parent in nodes[u][2]:
                if parent in xy:
                    px, py = xy[parent]
                    ax.plot([px, x], [py, y], color=style.RULE, linewidth=1.0, zorder=1, solid_capstyle="round")
        for u, (x, y) in xy.items():
            generation, op, _, fit, size, _ = nodes[u]
            label, color = ANCESTRY_OPS.get(op, (op, style.FAINT))
            value = (("0" if fit < ERROR_FLOOR else f"{fit:.3g}") if kind == "sr" else f"{fit:.0f} %")
            dark = op in ("cx", "mut", "cx+mut", "elite")
            is_root = u == root
            ax.text(x, y, value, ha="center", va="center", zorder=3, fontsize=8,
                    fontfamily="monospace", color="#FFFFFF" if dark else style.INK,
                    fontweight="bold" if is_root else "normal",
                    bbox=dict(boxstyle="round,pad=0.4,rounding_size=0.35", facecolor=color,
                              edgecolor=(style.INK if color == style.ACCENT else style.ACCENT) if is_root else color,
                              linewidth=2.2 if is_root else 0.8))   # the frame must differ from the fill
        for k, g in enumerate(gens):
            ax.text(k * dx, rows * dy - dy / 2 + 0.28, f"gen {g}", ha="center", va="top", fontsize=8.5,
                    color=style.MUTED)
        handles = [matplotlib.patches.Patch(facecolor=c, edgecolor=c, label=l)
                   for op_, (l, c) in ANCESTRY_OPS.items() if any(rec[1] == op_ for rec in nodes.values())]
        legend = fig.legend(handles=handles, loc="upper left", ncol=len(handles), fontsize=8, frameon=False,
                            handlelength=1.2, columnspacing=1.2, bbox_to_anchor=(0.01, 0.995))
        for text in legend.get_texts():
            text.set_color(style.INK)
        w_px, h_px = int(round(width_in * 100)), int(round(height_in * 100))
        return _png(fig), w_px, h_px


SYMBOLS = {"add": "+", "sub": "−", "mul": "×", "protectedDiv": "÷", "analyticQuotient": "aq",
           "neg": "neg", "sin": "sin", "cos": "cos"}
TREE_DX, TREE_DY, TREE_PAD = 0.62, 0.6, 0.45   # inches between leaves / levels, border
MAX_TREE_INCHES = 60


def _layout(nodes):
    """Tidy layout of a prefix-ordered tree: leaves get consecutive x slots,
    every parent sits centred above its children."""
    pos, edges = {}, []
    next_leaf = 0
    open_nodes = []   # [index, depth, children so far] of the inner nodes not complete yet
    for i, (_, arity, _) in enumerate(nodes):   # prefix order, without recursion (deep trees)
        depth = len(open_nodes)
        if arity:
            open_nodes.append([i, depth, []])
            continue
        pos[i] = (next_leaf, depth)
        next_leaf += 1
        done = i
        while open_nodes:   # parents whose last child is now placed sit centred above them
            open_nodes[-1][2].append(done)
            parent, p_depth, children = open_nodes[-1]
            if len(children) < nodes[parent][1]:
                break
            open_nodes.pop()
            pos[parent] = (sum(pos[c][0] for c in children) / len(children), p_depth)
            edges.extend((parent, c) for c in children)
            done = parent
        if not open_nodes and len(pos) == len(nodes):
            break
    return pos, edges, next_leaf, max(d for _, d in pos.values())


def render_tree(nodes):
    """nodes: prefix list of (label, arity, kind) with kind in {"function", "variable", "constant"}.
    Returns (png bytes, width px, height px) at the logical (1x) size."""
    with _lock, matplotlib.rc_context(_rc()):
        pos, edges, leaves, depth = _layout(nodes)
        width = max(leaves - 1, 1) * TREE_DX + 2 * TREE_PAD
        height = depth * TREE_DY + 2 * TREE_PAD
        shrink = min(1.0, MAX_TREE_INCHES / max(width, height))
        fig = Figure(figsize=(width * shrink, height * shrink), dpi=100, facecolor=_panel())
        ax = fig.add_axes((0, 0, 1, 1))
        ax.set_axis_off()
        ax.set_xlim(-TREE_PAD, width - TREE_PAD)
        ax.set_ylim(height - TREE_PAD, -TREE_PAD)
        xy = {i: (x * TREE_DX, d * TREE_DY) for i, (x, d) in pos.items()}
        for a, b in edges:
            (x0, y0), (x1, y1) = xy[a], xy[b]
            ax.plot([x0, x1], [y0, y1], color=style.RULE, linewidth=1.1 * shrink, zorder=1,
                    solid_capstyle="round")
        colors = {"function": (style.INK, "#FFFFFF"), "variable": (style.TEAL, "#FFFFFF"),
                  "constant": (_panel(), style.INK)}
        for i, (label, arity, kind) in enumerate(nodes):
            face, text = colors[kind]
            x, y = xy[i]
            label = SYMBOLS.get(label, label)
            ax.text(x, y, label, ha="center", va="center", zorder=2, color=text,
                    fontsize=(11 if len(label) <= 2 else 9) * shrink,
                    fontfamily="monospace", fontweight="bold" if kind == "function" else "normal",
                    bbox=dict(boxstyle="round,pad=0.38,rounding_size=0.5", facecolor=face,
                              edgecolor=style.INK if kind == "constant" else face, linewidth=0.8 * shrink))
        w_px, h_px = int(round(fig.get_figwidth() * 100)), int(round(fig.get_figheight() * 100))
        return _png(fig), w_px, h_px
