import asyncio
import math
import multiprocessing
import os
import time
import warnings
from concurrent.futures import ProcessPoolExecutor

import flet as ft
import numpy as np

import clipboard_image
import config
import datasets
import experiments
import fast
import gp
import plots
import problems
import style as S
import surface_workers
import operators
from code_editor import CodeEditor
import synthesis
import webserver

warnings.filterwarnings("ignore")

FUNCTION_ROWS = [
    ("add(a, b)", "Function", "Adds two numbers.", "add_func"),
    ("sub(a, b)", "Function", "Subtracts two numbers.", "sub_func"),
    ("mul(a, b)", "Function", "Multiplies two numbers.", "mul_func"),
    ("protectedDiv(a, b)", "Function", "Protected division of two numbers. If b equals 0 then 1 is returned.", "protectedDiv_func"),
    ("analyticQuotient(a, b)", "Function", "Divides two numbers using the formula a / sqrt(1 + b**2).", "analyticQuotient_func"),
    ("neg(a)", "Function", "Negates the value of a.", "neg_func"),
    ("sin(a)", "Function", "Sine of a.", "sin_func"),
    ("cos(a)", "Function", "Cosine of a.", "cos_func"),
    ("ERC Int [-1, 1]", "Terminal", "Ephemeral random constant of type integer from the interval [-1, 1].", "int101_erc"),
]
FUNCTION_FLAGS = [row[3] for row in FUNCTION_ROWS if row[1] == "Function"]

# --------------------------------------------------------------------------
# Small building blocks
# --------------------------------------------------------------------------

def mono(value="", size=13, color=S.INK, weight=None, **kw):
    return ft.Text(value, font_family=S.MONO, size=size, color=color, weight=weight, **kw)


def caps(value, color=S.MUTED, size=10.5):
    """Small section label (sentence case, semi-bold)."""
    return ft.Text(value, size=size + 2, color=color, weight=ft.FontWeight.W_600,
                   max_lines=1, overflow=ft.TextOverflow.ELLIPSIS)


def card(content, padding=16, expand=None, height=None, bgcolor=S.PANEL):
    return ft.Container(content=content, padding=padding, expand=expand, height=height,
                        bgcolor=bgcolor, border=ft.Border.all(1, S.RULE),
                        border_radius=ft.BorderRadius.all(4))


def link_button(icon, label, on_click):
    """Accent text button with icon (used for 'Custom problem' and 'Show tree')."""
    return ft.TextButton(
        content=ft.Row([ft.Icon(icon, size=15), ft.Text(label, size=12.5)], spacing=4, tight=True),
        on_click=on_click,
        style=ft.ButtonStyle(color={ft.ControlState.DEFAULT: S.ACCENT, ft.ControlState.DISABLED: S.FAINT},
                             padding=ft.Padding.symmetric(horizontal=4, vertical=0),
                             overlay_color=ft.Colors.with_opacity(0.06, S.ACCENT), mouse_cursor=HAND),
        height=28,
    )


# GPX_DRAG_LOG=<file>: timings of each 3D drag (events, frames), for tuning
DRAG_LOG = [] if os.environ.get("GPX_DRAG_LOG") else None
HAND = {ft.ControlState.DISABLED: ft.MouseCursor.BASIC, ft.ControlState.DEFAULT: ft.MouseCursor.CLICK}


def hand(control, enabled=True):
    """Hand cursor over a clickable container (Flet containers have no cursor setting of their own)."""
    return ft.GestureDetector(content=control, expand=control.expand,   # a Row lays out the wrapper, not the control
                              mouse_cursor=ft.MouseCursor.CLICK if enabled else ft.MouseCursor.BASIC)


def splitter(on_drag):
    """Handle between two areas: drag it up or down to change their heights."""
    grip = ft.Container(width=40, height=4, border_radius=2, bgcolor=S.RULE)
    return ft.GestureDetector(
        content=ft.Container(grip, height=14, alignment=ft.Alignment.CENTER),
        mouse_cursor=ft.MouseCursor.RESIZE_ROW, drag_interval=16,
        on_vertical_drag_update=lambda e: on_drag(e.local_delta.y if e.local_delta is not None else 0.0))


def side_by_side(left, right, expand=None, height=None):
    """Two cards next to each other with a handle between them: drag it left or right to
    change their widths (each keeps at least 20 %)."""
    state = {"ratio": 0.5, "width": 1000.0}

    def apply():
        left.expand = max(1, round(1000 * state["ratio"]))
        right.expand = max(1, 1000 - left.expand)

    def drag(dx):
        ratio = min(0.8, max(0.2, state["ratio"] + dx / max(state["width"], 1.0)))
        if ratio != state["ratio"]:
            state["ratio"] = ratio
            apply()
            row.update()

    def sized(e):
        state["width"] = float(e.width) or state["width"]

    grip = ft.Container(width=4, height=40, border_radius=2, bgcolor=S.RULE)
    handle = ft.GestureDetector(
        content=ft.Container(grip, width=16, alignment=ft.Alignment.CENTER),
        mouse_cursor=ft.MouseCursor.RESIZE_COLUMN, drag_interval=16,
        on_horizontal_drag_update=lambda e: drag(e.local_delta.x if e.local_delta is not None else 0.0))
    apply()
    row = ft.Row([left, handle, right], spacing=0, expand=True, vertical_alignment=ft.CrossAxisAlignment.STRETCH)
    return ft.Container(row, expand=expand, height=height, on_size_change=sized, size_change_interval=100)


# Expert mode: the sliders become number fields, limited only by the data type and these
# real limits (None = open); probabilities and rates stay in [0, 1].
HARD_LIMITS = {
    "training_set_size": (1, None), "noise_intensity": (0.0, None), "outlier_share": (0.0, 1.0),
    "validation_set_size": (1, None), "generations": (1, None), "population_size": (1, None),
    "crossover_probability": (0.0, 1.0), "mutation_probability": (0.0, 1.0), "tourn_size": (1, None),
    "parsimony_size": (1.0, 2.0), "lexicase_epsilon": (0.0, None),
    "ds_rate": (0.0, 1.0), "parent_sampling_rate": (0.0, 1.0), "distance_interval": (1, None),
    "outlier_rate": (0.0, 1.0), "min_initial_tree_depth": (0, None), "max_initial_tree_depth": (1, None),
    "max_tree_depth": (1, gp.MAX_DEPTH),
    "inter_gen_pause": (0.0, None), "max_tree_size": (1, None), "elitism": (0, None),
    "inner_node_probability": (0.0, 1.0), "subtree_min_depth": (0, 90), "subtree_max_depth": (0, 90),
    "exp_runs": (2, None), "exp_workers": (1, None),
}


class ParamSlider:
    """Label + live value + slider. `on_commit` fires when the user lets go.
    In expert mode (set_expert) a number field replaces the slider."""

    instances = []
    editing = None   # the slider whose number field has the keyboard focus (no run shortcuts then)

    def __init__(self, label, attr, lo, hi, fmt, divisions=None, on_commit=None, width=None, compact=False):
        self.attr, self.fmt, self.on_commit = attr, fmt, on_commit
        value = getattr(config, attr)
        # Decided once: the slider may report 0 as an int, which must not turn a
        # float setting (e.g. the inter-generation pause) into an integer one.
        self.is_float = isinstance(value, float)
        self.hard = HARD_LIMITS.get(attr, (None, None))
        self.range = [lo, hi]   # the slider's range; only a problem's own sizes may widen it (see set)
        self.expert = False
        self.last_committed = value   # on_commit only fires when the value really changed
        self.value_text = mono(fmt(value), size=12.5, text_align=ft.TextAlign.RIGHT)
        self.slider = ft.Slider(
            min=lo, max=hi, value=min(max(value, lo), hi), divisions=divisions if divisions else int(hi - lo),
            on_change=self._changed, on_change_end=self._committed,
            padding=ft.Padding.symmetric(horizontal=0, vertical=4), width=width,
        )
        self.field = ft.TextField(
            value=self._text(value), visible=False, dense=True, width=width, border_color=S.RULE,
            filled=True, fill_color="#FFFFFF",
            focused_border_color=S.INK, border_radius=4, keyboard_type=ft.KeyboardType.NUMBER,
            text_style=ft.TextStyle(font_family=S.MONO, size=12.5, color=S.INK),
            content_padding=ft.Padding.symmetric(horizontal=10, vertical=8),
            on_change=self._typed, on_focus=self._field_focus, on_blur=self._field_blur,
            on_submit=self._field_done)
        self.label_text = ft.Text(label, size=13, color=S.INK)
        if compact:
            self.slider_box = ft.Container(self.slider, expand=True)
            self.value_box = ft.Container(self.value_text, width=44)
            self.field.expand, self.field.width = True, None
            self.control = ft.Row([self.slider_box, self.value_box, self.field], spacing=8, width=width)
        else:
            self.control = ft.Column(
                [ft.Row([self.label_text, self.value_text],
                        alignment=ft.MainAxisAlignment.SPACE_BETWEEN),
                 self.slider, ft.Container(self.field, padding=ft.Padding.only(top=4, bottom=2))],
                spacing=0,
            )
        ParamSlider.instances.append(self)

    def _text(self, v):
        return f"{float(v):.10g}" if self.is_float else str(int(v))

    def _parse(self, text):
        """(value, error text) of what was typed into the expert field."""
        try:
            v = float(text) if self.is_float else int(str(text).strip())
        except (TypeError, ValueError):
            return None, "a number" if self.is_float else "a whole number"
        if self.is_float and not math.isfinite(v):
            return None, "a finite number"
        lo, hi = self.hard
        if lo is not None and hi is not None and not lo <= v <= hi:
            return None, f"{lo:g} – {hi:g}"
        if lo is not None and v < lo:
            return None, f"at least {lo:g}"
        if hi is not None and v > hi:
            return None, f"at most {hi:g}"
        return v, None

    def _apply(self, v):
        setattr(config, self.attr, v)   # expert values may leave the slider's range; the slider is hidden
        self.slider.value = min(max(v, self.range[0]), self.range[1])
        self.value_text.value = self.fmt(v)

    def _typed(self, e):
        v, error = self._parse(self.field.value)
        self.field.error = error
        if error is None:
            self._apply(v)
        self.field.update()

    def _field_focus(self, e):
        ParamSlider.editing = self

    def _field_blur(self, e):
        if ParamSlider.editing is self:
            ParamSlider.editing = None
        return self._field_done(e)

    def _field_done(self, e):
        v, error = self._parse(self.field.value)
        if error is None:
            return self._commit()

    def _commit(self):
        """on_commit if the value changed since the last commit (leaving a field or letting go of the
        slider without a change must not, e.g., draw new training data)."""
        value = getattr(config, self.attr)
        if value == self.last_committed:
            return None
        self.last_committed = value
        if self.on_commit:
            return self.on_commit()
        return None

    def clamp(self):
        """Leaving expert mode: a value outside the slider's range becomes its minimum or maximum.
        Returns True if the value changed."""
        value = getattr(config, self.attr)
        lo, hi = self.range
        clamped = min(max(value, lo), hi)
        clamped = float(clamped) if self.is_float else int(round(clamped))
        self.slider.min, self.slider.max, self.slider.value = lo, hi, clamped
        self.value_text.value = self.fmt(clamped)
        if clamped != value:
            setattr(config, self.attr, clamped)
            self.last_committed = clamped
            return True
        return False

    def set_expert(self, flag):
        """Returns True if leaving expert mode changed the value (see clamp)."""
        changed = False if flag else self.clamp()
        self.expert = flag
        self.slider.visible = self.value_text.visible = not flag
        for box in (getattr(self, "slider_box", None), getattr(self, "value_box", None)):
            if box is not None:
                box.visible = not flag   # otherwise the empty boxes push the field aside
        self.field.visible = flag
        if flag:
            self.field.value = self._text(getattr(config, self.attr))
            self.field.error = None
        return changed

    @property
    def value(self):
        v = self.slider.value
        return round(float(v), 10 if self.expert else 4) if self.is_float else int(round(v))

    def set(self, value):
        """A value set by the app (e.g. the fixed training size of a problem) widens the range if needed."""
        if not self.expert:
            self.range = [min(self.range[0], value), max(self.range[1], value)]
        self.slider.min, self.slider.max = self.range
        self.slider.value = min(max(value, self.range[0]), self.range[1])
        setattr(config, self.attr, value)
        self.last_committed = value
        self.value_text.value = self.fmt(value)
        self.field.value = self._text(value)
        self.field.error = None

    def _changed(self, e):
        setattr(config, self.attr, self.value)
        self.value_text.value = self.fmt(self.value)
        self.value_text.update()

    def _committed(self, e):
        self._changed(e)
        return self._commit()

    @property
    def disabled(self):
        return self.slider.disabled

    @disabled.setter
    def disabled(self, flag):
        self.slider.disabled = self.field.disabled = flag
        self.field.disabled = flag
        self.value_text.color = S.FAINT if flag else S.INK
        self.field.text_style = ft.TextStyle(font_family=S.MONO, size=12.5, color=S.FAINT if flag else S.INK)


# --------------------------------------------------------------------------
# App
# --------------------------------------------------------------------------

async def main(page: ft.Page):
    config.load_settings()   # the settings of the last session
    if config.target_function not in problems.REGISTRY:
        config.target_function = config.DEFAULTS["target_function"]
    page.title = f"{config.APP_NAME} v{config.APP_VERSION}"
    page.bgcolor = S.PAPER
    page.padding = 0
    page.spacing = 0
    page.theme_mode = ft.ThemeMode.LIGHT
    page.theme = ft.Theme(
        font_family=S.SANS,
        use_material3=True,
        color_scheme=ft.ColorScheme(
            primary=S.ACCENT, on_primary="#FFFFFF", secondary=S.TEAL,
            surface=S.PANEL, on_surface=S.INK, on_surface_variant=S.MUTED,
            outline=S.RULE, outline_variant=S.RULE, surface_container_highest=S.RULE_SOFT,
        ),
        slider_theme=ft.SliderTheme(
            active_track_color=S.INK, inactive_track_color=S.RULE, thumb_color=S.INK,
            overlay_color=ft.Colors.with_opacity(0.08, S.INK), track_height=2.5,
            disabled_active_track_color=S.FAINT, disabled_inactive_track_color=S.RULE,
            disabled_thumb_color=S.FAINT, value_indicator_color=S.INK,
        ),
        # DISABLED first: a switch without effect (or locked during a run) must look inactive
        switch_theme=ft.SwitchTheme(
            thumb_color={ft.ControlState.DISABLED: S.PANEL, ft.ControlState.SELECTED: "#FFFFFF",
                         ft.ControlState.DEFAULT: S.MUTED},
            track_color={ft.ControlState.DISABLED: S.RULE_SOFT, ft.ControlState.SELECTED: S.INK,
                         ft.ControlState.DEFAULT: S.PAPER},   # same colour as the sliders
            track_outline_color={ft.ControlState.DISABLED: S.RULE, ft.ControlState.SELECTED: S.INK,
                                 ft.ControlState.DEFAULT: S.FAINT},
        ),
        divider_theme=ft.DividerTheme(color=S.RULE, thickness=1, space=1),
        # hand cursor on every enabled button
        text_button_theme=ft.TextButtonTheme(style=ft.ButtonStyle(mouse_cursor=HAND)),
        icon_button_theme=ft.IconButtonTheme(style=ft.ButtonStyle(mouse_cursor=HAND)),
        filled_button_theme=ft.FilledButtonTheme(style=ft.ButtonStyle(mouse_cursor=HAND)),
        segmented_button_theme=ft.SegmentedButtonTheme(style=ft.ButtonStyle(mouse_cursor=HAND)),
        tooltip_theme=ft.TooltipTheme(text_style=ft.TextStyle(size=11, color=S.PANEL),
                                      decoration=ft.BoxDecoration(bgcolor=S.INK, border_radius=3)),
    )
    page.window.min_width = 1100
    page.window.min_height = 720
    # size of the last session (its normal size, also when it ended maximized or in full screen)
    page.window.width = max(page.window.min_width, config.window_width)
    page.window.height = max(page.window.min_height, config.window_height)
    page.window.bgcolor = S.PAPER
    page.window.icon = "icon.ico"   # title bar and taskbar (Windows only; from assets/)
    # The client starts hidden (AppView.FLET_APP_HIDDEN); mirror that here so
    # that setting `visible = True` after centering is sent as a real change.
    page.window.visible = False
    page.update()

    run: gp.GPRun | None = None
    # Inputs are locked only while a run is running or paused: a finished run keeps showing its own
    # settings (run.settings), and "Run again" starts with the current ones.
    run_state = {"locked": False}
    last_snapshot: gp.Snapshot | None = None
    render_lock = asyncio.Lock()

    def current_problem():
        return problems.get(config.target_function)

    def regenerate_data():
        config.training_X, config.training_y, config.validation_X, config.validation_y = problems.make_data(
            current_problem(), config.training_set_size, config.validation_set_size,
            config.noise_intensity, config.outlier_share)

    regenerate_data()

    # ---- plots -----------------------------------------------------------

    def plot_image(png):
        # FILL: the PNG is rendered at the container's size, so it is never
        # distorted once the (debounced) re-render after a resize has landed.
        return ft.Image(src=png, fit=ft.BoxFit.FILL, gapless_playback=True,
                        filter_quality=ft.FilterQuality.HIGH, expand=True)

    error_img = plot_image(plots.render_error_plot(None, config.generations, config.plot_adjust,
                                                   percent=current_problem().kind != "sr"))
    fit_img = plot_image(plots.render_fit_plot(None, current_problem(), config.training_X, config.training_y,
                                               config.plot_adjust, test_size=len(config.validation_y),
                                               test_y=config.validation_y))

    def holdout_name(settings):
        kind = settings["problem"].kind
        return {"sr": "validation", "ps": "test"}.get(kind)   # Boolean: full truth table, no test set

    def show(problem, v):
        """Value as shown to the user: 1/0 for Boolean problems, Python repr otherwise."""
        if problem.kind == "logic" and isinstance(v, bool):
            return "1" if v else "0"
        if problem.kind == "logic" and v in ("True", "False"):
            return "1" if v == "True" else "0"
        return repr(list(v)) if isinstance(v, tuple) else repr(v)

    plot_sizes = {"error": plots.DEFAULT_SIZE, "fit": plots.DEFAULT_SIZE}
    card_view = {"error": "main", "fit": "main"}   # "main" or a key of plots.DYNAMICS_VIEWS

    def render_card(which, snap, settings, size, angles=None, scale=None, draft=False):
        """PNG of one plot card in its current view."""
        problem = settings["problem"]
        if card_view[which] != "main":
            return plots.render_dynamics(card_view[which], snap, settings["generations"], size, problem.kind)
        if which == "error":
            return plots.render_error_plot(snap, settings["generations"], config.plot_adjust, size,
                                           holdout=holdout_name(settings), percent=problem.kind != "sr")
        return plots.render_fit_plot(snap, problem, settings["training_X"], settings["training_y"],
                                     config.plot_adjust, size, angles or view["angles"],
                                     scale or plots.SCALE, len(settings["validation_y"]), settings["validation_y"],
                                     draft=draft)
    render_state = {"dirty": False, "busy": False}
    resize_timer = {"handle": None}

    async def refresh_plots():
        """Render both plots off the UI thread and swap them in one go.

        Requests arriving while a render is in progress are coalesced into
        one follow-up render of the latest state.
        """
        render_state["dirty"] = True
        if render_state["busy"]:
            return
        render_state["busy"] = True
        try:
            while render_state["dirty"]:
                render_state["dirty"] = False
                snap = last_snapshot if run is not None else None
                settings = run.settings if run is not None else config.run_settings()
                err_size, fit_size = plot_sizes["error"], plot_sizes["fit"]
                err_png, fit_png = await asyncio.to_thread(lambda: (
                    render_card("error", snap, settings, err_size),
                    render_card("fit", snap, settings, fit_size),
                ))
                error_img.src, fit_img.src = err_png, fit_png
                error_img.update()
                fit_img.update()
        finally:
            render_state["busy"] = False

    view = {"angles": plots.DEFAULT_VIEW}
    fit_state = {"dirty": False, "busy": False, "fast": False}

    async def refresh_fit(fast=False):
        """Re-render only the fit plot (3D rotation). `fast` = 1x resolution while dragging."""
        fit_state["dirty"], fit_state["fast"] = True, fast
        if fit_state["busy"]:
            return
        fit_state["busy"] = True
        try:
            while fit_state["dirty"]:
                fit_state["dirty"] = False
                snap = last_snapshot if run is not None else None
                settings = run.settings if run is not None else config.run_settings()
                png = await asyncio.to_thread(render_card, "fit", snap, settings, plot_sizes["fit"],
                                              view["angles"], 1 if fit_state["fast"] else plots.SCALE,
                                              fit_state["fast"])
                fit_img.src = png
                fit_img.update()
        finally:
            fit_state["busy"] = False

    def is_3d():
        if card_view["fit"] != "main":
            return False
        problem = run.settings["problem"] if run is not None else current_problem()
        return problem.kind == "sr" and problem.inputs == 2 and not getattr(problem, "dataset", False)

    def set_fit_cursor(cursor):
        fit_gesture.mouse_cursor = cursor
        fit_gesture.update()

    # While dragging, frames are rendered in parallel by worker processes (surface_workers);
    # only a frame newer than the one on screen is shown, and none after the drag has ended
    # (then the sharp PNG of refresh_fit takes over).
    drag = {"active": False, "seq": 0, "shown": 0, "inflight": 0, "pending": False}

    async def drag_frame():
        if not surface_workers.ready():
            await refresh_fit(fast=True)
            return
        if drag["inflight"] >= surface_workers.WORKERS:
            drag["pending"] = True   # rendered by the next free worker, with the angles of then
            return
        drag["seq"] += 1
        seq = drag["seq"]
        snap = last_snapshot if run is not None else None
        settings = run.settings if run is not None else config.run_settings()
        drag["inflight"] += 1
        t_submit = time.perf_counter()
        try:
            png = await surface_workers.render(snap, settings["problem"], settings["training_X"],
                                               settings["training_y"], config.plot_adjust, plot_sizes["fit"],
                                               view["angles"])
        finally:
            drag["inflight"] -= 1
        t_done = time.perf_counter()
        if png is None:
            await refresh_fit(fast=True)
        elif drag["active"] and seq > drag["shown"] and is_3d():
            drag["shown"] = seq
            fit_img.src = png
            fit_img.update()
            if DRAG_LOG is not None:
                DRAG_LOG.append(("frame", t_submit, t_done, time.perf_counter(), len(png)))
        if drag["pending"] and drag["active"]:
            drag["pending"] = False
            await drag_frame()

    def on_fit_pan_start(e):
        if is_3d():
            drag["active"], drag["pending"] = True, False
            surface_workers.start()
            set_fit_cursor(ft.MouseCursor.GRABBING)

    async def on_fit_pan_update(e: ft.DragUpdateEvent):
        if not is_3d() or e.local_delta is None:
            return
        elev, azim = view["angles"]
        azim -= e.local_delta.x * 0.5
        elev = min(90.0, max(-5.0, elev + e.local_delta.y * 0.4))
        view["angles"] = (elev, azim)
        if DRAG_LOG is not None:
            DRAG_LOG.append(("event", time.perf_counter(), surface_workers.ready(), drag["inflight"]))
        await drag_frame()

    async def on_fit_pan_end(e):
        drag["active"], drag["pending"] = False, False
        if DRAG_LOG:
            with open(os.environ["GPX_DRAG_LOG"], "a") as f:
                f.write(f"--- drag, size {plot_sizes['fit']}, workers {surface_workers.WORKERS}\n")
                for row in DRAG_LOG:
                    f.write(" ".join(f"{v:.4f}" if isinstance(v, float) else str(v) for v in row) + "\n")
            DRAG_LOG.clear()
        if is_3d():
            set_fit_cursor(ft.MouseCursor.GRAB)
            await refresh_fit(fast=False)

    async def on_fit_double_tap(e):
        if is_3d():
            view["angles"] = plots.DEFAULT_VIEW
            await refresh_fit(fast=False)

    # Program synthesis: hovering a case cell shows the case and the best program's output.
    case_info = mono("", size=12.5, color=S.INK, max_lines=1, overflow=ft.TextOverflow.ELLIPSIS,
                     text_align=ft.TextAlign.RIGHT)

    def on_fit_hover(e: ft.HoverEvent):
        if is_3d():
            surface_workers.start()   # warm up the 3D renderers before the first drag
        settings = run.settings if run is not None else config.run_settings()
        problem = settings["problem"]
        if problem.kind == "sr" or e.local_position is None or card_view["fit"] != "main":
            return
        n = len(settings["training_y"])
        idx = plots.case_index_at(problem, n, plot_sizes["fit"], e.local_position.x, e.local_position.y)
        if idx is None:
            text = ""
        else:
            inputs, expected = settings["training_X"][idx], settings["training_y"][idx]
            text = f"#{idx + 1}  {problem.describe_inputs(inputs, 48)}  →  {show(problem, expected)}"
            snap = last_snapshot
            if snap is not None and snap.best_outputs is not None and idx < len(snap.best_outputs):
                mark = "✓" if snap.best_solved[idx] else "✗"
                text += (f"   best {show(problem, snap.best_outputs[idx])} {mark}"
                         f"   {100 * snap.case_share[idx]:.0f} % solve it")
        if text != case_info.value:
            case_info.value = text
            case_info.update()

    def open_case_dialog(idx):
        """Full details of one training case (the hover line may be cut off)."""
        settings = run.settings if run is not None else config.run_settings()
        problem = settings["problem"]
        n = len(settings["training_y"])
        inputs, expected = settings["training_X"][idx], settings["training_y"][idx]
        snap = last_snapshot if run is not None else None

        def section(label, *controls):
            return ft.Column([caps(label), *controls], spacing=6)

        def value(v):
            return mono(show(problem, v), size=13.5, selectable=True)

        rows = [section("Input", *[
            ft.Row([mono(f"{name} =", size=13.5, color=S.TEAL, weight=ft.FontWeight.W_600),
                    ft.Container(value(v), expand=True)],
                   spacing=8, vertical_alignment=ft.CrossAxisAlignment.START)
            for name, v in zip(problem.arg_names, inputs)])]
        rows.append(section("Expected output", value(expected)))
        if snap is not None and snap.best_outputs is not None and idx < len(snap.best_outputs):
            ok = bool(snap.best_solved[idx])
            rows.append(section(
                "Best program",
                ft.Row([mono(show(problem, snap.best_outputs[idx]), size=13.5, selectable=True),
                        ft.Text("✓ solved" if ok else f"✗ error {snap.best_errors[idx]:g}", size=13.5,
                                color=S.GREEN if ok else S.ACCENT, weight=ft.FontWeight.W_600)],
                       spacing=14, wrap=True)))
            notes = [f"Solved by {100 * snap.case_share[idx]:.0f} % of the population"]
            if snap.downsample is not None and idx in set(np.asarray(snap.downsample).tolist()):
                notes.append("In the current down-sample")
            if snap.outliers is not None and idx in set(np.asarray(snap.outliers).tolist()):
                notes.append("Excluded as outlier (ROIDS)")
            rows.append(section("Population", *[ft.Text(t, size=13.5, color=S.INK) for t in notes]))

        kind = ("Truth-table row" if problem.kind == "logic" else
                "Edge case" if idx < problem.edge_count(n) else "Random case")

        def close(_=None):
            dialog_state["open"] = False
            page.pop_dialog()

        dialog = ft.AlertDialog(
            modal=False, bgcolor=S.PANEL, shape=ft.RoundedRectangleBorder(radius=6),
            title=ft.Row([ft.Text(f"Training case #{idx + 1}", size=18, weight=ft.FontWeight.W_600, color=S.INK),
                          ft.Container(expand=True), tag(kind)],
                         vertical_alignment=ft.CrossAxisAlignment.CENTER),
            content=ft.Container(ft.Column(rows, spacing=18, tight=True), width=560),
            actions=[ft.TextButton("Close", on_click=close, style=ft.ButtonStyle(color=S.INK))],
            on_dismiss=lambda e: dialog_state.update(open=False),
        )
        dialog_state["open"] = True
        page.show_dialog(dialog)

    def on_fit_tap(e: ft.TapEvent):
        settings = run.settings if run is not None else config.run_settings()
        problem = settings["problem"]
        if problem.kind == "sr" or e.local_position is None or card_view["fit"] != "main":
            return
        n = len(settings["training_y"])
        idx = plots.case_index_at(problem, n, plot_sizes["fit"], e.local_position.x, e.local_position.y)
        if idx is not None:
            open_case_dialog(idx)

    def on_fit_exit(e):
        if case_info.value:
            case_info.value = ""
            case_info.update()

    fit_gesture = ft.GestureDetector(
        content=fit_img, expand=True, drag_interval=25, mouse_cursor=ft.MouseCursor.BASIC,
        on_pan_start=on_fit_pan_start, on_pan_update=on_fit_pan_update, on_pan_end=on_fit_pan_end,
        on_double_tap=on_fit_double_tap, on_hover=on_fit_hover, on_exit=on_fit_exit, hover_interval=40,
        on_tap_up=on_fit_tap,
    )

    def plot_resized(which):
        # size_change_interval=0 reports every frame, so the *final* size of a
        # maximise / restore is never dropped. Rendering is debounced: one
        # render shortly after the last size change.
        def handler(e: ft.LayoutSizeChangeEvent):
            size = (int(e.width), int(e.height))
            if size[0] <= 0 or size[1] <= 0 or size == plot_sizes[which]:
                return
            plot_sizes[which] = size
            if resize_timer["handle"] is not None:
                resize_timer["handle"].cancel()
            loop = asyncio.get_running_loop()
            resize_timer["handle"] = loop.call_later(0.15, lambda: loop.create_task(refresh_plots()))
        return handler

    # ---- settings: problem & evolution (sidebar) -------------------------

    async def on_training_commit():
        regenerate_data()
        await refresh_plots()

    async def on_validation_commit():
        problem = current_problem()
        if problem.kind == "ps":
            regenerate_data()   # test cases are drawn together with the training cases
            await refresh_plots()
        else:
            config.validation_X = problems.make_validation_set(problem, config.validation_set_size)
            config.validation_y = problems.evaluate(problem.expression, config.validation_X).tolist()

    async def on_generations_commit():
        await refresh_plots()

    s_train = ParamSlider("Training set size", "training_set_size", 1, 500, lambda v: f"{int(v)}", on_commit=on_training_commit)
    s_noise = ParamSlider("Training noise", "noise_intensity", 0.0, 1.0, lambda v: f"{v:.2f}", divisions=100, on_commit=on_training_commit)
    s_outl = ParamSlider("Training outliers", "outlier_share", 0.0, 0.2, lambda v: f"{v:.2f}", divisions=20, on_commit=on_training_commit)
    s_val = ParamSlider("Validation set size", "validation_set_size", 1, 2000, lambda v: f"{int(v)}", on_commit=on_validation_commit)
    s_gens = ParamSlider("Generations", "generations", 1, 500, lambda v: f"{int(v)}", on_commit=on_generations_commit)
    s_pop = ParamSlider("Population size", "population_size", 1, 500, lambda v: f"{int(v)}")
    s_cx = ParamSlider("Crossover probability", "crossover_probability", 0.0, 1.0, lambda v: f"{v:.2f}", divisions=100)
    s_mut = ParamSlider("Mutation probability", "mutation_probability", 0.0, 1.0, lambda v: f"{v:.2f}", divisions=100)

    def problem_options():
        """Problems grouped by kind, each group under a (not selectable) section title."""
        options = []
        for kind, group in problems.ordered().items():
            if not group:
                continue
            options.append(ft.DropdownOption(key=f"__section_{kind}", text=problems.KIND_TITLES[kind],
                                             content=caps(problems.KIND_TITLES[kind], color=S.ACCENT), disabled=True))
            for p in group:
                right = p.name or ({"ps": "program", "logic": "logic"}.get(p.kind, "custom") if p.custom else "")
                content = ft.Row([mono(p.expression, size=12.5), ft.Container(expand=True),
                                  ft.Text(right, size=12.5, color=S.INK)], spacing=8)
                options.append(ft.DropdownOption(key=p.expression, text=p.expression, content=content))
        return options

    ps_state = {"auto": False}      # program synthesis blocks were switched on automatically
    logic_state = {"auto": False}   # logic gates were set automatically

    def set_ps_blocks(on):
        for flag in synthesis.GROUP_FLAGS:
            setattr(config, flag, on)
            ps_switches[flag].value = on

    def set_gates(flags):
        for flag in synthesis.GATE_FLAGS:
            setattr(config, flag, flag in flags)
            logic_switches[flag].value = flag in flags

    async def select_problem(expression):
        if run is not None and not run_state["locked"]:
            await reset()   # the finished run belongs to the previous problem
        config.target_function = expression
        target_dropdown.value = expression
        problem = current_problem()
        messages = []
        if problem.kind == "ps":
            s_train.set(problem.train_size)
            s_val.set(problem.test_size)
            if not all(getattr(config, f) for f in synthesis.GROUP_FLAGS):
                set_ps_blocks(True)
                ps_state["auto"] = True
                messages.append("Program synthesis building blocks enabled")
        elif ps_state["auto"]:
            set_ps_blocks(False)
            ps_state["auto"] = False
            messages.append("Program synthesis building blocks disabled")
        if getattr(problem, "dataset", False) and problem.kind == "sr":
            s_train.set(problem.train_size)   # fixed split of the data set
            s_val.set(problem.test_size)
        if problem.kind == "logic":
            s_train.set(problem.train_size)   # the full truth table
            if [f for f in synthesis.GATE_FLAGS if getattr(config, f)] != list(problem.gates):
                set_gates(problem.gates)
                logic_state["auto"] = True
                names = ", ".join(f.split("_")[1].upper() for f in problem.gates)
                messages.append(f"Logic gates set to {names}")
        elif logic_state["auto"]:
            set_gates(())
            logic_state["auto"] = False
            messages.append("Logic gates disabled")
        if messages:
            toast("  ·  ".join(messages), seconds=4)
        regenerate_data()
        sync_problem_views()
        await refresh_plots()
        page.update()

    async def changed_target_function(e):
        await select_problem(target_dropdown.value)

    target_dropdown = ft.Dropdown(
        value=config.target_function,
        options=problem_options(),
        on_select=changed_target_function,
        expand=True, dense=True, filled=True, fill_color=S.PANEL,
        border_color=S.RULE, focused_border_color=S.INK, border_radius=4,
        text_style=ft.TextStyle(font_family=S.MONO, size=12.5, color=S.INK),
        menu_style=ft.MenuStyle(bgcolor=S.PANEL), menu_width=420,
        enable_search=False,
    )

    # ---- custom benchmark problems ---------------------------------------

    dialog_state = {"open": False}

    def open_custom_dialog(e):
        """Dialog for a custom problem: a target function f(x[, y]) (symbolic regression)
        or a Python function (program synthesis). Built fresh each time."""
        def input_field(label, value, width=None, expand=None, autofocus=False, hint=None, **kw):
            return ft.TextField(
                label=label, value=value, hint_text=hint, width=width, expand=expand, autofocus=autofocus,
                text_style=ft.TextStyle(font_family=S.MONO, size=14, color=S.INK),
                border_color=S.RULE, focused_border_color=S.INK, border_radius=4, dense=True, **kw,
            )

        mode = {"kind": "sr", "types": set()}
        defaults = synthesis.InputSpec()
        field = input_field("f(x) or f(x, y)", "", autofocus=True, hint="x**3 - 2*x")
        logic_field = input_field("Boolean expression", "", hint="(a and not b) or (c ^ d)", visible=False)
        name_field = input_field("Name (optional)", "", hint="e.g. My-Quartic")
        code_field = CodeEditor(synthesis.TEMPLATE, lines=12, label="Python function", width=640)   # highlighted
        code_field.visible = False
        lo_field = input_field("x, y from", "-1", expand=True)
        hi_field = input_field("to", "1", expand=True)
        domain_row = ft.Row([lo_field, hi_field], spacing=12)

        # Random inputs of a program, one row per parameter type used in the signature.
        def pair(lo, hi):
            return input_field("from", f"{lo:g}", expand=True), input_field("to", f"{hi:g}", expand=True)

        int_lo, int_hi = pair(*defaults.int_range)
        flt_lo, flt_hi = pair(*defaults.float_range)
        str_lo, str_hi = pair(*defaults.str_length)
        vlen_lo, vlen_hi = pair(*defaults.vec_length)
        vval_lo, vval_hi = pair(*defaults.vec_range)
        alphabet = ft.Dropdown(
            value=defaults.alphabet, options=[ft.DropdownOption(key=k, text=k) for k in synthesis.ALPHABETS],
            expand=True, dense=True, filled=True, fill_color=S.PANEL, border_color=S.RULE,
            focused_border_color=S.INK, border_radius=4, text_style=ft.TextStyle(size=13, color=S.INK),
            menu_style=ft.MenuStyle(bgcolor=S.PANEL), enable_search=False, label="characters",
        )
        custom_chars = input_field("custom characters", "", expand=True, visible=False)

        def spec_row(title, *controls):
            return ft.Column([caps(title), ft.Row(list(controls), spacing=12)], spacing=6, visible=False)

        int_row = spec_row("int inputs", int_lo, int_hi)
        flt_row = spec_row("float inputs", flt_lo, flt_hi)
        str_row = ft.Column([caps("str inputs  ·  length"), ft.Row([str_lo, str_hi], spacing=12),
                             ft.Row([alphabet, custom_chars], spacing=12)], spacing=6, visible=False)
        vec_row = ft.Column([caps("list[int] inputs  ·  length"), ft.Row([vlen_lo, vlen_hi], spacing=12),
                             caps("list[int] inputs  ·  values"), ft.Row([vval_lo, vval_hi], spacing=12)],
                            spacing=6, visible=False)
        spec_rows = {synthesis.Int: int_row, synthesis.Num: flt_row, synthesis.Str: str_row, synthesis.Vec: vec_row}

        info = ft.Text("", size=13, color=S.INK, max_lines=2, overflow=ft.TextOverflow.ELLIPSIS)
        info_box = ft.Container(info, height=40, alignment=ft.Alignment.TOP_LEFT)   # fixed: no jumps while typing
        add_btn = ft.FilledButton(
            "Add problem", disabled=True,
            style=ft.ButtonStyle(bgcolor={ft.ControlState.DEFAULT: S.ACCENT, ft.ControlState.DISABLED: S.RULE},
                                 color={ft.ControlState.DEFAULT: "#FFFFFF", ft.ControlState.DISABLED: S.FAINT},
                                 shape=ft.RoundedRectangleBorder(radius=4)))

        def number(f):
            try:
                v = float(f.value)
            except (TypeError, ValueError):
                return math.nan
            return int(v) if v.is_integer() else v

        def read_spec():
            return synthesis.InputSpec(
                int_range=(number(int_lo), number(int_hi)), float_range=(number(flt_lo), number(flt_hi)),
                str_length=(number(str_lo), number(str_hi)), alphabet=alphabet.value or "lowercase",
                custom_chars=custom_chars.value or "", vec_length=(number(vlen_lo), number(vlen_hi)),
                vec_range=(number(vval_lo), number(vval_hi)))

        def fail(msg):
            info.value, info.color = msg, "#B00020"
            add_btn.disabled = True
            return None

        def sync_spec_rows():
            ps = mode["kind"] == "ps"
            for t, row in spec_rows.items():
                row.visible = ps and t in mode["types"]
            custom_chars.visible = alphabet.value == "custom"

        def check():
            if mode["kind"] == "dataset":
                return check_dataset()
            if mode["kind"] == "logic":
                name = (name_field.value or "").strip()
                if name in problems.REGISTRY and not problems.REGISTRY[name].custom:
                    return fail(f"'{name}' is the name of a built-in problem.")
                try:
                    problem = synthesis.parse_logic(logic_field.value, name)
                except synthesis.ProgramError as exc:
                    return fail(str(exc) if (logic_field.value or "").strip() else "")
                info.value = (f"{len(problem.arg_names)} inputs ({', '.join(problem.arg_names)})  ·  "
                              f"{problem.train_size} truth-table rows")
                if problem.expression in problems.REGISTRY:
                    info.value += "  (replaces the existing problem)"
                info.color = S.TEAL
                add_btn.disabled = False
                return problem
            if mode["kind"] == "ps":
                try:
                    name, signature, types, constants = synthesis.check_signature(code_field.value)
                except synthesis.ProgramError as exc:
                    return fail(str(exc))   # the input rows stay as they are while the code is incomplete
                mode["types"] = types
                sync_spec_rows()
                spec = read_spec()
                if any(math.isnan(v) for pair_ in (spec.int_range, spec.float_range, spec.str_length,
                                                    spec.vec_length, spec.vec_range) for v in pair_):
                    return fail("Enter numbers in all range fields.")
                try:
                    spec.validate(types)
                except synthesis.ProgramError as exc:
                    return fail(str(exc))
                found = synthesis.format_constants(*constants)
                info.value = (signature + (f"   constants {found}" if found else "")
                              + ("  (replaces the existing problem)" if name in problems.REGISTRY else ""))
                info.color = S.TEAL
                add_btn.disabled = False
                return spec
            domain = (number(lo_field), number(hi_field))
            try:
                expr, inputs = problems.parse_expression(field.value, domain)
            except problems.ExpressionError as exc:
                return fail(str(exc) if (field.value or "").strip() else "")
            square = "²" if inputs == 2 else ""
            info.value = f"{'x, y' if inputs == 2 else 'x'} ∈ [{domain[0]:g}, {domain[1]:g}]{square}"
            if expr in problems.REGISTRY:
                info.value += "  (already in the list)"
            info.color = S.TEAL
            add_btn.disabled = False
            return expr, inputs, domain

        # ---- data set import (fourth tab) --------------------------------
        ds = {"text": None, "det": None, "source": ""}

        def small_select(label, options, value=None, width=None, expand=None):
            return ft.Dropdown(
                label=label, value=value, options=[ft.DropdownOption(key=k, text=t) for k, t in options],
                width=width, expand=expand, dense=True, filled=True, fill_color=S.PANEL, border_color=S.RULE,
                focused_border_color=S.INK, border_radius=4, text_style=ft.TextStyle(size=13, color=S.INK),
                menu_style=ft.MenuStyle(bgcolor=S.PANEL), enable_search=False)

        file_label = mono("No file chosen", size=12.5, color=S.INK, expand=True, max_lines=1,
                          overflow=ft.TextOverflow.ELLIPSIS)
        delim_dd = small_select("delimiter", [(k, k) for k in datasets.DELIMITERS], expand=True)
        header_cb = ft.Checkbox(label="first row is a header", value=False, active_color=S.INK,
                                label_style=ft.TextStyle(size=13, color=S.INK))
        ds_kind = ft.SegmentedButton(
            segments=[ft.Segment(value="sr", label=ft.Text("Symbolic regression")),
                      ft.Segment(value="ps", label=ft.Text("Program synthesis")),
                      ft.Segment(value="logic", label=ft.Text("Logic"))],
            selected=["sr"], show_selected_icon=False,
            style=ft.ButtonStyle(color={ft.ControlState.SELECTED: "#FFFFFF", ft.ControlState.DEFAULT: S.INK},
                                 bgcolor={ft.ControlState.SELECTED: S.TEAL, ft.ControlState.DEFAULT: S.PANEL},
                                 side=ft.BorderSide(1, S.RULE), shape=ft.RoundedRectangleBorder(radius=4)))
        target_dd = small_select("target column", [], expand=True)
        inputs_wrap = ft.Row(wrap=True, spacing=4, run_spacing=0)
        types_wrap = ft.Row(wrap=True, spacing=8, run_spacing=8)
        types_box = ft.Column([caps("column types"), types_wrap], spacing=6, visible=False)
        share_text = mono("30 %", size=12.5)
        share_slider = ft.Slider(min=0, max=0.5, divisions=10, value=0.3, expand=True,
                                 padding=ft.Padding.symmetric(horizontal=0, vertical=4))
        share_row = ft.Row([ft.Text("validation share", size=13, color=S.INK), share_slider, share_text],
                           spacing=10, vertical_alignment=ft.CrossAxisAlignment.CENTER)
        standardize_cb = ft.Checkbox(label="standardize columns (z-scores)", value=True, active_color=S.INK,
                                     label_style=ft.TextStyle(size=13, color=S.INK))
        preview = ft.Column(spacing=2)
        ds_settings = ft.Column([
            ft.Row([delim_dd, header_cb], spacing=16, vertical_alignment=ft.CrossAxisAlignment.CENTER),
            caps("problem type"), ds_kind,
            target_dd, caps("input columns"), inputs_wrap, types_box, share_row, standardize_cb,
            caps("preview"),
            ft.Container(ft.Row([preview], scroll=ft.ScrollMode.AUTO), padding=10, bgcolor=S.PAPER,
                         border=ft.Border.all(1, S.RULE), border_radius=4),
        ], spacing=10, visible=False, horizontal_alignment=ft.CrossAxisAlignment.STRETCH)

        async def choose_file(e):
            files = await file_picker.pick_files(dialog_title="Load a data set", allow_multiple=False)
            if not files or not files[0].path:
                return
            try:
                text = await asyncio.to_thread(datasets.read_file, files[0].path)
                load_text(text, os.path.basename(files[0].path))
            except datasets.DatasetError as exc:
                fail(str(exc))
            dialog.update()

        file_btn = ft.OutlinedButton(
            content=ft.Row([ft.Icon(ft.Icons.UPLOAD_FILE, size=17, color=S.INK),
                            ft.Text("Choose file …", size=13, color=S.INK)], spacing=8, tight=True),
            on_click=choose_file, style=ft.ButtonStyle(side=ft.BorderSide(1, S.FAINT),
                                                       shape=ft.RoundedRectangleBorder(radius=4)))
        file_row = ft.Row([file_btn, file_label], spacing=12, visible=False,
                          vertical_alignment=ft.CrossAxisAlignment.CENTER)

        def populate(det):
            """Show a detection result in the controls (all editable afterwards)."""
            ds["det"] = det
            delim_dd.value, header_cb.value = det.delimiter, det.header
            ds_kind.selected = [det.kind]
            target_dd.options = [ft.DropdownOption(key=str(i), text=f"{n}  ({t})")
                                 for i, (n, t) in enumerate(zip(det.names, det.types))]
            target_dd.value = str(det.target)
            def column_box(i, n):
                # fixed width per entry, otherwise Flet stretches each labelled checkbox to a full line
                cb = ft.Checkbox(label=n, value=i in det.inputs, data=i, active_color=S.INK, on_change=changed,
                                 label_style=ft.TextStyle(size=12.5, color=S.INK, font_family=S.MONO))
                return ft.Container(cb, width=min(260, 48 + 8 * len(n)), data=cb)
            inputs_wrap.controls = [column_box(i, n) for i, n in enumerate(det.names)]
            types_wrap.controls = [
                small_select(n, [(t, t) for t in datasets.TYPES], value=t, width=150)
                for n, t in zip(det.names, det.types)]
            for i, dd in enumerate(types_wrap.controls):
                dd.data, dd.on_select = i, changed
            header = [mono(f"{n}", size=12, weight=ft.FontWeight.W_600) for n in det.names]
            kinds = [mono(t, size=11, color=S.TEAL) for t in det.types]
            cols = [ft.Column([header[j], kinds[j]] + [mono(r[j][:18], size=12) for r in det.rows[:5]],
                              spacing=2, width=max(70, min(170, 9 * max(len(det.names[j]), 6))))
                    for j in range(len(det.names))]
            preview.controls = [ft.Row(cols, spacing=10, vertical_alignment=ft.CrossAxisAlignment.START),
                                mono(f"{len(det.rows)} rows", size=11, color=S.MUTED)]

        def load_text(text, source):
            ds["text"], ds["source"] = text, source
            file_label.value = source
            if not (name_field.value or "").strip():
                name_field.value = os.path.splitext(source)[0][:40]
            populate(datasets.analyse(text))
            check()

        def reparse(e=None):
            if ds["text"] is None:
                return
            try:
                populate(datasets.analyse(ds["text"], delim_dd.value, bool(header_cb.value)))
            except datasets.DatasetError as exc:
                fail(str(exc))
            changed()

        def check_dataset():
            det = ds["det"]
            if det is None:
                return fail("Choose a file (CSV, TSV, DAT, …).")
            kind = ds_kind.selected[0] if ds_kind.selected else "sr"
            target = int(target_dd.value)
            for box in inputs_wrap.controls:
                box.data.disabled = box.data.data == target
            types_box.visible = kind == "ps"
            share_row.visible = kind != "logic"
            standardize_cb.visible = kind == "sr"
            share_text.value = f"{100 * share_slider.value:.0f} %"
            name = (name_field.value or "").strip() or os.path.splitext(ds["source"])[0]
            if name in problems.REGISTRY and not problems.REGISTRY[name].custom:
                return fail(f"'{name}' is the name of a built-in problem.")
            spec = datasets.DatasetSpec(
                name=name, kind=kind, names=list(det.names), types=[dd.value for dd in types_wrap.controls],
                target=target, inputs=[b.data.data for b in inputs_wrap.controls if b.data.value and b.data.data != target],
                rows=det.rows, test_share=round(share_slider.value, 2), standardize=bool(standardize_cb.value),
                source=ds["source"])
            try:
                problem = datasets.build(spec)
            except datasets.DatasetError as exc:
                return fail(str(exc))
            if kind == "logic":
                text = f"{problem.train_size} of {2 ** problem.inputs} input combinations  ·  {problem.inputs} inputs"
            elif kind == "sr":
                text = (f"{len(problem.train[1])} training / {len(problem.test[1])} validation rows  ·  "
                        f"{problem.inputs} inputs")
            else:
                text = f"{problem.train_size} training / {problem.test_size} test rows  ·  {problem.inputs} inputs"
            info.value = text + (f"  ·  {problem.info}" if problem.info else "") + \
                ("  (replaces the existing problem)" if name in problems.REGISTRY else "")
            info.color = S.TEAL
            add_btn.disabled = False
            return problem

        def close(_=None):
            dialog_state["open"] = False
            page.pop_dialog()

        async def add(_=None):
            parsed = check()
            if parsed is None:
                dialog.update()
                return
            if mode["kind"] in ("logic", "dataset"):
                problem = problems.add_custom_program(parsed)
            elif mode["kind"] == "ps":
                info.value, info.color = "Generating training and test cases …", S.INK
                info.update()
                try:
                    problem = await asyncio.to_thread(synthesis.parse_program, code_field.value, parsed)
                except synthesis.ProgramError as exc:
                    fail(str(exc))
                    dialog.update()
                    return
                problems.add_custom_program(problem)
            else:
                problem = problems.add_custom(*parsed, name=(name_field.value or "").strip())
            target_dropdown.options = problem_options()
            close()
            await select_problem(problem.expression)

        def switch_mode(e):
            mode["kind"] = kind_switch.selected[0] if kind_switch.selected else "sr"
            kind = mode["kind"]
            field.visible = domain_row.visible = kind == "sr"
            code_field.visible = kind == "ps"
            logic_field.visible = kind == "logic"
            file_row.visible = ds_settings.visible = kind == "dataset"
            name_field.visible = kind != "ps"   # programs are named after their function
            check()
            dialog.update()

        def changed(e=None):
            check()
            dialog.update()

        kind_switch = ft.SegmentedButton(
            segments=[ft.Segment(value="sr", label=ft.Text("Function f(x, y)")),
                      ft.Segment(value="ps", label=ft.Text("Python program")),
                      ft.Segment(value="logic", label=ft.Text("Logic function")),
                      ft.Segment(value="dataset", label=ft.Text("Dataset"))],
            selected=["sr"], on_change=switch_mode, show_selected_icon=False,
            style=ft.ButtonStyle(color={ft.ControlState.SELECTED: "#FFFFFF", ft.ControlState.DEFAULT: S.INK},
                                 bgcolor={ft.ControlState.SELECTED: S.INK, ft.ControlState.DEFAULT: S.PANEL},
                                 side=ft.BorderSide(1, S.RULE), shape=ft.RoundedRectangleBorder(radius=4)),
        )
        for f in (field, code_field, lo_field, hi_field, int_lo, int_hi, flt_lo, flt_hi, str_lo, str_hi,
                  vlen_lo, vlen_hi, vval_lo, vval_hi, custom_chars, logic_field, name_field):
            f.on_change = changed
        alphabet.on_select = changed
        target_dd.on_select = changed
        ds_kind.on_change = changed
        share_slider.on_change = changed
        standardize_cb.on_change = changed
        delim_dd.on_select = reparse
        header_cb.on_change = reparse
        for f in (field, lo_field, hi_field, logic_field, name_field):
            f.on_submit = add
        add_btn.on_click = add
        dialog = ft.AlertDialog(
            modal=True, bgcolor=S.PANEL, shape=ft.RoundedRectangleBorder(radius=6),
            title=ft.Text("Custom benchmark problem", size=18, weight=ft.FontWeight.W_600, color=S.INK),
            content=ft.Column([kind_switch, field, logic_field, code_field.control, file_row, name_field, domain_row,
                               int_row, flt_row, str_row, vec_row, ds_settings, info_box],
                              tight=True, spacing=14, width=640,
                              horizontal_alignment=ft.CrossAxisAlignment.STRETCH),
            scrollable=True,
            actions=[ft.TextButton("Cancel", on_click=close, style=ft.ButtonStyle(color=S.INK)), add_btn],
            actions_alignment=ft.MainAxisAlignment.END,
        )
        dialog.data = load_text   # lets tests feed a file without the native file dialog
        dialog_state["open"] = True
        page.show_dialog(dialog)

    custom_btn = link_button(ft.Icons.ADD, "Custom problem", open_custom_dialog)

    # Toast: a small message bar, centred at the bottom of the window.
    # Flutter cannot update a SnackBar once shown, so the countdown needs our own bar.
    toast_label = ft.Text("", color=S.PANEL, font_family=S.MONO, size=13, max_lines=1,
                          overflow=ft.TextOverflow.ELLIPSIS, expand=True)
    toast_btn = ft.TextButton("", visible=False, style=ft.ButtonStyle(color="#F2B8A2"))
    toast_layer = ft.Container(
        ft.Row([ft.Container(ft.Row([toast_label, toast_btn], spacing=12,
                                    vertical_alignment=ft.CrossAxisAlignment.CENTER),
                             bgcolor=S.INK, border_radius=4, width=520, height=46,
                             padding=ft.Padding.only(left=16, right=6, top=0, bottom=0))],
               alignment=ft.MainAxisAlignment.CENTER),
        left=0, right=0, bottom=28, visible=False,
    )
    page.overlay.append(toast_layer)
    toast_state = {"token": 0}

    async def show_toast(text, action=None, on_action=None, seconds=4):
        toast_state["token"] += 1
        token = toast_state["token"]

        def hide():
            toast_layer.visible = False
            toast_layer.update()

        async def clicked(e):
            toast_state["token"] += 1
            hide()
            await on_action()

        toast_label.value = text
        toast_btn.visible = action is not None
        toast_btn.on_click = clicked
        toast_layer.visible = True
        for left in range(seconds, 0, -1):
            if action is not None:
                toast_btn.content = f"{action} ({left})"
            toast_layer.update()
            await asyncio.sleep(1)
            if toast_state["token"] != token:
                return
        hide()

    def toast(text, **kw):
        asyncio.get_running_loop().create_task(show_toast(text, **kw))

    async def delete_problem(e):
        problem = problems.remove_custom(config.target_function)
        if problem is None:
            return
        target_dropdown.options = problem_options()
        await select_problem(problems.BUILTIN[0].expression)

        async def undo():
            problems.restore_custom(problem)
            target_dropdown.options = problem_options()
            await select_problem(problem.expression)

        toast(f"Deleted  {problem.expression}", action="Undo", on_action=undo, seconds=5)

    delete_btn = ft.IconButton(
        icon=ft.Icons.DELETE_OUTLINE, icon_size=19, tooltip="Delete this custom problem", on_click=delete_problem,
        width=40, height=40, visible=False,
        style=ft.ButtonStyle(icon_color={ft.ControlState.DEFAULT: S.ACCENT, ft.ControlState.DISABLED: S.FAINT},
                             shape=ft.RoundedRectangleBorder(radius=4)),
    )

    # ---- function & terminal set ----------------------------------------

    switches = {}

    def changed_function_switch(e):
        if not any(switches[f].value for f in FUNCTION_FLAGS):
            e.control.value = True
            show_message("At least one function must always be selected.")
            return
        for flag, sw in switches.items():
            setattr(config, flag, sw.value)

    for _, _, _, flag in FUNCTION_ROWS:
        switches[flag] = ft.Switch(value=getattr(config, flag), on_change=changed_function_switch)

    def changed_ps_switch(e):
        for flag, sw in ps_switches.items():
            setattr(config, flag, sw.value)
        ps_state["auto"] = False   # the user chose; do not switch them off automatically later

    ps_switches = {flag: ft.Switch(value=getattr(config, flag), on_change=changed_ps_switch)
                   for flag in synthesis.GROUP_FLAGS}

    def changed_logic_switch(e):
        for flag, sw in logic_switches.items():
            setattr(config, flag, sw.value)
        logic_state["auto"] = False

    logic_switches = {flag: ft.Switch(value=getattr(config, flag), on_change=changed_logic_switch)
                      for flag in synthesis.GATE_FLAGS}
    ps_signature = mono("", size=12.5, color=S.INK)
    logic_signature = mono("", size=12.5, color=S.INK)

    def terminal_state(active):
        return ft.Text("in use" if active else "not used", size=12.5, color=S.TEAL if active else S.INK,
                       font_family=S.MONO)

    x_state, y_state = terminal_state(True), terminal_state(False)

    # ---- additional parameters ------------------------------------------

    def changed_selection(e):
        config.selection_method = selection_dropdown.value
        sync_operator_buttons()
        sync_selection_controls()
        sync_problem_views()

    def changed_init(e):
        config.initialization = init_dropdown.value
        sync_operator_buttons()

    def small_dropdown(value, options, on_select):
        return ft.Dropdown(
            value=value, options=[ft.DropdownOption(key=k, text=t) for k, t in options], on_select=on_select,
            width=410, dense=True, filled=True, fill_color=S.PANEL, border_color=S.RULE,
            focused_border_color=S.INK, border_radius=4, text_style=ft.TextStyle(size=13, color=S.INK),
            menu_style=ft.MenuStyle(bgcolor=S.PANEL), enable_search=False,
        )

    selection_dropdown = small_dropdown(config.selection_method, gp.SELECTION_METHODS, changed_selection)
    init_dropdown = small_dropdown(config.initialization, [(m, m) for m in ("Ramped Half-n-Half", "Full", "Grow")],
                                   changed_init)

    def depth_commit():
        # keep min initial depth strictly below max initial depth
        if s_min_depth.value >= s_max_depth.value:
            s_min_depth.set(s_max_depth.value - 1)
            s_min_depth.control.update()

    W = 300
    s_tourn = ParamSlider("", "tourn_size", 2, 9, lambda v: f"{int(v)}", width=W, compact=True)
    s_parsimony = ParamSlider("", "parsimony_size", 1.0, 2.0, lambda v: f"{v:.2f}", divisions=20, width=W, compact=True)
    s_eps = ParamSlider("", "lexicase_epsilon", 0.0, 1.0, lambda v: f"{v:.2f}", divisions=100, width=W, compact=True)

    def changed_dt_order(e):
        config.double_tournament_order = dt_order_dd.value

    dt_order_dd = small_dropdown(config.double_tournament_order,
                                 [(o, o) for o in operators.OPTIONS["double_tournament_order"]], changed_dt_order)
    dt_order_dd.width = 326

    def changed_downsampling(e):
        config.downsampling = downsampling_dd.value
        sync_selection_controls()
        sync_problem_views()

    downsampling_dd = small_dropdown(config.downsampling, [(o, o) for o in operators.OPTIONS["downsampling"]],
                                     changed_downsampling)
    downsampling_dd.width = 326
    s_ds = ParamSlider("", "ds_rate", 0.01, 0.5, lambda v: f"{v:.2f}", divisions=49, width=W, compact=True)
    s_rho = ParamSlider("", "parent_sampling_rate", 0.01, 0.2, lambda v: f"{v:.2f}", divisions=19, width=W, compact=True)
    s_k = ParamSlider("", "distance_interval", 1, 50, lambda v: f"{int(v)}", width=W, compact=True)
    s_gamma = ParamSlider("", "outlier_rate", 0.0, 0.3, lambda v: f"{v:.3f}".rstrip("0").rstrip("."), divisions=120, width=W, compact=True)
    s_min_depth = ParamSlider("", "min_initial_tree_depth", 1, 8, lambda v: f"{int(v)}", width=W, compact=True, on_commit=depth_commit)
    s_max_depth = ParamSlider("", "max_initial_tree_depth", 2, 9, lambda v: f"{int(v)}", width=W, compact=True, on_commit=depth_commit)
    s_max_tree = ParamSlider("", "max_tree_depth", 10, 30, lambda v: f"{int(v)}", width=W, compact=True)
    s_pause = ParamSlider("", "inter_gen_pause", 0.0, 5.0, lambda v: f"{v:.1f}s", divisions=50, width=W, compact=True)

    def adaptive():
        """An adaptive control may switch operators during the run: then all their parameters count."""
        return config.parameter_control != "None"

    def sync_selection_controls(locked=None):
        """Only the parameters of the chosen selection method are editable (all of them with an
        adaptive control, which may switch the method)."""
        locked = run_state["locked"] if locked is None else locked
        method = config.selection_method
        free = adaptive()
        s_tourn.disabled = locked or not (free or method in ("Tournament", "Double tournament", "ROIDS"))
        s_parsimony.disabled = dt_order_dd.disabled = locked or not (free or method == "Double tournament")
        s_eps.disabled = locked or not (free or method == "Fixed Epsilon-Lexicase")
        own = method in gp.DOWNSAMPLED_METHODS          # methods with a down-sampling of their own
        can = not own and method != "Random"            # the others may use the down-sampling setting
        extra = can and config.downsampling != "None"
        downsampling_dd.disabled = locked or not (free or can)
        s_ds.disabled = locked or not (free or own or extra)
        s_rho.disabled = s_k.disabled = locked or not (
            free or method in gp.INFORMED_METHODS or (extra and config.downsampling == "Informed"))
        s_gamma.disabled = locked or not (free or method == "ROIDS")

    # advanced parameters (shown with the "Advanced" switch)
    def changed_advanced(e=None):
        config.crossover_type = crossover_dd.value
        config.mutation_type = mutation_dd.value
        config.subtree_method = subtree_dd.value
        config.ephemeral_mode = ephemeral_dd.value
        config.bloat_control = bloat_dd.value
        config.fitness_function = fitness_dd.value
        config.parameter_control = control_dd.value
        sync_operator_buttons()
        try:
            config.seed = max(0, int(seed_field.value or 0))
            seed_field.error = None
        except ValueError:
            seed_field.error = "whole number, 0 = random"
        sync_selection_controls()
        sync_advanced_controls()
        sync_problem_views()

    def adv_dropdown(value, options):
        return small_dropdown(value, [(o, o) for o in options], changed_advanced)

    crossover_dd = adv_dropdown(config.crossover_type, gp.CROSSOVERS)
    mutation_dd = adv_dropdown(config.mutation_type, gp.MUTATIONS)
    bloat_dd = adv_dropdown(config.bloat_control, gp.BLOAT_CONTROLS)
    fitness_dd = adv_dropdown(config.fitness_function, gp.FITNESS_FUNCTIONS)
    control_dd = adv_dropdown(config.parameter_control, operators.BUILTIN["control"])
    subtree_dd = adv_dropdown(config.subtree_method, operators.OPTIONS["subtree_method"])
    ephemeral_dd = adv_dropdown(config.ephemeral_mode, operators.OPTIONS["ephemeral_mode"])
    subtree_dd.width = ephemeral_dd.width = 326   # as wide as the operator dropdowns above and below

    def subtree_commit(changed):
        """Keep the minimum depth of new subtrees at most the maximum."""
        def commit():
            if s_sub_min.value > s_sub_max.value:
                other = s_sub_max if changed is s_sub_min else s_sub_min
                other.set(changed.value)
                other.control.update()
        return commit

    s_inner = ParamSlider("", "inner_node_probability", 0.0, 1.0, lambda v: f"{v:.2f}", divisions=100, width=W,
                          compact=True)
    s_sub_min = ParamSlider("", "subtree_min_depth", 0, 8, lambda v: f"{int(v)}", width=190, compact=True)
    s_sub_max = ParamSlider("", "subtree_max_depth", 0, 8, lambda v: f"{int(v)}", width=190, compact=True)
    s_sub_min.on_commit, s_sub_max.on_commit = subtree_commit(s_sub_min), subtree_commit(s_sub_max)
    s_size = ParamSlider("", "max_tree_size", 10, 300, lambda v: f"{int(v)}", width=W, compact=True)
    s_elite = ParamSlider("", "elitism", 0, 10, lambda v: f"{int(v)}", width=W, compact=True)
    s_erc_lo = ParamSlider("", "erc_lo", -10, 0, lambda v: f"{int(v)}", width=190, compact=True)
    s_erc_hi = ParamSlider("", "erc_hi", 0, 10, lambda v: f"{int(v)}", width=190, compact=True)
    seed_field = ft.TextField(value=str(config.seed), width=160, dense=True, border_color=S.RULE,
                              focused_border_color=S.INK, border_radius=4, on_change=changed_advanced,
                              text_style=ft.TextStyle(font_family=S.MONO, size=13, color=S.INK),
                              hint_text="0 = random", keyboard_type=ft.KeyboardType.NUMBER)
    def changed_numba(e):
        config.numba_acceleration = bool(numba_switch.value)

    numba_switch = ft.Switch(value=config.numba_acceleration and fast.AVAILABLE, on_change=changed_numba,
                             disabled=not fast.AVAILABLE)
    def changed_deep(e):
        config.deep_acceleration = bool(deep_switch.value)

    deep_switch = ft.Switch(value=config.deep_acceleration, on_change=changed_deep)
    advanced_controls = [numba_switch, deep_switch, crossover_dd, mutation_dd, s_inner, s_sub_min, s_sub_max, subtree_dd, ephemeral_dd,
                         bloat_dd, fitness_dd, control_dd, s_size, s_elite, s_erc_lo, s_erc_hi,
                         seed_field]

    def sync_advanced_controls():
        locked = run_state["locked"]
        sr = current_problem().kind == "sr"
        free = adaptive()   # the operators may change during the run
        mutation = config.mutation_type
        new_subtrees = mutation in ("Uniform (subtree)", "Mixed") or mutation not in operators.BUILTIN["mutation"]
        s_inner.disabled = locked or not (free or config.crossover_type == "Leaf-biased")
        s_sub_min.disabled = s_sub_max.disabled = subtree_dd.disabled = locked or not (free or new_subtrees)
        ephemeral_dd.disabled = locked or not (free or mutation in ("Ephemeral constants", "Mixed"))
        s_size.disabled = locked or not (free or config.bloat_control == "Depth + size limit")
        numba_switch.disabled = locked or not fast.AVAILABLE
        fitness_dd.disabled = locked or not sr
        s_erc_lo.disabled = s_erc_hi.disabled = locked or not sr

    # ---- custom operators (Python functions) ----------------------------------

    operator_dropdowns = {"fitness": fitness_dd, "selection": selection_dropdown, "crossover": crossover_dd,
                          "mutation": mutation_dd, "initialization": init_dropdown, "bloat": bloat_dd,
                          "control": control_dd}

    def refresh_operator_options():
        """Built-in choices plus the saved custom operators in every operator dropdown."""
        for kind, dropdown in operator_dropdowns.items():
            builtin = list(gp.SELECTION_METHODS) if kind == "selection" else [(o, o) for o in operators.BUILTIN[kind]]
            dropdown.options = ([ft.DropdownOption(key=k, text=t) for k, t in builtin]
                                + [ft.DropdownOption(key=name, text=f"{name}  (custom)")
                                   for name in operators.REGISTRY[kind]])
            setting = operators.KINDS[kind].setting
            if getattr(config, setting) not in {o.key for o in dropdown.options}:
                setattr(config, setting, operators.BUILTIN[kind][0])   # its custom operator was deleted
            dropdown.value = getattr(config, setting)
        sync_operator_buttons()

    def open_code_dialog(title, kinds, kind, registry, templates, check, test, save, on_saved, code=None,
                         noun=None):
        """Dialog to add or edit Python code (custom operators and custom functions).

        kinds: [(key, label)]; registry: key -> {name: code}; check(key, code) -> (name, text),
        raising with a readable message; test(key, code) -> error text or None (runs in a thread);
        save(key, code) -> name; on_saved(key, name)."""
        state = {"kind": kind}
        editor = CodeEditor(code if code is not None else templates[kind], lines=15, label="Python function",
                            width=900)
        info = ft.Text("", size=13, color=S.INK, max_lines=2, overflow=ft.TextOverflow.ELLIPSIS)
        saved_dd = ft.Dropdown(label="Open a saved one to edit it", dense=True, filled=True, fill_color=S.PANEL,
                               border_color=S.RULE, focused_border_color=S.INK, border_radius=4,
                               text_style=ft.TextStyle(size=13, color=S.INK), menu_style=ft.MenuStyle(bgcolor=S.PANEL),
                               enable_search=False, visible=False)
        save_btn = ft.FilledButton(
            "Add", disabled=True,
            style=ft.ButtonStyle(bgcolor={ft.ControlState.DEFAULT: S.ACCENT, ft.ControlState.DISABLED: S.RULE},
                                 color={ft.ControlState.DEFAULT: "#FFFFFF", ft.ControlState.DISABLED: S.FAINT},
                                 shape=ft.RoundedRectangleBorder(radius=4)))

        def sync_saved():
            names = list(registry[state["kind"]])
            saved_dd.options = [ft.DropdownOption(key=n, text=n) for n in names]
            saved_dd.value = None
            saved_dd.visible = bool(names)

        def check_code():
            try:
                name, text = check(state["kind"], editor.value)
            except Exception as exc:
                info.value, info.color = str(exc), "#B00020"
                save_btn.disabled, save_btn.content = True, "Add"
                return None
            exists = name in registry[state["kind"]]
            info.value = text + ("  ·  saving replaces the saved version" if exists else "")
            info.color = S.TEAL
            save_btn.disabled, save_btn.content = False, ("Save changes" if exists else "Add")
            return name

        def changed(e=None):
            check_code()
            dialog.update()

        def switch_kind(e):
            state["kind"] = kind_seg.selected[0] if kind_seg.selected else kinds[0][0]
            editor.value = templates[state["kind"]]
            sync_saved()
            changed()

        def load_saved(e):
            if saved_dd.value in registry[state["kind"]]:
                editor.value = registry[state["kind"]][saved_dd.value]
                changed()

        async def do_save(e):
            key, code_ = state["kind"], editor.value
            name = check_code()
            if name is None:
                return
            info.value, info.color = "Testing …", S.INK
            save_btn.disabled = True
            dialog.update()
            error = await asyncio.to_thread(test, key, code_)
            if error:
                info.value, info.color = error, "#B00020"
                save_btn.disabled = False
                dialog.update()
                return
            replaced = name in registry[key]
            save(key, code_)
            page.pop_dialog()
            on_saved(key, name)
            what = noun.get(key) if isinstance(noun, dict) else (noun or dict(kinds)[key])
            toast(f"{what} '{name}' {'saved' if replaced else 'added'}", seconds=4)
            page.update()

        kind_seg = ft.SegmentedButton(
            segments=[ft.Segment(value=k, label=ft.Text(label, size=13, no_wrap=True)) for k, label in kinds],
            selected=[kind], on_change=switch_kind, show_selected_icon=False,
            style=ft.ButtonStyle(color={ft.ControlState.SELECTED: "#FFFFFF", ft.ControlState.DEFAULT: S.INK},
                                 bgcolor={ft.ControlState.SELECTED: S.INK, ft.ControlState.DEFAULT: S.PANEL},
                                 side=ft.BorderSide(1, S.RULE), shape=ft.RoundedRectangleBorder(radius=4)))
        editor.on_change = changed
        saved_dd.on_select = load_saved
        save_btn.on_click = do_save
        sync_saved()
        check_code()
        dialog = ft.AlertDialog(
            modal=True, bgcolor=S.PANEL, shape=ft.RoundedRectangleBorder(radius=6),
            title=ft.Text(title, size=18, weight=ft.FontWeight.W_600, color=S.INK),
            content=ft.Column([kind_seg, saved_dd, editor.control,
                               ft.Container(info, height=40, alignment=ft.Alignment.TOP_LEFT)],
                              tight=True, spacing=14, width=900, horizontal_alignment=ft.CrossAxisAlignment.STRETCH),
            scrollable=True,
            actions=[ft.TextButton("Cancel", on_click=lambda _: page.pop_dialog(), style=ft.ButtonStyle(color=S.INK)),
                     save_btn],
            actions_alignment=ft.MainAxisAlignment.END,
        )
        page.show_dialog(dialog)

    OPERATOR_KINDS = [(k.key, k.label or k.title.replace(" function", "")) for k in operators.KINDS.values()]

    def check_operator(kind, code):
        name = operators.check(kind, code)
        return name, f"{name}  ·  custom {operators.KINDS[kind].title.lower()}"

    def operator_saved(kind, name):
        setattr(config, operators.KINDS[kind].setting, name)
        refresh_operator_options()
        if kind in ("fitness", "crossover", "mutation", "bloat", "control"):   # these are advanced parameters
            advanced_switch.value = advanced_card.visible = True
        sync_selection_controls()
        sync_advanced_controls()
        sync_problem_views()

    def open_operator_dialog(e=None, kind="selection", code=None):
        open_code_dialog("Add or edit a custom operator", OPERATOR_KINDS, kind, operators.REGISTRY,
                         operators.TEMPLATES, check_operator,
                         lambda k, c: operators.smoke_test(k, c, config.run_settings()),
                         operators.add, operator_saved, code,
                         noun={k.key: k.title for k in operators.KINDS.values()})

    # edit / delete icons next to each operator dropdown (shown while a custom operator is chosen)
    operator_buttons = {}

    def operator_control(kind):
        dropdown = operator_dropdowns[kind]
        dropdown.width = 326
        title = operators.KINDS[kind].title.lower()

        def edit(e):
            name = getattr(config, operators.KINDS[kind].setting)
            open_operator_dialog(kind=kind, code=operators.REGISTRY[kind].get(name))

        async def delete(e):
            setting = operators.KINDS[kind].setting
            name = getattr(config, setting)
            code = operators.REGISTRY[kind].get(name)
            if code is None:
                return
            operators.delete(kind, name)
            refresh_operator_options()
            sync_selection_controls()
            sync_advanced_controls()
            sync_problem_views()
            page.update()

            async def undo():
                operators.REGISTRY[kind][name] = code
                operators.save()
                setattr(config, setting, name)
                refresh_operator_options()
                sync_selection_controls()
                sync_advanced_controls()
                sync_problem_views()
                page.update()

            toast(f"Deleted  {name}", action="Undo", on_action=undo, seconds=5)

        edit_btn = ft.IconButton(ft.Icons.EDIT_OUTLINED, icon_size=18, tooltip=f"Edit this custom {title}",
                                 on_click=edit, width=40, height=40, visible=False,
                                 style=ft.ButtonStyle(icon_color={ft.ControlState.DEFAULT: S.ACCENT,
                                                                  ft.ControlState.DISABLED: S.FAINT},
                                                      shape=ft.RoundedRectangleBorder(radius=4)))
        trash_btn = ft.IconButton(ft.Icons.DELETE_OUTLINE, icon_size=19, tooltip=f"Delete this custom {title}",
                                  on_click=delete, width=40, height=40, visible=False,
                                  style=ft.ButtonStyle(icon_color={ft.ControlState.DEFAULT: S.ACCENT,
                                                                   ft.ControlState.DISABLED: S.FAINT},
                                                       shape=ft.RoundedRectangleBorder(radius=4)))
        operator_buttons[kind] = (edit_btn, trash_btn)
        return ft.Row([dropdown, edit_btn, trash_btn], spacing=2)

    def sync_operator_buttons():
        for kind, (edit_btn, trash_btn) in operator_buttons.items():
            custom = operators.is_custom(kind, getattr(config, operators.KINDS[kind].setting))
            edit_btn.visible = trash_btn.visible = custom
            edit_btn.disabled = trash_btn.disabled = run_state["locked"]

    operator_btn = link_button(ft.Icons.EDIT_NOTE, "Add / edit custom operators", open_operator_dialog)
    refresh_operator_options()

    async def changed_plot_adjust(e):
        config.plot_adjust = plot_adjust_switch.value
        await refresh_plots()

    plot_adjust_switch = ft.Switch(value=config.plot_adjust, on_change=changed_plot_adjust)

    # Controls locked while a run exists (they define the run).
    run_locked = [*advanced_controls, *ps_switches.values(), *logic_switches.values(), s_train, s_noise, s_outl, s_val, s_gens, s_pop, s_cx, s_mut, target_dropdown, custom_btn, delete_btn,
                  *switches.values(), selection_dropdown, init_dropdown, operator_btn,
                  s_min_depth, s_max_depth, s_max_tree]

    # ---- run view: best individual ---------------------------------------

    expression_text = mono("", size=13.5, selectable=True)
    expression_footer = mono("", size=13, color=S.INK)

    # Generation slider (replaces the progress bar). Accent up to the generation shown,
    # light accent up to the generation computed so far. At the right end it follows the
    # run live; moved back, the view stays on that generation.
    scrub = {"gen": None}   # None = follow the run

    async def on_gen_slide(e):
        nonlocal last_snapshot
        if run is None or run.computed == 0:
            return
        computed = run.computed
        g = max(1, min(int(round(gen_slider.value)), computed))
        gen_slider.value = g
        scrub["gen"] = None if g >= computed else g
        last_snapshot = run.snapshot(g)
        show_snapshot(last_snapshot)
        gen_slider.update()
        expression_card.update()
        await refresh_plots()
        await refresh_tree()
        await refresh_population()

    gen_slider = ft.Slider(
        # no divisions: tick marks per generation would make the track look dotted;
        # the handler snaps to whole generations instead
        min=1, max=2, value=1, disabled=True, on_change=on_gen_slide,
        active_color=S.ACCENT, inactive_color=S.RULE, thumb_color=S.ACCENT,
        secondary_active_color="#E8A890", secondary_track_value=1,
        padding=ft.Padding.symmetric(horizontal=0, vertical=4), tooltip="Scroll through the generations",
    )

    def sync_gen_slider(live: gp.Snapshot):
        total = max(live.generations, 2)
        gen_slider.max = total
        gen_slider.secondary_track_value = max(1, live.generation)
        if scrub["gen"] is None:
            gen_slider.value = max(1, live.generation)

    def reset_readouts():
        expression_text.value = ""
        expression_footer.value = f"Generation – / {config.generations}"
        scrub["gen"] = None
        gen_slider.value = gen_slider.secondary_track_value = 1
        tree_btn.disabled = pop_btn.disabled = True

    def show_snapshot(snap: gp.Snapshot):
        n = snap.generation
        if n:
            expression_text.value = snap.best_expression
            if snap.best_scale:   # linear scaling: output = a + b · f(x)
                a, b = snap.best_scale
                expression_text.value = f"{a:.6g} + {b:.6g} · {snap.best_expression}"
            expression_footer.value = f"Generation {n} / {snap.generations}  ·  {snap.best_size} nodes"
            tree_btn.disabled = pop_btn.disabled = False

    # ---- tree view -------------------------------------------------------

    tree_state = {"dialog": None, "image": None, "key": None, "title": None, "follow": True, "nodes": None,
                  "uid": None, "mode": "tree", "depth": 4, "controls": None}

    def ancestry_nodes():
        if run is None or tree_state["uid"] is None:
            return {}
        return run.ancestry(tree_state["uid"], tree_state["depth"])

    def render_tree_view():
        """PNG (with display size) of the dialog's current view: the tree or the ancestry."""
        if tree_state["mode"] == "ancestry":
            nodes = ancestry_nodes()
            if nodes:
                return plots.render_ancestry(nodes, tree_state["uid"], run.settings["problem"].kind)
        return plots.render_tree(tree_state["nodes"])

    async def refresh_tree():
        """Render the current best individual's tree into the open tree dialog."""
        snap = last_snapshot
        if (tree_state["dialog"] is None or not tree_state["follow"] or snap is None
                or not snap.best_nodes):
            return
        key = (snap.generation, snap.best_expression)
        if key == tree_state["key"]:
            return
        tree_state["key"] = key
        tree_state["nodes"], tree_state["uid"] = snap.best_nodes, snap.best_uid
        png, w, h = await asyncio.to_thread(render_tree_view)
        if tree_state["dialog"] is None:
            return
        tree_state["image"].src = png
        tree_state["image"].width, tree_state["image"].height = w, h
        tree_state["title"].value = f"Generation {snap.generation}  ·  {snap.best_size} nodes  ·  depth {snap.best_depth}"
        tree_state["dialog"].update()

    def close_tree(_=None):
        tree_state.update(dialog=None, key=None)
        page.pop_dialog()

    async def open_tree(e):
        snap = last_snapshot
        if snap is None or not snap.best_nodes:
            return
        await show_tree(snap.best_nodes, "Best individual",
                        f"Generation {snap.generation}  ·  {snap.best_size} nodes  ·  depth {snap.best_depth}",
                        follow=True, key=(snap.generation, snap.best_expression), uid=snap.best_uid)

    async def show_tree(nodes, heading, subtitle, follow=False, key=None, uid=None):
        """Tree dialog; `follow` = keep showing the current best individual. The switch in the title
        shows the individual's ancestry instead: its parents, their parents, … over the last generations."""
        tree_state.update(nodes=nodes, uid=uid)
        if tree_state["mode"] == "ancestry" and (uid is None or run is None):
            tree_state["mode"] = "tree"
        png, w, h = await asyncio.to_thread(render_tree_view)
        # Small trees are shown at their natural size, large ones are scaled to fit;
        # scroll / pinch to zoom, drag to pan.
        image = ft.Image(src=png, width=w, height=h, fit=ft.BoxFit.CONTAIN,
                         gapless_playback=True, filter_quality=ft.FilterQuality.HIGH)
        title = mono(subtitle, size=13, color=S.INK)
        grab = ft.GestureDetector(content=ft.Container(image, alignment=ft.Alignment.CENTER),
                                  mouse_cursor=ft.MouseCursor.GRAB)

        def cursor(c):
            grab.mouse_cursor = c
            grab.update()

        viewer = ft.InteractiveViewer(content=grab, min_scale=0.5, max_scale=8, boundary_margin=ft.Margin.all(200),
                                      on_interaction_start=lambda e: cursor(ft.MouseCursor.GRABBING),
                                      on_interaction_end=lambda e: cursor(ft.MouseCursor.GRAB))
        win_w = page.width or page.window.width or 1400
        win_h = page.height or page.window.height or 900

        async def redraw():
            png_, w_, h_ = await asyncio.to_thread(render_tree_view)
            image.src, image.width, image.height = png_, w_, h_
            depth_dd.visible = tree_state["mode"] == "ancestry"
            dialog.update()

        async def switch_view(e):
            tree_state["mode"] = view_seg.selected[0] if view_seg.selected else "tree"
            await redraw()

        async def changed_depth(e):
            tree_state["depth"] = int(depth_dd.value)
            await redraw()

        view_seg = ft.SegmentedButton(
            segments=[ft.Segment(value="tree", label=ft.Text("Parse tree")),
                      ft.Segment(value="ancestry", label=ft.Text("Ancestry tree"))],
            selected=[tree_state["mode"]], on_change=switch_view, show_selected_icon=False,
            disabled=uid is None or run is None,
            style=ft.ButtonStyle(color={ft.ControlState.SELECTED: "#FFFFFF", ft.ControlState.DEFAULT: S.INK},
                                 bgcolor={ft.ControlState.SELECTED: S.INK, ft.ControlState.DEFAULT: S.PANEL},
                                 side=ft.BorderSide(1, S.RULE), shape=ft.RoundedRectangleBorder(radius=4)))
        depth_dd = ft.Dropdown(
            value=str(tree_state["depth"]), options=[ft.DropdownOption(key=str(d), text=f"{d} generations back")
                                                    for d in (2, 4, 6, 8, 12)],
            on_select=changed_depth, width=190, dense=True, filled=True, fill_color=S.PANEL, border_color=S.RULE,
            focused_border_color=S.INK, border_radius=4, text_style=ft.TextStyle(size=13, color=S.INK),
            menu_style=ft.MenuStyle(bgcolor=S.PANEL), enable_search=False,
            visible=tree_state["mode"] == "ancestry")
        dialog = ft.AlertDialog(
            modal=False, bgcolor=S.PANEL, shape=ft.RoundedRectangleBorder(radius=6),
            title=ft.Row([ft.Text(heading, size=18, weight=ft.FontWeight.W_600, color=S.INK), view_seg, depth_dd,
                          ft.Container(expand=True), title], spacing=16,
                         vertical_alignment=ft.CrossAxisAlignment.CENTER),
            content=ft.Container(plot_menu("tree", viewer), width=min(1200, win_w - 140), height=min(760, win_h - 230),
                                 bgcolor=S.PANEL, border=ft.Border.all(1, S.RULE), border_radius=4,
                                 clip_behavior=ft.ClipBehavior.HARD_EDGE),
            actions=[ft.TextButton("Close", on_click=close_tree, style=ft.ButtonStyle(color=S.INK))],
            on_dismiss=lambda e: tree_state.update(dialog=None, key=None),
        )
        tree_state.update(dialog=dialog, image=image, title=title, key=key, follow=follow)
        page.show_dialog(dialog)

    tree_btn = link_button(ft.Icons.ACCOUNT_TREE_OUTLINED, "Show tree", open_tree)

    async def open_population_click(e):
        await open_population()

    pop_btn = link_button(ft.Icons.GRID_VIEW, "Show population", open_population_click)

    # ---- transport -------------------------------------------------------

    play_icon = ft.Icon(ft.Icons.PLAY_ARROW_ROUNDED, color="#FFFFFF", size=22)
    play_label = ft.Text("Run", size=14, color="#FFFFFF", weight=ft.FontWeight.W_600)

    def transport_button(icon, tooltip, on_click):
        return ft.IconButton(
            icon=icon, tooltip=tooltip, on_click=on_click, icon_size=20, width=44, height=44,
            style=ft.ButtonStyle(
                shape=ft.RoundedRectangleBorder(radius=4),
                side={ft.ControlState.DEFAULT: ft.BorderSide(1, S.FAINT),
                      ft.ControlState.DISABLED: ft.BorderSide(1, S.RULE)},
                icon_color={ft.ControlState.DEFAULT: S.INK, ft.ControlState.DISABLED: S.FAINT},
                bgcolor={ft.ControlState.DEFAULT: S.PANEL, ft.ControlState.DISABLED: S.SIDEBAR,
                         ft.ControlState.HOVERED: S.PAPER},
            ),
        )

    async def on_play_click(e):
        await toggle_play()

    async def on_step_click(e):
        await step()

    async def on_reset_click(e):
        await reset()

    play_btn = ft.Container(
        content=ft.Row([play_icon, play_label], spacing=6, alignment=ft.MainAxisAlignment.CENTER),
        bgcolor=S.ACCENT, border_radius=4, height=44, expand=True, on_click=on_play_click,
        ink=True, tooltip="Run / pause  (Space)",
    )
    step_btn = transport_button(ft.Icons.SKIP_NEXT_ROUNDED, "Next generation  (→)", on_step_click)
    reset_btn = transport_button(ft.Icons.RESTART_ALT_ROUNDED, "Reset run  (R)", on_reset_click)

    def apply_state(state):
        """Sync buttons and locked inputs with the run state."""
        if state == "running":
            play_icon.icon, play_label.value, play_btn.bgcolor = ft.Icons.PAUSE_ROUNDED, "Pause", S.INK
        elif state == "paused":
            play_icon.icon, play_label.value, play_btn.bgcolor = ft.Icons.PLAY_ARROW_ROUNDED, "Resume", S.ACCENT
        elif state in ("finished", "error"):
            play_icon.icon, play_label.value, play_btn.bgcolor = ft.Icons.REPLAY_ROUNDED, "Run again", S.ACCENT
        else:
            play_icon.icon, play_label.value, play_btn.bgcolor = ft.Icons.PLAY_ARROW_ROUNDED, "Run", S.ACCENT

        step_btn.disabled = state not in ("idle", "paused")
        reset_btn.disabled = state == "idle"

        locked = run_state["locked"] = state in ("running", "paused")
        for c in run_locked:
            c.disabled = locked
        sync_selection_controls(locked)
        sync_advanced_controls()
        sync_operator_buttons()
        if function_lock["locked"] != run_state["locked"]:   # apply_state runs every generation: rebuild only on change
            rebuild_function_rows()
        function_btn.disabled = settings_reset_btn.disabled = run_state["locked"]
        sync_problem_views()
        tree_btn.disabled = pop_btn.disabled = last_snapshot is None or not last_snapshot.best_nodes
        gen_slider.disabled = run is None

    async def watch(current: gp.GPRun):
        """Mirror the run into the UI. Only re-renders when something changed."""
        nonlocal last_snapshot
        seen = (-1, None)
        while run is current:
            snap = current.snapshot()
            key = (snap.generation, snap.state)
            if key != seen:
                sync_gen_slider(snap)
                following = scrub["gen"] is None
                if following:
                    last_snapshot = snap
                    show_snapshot(snap)
                    if key[0] != seen[0]:
                        await refresh_plots()
                if run is not current:  # reset while rendering
                    return
                apply_state(snap.state)
                page.update()
                current.ack(snap.generation)   # the evolution may continue with the next generation
                if following and key[0] != seen[0]:
                    await refresh_tree()
                    await refresh_population()
                if snap.state == "error" and seen[1] != "error":
                    show_message(f"Run stopped: {snap.error}")
                seen = key
            if snap.state in ("finished", "stopped", "error"):
                return
            await asyncio.sleep(0.05)

    async def start_run(paused=False):
        nonlocal run, last_snapshot
        if run is not None:
            run.stop()
        last_snapshot = None
        reset_readouts()
        run = gp.GPRun(config.run_settings(), start_paused=paused)
        run.start()
        apply_state("paused" if paused else "running")
        page.update()
        page.run_task(watch, run)

    async def toggle_play():
        state = run.snapshot().state if run else "idle"
        if state == "running":
            run.pause()
        elif state == "paused":
            run.resume()
        else:
            await start_run()

    async def step():
        state = run.snapshot().state if run else "idle"
        if state == "idle" or run is None:
            await start_run(paused=True)
        elif state == "paused":
            run.step()

    async def reset():
        nonlocal run, last_snapshot
        if run is None:
            return
        run.stop()
        run, last_snapshot = None, None
        reset_readouts()
        apply_state("idle")
        await refresh_plots()
        page.update()

    async def on_key(e: ft.KeyboardEvent):
        # Keyboard events reach the page whatever has the focus: the shortcuts belong to the Run tab
        # only (typing a space in, say, a configuration name in Experiments must not start a run),
        # and not while a number is typed into a slider's field.
        if (e.ctrl or e.meta or e.alt or dialog_state["open"] or current_tab["index"] != 0
                or ParamSlider.editing is not None):
            return
        if e.key in (" ", "Space"):
            await toggle_play()
        elif e.key == "Arrow Right" and not step_btn.disabled:
            await step()
        elif e.key.upper() == "R" and run is not None:
            await reset()

    current_tab = {"index": 0}   # 0 = Run (see tab_changed)
    page.on_keyboard_event = on_key

    def show_message(text):
        toast(text, seconds=4)

    # ---- layout: run tab -------------------------------------------------

    def section(title, controls):
        return ft.Column([caps(title), ft.Container(height=2), *controls], spacing=10)

    sidebar = ft.Container(
        width=340, bgcolor=S.SIDEBAR,
        border=ft.Border(right=ft.BorderSide(1, S.RULE)),
        padding=ft.Padding.only(left=24, right=24, top=20, bottom=20),
        content=ft.Column(
            [
                ft.Column(
                    [
                        section("Problem", [
                            ft.Column([ft.Row([ft.Text("Benchmark function", size=13, color=S.INK), custom_btn],
                                              alignment=ft.MainAxisAlignment.SPACE_BETWEEN,
                                              vertical_alignment=ft.CrossAxisAlignment.CENTER),
                                       ft.Row([target_dropdown, delete_btn], spacing=4)], spacing=4),
                            s_train.control, s_noise.control, s_outl.control, s_val.control,
                        ]),
                        ft.Container(height=14),
                        section("Evolution", [s_gens.control, s_pop.control, s_cx.control, s_mut.control]),
                    ],
                    spacing=0, expand=True, scroll=ft.ScrollMode.AUTO,
                ),
                ft.Divider(),
                ft.Container(height=6),
                ft.Row([reset_btn, step_btn, hand(play_btn)], spacing=8),
            ],
            spacing=8,
        ),
    )

    error_title, fit_title = caps("Mean squared error"), caps("Best individual vs. data")
    main_titles = {"error": "", "fit": ""}

    def sync_card_titles():
        for which, title in (("error", error_title), ("fit", fit_title)):
            mode = card_view[which]
            title.value = main_titles[which] if mode == "main" else plots.DYNAMICS_VIEWS[mode]

    def view_menu(which, title):
        """Card title that opens a menu to switch the card between its main view and
        the population dynamics views."""
        async def pick(e):
            card_view[which] = e.control.data
            sync_card_titles()
            sync_problem_views()
            page.update()
            await refresh_plots()

        def item(key, label):
            return ft.PopupMenuItem(content=ft.Text(label, size=13, color=S.INK), data=key, on_click=pick,
                                    checked=False)

        main_label = "Error over generations" if which == "error" else "Problem view (data, cases, truth table)"
        return ft.PopupMenuButton(
            content=ft.Row([title, ft.Icon(ft.Icons.EXPAND_MORE, size=16, color=S.MUTED)], spacing=2, tight=True),
            items=[item("main", main_label)] + [item(k, v) for k, v in plots.DYNAMICS_VIEWS.items()],
            tooltip="Choose what this plot shows", menu_position=ft.PopupMenuPosition.UNDER,
            bgcolor=S.PANEL)

    # ---- copy / save plots (right-click menu) -----------------------------

    clipboard = ft.Clipboard()
    file_picker = ft.FilePicker()
    url_launcher = ft.UrlLauncher()

    async def save_bytes(data, title, file_name, ext):
        """Save an export: a save dialog on the desktop, a normal download in the browser."""
        try:
            if page.web:
                await url_launcher.launch_url(webserver.offer(data, file_name), web_only_window_name="_self")
                toast(f"Downloading  {file_name}", seconds=4)
                return
            path = await file_picker.save_file(dialog_title=title, file_name=file_name,
                                               file_type=ft.FilePickerFileType.CUSTOM, allowed_extensions=[ext],
                                               src_bytes=data)
            if path:
                toast(f"Saved  {os.path.basename(path)}", seconds=4)
        except Exception as exc:
            toast(f"Could not save the file: {exc}", seconds=5)

    def plot_file_name(which, ext):
        if which.startswith("exp_"):
            name = file_stem(problem_title(exp["view"])) if exp["view"] else "experiment"
            part = "final" if which == "exp_final" else ("convergence" if exp_view["mode"] == "main"
                                                         else exp_view["mode"])
            return f"gpexplore_experiment_{name}_{part}.{ext}"
        settings = run.settings if run is not None else config.run_settings()
        problem = settings["problem"]
        name = problems.title(problem)
        name = "".join(c if c.isalnum() or c in "-_" else "-" for c in name).strip("-")[:40] or "problem"
        gen = f"_gen{last_snapshot.generation}" if last_snapshot is not None and last_snapshot.generation else ""
        part = which if which in ("error", "tree") else ("cases" if problem.kind != "sr" else "fit")
        if which == "tree" and tree_state["mode"] == "ancestry":
            part = "ancestry"
        if which in card_view and card_view[which] != "main":
            part = card_view[which]
        return f"gpexplore_{name}_{part}{gen}.{ext}"

    def render_for_export(which, fmt):
        """The plot exactly as shown (current generation, size and 3D view) in `fmt`."""
        snap = last_snapshot if run is not None else None
        settings = run.settings if run is not None else config.run_settings()
        with plots.output_format(fmt):
            if which.startswith("exp_"):
                return render_experiment_plot(which)
            if which == "tree":
                return render_tree_view()[0]   # the tree or the ancestry, as shown
            return render_card(which, snap, settings, plot_sizes[which])

    async def plot_menu_action(which, index):
        try:
            if index == 0:   # copy
                data = await asyncio.to_thread(render_for_export, which, "png")
                try:
                    await clipboard.set_image(data)
                except Exception:   # not supported by Flet on this desktop platform
                    await asyncio.to_thread(clipboard_image.copy_png, data)
                toast("Plot copied to the clipboard", seconds=3)
                return
            fmt = "png" if index == 1 else "pdf"
            data = await asyncio.to_thread(render_for_export, which, fmt)
            await save_bytes(data, f"Save plot as {fmt.upper()}", plot_file_name(which, fmt), fmt)
        except Exception as exc:  # e.g. clipboard not available on this platform
            toast(f"Could not export the plot: {exc}", seconds=5)

    def plot_menu(which, content):
        def item(icon, label):
            return ft.PopupMenuItem(content=ft.Row([ft.Icon(icon, size=17, color=S.INK),
                                                    ft.Text(label, size=13, color=S.INK)], spacing=10))

        async def selected(e: ft.ContextMenuSelectEvent):
            if e.item_index is not None:
                await plot_menu_action(which, e.item_index)

        return ft.ContextMenu(
            content=content, expand=True, on_select=selected,
            secondary_items=[item(ft.Icons.CONTENT_COPY, "Copy image"),
                             item(ft.Icons.IMAGE_OUTLINED, "Save as PNG …"),
                             item(ft.Icons.PICTURE_AS_PDF_OUTLINED, "Save as PDF …")],
        )

    def plot_card(title, image, which, extra=None):
        # only the left card can switch views; the right one always shows the problem
        menu = view_menu(which, title) if which == "error" else title
        header = ft.Row([menu, ft.Container(extra, expand=True, alignment=ft.Alignment.CENTER_RIGHT)],
                        spacing=12, vertical_alignment=ft.CrossAxisAlignment.CENTER) if extra else \
            ft.Row([menu], spacing=0)
        return card(
            ft.Column(
                [header,
                 ft.Container(plot_menu(which, image), expand=True, on_size_change=plot_resized(which),
                              size_change_interval=0)],
                # STRETCH: the plot container takes the full card width (and reports it), instead
                # of shrinking to the width of the previously rendered image.
                spacing=6, expand=True, horizontal_alignment=ft.CrossAxisAlignment.STRETCH,
            ),
            padding=ft.Padding.only(left=14, right=14, top=12, bottom=10), expand=1,
        )

    section_names = {"sr": "Symbolic regression functions", "ps": "Program synthesis building blocks",
                     "logic": "Logic gates"}
    section_titles = {k: caps(v) for k, v in section_names.items()}

    def sync_block_switches():
        """Grey out switches that have no effect on the selected problem (e.g. sin and cos
        for a Boolean problem) and say so in the section title."""
        usable = gp.applicable_flags(config.run_settings())
        locked = run_state["locked"]
        groups = {"sr": switches, "ps": ps_switches, "logic": logic_switches}
        for key, group in groups.items():
            for flag, switch in group.items():
                switch.disabled = locked or flag not in usable
            unused = not any(flag in usable for flag in group)
            title = section_names[key] + ("  ·  not used for this problem" if unused else "")
            section_titles[key].value = title

    def sync_problem_views():
        """Controls that depend on the selected problem (regression vs. program synthesis)."""
        problem = current_problem()
        sync_block_switches()
        ps = problem.kind != "sr"            # program synthesis or Boolean logic
        logic = problem.kind == "logic"
        dataset = getattr(problem, "dataset", False)
        two = not ps and problem.inputs == 2 and not dataset
        x_state.value, x_state.color = ("not used", S.INK) if ps or dataset else ("in use", S.TEAL)
        y_state.value, y_state.color = ("in use", S.TEAL) if two else ("not used", S.INK)
        delete_btn.visible = problem.custom
        fit_gesture.mouse_cursor = (ft.MouseCursor.GRAB if two else
                                    ft.MouseCursor.CLICK if ps else ft.MouseCursor.BASIC)
        main_titles["error"] = "Solved cases" if ps else gp.FITNESS_TITLES.get(config.fitness_function,
                                                                                  "Mean squared error")
        case_info.value = ""
        main_titles["fit"] = ("Truth table (Karnaugh map)" if logic else "Training cases" if ps else
                              "Prediction vs. measured value" if dataset else "Best individual vs. data")
        sync_card_titles()
        if card_view["fit"] != "main":
            fit_gesture.mouse_cursor = ft.MouseCursor.BASIC
        s_val.label_text.value = "Test set size" if ps else "Validation set size"
        if logic or dataset:
            # Boolean problems use the whole truth table, data sets their fixed split
            s_train.disabled = s_val.disabled = True
            logic_signature.value = (f"{', '.join(problem.arg_names)} → output   ({problem.train_size} rows)"
                                     if logic else "—")
        else:
            s_train.disabled = s_val.disabled = run_state["locked"]
            logic_signature.value = "—"
        if ps and not logic:
            # noise and outliers only make sense for numeric regression targets
            s_noise.disabled = s_outl.disabled = True
            params = ", ".join(f"{n}: {synthesis.TYPE_NAMES[t]}" for n, t in zip(problem.arg_names, problem.arg_types))
            consts = synthesis.format_constants(problem.int_constants, getattr(problem, "float_constants", ()),
                                                problem.str_constants)
            ps_signature.value = (f"{params} → {synthesis.TYPE_NAMES[problem.ret]}"
                                  + (f"   constants {consts}" if consts else ""))
        elif logic:
            s_noise.disabled = s_outl.disabled = True
            ps_signature.value = "—"
        else:
            s_noise.disabled = s_outl.disabled = run_state["locked"] or dataset
            ps_signature.value = "—"

    plots_row = side_by_side(plot_card(error_title, error_img, "error"),
                             plot_card(fit_title, fit_gesture, "fit", case_info), expand=True)

    expression_card = card(
        ft.Column(
            [
                ft.Row([caps("Best individual"), ft.Container(expand=True), pop_btn, tree_btn], spacing=4,
                       vertical_alignment=ft.CrossAxisAlignment.CENTER),
                ft.Column([expression_text], scroll=ft.ScrollMode.AUTO, expand=True),
                expression_footer,
            ],
            spacing=8,
        ),
        padding=ft.Padding.only(left=16, right=12, top=10, bottom=14), height=308,
    )

    def resize_expression(dy):
        """Splitter between the plots and the best individual (dragging down makes the plots taller)."""
        limit = max(140, (page.height or page.window.height or 900) - 360)
        height = min(limit, max(110, expression_card.height - dy))
        if height != expression_card.height:
            expression_card.height = height
            expression_card.update()

    run_view = ft.Row(
        [
            sidebar,
            ft.Container(
                ft.Column([plots_row, splitter(resize_expression), expression_card, ft.Container(height=10),
                           gen_slider], spacing=0),
                expand=True, padding=ft.Padding.only(left=20, right=20, top=20, bottom=18),
            ),
        ],
        spacing=0, expand=True, vertical_alignment=ft.CrossAxisAlignment.STRETCH,
    )

    # ---- layout: list-style tabs ----------------------------------------

    def tag(text):
        return ft.Container(ft.Text(text, size=11.5, color=S.INK, font_family=S.MONO),
                            padding=ft.Padding.symmetric(horizontal=7, vertical=2),
                            border=ft.Border.all(1, S.RULE), border_radius=3)

    def list_row(name, kind, description, control, name_mono=True, control_width=410):
        return ft.Container(
            ft.Row(
                [
                    ft.Container(mono(name, size=13) if name_mono else ft.Text(name, size=13, color=S.INK,
                                                                              weight=ft.FontWeight.W_500), width=240),
                    ft.Container(tag(kind) if kind else None, width=100, alignment=ft.Alignment.CENTER_LEFT),
                    ft.Text(description, size=13, color=S.INK, expand=True),
                    ft.Container(control, width=control_width, alignment=ft.Alignment.CENTER_LEFT),
                ],
                spacing=16, vertical_alignment=ft.CrossAxisAlignment.CENTER,
            ),
            padding=ft.Padding.symmetric(horizontal=20, vertical=10),
            border=ft.Border(bottom=ft.BorderSide(1, S.RULE_SOFT)),
        )

    def list_header(*labels, control_width=410):
        widths = [240, 100, None, control_width]
        return ft.Container(
            ft.Row([ft.Container(caps(l, size=9.5), width=w, expand=w is None) for l, w in zip(labels, widths)],
                   spacing=16),
            padding=ft.Padding.symmetric(horizontal=20, vertical=10),
            border=ft.Border(bottom=ft.BorderSide(1, S.RULE)),
        )

    def page_intro(title):
        return ft.Text(title, size=22, weight=ft.FontWeight.W_600, color=S.INK)

    # ---- population browser ---------------------------------------------

    pop_title = mono("Start a run to see its population.", size=13, color=S.INK)
    pop_sort = {"col": 1, "desc": None}   # column of (expression, training, validation, size, depth)
    pop_list = ft.ListView(expand=True, spacing=0)
    pop_header_row = ft.Row(spacing=16)
    pop_state = {"dialog": None}
    POP_COLS = [("#", 46), ("training", 120), ("validation", 120), ("size", 70), ("depth", 70), ("expression", None)]

    def pop_metric_names(problem):
        if problem.kind == "sr":
            return "training error", "validation error"
        if problem.kind == "ps":
            return "training solved", "test solved*"
        return "rows correct", ""

    def pop_fmt(problem, v):
        if v is None or (isinstance(v, float) and not math.isfinite(v)):
            return "inf"
        return f"{v:.4g}" if problem.kind == "sr" else f"{v:.1f} %"

    async def refresh_population():
        if pop_state["dialog"] is None:
            return
        snap = last_snapshot
        settings = run.settings if run is not None else config.run_settings()
        problem = settings["problem"]
        names = pop_metric_names(problem)
        labels = ["#", names[0], names[1], "size", "depth", "expression"]

        def header(i, text, width):
            async def sort(e):
                pop_sort["desc"] = (not pop_sort["desc"]) if pop_sort["col"] == i else None
                pop_sort["col"] = i
                await refresh_population()
            arrow = "" if pop_sort["col"] != i else (" ↓" if sort_desc() else " ↑")
            return hand(ft.Container(caps(text + arrow, size=9.5), on_click=sort if i > 0 else None, height=32,
                                     alignment=ft.Alignment.CENTER_LEFT, width=width, expand=width is None), i > 0)

        def sort_desc():
            if pop_sort["desc"] is not None:
                return pop_sort["desc"]
            # default: best first (lower error for regression, more solved cases otherwise)
            return problem.kind != "sr" if pop_sort["col"] in (1, 2) else False

        pop_header_row.controls = [header(i, t, w) for i, ((_, w), t) in enumerate(zip(POP_COLS, labels))
                                   if not (i == 2 and not t)]
        if snap is None or not snap.population:
            pop_title.value = "Start a run to see its population."
            pop_list.controls = []
            page.update()
            return
        rows = list(snap.population)
        col = pop_sort["col"]
        key = (lambda r: r[col] if isinstance(r[col], (int, float)) and math.isfinite(r[col]) else math.inf) \
            if col else (lambda r: 0)
        ranked = sorted(rows, key=lambda r: r[1] if math.isfinite(r[1]) else math.inf,
                        reverse=problem.kind != "sr")
        rank = {id(r): i + 1 for i, r in enumerate(ranked)}
        rows.sort(key=key, reverse=sort_desc())
        if col == 0:
            rows = ranked

        def row(r):
            async def open_individual(e):
                nodes = gp.tree_nodes_from_string(r[0], set(snap.variables))
                await show_tree(nodes, f"Individual #{rank[id(r)]}",
                                f"Generation {snap.generation}  ·  {r[3]} nodes  ·  depth {r[4]}",
                                uid=r[5] if len(r) > 5 else None)
            cells = [mono(str(rank[id(r)]), size=12.5), mono(pop_fmt(problem, r[1]), size=12.5),
                     mono(pop_fmt(problem, r[2]), size=12.5), mono(str(r[3]), size=12.5), mono(str(r[4]), size=12.5),
                     mono(r[0], size=12, max_lines=1, overflow=ft.TextOverflow.ELLIPSIS)]
            widths = [w for _, w in POP_COLS]
            items = [ft.Container(c, width=w, expand=w is None) for c, w in zip(cells, widths)
                     if not (c is cells[2] and not names[1])]
            return hand(ft.Container(ft.Row(items, spacing=16), padding=ft.Padding.symmetric(horizontal=20, vertical=7),
                                     border=ft.Border(bottom=ft.BorderSide(1, S.RULE_SOFT)), on_click=open_individual,
                                     ink=True, tooltip="Show the tree"))
        unique = len({r[0] for r in snap.population})
        pop_title.value = (f"Generation {snap.generation}  ·  {len(rows)} individuals  ·  {unique} different trees"
                           + ("   (* on a sample of the test set)" if problem.kind == "ps" else ""))
        pop_list.controls = [row(r) for r in rows]
        page.update()

    def close_population(_=None):
        pop_state["dialog"] = None
        page.pop_dialog()

    async def open_population():
        if last_snapshot is None or not last_snapshot.population:
            return
        win_w = page.width or page.window.width or 1400
        win_h = page.height or page.window.height or 900
        dialog = ft.AlertDialog(
            modal=False, bgcolor=S.PANEL, shape=ft.RoundedRectangleBorder(radius=6),
            title=ft.Row([ft.Text("Population", size=18, weight=ft.FontWeight.W_600, color=S.INK),
                          ft.Container(expand=True), pop_title], vertical_alignment=ft.CrossAxisAlignment.CENTER),
            content=ft.Container(
                ft.Column([ft.Container(pop_header_row, padding=ft.Padding.symmetric(horizontal=20, vertical=6),
                                        border=ft.Border(bottom=ft.BorderSide(1, S.RULE))),
                           pop_list], spacing=0, expand=True),
                width=min(1200, win_w - 140), height=min(760, win_h - 230), bgcolor=S.PANEL,
                border=ft.Border.all(1, S.RULE), border_radius=4, clip_behavior=ft.ClipBehavior.HARD_EDGE),
            actions=[ft.TextButton("Close", on_click=close_population, style=ft.ButtonStyle(color=S.INK))],
            on_dismiss=lambda e: pop_state.update(dialog=None),
        )
        pop_state["dialog"] = dialog
        page.show_dialog(dialog)
        await refresh_population()

    # ---- custom functions of the function sets ------------------------------

    function_rows = {area: ft.Column(spacing=0) for area in operators.AREAS}
    FUNCTION_AREAS = list(operators.AREAS.items())

    def check_function_code(area, code):
        name, _, _, signature = operators.check_function(area, code)
        return name, f"{signature}  ·  custom {operators.AREAS[area].lower()} function"

    def function_saved(area, name):
        rebuild_function_rows()
        sync_problem_views()

    def open_function_dialog(e=None, area=None, code=None):
        kind = current_problem().kind
        open_code_dialog("Add or edit a custom function", FUNCTION_AREAS, area or ("sr" if kind == "sr" else kind),
                         operators.FUNCTIONS, operators.FUNCTION_TEMPLATES, check_function_code,
                         operators.test_function, operators.add_function, function_saved, code, noun="Function")

    function_lock = {"locked": None}

    def rebuild_function_rows():
        locked = run_state["locked"]
        function_lock["locked"] = locked
        for area, column in function_rows.items():
            rows = []
            for name, code in operators.FUNCTIONS[area].items():
                signature = operators.check_function(area, code)[3]

                def toggled(e, name=name):
                    operators.set_enabled(name, e.control.value)

                def edit(e, area=area, code=code):
                    open_function_dialog(area=area, code=code)

                async def delete(e, area=area, name=name, code=code):
                    enabled = operators.ENABLED.get(name, True)
                    operators.delete_function(area, name)
                    rebuild_function_rows()
                    page.update()

                    async def undo():
                        operators.FUNCTIONS[area][name] = code
                        operators.ENABLED[name] = enabled
                        operators.save_functions()
                        rebuild_function_rows()
                        page.update()

                    toast(f"Deleted  {name}", action="Undo", on_action=undo, seconds=5)

                icon_style = ft.ButtonStyle(icon_color={ft.ControlState.DEFAULT: S.ACCENT, ft.ControlState.DISABLED: S.FAINT},
                                            shape=ft.RoundedRectangleBorder(radius=4))
                trash_style = ft.ButtonStyle(icon_color={ft.ControlState.DEFAULT: S.ACCENT,
                                                         ft.ControlState.DISABLED: S.FAINT},
                                             shape=ft.RoundedRectangleBorder(radius=4))
                description = ft.Row([
                    mono(signature, size=12.5, expand=True, max_lines=1, overflow=ft.TextOverflow.ELLIPSIS),
                    ft.IconButton(ft.Icons.EDIT_OUTLINED, icon_size=18, tooltip="Edit this custom function",
                                  on_click=edit, disabled=locked, width=36, height=36, style=icon_style),
                    ft.IconButton(ft.Icons.DELETE_OUTLINE, icon_size=19, tooltip="Delete this custom function",
                                  on_click=delete, disabled=locked, width=36, height=36, style=trash_style)],
                    spacing=2, expand=True, vertical_alignment=ft.CrossAxisAlignment.CENTER)
                switch = ft.Switch(value=operators.ENABLED.get(name, True), on_change=toggled, disabled=locked)
                rows.append(ft.Container(
                    ft.Row([ft.Container(mono(name, size=13), width=240),
                            ft.Container(tag("Custom"), width=100, alignment=ft.Alignment.CENTER_LEFT),
                            description, ft.Container(switch, width=80, alignment=ft.Alignment.CENTER_LEFT)],
                           spacing=16, vertical_alignment=ft.CrossAxisAlignment.CENTER),
                    padding=ft.Padding.symmetric(horizontal=20, vertical=4),
                    border=ft.Border(bottom=ft.BorderSide(1, S.RULE_SOFT))))
            column.controls = rows

    function_btn = link_button(ft.Icons.EDIT_NOTE, "Add / edit custom functions", open_function_dialog)
    rebuild_function_rows()

    primitives_view = ft.Container(
        ft.Column(
            [
                ft.Row([page_intro("Function & terminal set"), ft.Container(expand=True), function_btn],
                       vertical_alignment=ft.CrossAxisAlignment.CENTER),
                section_titles["sr"],
                card(ft.Column([list_header("Name", "Type", "Description", "Use", control_width=80),
                                *[list_row(n, k, d, switches[f], control_width=80) for n, k, d, f in FUNCTION_ROWS],
                                function_rows["sr"],
                                list_row("x", "Terminal", "First input variable (all problems).", x_state, control_width=80),
                                list_row("y", "Terminal", "Second input variable (two-input problems only).", y_state,
                                         control_width=80)],
                               spacing=0), padding=0),
                ft.Container(height=4),
                section_titles["ps"],
                card(ft.Column([list_header("Name", "Type", "Description", "Use", control_width=80),
                                *[list_row(label, "Function", desc, ps_switches[flag], control_width=80)
                                  for flag, label, desc, _ in synthesis.GROUPS],
                                function_rows["ps"],
                                list_row(synthesis.ELEM, "Terminal", "Current element inside a loop.",
                                         mono("with loops", size=12.5), control_width=80),
                                list_row("inputs", "Terminal", "Inputs and constants of the selected problem.",
                                         ft.Container(), control_width=0),
                                ft.Container(ps_signature, padding=ft.Padding.only(left=20, right=20, bottom=12))],
                               spacing=0), padding=0),
                ft.Container(height=4),
                section_titles["logic"],
                card(ft.Column([list_header("Name", "Type", "Description", "Use", control_width=80),
                                *[list_row(gate.name, "Gate", synthesis.GATE_DESCRIPTIONS[flag], logic_switches[flag],
                                           control_width=80) for flag, gate in synthesis.LOGIC_GATES],
                                function_rows["logic"],
                                list_row("inputs", "Terminal", "Input bits of the selected Boolean problem.",
                                         ft.Container(), control_width=0),
                                ft.Container(logic_signature, padding=ft.Padding.only(left=20, right=20, bottom=12))],
                               spacing=0), padding=0),
            ],
            spacing=20, scroll=ft.ScrollMode.AUTO,
        ),
        padding=ft.Padding.only(left=32, right=32, top=24, bottom=24), expand=True,
    )

    async def toggle_expert(e):
        """All sliders (also in the sidebar and the Experiments tab) become free number fields; leaving
        expert mode puts values outside a slider's range to its minimum or maximum."""
        config.expert_mode = bool(e.control.value)
        await apply_expert_mode()
        page.update()

    async def apply_expert_mode():
        for slider in ParamSlider.instances:
            if slider.set_expert(config.expert_mode) and slider.on_commit:
                result = slider.on_commit()   # e.g. new training data for a clamped training size
                if asyncio.iscoroutine(result):
                    await result
        page.update()

    expert_switch = ft.Switch(value=config.expert_mode, on_change=toggle_expert)

    advanced_card = card(ft.Column([
        list_header("Advanced parameter", "Scope", "Description", "Value"),
        list_row("Parsimony size", "GP", "Double tournament: the smaller of two individuals wins with "
                 "probability size / 2.",
                 s_parsimony.control, name_mono=False),
        list_row("Tournament order", "GP", "Double tournament: which of the two tournaments comes first.", dt_order_dd, name_mono=False),
        list_row("Fixed ε", "GP", "Error tolerance per case (ε-lexicase with fixed ε).", s_eps.control, name_mono=False),
        list_row("Parent sampling rate s", "GP", "Share of the population used to measure distances "
                 "between training cases (informed down-sampling and ROIDS).", s_rho.control, name_mono=False),
        list_row("Distance interval k", "GP", "Generations between two updates of the case distances "
                 "(informed down-sampling and ROIDS).", s_k.control, name_mono=False),
        list_row("Crossover", "GP", "One-point subtree crossover; leaf-biased picks the crossover point "
                 "among the inner nodes with the probability below (Koza).", operator_control("crossover"),
                 name_mono=False),
        list_row("Inner-node probability", "GP", "Leaf-biased crossover: probability that the crossover point "
                 "is an inner node rather than a leaf (Koza: 0.9).", s_inner.control, name_mono=False),
        list_row("Mutation", "GP", "Uniform: new random subtree; node replacement: one node by one of the same "
                 "arity; shrink: subtree by one of its own subtrees; ephemeral: change constants; mixed: "
                 "one of these at random.", operator_control("mutation"), name_mono=False),
        list_row("New subtree depth", "GP", "Uniform mutation (and random_subtree of custom mutations): "
                 "minimum and maximum depth of the new subtree.",
                 ft.Row([s_sub_min.control, s_sub_max.control], spacing=10), name_mono=False),
        list_row("New subtree method", "GP", "How the new subtree is built: full (all leaves at the drawn "
                 "depth), grow (leaves at any depth) or ramped half-and-half (either, at random).", subtree_dd,
                 name_mono=False),
        list_row("Ephemeral mutation", "GP", "Ephemeral mutation: replaces one random constant of the tree "
                 "or all of them by new random values.", ephemeral_dd, name_mono=False),
        list_row("Fitness function", "GP", "Symbolic regression only. Linear scaling (Keijzer 2003) fits "
                 "a + b·f(x) to the data before measuring the error.", operator_control("fitness"), name_mono=False),
        list_row("Random constants", "GP", "Symbolic regression only: range of the integer random constants "
                 "(ERC).", ft.Row([s_erc_lo.control, s_erc_hi.control], spacing=10), name_mono=False),
        list_row("Bloat control", "GP", "Depth limit (always), additionally a size limit, or lexicographic "
                 "parsimony: equal fitness → smaller tree wins tournaments.", operator_control("bloat"), name_mono=False),
        list_row("Max. tree size", "GP", "Maximum number of nodes (with the size limit).", s_size.control,
                 name_mono=False),
        list_row("Adaptive control", "GP", "Changes parameters during the run, e.g. with the generation or the "
                 "diversity: presets, or your own Python function (Add / edit custom operators → Adaptive control). "
                 "Its effect: the plot view \"Parameters over generations\".", operator_control("control"),
                 name_mono=False),
        list_row("Numba acceleration", "App", "Evaluates regression trees and runs lexicase selection as "
                 "compiled code (numba): same results, shorter runs." + ("" if fast.AVAILABLE else
                 " Not available: numba is not installed."), numba_switch, name_mono=False),
        list_row("Deep-tree acceleration", "App", "Symbolic regression: children are evaluated only from the changed "
                 "subtree upwards (Langdon & Banzhaf 2022); off: as in DEAP. Same results, shorter runs.",
                 deep_switch, name_mono=False),
        list_row("Expert mode", "App", "Number fields instead of sliders, everywhere in the app: values outside "
                 "the slider ranges become possible, only real limits remain (e.g. probabilities 0 – 1). "
                 "Switching back moves such values to the nearest end of the slider.", expert_switch, name_mono=False),
    ], spacing=0), padding=0)
    advanced_card.visible = config.show_advanced

    advanced_switch = ft.Switch(value=config.show_advanced)

    async def toggle_advanced(e):
        config.show_advanced = advanced_card.visible = bool(e.control.value)
        advanced_card.update()
        if config.show_advanced:   # bring the new rows into view: the switch moves to the top
            await asyncio.sleep(0.05)   # the card has to be laid out first
            await parameters_column.scroll_to(scroll_key="advanced_switch", duration=600)

    def open_reset_dialog(e):
        """All parameters back to their defaults, after typing "yes"."""
        answer = ft.TextField(label='Type "yes" to confirm', dense=True, autofocus=True, border_color=S.RULE,
                              focused_border_color=S.INK, border_radius=4, width=260,
                              text_style=ft.TextStyle(size=13, color=S.INK))
        confirm = ft.FilledButton(
            "Reset", disabled=True,
            style=ft.ButtonStyle(bgcolor={ft.ControlState.DEFAULT: S.ACCENT, ft.ControlState.DISABLED: S.RULE},
                                 color={ft.ControlState.DEFAULT: "#FFFFFF", ft.ControlState.DISABLED: S.FAINT},
                                 shape=ft.RoundedRectangleBorder(radius=4)))

        def typed(e):
            confirm.disabled = (answer.value or "").strip().lower() != "yes"
            confirm.update()

        async def do_reset(e):
            if (answer.value or "").strip().lower() != "yes":
                return
            page.pop_dialog()
            await reset_to_defaults()

        answer.on_change = typed
        answer.on_submit = do_reset
        confirm.on_click = do_reset
        page.show_dialog(ft.AlertDialog(
            modal=True, bgcolor=S.PANEL, shape=ft.RoundedRectangleBorder(radius=6),
            title=ft.Text("Reset all settings to their defaults?", size=18, weight=ft.FontWeight.W_600, color=S.INK),
            content=ft.Column([
                ft.Text("All parameters (problem, evolution, function set, additional and advanced parameters, "
                        "experiment settings) go back to their default values. Custom problems, operators and "
                        "functions stay saved.", size=13, color=S.INK),
                answer], tight=True, spacing=16, width=520),
            actions=[ft.TextButton("Cancel", on_click=lambda _: page.pop_dialog(), style=ft.ButtonStyle(color=S.INK)),
                     confirm],
            actions_alignment=ft.MainAxisAlignment.END))

    async def reset_to_defaults():
        if exp["running"]:
            toast("Stop the experiment first.", seconds=4)
            return
        config.reset_settings()
        config.exp_workers = max(1, CORES - 1)
        for slider in ParamSlider.instances:
            slider.set(getattr(config, slider.attr))
            slider.set_expert(config.expert_mode)
        for group in (switches, ps_switches, logic_switches):
            for flag, switch in group.items():
                switch.value = getattr(config, flag)
        seed_field.value, seed_field.error = str(config.seed), None
        subtree_dd.value, ephemeral_dd.value = config.subtree_method, config.ephemeral_mode
        dt_order_dd.value, downsampling_dd.value = config.double_tournament_order, config.downsampling
        numba_switch.value = config.numba_acceleration and fast.AVAILABLE
        deep_switch.value = config.deep_acceleration
        exp_seed_field.value, exp_seed_field.error = str(config.exp_seed), None
        plot_adjust_switch.value = config.plot_adjust
        advanced_switch.value = advanced_card.visible = config.show_advanced
        expert_switch.value = config.expert_mode
        refresh_operator_options()
        sync_selection_controls()
        sync_advanced_controls()
        await select_problem(config.target_function)
        toast("All settings reset to their defaults", seconds=4)
        page.update()

    settings_reset_btn = link_button(ft.Icons.RESTART_ALT, "Reset all settings to defaults", open_reset_dialog)

    advanced_switch.on_change = toggle_advanced
    parameters_column = ft.Column(
        [
            ft.Row([page_intro("Additional parameters"), ft.Container(expand=True), operator_btn],
                   vertical_alignment=ft.CrossAxisAlignment.CENTER),
            card(ft.Column([
                list_header("Parameter", "Scope", "Description", "Value"),
                list_row("Selection method", "GP", "Method for selecting individuals for the next generation.", operator_control("selection"), name_mono=False),
                list_row("Tournament size", "GP", "Size of each tournament (tournament, double tournament and ROIDS).", s_tourn.control, name_mono=False),
                list_row("Down-sampling", "GP", "Selection on a random or informed subset of the training cases, "
                         "new in each generation (for the methods without their own down-sampling).", downsampling_dd,
                         name_mono=False),
                list_row("Down-sample rate d", "GP", "Share of training cases in each down-sample (down-sampling setting, "
                         "DS / IDS lexicase methods and ROIDS).", s_ds.control, name_mono=False),
                list_row("Outlier rate γ", "GP", "Share of training cases with the highest mean error that are "
                         "excluded before down-sampling (ROIDS only).", s_gamma.control, name_mono=False),
                list_row("Initialization method", "GP", "How the individuals of the first generation are created.", operator_control("initialization"), name_mono=False),
                list_row("Min. initial tree depth", "GP", "Minimum tree depth of the individuals in the first generation.", s_min_depth.control, name_mono=False),
                list_row("Max. initial tree depth", "GP", "Maximum tree depth of the individuals in the first generation.", s_max_depth.control, name_mono=False),
                list_row("Max. tree depth", "GP", "Maximum tree depth allowed after crossover and mutation.", s_max_tree.control, name_mono=False),
                list_row("Elitism", "GP", "Best individuals copied unchanged into the next generation.", s_elite.control,
                         name_mono=False),
                list_row("Random seed", "GP", "Same seed and settings give the same run; 0 = a new random run each time.",
                         seed_field, name_mono=False),
                list_row("Plot adjustment", "App", "Use fixed, problem-specific axis limits instead of autoscaling.", plot_adjust_switch, name_mono=False),
                list_row("Inter-generation pause", "App", "Minimum pause between two displayed generations. Can be changed during a run.", s_pause.control, name_mono=False),
            ], spacing=0), padding=0),
            ft.Row([advanced_switch, ft.Text("Show advanced settings / parameters", size=13, color=S.INK),
                    ft.Container(expand=True), settings_reset_btn],
                   spacing=10, vertical_alignment=ft.CrossAxisAlignment.CENTER,
                   key=ft.ScrollKey("advanced_switch")),
            advanced_card,
        ],
        spacing=20, scroll=ft.ScrollMode.AUTO,
    )
    parameters_view = ft.Container(
        parameters_column,
        padding=ft.Padding.only(left=32, right=32, top=24, bottom=24), expand=True,
    )

    about_view = ft.Container(
        ft.Row(
            [
                ft.Column(
                    [
                        ft.Text(f"{config.APP_NAME} {config.APP_VERSION}", size=34, weight=ft.FontWeight.W_700, color=S.INK),
                        ft.Container(height=8),
                        caps("Source"),
                        ft.TextButton("github.com/domsob/gpexplore", url="https://github.com/domsob/gpexplore",
                                      style=ft.ButtonStyle(color=S.ACCENT, padding=0,
                                                           text_style=ft.TextStyle(font_family=S.MONO, size=13))),
                        caps("Contributors"),
                        ft.Text("Dominik Sobania", size=14, color=S.INK),
                        ft.Container(height=8),
                        caps("Keyboard"),
                        mono("Space   run / pause / resume\n→       next generation\nR       reset run", size=12.5, color=S.INK),
                    ],
                    width=380, spacing=8, scroll=ft.ScrollMode.AUTO,
                ),
                ft.Column(
                    [caps("License"),
                     ft.Container(
                         ft.ListView([mono(config.license_text, size=12, color=S.INK, selectable=True)],
                                     expand=True, padding=16),
                         expand=True, alignment=ft.Alignment.TOP_LEFT, bgcolor=S.PANEL,
                         border=ft.Border.all(1, S.RULE), border_radius=ft.BorderRadius.all(4),
                     )],
                    spacing=10, expand=True, horizontal_alignment=ft.CrossAxisAlignment.STRETCH,
                ),
            ],
            spacing=48, vertical_alignment=ft.CrossAxisAlignment.STRETCH,
        ),
        padding=ft.Padding.only(left=32, right=32, top=28, bottom=24), expand=True,
    )

    # ---- experiments (repeated runs, statistics) ---------------------------

    CORES = experiments.cpu_cores()
    config.exp_workers = max(1, min(config.exp_workers or CORES - 1, CORES))
    # configs: the compared settings; problems: problem keys, worked through one after the other;
    # results: problem key -> per configuration a list of run results (None while open);
    # view: the problem whose results are shown
    exp = {"configs": [], "problems": [], "results": {}, "analyses": {}, "view": None, "pinned": False,
           "pool": None, "running": False, "stop": False, "started": 0.0, "done": 0, "total": 0}
    exp_sizes = {"exp_final": plots.DEFAULT_SIZE, "exp_conv": plots.DEFAULT_SIZE}
    exp_view = {"mode": "main"}   # right plot: median curves of the compared value, or a dynamics view
    exp_timer = {"handle": None}

    # settings shown in the configuration list (only those that differ between configurations)
    DESCRIBED = [("selection_method", "selection", None), ("population_size", "population", None),
                 ("generations", "generations", None), ("crossover_probability", "crossover", None),
                 ("mutation_probability", "mutation", None), ("tourn_size", "tournament", None),
                 ("parsimony_size", "parsimony size", None), ("double_tournament_order", "order", None),
                 ("lexicase_epsilon", "fixed ε", None),
                 ("downsampling", "down-sampling", None), ("ds_rate", "down-sample rate", None), ("initialization", "init", None),
                 ("max_tree_depth", "max depth", None), ("crossover_type", "crossover type", None),
                 ("inner_node_probability", "inner nodes", None),
                 ("mutation_type", "mutation type", None), ("subtree_min_depth", "subtree min. depth", None),
                 ("subtree_max_depth", "subtree max. depth", None), ("subtree_method", "subtree method", None),
                 ("ephemeral_mode", "ephemeral", None), ("bloat_control", "bloat", None),
                 ("max_tree_size", "max size", None), ("elitism", "elitism", None),
                 ("fitness_function", "fitness", None), ("numba_acceleration", "numba", None),
                 ("deep_acceleration", "deep trees", None), ("parameter_control", "control", None),
                 ("training_y", "training cases", len),
                 ("validation_y", "validation cases", len)]
    BLOCK_FLAGS = [k for k in config.RUN_SETTING_KEYS if k.endswith("_func") or k.endswith("_erc")
                   or k.startswith("ps_") or k.startswith("logic_")]

    def describe(settings, keys):
        parts = []
        for key, name, conv in DESCRIBED:
            if key in keys:
                value = conv(settings[key]) if conv else settings.get(key)
                parts.append(f"{name} {value}")
        if "blocks" in keys:
            parts.append("blocks " + ", ".join(k.split("_")[1] if "_" in k else k
                                               for k in BLOCK_FLAGS if settings.get(k)))
        return "  ·  ".join(parts)

    def differing_keys():
        settings = [c["settings"] for c in exp["configs"]]
        if len(settings) < 2:
            return {"selection_method", "population_size", "generations", "crossover_probability",
                    "mutation_probability"}
        keys = set()
        for key, _, conv in DESCRIBED:
            values = [repr(conv(s[key]) if conv else s.get(key)) for s in settings]
            if len(set(values)) > 1:
                keys.add(key)
        if len({tuple(bool(s.get(k)) for k in BLOCK_FLAGS) for s in settings}) > 1:
            keys.add("blocks")
        return keys or {"selection_method"}

    exp_rows = ft.Column(spacing=0)
    cores_text = mono(f"{CORES} CPU cores", size=13)
    exp_status = mono("", size=12.5, max_lines=1)
    exp_progress = ft.ProgressBar(value=0, color=S.ACCENT, bgcolor=S.RULE, bar_height=4, border_radius=2)

    def exp_labels():
        return [c["label"].strip() or f"Configuration {i + 1}" for i, c in enumerate(exp["configs"])]

    def exp_problem(key=None):
        return problems.REGISTRY.get(key or exp["view"] or "")

    def exp_kind(key=None):
        problem = exp_problem(key)
        return problem.kind if problem is not None else "sr"

    def problem_title(key):
        """Short unique name: Koza-1 etc. for regression, the problem itself otherwise
        (the name of program synthesis / logic problems is their source, e.g. PSB1)."""
        problem = problems.REGISTRY.get(key)
        return problems.title(problem) if problem is not None else key

    def exp_groups(key=None):
        res = exp["results"].get(key or exp["view"])
        if res:
            return [[r for r in runs if r is not None] for runs in res]
        return [[] for _ in exp["configs"]]

    def exp_generations():
        return max([c["settings"]["generations"] for c in exp["configs"]] or [config.generations])

    def clear_results():
        exp["results"] = {}
        exp["analyses"] = {}
        exp["total"] = exp["done"] = 0

    def file_stem(text):
        return "".join(ch if ch.isalnum() or ch in "-_" else "-" for ch in text).strip("-")[:40] or "problem"

    def rebuild_exp_rows():
        keys = differing_keys()
        rows = []
        for i, c in enumerate(exp["configs"]):
            color = S.CATEGORICAL[i % len(S.CATEGORICAL)]

            def renamed(e, c=c):
                c["label"] = e.control.value
                refresh_stats()
                exp_stats.update()
                exp_overview.update()

            async def rename_done(e):
                await refresh_experiment()

            def removed(e, c=c):
                exp["configs"].remove(c)
                clear_results()
                sync_exp_controls()
                sync_exp_status()
                page.run_task(refresh_experiment)

            field_ = ft.TextField(value=c["label"], width=220, dense=True,
                                  border_color=S.RULE_SOFT if exp["running"] else S.RULE,
                                  focused_border_color=S.INK, border_radius=4, on_change=renamed, on_blur=rename_done,
                                  on_submit=rename_done,
                                  text_style=ft.TextStyle(size=13, color=S.FAINT if exp["running"] else S.INK),
                                  disabled=exp["running"])
            rows.append(ft.Container(
                ft.Row([ft.Container(width=10, height=10, bgcolor=color, border_radius=2), field_,
                        mono(describe(c["settings"], keys), size=12.5, expand=True, max_lines=2,
                             overflow=ft.TextOverflow.ELLIPSIS),
                        ft.IconButton(ft.Icons.DELETE_OUTLINE, icon_size=18, tooltip="Remove configuration",
                                      on_click=removed, disabled=exp["running"],
                                      style=ft.ButtonStyle(icon_color={ft.ControlState.DEFAULT: S.MUTED,
                                                                       ft.ControlState.DISABLED: S.RULE}))],
                       spacing=14, vertical_alignment=ft.CrossAxisAlignment.CENTER),
                padding=ft.Padding.symmetric(horizontal=20, vertical=6),
                border=ft.Border(bottom=ft.BorderSide(1, S.RULE_SOFT))))
        if not rows:
            rows.append(ft.Container(ft.Text("Add the current settings as the first configuration, change "
                                             "something (e.g. the selection method) and add them again.",
                                             size=13, color=S.INK),
                                     padding=ft.Padding.symmetric(horizontal=20, vertical=14)))
        exp_rows.controls = rows
        exp_add_btn.disabled = exp["running"] or len(exp["configs"]) >= len(S.CATEGORICAL)
        exp_script_btn.disabled = not exp["configs"]
        rebuild_problem_chips()

    # ---- problems of the experiment (batch) ----

    exp_problem_row = ft.Row(spacing=8, wrap=True, run_spacing=6, expand=True)

    def problem_progress(key):
        res = exp["results"].get(key)
        if not res or not exp["total"]:
            return ""
        done = sum(r is not None for runs in res for r in runs)
        return f"{done}/{sum(len(runs) for runs in res)}"

    def rebuild_problem_chips():
        chips = []
        for key in exp["problems"]:
            selected = key == exp["view"]
            ink = S.PANEL if selected else S.INK
            greyed = ft.Colors.with_opacity(0.3, S.PANEL) if selected else S.RULE   # disabled ×, visibly off

            async def show(e, key=key):
                exp["view"] = key
                exp["pinned"] = exp["running"]   # stays on this problem while the batch goes on
                rebuild_problem_chips()
                await refresh_experiment()

            async def remove(e, key=key):
                exp["problems"].remove(key)
                exp["results"].pop(key, None)
                exp["analyses"].pop(key, None)
                if exp["view"] == key:
                    exp["view"] = exp["problems"][0] if exp["problems"] else None
                sync_exp_controls()
                await refresh_experiment()

            progress = problem_progress(key)
            chips.append(hand(ft.Container(
                ft.Row([mono(problem_title(key), size=12.5, color=ink, max_lines=1),
                        *([mono(progress, size=11, color=ink)] if progress else []),
                        ft.IconButton(ft.Icons.CLOSE, icon_size=14, width=26, height=26, padding=0,
                                      tooltip="Remove problem", on_click=remove,
                                      disabled=exp["running"] or len(exp["problems"]) < 2,
                                      style=ft.ButtonStyle(icon_color={ft.ControlState.DEFAULT: ink,
                                                                       ft.ControlState.DISABLED: greyed}))],
                       spacing=6, tight=True, vertical_alignment=ft.CrossAxisAlignment.CENTER),
                bgcolor=S.INK if selected else S.PANEL, border=ft.Border.all(1, S.INK if selected else S.RULE),
                border_radius=4, padding=ft.Padding.only(left=10, right=2, top=1, bottom=1), on_click=show,
                tooltip="Show the results of this problem")))
        choose_problems_btn.disabled = exp["running"]
        exp_problem_row.controls = chips + [choose_problems_btn]

    def open_problem_dialog(e):
        """All problems with a checkbox; the checked ones are worked through one after the other."""
        boxes, rows = [], []
        for kind, group in problems.ordered().items():
            if not group:
                continue
            rows.append(ft.Container(caps(problems.KIND_TITLES[kind]), padding=ft.Padding.only(top=12, bottom=4)))
            for p in group:
                box = ft.Checkbox(value=p.expression in exp["problems"], data=p.expression, active_color=S.INK)
                boxes.append(box)
                right = p.name if p.name and p.name != p.expression else ""
                rows.append(ft.Row([box, mono(p.expression, size=12.5, expand=True, max_lines=1,
                                              overflow=ft.TextOverflow.ELLIPSIS),
                                    ft.Text(right, size=12.5, color=S.INK)], spacing=8,
                                   vertical_alignment=ft.CrossAxisAlignment.CENTER))

        async def apply(_):
            picked = [b.data for b in boxes if b.value]
            if not picked:
                toast("Choose at least one problem.", seconds=4)
                return
            # problems already in the list keep their place, new ones are appended
            chosen = [k for k in exp["problems"] if k in picked] + [k for k in picked if k not in exp["problems"]]
            for key in set(exp["problems"]) - set(chosen):
                exp["results"].pop(key, None)
                exp["analyses"].pop(key, None)
            exp["problems"] = chosen
            if exp["view"] not in chosen:
                exp["view"] = chosen[0]
            exp["total"] = 0
            page.pop_dialog()
            sync_exp_controls()
            sync_exp_status()
            await refresh_experiment()

        win_h = page.height or page.window.height or 900
        page.show_dialog(ft.AlertDialog(
            modal=True, bgcolor=S.PANEL, shape=ft.RoundedRectangleBorder(radius=6),
            title=ft.Text("Problems of the experiment", size=18, weight=ft.FontWeight.W_600, color=S.INK),
            content=ft.Container(ft.Column(rows, spacing=0, scroll=ft.ScrollMode.AUTO), width=560,
                                 height=min(560, win_h - 230)),
            actions=[ft.TextButton("Cancel", on_click=lambda _: page.pop_dialog(), style=ft.ButtonStyle(color=S.INK)),
                     ft.FilledButton("Apply", on_click=apply,
                                     style=ft.ButtonStyle(bgcolor=S.ACCENT, color="#FFFFFF",
                                                          shape=ft.RoundedRectangleBorder(radius=4)))],
            actions_alignment=ft.MainAxisAlignment.END))

    choose_problems_btn = link_button(ft.Icons.ADD, "Choose problems", open_problem_dialog)

    async def add_configuration(e):
        settings = config.run_settings()
        problem = settings["problem"]
        if not exp["problems"]:
            exp["problems"].append(problem.expression)   # by default the selected problem
            exp["view"] = problem.expression
        label = dict(gp.SELECTION_METHODS).get(settings["selection_method"], settings["selection_method"])
        label = label.split(" (")[0]
        method = settings["selection_method"]
        if (settings.get("downsampling", "None") != "None" and method not in gp.DOWNSAMPLED_METHODS
                and method != "Random"):
            label += " + " + ("DS" if settings["downsampling"] == "Random" else "IDS")
        taken = {c["label"] for c in exp["configs"]}
        if label in taken:
            k = 2
            while f"{label} {k}" in taken:
                k += 1
            label = f"{label} {k}"
        exp["configs"].append({"label": label, "settings": experiments.job_settings(settings),
                               "problem": problem.expression,
                               "sizes": (config.training_set_size, config.validation_set_size,
                                         config.noise_intensity, config.outlier_share)})
        clear_results()
        sync_exp_controls()
        sync_exp_status()
        await refresh_experiment()

    exp_add_btn = link_button(ft.Icons.ADD, "Add current settings", add_configuration)

    async def reset_experiment(e):
        """Empty the Experiments tab (configurations, problems, results); undo is offered."""
        if exp["running"]:
            return
        saved = {key: exp.get(key) for key in ("configs", "problems", "results", "analyses", "view", "total", "done",
                                               "elapsed")}
        saved_view = exp_view["mode"]
        exp.update(configs=[], problems=[], results={}, analyses={}, view=None, total=0, done=0, pinned=False)
        exp_view["mode"] = "main"
        sync_exp_controls()
        sync_exp_status()
        await refresh_experiment()

        async def undo():
            if exp["running"]:
                return
            exp.update(saved)
            exp_view["mode"] = saved_view
            sync_exp_controls()
            sync_exp_status()
            await refresh_experiment()

        toast("Experiment reset", action="Undo", on_action=undo, seconds=6)

    exp_reset_btn = link_button(ft.Icons.RESTART_ALT, "Reset experiment", reset_experiment)

    async def export_script(e):
        """The experiment as a Python script using gpexplore's API (to run it without the app)."""
        if not exp["configs"]:
            return
        keys = [k for k in exp["problems"] if k in problems.REGISTRY] or [exp["configs"][0]["problem"]]
        stem = "gpexplore_experiment_" + (file_stem(problem_title(keys[0])) if len(keys) == 1 else "batch")
        try:
            text = await asyncio.to_thread(experiments.python_script, list(exp["configs"]), keys,
                                           int(s_runs.value), int(config.exp_seed), stem)
        except Exception as exc:
            toast(f"Could not write the script: {exc}", seconds=6)
            return
        await save_bytes(text.encode("utf-8"), "Export experiment as Python script", stem + ".py", "py")

    exp_script_btn = link_button(ft.Icons.CODE, "Python script", export_script)
    exp_script_btn.disabled = True

    def problem_data(problem, sizes):
        """Data of a problem for the batch, reproducible from the first seed."""
        train, val, noise, outliers = sizes
        seed = config.exp_seed
        if problem.kind != "sr":
            try:
                tr, te = problem.generate(train, val, seed=seed)
            except TypeError:
                tr, te = problem.generate(train, val)
            return [i for i, _ in tr], [o for _, o in tr], [i for i, _ in te], [o for _, o in te]
        state = np.random.get_state()
        np.random.seed(seed)
        try:
            return problems.make_data(problem, train, val, noise, outliers)
        finally:
            np.random.set_state(state)

    def settings_for(c, key, cache):
        """Settings of configuration c on problem `key`. On the problem it was added on, the
        configuration keeps its building blocks; on other problems its GP parameters stay, while
        the building blocks are set as selecting the problem in the Run tab would set them.
        The data always come from the first seed and the data sizes (not from the Run tab, whose
        data are drawn anew e.g. when the training size changes): configurations with the same
        data sizes run on the same data, and the batch can be repeated."""
        settings = dict(c["settings"])
        problem = problems.get(key)
        if key != c["problem"]:
            origin = problems.REGISTRY.get(c["problem"])
            settings["target_function"] = key
            if problem.kind == "ps" or (origin is not None and origin.kind == "ps"):
                for flag in synthesis.GROUP_FLAGS:
                    settings[flag] = problem.kind == "ps"
            if problem.kind == "logic":
                for flag in synthesis.GATE_FLAGS:
                    settings[flag] = flag in problem.gates
        train, val, noise, outliers = c["sizes"]
        if problem.kind == "ps" or (problem.kind == "sr" and getattr(problem, "dataset", False)):
            train, val = problem.train_size, problem.test_size
        elif problem.kind == "logic":
            train = problem.train_size
        if problem.kind != "sr" or getattr(problem, "dataset", False):
            noise = outliers = 0.0
        sizes = (int(train), int(val), float(noise), float(outliers))
        if (key, sizes) not in cache:
            cache[(key, sizes)] = problem_data(problem, sizes)
        settings["training_X"], settings["training_y"], settings["validation_X"], settings["validation_y"] = \
            cache[(key, sizes)]
        return settings

    def changed_exp_seed(e):
        try:
            config.exp_seed = max(1, int(exp_seed_field.value))
            exp_seed_field.error = None
        except ValueError:
            exp_seed_field.error = "whole number ≥ 1"
        exp_seed_field.update()

    s_runs = ParamSlider("Runs per configuration", "exp_runs", 2, 100, lambda v: str(int(v)), divisions=98)
    s_workers = ParamSlider("Parallel runs", "exp_workers", 1, max(CORES, 2), lambda v: str(int(v)),
                            divisions=max(CORES - 1, 1))
    exp_seed_field = ft.TextField(value=str(config.exp_seed), width=120, dense=True, border_color=S.RULE,
                                  focused_border_color=S.INK, border_radius=4, on_change=changed_exp_seed,
                                  label="First seed", text_style=ft.TextStyle(font_family=S.MONO, size=13, color=S.INK),
                                  keyboard_type=ft.KeyboardType.NUMBER)

    exp_run_icon = ft.Icon(ft.Icons.PLAY_ARROW_ROUNDED, color="#FFFFFF", size=22)
    exp_run_label = ft.Text("Run experiment", size=14, color="#FFFFFF", weight=ft.FontWeight.W_600)

    def sync_exp_controls():
        running = exp["running"]
        exp_run_icon.icon = ft.Icons.STOP_ROUNDED if running else ft.Icons.PLAY_ARROW_ROUNDED
        exp_run_label.value = "Stop" if running else "Run experiment"
        exp_run_btn.bgcolor = S.INK if running else (S.ACCENT if exp["configs"] and exp["problems"] else S.FAINT)
        exp_run_hand.mouse_cursor = (ft.MouseCursor.CLICK if running or (exp["configs"] and exp["problems"])
                                     else ft.MouseCursor.BASIC)
        s_runs.disabled = s_workers.disabled = running
        exp_seed_field.disabled = running
        exp_reset_btn.disabled = running or not (exp["configs"] or exp["problems"] or exp["results"])
        exp_seed_field.text_style = ft.TextStyle(font_family=S.MONO, size=13, color=S.FAINT if running else S.INK)
        exp_seed_field.border_color = S.RULE_SOFT if running else S.RULE
        rebuild_exp_rows()

    def fmt_time(seconds):
        seconds = int(round(seconds))
        return f"{seconds // 60}:{seconds % 60:02d}"

    def sync_exp_status():
        if exp["total"]:
            elapsed = time.time() - exp["started"] if exp["running"] else exp.get("elapsed", 0.0)
            text = f"{exp['done']} / {exp['total']} runs  ·  {fmt_time(elapsed)}"
            if exp["running"] and exp["done"]:
                left = elapsed / exp["done"] * (exp["total"] - exp["done"])
                text += f"  ·  ~{fmt_time(left)} left"
            elif not exp["running"] and exp["done"] < exp["total"]:
                text += "  ·  stopped"
            exp_status.value = text
            exp_progress.value = exp["done"] / exp["total"]
        else:
            exp_status.value = ""
            exp_progress.value = 0

    def stop_pool():
        pool = exp["pool"]
        exp["pool"] = None
        if pool is None:
            return
        for process in list(getattr(pool, "_processes", {}).values() or []):
            try:
                process.terminate()
            except Exception:
                pass
        pool.shutdown(wait=False, cancel_futures=True)

    def follow_batch():
        """While a batch runs, show the first problem that is not finished yet (unless the user picked one)."""
        if exp["pinned"]:
            return
        for key in exp["problems"]:
            res = exp["results"].get(key)
            if res and any(r is None for runs in res for r in runs):
                exp["view"] = key
                return

    async def toggle_experiment(e):
        if exp["running"]:
            exp["stop"] = True
            stop_pool()
            return
        keys = [k for k in exp["problems"] if k in problems.REGISTRY]
        if not exp["configs"] or not keys:
            toast("Add at least one configuration first.", seconds=4)
            return
        if exp_seed_field.error:
            return
        runs, workers = s_runs.value, s_workers.value
        configs = list(exp["configs"])
        cache = {}
        try:
            jobs = []
            for key in keys:   # problems one after the other; run r of every configuration uses the same seed
                per_config = [await asyncio.to_thread(settings_for, c, key, cache) for c in configs]
                for r in range(runs):
                    for ci, settings in enumerate(per_config):
                        jobs.append((key, ci, r, (experiments.job_settings(settings), key, config.exp_seed + r)))
        except Exception as exc:
            toast(f"Could not prepare the problems: {exc}", seconds=6)
            return
        exp["results"] = {k: [[None] * runs for _ in configs] for k in keys}
        exp["analyses"] = {}
        exp.update(running=True, stop=False, started=time.time(), done=0, total=len(jobs), pinned=False)
        follow_batch()
        sync_exp_controls()
        sync_exp_status()
        page.update()
        await refresh_experiment()
        pool = ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn"))
        exp["pool"] = pool
        tasks = []

        async def job(future, key, ci, r):
            return key, ci, r, await asyncio.wrap_future(future)
        try:
            for key, ci, r, payload in jobs:
                tasks.append(asyncio.ensure_future(job(pool.submit(experiments.run_one, payload), key, ci, r)))
            last_draw = 0.0
            failure = None
            for next_done in asyncio.as_completed(tasks):
                try:
                    key, ci, r, result = await next_done
                except (Exception, asyncio.CancelledError) as exc:   # Stop, or a failed run
                    if not exp["stop"] and failure is None:
                        failure = exc
                    continue
                exp["results"][key][ci][r] = result
                exp["done"] += 1
                sync_exp_status()
                exp_status.update()
                exp_progress.update()
                if time.time() - last_draw > 1.0:
                    last_draw = time.time()
                    follow_batch()
                    try:
                        rebuild_problem_chips()
                        await refresh_experiment()
                    except Exception:        # a failed redraw must not end the experiment
                        pass
            if failure is not None:
                toast(f"A run failed: {failure}", seconds=6)
        finally:
            exp["elapsed"] = time.time() - exp["started"]
            exp["running"] = False
            stop_pool()
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            sync_exp_controls()
            sync_exp_status()
            page.update()
            await refresh_experiment()

    exp_run_btn = ft.Container(
        content=ft.Row([exp_run_icon, exp_run_label], spacing=6, alignment=ft.MainAxisAlignment.CENTER),
        bgcolor=S.FAINT, border_radius=4, height=44, width=200, on_click=toggle_experiment, ink=True,
    )
    exp_run_hand = hand(exp_run_btn)

    # results: plots and statistics (of the problem selected above), overview over all problems

    exp_final_img = plot_image(plots.render_experiment_final([], [], "sr"))
    exp_conv_img = plot_image(plots.render_experiment_convergence([], [], "sr", config.generations))
    exp_final_title = caps("Final values")
    exp_conv_title = caps("Median over the runs (IQR band)")
    exp_stats_title = caps("Statistics")
    exp_stats = ft.Column(spacing=0)
    exp_overview_rows = ft.Column(spacing=0)
    exp_render_lock = asyncio.Lock()

    def render_experiment_plot(which, size=None):
        groups, labels, kind = exp_groups(), exp_labels(), exp_kind()
        size = size or exp_sizes[which]
        if which == "exp_final":
            return plots.render_experiment_final(groups, labels, kind, size)
        if exp_view["mode"] != "main":
            return plots.render_experiment_dynamics(exp_view["mode"], groups, labels, kind, exp_generations(), size)
        return plots.render_experiment_convergence(groups, labels, kind, exp_generations(), size)

    async def pick_exp_view(e):
        exp_view["mode"] = e.control.data
        await refresh_experiment()

    exp_conv_menu = ft.PopupMenuButton(
        content=ft.Row([exp_conv_title, ft.Icon(ft.Icons.EXPAND_MORE, size=16, color=S.MUTED)], spacing=2, tight=True),
        items=[ft.PopupMenuItem(content=ft.Text(label, size=13, color=S.INK), data=key, on_click=pick_exp_view)
               for key, label in [("main", "Compared value over generations")] + list(plots.DYNAMICS_VIEWS.items())],
        tooltip="Choose what this plot shows", menu_position=ft.PopupMenuPosition.UNDER, bgcolor=S.PANEL)

    def exp_resized(which):
        def handler(e: ft.LayoutSizeChangeEvent):
            size = (int(e.width), int(e.height))
            if size[0] <= 0 or size[1] <= 0 or size == exp_sizes[which]:
                return
            exp_sizes[which] = size
            if exp_timer["handle"] is not None:
                exp_timer["handle"].cancel()
            loop = asyncio.get_running_loop()
            exp_timer["handle"] = loop.call_later(0.15, lambda: loop.create_task(refresh_experiment()))
        return handler

    def stat_cell(text, width=None, bold=False, color=S.INK, font=True, lines=None):
        t = (mono(text, size=12.5, color=color, weight=ft.FontWeight.W_600 if bold else None, max_lines=lines) if font
             else ft.Text(text, size=13, color=color, weight=ft.FontWeight.W_500, max_lines=lines))
        return ft.Container(t, width=width, expand=width is None)

    def stat_row(cells, header=False):
        return ft.Container(ft.Row(cells, spacing=16, vertical_alignment=ft.CrossAxisAlignment.CENTER),
                            padding=ft.Padding.symmetric(horizontal=20, vertical=7 if not header else 9),
                            border=ft.Border(bottom=ft.BorderSide(1, S.RULE if header else S.RULE_SOFT)))

    def p_text(p):
        if not math.isfinite(p):
            return "–"
        return "< 0.001" if p < 0.001 else f"{p:.3f}"

    def value_fmt(kind):
        if kind == "sr":
            return lambda v: "inf" if v == math.inf else "–" if not math.isfinite(v) else plots._short_number(v)
        return lambda v: "–" if not math.isfinite(v) else f"{v:.1f} %"

    def analysis_of(key):
        """Analysis of one problem, or None while a configuration has fewer than two finished runs."""
        groups = exp_groups(key)
        if not groups or not all(len(g) >= 2 for g in groups):
            exp["analyses"].pop(key, None)
            return None
        a = experiments.analyse(groups, exp_labels(), exp_kind(key))
        exp["analyses"][key] = a
        return a

    def refresh_stats():
        """Summary per configuration and pairwise tests (Mann-Whitney U, Vargha-Delaney A12; for program
        synthesis / logic also Fisher's exact test on success; Holm over all p values of the table)."""
        kind = exp_kind()
        for key in exp["problems"]:
            if key != exp["view"]:
                analysis_of(key)
        a = analysis_of(exp["view"]) if exp["view"] else None
        refresh_overview()
        controls = []
        if a is None:
            exp_stats.controls = [ft.Container(ft.Text("Results appear here once every configuration has at "
                                                       "least two finished runs.", size=13, color=S.INK),
                                               padding=ft.Padding.symmetric(horizontal=20, vertical=14))]
            return
        fmt = value_fmt(kind)
        success = kind != "sr"
        w = [None, 60, 110, 170, 170] + ([200] if success else []) + [90]
        head = ["Configuration", "Runs", "Median", "IQR", "Mean ± SD"] + (["Success (95 % CI)"] if success else []) \
            + ["Time / run"]
        controls.append(stat_row([ft.Container(caps(h, size=9.5), width=x, expand=x is None)
                                  for h, x in zip(head, w)], header=True))
        best = (max if a.higher_is_better else min)(
            (row["median"] for row in a.summary if math.isfinite(row["median"])), default=None)
        for i, row in enumerate(a.summary):
            color = S.CATEGORICAL[i % len(S.CATEGORICAL)]
            name = ft.Row([ft.Container(width=10, height=10, bgcolor=color, border_radius=2),
                           ft.Text(row["label"], size=13, color=S.INK, weight=ft.FontWeight.W_500)], spacing=10)
            cells = [ft.Container(name, expand=True), stat_cell(str(row["n"]), 60),
                     stat_cell(fmt(row["median"]), 110, bold=row["median"] == best),
                     stat_cell(f"{fmt(row['q1'])} – {fmt(row['q3'])}", 170),
                     stat_cell(f"{fmt(row['mean'])} ± {fmt(row['sd'])}", 170)]
            if success:
                lo, hi = row["success_ci"]
                cells.append(stat_cell(f"{row['successes']}/{row['n']}  ({100 * lo:.0f}–{100 * hi:.0f} %)", 200))
            cells.append(stat_cell(f"{row['seconds']:.1f} s", 90))
            controls.append(stat_row(cells))

        if a.pairs:
            controls.append(ft.Container(height=14))
            if a.omnibus is not None:
                controls.append(ft.Container(
                    mono(f"Kruskal-Wallis test over all configurations:  H = {a.omnibus['H']:.2f},  "
                         f"p {'' if a.omnibus['p'] < 0.001 else '= '}{p_text(a.omnibus['p'])}", size=12.5),
                    padding=ft.Padding.only(left=20, right=20, bottom=10)))
            w2 = [None, 170, 170] + ([170] if success else []) + [260]
            head2 = ["Comparison", "p  (Mann-Whitney U, Holm)", "Â12  (effect size)"] + \
                (["p  (Fisher exact, Holm)"] if success else []) + ["Result  (α = 0.05)"]
            controls.append(stat_row([ft.Container(caps(h, size=9.5), width=x, expand=x is None)
                                      for h, x in zip(head2, w2)], header=True))
            for pair in a.pairs:
                cells = [stat_cell(f"{pair['a']}  vs.  {pair['b']}", font=False),
                         stat_cell(p_text(pair["p_holm"]), 170, bold=pair["p_holm"] < 0.05),
                         stat_cell(f"{pair['a12']:.2f}  {experiments.a12_magnitude(pair['a12'])}", 170)]
                if success:
                    cells.append(stat_cell(p_text(pair["fisher_p_holm"]), 170, bold=pair["fisher_p_holm"] < 0.05))
                cells.append(stat_cell(pair["verdict"], 260, color=S.ACCENT if pair["significant"] else S.INK,
                                       font=False))
                controls.append(stat_row(cells))
        exp_stats.controls = controls

    def overview_cell(row, kind):
        fmt = value_fmt(kind)
        text = fmt(row["median"])
        if kind != "sr":
            text += f"  ·  {row['successes']}/{row['n']}"
        return text

    def overview_verdict(a):
        wins = [p["verdict"].replace(" better", f" better than {p['b'] if p['winner'] == p['a'] else p['a']}")
                if p["winner"] else f"{p['a']} vs. {p['b']}: {p['verdict']}"
                for p in a.pairs if p["significant"]]
        return "; ".join(wins) if wins else "no significant difference"

    def refresh_overview():
        """One row per problem: median (and successes) per configuration, significant differences."""
        show = len(exp["problems"]) > 1 and any(exp["analyses"].get(k) for k in exp["problems"])
        exp_overview.visible = show
        if not show:
            return
        labels = exp_labels()
        w = [None] + [150] * len(labels) + [300]
        head = [ft.Container(caps("Problem", size=9.5), expand=True)]
        for i, label in enumerate(labels):
            head.append(ft.Container(ft.Row([ft.Container(width=8, height=8, bgcolor=S.CATEGORICAL[i % len(S.CATEGORICAL)],
                                                          border_radius=2), caps(label, size=9.5)], spacing=6),
                                     width=150))
        head.append(ft.Container(caps("Result  (α = 0.05)", size=9.5), width=300))
        rows = [stat_row(head, header=True)]
        for key in exp["problems"]:
            a = exp["analyses"].get(key)
            kind = exp_kind(key)
            cells = [stat_cell(problem_title(key), lines=1)]
            if a is None:
                cells += [stat_cell("–", 150) for _ in labels] + [stat_cell("", 300)]
            else:
                best = (max if a.higher_is_better else min)(
                    (row["median"] for row in a.summary if math.isfinite(row["median"])), default=None)
                cells += [stat_cell(overview_cell(row, kind), 150, bold=row["median"] == best) for row in a.summary]
                verdict = overview_verdict(a)
                cells.append(stat_cell(verdict, 300, font=False, lines=3,
                                       color=S.ACCENT if any(p["significant"] for p in a.pairs) else S.INK))
            rows.append(stat_row(cells))
        exp_overview_rows.controls = rows

    async def refresh_experiment():
        kind = exp_kind()
        metric, _ = experiments.primary_metric(kind)
        exp_final_title.value = (("Final " + metric) if kind == "sr" else
                                 f"Success rate (95 % CI) and {metric}")
        exp_conv_title.value = (f"Median {metric} over the runs (IQR band)" if exp_view["mode"] == "main" else
                                f"{plots.EXPERIMENT_DYNAMICS_TITLES[exp_view['mode']]} · median over the runs"
                                if exp_view["mode"] != "blocks" else plots.EXPERIMENT_DYNAMICS_TITLES["blocks"])
        title = problem_title(exp["view"]) if exp["view"] else ""
        exp_stats_title.value = f"Statistics  ·  {title}" if len(exp["problems"]) > 1 else "Statistics"
        refresh_stats()
        sync_result_header()
        exp_export_row.visible = any(exp["analyses"].values())
        page.update()
        async with exp_render_lock:
            final, conv = await asyncio.gather(asyncio.to_thread(render_experiment_plot, "exp_final"),
                                               asyncio.to_thread(render_experiment_plot, "exp_conv"))
        exp_final_img.src, exp_conv_img.src = final, conv
        page.update()

    def finished_sets():
        """(problem title, kind, groups) of every problem with finished runs."""
        return [(problem_title(k), exp_kind(k), exp_groups(k)) for k in exp["problems"]
                if any(exp_groups(k))]

    def export_stem():
        keys = [k for k in exp["problems"] if any(exp_groups(k))]
        return file_stem(problem_title(keys[0])) if len(keys) == 1 else "batch"

    async def run_export(what, make, title, file_name, ext):
        """Prepare an export off the UI thread; meanwhile a spinner says what is being prepared and
        the export buttons are disabled. Then the save dialog opens."""
        if exp_busy.visible or not finished_sets():
            return
        exp_busy_text.value = f"Preparing {what} …"
        exp_busy.visible = True
        for button in exp_export_buttons:
            button.disabled = True
        page.update()
        try:
            data = await asyncio.to_thread(make)
        except Exception as exc:
            toast(f"Could not create the {what}: {exc}", seconds=6)
            return
        finally:
            exp_busy.visible = False
            for button in exp_export_buttons:
                button.disabled = False
            page.update()
        await save_bytes(data, title, file_name, ext)

    async def export_experiment(variant):
        sets, labels = finished_sets(), exp_labels()
        make = (lambda: experiments.to_csv_raw(sets, labels)) if variant == "raw" else \
            (lambda: experiments.to_csv_aggregated(sets, labels))
        await run_export(f"{variant} CSV", make, f"Save {variant} results as CSV",
                         f"gpexplore_experiment_{export_stem()}_{variant}.csv", "csv")

    async def export_aggregated(e):
        await export_experiment("aggregated")

    async def export_raw(e):
        await export_experiment("raw")

    def report_overview():
        """Text of the first PDF page: setup, configurations and results of all problems."""
        labels = exp_labels()
        runs = max((len(runs) for res in exp["results"].values() for runs in res), default=0)
        keys = differing_keys()
        lines = [("h1", "GPExplore experiment"),
                 ("text", f"{time.strftime('%Y-%m-%d %H:%M')}  ·  {runs} runs per configuration  ·  seeds "
                          f"{config.exp_seed}–{config.exp_seed + max(runs, 1) - 1}  ·  "
                          f"{len(exp['problems'])} problem{'s' if len(exp['problems']) != 1 else ''}", S.MUTED),
                 ("h2", "Configurations")]
        for i, c in enumerate(exp["configs"]):
            lines.append(("swatch", f"{labels[i]:<22} {describe(c['settings'], keys)}",
                          S.CATEGORICAL[i % len(S.CATEGORICAL)]))
        lines.append(("h2", "Results  (pairwise: Mann-Whitney U and, for program synthesis / logic, Fisher's "
                            "exact test; Holm-corrected; α = 0.05)"))
        for key in exp["problems"]:
            a = exp["analyses"].get(key)
            kind = exp_kind(key)
            metric, higher = experiments.primary_metric(kind)
            lines.append(("text", f"{problem_title(key)}   ({problems.KIND_TITLES[kind]}; {metric}, "
                                  f"{'higher' if higher else 'lower'} is better)"))
            if a is None:
                lines.append(("mono", "    not enough finished runs"))
                continue
            fmt = value_fmt(kind)
            for row in a.summary:
                iqr = f"{fmt(row['q1'])} – {fmt(row['q3'])}"
                text = f"    {row['label']:<22} median {fmt(row['median']):<10} IQR {iqr:<22}"
                if kind != "sr":
                    text += f"   success {row['successes']}/{row['n']}"
                lines.append(("mono", text))
            for p in a.pairs:
                text = f"    {p['a']} vs. {p['b']}: p = {p_text(p['p_holm'])}, Â12 = {p['a12']:.2f}"
                if kind != "sr":
                    text += f", Fisher p = {p_text(p['fisher_p_holm'])}"
                lines.append(("mono", text + f"  →  {p['verdict']}", S.ACCENT if p["significant"] else S.INK))
            if a.omnibus is not None:
                lines.append(("mono", f"    Kruskal-Wallis: H = {a.omnibus['H']:.2f}, p = {p_text(a.omnibus['p'])}"))
        return lines

    def build_report():
        labels, generations = exp_labels(), exp_generations()
        sections = []
        for key in exp["problems"]:
            groups = exp_groups(key)
            if not any(groups):
                continue
            problem, kind = exp_problem(key), exp_kind(key)
            sub = [problems.KIND_TITLES[kind]]
            if problem is not None and problems.title(problem) != problem.expression:
                sub.append(problem.expression)
            sub.append(f"{max(len(g) for g in groups)} runs per configuration")
            sub.append("configurations: " + ", ".join(labels))
            sections.append((problem_title(key), "  ·  ".join(sub),
                             plots.experiment_report_plots(kind, groups, labels, generations)))
        return plots.render_experiment_report(report_overview(), sections)

    async def export_pdf(e):
        await run_export("PDF report", build_report, "Save all plots as PDF",
                         f"gpexplore_experiment_{export_stem()}.pdf", "pdf")

    async def export_zip(e):
        sets, labels, generations = finished_sets(), exp_labels(), exp_generations()
        await run_export("ZIP with all plots", lambda: plots.experiment_plots_zip(sets, labels, generations),
                         "Save all plots as ZIP (PNG and PDF)", f"gpexplore_experiment_{export_stem()}_plots.zip",
                         "zip")

    exp_busy_text = ft.Text("", size=12.5, color=S.INK)
    exp_busy = ft.Row([ft.ProgressRing(width=14, height=14, stroke_width=2, color=S.ACCENT), exp_busy_text],
                      spacing=8, visible=False, vertical_alignment=ft.CrossAxisAlignment.CENTER)
    exp_export_buttons = [link_button(ft.Icons.PICTURE_AS_PDF_OUTLINED, "Plots as PDF", export_pdf),
                          link_button(ft.Icons.FOLDER_ZIP_OUTLINED, "Plots as ZIP", export_zip),
                          link_button(ft.Icons.TABLE_CHART_OUTLINED, "Aggregated CSV", export_aggregated),
                          link_button(ft.Icons.DATASET_OUTLINED, "Raw CSV", export_raw)]
    exp_export_row = ft.Row([exp_busy, ft.Container(width=8), *exp_export_buttons], spacing=8, visible=False,
                            vertical_alignment=ft.CrossAxisAlignment.CENTER)

    def exp_plot_card(title, image, which):
        return card(ft.Column([ft.Row([title], height=22),
                               ft.Container(plot_menu(which, image), expand=True, on_size_change=exp_resized(which),
                                            size_change_interval=0)],
                              spacing=6, expand=True, horizontal_alignment=ft.CrossAxisAlignment.STRETCH),
                    padding=ft.Padding.only(left=14, right=14, top=12, bottom=10), expand=1)

    exp_overview = card(ft.Column([
        ft.Container(ft.Row([caps("Overview over all problems")], vertical_alignment=ft.CrossAxisAlignment.CENTER),
                     padding=ft.Padding.only(left=20, right=14, top=12, bottom=12),
                     border=ft.Border(bottom=ft.BorderSide(1, S.RULE))),
        exp_overview_rows], spacing=0), padding=0)
    exp_overview.visible = False

    # which problem the results below belong to
    exp_result_title = ft.Text("", size=18, weight=ft.FontWeight.W_600, color=S.INK)
    exp_result_sub = mono("", size=12.5, max_lines=1, overflow=ft.TextOverflow.ELLIPSIS)
    exp_result_count = mono("", size=12.5)

    async def step_problem(delta):
        keys = exp["problems"]
        if exp["view"] not in keys:
            return
        exp["view"] = keys[(keys.index(exp["view"]) + delta) % len(keys)]
        exp["pinned"] = exp["running"]
        rebuild_problem_chips()
        await refresh_experiment()

    async def previous_problem(e):
        await step_problem(-1)

    async def next_problem(e):
        await step_problem(1)

    exp_result_nav = ft.Row([
        ft.IconButton(ft.Icons.CHEVRON_LEFT, icon_size=20, tooltip="Previous problem", on_click=previous_problem,
                      style=ft.ButtonStyle(icon_color=S.INK)),
        exp_result_count,
        ft.IconButton(ft.Icons.CHEVRON_RIGHT, icon_size=20, tooltip="Next problem", on_click=next_problem,
                      style=ft.ButtonStyle(icon_color=S.INK))], spacing=0, visible=False)
    exp_result_header = card(ft.Row([
        ft.Container(width=4, height=44, bgcolor=S.ACCENT, border_radius=2),
        ft.Column([caps("Results for"), ft.Row([exp_result_title, exp_result_sub], spacing=14,
                                                vertical_alignment=ft.CrossAxisAlignment.END)],
                  spacing=2, expand=True),
        exp_result_nav], spacing=14, vertical_alignment=ft.CrossAxisAlignment.CENTER),
        padding=ft.Padding.only(left=14, right=10, top=10, bottom=10))
    exp_result_header.visible = False

    def sync_result_header():
        key = exp["view"]
        exp_result_header.visible = key is not None
        if key is None:
            return
        problem = exp_problem(key)
        kind = exp_kind(key)
        parts = [problems.KIND_TITLES[kind]]
        if problem is not None and problems.title(problem) != problem.expression:
            parts.append(problem.expression)
        res = exp["results"].get(key)
        if res:
            done = sum(r is not None for runs in res for r in runs)
            parts.append(f"{done} / {sum(len(runs) for runs in res)} runs finished")
        exp_result_title.value = problem_title(key)
        exp_result_sub.value = "  ·  ".join(parts)
        n = len(exp["problems"])
        exp_result_nav.visible = n > 1
        exp_result_count.value = f"{exp['problems'].index(key) + 1} / {n}" if key in exp["problems"] else ""

    exp_plots_box = side_by_side(exp_plot_card(exp_final_title, exp_final_img, "exp_final"),
                                 exp_plot_card(exp_conv_menu, exp_conv_img, "exp_conv"), height=400)

    def resize_exp_plots(dy):
        height = min(1200, max(240, exp_plots_box.height + dy))
        if height != exp_plots_box.height:
            exp_plots_box.height = height
            exp_plots_box.update()

    experiments_view = ft.Container(
        ft.Column([
            ft.Row([page_intro("Experiments"), ft.Container(expand=True), cores_text],
                   vertical_alignment=ft.CrossAxisAlignment.CENTER),
            card(ft.Column([
                ft.Container(ft.Row([caps("Configurations"), ft.Container(expand=True), exp_script_btn,
                                     ft.Container(width=8), exp_reset_btn,
                                     ft.Container(width=8), exp_add_btn],
                                    spacing=8, vertical_alignment=ft.CrossAxisAlignment.CENTER),
                             padding=ft.Padding.only(left=20, right=14, top=8, bottom=8),
                             border=ft.Border(bottom=ft.BorderSide(1, S.RULE))),
                exp_rows,
                ft.Container(ft.Row([ft.Container(caps("Problems"), width=96), exp_problem_row],
                                    spacing=8, vertical_alignment=ft.CrossAxisAlignment.CENTER),
                             padding=ft.Padding.only(left=20, right=14, top=10, bottom=10),
                             border=ft.Border(bottom=ft.BorderSide(1, S.RULE_SOFT))),
                ft.Container(ft.Row([ft.Container(s_runs.control, width=240), ft.Container(s_workers.control, width=240),
                                     exp_seed_field, ft.Container(expand=True),
                                     ft.Column([exp_status, exp_progress], spacing=6, width=320,
                                               horizontal_alignment=ft.CrossAxisAlignment.STRETCH),
                                     exp_run_hand],
                                    spacing=28, vertical_alignment=ft.CrossAxisAlignment.CENTER),
                             padding=ft.Padding.symmetric(horizontal=20, vertical=14)),
            ], spacing=0), padding=0),
            exp_overview,
            exp_result_header,
            ft.Column([exp_plots_box, splitter(resize_exp_plots)], spacing=2),
            card(ft.Column([
                ft.Container(ft.Row([exp_stats_title, ft.Container(expand=True), exp_export_row],
                                    vertical_alignment=ft.CrossAxisAlignment.CENTER),
                             padding=ft.Padding.only(left=20, right=14, top=8, bottom=8),
                             border=ft.Border(bottom=ft.BorderSide(1, S.RULE))),
                exp_stats], spacing=0), padding=0),
        ], spacing=16, scroll=ft.ScrollMode.AUTO),
        padding=ft.Padding.only(left=32, right=32, top=24, bottom=24), expand=True,
    )
    sync_exp_controls()
    refresh_stats()

    # ---- header + tabs ---------------------------------------------------

    def tab(label, icon):
        return ft.Tab(label=ft.Row([ft.Icon(icon, size=16), ft.Text(label)], spacing=7, tight=True), height=52)

    tab_bar = ft.TabBar(
        tabs=[tab("Run", ft.Icons.PLAY_ARROW_OUTLINED), tab("Function & Terminal Set", ft.Icons.FUNCTIONS),
              tab("Additional Parameters", ft.Icons.TUNE), tab("Experiments", ft.Icons.SCIENCE_OUTLINED),
              tab("Info", ft.Icons.INFO_OUTLINE)],
        scrollable=True, tab_alignment=ft.TabAlignment.START, divider_color=ft.Colors.TRANSPARENT,
        indicator_color=S.ACCENT, indicator_thickness=2, indicator_size=ft.TabBarIndicatorSize.LABEL,
        label_color=S.INK, unselected_label_color=S.MUTED,
        label_text_style=ft.TextStyle(size=13, weight=ft.FontWeight.W_600),
        unselected_label_text_style=ft.TextStyle(size=13, weight=ft.FontWeight.W_500),
        overlay_color=ft.Colors.TRANSPARENT, expand=True,
    )

    header = ft.Container(
        ft.Row(
            [
                ft.Row(
                    [ft.Text("GP", size=17, weight=ft.FontWeight.W_800, color=S.ACCENT, font_family=S.MONO),
                     ft.Text("Explore", size=17, weight=ft.FontWeight.W_600, color=S.INK)],
                    spacing=0, width=316,
                ),
                tab_bar,
            ],
            spacing=0, vertical_alignment=ft.CrossAxisAlignment.CENTER,
        ),
        padding=ft.Padding.only(left=24, right=24), height=52, bgcolor=S.PANEL,
        border=ft.Border(bottom=ft.BorderSide(1, S.RULE)),
    )

    def tab_changed(e):
        try:
            current_tab["index"] = int(e.data)
        except (TypeError, ValueError):
            current_tab["index"] = e.control.selected_index

    tabs = ft.Tabs(
        length=5, selected_index=0, animation_duration=0, expand=True, on_change=tab_changed,
        content=ft.Column(
            [header, ft.TabBarView(controls=[run_view, primitives_view, parameters_view,
                                             experiments_view, about_view], expand=True)],
            spacing=0, expand=True,
        ),
    )

    if config.expert_mode:   # as in the last session
        for slider in ParamSlider.instances:
            slider.set_expert(True)
    elif [slider for slider in ParamSlider.instances if slider.clamp()]:   # expert values of an earlier session
        regenerate_data()
    reset_readouts()
    apply_state("idle")
    sync_problem_views()
    page.add(tabs)

    async def autosave():
        """Settings are kept across sessions: written whenever they change (checked every 2 s)."""
        saved = None
        while True:
            current = config.snapshot()
            if current != saved:
                await asyncio.to_thread(config.save_settings)
                saved = current
            await asyncio.sleep(2)

    page.run_task(autosave)

    # The window starts hidden (see ft.run below) and is shown once it sits in
    # the middle of the screen. Flet's window manager picks the display under
    # the mouse cursor, so on multi-monitor setups the app opens where you are.
    if not page.web:   # in the browser there is no window to place
        def window_changed(e):
            """Remember the window size for the next start; not the size while maximized or in full
            screen, so the app always starts as a normal window."""
            w = page.window
            if (e.type == ft.WindowEventType.RESIZED and not w.maximized and not w.full_screen
                    and w.width and w.height):
                size = int(round(w.width)), int(round(w.height))
                if size != (config.window_width, config.window_height):
                    config.window_width, config.window_height = size
                    config.save_settings()   # right away: the app may be quit before the next autosave

        await page.window.center()
        page.window.visible = True
        page.update()
        await page.window.to_front()
        page.window.on_event = window_changed
        page.update()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="GPExplore: genetic programming to explore and teach.")
    parser.add_argument("--web", action="store_true", help="open in the web browser instead of the desktop window")
    parser.add_argument("--port", type=int, default=8550, help="port of the web interface (with --web)")
    args = parser.parse_args()
    if args.web:
        webserver.run(main, args.port)   # the Flet app plus a download route for exports
    else:
        ft.run(main, view=ft.AppView.FLET_APP_HIDDEN, assets_dir=webserver.ASSETS_DIR)
