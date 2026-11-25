"""Python code field with syntax highlighting.

Flet has no editable code editor, so the field is a stack of two layers with the same
font and metrics: below, a text with coloured spans; above, the actual TextField whose own
text is transparent (cursor and selection stay visible). Every change re-colours the
layer below. Colours only (no bold or italic), so every character keeps its width.
"""

import keyword
import re

import flet as ft

import style as S

KEYWORDS = set(keyword.kwlist) | {"match", "case"}
CONSTANTS = {"True", "False", "None"}
BUILTINS = {"abs", "all", "any", "bool", "dict", "divmod", "enumerate", "float", "int", "isinstance", "len",
            "list", "map", "max", "min", "pow", "print", "range", "reversed", "round", "set", "sorted", "str",
            "sum", "tuple", "zip", "math", "random"}

COLORS = {
    "keyword": S.ACCENT,
    "string": S.TEAL,
    "comment": S.MUTED,
    "number": S.CATEGORICAL[0],
    "builtin": S.CATEGORICAL[6],
    "constant": S.CATEGORICAL[6],
    "function": S.CATEGORICAL[5],
}

_TOKEN = re.compile(
    r"(?P<comment>#[^\n]*)"
    r"|(?P<string>[rbfuRBFU]{0,2}(?:\"\"\"[\s\S]*?(?:\"\"\"|$)|'''[\s\S]*?(?:'''|$)|\"[^\"\n]*\"?|'[^'\n]*'?))"
    r"|(?P<number>\b\d[\d_]*(?:\.\d*)?(?:[eE][+-]?\d+)?\b|\.\d+\b)"
    r"|(?P<name>[A-Za-z_]\w*)"
)


def tokens(code):
    """(text, kind) pieces of `code`; kind None is plain text."""
    out, pos, after_def = [], 0, False
    for m in _TOKEN.finditer(code):
        if m.start() > pos:
            out.append((code[pos:m.start()], None))
        kind, text = m.lastgroup, m.group()
        if kind == "name":
            if after_def:
                kind = "function"
            elif text in CONSTANTS:
                kind = "constant"
            elif text in KEYWORDS:
                kind = "keyword"
            elif text in BUILTINS:
                kind = "builtin"
            else:
                kind = None
            after_def = text in ("def", "class")
        else:
            after_def = False
        out.append((text, kind))
        pos = m.end()
    if pos < len(code):
        out.append((code[pos:], None))
    return out


def spans(code):
    return [ft.TextSpan(text, ft.TextStyle(color=COLORS[kind] if kind else S.INK, font_family=S.MONO,
                                           letter_spacing=0)) for text, kind in tokens(code)]


INDENT = 4
_DEDENT_AFTER = re.compile(r"^\s*(return\b|raise\b|pass\s*$|break\s*$|continue\s*$)")
_CONTINUATION = re.compile(r"^(\s*)(else|elif\b.*|except\b.*|finally)\s*:\s*$")


_OPENERS = {"else": {"if", "elif", "for", "while", "try", "except"}, "elif": {"if", "elif"},
            "except": {"try", "except"}, "finally": {"try", "except", "else"}}


def _code_part(line):
    """The line without a trailing comment (a # inside a string is good enough to ignore here)."""
    return line.split("#", 1)[0].rstrip()


def auto_indent(old, new):
    """Editor behaviour after a one-character edit, as in VS Code: a new line keeps the indentation
    of the previous one, one level more after a line ending in ':' and one level less after
    return / raise / pass / break / continue; typing the ':' of else / elif / except / finally
    moves that line back to the level of its if / try. Returns (text, cursor) or None."""
    if len(new) != len(old) + 1:
        return None
    p = 0
    while p < len(old) and old[p] == new[p]:
        p += 1
    typed = new[p]
    if typed == "\n":
        previous = new[:p].rsplit("\n", 1)[-1]
        indent = len(previous) - len(previous.lstrip(" "))
        code = _code_part(previous)
        if code.endswith(":"):
            indent += INDENT
        elif _DEDENT_AFTER.match(code):
            indent = max(0, indent - INDENT)
        if not indent:
            return None
        text = new[:p + 1] + " " * indent + new[p + 1:]
        return text, p + 1 + indent
    if typed == ":":
        start = new.rfind("\n", 0, p) + 1
        end = new.find("\n", p)
        line = new[start:end if end >= 0 else len(new)]
        m = _CONTINUATION.match(line)
        if not m:
            return None
        indent = len(m.group(1))
        word = m.group(2).split()[0].rstrip(":")
        openers = _OPENERS[word]
        # the level of the statement this line continues: the closest earlier if / try / ... at or
        # left of this line (VS Code moves `else:` there and leaves it alone if it is already there)
        target = None
        for earlier in reversed(new[:start].split("\n")):
            if earlier.strip():
                level = len(earlier) - len(earlier.lstrip(" "))
                if level <= indent and earlier.strip().split(" ")[0].rstrip(":") in openers:
                    target = level
                    break
        if target is None or target == indent:
            return None
        text = new[:start] + " " * target + line.lstrip(" ") + new[start + len(line):]
        return text, p + 1 - (indent - target)
    return None


class CodeEditor:
    """Multi-line Python editor; use `.control` in layouts, `.value`, `.on_change`, `.visible`."""

    SIZE, HEIGHT = 13, 1.45   # font size and line height factor (identical in both layers)

    def __init__(self, value="", lines=14, label=None, on_change=None, width=640):
        self.on_change = on_change
        self._last = value or ""
        inner = width - 2 * 12 - 2   # padding and border: both layers wrap at exactly this width
        # letter_spacing 0: Material gives text fields and texts different default spacings
        text_style = ft.TextStyle(font_family=S.MONO, size=self.SIZE, height=self.HEIGHT, letter_spacing=0)
        self.layer = ft.Text(spans=spans(value), style=text_style, width=inner)
        self.field = ft.TextField(
            value=value, multiline=True, min_lines=lines, max_lines=None, width=inner,
            border=ft.InputBorder.NONE, content_padding=ft.Padding.all(0), dense=True,
            text_style=ft.TextStyle(font_family=S.MONO, size=self.SIZE, height=self.HEIGHT, letter_spacing=0,
                                    color=ft.Colors.TRANSPARENT),
            cursor_color=S.INK, selection_color=ft.Colors.with_opacity(0.25, S.TEAL), on_change=self._changed)
        box_height = round(lines * self.SIZE * self.HEIGHT) + 24
        editor = ft.Container(
            ft.Column([ft.Stack([self.layer, self.field])], scroll=ft.ScrollMode.AUTO),
            height=box_height, width=width, padding=ft.Padding.symmetric(horizontal=12, vertical=12),
            border=ft.Border.all(1, S.RULE), border_radius=4, bgcolor=S.PANEL)
        self.control = ft.Column(([ft.Text(label, size=12.5, color=S.INK)] if label else []) + [editor], spacing=4)

    def _changed(self, e):
        new = self.field.value or ""
        edit = auto_indent(self._last, new)
        if edit is not None:
            new, cursor = edit
            self.field.value = new
            self.field.selection = ft.TextSelection(base_offset=cursor, extent_offset=cursor)
            self.field.update()
        self._last = new
        self.layer.spans = spans(new)
        self.layer.update()
        if self.on_change:
            return self.on_change(e)

    @property
    def value(self):
        return self.field.value or ""

    @value.setter
    def value(self, code):
        self.field.value = code
        self._last = code or ""
        self.layer.spans = spans(code or "")

    @property
    def visible(self):
        return self.control.visible

    @visible.setter
    def visible(self, flag):
        self.control.visible = flag
