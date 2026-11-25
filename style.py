"""Colours and fonts shared by the UI and the plots."""

import platform

# A warm "lab notebook" palette: paper, ink and two signal colours.
PAPER = "#F2EFE7"       # window background
SIDEBAR = "#E9E5DA"     # settings column
PANEL = "#FBFAF6"       # cards / plot background
INK = "#1F1F1C"         # primary text
MUTED = "#6F6D65"       # secondary text
FAINT = "#A3A097"       # tertiary text, grid labels
RULE = "#D9D4C7"        # hairlines
RULE_SOFT = "#E8E4DA"   # grid lines inside plots

ACCENT = "#C8431D"      # vermilion: best individual, primary action
ACCENT_DARK = "#A5350F"
ACCENT_SOFT = "#F3DDD3"
TEAL = "#1E5A61"        # population averages
TEAL_SOFT = "#8DB1B4"
OCHRE = "#B8860B"       # outliers
GREEN = "#2E7D4F"       # program synthesis: solved
GREEN_LIGHT = "#EDEFE6" # program synthesis: solved by nobody

# Categorical palette for many series (building-block usage), validated with the
# dataviz palette checker on the plot surface: fixed order, never cycled.
CATEGORICAL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]

_system = platform.system()
if _system == "Darwin":
    SANS, MONO = "Helvetica Neue", "Menlo"
elif _system == "Windows":
    SANS, MONO = "Segoe UI", "Consolas"
else:
    SANS, MONO = "DejaVu Sans", "DejaVu Sans Mono"

# Matplotlib falls back along this list if the first font is missing.
PLOT_SANS = [SANS, "Helvetica Neue", "Segoe UI", "Arial", "DejaVu Sans"]
PLOT_MONO = [MONO, "Menlo", "Consolas", "DejaVu Sans Mono"]
