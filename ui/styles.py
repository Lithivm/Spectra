"""Spectra design tokens — single source of truth for colors, fonts, geometry.

Every UI module must pull colors/sizes from here; no raw hex in widgets.
"""

# ── Backgrounds ───────────────────────────────────────────────────────────────
BG_BASE = "#222526"      # window root
BG_SURFACE = "#303436"  # cards / panels (raised)
BG_RAISED = "#292d2e"   # hover on surface rows
BG_WELL = "#1a1d1f"     # sunken wells: progress tracks, code areas, icon buttons
BG_CANVAS = "#1a1d1f"   # spectrogram canvas == LUT flat floor (must stay in sync)
CARD_INSET = 4          # inset well between card border and content

# ── Borders ───────────────────────────────────────────────────────────────────
BORDER_SUB = "#2e3133"  # card outlines
BORDER_MID = "#3d4143"  # inputs / dividers

# ── Accents ───────────────────────────────────────────────────────────────────
ACCENT = "#F0EDE8"      # warm off-white — primary action & highlights
ACCENT_HOVER = "#FFFFFF"    # primary hover (brighter)
ACCENT_PRESSED = "#E2DFD8"  # primary pressed (darker)
ACCENT_ALT = "#1C1B19"     # text on accent surfaces

# ── Status ────────────────────────────────────────────────────────────────────
ACCENT_GRN = "#34d399"   # success / ok
ACCENT_AMB = "#F5A623"  # warning
ACCENT_RED = "#E0554D"  # error / clipping

# ── Text ──────────────────────────────────────────────────────────────────────
TEXT_PRI = "#F0EDE8"    # primary text (== ACCENT)
TEXT_SEC = "#A09D96"    # secondary
TEXT_DIM = "#5C5952"    # muted / placeholder

# ── Axis & markers ────────────────────────────────────────────────────────────
AXIS_TEXT = "#AAA6A1"   # axis tick labels
AXIS_TICK = "#96918C"   # axis tick marks
AXIS_GRID = "rgba(120, 120, 120, 40)"  # grid lines
COLORBAR_BORDER = "#55534F"

CURSOR_LINE = "#FFFFFF"     # hover cursor line (alpha applied at use site)
PLAY_CURSOR = "#F0EDE8"    # playback position line
CUTOFF_LINE = ACCENT_RED   # high-freq cutoff annotation
WAVEFORM_LINE = ACCENT     # waveform envelope (alpha applied at use site)

# ── Overlays & progress ───────────────────────────────────────────────────────
OVERLAY_BG = "rgba(0, 0, 0, 140)"    # loading overlay scrim
PROGRESS_FILL = ACCENT               # progress chunk / floating bar fill
PROGRESS_BG = BG_WELL                # progress track

# ── Font ───────────────────────────────────────────────────────────────────────
FONT_FAMILY = "Segoe UI"
FS_XS = 9    # section labels / axis ticks
FS_SM = 10   # status bar / captions
FS_BODY = 11 # rows, toolbar labels
FS_MD = 12   # dialog body
FS_LG = 13   # panel titles
FS_XL = 15   # brand

# ── Geometry ───────────────────────────────────────────────────────────────────
CORNER_SM = 6    # buttons / combos / rows
CORNER_MD = 8    # floating chips / progress labels
CORNER_LG = 12   # cards / dialogs
SIDE = 36        # Y-axis width == colorbar width (single alignment constant)
