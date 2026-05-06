"""Theme — curated palette and styling helpers (Claude/GPT dark)."""

# Palette
COLORS = {
    "bg":       "#0f1117",
    "bg2":      "#161922",
    "bg3":      "#1c1f2a",
    "bg4":      "#212430",
    "surface":  "#0f1117",
    "primary":  "#6e8efb",
    "primary2": "#4361f7",
    "secondary": "#7a8190",
    "text":     "#e8ecf1",
    "text2":    "#8b92a0",
    "text3":    "#4a4f5b",
    "red":      "#ff6b6b",
    "green":    "#4ecb71",
    "yellow":   "#f5c542",
    "blue":     "#6e8efb",
    "purple":   "#a78bfa",
    "border":   "#262a36",
    "hover":    "#202430",
}

# Corner radii
R = 8
R_MD = 12
R_LG = 16
R_XL = 20
R_PILL = 999

# Shadow radii
SHADOW = {
    "sm": 6,
    "md": 12,
    "lg": 18,
    "xl": 24,
}

def gradient(*stops, vertical=True):
    """Build a qlineargradient expression."""
    axis = ("x1:0, y1:0, x2:0, y2:1" if vertical else "x1:0, y1:0, x2:1, y2:0")
    parts = ",".join(f"stop:{i} {c}" for i, c in enumerate(stops))
    return f"qlineargradient({axis}, {parts})"

def shadow(style="md", offset=(0, 3)):
    """Return shadow style string."""
    r = SHADOW.get(style, 12)
    dx, dy = offset
    return f"drop-shadow:{dx}px {dy}px {r}px #00000040"

def glass(border=True, bg="bg2"):
    """Glassy panel background."""
    s = f"background-color: {COLORS[bg]}ee; border-radius: {R_MD}px;"
    if border:
        s += f" border: 1px solid {COLORS['border']}60;"
    return s
