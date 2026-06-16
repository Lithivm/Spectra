"""Colour palette registry + LUT builder — zero Qt dependencies, numpy only.

Public API:
    PALETTE        — name → display label
    build_lut_np(palette_name) → (256, 4) uint8 RGBA numpy array
    is_spectra(palette_name)   → True if palette uses custom brightness curve
"""

import numpy as np

# ── Display labels ─────────────────────────────────────────────────
# spectra is the custom palette with brightness curve; others are linear matplotlib

PALETTE: dict[str, str] = {
    "spectra":  "Spectra (custom)",
    "inferno":  "Inferno",
    "viridis":  "Viridis",
    "plasma":   "Plasma",
    "magma":    "Magma",
    "hot":      "Hot (black→red→yellow)",
    "coolwarm": "Coolwarm",
    "seismic":  "Seismic",
    "turbo":    "Turbo (Google)",
    "jet":      "Jet (rainbow)",
}

# ── LUT constants ──────────────────────────────────────────────────

LUT_SIZE = 256
DB_MIN = -120.0
DB_MAX = 0.0

# ── Custom spectra palette (inherits old inferno custom stops) ────

SPECTRA_STOPS: list[tuple[float, tuple[float, float, float]]] = [
    (0.00, (0.00, 0.00, 0.00)),       # black — noise floor
    (0.10, (0.00, 0.00, 0.02)),        # near-black
    (0.24, (0.08, 0.01, 0.34)),
    (0.42, (0.37, 0.07, 0.43)),
    (0.59, (0.69, 0.16, 0.21)),
    (0.76, (0.92, 0.37, 0.07)),
    (0.89, (0.99, 0.65, 0.04)),
    (1.00, (0.99, 0.88, 0.37)),
]

# ── Standard matplotlib colormap names ─────────────────────────────

STANDARD_CMAPS = [
    "inferno", "viridis", "plasma", "magma",
    "hot", "coolwarm", "seismic", "turbo", "jet",
]

# ── Brightness curve presets ───────────────────────────────────────
# Keys match shader uniforms: u_curve_power, u_curve_lo, u_curve_span
# Standard palettes use identity (power=1, lo=0, span=1) — no transform.

LINEAR_CURVE = {"power": 1.0, "lo": 0.0, "span": 1.0}

SPECTRA_CURVE = {"power": 0.5, "lo": 0.15, "span": 0.70}


def is_spectra(palette_name: str) -> bool:
    """Return True if palette uses custom brightness curve (spectra only)."""
    return palette_name == "spectra"


def get_curve_params(palette_name: str) -> dict[str, float]:
    """Return brightness curve parameters for a palette.

    Returns dict with keys ``power``, ``lo``, ``span``.
    """
    if palette_name == "spectra":
        return SPECTRA_CURVE
    return LINEAR_CURVE


# ── LUT builder (pure numpy) ───────────────────────────────────────

def _rgb_lerp(stops: list, t: float) -> tuple[float, float, float]:
    """Linear interpolation between RGB anchor stops."""
    if t <= stops[0][0]:
        return stops[0][1]
    if t >= stops[-1][0]:
        return stops[-1][1]
    for i in range(len(stops) - 1):
        t0, c0 = stops[i]
        t1, c1 = stops[i + 1]
        if t0 <= t <= t1:
            f = (t - t0) / (t1 - t0) if t1 > t0 else 0.0
            return (
                c0[0] + f * (c1[0] - c0[0]),
                c0[1] + f * (c1[1] - c0[1]),
                c0[2] + f * (c1[2] - c0[2]),
            )
    return stops[-1][1]


_lut_np_cache: dict[str, np.ndarray] = {}


def _export_standard_lut(name: str) -> np.ndarray:
    """Export a standard matplotlib colormap as (256, 4) uint8 RGBA LUT."""
    import matplotlib.cm as cm
    cmap = cm.get_cmap(name, 256)
    rgb = (cmap(np.linspace(0, 1, 256))[:, :3] * 255).astype(np.uint8)
    arr = np.zeros((256, 4), dtype=np.uint8)
    arr[:, :3] = rgb
    arr[:, 3] = 255
    return arr


def build_lut_np(palette_name: str = "spectra") -> np.ndarray:
    """Return shape=(256, 4) uint8 RGBA LUT. Cached per palette name."""
    cached = _lut_np_cache.get(palette_name)
    if cached is not None:
        return cached

    if palette_name == "spectra":
        # Custom palette via interpolation
        stops = SPECTRA_STOPS
        arr = np.zeros((LUT_SIZE, 4), dtype=np.uint8)
        for i in range(LUT_SIZE):
            x = i / (LUT_SIZE - 1)
            r, g, b = _rgb_lerp(stops, x)
            arr[i] = [int(r * 255), int(g * 255), int(b * 255), 255]
        _lut_np_cache[palette_name] = arr
        return arr

    if palette_name in STANDARD_CMAPS:
        arr = _export_standard_lut(palette_name)
        _lut_np_cache[palette_name] = arr
        return arr

    # Fallback to spectra
    return build_lut_np("spectra")
