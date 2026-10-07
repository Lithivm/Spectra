"""Colour palette registry + LUT builder — pure numpy, zero external deps.

Public API:
    PALETTE        — name → display label
    build_lut_np(palette_name) → (256, 4) uint8 RGBA numpy array
    is_spectra(palette_name)   → True if palette uses custom brightness curve
"""

import numpy as np

# ── Display labels ─────────────────────────────────────────────────

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

# ── Brightness curve presets ───────────────────────────────────────
# Keys match shader uniforms: u_curve_power, u_curve_lo, u_curve_span
# Standard palettes use identity (power=1, lo=0, span=1) — no transform.

LINEAR_CURVE = {"power": 1.0, "lo": 0.0, "span": 1.0}

SPECTRA_CURVE = {"power": 0.5, "lo": 0.15, "span": 0.70}


# ── Color stops — sampled from matplotlib originals ────────────────
# Each entry: list of (position, (R, G, B)) with R/G/B in [0, 1].
# Positions must be sorted ascending from 0.0 to 1.0.

INFERNO_STOPS = [
    (0.000, (0.000, 0.000, 0.016)),
    (0.130, (0.058, 0.024, 0.208)),
    (0.250, (0.192, 0.043, 0.396)),
    (0.380, (0.365, 0.055, 0.471)),
    (0.500, (0.541, 0.090, 0.435)),
    (0.630, (0.718, 0.165, 0.318)),
    (0.750, (0.867, 0.282, 0.165)),
    (0.880, (0.969, 0.478, 0.024)),
    (1.000, (0.988, 0.998, 0.645)),
]

VIRIDIS_STOPS = [
    (0.000, (0.267, 0.004, 0.329)),
    (0.130, (0.282, 0.141, 0.458)),
    (0.250, (0.254, 0.265, 0.530)),
    (0.380, (0.207, 0.372, 0.553)),
    (0.500, (0.164, 0.471, 0.558)),
    (0.630, (0.128, 0.567, 0.551)),
    (0.750, (0.135, 0.659, 0.518)),
    (0.880, (0.267, 0.749, 0.441)),
    (1.000, (0.993, 0.906, 0.144)),
]

PLASMA_STOPS = [
    (0.000, (0.050, 0.030, 0.528)),
    (0.130, (0.290, 0.012, 0.628)),
    (0.250, (0.484, 0.012, 0.658)),
    (0.380, (0.655, 0.055, 0.600)),
    (0.500, (0.800, 0.134, 0.490)),
    (0.630, (0.906, 0.255, 0.365)),
    (0.750, (0.965, 0.412, 0.232)),
    (0.880, (0.980, 0.596, 0.100)),
    (1.000, (0.940, 0.975, 0.131)),
]

MAGMA_STOPS = [
    (0.000, (0.001, 0.000, 0.014)),
    (0.130, (0.063, 0.028, 0.200)),
    (0.250, (0.196, 0.047, 0.380)),
    (0.380, (0.361, 0.055, 0.482)),
    (0.500, (0.529, 0.082, 0.522)),
    (0.630, (0.702, 0.149, 0.494)),
    (0.750, (0.859, 0.267, 0.396)),
    (0.880, (0.965, 0.451, 0.306)),
    (1.000, (0.987, 0.991, 0.750)),
]

HOT_STOPS = [
    (0.000, (0.000, 0.000, 0.000)),
    (0.333, (1.000, 0.000, 0.000)),
    (0.667, (1.000, 1.000, 0.000)),
    (1.000, (1.000, 1.000, 1.000)),
]

COOLWARM_STOPS = [
    (0.000, (0.230, 0.299, 0.754)),
    (0.250, (0.477, 0.565, 0.922)),
    (0.500, (0.865, 0.865, 0.865)),
    (0.750, (0.922, 0.565, 0.477)),
    (1.000, (0.706, 0.016, 0.150)),
]

SEISMIC_STOPS = [
    (0.000, (0.000, 0.000, 0.350)),
    (0.250, (0.000, 0.250, 0.950)),
    (0.500, (1.000, 1.000, 1.000)),
    (0.750, (0.950, 0.250, 0.000)),
    (1.000, (0.350, 0.000, 0.000)),
]

TURBO_STOPS = [
    (0.000, (0.189, 0.071, 0.232)),
    (0.100, (0.251, 0.252, 0.634)),
    (0.200, (0.162, 0.440, 0.900)),
    (0.300, (0.058, 0.627, 0.950)),
    (0.400, (0.131, 0.780, 0.692)),
    (0.500, (0.394, 0.875, 0.452)),
    (0.600, (0.658, 0.916, 0.250)),
    (0.700, (0.874, 0.882, 0.150)),
    (0.800, (0.972, 0.700, 0.120)),
    (0.900, (0.900, 0.400, 0.050)),
    (1.000, (0.478, 0.015, 0.010)),
]

JET_STOPS = [
    (0.000, (0.000, 0.000, 0.500)),
    (0.125, (0.000, 0.000, 1.000)),
    (0.375, (0.000, 1.000, 1.000)),
    (0.500, (0.000, 1.000, 0.000)),
    (0.625, (1.000, 1.000, 0.000)),
    (0.875, (1.000, 0.000, 0.000)),
    (1.000, (0.500, 0.000, 0.000)),
]

# Master table — maps palette name → color stops
_STOPS_TABLE: dict[str, list[tuple[float, tuple[float, float, float]]]] = {
    "spectra": [
        (0.00, (0.00, 0.00, 0.00)),
        (0.10, (0.00, 0.00, 0.02)),
        (0.24, (0.08, 0.01, 0.34)),
        (0.42, (0.37, 0.07, 0.43)),
        (0.59, (0.69, 0.16, 0.21)),
        (0.76, (0.92, 0.37, 0.07)),
        (0.89, (0.99, 0.65, 0.04)),
        (1.00, (0.99, 0.88, 0.37)),
    ],
    "inferno":  INFERNO_STOPS,
    "viridis":  VIRIDIS_STOPS,
    "plasma":   PLASMA_STOPS,
    "magma":    MAGMA_STOPS,
    "hot":      HOT_STOPS,
    "coolwarm": COOLWARM_STOPS,
    "seismic":  SEISMIC_STOPS,
    "turbo":    TURBO_STOPS,
    "jet":      JET_STOPS,
}


# ── Public helpers ─────────────────────────────────────────────────

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


def _build_lut_from_stops(stops: list[tuple[float, tuple[float, float, float]]]) -> np.ndarray:
    """Build a (256, 4) uint8 RGBA LUT from color stops via linear interpolation."""
    arr = np.zeros((LUT_SIZE, 4), dtype=np.uint8)
    for i in range(LUT_SIZE):
        x = i / (LUT_SIZE - 1)
        r, g, b = _rgb_lerp(stops, x)
        arr[i] = [int(r * 255), int(g * 255), int(b * 255), 255]
    return arr


_lut_np_cache: dict[str, np.ndarray] = {}


def build_lut_np(palette_name: str = "spectra") -> np.ndarray:
    """Return shape=(256, 4) uint8 RGBA LUT. Cached per palette name."""
    cached = _lut_np_cache.get(palette_name)
    if cached is not None:
        return cached

    stops = _STOPS_TABLE.get(palette_name)
    if stops is None:
        # Unknown name → fallback to spectra
        stops = _STOPS_TABLE["spectra"]

    arr = _build_lut_from_stops(stops)
    _lut_np_cache[palette_name] = arr
    return arr
