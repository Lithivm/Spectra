"""SpectrogramRenderer — 将频谱数据渲染为 matplotlib 图像。"""

from __future__ import annotations

from typing import Any

import matplotlib
matplotlib.use("QtAgg")

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors
import numpy as np


# ------------------------------------------------------------------
# 配色方案
# ------------------------------------------------------------------
# Classic warm-tone spectrogram palette
_rx_colors = [
    (0.000, 0.000, 0.000),   # #000000  black
    (0.102, 0.000, 0.314),   # #1a0050  deep purple
    (0.000, 0.188, 0.565),   # #003090  deep blue
    (0.000, 0.502, 0.753),   # #0080c0  blue
    (0.000, 0.784, 0.784),   # #00c8c8  cyan
    (0.502, 0.753, 0.000),   # #80c000  yellow-green
    (0.878, 0.816, 0.000),   # #e0d000  yellow
    (0.878, 0.376, 0.000),   # #e06000  orange
    (0.753, 0.000, 0.000),   # #c00000  deep red
    (1.000, 0.251, 0.251),   # #ff4040  bright red
    (1.000, 1.000, 1.000),   # #ffffff  white
]
_cm_rx = matplotlib.colors.LinearSegmentedColormap.from_list("rx", _rx_colors)

PALETTE = {
    "rx": ("RX (iZotope style)", _cm_rx),
    "viridis": ("Viridis", cm.viridis),
    "plasma": ("Plasma", cm.plasma),
    "inferno": ("Inferno", cm.inferno),
    "magma": ("Magma", cm.magma),
    "cividis": ("Cividis", cm.cividis),
    "hot": ("Hot (black→red→yellow)", cm.hot),
    "coolwarm": ("Coolwarm", cm.coolwarm),
    "seismic": ("Seismic", cm.seismic),
    "ice": ("Ice", matplotlib.colors.LinearSegmentedColormap.from_list("ice", [
        (0.00, 0.00, 0.08), (0.20, 0.00, 0.10, 0.28), (0.40, 0.00, 0.25, 0.50),
        (0.60, 0.10, 0.50, 0.75), (0.80, 0.50, 0.80, 0.95), (1.00, 0.95, 0.98, 1.00),
    ])),
    "fire": ("Fire", matplotlib.colors.LinearSegmentedColormap.from_list("fire", [
        (0.00, 0.00, 0.00), (0.15, 0.12, 0.00, 0.00), (0.35, 0.40, 0.08, 0.00),
        (0.55, 0.75, 0.25, 0.00), (0.75, 0.95, 0.55, 0.05),
        (0.90, 1.00, 0.82, 0.20), (1.00, 1.00, 1.00, 0.85),
    ])),
    "aurora": ("Aurora", matplotlib.colors.LinearSegmentedColormap.from_list("aurora", [
        (0.00, 0.02, 0.02, 0.15), (0.20, 0.05, 0.20, 0.35), (0.40, 0.10, 0.45, 0.30),
        (0.60, 0.30, 0.65, 0.25), (0.80, 0.70, 0.80, 0.45), (1.00, 0.95, 0.95, 0.80),
    ])),
}


class SpectrogramRenderer:
    """将频谱数据渲染到 matplotlib Axes。"""

    def __init__(self) -> None:
        self._fig, self._ax = plt.subplots(figsize=(10, 4))
        self._im = None
        self._cbar = None
        self._title = ""
        self._palette = "viridis"
        self._db_min = -100.0
        self._db_max = 0.0
        self._show_cbar = True
        self._y_axis_scale = "log"
        self._freq_labels: list[float] | None = None

    # ---- setters ----
    def set_title(self, title: str) -> None:
        self._title = title

    def set_palette(self, name: str) -> None:
        if name in PALETTE:
            self._palette = name
        else:
            self._palette = "viridis"

    def set_db_range(self, db_min: float, db_max: float) -> None:
        self._db_min = db_min
        self._db_max = db_max

    def set_y_scale(self, scale: str) -> None:
        self._y_axis_scale = scale

    def set_freq_labels(self, labels: list[float] | None) -> None:
        self._freq_labels = labels

    def show_colorbar(self, show: bool = True) -> None:
        self._show_cbar = show

    # ---- render ----
    def render(self, freqs: np.ndarray, times: np.ndarray, db: np.ndarray) -> None:
        """将频谱绘制到 self._ax。"""
        self._ax.clear()

        self._im = self._ax.imshow(
            db,
            aspect='auto',
            origin='lower',
            extent=[times[0], times[-1], freqs[0], freqs[-1]],
            cmap=PALETTE[self._palette][1],
            vmin=self._db_min,
            vmax=self._db_max,
            interpolation='nearest',
        )

        self._ax.set_yscale(self._y_axis_scale)
        self._ax.set_ylabel("频率 (Hz)")
        self._ax.set_xlabel("时间 (s)")
        self._ax.set_title(self._title)

        if self._freq_labels:
            self._ax.set_yticks(self._freq_labels, [f"{f:.0f}" for f in self._freq_labels])

        if self._show_cbar:
            if self._cbar is not None:
                self._cbar.remove()
                self._cbar = None
            self._cbar = self._fig.colorbar(self._im, ax=self._ax, pad=0.05)
            self._cbar.set_label("幅度 (dB)")

        self._ax.grid(True, alpha=0.3)

    def get_figure(self) -> Any:
        """返回 matplotlib Figure 对象（可用于嵌入 QtWidget）。"""
        return self._fig

    def get_axes(self) -> Any:
        """返回 Axes，供外部进一步修改。"""
        return self._ax


# Backward-compatible alias used by main_window.py
Renderer = SpectrogramRenderer
