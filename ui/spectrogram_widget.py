"""SpectrogramWidget — iZotope RX-style deep-background spectrogram.

Key features:
- -90 dB noise floor → deep black background
- -30 dB knee → only musical signals light up
- 75% overlap + Gamma 1.0 → smooth texture, curve-driven shaping
- Cubic interpolation on low frequencies → eliminates mosaic
- QImage-based blit rendering → single-pass, no per-pixel drawRect
"""

import os
import sys
import numpy as np
from PyQt6.QtWidgets import QWidget, QLabel
from PyQt6.QtGui import QPainter, QColor, QFont, QPen, QImage
from PyQt6.QtCore import Qt, QRectF, QPointF, pyqtSignal
from PyQt6.QtOpenGLWidgets import QOpenGLWidget
from OpenGL.GL import *
from lang import t


def _load_shader(name: str) -> str:
    """Load a GLSL shader source file, with PyInstaller support."""
    # Try relative to this file's directory first (dev and --onedir)
    base = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(base, "shaders", name)
    if not os.path.exists(path):
        # PyInstaller --onefile: look under sys._MEIPASS
        base2 = os.path.join(sys._MEIPASS, "ui", "shaders")
        path = os.path.join(base2, name)
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

# ── Palette anchor stops ──────────────────────────────────────────
_PALETTE_STOPS: dict[str, list[tuple[float, tuple[float, float, float]]]] = {
    "rx": [
        (0.00, (0.000, 0.000, 0.000)),       # black (-120 dB)
        (0.08, (0.000, 0.020, 0.120)),       # near-black deep blue (-110 dB)
        (0.18, (0.050, 0.050, 0.250)),       # dark violet-blue
        (0.30, (0.100, 0.120, 0.400)),       # purple
        (0.40, (0.000, 0.350, 0.550)),       # blue-cyan
        (0.48, (0.000, 0.550, 0.600)),       # cyan — RX primary
        (0.55, (0.550, 0.420, 0.000)),       # warm brown (~ -60 dB)
        (0.62, (0.880, 0.520, 0.000)),       # orange (~ -45 dB, knee)
        (0.72, (0.950, 0.320, 0.000)),       # orange-red
        (0.82, (0.920, 0.120, 0.000)),       # deep red
        (0.91, (0.980, 0.280, 0.080)),       # bright red-orange
        (0.97, (1.000, 0.650, 0.250)),       # bright warm highlight
        (1.00, (1.000, 1.000, 1.000)),       # white (0 dB)
    ],
    "inferno": [
        (0.00, (0.00, 0.00, 0.02)),
        (0.15, (0.08, 0.01, 0.34)),
        (0.35, (0.37, 0.07, 0.43)),
        (0.55, (0.69, 0.16, 0.21)),
        (0.75, (0.92, 0.37, 0.07)),
        (0.90, (0.99, 0.65, 0.04)),
        (1.00, (0.99, 0.88, 0.37)),
    ],
    "viridis": [
        (0.00, (0.27, 0.00, 0.33)),
        (0.25, (0.28, 0.14, 0.46)),
        (0.50, (0.13, 0.53, 0.56)),
        (0.75, (0.37, 0.77, 0.37)),
        (1.00, (0.99, 0.91, 0.14)),
    ],
    "plasma": [
        (0.00, (0.05, 0.03, 0.53)),
        (0.25, (0.45, 0.01, 0.61)),
        (0.50, (0.62, 0.26, 0.37)),
        (0.75, (0.85, 0.53, 0.10)),
        (1.00, (0.94, 0.98, 0.13)),
    ],
    "magma": [
        (0.00, (0.001462, 0.000466, 0.013866)),
        (0.15, (0.156511, 0.034391, 0.404977)),
        (0.35, (0.407590, 0.102322, 0.381350)),
        (0.55, (0.706747, 0.173181, 0.289928)),
        (0.75, (0.916242, 0.385741, 0.110804)),
        (0.90, (0.987622, 0.643683, 0.038760)),
        (1.00, (0.987053, 0.875393, 0.372698)),
    ],
    "ice": [
        (0.00, (0.00, 0.00, 0.08)),
        (0.20, (0.00, 0.10, 0.28)),
        (0.40, (0.00, 0.25, 0.50)),
        (0.60, (0.10, 0.50, 0.75)),
        (0.80, (0.50, 0.80, 0.95)),
        (1.00, (0.95, 0.98, 1.00)),
    ],
    "fire": [
        (0.00, (0.00, 0.00, 0.00)),
        (0.15, (0.12, 0.00, 0.00)),
        (0.35, (0.40, 0.08, 0.00)),
        (0.55, (0.75, 0.25, 0.00)),
        (0.75, (0.95, 0.55, 0.05)),
        (0.90, (1.00, 0.82, 0.20)),
        (1.00, (1.00, 1.00, 0.85)),
    ],
    "aurora": [
        (0.00, (0.02, 0.02, 0.15)),
        (0.20, (0.05, 0.20, 0.35)),
        (0.40, (0.10, 0.45, 0.30)),
        (0.60, (0.30, 0.65, 0.25)),
        (0.80, (0.70, 0.80, 0.45)),
        (1.00, (0.95, 0.95, 0.80)),
    ],
}

LUT_SIZE = 256
DB_MIN = -120.0
DB_MAX = 0.0
GAMMA = 1.0
KNEE_DB = -45.0   # lower knee — more signal stays in the dark region
NOISE_DB = -110.0  # noise floor crush starts here, pure black at DB_MIN


# ── LUT helpers ────────────────────────────────────────────────────

def _rgb_lerp(stops: list, t: float) -> tuple[float, float, float]:
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


def build_lut(palette_name: str = "rx") -> list[QColor]:
    """Precompute 256-entry colour LUT.

    Three-region dB→brightness curve:
      DB_MIN … NOISE_DB : power-law crush  →  near-black  (deep background)
      NOISE_DB … KNEE_DB : smooth power-law ramp  →  gradual colour emergence
      KNEE_DB … 0 dB     : power-law rise  →  bright musical peaks

    Gamma = 1.0 (linear — curve handles the shaping).
    """
    stops = _PALETTE_STOPS.get(palette_name, _PALETTE_STOPS["rx"])

    # Normalised positions
    nf = (NOISE_DB - DB_MIN) / (DB_MAX - DB_MIN)
    kn = (KNEE_DB - DB_MIN) / (DB_MAX - DB_MIN)

    lut: list[QColor] = []
    for i in range(LUT_SIZE):
        x = i / (LUT_SIZE - 1)          # 0 = DB_MIN, 1 = 0 dB

        if x < nf:
            # noise floor: soft crush to black — only pure black near DB_MIN
            t = (x / nf) ** 2.0 * 0.06
        elif x < kn:
            # mid-range: slow ramp, stays dark
            s = (x - nf) / (kn - nf)
            t = 0.06 + (s ** 1.8) * 0.39
        else:
            # above knee: fast brightening
            s = (x - kn) / (1.0 - kn)
            t = 0.45 + 0.55 * (s ** 0.4)

        t = max(0.0, min(1.0, t))
        # Gamma correction (currently 1.0 — identity)
        t = t ** GAMMA
        r, g, b = _rgb_lerp(stops, t)
        lut.append(QColor(int(r * 255), int(g * 255), int(b * 255), 250))
    return lut


def build_lut_np(palette_name: str = "rx") -> np.ndarray:
    """Return shape=(256,4) uint8 numpy LUT for vectorised rendering."""
    qcolors = build_lut(palette_name)
    arr = np.zeros((LUT_SIZE, 4), dtype=np.uint8)
    for i, c in enumerate(qcolors):
        arr[i] = [c.red(), c.green(), c.blue(), c.alpha()]
    return arr


# ══════════════════════════════════════════════════════════════════════
# Axis / colorbar widgets — painted alongside the GL spectrogram
# ══════════════════════════════════════════════════════════════════════

class _YAxisWidget(QWidget):
    """Frequency axis (left). Paints labels + short tick marks."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._freqs: np.ndarray | None = None
        self._yscale_mode = "linear"
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)

    def set_data(self, freqs, yscale_mode: str = "linear") -> None:
        self._freqs = np.asarray(freqs) if freqs is not None else None
        self._yscale_mode = yscale_mode
        self.update()

    def paintEvent(self, event) -> None:
        if self._freqs is None or len(self._freqs) == 0:
            return
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        w, h = self.width(), self.height()
        pad_top, pad_bot = 10, 10
        h_eff = h - pad_top - pad_bot

        font = QFont("Segoe UI, sans-serif", 7)
        painter.setFont(font)

        nyquist = float(self._freqs[-1])
        _ALL_TICKS = [
            0, 20, 50, 100, 200, 500, 1000, 2000, 5000,
            10000, 16000, 20000, 22050, 24000, 32000,
            40000, 44100, 48000, 64000, 80000, 88200, 96000,
        ]
        candidates = [t for t in _ALL_TICKS if t == 0 or t <= nyquist * 1.01]
        MAX_TICKS = 11
        if len(candidates) <= MAX_TICKS:
            visible_ticks = candidates
        else:
            inner = candidates[1:-1]
            n_inner = MAX_TICKS - 2
            step = len(inner) / n_inner
            sampled = [inner[min(int(i * step), len(inner) - 1)] for i in range(n_inner)]
            visible_ticks = [candidates[0]] + sampled + [candidates[-1]]

        f_min = max(float(self._freqs[0]), 1.0)
        f_max = float(self._freqs[-1])

        last_y = None
        for tick in visible_ticks:
            if tick > nyquist * 1.02:
                continue

            if self._yscale_mode == "log":
                frac = 0.0 if tick == 0 else (
                    (np.log10(max(tick, 1.0)) - np.log10(f_min)) /
                    (np.log10(f_max) - np.log10(f_min))
                )
            elif self._yscale_mode == "mel":
                import librosa
                m_tick = librosa.hz_to_mel(max(tick, 1.0))
                m_min = librosa.hz_to_mel(f_min)
                m_max = librosa.hz_to_mel(f_max)
                frac = 0.0 if tick == 0 else (m_tick - m_min) / (m_max - m_min)
            else:
                frac = (tick - f_min) / (f_max - f_min) if f_max > f_min else 0.5

            frac = max(0.0, min(1.0, frac))
            y = int(pad_top + h_eff - frac * h_eff)

            if last_y is not None and abs(y - last_y) < 15:
                continue
            last_y = y

            # Tick mark (right edge, shorter)
            painter.setPen(QColor(150, 145, 140))
            painter.drawLine(QPointF(w - 4, y), QPointF(w, y))

            # Label
            if tick == 0:
                label = "0"
            elif tick >= 1000:
                label = f"{tick / 1000:.0f}k" if tick % 1000 == 0 else f"{tick / 1000:.1f}k"
            else:
                label = f"{tick}"
            painter.setPen(QColor(170, 166, 161))
            painter.drawText(
                QRectF(1, y - 9, w - 7, 18),
                Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter,
                label,
            )

        painter.end()


class _XAxisWidget(QWidget):
    """Time axis (bottom). Paints labels + short tick marks."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._duration = 0.0
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)

    def set_data(self, duration: float) -> None:
        self._duration = duration
        self.update()

    def paintEvent(self, event) -> None:
        if self._duration <= 0:
            return
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        w, h = self.width(), self.height()
        SIDE = 36
        left, right = SIDE, w - SIDE
        rw = right - left
        if rw <= 0:
            return

        font = QFont("Segoe UI, sans-serif", 8)
        painter.setFont(font)

        num_ticks = min(8, max(4, rw // 80))
        for i in range(num_ticks):
            t = (i / (num_ticks - 1)) * self._duration if num_ticks > 1 else 0
            x = l + int((i / (num_ticks - 1)) * rw) if num_ticks > 1 else l

            # Tick mark (top edge)
            painter.setPen(QColor(150, 145, 140))
            painter.drawLine(QPointF(x, 0), QPointF(x, 5))

            # Label
            if t < 1:
                label = f"{t * 1000:.0f}ms"
            elif t < 60:
                label = f"{int(t)}s"
            else:
                m = int(t // 60)
                s = int(t % 60)
                label = f"{m}m{s:02d}s"
            painter.setPen(QColor(170, 166, 161))
            painter.drawText(
                QRectF(x - 35, 5, 70, h - 5),
                Qt.AlignmentFlag.AlignCenter, label,
            )

        painter.end()


class _ColorBarWidget(QWidget):
    """dB colorbar (right). Paints gradient + tick marks + dB labels."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._lut: np.ndarray = np.zeros((256, 4), dtype=np.uint8)
        self._bar_img: QImage | None = None
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)

    def set_data(self, lut_np: np.ndarray) -> None:
        self._lut = np.ascontiguousarray(lut_np[:, :4]) if lut_np is not None else np.zeros((256, 4), dtype=np.uint8)
        self._bar_img = None
        self.update()

    def _build_bar_image(self) -> None:
        """Build a vertical 1-pixel-wide gradient image — top = 0 dB, bottom = -90 dB."""
        lut = self._lut
        n_lut = len(lut)
        img = QImage(1, n_lut, QImage.Format.Format_RGBA8888)
        for i in range(n_lut):
            r, g, b, a = int(lut[i][0]), int(lut[i][1]), int(lut[i][2]), int(lut[i][3])
            # Row 0 = top = 0 dB (lut[-1]), row n_lut-1 = bottom = -90 dB (lut[0])
            img.setPixelColor(0, n_lut - 1 - i, QColor(r, g, b, a))
        self._bar_img = img

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        w, h = self.width(), self.height()
        SIDE = 36
        PROTRUDE = 5
        pad_top = SIDE - PROTRUDE
        pad_bot = SIDE - PROTRUDE
        h_eff = h - pad_top - pad_bot
        if h_eff <= 0:
            painter.end()
            return

        bar_x = 2
        bar_w = 7

        # Gradient bar — use scaled QImage instead of per-pixel fillRect
        if self._bar_img is None:
            self._build_bar_image()
        painter.drawImage(QRectF(bar_x, pad_top, bar_w, h_eff),
                          self._bar_img,
                          QRectF(0, 0, 1, self._bar_img.height()))

        # Border
        painter.setPen(QColor(85, 83, 79))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawRect(bar_x, pad_top, bar_w, h_eff)

        # dB ticks
        font = QFont("Segoe UI, sans-serif", 7)
        painter.setFont(font)
        DB_MIN_V = -120.0
        DB_MAX_V = 0.0
        db_ticks = [0, -20, -40, -60, -80, -100, -120]

        for db_val in db_ticks:
            frac = (db_val - DB_MAX_V) / (DB_MIN_V - DB_MAX_V)
            y = int(pad_top + frac * h_eff)
            painter.setPen(QColor(150, 145, 140))
            painter.drawLine(QPointF(bar_x + bar_w, y), QPointF(bar_x + bar_w + 3, y))
            painter.setPen(QColor(170, 166, 161))
            painter.drawText(
                QRectF(bar_x + bar_w + 4, y - 7, w - bar_x - bar_w - 4, 14),
                Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
                f"{db_val}",
            )

        painter.end()


# ══════════════════════════════════════════════════════════════════════
# SpectrogramGLWidget — GPU-accelerated spectrogram via QOpenGLWidget
# ══════════════════════════════════════════════════════════════════════

class SpectrogramGLWidget(QOpenGLWidget):
    """OpenGL spectrogram renderer.

    The raw STFT dB matrix is uploaded as a GL_R32F 2-D texture.
    Y-axis log/linear mapping, brightness normalisation, and colormap
    lookup all run in a GLSL fragment shader — no CPU preprocessing
    after initial upload.
    """

    seekRequested = pyqtSignal(float)  # seconds

    def __init__(self, parent=None):
        super().__init__(parent)
        self._data: np.ndarray | None = None       # (n_freqs, n_frames) float32 dB
        self._tex_id: int | None = None
        self._gl_program: int | None = None
        self._vao: int | None = None
        self._palette_name = "inferno"
        self._lut_tex_id: int | None = None
        self._lut_np: np.ndarray = np.zeros((256, 4), dtype=np.uint8)
        self._vmin = -120.0
        self._vmax = 0.0
        self._yscale_mode = "linear"
        self._freq_min = 20.0
        self._freq_max = 22050.0
        self._needs_upload = False
        self.frequencies = None
        self.duration = 0.0

        # ── Loading overlay — QLabel avoids QPainter text on FBO ───
        self._progress_label = QLabel(self)
        self._progress_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._progress_label.setFixedSize(200, 50)
        self._progress_label.setStyleSheet(
            "QLabel {"
            "  background: rgba(0, 0, 0, 180);"
            "  border-radius: 8px;"
            "  color: #F0EDE8;"
            "  font-size: 18px;"
            "  font-weight: 600;"
            "}"
        )
        self._progress_label.setVisible(False)

        # ── Cutoff annotation ─────────────────────────────────────
        self._cutoff_hz: float | None = None
        self._show_cutoff = True

        # ── Playhead ──────────────────────────────────────────────
        self.playhead_pos: float = -1.0  # seconds, -1 = hidden; set by main_window
        self._on_playhead_drag: callable | None = None

        # ── Streaming state ────────────────────────────────────────
        self._is_streaming = False
        self._stream_total = 0
        self._stream_filled = 0
        self._stream_needs_realloc = False
        self._pending_blocks: list[tuple[int, np.ndarray]] = []

        self._rebuild_lut()

    # ── Public API ──────────────────────────────────────────────────

    def set_data(self, data: np.ndarray) -> None:
        """Receive (n_freqs, n_frames) float32 dB matrix."""
        self._data = data.astype(np.float32)
        self._stream_total = data.shape[1]
        self._stream_filled = data.shape[1]
        self._is_streaming = False
        self._vmin = -120.0
        self._vmax = 0.0
        self._needs_upload = True
        self.update()

    def set_audio(self, data: dict) -> None:
        db = data.get('spectrogram', None)
        if db is not None and db.size > 0:
            self.frequencies = data.get('fft_freqs', None)
            self.duration = data.get('duration', 0.0)
            self.start_time = data.get('start_time', 0.0)
            freqs = data.get('fft_freqs', None)
            if freqs is not None and len(freqs) > 0:
                self._freq_min = float(freqs[0])
                self._freq_max = float(freqs[-1])
            self.set_data(db)

    def begin_stream(self, n_freqs: int, total_cols: int,
                     freqs: np.ndarray, duration: float) -> None:
        """Start streaming mode — pre-allocate texture, show axes."""
        self._is_streaming = True
        self._stream_total = total_cols
        self._stream_filled = 0
        self._stream_needs_realloc = True
        self._pending_blocks.clear()
        self.frequencies = freqs
        self.duration = duration
        self._freq_min = float(freqs[0]) if len(freqs) > 0 else 20.0
        self._freq_max = float(freqs[-1]) if len(freqs) > 0 else 22050.0
        self._data = None
        self._needs_upload = False
        self.update()

    def push_block(self, start_col: int, block_db: np.ndarray) -> None:
        """Enqueue a block of columns for the next paintGL pass."""
        self._pending_blocks.append((start_col, block_db))
        self.update()

    def end_stream(self, full_db: np.ndarray) -> None:
        """Finish streaming — hold the complete matrix for resize repaints."""
        self._data = np.ascontiguousarray(full_db, dtype=np.float32)
        self._stream_filled = full_db.shape[1]
        self._is_streaming = False
        self.update()

    def set_palette(self, name: str) -> None:
        self._palette_name = name
        self._rebuild_lut()
        if self.isValid():
            self.makeCurrent()
            self._upload_lut()
            self.doneCurrent()
        self.update()

    def _on_yscale_changed(self, scale: str) -> None:
        self._yscale_mode = scale
        self.update()

    def get_figure(self):
        return None

    # ── Cutoff line ───────────────────────────────────────────────────

    def set_cutoff_line(self, hz: float | None) -> None:
        self._cutoff_hz = hz
        self.update()

    def set_playhead(self, seconds: float) -> None:
        """Set playhead position — called by main_window only."""
        if seconds != self.playhead_pos:
            self.playhead_pos = seconds
            self.update()

    # ── LUT ─────────────────────────────────────────────────────────

    # ── Progress bar ────────────────────────────────────────────────────

    def show_progress(self, _pct: float = 0.0) -> None:
        """Show the loading overlay."""
        self._progress_label.setText(t("加载中…", "Loading…"))
        self._reposition_progress_label()
        self._progress_label.setVisible(True)

    def hide_progress(self) -> None:
        """Hide the loading overlay."""
        self._progress_label.setVisible(False)

    def _reposition_progress_label(self) -> None:
        w, h = self.width(), self.height()
        lw, lh = 200, 50
        self._progress_label.move((w - lw) // 2, (h - lh) // 2)

    def _paint_cutoff_overlay(self, painter: QPainter) -> None:
        if self._cutoff_hz is None or self._cutoff_hz <= 0:
            return
        if not hasattr(self, '_freq_min') or not hasattr(self, '_freq_max'):
            return
        f_min, f_max = self._freq_min, self._freq_max
        if f_min <= 0 or f_max <= f_min:
            return

        cutoff = self._cutoff_hz
        if self._yscale_mode == "log":
            f_min_safe = max(f_min, 1.0)
            frac = (np.log10(max(cutoff, 1.0)) - np.log10(f_min_safe)) / (
                np.log10(f_max) - np.log10(f_min_safe))
        elif self._yscale_mode == "mel":
            import librosa
            m_cut = librosa.hz_to_mel(max(cutoff, 1.0))
            m_min = librosa.hz_to_mel(f_min)
            m_max = librosa.hz_to_mel(f_max)
            frac = (m_cut - m_min) / (m_max - m_min) if m_max > m_min else 0.5
        else:
            frac = (cutoff - f_min) / (f_max - f_min) if f_max > f_min else 0.5

        frac = max(0.0, min(1.0, frac))
        # Match paintGL insets: 2px margins
        ml, mr = 2, 2
        y = int(self.height() * (1.0 - frac))

        color = QColor("#E0554D")
        pen = QPen(color)
        pen.setStyle(Qt.PenStyle.DashLine)
        pen.setWidth(1)
        painter.setPen(pen)
        painter.drawLine(ml, y, self.width() - mr - 1, y)

        font = QFont("Segoe UI, sans-serif", 8)
        painter.setFont(font)
        painter.setPen(color)
        label = f"Cutoff: {cutoff / 1000:.1f} kHz"
        painter.drawText(
            QRectF(ml + 4, y - 14, 140, 14),
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
            label,
        )

    def _rebuild_lut(self) -> None:
        lut_np = build_lut_np(self._palette_name)
        self._lut_np = np.ascontiguousarray(lut_np[:, :4])

    # ── OpenGL lifecycle ────────────────────────────────────────────

    def initializeGL(self) -> None:
        glClearColor(0, 0, 0, 1)
        glPixelStorei(GL_UNPACK_ALIGNMENT, 1)

        vert_src = _load_shader("spectrogram.vert")
        frag_src = _load_shader("spectrogram.frag")

        def _compile_shader(src: str, shader_type: int) -> int:
            s = glCreateShader(shader_type)
            glShaderSource(s, src)
            glCompileShader(s)
            if not glGetShaderiv(s, GL_COMPILE_STATUS):
                err = glGetShaderInfoLog(s).decode(errors='replace')
                raise RuntimeError(f"Shader compile error: {err}")
            return s

        vert = _compile_shader(vert_src, GL_VERTEX_SHADER)
        frag = _compile_shader(frag_src, GL_FRAGMENT_SHADER)

        self._gl_program = glCreateProgram()
        glAttachShader(self._gl_program, vert)
        glAttachShader(self._gl_program, frag)
        glLinkProgram(self._gl_program)
        if not glGetProgramiv(self._gl_program, GL_LINK_STATUS):
            err = glGetProgramInfoLog(self._gl_program).decode(errors='replace')
            raise RuntimeError(f"Shader link error: {err}")
        glDeleteShader(vert)
        glDeleteShader(frag)

        self._u_spec      = glGetUniformLocation(self._gl_program, "u_spec")
        self._u_colormap  = glGetUniformLocation(self._gl_program, "u_colormap")
        self._u_vmin      = glGetUniformLocation(self._gl_program, "u_vmin")
        self._u_vmax      = glGetUniformLocation(self._gl_program, "u_vmax")
        self._u_log_scale = glGetUniformLocation(self._gl_program, "u_log_scale")
        self._u_f_min        = glGetUniformLocation(self._gl_program, "u_f_min")
        self._u_f_max        = glGetUniformLocation(self._gl_program, "u_f_max")
        self._u_filled_cols  = glGetUniformLocation(self._gl_program, "u_filled_cols")
        self._u_total_cols   = glGetUniformLocation(self._gl_program, "u_total_cols")
        self._u_n_freqs      = glGetUniformLocation(self._gl_program, "u_n_freqs")

        self._vao = glGenVertexArrays(1)

        self._tex_id = glGenTextures(1)
        self._lut_tex_id = glGenTextures(1)

        self._upload_lut()
        if self._data is not None:
            self._upload_texture()

    def paintEvent(self, event) -> None:
        super().paintEvent(event)
        # Only create QPainter when there's an overlay to draw —
        # avoids unnecessary OpenGL framebuffer resolves that cause flicker.
        has_overlay = (
            (self._show_cutoff and self._cutoff_hz is not None and self._cutoff_hz > 0)
            or (self.playhead_pos >= 0 and self.duration > 0)
        )
        if not has_overlay:
            return
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        if self._show_cutoff and self._cutoff_hz is not None and self._cutoff_hz > 0:
            self._paint_cutoff_overlay(painter)
        if self.playhead_pos >= 0 and self.duration > 0:
            self._paint_playhead(painter)
        painter.end()

    def _paint_playhead(self, painter: QPainter) -> None:
        x = int(self.playhead_pos / self.duration * self.width()) if self.duration > 0 else 0
        painter.setPen(QPen(QColor("#e8e6e2"), 1))
        painter.drawLine(x, 0, x, self.height())

    def mousePressEvent(self, event) -> None:
        if (event.button() == Qt.MouseButton.LeftButton
                and self.playhead_pos >= 0 and self.duration > 0):
            px = int(self.playhead_pos / self.duration * self.width())
            if abs(event.position().x() - px) <= 20:
                self._dragging = True
                self.setCursor(Qt.CursorShape.SizeHorCursor)
                return
        # Click anywhere on the spectrogram seeks immediately
        if event.button() == Qt.MouseButton.LeftButton and self.duration > 0:
            secs = max(0, min(event.position().x() / self.width(), 1.0)) * self.duration
            self.playhead_pos = secs
            self.seekRequested.emit(secs)
            self.update()
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:
        if getattr(self, '_dragging', False) and self.duration > 0:
            secs = max(0, min(event.position().x() / self.width(), 1.0)) * self.duration
            self.playhead_pos = secs
            if self._on_playhead_drag is not None:
                self._on_playhead_drag(secs)
            self.update()  # smooth visual drag, no seek signal
        else:
            super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event) -> None:
        if getattr(self, '_dragging', False) and event.button() == Qt.MouseButton.LeftButton:
            self._dragging = False
            self.setCursor(Qt.CursorShape.ArrowCursor)
            if self.duration > 0:
                secs = max(0, min(event.position().x() / self.width(), 1.0)) * self.duration
                self.seekRequested.emit(secs)  # single seek on release
        else:
            super().mouseReleaseEvent(event)

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._reposition_progress_label()

    def resizeGL(self, w: int, h: int) -> None:
        glViewport(0, 0, w, h)

    def paintGL(self) -> None:
        w = int(self.width() * self.devicePixelRatio())
        h = int(self.height() * self.devicePixelRatio())

        # ── Streaming: allocate noise-floor texture on first frame ──
        if self._stream_needs_realloc and self._is_streaming:
            n_freqs = len(self.frequencies) if self.frequencies is not None else 1025
            total_cols = self._stream_total
            blank = np.full((n_freqs, total_cols), -120.0, dtype=np.float32)
            glBindTexture(GL_TEXTURE_2D, self._tex_id)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
            glTexImage2D(GL_TEXTURE_2D, 0, GL_R32F,
                         total_cols, n_freqs, 0,
                         GL_RED, GL_FLOAT, blank)
            self._data = blank
            self._stream_needs_realloc = False
            glBindTexture(GL_TEXTURE_2D, 0)

        # ── Non-streaming: legacy upload ──
        if self._needs_upload and self._data is not None and not self._is_streaming:
            self._upload_texture()
            self._needs_upload = False

        # ── Drain pending streaming blocks ──
        if self._pending_blocks:
            glBindTexture(GL_TEXTURE_2D, self._tex_id)
            max_filled = self._stream_filled
            for c0, blk in self._pending_blocks:
                blk = np.ascontiguousarray(blk, dtype=np.float32)
                n_freqs, bw = blk.shape
                glTexSubImage2D(GL_TEXTURE_2D, 0, c0, 0, bw, n_freqs,
                               GL_RED, GL_FLOAT, blk)
                max_filled = max(max_filled, c0 + bw)
            self._stream_filled = max_filled
            self._pending_blocks.clear()
            glBindTexture(GL_TEXTURE_2D, 0)

        # Minimal inset — axes are now separate widgets
        ml, mr, mt, mb = 2, 2, 2, 2
        rw = max(w - ml - mr, 1)
        rh = max(h - mt - mb, 1)

        glViewport(0, 0, w, h)
        glDisable(GL_SCISSOR_TEST)
        glClear(GL_COLOR_BUFFER_BIT)
        glEnable(GL_SCISSOR_TEST)
        glScissor(ml, mb, rw, rh)
        if self._data is None:
            glDisable(GL_SCISSOR_TEST)
            return

        glUseProgram(self._gl_program)

        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, self._tex_id)
        glUniform1i(self._u_spec, 0)

        glActiveTexture(GL_TEXTURE1)
        glBindTexture(GL_TEXTURE_2D, self._lut_tex_id)
        glUniform1i(self._u_colormap, 1)

        glUniform1f(self._u_vmin, self._vmin)
        glUniform1f(self._u_vmax, self._vmax)
        glUniform1i(self._u_log_scale, 1 if self._yscale_mode == "log" else 0)
        glUniform1f(self._u_f_min, self._freq_min)
        glUniform1f(self._u_f_max, self._freq_max)
        glUniform1i(self._u_filled_cols, self._stream_filled)
        glUniform1i(self._u_total_cols, self._stream_total)
        glUniform1f(self._u_n_freqs, float(self._data.shape[0]) if self._data is not None else 1025.0)

        glBindVertexArray(self._vao)
        glDrawArrays(GL_TRIANGLE_STRIP, 0, 4)
        glBindVertexArray(0)

        glUseProgram(0)
        glDisable(GL_SCISSOR_TEST)

    # ── Texture upload ──────────────────────────────────────────────

    def _upload_texture(self) -> None:
        if self._data is None:
            return
        data = np.ascontiguousarray(self._data, dtype=np.float32)
        n_freqs, n_frames = data.shape

        glBindTexture(GL_TEXTURE_2D, self._tex_id)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_R32F,
                     n_frames, n_freqs, 0,
                     GL_RED, GL_FLOAT, data)
        glBindTexture(GL_TEXTURE_2D, 0)

    def _upload_lut(self) -> None:
        glBindTexture(GL_TEXTURE_2D, self._lut_tex_id)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8,
                     256, 1, 0, GL_RGBA, GL_UNSIGNED_BYTE,
                     self._lut_np.tobytes())
        glBindTexture(GL_TEXTURE_2D, 0)
