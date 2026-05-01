"""SpectrogramWidget — iZotope RX-style deep-background spectrogram.

Key features:
- -90 dB noise floor → deep black background
- -30 dB knee → only musical signals light up
- 75% overlap + Gamma 1.0 → smooth texture, curve-driven shaping
- Cubic interpolation on low frequencies → eliminates mosaic
- QImage-based blit rendering → single-pass, no per-pixel drawRect
"""

import time
import ctypes

import numpy as np
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout
from PyQt6.QtGui import QPainter, QColor, QPen, QFont, QBrush, QPainterPath, QImage
from PyQt6.QtCore import Qt, QRectF, QPointF, pyqtSignal, QObject
from PyQt6.QtOpenGLWidgets import QOpenGLWidget
from OpenGL.GL import *

# ── Progress pipeline phases ────────────────────────────────────────

ProgressPhase = str  # "decoded" | "stft_done" | "render_done"

from lang import t
from ui.styles import BORDER_MID, TEXT_DIM

# ── Palette anchor stops ──────────────────────────────────────────
_PALETTE_STOPS: dict[str, list[tuple[float, tuple[float, float, float]]]] = {
    "rx": [
        (0.00, (0.000, 0.000, 0.000)),       # black
        (0.15, (0.000, 0.020, 0.120)),       # near-black deep blue
        (0.30, (0.000, 0.200, 0.350)),       # deep blue-cyan
        (0.48, (0.000, 0.550, 0.600)),       # cyan ← RX default primary
        (0.62, (0.200, 0.600, 0.300)),       # cyan-green transition
        (0.72, (0.700, 0.500, 0.000)),       # orange-yellow
        (0.82, (0.900, 0.280, 0.000)),       # orange
        (0.91, (0.800, 0.050, 0.000)),       # deep red-orange
        (0.97, (1.000, 0.400, 0.200)),       # bright orange-red
        (1.00, (1.000, 1.000, 1.000)),       # white
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
DB_MIN = -90.0
DB_MAX = 0.0
GAMMA = 1.0
KNEE_DB = -45.0   # lower knee — more signal stays in the dark region
NOISE_DB = -75.0  # deeper noise floor crush


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
            # noise floor: fast crush to black
            t = (x / nf) ** 4.0 * 0.01
        elif x < kn:
            # mid-range: slow ramp, stays dark
            s = (x - nf) / (kn - nf)
            t = 0.01 + (s ** 1.8) * 0.44
        else:
            # above knee: fast brightening
            s = (x - kn) / (1.0 - kn)
            t = 0.45 + 0.55 * (s ** 0.5)

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


# ── Frequency scale resampling ─────────────────────────────────────

def _resample_freq_scale(
    data: np.ndarray,
    freqs_in: np.ndarray,
    scale: str,
    sr: float,
    target_bins: int = 1024,
) -> tuple[np.ndarray, np.ndarray]:
    """Resample spectrogram to a target frequency scale.

    Uses cubic interpolation below 3 kHz to eliminate mosaic artefacts,
    linear interpolation above for transient preservation.

    Scales: 'linear', 'log', 'mel', 'bark'
    """
    from scipy.interpolate import interp1d

    f_min = max(freqs_in[1] if len(freqs_in) > 1 else freqs_in[0], 40.0)
    f_max = min(freqs_in[-1], sr / 2.0)

    if scale == "linear":
        target_freqs = np.linspace(f_min, f_max, target_bins)
    elif scale == "mel":
        import librosa
        target_freqs = librosa.mel_frequencies(n_mels=target_bins, fmin=f_min, fmax=f_max)
    elif scale == "bark":
        z_min = 26.81 / (1.0 + 1960.0 / f_min) - 0.53
        z_max = 26.81 / (1.0 + 1960.0 / f_max) - 0.53
        z = np.linspace(z_min, z_max, target_bins)
        target_freqs = 1960.0 / (26.81 / (z + 0.53) - 1.0)
    else:  # log
        target_freqs = np.logspace(np.log10(f_min), np.log10(f_max), target_bins)

    # Ensure strictly increasing source frequencies
    mask = np.ones(len(freqs_in), dtype=bool)
    for i in range(1, len(freqs_in)):
        if freqs_in[i] <= freqs_in[i - 1]:
            mask[i] = False
    freqs_clean = freqs_in[mask]
    data_clean = data[mask, :]

    CROSSOVER = 3000.0
    lo_mask = target_freqs <= CROSSOVER
    hi_mask = target_freqs > CROSSOVER

    result = np.zeros((len(target_freqs), data_clean.shape[1]), dtype=np.float32)

    if np.any(lo_mask):
        cubic = interp1d(
            freqs_clean, data_clean, axis=0,
            kind='cubic', bounds_error=False, fill_value=DB_MIN,
        )
        result[lo_mask, :] = cubic(target_freqs[lo_mask]).astype(np.float32)

    if np.any(hi_mask):
        linear = interp1d(
            freqs_clean, data_clean, axis=0,
            kind='linear', bounds_error=False, fill_value=DB_MIN,
        )
        result[hi_mask, :] = linear(target_freqs[hi_mask]).astype(np.float32)

    return result, target_freqs.astype(np.float64)


# ── Widget ─────────────────────────────────────────────────────────

class SpectrogramWidget(QWidget):
    """Deep-background professional spectrogram — iZotope RX style."""

    def __init__(self):
        super().__init__()
        self.setMinimumHeight(240)

        # Data
        self.audio = None
        self.frequencies = None
        self.data = None
        self.start_time = 0.0
        self.duration = 0.0
        self._times = None
        self._original_freqs = None
        self._original_data = None

        # Settings
        self._palette_name = "inferno"
        self._yscale = "linear"
        self._mode = "standard"
        self._pre_emphasis = True
        self._pre_emphasis_crossover = 500.0

        # LUT
        self._lut = build_lut(self._palette_name)
        self._lut_np = build_lut_np(self._palette_name)

        # QImage cache
        self._cached_image: QImage | None = None
        self._cache_widget_size: tuple = (0, 0)
        self._cache_palette: str = ""
        self._cache_yscale: str = ""

    # ── Public API ─────────────────────────────────────────────

    def set_audio(self, data: dict) -> None:
        t0 = time.perf_counter()
        self.audio = data
        self._original_freqs = np.asarray(data.get('fft_freqs', []), dtype=np.float64)
        self._original_data = data.get('spectrogram', None)
        self.start_time = data.get('start_time', 0.0)
        self.duration = data.get('duration', 0.0)
        self._times = data.get('times', None)
        self._mode = data.get('mode', 'standard')

        if self._original_data is not None and self._original_data.size > 0:
            print(f"[spectrogram] shape={self._original_data.shape}, "
                  f"min={self._original_data.min():.1f}, max={self._original_data.max():.1f}, "
                  f"mode={self._mode}")

            sr = self.audio.get('sample_rate', 44100)
            data_work = self._original_data.copy()
            freqs_work = self._original_freqs.copy()

            # High-frequency pre-emphasis: +6 dB/octave above crossover
            if self._pre_emphasis:
                mask = freqs_work > self._pre_emphasis_crossover
                if np.any(mask):
                    boost_db = np.zeros(len(freqs_work), dtype=np.float64)
                    boost_db[mask] = 6.0 * np.log2(
                        freqs_work[mask] / self._pre_emphasis_crossover)
                    data_work = data_work + boost_db[:, np.newaxis]

            # Clamp to DB range
            data_work = np.clip(data_work, DB_MIN, DB_MAX)

            # ── Downsample time axis to screen width before freq resample ──
            rw = max(self.width() - 80, 200)
            n_frames = data_work.shape[1]
            if n_frames > rw:
                chunk = n_frames // rw
                data_work = data_work[:, :chunk * rw].reshape(
                    data_work.shape[0], rw, chunk
                ).max(axis=2)  # (n_freqs, rw)

            # Resample to target frequency scale — match viewport resolution
            target_bins = min(2048, max(512, int(self.height()) * 2))
            t_resample = time.perf_counter()
            data_resampled, freqs_resampled = _resample_freq_scale(
                data_work, freqs_work, self._yscale, sr, target_bins,
            )
            print(f"[PROFILE] _resample_freq_scale: {time.perf_counter() - t_resample:.3f}s ({self._yscale}, {target_bins} bins)")

            # Guard monotonicity
            diffs = np.diff(freqs_resampled)
            if not np.all(diffs > 0):
                keep = np.ones(len(freqs_resampled), dtype=bool)
                for i in range(1, len(freqs_resampled)):
                    if freqs_resampled[i] <= freqs_resampled[i - 1]:
                        keep[i] = False
                freqs_resampled = freqs_resampled[keep]
                data_resampled = data_resampled[keep, :]

            self.frequencies = freqs_resampled
            self.data = data_resampled

            print(f"[spectrogram] resample ({self._yscale}): "
                  f"{len(freqs_resampled)} bins, "
                  f"{freqs_resampled[0]:.0f}–{freqs_resampled[-1]:.0f} Hz")
        else:
            self.frequencies = None
            self.data = None

        self._cached_image = None
        print(f"[PROFILE] set_audio total: {time.perf_counter() - t0:.3f}s")
        self._first_paint = True
        self.update()

    # ── Painting ───────────────────────────────────────────────

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        w, h = self.width(), self.height()
        rect = QRectF(0, 0, w, h)

        clip_path = QPainterPath()
        clip_path.addRoundedRect(rect, 10, 10)
        painter.setClipPath(clip_path)

        if self.data is None or self.data.size == 0:
            self._draw_empty(painter, rect)
            painter.end()
            return

        ml, mr, mt, mb = 62, 46, 6, 26
        rw = int(rect.width()) - ml - mr
        rh = int(rect.height()) - mt - mb
        if rw <= 0 or rh <= 0:
            painter.end()
            return

        # self._draw_spectrogram_image(painter, ml, mt, rw, rh)
        self._draw_axes(painter, ml, mr, mt, mb, rw, rh)
        # self._draw_colorbar(painter, ml, mr, mt, mb, rw, rh)
        painter.end()

    # ── QImage spectrogram rendering ───────────────────────────

    def _draw_spectrogram_image(
        self, painter: QPainter, ml: int, mt: int, rw: int, rh: int,
    ) -> None:
        """Build a QImage from dB data via LUT, blit in one pass."""
        t0 = time.perf_counter()
        first_paint = getattr(self, '_first_paint', False)

        mag = self.data.T  # (n_frames, n_freqs)
        n_frames, n_freqs = mag.shape
        t_down = time.perf_counter()

        # ── Time downsampling: vectorised RMS power average (75% overlap) ──
        if n_frames > rw:
            chunk_size = max(8, int(np.ceil(n_frames / rw * 4.0)))
            hop_t = max(1, chunk_size // 4)   # 75% overlap
            num_chunks = min(rw, (n_frames - chunk_size) // hop_t + 1)
            window = np.hanning(chunk_size).astype(np.float32)
            wsum = window.sum()

            # Build output in blocks to keep memory ~<100 MB
            BLK = 128
            parts = []
            for b in range(0, num_chunks, BLK):
                end = min(b + BLK, num_chunks)
                starts = np.arange(b, end) * hop_t                # (blk,)
                idx = starts[:, None] + np.arange(chunk_size)      # (blk, chunk_size)
                chunks = mag[idx]                                   # (blk, chunk_size, n_freqs)
                pwr = 10.0 ** (chunks / 10.0)
                weighted = (pwr * window[np.newaxis, :, np.newaxis]).sum(axis=1) / wsum
                parts.append(10.0 * np.log10(weighted + 1e-12))

            mag = np.concatenate(parts, axis=0).astype(np.float32) if parts else mag
            n_frames = mag.shape[0]

        t_numpy = time.perf_counter()

        # ── Fully vectorised rendering, no Python loops ──────────────
        db_range = DB_MAX - DB_MIN
        lut_max = LUT_SIZE - 1
        lut_np = self._lut_np          # shape (256, 4) uint8

        # 1. Frequency axis interpolation: row → freq bin, in one pass
        row_idx = (np.arange(rh, dtype=np.float32)[::-1] + 0.5) / rh * n_freqs
        row_lo = np.clip(row_idx.astype(np.int32), 0, n_freqs - 1)
        row_hi = np.clip(row_lo + 1, 0, n_freqs - 1)
        row_w = (row_idx - row_lo).astype(np.float32)  # shape (rh,)

        # 2. Time-axis resample (n_frames → rw)
        if n_frames > rw:
            chunk = n_frames // rw          # frames per output column
            use = chunk * rw                # truncate to exact multiple
            reshaped = mag[:use].reshape(rw, chunk, n_freqs)
            cols = reshaped.max(axis=1) * 0.7 + reshaped.mean(axis=1) * 0.3
        else:
            src_x = np.linspace(0, n_frames - 1, rw, dtype=np.float32)
            src_lo = np.clip(src_x.astype(np.int32), 0, n_frames - 1)
            src_hi = np.clip(src_lo + 1, 0, n_frames - 1)
            src_w = (src_x - src_lo).astype(np.float32)
            cols = mag[src_lo] + src_w[:, None] * (mag[src_hi] - mag[src_lo])

        # Frequency-axis interpolation → (rw, rh)
        vals = cols[:, row_lo] + row_w * (cols[:, row_hi] - cols[:, row_lo])

        # 3. dB → LUT index
        lut_idx = np.clip(
            ((vals - DB_MIN) / db_range * lut_max),
            0, lut_max
        ).astype(np.uint8)                                 # (rw, rh)

        # 4. LUT lookup → RGBA, transpose to (rh, rw, 4)
        rgba = lut_np[lut_idx]                             # (rw, rh, 4)
        img = rgba.transpose(1, 0, 2).copy()               # (rh, rw, 4)

        t_qimg = time.perf_counter()
        qimg = QImage(img.tobytes(), rw, rh, rw * 4, QImage.Format.Format_RGBA8888)
        self._cached_image = qimg.copy()
        t_blit = time.perf_counter()
        self._cache_widget_size = (self.width(), self.height())
        self._cache_palette = self._palette_name
        self._cache_yscale = self._yscale

        painter.drawImage(ml, mt, self._cached_image)

        if first_paint:
            self._first_paint = False
            print(f"[PROFILE] first paint breakdown: "
                  f"downsample={t_numpy - t_down:.3f}s, "
                  f"numpy+lut={t_qimg - t_numpy:.3f}s, "
                  f"qimage={t_blit - t_qimg:.3f}s, "
                  f"blit={time.perf_counter() - t_blit:.3f}s, "
                  f"total={time.perf_counter() - t0:.3f}s "
                  f"({rw}x{rh}, n_freqs_in={n_freqs})")

    # ── Axes ────────────────────────────────────────────────────

    def _draw_axes(
        self, painter: QPainter, ml: int, mr: int,
        mt: int, mb: int, rw: int, rh: int,
    ) -> None:
        font = QFont("system-ui, sans-serif", 9)
        painter.setFont(font)

        freqs_arr = np.asarray(self.frequencies)
        n_freqs = len(freqs_arr)
        max_freq = freqs_arr[-1] if n_freqs > 0 else 22050.0

        nyquist = float(freqs_arr[-1]) if n_freqs > 0 else 20000.0

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
            sampled = [inner[min(int(i * step), len(inner)-1)] for i in range(n_inner)]
            visible_ticks = [candidates[0]] + sampled + [candidates[-1]]

        last_y = None

        for tick in visible_ticks:
            if tick > max_freq * 1.02:
                continue
            # Calculate normalised position directly from scale formula,
            # independent of the bin distribution of the underlying array.
            f_min = max(freqs_arr[0], 1.0)
            f_max = freqs_arr[-1]

            if self._yscale == "log":
                if tick == 0:
                    frac = 0.0
                else:
                    frac = (np.log10(max(tick, 1.0)) - np.log10(f_min)) / \
                           (np.log10(f_max) - np.log10(f_min))
            elif self._yscale == "mel":
                import librosa
                m_tick = librosa.hz_to_mel(max(tick, 1.0))
                m_min = librosa.hz_to_mel(f_min)
                m_max = librosa.hz_to_mel(f_max)
                frac = 0.0 if tick == 0 else (m_tick - m_min) / (m_max - m_min)
            else:  # linear
                frac = (tick - freqs_arr[0]) / (f_max - freqs_arr[0]) if f_max > freqs_arr[0] else 0.5

            frac = max(0.0, min(1.0, frac))
            y = mt + rh - int(frac * rh)

            if last_y is not None and abs(y - last_y) < 18:
                continue
            last_y = y

            grid_c = QColor(BORDER_MID)
            grid_c.setAlpha(38)
            painter.setPen(grid_c)
            painter.drawLine(QPointF(ml - 4, y), QPointF(ml + rw + mr - 10, y))

            painter.setPen(QColor(TEXT_DIM))
            painter.drawLine(QPointF(ml - 6, y), QPointF(ml, y))

            if tick == 0:
                label = "0"
            elif tick >= 1000:
                label = f"{tick / 1000:.0f}k" if tick % 1000 == 0 else f"{tick / 1000:.1f}k"
            else:
                label = f"{tick}Hz"
            painter.drawText(
                QRectF(0, y - 10, ml - 10, 20),
                Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter,
                label,
            )

        if self.duration > 0:
            painter.setFont(QFont("system-ui, sans-serif", 9))
            num_ticks = min(8, max(4, rw // 80))
            for i in range(num_ticks):
                t = (i / (num_ticks - 1)) * self.duration if num_ticks > 1 else 0
                x = ml + int((i / (num_ticks - 1)) * rw) if num_ticks > 1 else ml
                painter.setPen(QColor(BORDER_MID))
                painter.drawLine(QPointF(x, mt + rh), QPointF(x, mt + rh + 4))
                painter.setPen(QColor(TEXT_DIM))
                if t < 1:
                    label = f"{t * 1000:.0f}ms"
                elif t < 60:
                    label = f"{t:.1f}s"
                else:
                    label = f"{t / 60:.1f}min"
                painter.drawText(
                    QRectF(x - 30, mt + rh + 6, 60, 16),
                    Qt.AlignmentFlag.AlignCenter, label,
                )

    # ── Colour bar ──────────────────────────────────────────────

    def _draw_colorbar(
        self, painter: QPainter, ml: int, mr: int,
        mt: int, mb: int, rw: int, rh: int,
    ) -> None:
        bar_x = ml + rw + 8
        bar_w = 10

        for py in range(rh):
            idx = int((rh - 1 - py) / (rh - 1) * (LUT_SIZE - 1)) if rh > 1 else 0
            idx = max(0, min(LUT_SIZE - 1, idx))
            painter.fillRect(int(bar_x), int(mt + py), int(bar_w), 2, self._lut[idx])

        border_c = QColor(TEXT_DIM)
        border_c.setAlpha(80)
        painter.setPen(border_c)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawRect(bar_x, mt, bar_w, rh)

        font = QFont("system-ui, sans-serif", 8)
        painter.setFont(font)
        db_ticks = [0, -20, -40, -60, -80, -100]
        for db_val in db_ticks:
            frac = (db_val - DB_MAX) / (DB_MIN - DB_MAX)
            y = mt + int(frac * rh)
            painter.setPen(QColor(TEXT_DIM))
            painter.drawText(
                QRectF(bar_x + bar_w + 3, y - 8, mr - bar_w - 14, 16),
                Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
                f"{db_val}",
            )

    def _draw_empty(self, painter: QPainter, rect: QRectF) -> None:
        painter.setPen(QColor(BORDER_MID))
        font = QFont("system-ui, sans-serif", 13)
        painter.setFont(font)
        painter.drawText(
            rect, Qt.AlignmentFlag.AlignCenter,
            t("声谱图 — 打开音频文件查看", "Spectrogram — open an audio file to view"),
        )

    # ── Callbacks ───────────────────────────────────────────────

    def _on_palette_changed(self, name: str) -> None:
        self._palette_name = name
        self._lut = build_lut(name)
        self._lut_np = build_lut_np(name)
        self._cached_image = None
        self.update()

    def _on_yscale_changed(self, scale: str) -> None:
        self._yscale = scale
        if self._original_data is not None and self._original_data.size > 0:
            sr = self.audio.get('sample_rate', 44100) if self.audio else 44100
            data_w = self._original_data.copy()

            if self._pre_emphasis and self._original_freqs is not None:
                mask = self._original_freqs > self._pre_emphasis_crossover
                if np.any(mask):
                    boost_db = np.zeros(len(self._original_freqs), dtype=np.float64)
                    boost_db[mask] = 6.0 * np.log2(
                        self._original_freqs[mask] / self._pre_emphasis_crossover)
                    data_w = data_w + boost_db[:, np.newaxis]

            data_w = np.clip(data_w, DB_MIN, DB_MAX)

            rw = max(self.width() - 80, 200)
            n_frames = data_w.shape[1]
            if n_frames > rw:
                chunk = n_frames // rw
                data_w = data_w[:, :chunk * rw].reshape(
                    data_w.shape[0], rw, chunk
                ).max(axis=2)

            data_r, freqs_r = _resample_freq_scale(
                data_w, self._original_freqs, scale, sr,
                min(2048, max(512, int(self.height()) * 2)),
            )
            diffs = np.diff(freqs_r)
            if not np.all(diffs > 0):
                keep = np.ones(len(freqs_r), dtype=bool)
                for i in range(1, len(freqs_r)):
                    if freqs_r[i] <= freqs_r[i - 1]:
                        keep[i] = False
                freqs_r = freqs_r[keep]
                data_r = data_r[keep, :]
            self.frequencies = freqs_r
            self.data = data_r
        self._cached_image = None
        self.update()

    def _on_pre_emphasis_toggled(self, enabled: bool) -> None:
        self._pre_emphasis = enabled
        if self.audio and self._original_data is not None:
            self.set_audio({
                'spectrogram': self._original_data,
                'fft_freqs': self._original_freqs,
                'times': self._times,
                'start_time': self.start_time,
                'duration': self.duration,
                'sample_rate': self.audio.get('sample_rate', 44100),
                'mode': self._mode,
            })

    # ── Matplotlib export ───────────────────────────────────────

    def get_figure(self):
        import matplotlib
        matplotlib.use("Qt5Agg")
        import matplotlib.pyplot as plt

        if self.data is None or self.data.size == 0:
            return None
        if self._times is None or len(self.frequencies) == 0:
            return None

        fig, ax = plt.subplots(figsize=(10, 4))
        ax.imshow(
            self.data, aspect='auto', origin='lower',
            extent=[self._times[0], self._times[-1],
                    self.frequencies[0], self.frequencies[-1]],
            cmap=self._palette_name, vmin=DB_MIN, vmax=DB_MAX,
            interpolation='bilinear',
        )
        ax.set_ylabel("Frequency (Hz)")
        ax.set_xlabel("Time (s)")
        ax.grid(True, alpha=0.3)
        return fig


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
        l, r = SIDE, w - SIDE
        rw = r - l
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
                label = f"{t:.1f}s"
            else:
                label = f"{t / 60:.1f}min"
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
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)

    def set_data(self, lut_np: np.ndarray) -> None:
        self._lut = np.ascontiguousarray(lut_np[:, :4]) if lut_np is not None else np.zeros((256, 4), dtype=np.uint8)
        self.update()

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        w, h = self.width(), self.height()
        # Colorbar spans all 3 grid rows — align gradient to spectrogram
        # with 5 px protrusion above / below
        SIDE = 36
        PROTRUDE = 5
        pad_top = SIDE - PROTRUDE
        pad_bot = SIDE - PROTRUDE
        h_eff = h - pad_top - pad_bot
        if h_eff <= 0:
            painter.end()
            return

        bar_x = 2
        bar_w = 5
        lut = self._lut
        n_lut = len(lut)

        # Gradient bar
        for py in range(h_eff):
            idx = int((h_eff - 1 - py) / max(h_eff - 1, 1) * (n_lut - 1))
            idx = max(0, min(n_lut - 1, idx))
            r, g, b, a = int(lut[idx][0]), int(lut[idx][1]), int(lut[idx][2]), int(lut[idx][3])
            painter.fillRect(bar_x, pad_top + py, bar_w, 2, QColor(r, g, b, a))

        # Border
        painter.setPen(QColor(85, 83, 79))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawRect(bar_x, pad_top, bar_w, h_eff)

        # dB ticks
        font = QFont("Segoe UI, sans-serif", 7)
        painter.setFont(font)
        DB_MIN_V = -90.0
        DB_MAX_V = 0.0
        db_ticks = [0, -20, -40, -60, -80]

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

    def __init__(self, parent=None):
        super().__init__(parent)
        self._data: np.ndarray | None = None       # (n_freqs, n_frames) float32 dB
        self._tex_id: int | None = None
        self._gl_program: int | None = None
        self._vao: int | None = None
        self._palette_name = "inferno"
        self._lut_tex_id: int | None = None
        self._lut_np: np.ndarray = np.zeros((256, 4), dtype=np.uint8)
        self._vmin = -90.0
        self._vmax = 0.0
        self._yscale_mode = "linear"
        self._freq_min = 20.0
        self._freq_max = 22050.0
        self._needs_upload = False
        self.frequencies = None
        self.duration = 0.0

        # ── Loading overlay ────────────────────────────────────────
        self._progress_visible = False

        self._rebuild_lut()

    # ── Public API ──────────────────────────────────────────────────

    def set_data(self, data: np.ndarray) -> None:
        """Receive (n_freqs, n_frames) float32 dB matrix."""
        self._data = data.astype(np.float32)
        self._vmin = -120.0
        self._vmax = 0.0
        print(f"[GL] set_data: shape={data.shape}")
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

    def set_palette(self, name: str) -> None:
        self._palette_name = name
        self._rebuild_lut()
        if self.isValid():
            self.makeCurrent()
            self._upload_lut()
            self.doneCurrent()
        self.update()

    def _on_palette_changed(self, name: str) -> None:
        self.set_palette(name)

    def _on_yscale_changed(self, scale: str) -> None:
        self._yscale_mode = scale
        self.update()

    def get_figure(self):
        return None

    # ── LUT ─────────────────────────────────────────────────────────

    # ── Progress bar ────────────────────────────────────────────────────

    def show_progress(self, _pct: float = 0.0) -> None:
        """Show the loading overlay."""
        self._progress_visible = True
        self.repaint()

    def hide_progress(self) -> None:
        """Hide the loading overlay."""
        self._progress_visible = False
        self.update()

    def _paint_progress_overlay(self, painter: QPainter) -> None:
        if not self._progress_visible:
            return
        painter.save()
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        w, h = self.width(), self.height()
        cx, cy = w / 2.0, h / 2.0

        # Semi-transparent backdrop
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(0, 0, 0, 120))
        painter.drawRoundedRect(QRectF(cx - 60, cy - 16, 120, 32), 8, 8)

        # Text label
        font = QFont("Segoe UI, sans-serif", 12)
        font.setWeight(QFont.Weight.Medium)
        painter.setFont(font)
        painter.setPen(QColor(210, 207, 202))
        painter.drawText(
            QRectF(cx - 60, cy - 16, 120, 32),
            Qt.AlignmentFlag.AlignCenter,
            t("加载中…", "Loading…"),
        )

        painter.restore()

    def _rebuild_lut(self) -> None:
        lut_np = build_lut_np(self._palette_name)
        self._lut_np = np.ascontiguousarray(lut_np[:, :4])

    # ── OpenGL lifecycle ────────────────────────────────────────────

    def initializeGL(self) -> None:
        glClearColor(0, 0, 0, 1)

        vert_src = """
        #version 330 core
        const vec2 VERTS[4] = vec2[](
            vec2(-1,-1), vec2(1,-1), vec2(-1,1), vec2(1,1)
        );
        const vec2 UVS[4] = vec2[](
            vec2(0,0), vec2(1,0), vec2(0,1), vec2(1,1)
        );
        out vec2 uv;
        void main() {
            uv = UVS[gl_VertexID];
            gl_Position = vec4(VERTS[gl_VertexID], 0, 1);
        }
        """

        frag_src = """
        #version 330 core
        in vec2 uv;
        out vec4 fragColor;
        uniform sampler2D u_spec;
        uniform sampler2D u_colormap;
        uniform float u_vmin;
        uniform float u_vmax;
        uniform int u_log_scale;

        void main() {
            // Y-axis: log or linear
            float y;
            if (u_log_scale == 1) {
                float f_min_log  = log(20.0);
                float f_max_log  = log(22050.0);
                float f_val_log  = f_min_log + uv.y * (f_max_log - f_min_log);
                float f_norm = (exp(f_val_log) - 20.0) / (22050.0 - 20.0);
                y = clamp(f_norm, 0.0, 1.0);
            } else {
                y = uv.y;
            }

            float db = texture(u_spec, vec2(uv.x, y)).r;
            float t = clamp((db - u_vmin) / (u_vmax - u_vmin), 0.0, 1.0);

            fragColor = texture(u_colormap, vec2(t, 0.5));
        }
        """

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
        self._u_margin_l = glGetUniformLocation(self._gl_program, "u_margin_l")
        self._u_margin_b = glGetUniformLocation(self._gl_program, "u_margin_b")
        self._u_scale_x = glGetUniformLocation(self._gl_program, "u_scale_x")
        self._u_scale_y = glGetUniformLocation(self._gl_program, "u_scale_y")

        self._vao = glGenVertexArrays(1)

        self._tex_id = glGenTextures(1)
        self._lut_tex_id = glGenTextures(1)

        self._upload_lut()
        if self._data is not None:
            self._upload_texture()

    def paintEvent(self, event) -> None:
        super().paintEvent(event)
        if self._progress_visible:
            painter = QPainter(self)
            painter.setRenderHint(QPainter.RenderHint.Antialiasing)
            self._paint_progress_overlay(painter)
            painter.end()

    def resizeGL(self, w: int, h: int) -> None:
        print(f"[GL] resizeGL called")
        glViewport(0, 0, w, h)

    def paintGL(self) -> None:
        print(f"[GL] paintGL called")
        if self._needs_upload and self._data is not None:
            self._upload_texture()
            self._needs_upload = False

        # Minimal inset — axes are now separate widgets
        ml, mr, mt, mb = 2, 2, 2, 2
        w = int(self.width() * self.devicePixelRatio())
        h = int(self.height() * self.devicePixelRatio())
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
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_R32F,
                     n_frames, n_freqs, 0,
                     GL_RED, GL_FLOAT, data.tobytes())
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
