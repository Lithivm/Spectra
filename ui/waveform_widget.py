"""WaveformWidget — displays a waveform visualization."""

import numpy as np

from PyQt6.QtWidgets import QWidget
from PyQt6.QtGui import QPainter, QColor, QBrush, QPolygonF, QFont
from lang import t, on_lang_change
from ui.styles import BORDER_MID
from PyQt6.QtCore import Qt, QRectF, QPointF

_ENVELOPE_SIZE = 4096


class WaveformWidget(QWidget):
    """Shows the audio waveform in a smooth, visually pleasing style."""

    def __init__(self):
        super().__init__()
        self.setMinimumHeight(100)
        self.audio = None
        self.duration = 0.0
        self._envelope_cache: np.ndarray | None = None
        self._cached_polygon: QPolygonF | None = None
        on_lang_change(lambda _lang: self.update() if self.audio is None else None)

    def _build_envelope(self, channel_samples: np.ndarray) -> np.ndarray:
        n = len(channel_samples)
        if n <= _ENVELOPE_SIZE:
            return np.abs(channel_samples)
        chunk_size = n // _ENVELOPE_SIZE
        trimmed = channel_samples[:chunk_size * _ENVELOPE_SIZE]
        return np.max(np.abs(trimmed.reshape(_ENVELOPE_SIZE, chunk_size)), axis=1)

    def set_audio(self, waveform: np.ndarray, sample_rate: int, duration: float) -> None:
        self.audio = waveform
        self.duration = duration
        self._envelope_cache = None
        self._cached_polygon = None

        if waveform is not None and waveform.size > 0:
            self._envelope_cache = self._build_envelope(waveform)
            self._rebuild_polygon()

        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        rect = QRectF(0, 0, self.width(), self.height())

        if self._envelope_cache is None:
            self._draw_empty(painter, rect)
            painter.end()
            return

        self._draw_waveform(painter, rect)
        self._draw_axes(painter, rect)
        painter.end()

    def _draw_waveform(self, painter, rect):
        if self._cached_polygon is None:
            return
        painter.setPen(Qt.PenStyle.NoPen)
        line_color = QColor("#e8e6e2")
        line_color.setAlpha(220)
        painter.setBrush(QBrush(line_color))
        painter.drawPolygon(self._cached_polygon)

    def _rebuild_polygon(self) -> None:
        """Pre-compute the waveform QPolygonF from the envelope cache."""
        envelope = self._envelope_cache
        if envelope is None or len(envelope) == 0:
            self._cached_polygon = None
            return

        rw = self.width()
        rh = self.height()
        n = len(envelope)

        # 向量化计算所有坐标点
        indices = np.arange(n, dtype=np.float64)
        xs = (indices / max(1, n - 1) * rw).astype(np.int32) if n > 1 else np.full(n, rw // 2, dtype=np.int32)
        half = rh // 2
        ys_upper = (half - envelope * half).astype(np.int32)
        ys_lower = half + (half - ys_upper)

        # 上半部分 + 下半部分反转
        all_x = np.concatenate([xs, xs[::-1]])
        all_y = np.concatenate([ys_upper, ys_lower[::-1]])

        self._cached_polygon = QPolygonF(
            QPointF(float(x), float(y)) for x, y in zip(all_x, all_y)
        ) if n > 0 else None

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        if self._envelope_cache is not None:
            self._rebuild_polygon()

    def _draw_axes(self, painter, rect):
        painter.setPen(QColor(BORDER_MID))
        cy = int(rect.height() // 2)
        painter.drawLine(0, cy, int(rect.width()), cy)

    def _draw_empty(self, painter, rect):
        painter.setPen(QColor(BORDER_MID))
        font = QFont("system-ui, sans-serif", 13)
        painter.setFont(font)
        painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, t("波形图 — 打开音频文件查看", "Waveform — open an audio file to view"))
