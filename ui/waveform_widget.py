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
        self.samples = None
        self.duration = 0.0
        self._envelope_cache: np.ndarray | None = None
        on_lang_change(lambda _lang: self.update() if not self.audio else None)

    def _build_envelope(self, channel_samples: np.ndarray) -> np.ndarray:
        n = len(channel_samples)
        if n <= _ENVELOPE_SIZE:
            return np.abs(channel_samples)
        chunk_size = n // _ENVELOPE_SIZE
        trimmed = channel_samples[:chunk_size * _ENVELOPE_SIZE]
        return np.max(np.abs(trimmed.reshape(_ENVELOPE_SIZE, chunk_size)), axis=1)

    def set_audio(self, data):
        self.audio = data
        self.samples = data.get('samples') or data.get('waveform', [])
        self.duration = data.get('duration', 0.0)
        self._envelope_cache = None

        if self.samples is not None and hasattr(self.samples, 'size') and self.samples.size > 0:
            if isinstance(self.samples, list) and len(self.samples) > 0 and isinstance(self.samples[0], list):
                raw = np.asarray(self.samples[0])
            else:
                raw = np.asarray(self.samples)
            self._envelope_cache = self._build_envelope(raw)

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
        envelope = self._envelope_cache
        if envelope is None or len(envelope) == 0:
            return

        rw = int(rect.width())
        rh = int(rect.height())
        n = len(envelope)

        points_upper = []
        for i, val in enumerate(envelope):
            x = int((i / max(1, n - 1)) * rw) if n > 1 else int(rw // 2)
            y = int(rh // 2 - val * rh // 2)
            points_upper.append((x, y))

        points_lower = [(x, rh // 2 + (rh // 2 - y)) for x, y in points_upper]
        full_path = points_upper + list(reversed(points_lower))

        if full_path:
            painter.setPen(Qt.PenStyle.NoPen)
            line_color = QColor("#e8e6e2")
            line_color.setAlpha(220)
            painter.setBrush(QBrush(line_color))
            polygon = QPolygonF(QPointF(x, y) for x, y in full_path)
            painter.drawPolygon(polygon)

    def _draw_axes(self, painter, rect):
        painter.setPen(QColor(BORDER_MID))
        cy = int(rect.height() // 2)
        painter.drawLine(0, cy, int(rect.width()), cy)

    def _draw_empty(self, painter, rect):
        painter.setPen(QColor(BORDER_MID))
        font = QFont("system-ui, sans-serif", 13)
        painter.setFont(font)
        painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, t("波形图 — 打开音频文件查看", "Waveform — open an audio file to view"))