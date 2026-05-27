"""WaveformWidget — displays a waveform visualization."""

import numpy as np

from PyQt6.QtWidgets import QWidget
from PyQt6.QtGui import QPainter, QColor, QBrush, QPolygonF, QFont, QPen
from lang import t, on_lang_change
from ui.styles import BORDER_MID
from PyQt6.QtCore import Qt, QRectF, QPointF, pyqtSignal

_ENVELOPE_SIZE = 4096


class WaveformWidget(QWidget):
    """Shows the audio waveform in a smooth, visually pleasing style."""

    seekRequested = pyqtSignal(float)  # seconds

    def __init__(self):
        super().__init__()
        self.setMinimumHeight(100)
        self.audio = None
        self.duration = 0.0
        self._envelope_cache: np.ndarray | None = None
        self._cached_polygon: QPolygonF | None = None
        self.playhead_pos: float = -1.0       # seconds, -1 = hidden; owned by main_window
        self._on_playhead_drag: callable | None = None
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

    def set_playhead(self, seconds: float) -> None:
        """Set playhead position — called by main_window only."""
        if seconds != self.playhead_pos:
            self.playhead_pos = seconds
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
        if self.playhead_pos >= 0 and self.duration > 0:
            self._draw_playhead(painter, rect)
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

        points_upper = []
        for i, val in enumerate(envelope):
            x = int((i / max(1, n - 1)) * rw) if n > 1 else int(rw // 2)
            y = int(rh // 2 - val * rh // 2)
            points_upper.append((x, y))

        points_lower = [(x, rh // 2 + (rh // 2 - y)) for x, y in points_upper]
        full_path = points_upper + list(reversed(points_lower))

        self._cached_polygon = QPolygonF(QPointF(x, y) for x, y in full_path) if full_path else None

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        if self._envelope_cache is not None:
            self._rebuild_polygon()

    def _draw_axes(self, painter, rect):
        painter.setPen(QColor(BORDER_MID))
        cy = int(rect.height() // 2)
        painter.drawLine(0, cy, int(rect.width()), cy)

    def _draw_playhead(self, painter, rect):
        x = int(self.playhead_pos / self.duration * rect.width())
        painter.setPen(QPen(QColor("#e8e6e2"), 1))
        painter.drawLine(x, 0, x, int(rect.height()))

    def _draw_empty(self, painter, rect):
        painter.setPen(QColor(BORDER_MID))
        font = QFont("system-ui, sans-serif", 13)
        painter.setFont(font)
        painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, t("波形图 — 打开音频文件查看", "Waveform — open an audio file to view"))

    # ── Mouse drag ─────────────────────────────────────────────────

    def mousePressEvent(self, event) -> None:
        if (event.button() == Qt.MouseButton.LeftButton
                and self.playhead_pos >= 0 and self.duration > 0):
            px = int(self.playhead_pos / self.duration * self.width())
            if abs(event.position().x() - px) <= 20:
                self._dragging = True
                self.setCursor(Qt.CursorShape.SizeHorCursor)
                return
        # Click anywhere on waveform seeks immediately
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
            self.update()
        else:
            super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event) -> None:
        if getattr(self, '_dragging', False) and event.button() == Qt.MouseButton.LeftButton:
            self._dragging = False
            self.setCursor(Qt.CursorShape.ArrowCursor)
            if self.duration > 0:
                secs = max(0, min(event.position().x() / self.width(), 1.0)) * self.duration
                self.seekRequested.emit(secs)
        else:
            super().mouseReleaseEvent(event)
