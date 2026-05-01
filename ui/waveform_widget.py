"""WaveformWidget — displays a waveform visualization."""

from PyQt6.QtWidgets import QWidget
from PyQt6.QtGui import QPainter, QColor, QBrush, QPolygonF, QFont
from lang import t
from ui.styles import BORDER_MID
from PyQt6.QtCore import Qt, QRectF, QPointF


class WaveformWidget(QWidget):
    """Shows the audio waveform in a smooth, visually pleasing style."""

    def __init__(self):
        super().__init__()
        self.setMinimumHeight(100)
        self.audio = None
        self.samples = None
        self.duration = 0.0

    def set_audio(self, data):
        self.audio = data
        self.samples = data.get('samples') or data.get('waveform', [])
        self.duration = data.get('duration', 0.0)

        # ── data validation ──
        if self.samples is not None and self.samples.size > 0:
            import numpy as np
            if isinstance(self.samples, list) and len(self.samples) > 0 and isinstance(self.samples[0], list):
                channel_samples = self.samples[0]
            else:
                channel_samples = self.samples
            cs_arr = np.asarray(channel_samples)
            print(f"[waveform] min={cs_arr.min():.6f}, max={cs_arr.max():.6f}, length={len(cs_arr)}")
        else:
            print("[waveform] samples is empty")

        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        rect = QRectF(0, 0, self.width(), self.height())

        if self.samples is None or (hasattr(self.samples, 'size') and self.samples.size == 0):
            self._draw_empty(painter, rect)
            painter.end()
            return

        # ── data validation before draw ──
        if isinstance(self.samples, list) and len(self.samples) > 0 and isinstance(self.samples[0], list):
            channel_samples = self.samples[0]
        else:
            channel_samples = self.samples
        cs_arr = __import__('numpy').asarray(channel_samples)
        print(f"[waveform/paintEvent] envelope input: min={cs_arr.min():.6f}, max={cs_arr.max():.6f}, len={len(cs_arr)}")

        self._draw_waveform(painter, rect)
        self._draw_axes(painter, rect)

        painter.end()

    def _draw_waveform(self, painter, rect):
        """Draw the waveform envelope as a single solid line."""
        samples = self.samples
        if samples is None or (hasattr(samples, 'size') and samples.size == 0):
            return

        # Use first channel only (already mixed to mono in AudioAnalyzer.waveform)
        if isinstance(samples, list) and len(samples) > 0:
            if isinstance(samples[0], list):
                channel_samples = samples[0]
            else:
                channel_samples = samples
        else:
            channel_samples = samples

        if channel_samples is None or (hasattr(channel_samples, 'size') and channel_samples.size == 0):
            return

        n = len(channel_samples)
        if n > rect.width() * 2:
            step = n / (rect.width() * 2)
            envelope = []
            for i in range(0, n, max(1, int(step))):
                chunk = channel_samples[i:min(i + int(step), n)]
                if len(chunk) > 0:
                    envelope.append(max(abs(s) for s in chunk))
        else:
            envelope = [abs(s) for s in channel_samples]

        if not envelope:
            return

        rw = int(rect.width())
        rh = int(rect.height())

        # Build upper half of the waveform envelope
        points_upper = []
        for i, val in enumerate(envelope):
            x = int((i / max(1, len(envelope) - 1)) * rw) if len(envelope) > 1 else int(rw // 2)
            y = int(rh // 2 - val * rh // 2)
            points_upper.append((x, y))

        # Build mirrored lower half
        points_lower = [(x, rh // 2 + (rh // 2 - y)) for x, y in points_upper]
        full_path = points_upper + list(reversed(points_lower))

        if full_path:
            # Single solid fill — no separate outline
            painter.setPen(Qt.PenStyle.NoPen)
            line_color = QColor("#c8c8c8")
            line_color.setAlpha(200)
            painter.setBrush(QBrush(line_color))
            polygon = QPolygonF(QPointF(x, y) for x, y in full_path)
            painter.drawPolygon(polygon)

    def _draw_axes(self, painter, rect):
        """Draw center line only."""
        painter.setPen(QColor(BORDER_MID))
        cy = int(rect.height() // 2)
        painter.drawLine(0, cy, int(rect.width()), cy)

    def _draw_empty(self, painter, rect):
        """Draw empty state."""
        painter.setPen(QColor(BORDER_MID))
        font = QFont("system-ui, sans-serif", 13)
        painter.setFont(font)
        painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, t("波形图 — 打开音频文件查看", "Waveform — open an audio file to view"))