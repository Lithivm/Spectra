"""Floating progress bar overlaid on SpectrogramGLWidget."""

from __future__ import annotations

import math
from typing import Optional

from PyQt6.QtCore import Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QColor, QPaintEvent, QPainter
from PyQt6.QtWidgets import QProgressBar, QWidget


class ProgressBar(QWidget):
    """Thin overlay bar for spectrogram work, parented to SpectrogramGLWidget."""

    progress_changed = pyqtSignal(float)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._value = 0.0
        self._visible = False
        self._anim_timer = QTimer(self)
        self._anim_timer.setSingleShot(True)
        self._anim_timer.timeout.connect(self._fade_out)
        self._fade_value = 0.0  # 0.0 → 1.0 opacity
        self._target_value = 0.0
        self._fade_started = False

        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)

        # Inferno-yellow: matches the colormap's warm tone
        self._fill_color = QColor("#ff9933")
        self._bg_color = QColor("#222222")

    def set_progress(self, fraction: float) -> None:
        """Set progress 0.0–1.0. Triggers fade-out timer on 1.0."""
        self._value = max(0.0, min(1.0, fraction))
        self._target_value = self._value
        self._visible = True
        self._fade_started = False
        self._fade_value = 0.0
        self.update()

        if self._value >= 1.0:
            self._anim_timer.start(250)  # wait a beat then fade
        else:
            self._anim_timer.stop()

    def _fade_out(self) -> None:
        """Crossfade to invisible over ~300 ms."""
        if self._fade_started:
            self._fade_value += 0.2
            if self._fade_value >= 1.0:
                self._fade_value = 1.0
                self._visible = False
                self.update()
                self._anim_timer.stop()
            else:
                self._anim_timer.start(50)
        else:
            self._fade_started = True
            self._anim_timer.start(50)

    # ── painting ────────────────────────────────────────────────────

    def _paint(self, painter: QPainter) -> None:
        w = self.width()
        h = self.height()

        if not self._visible or self._fade_value >= 1.0:
            return

        opacity = max(0.0, 1.0 - self._fade_value)
        painter.setOpacity(opacity)

        radius = h / 2
        bar_w = int(w * 0.5)
        fill_w = int(bar_w * self._value)

        # background
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setPen(Qt.PenStyle.NoPen)

        painter.setBrush(self._bg_color)
        painter.drawRoundedRect(
            (w - bar_w) // 2, (h - h) // 2, bar_w, h, radius, radius,
        )

        # fill
        painter.setBrush(self._fill_color)
        fill_x = (w - bar_w) // 2
        fill_y = (h - h) // 2
        painter.drawRoundedRect(fill_x, fill_y, fill_w, h, radius, radius)

        painter.setPen(Qt.PenStyle.NoPen)

    def paintEvent(self, ev: QPaintEvent) -> None:
        painter = QPainter(self)
        self._paint(painter)
        painter.end()

    def sizeHint(self):
        return self.size()  # we fill parent; no fixed hint needed
