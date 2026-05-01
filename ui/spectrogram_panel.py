"""Panel that lays out the spectrogram widget with placeholder zones.

Replaces the old single-widget approach in MainWindow so that each
component (Y axis, spectrogram, colorbar, X axis) has its own widget
in a grid layout. The spectrogram widget itself is not modified —
axis/colorbar drawing logic stays inside SpectrogramWidget for now
and will be migrated later.

Widget zones (4 regions):

    +---Y---+---spectrogram---+---colorbar---+
    |       |                 |               |
    |  Y-A  |    spectrogram  |  C-Bar      |
    |  xis  |    widget       |  Placeholder|
    |  Zone |                 |               |
    +-------+-----------------+---------------+
    |               X-Axis Zone              |
    +----------------------------------------+

"""

from __future__ import annotations

from PyQt5.QtWidgets import QGridLayout, QFrame, QWidget

from ui.spectrogram_widget import SpectrogramWidget


# ── Colors (matching existing palette) ──────────────────────────────

BG_DARK = "#2a2a2e"
BG_MID = "#222226"
BG_HIGHLIGHT = "#36363e"


# ── Placeholders ────────────────────────────────────────────────────

def _empty_widget(background: str = BG_MID, border_color: str = BG_HIGHLIGHT) -> QFrame:
    """Create a simple QFrame with a visible background color.

    Used as placeholder widgets for axis / colorbar zones.
    """
    frame = QFrame()
    frame.setStyleSheet(
        f"background-color: {background}; border: 1px solid {border_color};"
    )
    frame.setFrameShape(QFrame.Shape.StyledPanel)
    return frame


# ── Panel ───────────────────────────────────────────────────────────

class SpectrogramPanel(QWidget):
    """Top-level panel that arranges the spectrogram widget plus
    placeholder zones for Y-axis, colorbar, and X-axis.

    Public widgets (for later wiring up real axes / colorbars):
        - y_axis_placeholder (QFrame)
        - spectrogram_widget (SpectrogramWidget)
        - colorbar_placeholder (QFrame)
        - x_axis_placeholder (QFrame)

    Layout:
        Row 0: Y-Axis | Spectrogram | Colorbar
        Row 1: ───────────── X-Axis ─────────────
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)

        # ── Create children ────────────────────────────────────

        self.spectrogram_widget = SpectrogramWidget(self)

        self.y_axis_placeholder = _empty_widget(
            background=BG_DARK,
            border_color=BG_HIGHLIGHT,
        )
        self.y_axis_placeholder.setMinimumWidth(40)
        self.y_axis_placeholder.setMaximumWidth(60)

        self.colorbar_placeholder = _empty_widget(
            background=BG_DARK,
            border_color=BG_HIGHLIGHT,
        )
        self.colorbar_placeholder.setMinimumWidth(30)
        self.colorbar_placeholder.setMaximumWidth(50)

        self.x_axis_placeholder = _empty_widget(
            background=BG_DARK,
            border_color=BG_HIGHLIGHT,
        )
        self.x_axis_placeholder.setMinimumHeight(20)

        # ── Build grid ─────────────────────────────────────────

        self._layout = QGridLayout(self)
        self._layout.setContentsMargins(2, 2, 2, 2)
        self._layout.setSpacing(1)

        # Row 0: Y-axis | spectrogram | colorbar
        self._layout.addWidget(self.y_axis_placeholder, 0, 0)
        self._layout.addWidget(self.spectrogram_widget, 0, 1, 1, 1)
        self._layout.addWidget(self.colorbar_placeholder, 0, 2)

        # Row 1: X-axis (spans all columns)
        self._layout.addWidget(self.x_axis_placeholder, 1, 0, 1, 3)

        # Column stretch: spectrogram gets all remaining space
        self._layout.setColumnStretch(0, 0)   # Y-axis: fixed
        self._layout.setColumnStretch(1, 1)   # spectrogram: flexible
        self._layout.setColumnStretch(2, 0)   # colorbar: fixed
