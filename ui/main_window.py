"""Main application window — modern AI product aesthetic."""

from __future__ import annotations

import itertools
import logging
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from typing import Any

from PyQt6.QtCore import QThread, Qt, pyqtSignal, QTimer, QRectF, QPointF, QSize
from PyQt6.QtGui import (
    QDragEnterEvent, QDropEvent, QColor, QPainter, QPixmap,
    QKeySequence, QShortcut, QGuiApplication,
)

from ui.anim import FloatAnim
from ui.icons import render_icon
from PyQt6.QtWidgets import (
    QFileDialog, QHBoxLayout, QVBoxLayout,
    QLabel, QMainWindow, QMessageBox, QStatusBar, QWidget,
    QFrame, QPushButton, QComboBox, QSizePolicy,
    QGraphicsDropShadowEffect, QGridLayout,
)


from ui.playback_engine import PlaybackEngine
from analyzer._state import _stft_cache, _stft_lock
from analyzer.palette import PALETTE
from analyzer import is_audio_file, SUPPORTED_EXTENSIONS

logger = logging.getLogger(__name__)
from ui.metadata_panel import MetadataPanel
from ui.spectrogram_widget import SpectrogramGLWidget, _YAxisWidget, _XAxisWidget, _ColorBarWidget
from ui.waveform_widget import WaveformWidget
from ui.styles import (
    BG_BASE, BG_SURFACE, BG_RAISED, BG_CANVAS,
    BORDER_SUB, BORDER_MID,
    ACCENT, ACCENT_ALT, ACCENT_RED,
    PRIMARY_HOVER_A, PRIMARY_HOVER_B,
    TEXT_PRI, TEXT_SEC, TEXT_DIM,
    FONT_FAMILY, FS_SM, FS_BODY, FS_MD, FS_LG, CORNER_SM, CORNER_LG, SIDE, CARD_INSET,
)

BTN_H = 32  # unified toolbar control height


def _icon_dpr() -> float:
    """Device pixel ratio for icon rendering.

    Screen dpr is available from app start; a widget's own dpr can still be
    1.0 before the window is first shown, so icons render at screen scale.
    """
    screen = QGuiApplication.primaryScreen()
    return screen.devicePixelRatio() if screen else 1.0
from lang import t, toggle_lang, on_lang_change

if TYPE_CHECKING:
    from analyzer.core import AudioAnalyzer

APP_STYLESHEET = f"""
* {{
    font-family: "{FONT_FAMILY}", "SF Pro Display", sans-serif;
}}
QMainWindow {{
    background-color: {BG_BASE};
}}
QWidget {{
    background-color: transparent;
    color: {TEXT_PRI};
    font-size: {FS_MD}px;
}}
QStatusBar {{
    background-color: {BG_SURFACE};
    color: {TEXT_DIM};
    border-top: 1px solid {BORDER_SUB};
    font-size: {FS_SM}px;
    font-family: "Consolas", monospace;
    padding: 0 12px;
}}
QComboBox {{
    background-color: {BG_BASE};
    border: 1px solid {BORDER_MID};
    border-radius: {CORNER_SM}px;
    color: {TEXT_PRI};
    padding: 4px 10px;
    font-size: {FS_BODY}px;
    min-width: 80px;
}}
QComboBox:hover {{
    border-color: {ACCENT};
}}
QComboBox::drop-down {{
    border: none;
    width: 20px;
}}
QComboBox QAbstractItemView {{
    background-color: {BG_RAISED};
    border: 1px solid {BORDER_MID};
    border-radius: {CORNER_SM}px;
    selection-background-color: {ACCENT};
    selection-color: {ACCENT_ALT};
    color: {TEXT_PRI};
    padding: 4px;
    outline: none;
}}
QPushButton {{
    background-color: {BG_BASE};
    border: 1px solid {BORDER_MID};
    border-radius: {CORNER_SM}px;
    color: {TEXT_SEC};
    padding: 5px 16px;
    font-size: {FS_BODY}px;
    font-weight: 500;
}}
QPushButton:hover {{
    border-color: {ACCENT};
    color: {TEXT_PRI};
    background-color: rgba(255, 255, 255, 0.06);
}}
QPushButton:pressed {{
    background-color: rgba(255, 255, 255, 0.12);
}}
QPushButton#primary {{
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
        stop:0 {ACCENT}, stop:1 {ACCENT_ALT});
    border: none;
    color: white;
    font-weight: 600;
}}
QPushButton#primary:hover {{
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
        stop:0 {PRIMARY_HOVER_A}, stop:1 {PRIMARY_HOVER_B});
    color: white;
}}
QScrollBar:vertical {{
    background: transparent;
    width: 6px;
    border: none;
}}
QScrollBar::handle:vertical {{
    background: {BORDER_MID};
    border-radius: 3px;
    min-height: 24px;
}}
QScrollBar::handle:vertical:hover {{
    background: {TEXT_DIM};
}}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
    height: 0;
}}
QLabel {{
    color: {TEXT_SEC};
    background: transparent;
}}
"""


def _shadow(radius: int = 24, opacity: int = 70) -> QGraphicsDropShadowEffect:
    fx = QGraphicsDropShadowEffect()
    fx.setBlurRadius(radius)
    fx.setOffset(0, 6)
    c = QColor("#000000")
    c.setAlpha(opacity)
    fx.setColor(c)
    return fx


_card_ids = itertools.count()

def _card(radius: int = CORNER_LG) -> QWidget:
    name = f"_card_{next(_card_ids)}"
    w = QWidget()
    w.setObjectName(name)
    w.setStyleSheet(f"""
        #{name} {{
            background-color: {BG_SURFACE};
            border: 1px solid {BORDER_SUB};
            border-radius: {radius}px;
        }}
    """)
    return w


def _toolbar_btn_qss() -> str:
    """Unified toolbar button look — BG_BASE fill, matching the packaged build."""
    return (
        f"QPushButton {{ color: {TEXT_PRI}; background: {BG_BASE};"
        f" border: 1px solid {BORDER_MID}; border-radius: {CORNER_SM}px; }}"
        f" QPushButton:hover {{ border-color: {ACCENT}; }}"
    )


class _RootWidget(QWidget):
    """Root container — provides native edge-resize for the frameless window."""

    _MARGIN = 8  # px from window edge that triggers resize

    def paintEvent(self, event) -> None:
        # Explicitly fill BG_BASE. A bare QWidget whose stylesheet sets only
        # `background:` (no border) is NOT auto-filled by Qt — the buffer stays
        # black. Cards render because their stylesheet carries a border.
        super().paintEvent(event)
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor(BG_BASE))

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            win = self.window()
            geo = win.frameGeometry()
            gx, gy = event.globalPosition().x(), event.globalPosition().y()
            m = self._MARGIN
            edges = []
            if gx <= geo.left() + m:
                edges.append(Qt.Edge.LeftEdge)
            if gx >= geo.right() - m:
                edges.append(Qt.Edge.RightEdge)
            if gy <= geo.top() + m:
                edges.append(Qt.Edge.TopEdge)
            if gy >= geo.bottom() - m:
                edges.append(Qt.Edge.BottomEdge)
            if edges:
                wh = win.windowHandle()
                if wh is not None:
                    wh.startSystemResize(edges)
                    return
        super().mousePressEvent(event)


class _TitleBarCard(QWidget):
    """Toolbar card that doubles as the frameless window's title bar.

    Drag on any non-interactive area to move (empty space, labels and separators
    are mouse-transparent so events fall through here); double-click to
    maximize/restore. Interactive children (buttons, combos) keep their own events.
    """

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            win = self.window()
            self._drag_offset = event.globalPosition().toPoint() - win.frameGeometry().topLeft()
            self._dragging = True

    def mouseMoveEvent(self, event) -> None:
        if getattr(self, "_dragging", False):
            self.window().move(event.globalPosition().toPoint() - self._drag_offset)

    def mouseReleaseEvent(self, event) -> None:
        self._dragging = False

    def mouseDoubleClickEvent(self, event) -> None:
        win = self.window()
        if win.isMaximized():
            win.showNormal()
        else:
            win.showMaximized()


class _PlaybackSlider(QWidget):
    """Custom playback progress bar with a draggable handle.

    Spans the full grid row; the track is drawn with SIDE margins so it
    aligns exactly with the spectrogram between Y-axis and colorbar.
    """
    sliderPressed = pyqtSignal()
    sliderReleased = pyqtSignal()
    valueChanged = pyqtSignal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._value = 0       # 0–1000
        self._maximum = 1000
        self._dragging = False
        self.setMouseTracking(True)
        self._hover = False
        self._hover_grow = 0.0  # 0..1 handle grow animation
        self._grow_anim: FloatAnim | None = None
        self._view_t0 = 0.0   # spectrogram zoom range (fraction of duration)
        self._view_t1 = 1.0

    def setRange(self, minimum: int, maximum: int) -> None:
        self._maximum = maximum

    def setValue(self, value: int) -> None:
        if self._value != value and not self._dragging:
            self._value = max(0, min(self._maximum, value))
            self.update()

    def set_view_range(self, t0: float, t1: float) -> None:
        """Update the spectrogram zoom range (0–1 fractions of duration)."""
        self._view_t0 = t0
        self._view_t1 = t1
        self.update()

    def value(self) -> int:
        return self._value

    def sizeHint(self):
        return QSize(100, 20)

    def enterEvent(self, event) -> None:
        self._hover = True
        self._sync_grow()

    def leaveEvent(self, event) -> None:
        self._hover = False
        self._sync_grow()

    def _sync_grow(self) -> None:
        """Handle hover/drag grow 0→1 (120ms)."""
        target = 1.0 if (self._hover or self._dragging) else 0.0

        def tick(v):
            self._hover_grow = v
            self.update()

        if self._grow_anim is None:
            self._grow_anim = FloatAnim(120, tick)
        self._grow_anim.start_from(self._hover_grow, target)

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self._dragging = True
            self._sync_grow()
            self._update_from_mouse(event.position().x())
            self.sliderPressed.emit()

    def mouseMoveEvent(self, event) -> None:
        if self._dragging:
            self._update_from_mouse(event.position().x())

    def mouseReleaseEvent(self, event) -> None:
        if self._dragging and event.button() == Qt.MouseButton.LeftButton:
            self._dragging = False
            self._sync_grow()
            self._update_from_mouse(event.position().x())
            self.sliderReleased.emit()

    def _update_from_mouse(self, x: float) -> None:
        pad = SIDE
        w = self.width() - 2 * pad  # track width
        if w <= 0:
            return
        ratio = max(0.0, min(1.0, (x - pad) / w))
        new_val = int(ratio * self._maximum)
        if new_val != self._value:
            self._value = new_val
            self.valueChanged.emit(new_val)
            self.update()

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        pad = SIDE
        w = self.width() - 2 * pad  # track width = spectrogram width
        h = self.height()
        cy = h // 2

        # Track
        track_h = 4
        track_rect = QRectF(pad, cy - track_h // 2, w, track_h)
        painter.setBrush(QColor(BORDER_MID))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawRoundedRect(track_rect, 2, 2)

        # Zoom indicator — highlight the visible spectrogram range
        if self._view_t0 > 0.0 or self._view_t1 < 1.0:
            zl = pad + int(w * self._view_t0)
            zr = pad + int(w * self._view_t1)
            zoom_rect = QRectF(zl, cy - track_h // 2, max(zr - zl, 1), track_h)
            painter.setBrush(QColor(BORDER_MID).lighter(140))
            painter.drawRoundedRect(zoom_rect, 2, 2)

        # Progress fill
        if self._maximum > 0:
            fill_w = int(w * self._value / self._maximum)
            fill_rect = QRectF(pad, cy - track_h // 2, fill_w, track_h)
            painter.setBrush(QColor(ACCENT))
            painter.drawRoundedRect(fill_rect, 2, 2)

        # Handle
        if self._maximum > 0:
            handle_x = pad + int(w * self._value / self._maximum)
            handle_r = 4 + 2 * self._hover_grow
            painter.setBrush(QColor(ACCENT))
            painter.drawEllipse(QPointF(handle_x, cy), handle_r, handle_r)

        painter.end()


def safe_slot(fn):
    """Decorator: catch exceptions in Qt slots so they don't abort the process."""
    from functools import wraps
    @wraps(fn)
    def _wrapper(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except Exception:
            logger.exception("Unhandled in slot %s", fn.__qualname__)
    return _wrapper


class _SpectrumWorker(QThread):
    stream_init  = pyqtSignal(object, int, float)         # freqs, total_cols, duration
    stream_block = pyqtSignal(int, object)                 # c0, block_db
    stream_done  = pyqtSignal(object, object, object)      # freqs, times, full_db
    finished     = pyqtSignal(object, object, object, str) # legacy path
    fading       = pyqtSignal()

    def __init__(self, analyzer: AudioAnalyzer, n_fft: int = 2048,
                 mode: str = "multi") -> None:
        super().__init__()
        self._analyzer = analyzer
        self._n_fft = n_fft
        self._mode = mode

    def run(self) -> None:
        try:
            self._run_impl()
        except Exception:
            logger.exception("SpectrumWorker failed")
            self.fading.emit()

    def _run_impl(self) -> None:
        t0 = time.perf_counter()
        if self._mode == "standard":
            # ── Cache fast-path ──
            fp = str(self._analyzer.filepath) if self._analyzer.filepath else ""
            cache_key = (fp, "standard", self._n_fft)
            with _stft_lock:
                cached = _stft_cache.get(cache_key)
            if cached is not None:
                freqs, times, db = cached
                self.finished.emit(freqs, times, db, "standard")
                self.fading.emit()
                return
            # ── Streaming path ──
            def _init(freqs, total_cols, hop):
                self._freqs = freqs
                self._hop = hop
                self.stream_init.emit(freqs, total_cols, self._analyzer.duration)
            def _block(c0, blk):
                self.stream_block.emit(c0, blk)
            result = self._analyzer.spectrogram_db_streaming(
                n_fft=self._n_fft, block_cols=64,
                on_init=_init, on_block=_block,
                cancel_check=self.isInterruptionRequested)
            if result is not None:
                freqs, times, db = result
                self.stream_done.emit(freqs, times, db)
                logger.debug("STFT streaming (n_fft=%s, mode=%s): %.2fs",
                             self._n_fft, self._mode, time.perf_counter() - t0)
            elif not self.isInterruptionRequested():
                # File too short for streaming — fall back to non-streaming
                freqs, times, db = self._analyzer.spectrogram_db(
                    n_fft=self._n_fft, mode="standard")
                logger.debug("STFT fallback (n_fft=%s, mode=%s): %.2fs",
                             self._n_fft, "standard", time.perf_counter() - t0)
                self.finished.emit(freqs, times, db, "standard")
            # else: cancelled — emit nothing
        else:
            freqs, times, db = self._analyzer.spectrogram_db(
                n_fft=self._n_fft, mode=self._mode)
            logger.debug("STFT (n_fft=%s, mode=%s): %.2fs",
                         self._n_fft, self._mode, time.perf_counter() - t0)
            self.finished.emit(freqs, times, db, self._mode)
        self.fading.emit()


class _LoadWorker(QThread):
    """Load audio file + metadata in background — keeps UI responsive."""
    loaded = pyqtSignal()       # success — UI reads self.analyzer
    error  = pyqtSignal(str)    # failure

    def __init__(self, path: Path) -> None:
        super().__init__()
        self._path = path
        self.analyzer: AudioAnalyzer | None = None

    def run(self) -> None:
        try:
            from analyzer.core import AudioAnalyzer
            self.analyzer = AudioAnalyzer(self._path)
            self.loaded.emit()
        except Exception as e:
            self.error.emit(str(e))


class _PreloadWorker(QThread):
    """Background thread to warm up heavy libraries (librosa, pyfftw, scipy).
    Runs once at startup so the first file-open is instant."""
    finished = pyqtSignal()

    def run(self) -> None:
        try:
            from analyzer.core import _ensure_librosa
            _ensure_librosa()
            # Also warm up scipy (used by quality analysis)
            import scipy.signal  # noqa: F401
            import scipy.ndimage  # noqa: F401
        except Exception:
            pass
        self.finished.emit()


class _QualityWorker(QThread):
    finished = pyqtSignal(object)
    progress = pyqtSignal(int)

    def __init__(self, analyzer: AudioAnalyzer) -> None:
        super().__init__()
        self._analyzer = analyzer

    def run(self) -> None:
        self.progress.emit(33)  # audio decoded
        try:
            result = self._analyzer.analyze_quality(
                cancel_check=self.isInterruptionRequested,
            )
        except Exception:
            logger.exception("QualityWorker failed")
            result = None
        self.progress.emit(90)  # STFT done
        self.finished.emit(result)
        self.progress.emit(100)  # render done


class _BatchWorker(QThread):
    finished = pyqtSignal(object)           # list[dict] — flattened results
    progress = pyqtSignal(int, str, bool)    # n, filename, ok

    def __init__(self, files: list[Path]) -> None:
        super().__init__()
        self._files = files

    def run(self) -> None:
        from analyzer.core import AudioAnalyzer as AA
        from analyzer.metadata import get_metadata
        from analyzer.batch import flatten_analysis

        results = []
        for i, fp in enumerate(self._files):
            if self.isInterruptionRequested():
                break
            try:
                analyzer = AA(fp)
                qa = analyzer.analyze_quality()
                md = get_metadata(fp)
                row = flatten_analysis(md, qa, fp)
                results.append(row)
                self.progress.emit(i + 1, fp.name, True)
            except Exception as e:
                results.append({"filename": fp.name, "filepath": str(fp), "error": str(e)})
                self.progress.emit(i + 1, fp.name, False)
        self.finished.emit(results)


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setAcceptDrops(True)
        self._title = "Spectra"
        self._current_path: Path | None = None
        self._spectrum_worker: _SpectrumWorker | None = None
        self._quality_worker: _QualityWorker | None = None
        self._load_worker: _LoadWorker | None = None
        self._batch_worker: _BatchWorker | None = None
        self._batch_results: list[dict] = []
        self._wave: WaveformWidget | None = None
        self._spec: SpectrogramGLWidget | None = None
        self._y_axis: _YAxisWidget | None = None
        self._x_axis: _XAxisWidget | None = None
        self._colorbar: _ColorBarWidget | None = None
        self._meta: MetadataPanel | None = None
        self._analyzer: AudioAnalyzer | None = None
        self._current_palette = "spectra"
        self._fft_size = 8192
        self._mode = "standard"

        self._slider_dragging = False
        self._logo_ok = False  # brand logo loaded from assets/logo.png

        self._playback = PlaybackEngine(self)
        self._playback_timer = QTimer(self)
        self._playback_timer.setInterval(30)
        self._playback_timer.timeout.connect(self._on_playback_tick)
        self._playback_timer.start()

        self.setStyleSheet(APP_STYLESHEET)
        self._setup_ui()

        # Background preload: warm up heavy libraries while user sees the window
        self._preload_worker = _PreloadWorker()
        self._preload_worker.finished.connect(
            lambda: logger.debug("Preload complete"))
        self._preload_worker.start()

    def _setup_ui(self) -> None:
        self.setWindowTitle(self._title)
        self.resize(1440, 880)
        self.menuBar().setVisible(False)
        self._create_statusbar()

        self.setWindowFlag(Qt.WindowType.FramelessWindowHint)
        root = _RootWidget()
        root.setStyleSheet(f"background: {BG_BASE};")
        self.setCentralWidget(root)
        root_layout = QVBoxLayout(root)
        root_layout.setContentsMargins(12, 12, 12, 12)
        root_layout.setSpacing(10)

        # 顶部工具条 — 贯穿整个窗口宽度，右侧按钮落在窗口右边缘
        root_layout.addWidget(self._make_toolbar())

        # 下部：左侧卡片 + 右侧元数据面板
        bottom_layout = QHBoxLayout()
        bottom_layout.setSpacing(10)
        bottom_layout.setContentsMargins(0, 0, 0, 0)

        # 左侧
        left_layout = QVBoxLayout()
        left_layout.setSpacing(10)
        left_layout.setContentsMargins(0, 0, 0, 0)

        # 波形卡片 — left/right margins align with spectrogram (y-axis + colorbar)
        wave_card = _card()
        wave_card.setFixedHeight(130)
        wl = QVBoxLayout(wave_card)
        wl.setContentsMargins(SIDE + CARD_INSET, CARD_INSET, SIDE + CARD_INSET, CARD_INSET)
        self._wave = WaveformWidget()
        wl.addWidget(self._wave)
        left_layout.addWidget(wave_card)

        # 频谱卡片 — 内衬 well（_inner: bg=BG_CANVAS，视觉沉入）
        spec_card = _card()
        sl = QVBoxLayout(spec_card)
        sl.setContentsMargins(CARD_INSET, CARD_INSET, CARD_INSET, CARD_INSET)
        sl.setSpacing(0)

        # 内衬 well：_inner 承载 grid，bg=BG_CANVAS（视觉"沉入"卡片）
        _inner = QWidget()
        _inner.setObjectName("spec_well")
        _inner.setStyleSheet(
            f"background-color: {BG_CANVAS}; border-radius: {CORNER_LG - CARD_INSET}px;")

        # ---- Grid: filename | Y-axis | spectrogram | colorbar | X-axis ----
        _grid = QGridLayout()
        _grid.setContentsMargins(0, 0, 0, 0)
        _grid.setSpacing(0)

        _grid.setColumnMinimumWidth(0, SIDE)
        _grid.setColumnStretch(0, 0)
        _grid.setColumnStretch(1, 1)
        _grid.setColumnMinimumWidth(2, 36)
        _grid.setColumnStretch(2, 0)

        _grid.setRowMinimumHeight(0, SIDE)
        _grid.setRowStretch(0, 0)
        _grid.setRowStretch(1, 1)
        _grid.setRowMinimumHeight(2, 20)  # slider row
        _grid.setRowStretch(2, 0)
        _grid.setRowMinimumHeight(3, SIDE)
        _grid.setRowStretch(3, 0)

        # NOTE: SIDE is imported from ui.styles — single source of truth
        self._filename_widget = QLabel()
        self._filename_widget.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._filename_widget.setStyleSheet(
            f"color: {TEXT_DIM}; font-size: 11px; background: transparent;")
        _grid.addWidget(self._filename_widget, 0, 0, 1, 3)

        # Row 1
        self._y_axis = _YAxisWidget()
        self._y_axis.setFixedWidth(SIDE)
        _grid.addWidget(self._y_axis, 1, 0)

        self._spec = SpectrogramGLWidget()
        self._spec.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        _grid.addWidget(self._spec, 1, 1)

        # 空态提示（叠加在声谱图上，鼠标事件穿透；加载开始后隐藏）
        self._empty_hint = QLabel()
        self._empty_hint.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._empty_hint.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self._empty_hint.setStyleSheet(
            f"color: {TEXT_SEC}; font-family: '{FONT_FAMILY}'; font-size: {FS_LG}px;")
        self._empty_hint.setText(
            t("拖入音频文件，或按 Ctrl+O 打开",
              "Drop an audio file here, or press Ctrl+O"))
        _grid.addWidget(self._empty_hint, 1, 1)

        self._colorbar = _ColorBarWidget()
        self._colorbar.setFixedWidth(SIDE)
        _uc, _cp, _cl, _cs = self._spec.get_curve_params()
        self._colorbar.set_data(self._spec._lut_np, use_curve=_uc,
                                curve_power=_cp, curve_lo=_cl, curve_span=_cs)
        _grid.addWidget(self._colorbar, 1, 2)

        # Row 2: X-axis spans all 3 columns, data offset-aligned to spectrogram
        self._x_axis = _XAxisWidget()
        self._x_axis.setFixedHeight(SIDE)
        _grid.addWidget(self._x_axis, 3, 0, 1, 3)

        _inner.setLayout(_grid)
        sl.addWidget(_inner, 0)

        left_layout.addWidget(spec_card, stretch=1)

        bottom_layout.addLayout(left_layout, stretch=3)

        # 右侧元数据面板 — 从工具条下方开始，与左列顶部对齐
        self._meta = MetadataPanel()
        self._meta.setFixedWidth(310)
        bottom_layout.addWidget(self._meta)

        root_layout.addLayout(bottom_layout, stretch=1)

        self._spec.set_palette("spectra")

        # Floating cursor info label — child of spec_card, positioned at cursor x
        self._cursor_label = QLabel(spec_card)
        self._cursor_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._cursor_label.setStyleSheet(
            f"color: {TEXT_PRI}; font-size: {FS_BODY}px; font-family: 'Consolas', monospace;"
            f" background: transparent; border: 1px solid {BORDER_SUB};"
            f" border-radius: {CORNER_SM}px; padding: 2px 8px;"
        )
        self._cursor_label.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self._cursor_label.hide()

        # Playback slider — spans all columns; track inset by SIDE (see paintEvent)
        self._progress_slider = _PlaybackSlider()
        self._progress_slider.setFixedHeight(20)
        self._progress_slider.sliderPressed.connect(self._on_slider_pressed)
        self._progress_slider.sliderReleased.connect(self._on_slider_released)
        self._progress_slider.valueChanged.connect(self._on_slider_changed)
        _grid.addWidget(self._progress_slider, 2, 0, 1, 3)

        self._playback.state_changed.connect(self._on_playback_state)
        self._slider_dragging = False
        self._was_playing_before_drag = False

        # Cursor coordinate info — floating label above spectrogram
        self._spec.cursor_info.connect(self._on_cursor_info)
        self._spec.cursor_left.connect(self._on_cursor_left)
        self._spec.view_changed.connect(self._on_view_changed)

        on_lang_change(self._retranslate)

    def _load_brand_logo(self) -> None:
        """Replace brand text with the transparent S-mark (assets/logo_mark.png).

        Falls back to assets/logo.png, then plain "Spectra" text if the
        asset is missing/unreadable.
        DPR-aware: pixmap physical size = 44 * devicePixelRatio.
        """
        self._brand_label.setText("Spectra")
        pm = QPixmap()
        if hasattr(sys, "frozen"):
            base = Path(getattr(sys, "_MEIPASS", "."))
        else:
            base = Path(__file__).resolve().parent.parent
        for cand in (
            base / "assets" / "logo_mark.png",
            Path("assets") / "logo_mark.png",
            base / "assets" / "logo.png",
            Path("assets") / "logo.png",
        ):
            if not cand.exists():
                continue
            if QPixmap(str(cand)).isNull():
                break
            pm = QPixmap(str(cand))
            break
        if pm.isNull():
            self._logo_ok = False
            return
        dpr = max(self.devicePixelRatio() or 1.0, _icon_dpr())
        target = int(round(44 * dpr))
        # Scale in plain device pixels first; stamping dpr beforehand makes
        # scaled() work in logical coordinates and the result comes out wrong.
        pm = pm.scaled(target, target, Qt.AspectRatioMode.KeepAspectRatio,
                       Qt.TransformationMode.SmoothTransformation)
        if dpr != 1.0:
            pm.setDevicePixelRatio(dpr)
        self._brand_label.setPixmap(pm)
        self._logo_ok = True

    def _make_toolbar(self) -> QWidget:
        card = _TitleBarCard()
        name = f"_toolbar_{next(_card_ids)}"
        card.setObjectName(name)
        card.setStyleSheet(f"""
            #{name} {{
                background-color: {BG_SURFACE};
                border: 1px solid {BORDER_SUB};
                border-radius: {CORNER_LG}px;
            }}
        """)
        card.setFixedHeight(64)
        layout = QHBoxLayout(card)
        layout.setContentsMargins(10, 0, 16, 0)
        layout.setSpacing(12)

        self._brand_label = QLabel("Spectra")
        self._brand_label.setStyleSheet(f"""
            color: {TEXT_PRI};
            font-size: 15px;
            font-weight: 700;
            background: transparent;
            border: none;
        """)
        # 品牌区作为标题栏拖拽区（鼠标事件穿透到 _TitleBarCard）
        self._brand_label.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        layout.addWidget(self._brand_label)

        # Logo — 用 .exe 同款图标 assets/logo.png 替换品牌文字；失败保留文字
        self._load_brand_logo()

        sep0 = QFrame()
        sep0.setFrameShape(QFrame.Shape.VLine)
        sep0.setStyleSheet(f"background: {BORDER_SUB}; border: none; max-width: 1px;")
        sep0.setFixedHeight(20)
        layout.addWidget(sep0)

        self._open_btn = QPushButton(t("打开文件", "Open File"))
        self._open_btn.setFixedHeight(BTN_H)
        self._open_btn.setFixedWidth(100)
        self._open_btn.setStyleSheet(_toolbar_btn_qss())
        self._open_btn.clicked.connect(self._on_open_file)
        layout.addWidget(self._open_btn)

        # 播放标签
        self._play_label = QLabel(t("播放", "Play"))
        self._play_label.setStyleSheet(f"color: {TEXT_DIM}; font-size: 11px; background: transparent; border: none;")
        layout.addWidget(self._play_label)

        # Play / Pause
        self._play_btn = QPushButton()
        self._play_btn.setIcon(render_icon("play", 18, dpr=_icon_dpr()))
        self._play_btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self._play_btn.setFixedSize(36, BTN_H)
        self._play_btn.setStyleSheet(_toolbar_btn_qss())
        self._play_btn.clicked.connect(self._on_playback_toggle)
        layout.addWidget(self._play_btn)

        layout.addStretch()

        self._pal_label = QLabel(t("调色板", "Palette"))
        self._pal_label.setStyleSheet(f"color: {TEXT_DIM}; font-size: 11px; background: transparent; border: none;")
        layout.addWidget(self._pal_label)
        self._palette_combo = QComboBox()
        self._palette_combo.addItems(list(PALETTE.keys()))
        self._palette_combo.setCurrentText("spectra")
        self._palette_combo.setFixedHeight(BTN_H)
        self._palette_combo.currentTextChanged.connect(self._on_palette_changed)
        layout.addWidget(self._palette_combo)

        sep1 = QFrame()
        sep1.setFrameShape(QFrame.Shape.VLine)
        sep1.setStyleSheet(f"background: {BORDER_SUB}; border: none; max-width: 1px;")
        sep1.setFixedHeight(20)
        layout.addWidget(sep1)

        self._mode_label = QLabel(t("模式", "Mode"))
        self._mode_label.setStyleSheet(f"color: {TEXT_DIM}; font-size: 11px; background: transparent; border: none;")
        layout.addWidget(self._mode_label)
        self._mode_combo = QComboBox()
        self._mode_combo.addItems(["standard", "multi", "reassign"])
        self._mode_combo.setCurrentText("standard")
        self._mode_combo.setFixedHeight(BTN_H)
        self._mode_combo.setFixedWidth(88)
        self._mode_combo.currentTextChanged.connect(self._on_mode_changed)
        layout.addWidget(self._mode_combo)

        sep2 = QFrame()
        sep2.setFrameShape(QFrame.Shape.VLine)
        sep2.setStyleSheet(f"background: {BORDER_SUB}; border: none; max-width: 1px;")
        sep2.setFixedHeight(20)
        layout.addWidget(sep2)

        self._yscale_label = QLabel(t("刻度", "Scale"))
        self._yscale_label.setStyleSheet(f"color: {TEXT_DIM}; font-size: 11px; background: transparent; border: none;")
        layout.addWidget(self._yscale_label)
        self._yscale_combo = QComboBox()
        self._yscale_combo.addItems(["log", "mel", "bark", "linear"])
        self._yscale_combo.setCurrentText("linear")
        self._yscale_combo.setFixedHeight(BTN_H)
        self._yscale_combo.currentTextChanged.connect(self._on_yscale_changed)
        layout.addWidget(self._yscale_combo)

        sep3 = QFrame()
        sep3.setFrameShape(QFrame.Shape.VLine)
        sep3.setStyleSheet(f"background: {BORDER_SUB}; border: none; max-width: 1px;")
        sep3.setFixedHeight(20)
        layout.addWidget(sep3)

        self._fft_label = QLabel("FFT")
        self._fft_label.setStyleSheet(f"color: {TEXT_DIM}; font-size: 11px; background: transparent; border: none;")
        layout.addWidget(self._fft_label)
        self._fft_combo = QComboBox()
        self._fft_combo.addItems(["256", "512", "1024", "2048", "4096", "8192", "16384"])
        self._fft_combo.setCurrentText("8192")
        self._fft_combo.setFixedHeight(BTN_H)
        self._fft_combo.setFixedWidth(72)
        self._fft_combo.currentTextChanged.connect(self._on_fft_size_changed)
        layout.addWidget(self._fft_combo)

        sep4 = QFrame()
        sep4.setFrameShape(QFrame.Shape.VLine)
        sep4.setStyleSheet(f"background: {BORDER_SUB}; border: none; max-width: 1px;")
        sep4.setFixedHeight(20)
        layout.addWidget(sep4)

        self._save_btn = QPushButton(t("保存PNG", "Save PNG"))
        self._save_btn.setFixedHeight(BTN_H)
        self._save_btn.setStyleSheet(_toolbar_btn_qss())
        self._save_btn.clicked.connect(self._on_save_screenshot)
        layout.addWidget(self._save_btn)

        # language toggle
        self._lang_btn = QPushButton("中/EN")
        self._lang_btn.setFixedHeight(BTN_H)
        self._lang_btn.setMinimumWidth(64)
        self._lang_btn.setStyleSheet(_toolbar_btn_qss())
        self._lang_btn.clicked.connect(self._on_toggle_lang)
        layout.addWidget(self._lang_btn)

        # 窗口控制按钮（frameless 标题栏）— SVG 图标，不依赖字体渲染
        win_btn_qss = (
            f"QPushButton {{ background: {BG_BASE}; border: 1px solid {BORDER_MID};"
            f" border-radius: {CORNER_SM}px; }}"
            f" QPushButton:hover {{ background: {BORDER_SUB}; }}"
        )
        for name, slot in [("minimize", self._on_minimize), ("maximize", self._on_maximize_toggle)]:
            b = QPushButton()
            b.setIcon(render_icon(name, 14, dpr=_icon_dpr()))
            b.setIconSize(QSize(14, 14))
            b.setFixedSize(30, BTN_H)
            b.setCursor(Qt.CursorShape.PointingHandCursor)
            b.setStyleSheet(win_btn_qss)
            b.clicked.connect(slot)
            layout.addWidget(b)

        close_btn = QPushButton()
        close_btn.setIcon(render_icon("close", 13, dpr=_icon_dpr()))
        close_btn.setIconSize(QSize(13, 13))
        close_btn.setFixedSize(30, BTN_H)
        close_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        close_btn.setStyleSheet(
            f"QPushButton {{ background: {BG_BASE}; border: 1px solid {BORDER_MID};"
            f" border-radius: {CORNER_SM}px; }}"
            f" QPushButton:hover {{ background: {ACCENT_RED}; }}"
        )
        close_btn.clicked.connect(self.close)
        layout.addWidget(close_btn)

        # 装饰性控件（标签/分隔线）鼠标穿透 → 标题栏拖拽区覆盖整个非交互区域
        for w in (self._play_label, self._pal_label, self._mode_label,
                  self._yscale_label, self._fft_label,
                  sep0, sep1, sep2, sep3, sep4):
            w.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)

        self._setup_shortcuts_and_tooltips()

        return card

    def _setup_shortcuts_and_tooltips(self) -> None:
        """Keyboard shortcuts + tooltips (frameless window keeps native a11y)."""
        QShortcut(QKeySequence("Ctrl+O"), self).activated.connect(self._on_open_file)
        QShortcut(QKeySequence(Qt.Key.Key_Space), self).activated.connect(self._on_space_toggle)
        QShortcut(QKeySequence("Ctrl+S"), self).activated.connect(self._on_save_screenshot)
        self._update_tooltips()

    def _update_tooltips(self) -> None:
        self._open_btn.setToolTip(t("打开文件 (Ctrl+O)", "Open File (Ctrl+O)"))
        self._play_btn.setToolTip(t("播放/暂停 (空格)", "Play / Pause (Space)"))
        self._save_btn.setToolTip(t("保存PNG (Ctrl+S)", "Save PNG (Ctrl+S)"))

    def _on_space_toggle(self) -> None:
        """Space toggles playback unless an interactive control has focus."""
        from PyQt6.QtWidgets import QAbstractButton, QComboBox
        f = self.focusWidget()
        if isinstance(f, (QAbstractButton, QComboBox)):
            return  # let the focused control handle Space itself
        self._on_playback_toggle()

    def _create_statusbar(self) -> None:
        sb = QStatusBar()
        sb.setFixedHeight(24)
        self._zoom_hint = QLabel(
            t("滚轮: 缩放时间  Shift+滚轮: 缩放频率  双击: 重置",
              "Wheel: zoom time  Shift+Wheel: zoom freq  Dbl-click: reset"))
        self._zoom_hint.setStyleSheet(
            f"color: {TEXT_DIM}; font-size: {FS_SM}px; font-family: 'Consolas'; background: transparent;")
        sb.addWidget(self._zoom_hint)
        self._status_label = QLabel(t("就绪", "Ready"))
        self._status_label.setStyleSheet(
            f"color: {TEXT_DIM}; font-size: {FS_SM}px; font-family: 'Consolas'; background: transparent;")
        sb.addPermanentWidget(self._status_label)
        self.setStatusBar(sb)

    def _on_palette_changed(self, name: str) -> None:
        self._current_palette = name
        if self._spec:
            self._spec.set_palette(name)
            if self._colorbar:
                _uc, _cp, _cl, _cs = self._spec.get_curve_params()
                self._colorbar.set_data(self._spec._lut_np, use_curve=_uc,
                                        curve_power=_cp, curve_lo=_cl, curve_span=_cs)

    def _on_mode_changed(self, mode: str) -> None:
        self._mode = mode
        self._fft_combo.setEnabled(mode != "multi")
        if self._current_path:
            self._reload_spectrum()

    def _on_yscale_changed(self, scale: str) -> None:
        if self._spec:
            self._spec._on_yscale_changed(scale)
            if self._y_axis and self._spec.frequencies is not None:
                self._y_axis.set_data(self._spec.frequencies, scale,
                                      self._spec._view_f0, self._spec._view_f1)

    def _on_fft_size_changed(self, size_str: str) -> None:
        self._fft_size = int(size_str)
        if self._current_path:
            self._reload_spectrum()

    def _reload_spectrum(self) -> None:
        """Re-run only the spectrogram (streaming or not); skip quality & file I/O."""
        self._cancel_spectrum()
        if self._mode == "standard":
            # Clear cache so the scroll-fill animation always plays
            fp = str(self._analyzer.filepath) if self._analyzer.filepath else ""
            cache_key = (fp, "standard", self._fft_size)
            with _stft_lock:
                _stft_cache.pop(cache_key, None)
            self._spec.hide_progress()
            self._spectrum_worker = _SpectrumWorker(self._analyzer, self._fft_size, self._mode)
            self._spectrum_worker.stream_init.connect(self._on_stream_init)
            self._spectrum_worker.stream_block.connect(self._on_stream_block)
            self._spectrum_worker.stream_done.connect(self._on_stream_done)
            self._spectrum_worker.finished.connect(self._on_spectrum_done)
        else:
            self._spec.show_progress()
            self._spectrum_worker = _SpectrumWorker(self._analyzer, self._fft_size, self._mode)
            self._spectrum_worker.finished.connect(self._on_spectrum_done)
            self._spectrum_worker.fading.connect(lambda: self._spec.hide_progress())
        self._spectrum_worker.start()

    def _on_open_file(self) -> None:
        formats = sorted(SUPPORTED_EXTENSIONS)
        path_list, _ = QFileDialog.getOpenFileNames(
            self, t("选择音频文件（可多选）", "Select Audio File(s)"), str(Path.home()),
            "Audio Files (" + " ".join(f"*{ext}" for ext in formats) + ")"
        )
        if len(path_list) == 1:
            self._load_file(Path(path_list[0]))
        elif len(path_list) > 1:
            self._start_batch_analysis([Path(p) for p in path_list])

    @safe_slot
    def _on_spectrum_done(self, freqs: Any, times: Any, db: Any, mode: str = "multi") -> None:
        self._spec.set_audio({
            'spectrogram': db,
            'fft_freqs': freqs,
            'times': times,
            'start_time': 0.0,
            'duration': self._analyzer.duration,
            'sample_rate': self._analyzer.sample_rate,
            'mode': mode,
        })
        self._spec.hide_progress()
        self._y_axis.set_data(freqs, self._spec._yscale_mode,
                              self._spec._view_f0, self._spec._view_f1)
        self._x_axis.set_data(self._analyzer.duration,
                              self._spec._view_t0, self._spec._view_t1)
        _uc, _cp, _cl, _cs = self._spec.get_curve_params()
        self._colorbar.set_data(self._spec._lut_np, use_curve=_uc,
                                curve_power=_cp, curve_lo=_cl, curve_span=_cs)

    @safe_slot
    def _on_stream_init(self, freqs: np.ndarray, total_cols: int, duration: float) -> None:
        n_freqs = freqs.shape[0] if freqs is not None else 1025
        self._spec.begin_stream(n_freqs, total_cols, freqs, duration)
        self._y_axis.set_data(freqs, self._spec._yscale_mode,
                              self._spec._view_f0, self._spec._view_f1)
        self._x_axis.set_data(duration,
                              self._spec._view_t0, self._spec._view_t1)
        _uc, _cp, _cl, _cs = self._spec.get_curve_params()
        self._colorbar.set_data(self._spec._lut_np, use_curve=_uc,
                                curve_power=_cp, curve_lo=_cl, curve_span=_cs)

    @safe_slot
    def _on_stream_block(self, c0: int, blk: np.ndarray) -> None:
        self._spec.push_block(c0, blk)

    @safe_slot
    def _on_stream_done(self, freqs: np.ndarray, times: np.ndarray, full_db: np.ndarray) -> None:
        self._spec.end_stream(full_db)

    def _load_file(self, path: Path) -> None:
        """Kick off background audio load — non-blocking."""
        self._cancel_load()
        self._cancel_spectrum()
        self._cancel_quality()

        self._current_path = path
        self._spec.show_progress()
        self._empty_hint.hide()

        self._load_worker = _LoadWorker(path)
        self._load_worker.loaded.connect(self._on_load_done)
        self._load_worker.error.connect(self._on_load_error)
        self._load_worker.start()

    @safe_slot
    def _on_load_done(self) -> None:
        """Audio loaded — wire up quality, waveform, and spectrum workers."""
        analyzer = self._load_worker.analyzer
        old_load = self._load_worker
        self._load_worker = None
        self._hold_ref(old_load)
        self._analyzer = analyzer

        t0 = time.perf_counter()
        self._meta.load_metadata(analyzer)
        logger.debug("元数据: %.2fs", time.perf_counter() - t0)

        self._quality_worker = _QualityWorker(analyzer)
        self._quality_worker.finished.connect(self._on_quality_done)
        self._quality_worker.start()

        t0 = time.perf_counter()
        self._wave.set_audio(
            analyzer.waveform,
            analyzer.sample_rate,
            analyzer.duration,
        )
        logger.debug("波形渲染: %.2fs", time.perf_counter() - t0)

        self._playback.load(analyzer.waveform, analyzer.sample_rate)
        self._progress_slider.setRange(0, 1000)
        self._progress_slider.setValue(0)

        path = analyzer.filepath
        self.setWindowTitle(f"Spectra  —  {path.name}")
        self._status_label.setText(
            f"{path.name}  ·  {analyzer.sample_rate/1000:.1f} kHz  ·  "
            f"{int(analyzer.duration)//60}m{int(analyzer.duration)%60}s"
        )
        self._filename_widget.setText(path.name)

        self._spectrum_worker = _SpectrumWorker(analyzer, self._fft_size, self._mode)
        if self._mode == "standard":
            self._spec.hide_progress()
            self._spectrum_worker.stream_init.connect(self._on_stream_init)
            self._spectrum_worker.stream_block.connect(self._on_stream_block)
            self._spectrum_worker.stream_done.connect(self._on_stream_done)
            self._spectrum_worker.finished.connect(self._on_spectrum_done)
        else:
            self._spectrum_worker.finished.connect(self._on_spectrum_done)
            self._spectrum_worker.fading.connect(lambda: self._spec.hide_progress())
        self._spectrum_worker.start()

    @safe_slot
    def _on_load_error(self, msg: str) -> None:
        old_load = self._load_worker
        self._load_worker = None
        self._hold_ref(old_load)
        self._spec.hide_progress()
        path = self._current_path
        QMessageBox.critical(self, t("错误", "Error"),
                             t(f"无法加载:\n{path.name if path else msg}",
                               f"Cannot load:\n{path.name if path else msg}"))

    def _cancel_spectrum(self) -> None:
        if self._spectrum_worker is not None:
            old = self._spectrum_worker
            self._spectrum_worker = None
            old.requestInterruption()
            for sig in (old.stream_init, old.stream_block, old.stream_done,
                         old.finished, old.fading):
                try:
                    sig.disconnect()
                except TypeError:
                    pass  # signal was never connected
            self._hold_ref(old)

    def _cancel_quality(self) -> None:
        if self._quality_worker is not None:
            old = self._quality_worker
            self._quality_worker = None
            old.requestInterruption()
            for sig in (old.finished, old.progress):
                try:
                    sig.disconnect()
                except TypeError:
                    pass  # signal was never connected
            self._hold_ref(old)

    def _cancel_load(self) -> None:
        if self._load_worker is not None:
            old = self._load_worker
            self._load_worker = None
            old.requestInterruption()
            for sig in (old.loaded, old.error):
                try:
                    sig.disconnect()
                except TypeError:
                    pass  # signal was never connected
            self._hold_ref(old)

    def _hold_ref(self, worker: QThread) -> None:
        """Keep *worker* Python wrapper alive so GC can't __del__+wait()
        a live C++ thread. Automatically cleaned up on finished."""
        if not hasattr(self, '_zombies'):
            self._zombies: list[QThread] = []
        if worker.isRunning():
            self._zombies.append(worker)
            worker.finished.connect(lambda w=worker: self._remove_zombie(w))

    def _remove_zombie(self, worker: QThread) -> None:
        try:
            self._zombies.remove(worker)
        except (RuntimeError, ValueError):
            pass

    def _on_playback_tick(self) -> None:
        """Update slider position from DAC."""
        if not self._playback.is_playing or self._slider_dragging:
            return
        pos = self._playback.get_position()
        dur = self._analyzer.duration if self._analyzer else 0
        if dur > 0:
            self._progress_slider.setValue(int(pos / dur * 1000))
            # 同步声谱光标到播放位置（鼠标不在声谱区时）
            self._spec.update_playback_cursor(pos)

    def _on_playback_toggle(self) -> None:
        self._playback.toggle()

    def _on_playback_state(self, state: str) -> None:
        actual = self._playback.state
        if actual == "playing":
            self._play_btn.setIcon(render_icon("pause", 18, dpr=_icon_dpr()))
        else:
            self._play_btn.setIcon(render_icon("play", 18, dpr=_icon_dpr()))
            if actual == "stopped":
                self._progress_slider.setValue(0)
            # 停止/暂停时，若鼠标不在声谱区，清除光标
            if self._spec.stop_playback_cursor():
                self._cursor_label.hide()
                self._filename_widget.show()

    def _on_slider_pressed(self) -> None:
        self._slider_dragging = True
        self._was_playing_before_drag = self._playback.is_playing
        if self._was_playing_before_drag:
            self._playback.pause()

    def _on_slider_released(self) -> None:
        self._slider_dragging = False
        dur = self._analyzer.duration if self._analyzer else 0
        if dur > 0:
            secs = self._progress_slider.value() / 1000.0 * dur
            if self._was_playing_before_drag:
                self._playback.seek(secs)
                self._playback.play()
            else:
                self._playback.track_position(secs)

    def _on_slider_changed(self, value: int) -> None:
        if self._slider_dragging:
            dur = self._analyzer.duration if self._analyzer else 0
            if dur > 0:
                secs = self._progress_slider.value() / 1000.0 * dur
                self._playback.track_position(secs)
                self._spec.update_playback_cursor(secs)

    @safe_slot
    def _on_view_changed(self) -> None:
        """Sync axis widgets and slider with spectrogram view window."""
        spec = self._spec
        if self._spec.frequencies is not None:
            self._y_axis.set_data(spec.frequencies, spec._yscale_mode,
                                  spec._view_f0, spec._view_f1)
        if self._analyzer:
            self._x_axis.set_data(self._analyzer.duration,
                                  spec._view_t0, spec._view_t1)
        self._progress_slider.set_view_range(spec._view_t0, spec._view_t1)

    @safe_slot
    def _on_cursor_info(self, time_s: float, freq: float, db: float, px: int) -> None:
        """Show cursor coordinates in floating label above spectrogram."""
        if freq >= 1000:
            freq_str = f"{freq / 1000:.2f} kHz"
        else:
            freq_str = f"{freq:.1f} Hz"
        if time_s < 1:
            time_str = f"{time_s * 1000:.1f} ms"
        elif time_s < 60:
            time_str = f"{time_s:.3f} s"
        else:
            m = int(time_s // 60)
            s = time_s % 60
            time_str = f"{m}m{s:05.2f}s"
        text = f"{time_str}  ·  {freq_str}  ·  {db:.1f} dB"
        self._cursor_label.setText(text)
        self._cursor_label.adjustSize()
        # Position: center on cursor x, vertically centered in filename row
        lw = self._cursor_label.width()
        label_x = SIDE + CARD_INSET + px - lw // 2
        # Clamp so label stays within spectrogram area
        spec_right = self._spec.mapToParent(self._spec.rect().topRight()).x()
        label_x = max(SIDE + CARD_INSET, min(label_x, spec_right - lw))
        label_y = (SIDE + CARD_INSET - self._cursor_label.height()) // 2
        self._cursor_label.move(label_x, max(0, label_y))
        self._cursor_label.show()
        self._filename_widget.hide()

    @safe_slot
    def _on_cursor_left(self) -> None:
        """Restore filename when cursor leaves spectrogram."""
        self._cursor_label.hide()
        self._filename_widget.show()

    @safe_slot
    def _on_quality_done(self, qa: Any) -> None:
        self._meta.load_analysis(qa)
        old_quality = self._quality_worker
        self._quality_worker = None
        self._hold_ref(old_quality)
        if qa and "upsampling" in qa:
            cutoff = qa["upsampling"].get("cutoff_hz")
            self._spec.set_cutoff_line(cutoff)

    def _on_save_screenshot(self) -> None:
        if not self._spec:
            return
        default = str(self._current_path.with_suffix(".png").name) if self._current_path else "spectrogram.png"
        path_str, _ = QFileDialog.getSaveFileName(
            self, t("保存截图", "Save Screenshot"), default,
            "PNG Image (*.png)"
        )
        if path_str:
            img = self._spec.grabFramebuffer()
            if img:
                img.save(path_str, "PNG")
            self._status_label.setText(f"{t('已保存', 'Saved')}  {path_str}")

    def _start_batch_analysis(self, files: list[Path]) -> None:
        from ui.batch_dialog import BatchProgressDialog
        from analyzer.batch import export_batch_csv

        dialog = BatchProgressDialog(len(files), self)
        dialog.show()

        self._batch_results = []
        worker = _BatchWorker(files)
        self._batch_worker = worker

        def on_progress(n, name, ok):
            dialog.update_progress(n, name, ok)

        def on_finished(results):
            self._batch_results = results
            self._batch_worker = None
            dialog.finish()
            self._status_label.setText(
                t(f"批量完成: {len(results)} 个文件",
                  f"Batch done: {len(results)} files"))

        def on_export():
            dest, _ = QFileDialog.getSaveFileName(
                self, t("导出CSV", "Export CSV"), "batch_analysis.csv",
                "CSV Files (*.csv)")
            if dest:
                export_batch_csv(self._batch_results, Path(dest))
                self._status_label.setText(f"{t('已导出', 'Exported')} {dest}")

        def on_cancel():
            if worker.isRunning():
                worker.requestInterruption()

        worker.progress.connect(on_progress)
        worker.finished.connect(on_finished)
        dialog.cancelled.connect(on_cancel)
        dialog._export_btn.clicked.connect(on_export)

        worker.start()

    def dragEnterEvent(self, event: QDragEnterEvent) -> None:
        if event.mimeData().hasUrls():
            for url in event.mimeData().urls():
                if url.isLocalFile() and self._is_audio(url.toLocalFile()):
                    event.acceptProposedAction()
                    return
        event.ignore()

    def dropEvent(self, event: QDropEvent) -> None:
        audio_paths = [Path(url.toLocalFile()) for url in event.mimeData().urls()
                       if url.isLocalFile() and self._is_audio(url.toLocalFile())]
        if len(audio_paths) == 1:
            self._load_file(audio_paths[0])
        elif len(audio_paths) > 1:
            self._start_batch_analysis(audio_paths)

    def _on_toggle_lang(self) -> None:
        toggle_lang()
        from lang import LANG
        self._lang_btn.setText("EN" if LANG == "zh" else "中")

    def _retranslate(self, _lang: str | None = None) -> None:
        self._update_tooltips()
        if self._logo_ok:
            pass  # logo 已替换品牌文字，不覆盖
        else:
            self._brand_label.setText("Spectra")
        self._open_btn.setText(t("打开文件", "Open File"))
        self._save_btn.setText(t("保存PNG", "Save PNG"))
        self._pal_label.setText(t("调色板", "Palette"))
        self._mode_label.setText(t("模式", "Mode"))
        self._yscale_label.setText(t("刻度", "Scale"))
        self._play_label.setText(t("播放", "Play"))
        self._zoom_hint.setText(
            t("滚轮: 缩放时间  Shift+滚轮: 缩放频率  双击: 重置",
              "Wheel: zoom time  Shift+Wheel: zoom freq  Dbl-click: reset"))
        self._empty_hint.setText(
            t("拖入音频文件，或按 Ctrl+O 打开",
              "Drop an audio file here, or press Ctrl+O"))
        if not self._current_path:
            self._status_label.setText(t("就绪", "Ready"))
        self.setWindowTitle("Spectra")

    def _on_minimize(self) -> None:
        self.showMinimized()

    def _on_maximize_toggle(self) -> None:
        if self.isMaximized():
            self.showNormal()
        else:
            self.showMaximized()

    def showEvent(self, event) -> None:
        super().showEvent(event)
        if not getattr(self, "_rounded_applied", False):
            self._rounded_applied = True
            self._apply_win11_rounding()

    def _apply_win11_rounding(self) -> None:
        """Best-effort Win11 rounded corners for the frameless window (no-op elsewhere)."""
        try:
            import ctypes
            from ctypes import wintypes
            dwmapi = ctypes.windll.dwmapi  # Windows only
            DWMWA_WINDOW_CORNER_PREFERENCE = 33
            DWMWCP_ROUND = 2
            pref = ctypes.c_int(DWMWCP_ROUND)
            hwnd = int(self.winId())
            dwmapi.DwmSetWindowAttribute(
                wintypes.HWND(hwnd),
                DWMWA_WINDOW_CORNER_PREFERENCE,
                ctypes.byref(pref),
                ctypes.sizeof(pref),
            )
        except Exception:
            pass

    def closeEvent(self, event) -> None:
        self._cancel_spectrum()
        self._cancel_quality()
        if self._batch_worker is not None and self._batch_worker.isRunning():
            self._batch_worker.requestInterruption()
            self._batch_worker.wait(3000)
        super().closeEvent(event)

    @staticmethod
    def _is_audio(path: str) -> bool:
        return is_audio_file(path)
