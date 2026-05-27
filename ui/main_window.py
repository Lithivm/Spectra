"""Main application window — modern AI product aesthetic."""

from __future__ import annotations

import itertools
import logging
import sys
import time
import traceback
from pathlib import Path

import numpy as np
from typing import Any

from PyQt6.QtCore import QThread, Qt, pyqtSignal, QTimer
from PyQt6.QtGui import QDragEnterEvent, QDropEvent, QColor, QPainter
from PyQt6.QtWidgets import (
    QApplication, QFileDialog, QHBoxLayout, QVBoxLayout,
    QLabel, QMainWindow, QMessageBox, QStatusBar, QWidget,
    QFrame, QPushButton, QComboBox, QSizePolicy,
    QGraphicsDropShadowEffect, QGridLayout,
    QProgressBar,
)


from ui.playback_engine import PlaybackEngine
from analyzer.core import AudioAnalyzer, _stft_cache, _stft_lock
from analyzer.palette import PALETTE
from analyzer import is_audio_file, SUPPORTED_EXTENSIONS

logger = logging.getLogger(__name__)
from ui.metadata_panel import MetadataPanel
from ui.spectrogram_widget import SpectrogramGLWidget, _YAxisWidget, _XAxisWidget, _ColorBarWidget
from ui.waveform_widget import WaveformWidget
from ui.styles import (
    BG_BASE, BG_SURFACE, BG_RAISED,
    BORDER_SUB, BORDER_MID,
    ACCENT, ACCENT_ALT,
    TEXT_PRI, TEXT_SEC, TEXT_DIM,
)
from lang import t, toggle_lang, on_lang_change

APP_STYLESHEET = f"""
* {{
    font-family: "Segoe UI", "SF Pro Display", sans-serif;
}}
QMainWindow {{
    background-color: {BG_BASE};
}}
QWidget {{
    background-color: transparent;
    color: {TEXT_PRI};
    font-size: 12px;
}}
QStatusBar {{
    background-color: {BG_SURFACE};
    color: {TEXT_DIM};
    border-top: 1px solid {BORDER_SUB};
    font-size: 10px;
    font-family: "Consolas", monospace;
    padding: 0 12px;
}}
QComboBox {{
    background-color: {BG_RAISED};
    border: 1px solid {BORDER_MID};
    border-radius: 6px;
    color: {TEXT_PRI};
    padding: 4px 10px;
    font-size: 11px;
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
    border-radius: 6px;
    selection-background-color: {ACCENT};
    color: {TEXT_PRI};
    padding: 4px;
    outline: none;
}}
QPushButton {{
    background-color: {BG_RAISED};
    border: 1px solid {BORDER_MID};
    border-radius: 7px;
    color: {TEXT_SEC};
    padding: 5px 16px;
    font-size: 11px;
    font-weight: 500;
}}
QPushButton:hover {{
    border-color: {ACCENT};
    color: {TEXT_PRI};
    background-color: rgba(124, 106, 247, 0.08);
}}
QPushButton:pressed {{
    background-color: rgba(124, 106, 247, 0.15);
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
        stop:0 #E0B55A, stop:1 #C89A3A);
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

def _card(radius: int = 12) -> QWidget:
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
            self.analyzer = AudioAnalyzer(self._path)
            self.loaded.emit()
        except Exception as e:
            self.error.emit(str(e))


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
        self._current_palette = "inferno"
        self._fft_size = 8192
        self._mode = "standard"

        self._playback = PlaybackEngine(self)
        self._playback_timer = QTimer(self)
        self._playback_timer.setInterval(30)
        self._playback_timer.timeout.connect(self._on_playback_tick)
        self._playback_timer.start()

        self.setStyleSheet(APP_STYLESHEET)
        self._setup_ui()

    def _setup_ui(self) -> None:
        self.setWindowTitle(self._title)
        self.resize(1440, 880)
        self.menuBar().setVisible(False)
        self._create_statusbar()

        root = QWidget()
        root.setStyleSheet(f"background: {BG_BASE};")
        self.setCentralWidget(root)
        root_layout = QHBoxLayout(root)
        root_layout.setContentsMargins(12, 12, 12, 12)
        root_layout.setSpacing(10)

        # 左侧
        left_layout = QVBoxLayout()
        left_layout.setSpacing(10)
        left_layout.setContentsMargins(0, 0, 0, 0)

        left_layout.addWidget(self._make_toolbar())

        # 波形卡片 — left/right margins align with spectrogram (y-axis + colorbar)
        wave_card = _card(radius=10)
        wave_card.setFixedHeight(130)
        wl = QVBoxLayout(wave_card)
        wl.setContentsMargins(36, 0, 36, 0)
        self._wave = WaveformWidget()
        wl.addWidget(self._wave)
        left_layout.addWidget(wave_card)

        # 频谱卡片
        spec_card = _card(radius=10)
        sl = QVBoxLayout(spec_card)
        sl.setContentsMargins(0, 0, 0, 0)
        sl.setSpacing(0)

        # ---- Grid: filename | Y-axis | spectrogram | colorbar | X-axis ----
        _grid = QGridLayout()
        _grid.setContentsMargins(0, 0, 0, 0)
        _grid.setSpacing(0)

        SIDE = 36

        _grid.setColumnMinimumWidth(0, SIDE)
        _grid.setColumnStretch(0, 0)
        _grid.setColumnStretch(1, 1)
        _grid.setColumnMinimumWidth(2, 36)
        _grid.setColumnStretch(2, 0)

        _grid.setRowMinimumHeight(0, SIDE)
        _grid.setRowStretch(0, 0)
        _grid.setRowStretch(1, 1)
        _grid.setRowMinimumHeight(2, SIDE)
        _grid.setRowStretch(2, 0)

        # Row 0: filename
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

        self._colorbar = _ColorBarWidget()
        self._colorbar.setFixedWidth(36)
        self._colorbar.set_data(self._spec._lut_np)
        _grid.addWidget(self._colorbar, 0, 2, 3, 1)

        # Row 2: X-axis spans all 3 columns, data offset-aligned to spectrogram
        self._x_axis = _XAxisWidget()
        self._x_axis.setFixedHeight(SIDE)
        _grid.addWidget(self._x_axis, 2, 0, 1, 3)

        sl.addLayout(_grid, 0)

        left_layout.addWidget(spec_card, stretch=1)

        root_layout.addLayout(left_layout, stretch=3)

        # 右侧元数据面板
        self._meta = MetadataPanel()
        self._meta.setFixedWidth(310)
        root_layout.addWidget(self._meta)

        self._spec.set_palette("inferno")

        # Wire up seek signals from waveform + spectrogram → playback engine
        self._wave.seekRequested.connect(self._playback.seek)
        self._spec.seekRequested.connect(self._playback.seek)
        # Single playhead position sync — drag on either widget updates both
        self._wave._on_playhead_drag = self._on_playhead_drag
        self._spec._on_playhead_drag = self._on_playhead_drag
        self._playback.state_changed.connect(self._on_playback_state)

        on_lang_change(self._retranslate)

    def _make_toolbar(self) -> QWidget:
        card = _card(radius=10)
        card.setFixedHeight(52)
        layout = QHBoxLayout(card)
        layout.setContentsMargins(16, 0, 16, 0)
        layout.setSpacing(12)

        self._brand_label = QLabel("Spectra")
        self._brand_label.setStyleSheet(f"""
            color: {TEXT_PRI};
            font-size: 15px;
            font-weight: 700;
            background: transparent;
            border: none;
        """)
        layout.addWidget(self._brand_label)

        sep0 = QFrame()
        sep0.setFrameShape(QFrame.Shape.VLine)
        sep0.setStyleSheet(f"background: {BORDER_SUB}; border: none; max-width: 1px;")
        sep0.setFixedHeight(20)
        layout.addWidget(sep0)

        self._open_btn = QPushButton(t("打开文件", "Open File"))
        self._open_btn.setObjectName("primary")
        self._open_btn.setFixedHeight(32)
        self._open_btn.setFixedWidth(100)
        self._open_btn.clicked.connect(self._on_open_file)
        layout.addWidget(self._open_btn)

        # 播放标签
        self._play_label = QLabel(t("播放", "Play"))
        self._play_label.setStyleSheet(f"color: {TEXT_DIM}; font-size: 11px; background: transparent; border: none;")
        layout.addWidget(self._play_label)

        # Play / Pause
        self._play_btn = QPushButton("▶")
        self._play_btn.setFixedSize(36, 30)
        self._play_btn.setToolTip(t("播放/暂停", "Play / Pause"))
        self._play_btn.setStyleSheet(f"""
            QPushButton {{
                font-size: 14px; font-weight: bold;
                color: {TEXT_PRI}; background: #222526;
                border: 1px solid #444; border-radius: 4px;
                padding: 0px 0px 2px 0px;
            }}
            QPushButton:hover {{ border-color: {ACCENT}; background: #2a2d2f; }}
            QPushButton:pressed {{ background: #1a1c1d; }}
        """)
        self._play_btn.clicked.connect(self._on_playback_toggle)
        layout.addWidget(self._play_btn)

        layout.addStretch()

        self._pal_label = QLabel(t("调色板", "Palette"))
        self._pal_label.setStyleSheet(f"color: {TEXT_DIM}; font-size: 11px; background: transparent; border: none;")
        layout.addWidget(self._pal_label)
        self._palette_combo = QComboBox()
        self._palette_combo.addItems(list(PALETTE.keys()))
        self._palette_combo.setCurrentText("inferno")
        self._palette_combo.setFixedHeight(30)
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
        self._mode_combo.setFixedHeight(30)
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
        self._yscale_combo.setFixedHeight(30)
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
        self._fft_combo.setFixedHeight(30)
        self._fft_combo.setFixedWidth(72)
        self._fft_combo.currentTextChanged.connect(self._on_fft_size_changed)
        layout.addWidget(self._fft_combo)

        sep4 = QFrame()
        sep4.setFrameShape(QFrame.Shape.VLine)
        sep4.setStyleSheet(f"background: {BORDER_SUB}; border: none; max-width: 1px;")
        sep4.setFixedHeight(20)
        layout.addWidget(sep4)

        self._save_btn = QPushButton(t("保存PNG", "Save PNG"))
        self._save_btn.setFixedHeight(30)
        self._save_btn.clicked.connect(self._on_save_screenshot)
        layout.addWidget(self._save_btn)

        # language toggle
        self._lang_btn = QPushButton("中/EN")
        self._lang_btn.setFixedHeight(30)
        self._lang_btn.setMinimumWidth(60)
        self._lang_btn.clicked.connect(self._on_toggle_lang)
        layout.addWidget(self._lang_btn)

        return card

    def _create_statusbar(self) -> None:
        sb = QStatusBar()
        sb.setFixedHeight(24)
        self._status_label = QLabel(t("就绪", "Ready"))
        self._status_label.setStyleSheet(
            f"color: {TEXT_DIM}; font-size: 10px; font-family: 'Consolas'; background: transparent;")
        sb.addPermanentWidget(self._status_label)
        self.setStatusBar(sb)

    def _on_palette_changed(self, name: str) -> None:
        self._current_palette = name
        if self._spec:
            self._spec.set_palette(name)
            if self._colorbar:
                self._colorbar.set_data(self._spec._lut_np)

    def _on_mode_changed(self, mode: str) -> None:
        self._mode = mode
        self._fft_combo.setEnabled(mode != "multi")
        if self._current_path:
            self._reload_spectrum()

    def _on_yscale_changed(self, scale: str) -> None:
        if self._spec:
            self._spec._on_yscale_changed(scale)
            if self._y_axis and self._spec.frequencies is not None:
                self._y_axis.set_data(self._spec.frequencies, scale)

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
        t_set = time.perf_counter()
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
        self._y_axis.set_data(freqs, self._spec._yscale_mode)
        self._x_axis.set_data(self._analyzer.duration)
        self._colorbar.set_data(self._spec._lut_np)

    @safe_slot
    def _on_stream_init(self, freqs: np.ndarray, total_cols: int, duration: float) -> None:
        n_freqs = freqs.shape[0] if freqs is not None else 1025
        self._spec.begin_stream(n_freqs, total_cols, freqs, duration)
        self._y_axis.set_data(freqs, self._spec._yscale_mode)
        self._x_axis.set_data(duration)
        self._colorbar.set_data(self._spec._lut_np)

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
        self._wave.set_playhead(0.0)
        if self._spec is not None:
            self._spec.set_playhead(0.0)

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
                except Exception:
                    pass
            self._hold_ref(old)

    def _cancel_quality(self) -> None:
        if self._quality_worker is not None:
            old = self._quality_worker
            self._quality_worker = None
            old.requestInterruption()
            for sig in (old.finished, old.progress):
                try:
                    sig.disconnect()
                except Exception:
                    pass
            self._hold_ref(old)

    def _cancel_load(self) -> None:
        if self._load_worker is not None:
            old = self._load_worker
            self._load_worker = None
            old.requestInterruption()
            for sig in (old.loaded, old.error):
                try:
                    sig.disconnect()
                except Exception:
                    pass
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

    def _on_playhead_drag(self, seconds: float) -> None:
        """Sync playhead across both widgets during drag.
        Only update engine position when NOT playing — during playback,
        the seek-on-release handles it, avoiding callback interference.
        """
        if not self._playback.is_playing:
            self._playback.track_position(seconds)
        self._wave.playhead_pos = seconds
        self._spec.playhead_pos = seconds
        self._wave.update()
        self._spec.update()

    def _on_playback_tick(self) -> None:
        """Sync waveform + spectrogram playheads to DAC position."""
        if not self._playback.is_playing:
            return
        if getattr(self._wave, '_dragging', False) or getattr(self._spec, '_dragging', False):
            return  # don't fight the user's drag
        pos = self._playback.get_position()
        if self._wave is not None:
            self._wave.set_playhead(pos)
        if self._spec is not None:
            self._spec.set_playhead(pos)

    def _on_playback_toggle(self) -> None:
        self._playback.toggle()

    def _on_playback_state(self, state: str) -> None:
        # Use engine's actual state instead of the signal parameter to avoid
        # stale queued signals (from _on_stream_finished) overwriting current state.
        actual = self._playback.state
        if actual == "playing":
            self._play_btn.setText("‖")  # pause symbol
        else:
            self._play_btn.setText("▶")  # play symbol
            if actual == "stopped":
                self._wave.set_playhead(0.0)
                if self._spec is not None:
                    self._spec.set_playhead(0.0)

    @safe_slot
    def _on_quality_done(self, qa: Any) -> None:
        self._meta.load_analysis(qa)
        old_quality = self._quality_worker
        self._quality_worker = None
        self._hold_ref(old_quality)
        self._spec.hide_progress()
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
        self._brand_label.setText("Spectra")
        self._open_btn.setText(t("打开文件", "Open File"))
        self._save_btn.setText(t("保存PNG", "Save PNG"))
        self._pal_label.setText(t("调色板", "Palette"))
        self._mode_label.setText(t("模式", "Mode"))
        self._yscale_label.setText(t("刻度", "Scale"))
        self._play_label.setText(t("播放", "Play"))
        if not self._current_path:
            self._status_label.setText(t("就绪", "Ready"))
        self.setWindowTitle("Spectra")

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
