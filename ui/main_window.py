"""Main application window — modern AI product aesthetic."""

from __future__ import annotations

import itertools
import sys
import time
import traceback
from pathlib import Path
from typing import Any

from PyQt6.QtCore import QThread, Qt, pyqtSignal, QCoreApplication
from PyQt6.QtGui import QDragEnterEvent, QDropEvent, QColor
from PyQt6.QtWidgets import (
    QApplication, QFileDialog, QHBoxLayout, QVBoxLayout,
    QLabel, QMainWindow, QMessageBox, QStatusBar, QWidget,
    QFrame, QPushButton, QComboBox, QSizePolicy,
    QGraphicsDropShadowEffect, QGridLayout,
    QProgressBar,
)
from PyQt6.QtCore import QPropertyAnimation, QEasingCurve
from PyQt6.QtGui import QDragEnterEvent, QDropEvent, QColor, QPainter


from analyzer.core import AudioAnalyzer
from analyzer.spectrogram import PALETTE
from analyzer import is_audio_file, SUPPORTED_EXTENSIONS
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

class _SpectrumProgress(QWidget):
    """Floating progress bar overlay for the spectrogram."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self._set_stylesheet()
        self._opacity = QGraphicsOpacityEffect(self)
        self._opacity.setOpacity(0)
        self._progress = QProgressBar(self)
        self._progress.setFixedHeight(6)
        self._progress.setValue(0)
        self.setEffect(self._opacity)
        self._anim = QPropertyAnimation(self._opacity, b"opacity")
        self._anim.setDuration(800)
        self._anim.setEasingCurve(QEasingCurve.Type.InOutCubic)
        self._anim.finished.connect(self._on_anim_finished)

    def _set_stylesheet(self) -> None:
        self.setStyleSheet(f"""
            #_spectrum_progress {{
                background-color: rgba(20, 20, 24, 0.6);
                border: none;
                border-radius: 3px;
            }}
            #_spectrum_progress QProgressBar::chunk {{
                background-color: qlineargradient(
                    x1:0, y1:0, x2:1, y2:0,
                    stop:0 #ff7535,
                    stop:0.5 #ffa319,
                    stop:1 #ffcf49
                );
                border: none;
                border-radius: 3px;
            }}
        """)
        self.setObjectName("_spectrum_progress")

    def show_progress(self) -> None:
        self._opacity.setOpacity(0.9)
        self._anim.setStartValue(0.0)
        self._anim.setEndValue(0.9)
        self._anim.start()

    def update(self, value: int) -> None:
        self._progress.setValue(value)

    def fade_out(self) -> None:
        self._anim.setStartValue(0.9)
        self._anim.setEndValue(0.0)
        self._anim.start()

    def _on_anim_finished(self) -> None:
        if self._opacity.opacity() < 0.01:
            self.hide()

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        w, h = event.size().width(), event.size().height()
        bar_w = int(w * 0.5)
        self._progress.setGeometry(
            (w - bar_w) // 2,
            (h - 6) // 2,
            bar_w,
            6,
        )

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


class _SpectrumWorker(QThread):
    finished = pyqtSignal(object, object, object, str)
    fading = pyqtSignal()

    def __init__(self, analyzer: AudioAnalyzer, n_fft: int = 2048,
                 mode: str = "multi") -> None:
        super().__init__()
        self._analyzer = analyzer
        self._n_fft = n_fft
        self._mode = mode

    def run(self) -> None:
        t0 = time.perf_counter()
        freqs, times, db = self._analyzer.spectrogram_db(
            n_fft=self._n_fft, mode=self._mode)
        print(f"[TIMER] STFT (n_fft={self._n_fft}, mode={self._mode}): "
              f"{time.perf_counter() - t0:.2f}s")
        self.finished.emit(freqs, times, db, self._mode)
        self.fading.emit()


class _QualityWorker(QThread):
    finished = pyqtSignal(object)
    progress = pyqtSignal(int)

    def __init__(self, analyzer: AudioAnalyzer) -> None:
        super().__init__()
        self._analyzer = analyzer

    def run(self) -> None:
        self.progress.emit(33)  # audio decoded
        try:
            result = self._analyzer.analyze_quality()
        except Exception:
            result = None
        self.progress.emit(90)  # STFT done
        self.finished.emit(result)
        self.progress.emit(100)  # render done


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setAcceptDrops(True)
        self._title = "Spectra"
        self._current_path: Path | None = None
        self._spectrum_worker: _SpectrumWorker | None = None
        self._quality_worker: _QualityWorker | None = None
        self._wave: WaveformWidget | None = None
        self._spec: SpectrogramGLWidget | None = None
        self._y_axis: _YAxisWidget | None = None
        self._x_axis: _XAxisWidget | None = None
        self._colorbar: _ColorBarWidget | None = None
        self._meta: MetadataPanel | None = None
        self._analyzer: AudioAnalyzer | None = None
        self._current_palette = "inferno"
        self._fft_size = 2048
        self._mode = "standard"

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

        # 波形卡片
        wave_card = _card(radius=10)
        wave_card.setFixedHeight(130)
        wl = QVBoxLayout(wave_card)
        wl.setContentsMargins(0, 0, 0, 0)
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
        _grid.setColumnMinimumWidth(2, SIDE)
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
        self._colorbar.setFixedWidth(SIDE)
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

        self._spec._on_palette_changed("inferno")

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
        self._fft_combo.setCurrentText("2048")
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
        self._lang_btn.setFixedHeight(39)
        self._lang_btn.setFixedWidth(62)
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
            self._spec._on_palette_changed(name)
            if self._colorbar:
                self._colorbar.set_data(self._spec._lut_np)

    def _on_mode_changed(self, mode: str) -> None:
        self._mode = mode
        self._fft_combo.setEnabled(mode != "multi")
        if self._current_path:
            self._load_file(self._current_path)

    def _on_yscale_changed(self, scale: str) -> None:
        if self._spec:
            self._spec._on_yscale_changed(scale)
            if self._y_axis and self._spec.frequencies is not None:
                self._y_axis.set_data(self._spec.frequencies, scale)

    def _on_fft_size_changed(self, size_str: str) -> None:
        self._fft_size = int(size_str)
        if self._current_path:
            self._load_file(self._current_path)

    def _on_open_file(self) -> None:
        formats = ["*.wav", "*.mp3", "*.flac", "*.m4a", "*.ogg", "*.aac", "*.aiff", "*.opus", "*.ape"]
        path_str, _ = QFileDialog.getOpenFileName(
            self, t("打开音频文件", "Open Audio File"), str(Path.home()),
            "Audio Files (" + " ".join(formats) + ")"
        )
        if path_str:
            self._load_file(Path(path_str))

    def _on_spectrum_done(self, freqs: Any, times: Any, db: Any, mode: str = "multi") -> None:
        print(f"[TIMER] 频谱计算完成, 文件 {self._current_path.name}, mode={mode}")
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
        print(f"[PROFILE] _on_spectrum_done (set_audio + processEvents): {time.perf_counter() - t_set:.3f}s")
        QCoreApplication.processEvents()

    def _load_file(self, path: Path) -> None:
        try:
            self._spec.show_progress()
            self._cancel_spectrum()
            self._cancel_quality()
            t0 = time.perf_counter()
            self._analyzer = AudioAnalyzer(path)
            print(f"[TIMER] 音频解码: {time.perf_counter()-t0:.2f}s")
            t0 = time.perf_counter()
            self._meta.load_metadata(self._analyzer)
            print(f"[TIMER] 元数据: {time.perf_counter()-t0:.2f}s")
            # quality analysis runs in background
            self._quality_worker = _QualityWorker(self._analyzer)
            self._quality_worker.finished.connect(self._on_quality_done)
            self._quality_worker.progress.connect(self._spec.show_progress)
            self._quality_worker.start()
            t0 = time.perf_counter()
            self._wave.set_audio({
                'waveform': self._analyzer.waveform,
                'sample_rate': self._analyzer.sample_rate,
                'duration': self._analyzer.duration,
            })
            print(f"[TIMER] 波形渲染: {time.perf_counter()-t0:.2f}s")
            self._current_path = path
            self.setWindowTitle(f"Spectra  —  {path.name}")
            self._status_label.setText(
                f"{path.name}  ·  {self._analyzer.sample_rate/1000:.1f} kHz  ·  "
                f"{int(self._analyzer.duration)//60}m{int(self._analyzer.duration)%60}s"
            )
            self._filename_widget.setText(path.name)
            self._spectrum_worker = _SpectrumWorker(self._analyzer, self._fft_size, self._mode)
            self._spectrum_worker.finished.connect(self._on_spectrum_done)
            self._spectrum_worker.fading.connect(self._spec.hide_progress)
            self._spectrum_worker.start()
        except Exception:
            print(f"\n[LOAD ERROR] {path}\n")
            traceback.print_exc(file=sys.stderr)
            QMessageBox.critical(self, t("错误", "Error"), t(f"无法加载:\n{path.name}", f"Cannot load:\n{path.name}"))

    def _cancel_spectrum(self) -> None:
        if self._spectrum_worker is not None:
            try:
                self._spectrum_worker.requestInterruption()
                self._spectrum_worker.wait()
            except Exception:
                pass
            self._spectrum_worker = None

    def _cancel_quality(self) -> None:
        if self._quality_worker is not None:
            try:
                self._quality_worker.requestInterruption()
                self._quality_worker.wait()
            except Exception:
                pass
            self._quality_worker = None

    def _on_quality_done(self, qa: Any) -> None:
        self._meta.load_analysis(qa)
        self._quality_worker = None

    def _on_save_screenshot(self) -> None:
        if not self._spec:
            return
        default = str(self._current_path.with_suffix(".png").name) if self._current_path else "spectrogram.png"
        path_str, _ = QFileDialog.getSaveFileName(
            self, t("保存截图", "Save Screenshot"), default,
            "PNG Image (*.png);;JPEG Image (*.jpg)"
        )
        if path_str:
            img = self._spec.grabFramebuffer()
            if img:
                img.save(path_str, "PNG")
            self._status_label.setText(f"{t('已保存', 'Saved')}  {path_str}")

    def dragEnterEvent(self, event: QDragEnterEvent) -> None:
        if event.mimeData().hasUrls():
            for url in event.mimeData().urls():
                if url.isLocalFile() and self._is_audio(url.toLocalFile()):
                    event.acceptProposedAction()
                    return
        event.ignore()

    def dropEvent(self, event: QDropEvent) -> None:
        for url in event.mimeData().urls():
            if url.isLocalFile():
                self._load_file(Path(url.toLocalFile()))
                return

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
        if not self._current_path:
            self._status_label.setText(t("就绪", "Ready"))
        self.setWindowTitle("Spectra")

    def closeEvent(self, event) -> None:
        self._cancel_spectrum()
        self._cancel_quality()
        super().closeEvent(event)

    @staticmethod
    def _is_audio(path: str) -> bool:
        return Path(path).suffix.lower() in {
            ".wav", ".mp3", ".flac", ".ogg", ".m4a",
            ".aac", ".wma", ".aiff", ".mid", ".opus", ".ape",
        }
