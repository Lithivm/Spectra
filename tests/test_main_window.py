import sys
import pytest
from PyQt6.QtWidgets import QApplication, QFileDialog, QLabel


# ── helpers ──────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def app():
    app = QApplication.instance() or QApplication(sys.argv)
    yield app


def _build_main_window(app: QApplication):
    from ui.main_window import MainWindow
    mw = MainWindow()
    mw.resize(800, 600)
    mw.show()
    app.processEvents()
    return mw


class TestIsAudio:
    """Test the static `_is_audio` method on MainWindow."""

    def test_mp3(self, app):
        from ui.main_window import MainWindow
        assert MainWindow._is_audio("song.mp3") is True

    def test_wav(self, app):
        from ui.main_window import MainWindow
        assert MainWindow._is_audio("song.wav") is True

    def test_ogg(self, app):
        from ui.main_window import MainWindow
        assert MainWindow._is_audio("song.ogg") is True

    def test_flac(self, app):
        from ui.main_window import MainWindow
        assert MainWindow._is_audio("song.flac") is True

    def test_txt_rejected(self, app):
        from ui.main_window import MainWindow
        assert MainWindow._is_audio("readme.txt") is False

    def test_json_rejected(self, app):
        from ui.main_window import MainWindow
        assert MainWindow._is_audio("config.json") is False

    def test_png_rejected(self, app):
        from ui.main_window import MainWindow
        assert MainWindow._is_audio("photo.png") is False


class TestDragDrop:
    """Test drag-and-drop file handling."""

    def test_open_file_accepted_audio(self, app):
        _build_main_window(app)
        result = QFileDialog.getOpenFileName(None, "Open Audio", "", "Audio Files (*.mp3 *.wav *.ogg *.flac)")
        assert isinstance(result, tuple)

    def test_palette_switch(self, app):
        mw = _build_main_window(app)
        mw._on_palette_changed("viridis")
        assert mw._current_palette == "viridis"


class TestMainWindowBasics:
    """Basic MainWindow construction."""

    def test_window_created(self, app):
        mw = _build_main_window(app)
        assert mw.windowTitle() == "Spectra"

    def test_has_spectrogram(self, app):
        mw = _build_main_window(app)
        assert mw._spec is not None

    def test_has_metadata_panel(self, app):
        mw = _build_main_window(app)
        assert mw._meta is not None

    def test_empty_hint_visible_initially(self, app):
        mw = _build_main_window(app)
        assert mw._empty_hint.text() != ""
        assert mw._empty_hint.isVisible()

    def test_has_waveform(self, app):
        mw = _build_main_window(app)
        assert mw._wave is not None

    def test_has_axes(self, app):
        mw = _build_main_window(app)
        assert mw._y_axis is not None
        assert mw._x_axis is not None
        assert mw._colorbar is not None

    def test_has_filename_widget(self, app):
        mw = _build_main_window(app)
        assert isinstance(mw._filename_widget, QLabel)


class TestStyleApplied:
    """Verify that the stylesheet is applied."""

    def test_mainwindow_has_stylesheet(self, app):
        mw = _build_main_window(app)
        style = mw.styleSheet()
        assert "QMainWindow" in style

    def test_metadata_panel_has_stylesheet(self, app):
        mw = _build_main_window(app)
        style = mw._meta.styleSheet()
        assert "background-color" in style


class TestDragDropRejection:
    """Test that non-audio files are rejected in drag-and-drop."""

    def test_is_audio_rejects_non_media(self, app):
        from ui.main_window import MainWindow
        assert MainWindow._is_audio("readme.txt") is False
        assert MainWindow._is_audio("data.bin") is False
        assert MainWindow._is_audio("photo.jpeg") is False

    def test_is_audio_accepts_audio(self, app):
        from ui.main_window import MainWindow
        for ext in [".wav", ".mp3", ".flac", ".ogg", ".m4a", ".opus"]:
            assert MainWindow._is_audio(f"song{ext}") is True


class TestThemeSwitch:
    """Test theme/palette switching."""

    def test_initial_palette(self, app):
        mw = _build_main_window(app)
        assert mw._current_palette == "spectra"

    def test_mode_default(self, app):
        mw = _build_main_window(app)
        assert mw._mode == "standard"


class TestToolbar:
    """Verify toolbar widgets exist."""

    def test_toolbar_buttons_exist(self, app):
        mw = _build_main_window(app)
        # open button exists and is a QPushButton
        from PyQt6.QtWidgets import QPushButton
        assert isinstance(mw._open_btn, QPushButton)
        assert isinstance(mw._save_btn, QPushButton)
        assert isinstance(mw._lang_btn, QPushButton)

    def test_combo_boxes_exist(self, app):
        from PyQt6.QtWidgets import QComboBox
        mw = _build_main_window(app)
        assert isinstance(mw._palette_combo, QComboBox)
        assert isinstance(mw._mode_combo, QComboBox)
        assert isinstance(mw._fft_combo, QComboBox)
        assert isinstance(mw._yscale_combo, QComboBox)


class TestVisualUpgrade:
    """Round-4 visual upgrade: shader floor, LUT flat-floor sync, inset well, logo."""

    def test_shader_compiles(self, app):
        from ui.spectrogram_widget import SpectrogramGLWidget
        w = SpectrogramGLWidget()
        w.show()
        app.processEvents()
        w.makeCurrent()
        try:
            # initializeGL compiles + links the shader; raises RuntimeError on failure.
            w.initializeGL()
        finally:
            w.doneCurrent()

    def test_spectra_floor_matches_bg_canvas(self, app):
        from ui.styles import BG_CANVAS
        from ui.spectrogram_widget import SpectrogramGLWidget
        w = SpectrogramGLWidget()
        w.set_palette("spectra")
        r, g, b = (int(BG_CANVAS[i:i + 2], 16) / 255.0 for i in (1, 3, 5))
        assert abs(w._lut_floor[0] - r) < 0.01
        assert abs(w._lut_floor[1] - g) < 0.01
        assert abs(w._lut_floor[2] - b) < 0.01

    def test_spec_card_inset_well(self, app):
        from ui.styles import BG_CANVAS
        mw = _build_main_window(app)
        assert BG_CANVAS in mw._spec_card.styleSheet()

    def test_logo_label_has_pixmap(self, app):
        mw = _build_main_window(app)
        assert mw._logo_label.pixmap() is not None
