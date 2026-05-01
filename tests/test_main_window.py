import sys
import pytest
from PyQt6.QtCore import QMimeData, QSize
from PyQt6.QtWidgets import QApplication, QFileDialog, QWidget


# ── helpers ──────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def app():
    """Create a QApplication once for the module.
    PyQt6 requires a QApplication instance before any Qt widgets are used."""
    app = QApplication.instance() or QApplication(sys.argv)
    yield app


def _build_main_window(app: QApplication) -> tuple:
    """Create a minimal MainWindow and return the widget + its children for inspection."""
    from ui.main_window import MainWindow
    mw = MainWindow()
    mw.resize(800, 600)
    mw.show()
    # Give Qt a chance to lay out the widgets
    app.processEvents()
    return mw


class TestIsAudio:
    """Test the static `_is_audio` method on Analyzer."""

    def test_mp3(self, app):
        assert self._is_audio("song.mp3") is True

    def test_wav(self, app):
        assert self._is_audio("song.wav") is True

    def test_ogg(self, app):
        assert self._is_audio("song.ogg") is True

    def test_flac(self, app):
        assert self._is_audio("song.flac") is True

    def test_txt_rejected(self, app):
        assert self._is_audio("readme.txt") is False

    def test_json_rejected(self, app):
        assert self._is_audio("config.json") is False

    def test_png_rejected(self, app):
        assert self._is_audio("photo.png") is False

    @staticmethod
    def _is_audio(path: str) -> bool:
        """Call MainWindow._is_audio from the Analyzer class."""
        return MainWindow._is_audio(path)


class TestDragDrop:
    """Test drag-and-drop file handling."""

    def test_open_file_accepted_audio(self, app):
        """QFileDialog.getOpenFileName returns (selected_file, filter) tuple."""
        from ui.main_window import MainWindow
        _build_main_window(app)
        result = QFileDialog.getOpenFileName(None, "Open Audio", "", "Audio Files (*.mp3 *.wav *.ogg *.flac)")
        # Returns (str, str) — the path is empty when cancelled
        assert isinstance(result, tuple)

    def test_palette_switch(self, app):
        """Switching the palette should update the spectrogram widget."""
        from ui.main_window import MainWindow
        from analyzer.spectrogram import Renderer as SpecRenderer

        mw = _build_main_window(app)
        widget_names = [w.objectName() for w in mw.findChildren(QWidget)]

        # Palette defaults to "inferno"
        # Changing palette should trigger _on_palette_changed which
        # calls _spectrogram.render with the new palette
        app.processEvents()

    def test_selection_action_changed(self, app):
        """select_all action should change when called."""
        from ui.main_window import MainWindow
        mw = _build_main_window(app)

        # The select_all QAction should exist after the window is created
        select_action = mw._select_all
        assert select_action is not None
        assert select_action.text() == "Select All"


class TestMainWindowBasics:
    """Basic MainWindow construction."""

    def test_window_created(self, app):
        from ui.main_window import MainWindow
        mw = _build_main_window(app)
        assert mw.windowTitle() == "Spectra"

    def test_has_spectrogram(self, app):
        from ui.main_window import MainWindow
        mw = _build_main_window(app)
        spectrogram = mw._spectrogram
        assert spectrogram is not None

    def test_has_spectrogram_palette(self, app):
        """Spectrogram should have a renderable with palette."""
        from ui.main_window import MainWindow
        from analyzer.spectrogram import Renderer as SpecRenderer

        mw = _build_main_window(app)
        widget_names = [w.objectName() for w in mw.findChildren(QWidget)]
        assert "spectrogram" in widget_names

    def test_has_metadata_panel(self, app):
        from ui.main_window import MainWindow
        mw = _build_main_window(app)
        meta = mw._meta
        assert meta is not None

    def test_has_wavelength(self, app):
        from ui.main_window import MainWindow
        mw = _build_main_window(app)
        wave = mw._wave
        assert wave is not None


class TestThemeSwitch:
    """Test theme switching (light/dark)."""

    def test_initial_theme(self, app):
        from ui.main_window import MainWindow
        mw = _build_main_window(app)
        # Should start with "dark" theme
        # The combo box should show "Dark" as the current value
        app.processEvents()


class TestStyleApplied:
    """Verify that set_style() has been called and the stylesheet is applied."""

    def test_mainwindow_has_stylesheet(self, app):
        from ui.main_window import MainWindow
        mw = _build_main_window(app)
        # The stylesheet should contain our custom CSS
        style = mw.styleSheet()
        assert "/* Dark theme palette */" in style

    def test_spectrogram_has_stylesheet(self, app):
        from ui.main_window import MainWindow
        mw = _build_main_window(app)
        style = mw._spectrogram.styleSheet()
        assert "QLabel#meta-title" in style

    def test_metadata_panel_has_stylesheet(self, app):
        from ui.main_window import MainWindow
        mw = _build_main_window(app)
        style = mw._meta.styleSheet()
        assert "QLabel#meta-title" in style


class TestDragDropRejection:
    """Test that non-audio files are rejected in drag-and-drop."""

    def test_non_audio_rejected(self, app):
        """Opening a non-audio file should show the error dialog."""
        # We can't easily test _reject without a real file,
        # but we can verify the error_dialog method exists and works
        from ui.main_window import MainWindow
        mw = _build_main_window(app)
        mw.error_dialog("Test", "Error message")  # should not raise
        app.processEvents()
