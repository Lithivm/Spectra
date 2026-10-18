import sys

import numpy as np
import pytest
from PyQt6.QtWidgets import QApplication, QFileDialog, QLabel


# ── helpers ──────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def app():
    app = QApplication.instance() or QApplication(sys.argv)
    yield app


def _build_main_window(app: QApplication):
    from spectra.frontend.main_window import MainWindow
    mw = MainWindow()
    mw.resize(800, 600)
    mw.show()
    app.processEvents()
    return mw


@pytest.fixture(scope="module")
def mw(app):
    return _build_main_window(app)


class TestIsAudio:
    """Test the static `_is_audio` method on MainWindow."""

    def test_mp3(self, app):
        from spectra.frontend.main_window import MainWindow
        assert MainWindow._is_audio("song.mp3") is True

    def test_wav(self, app):
        from spectra.frontend.main_window import MainWindow
        assert MainWindow._is_audio("song.wav") is True

    def test_ogg(self, app):
        from spectra.frontend.main_window import MainWindow
        assert MainWindow._is_audio("song.ogg") is True

    def test_flac(self, app):
        from spectra.frontend.main_window import MainWindow
        assert MainWindow._is_audio("song.flac") is True

    def test_txt_rejected(self, app):
        from spectra.frontend.main_window import MainWindow
        assert MainWindow._is_audio("readme.txt") is False

    def test_json_rejected(self, app):
        from spectra.frontend.main_window import MainWindow
        assert MainWindow._is_audio("config.json") is False

    def test_png_rejected(self, app):
        from spectra.frontend.main_window import MainWindow
        assert MainWindow._is_audio("photo.png") is False


class TestDragDrop:
    """Test drag-and-drop file handling."""

    def test_open_file_accepted_audio(self, app, monkeypatch):
        """不能真的弹模态对话框（会挂住）——把 getOpenFileName 打桩成返回一个音频路径。"""
        from spectra.frontend.main_window import MainWindow
        monkeypatch.setattr(
            QFileDialog, "getOpenFileName",
            staticmethod(lambda parent, title, directory, name_filter: ("/tmp/song.mp3", "Audio Files (*.mp3)")),
        )
        _build_main_window(app)
        result = QFileDialog.getOpenFileName(None, "Open Audio", "", "Audio Files (*.mp3 *.wav *.ogg *.flac)")
        assert isinstance(result, tuple) and len(result) == 2
        assert MainWindow._is_audio(result[0]), result[0]

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
        from spectra.frontend.main_window import MainWindow
        assert MainWindow._is_audio("readme.txt") is False
        assert MainWindow._is_audio("data.bin") is False
        assert MainWindow._is_audio("photo.jpeg") is False

    def test_is_audio_accepts_audio(self, app):
        from spectra.frontend.main_window import MainWindow
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
    """Round-4 visual upgrade: generic open_btn, inset well, floor==canvas, brand logo."""

    def test_open_btn_uses_generic_style(self, mw):
        assert mw._open_btn.objectName() == ""

    def test_spec_well_inset(self, mw):
        from spectra.frontend.styles import BG_CANVAS
        well = mw._spec.parentWidget()
        assert well is not None
        assert well.objectName() == "spec_well"
        assert BG_CANVAS in well.styleSheet()

    def test_spectra_floor_matches_canvas(self):
        from spectra.backend.palette import build_lut_np
        lut = build_lut_np("spectra")
        assert tuple(lut[0][:3]) == (0, 0, 0)      # == BG_CANVAS #000000

    def test_brand_logo_loaded(self, mw):
        assert mw._logo_ok is True
        assert mw._brand_label.pixmap() is not None


# ── 播放声道路径 ───────────────────────────────────────────────────────

class _NullNode:
    """占位对象：属性访问返回自身，调用返回 None（用来顶掉 worker）。"""

    def __call__(self, *args, **kwargs):
        return None

    def connect(self, *args, **kwargs):
        return None

    def __getattr__(self, name):
        return self


class _Recorder:
    """顶掉 PlaybackEngine / WaveformWidget，只记录收到的数据形状。"""

    def __init__(self):
        self.audio = None
        self.sample_rate = None

    def load(self, audio, sample_rate):
        self.audio = np.asarray(audio)
        self.sample_rate = sample_rate

    def set_audio(self, audio, sample_rate, duration):
        self.audio = np.asarray(audio)
        self.sample_rate = sample_rate

    def __getattr__(self, name):
        return _NullNode()


class _FakeLoadWorker:
    """顶掉 _load_worker：_on_load_done 的 sender() 为空时回落到它。"""

    def __init__(self, analyzer):
        self.analyzer = analyzer

    def isRunning(self):
        return False

    def __getattr__(self, name):
        return _NullNode()


def _synth_stereo_analyzer():
    from pathlib import Path
    from spectra.backend.core import AudioAnalyzer

    a = AudioAnalyzer()
    a.data = np.vstack([np.array([0.4, 0.1, 0.2, -0.3], np.float32),
                        np.array([-0.4, 0.2, 0.3, 0.5], np.float32)])
    a.sample_rate = 48000
    a.channels = 2
    a.duration = a.data.shape[1] / a.sample_rate
    a.filepath = Path("synth.wav")
    a.metadata = {}
    return a


class TestPlaybackChannelPath:
    """播放必须收到真立体声（analyzer.data），波形显示仍走 mixdown。"""

    def test_playback_receives_stereo(self, app, monkeypatch):

        from spectra.frontend import main_window as mwmod

        mw = _build_main_window(app)
        play_rec = _Recorder()
        wave_rec = _Recorder()
        monkeypatch.setattr(mw, "_playback", play_rec)
        monkeypatch.setattr(mw._wave, "set_audio", wave_rec.set_audio)
        monkeypatch.setattr(mw._meta, "load_metadata", lambda analyzer: None)
        monkeypatch.setattr(mwmod, "_QualityWorker", lambda *a, **k: _NullNode())
        monkeypatch.setattr(mwmod, "_SpectrumWorker", lambda *a, **k: _NullNode())

        analyzer = _synth_stereo_analyzer()
        monkeypatch.setattr(mw, "_load_worker", _FakeLoadWorker(analyzer))
        mw._on_load_done()

        assert play_rec.audio.shape == (2, analyzer.data.shape[1])
        assert not np.allclose(play_rec.audio[0], play_rec.audio[1]), "播放被折叠成单声道"

    def test_waveform_still_uses_mixdown(self, app, monkeypatch):

        from spectra.frontend import main_window as mwmod

        mw = _build_main_window(app)
        play_rec = _Recorder()
        wave_rec = _Recorder()
        monkeypatch.setattr(mw, "_playback", play_rec)
        monkeypatch.setattr(mw._wave, "set_audio", wave_rec.set_audio)
        monkeypatch.setattr(mw._meta, "load_metadata", lambda analyzer: None)
        monkeypatch.setattr(mwmod, "_QualityWorker", lambda *a, **k: _NullNode())
        monkeypatch.setattr(mwmod, "_SpectrumWorker", lambda *a, **k: _NullNode())

        analyzer = _synth_stereo_analyzer()
        monkeypatch.setattr(mw, "_load_worker", _FakeLoadWorker(analyzer))
        mw._on_load_done()

        assert wave_rec.audio.ndim == 1
        assert np.allclose(wave_rec.audio, np.mean(analyzer.data, axis=0))
