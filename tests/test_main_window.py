import sys

import numpy as np
import pytest
from PyQt6.QtWidgets import QApplication, QFileDialog, QLabel


# ── helpers ──────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def app():
    app = QApplication.instance() or QApplication(sys.argv)
    yield app


class FakeSettings:
    """QSettings 替身：只用到 value/setValue。"""

    def __init__(self, initial=None):
        self.data = dict(initial or {})

    def value(self, key, default=None):
        return self.data.get(key, default)

    def setValue(self, key, val):
        self.data[key] = val


from spectra.backend.output_devices import (
    ChannelCapability, StreamInfo, enumerate_channels,
)

# 假设备表：MME / DirectSound / 单输入设备都不该出现在列表里
_TEST_DEVICES = [
    {"index": 0, "name": "MME default out", "hostapi": 0,
     "max_output_channels": 2, "max_input_channels": 0, "default_samplerate": 44100},
    {"index": 1, "name": "Mic only", "hostapi": 0,
     "max_output_channels": 0, "max_input_channels": 1, "default_samplerate": 44100},
    {"index": 2, "name": "DAC B", "hostapi": 1,
     "max_output_channels": 2, "max_input_channels": 0, "default_samplerate": 88200},
]
_TEST_HOSTAPIS = [{"name": "MME", "default_output_device": 0},
                  {"name": "Windows WASAPI", "default_output_device": 2}]
ENTRIES = enumerate_channels(_TEST_DEVICES, _TEST_HOSTAPIS, 0)


class _NullNode:
    def __call__(self, *args, **kwargs):
        return None

    def connect(self, *args, **kwargs):
        return None

    def __getattr__(self, name):
        return self


class FakeEngine:
    """PlaybackEngine 替身：capability() 只在缓存未命中时算一次探测。"""

    def __init__(self, entries=None, caps=None):
        self.entries = list(entries or ENTRIES)
        self._caps = dict(caps or {})
        self.probes = 0
        self.selected_key = None
        self.info = None
        self.state = "stopped"
        self.is_playing = False
        self.duration = 1.0
        self.state_changed = _NullNode()
        self.output_ready = _NullNode()

    def available_channels(self):
        return tuple(self.entries)

    def capability(self, ch):
        if ch.key not in self._caps:
            self.probes += 1
            self._caps[ch.key] = ChannelCapability(channel=ch, ok=True, dtype="float32")
        return self._caps[ch.key]

    def cached_capability(self, ch):
        return self._caps.get(ch.key)

    def _publish(self, ch, cap):
        self.selected_key = ch.key
        self.info = StreamInfo(channel=ch, dtype=cap.dtype, source_sr=0,
                               requested_sr=ch.mix_sr, actual_sr=ch.mix_sr,
                               fallback_from=cap.fallback_from)

    def use_channel(self, ch):
        self._publish(ch, ChannelCapability(channel=ch, ok=True, dtype="float32", probed=False))

    def select_output(self, ch):
        cap = self.capability(ch)
        self._publish(cap.channel, cap)
        return cap

    def get_position(self):
        return 0.0

    def __getattr__(self, name):
        return _NullNode()


class _SyncProbeWorker:
    """在调用线程里直接跑完的探测（事件循环不可靠时测试更好断言）。"""

    def __init__(self, engine, channel):
        self._engine = engine
        self._channel = channel
        self._slot = None
        self.ready = _NullNode()
        self.ready.connect = lambda slot: setattr(self, "_slot", slot)

    def start(self):
        self._slot(self._engine.capability(self._channel))


def _build_main_window(app: QApplication, settings=None, engine=None, show: bool = True):
    from spectra.frontend.main_window import MainWindow
    mw = MainWindow(settings=settings if settings is not None else FakeSettings(),
                    playback=engine if engine is not None else FakeEngine())
    mw.resize(800, 600)
    if show:
        # 显示会触发 GL widget 在 offscreen 平台上的 paint（历史上有 flaky），
        # 只测窗口/状态栏逻辑的用例不 show。
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
    """顶掉 PlaybackEngine / WaveformWidget，只记录收到的数据形状。

    位置查询给固定值，否则 _on_playback_tick 会在事件循环里算 pos/dur 炸掉。
    """

    def __init__(self):
        self.audio = None
        self.sample_rate = None
        self.state = "stopped"
        self.is_playing = False
        self.duration = 1.0

    def load(self, audio, sample_rate):
        self.audio = np.asarray(audio)
        self.sample_rate = sample_rate

    def set_audio(self, audio, sample_rate, duration):
        self.audio = np.asarray(audio)
        self.sample_rate = sample_rate

    def get_position(self):
        return 0.0

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


# ── 输出通道：展开列表 + 状态栏 + 持久化 + 启动探测一次 ──────────────────

def by_mode(mode):
    return next(e for e in ENTRIES if e.mode == mode)


class TestOutputSelection:
    def _window(self, app, monkeypatch, settings=None, engine=None):
        from spectra.frontend import main_window as mwmod
        monkeypatch.setattr(mwmod, "_ProbeWorker", _SyncProbeWorker)
        eng = engine or FakeEngine()
        mw = _build_main_window(app, settings=settings or FakeSettings(), engine=eng, show=False)
        return mw, eng

    def test_list_has_system_shared_exclusive(self, app, monkeypatch):
        mw, eng = self._window(app, monkeypatch)
        modes = sorted(a.data().mode for a in mw._output_actions.values())
        assert modes == ["exclusive", "shared", "system"]

    def test_current_selection_checked(self, app, monkeypatch):
        target = by_mode("exclusive")
        settings = FakeSettings({"output_channel": target.key})
        mw, eng = self._window(app, monkeypatch, settings)
        assert mw._current_output_channel().key == target.key
        assert mw._output_actions[target.key].isChecked()

    def test_missing_device_falls_back_to_system(self, app, monkeypatch):
        settings = FakeSettings({"output_channel": "Windows WASAPI|拔掉的 DAC|exclusive"})
        mw, eng = self._window(app, monkeypatch, settings)
        assert mw._current_output_channel().mode == "system"

    def test_startup_probes_exactly_once(self, app, monkeypatch):
        mw, eng = self._window(app, monkeypatch)
        assert eng.probes == 1                       # 启动自动探测一次，不多不少

    def test_probe_cached_per_channel(self, app, monkeypatch):
        mw, eng = self._window(app, monkeypatch)
        target = by_mode("exclusive")
        mw._choose_output(target)
        first = eng.probes
        assert first == 2                             # 新通道探一次
        mw._choose_output(target)
        assert eng.probes == first                    # 命中缓存不再探

    def test_selection_persisted_by_name_not_index(self, app, monkeypatch):
        mw, eng = self._window(app, monkeypatch)
        target = by_mode("exclusive")
        mw._choose_output(target)
        stored = mw._settings.data["output_channel"]
        assert stored == target.key
        # 重启后靠 host API 名 + 设备名找回，不是靠索引（拔插设备索引会重排）
        assert target.device_name in stored and target.mode in stored

    def test_status_bar_full_when_wide(self, app, monkeypatch):
        target = by_mode("exclusive")
        mw, eng = self._window(app, monkeypatch, FakeSettings({"output_channel": target.key}))
        mw.resize(900, 600)
        app.processEvents()
        assert mw._output_label.text() == target.status_text(full=True, tr=mw._tr)
        assert target.device_name in mw._output_label.text()

    def test_status_bar_narrow_keeps_exclusive_marker(self, app, monkeypatch):
        """缩到只剩协议时，独占的「独占」必须留住 —— 它意味着静音其他应用。"""
        target = by_mode("exclusive")
        mw, eng = self._window(app, monkeypatch, FakeSettings({"output_channel": target.key}))
        # 窗口窄到装不下「协议 · 独占 · 设备名」→ 设备名被挤掉，但「独占」必须留住
        # （resizeEvent 里做的事就是这一句，隐藏窗口不触发 resizeEvent，所以直接调）
        mw.resize(170, 300)
        mw._render_output_text()
        text = mw._output_label.text()
        assert target.device_name not in text
        assert "独占" in text

    def test_fallback_reported_not_hidden(self, app, monkeypatch):
        exclusive = by_mode("exclusive")
        shared = by_mode("shared")
        cap = ChannelCapability(channel=shared, ok=True, dtype="float32", fallback_from=exclusive)
        eng = FakeEngine(caps={exclusive.key: cap, shared.key: cap})
        mw, eng = self._window(app, monkeypatch, engine=eng)
        mw._choose_output(exclusive)
        assert mw._current_output_channel().key == shared.key    # 实际生效的是共享
        assert "回退" in mw._output_label.text()

    def test_unavailable_channel_marked_disabled(self, app, monkeypatch):
        exclusive = by_mode("exclusive")
        bad = ChannelCapability(channel=exclusive, ok=False, dtype="float32", error="device busy")
        eng = FakeEngine(caps={exclusive.key: bad})
        mw, eng = self._window(app, monkeypatch, engine=eng)
        assert mw._output_actions[exclusive.key].isEnabled() is False
        assert "不可用" in mw._output_actions[exclusive.key].text()
