"""播放路径测试 —— 合成信号 + sounddevice 打桩，不碰真设备。

声道布局（prepare_channels）与引擎接收的数据形状；量化部分见后面的 dtype 测试。
"""

import numpy as np
import pytest
from types import SimpleNamespace

import spectra.frontend.playback_engine as pe
from spectra.backend.convert import prepare_channels


# ── sounddevice 替身 ────────────────────────────────────────────────────

class FakeSD:
    """只提供枚举接口 —— 测试不调用 OutputStream，所以不会碰真设备。"""

    def __init__(self, devices, hostapis, default_output):
        self._devices = devices
        self._hostapis = hostapis
        self.default = SimpleNamespace(device=(-1, default_output))

    def query_devices(self, idx=None):
        if idx is None:
            return self._devices
        return self._devices[int(idx)]

    def query_hostapis(self):
        return self._hostapis


def _fake_sd(channels: int = 2, sr: int = 48000) -> FakeSD:
    devices = [
        {
            "index": 0,
            "name": "Fake MME Speakers",
            "hostapi": 0,
            "max_output_channels": channels,
            "default_samplerate": sr,
        },
        {
            "index": 1,
            "name": "Fake WASAPI DAC",
            "hostapi": 1,
            "max_output_channels": channels,
            "default_samplerate": sr,
        },
    ]
    hostapis = [
        {"name": "MME", "default_output_device": 0},
        {"name": "Windows WASAPI", "default_output_device": 1},
    ]
    return FakeSD(devices, hostapis, 1)


# ── prepare_channels ────────────────────────────────────────────────────

class TestPrepareChannels:
    def test_stereo_not_meaned(self):
        left = np.array([0.4, 0.1], dtype=np.float32)
        right = np.array([-0.4, 0.2], dtype=np.float32)
        out = prepare_channels(np.vstack([left, right]), 2)
        assert out.shape == (2, 2)
        assert np.array_equal(out[0], left)
        assert np.array_equal(out[1], right)
        # 关键：左右不同 → 没有被求和折叠
        assert not np.allclose(out[0], out[1])

    def test_mono_duplicated(self):
        mono = np.array([0.5, -0.25], dtype=np.float32)
        out = prepare_channels(mono, 2)
        assert out.shape == (2, 2)
        assert np.allclose(out[0], mono)
        assert np.allclose(out[1], mono)

    def test_pads_when_device_has_more_channels(self):
        out = prepare_channels(np.array([[0.3, 0.4]], dtype=np.float32), 2)
        assert out.shape == (2, 2)
        assert np.allclose(out[1], out[0])

    def test_folds_when_source_exceeds_device(self):
        """5.1 接立体声 DAC：平均折叠，不能截前 N 路（那会静默丢后环绕）。"""
        src = np.array([[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]], dtype=np.float32)
        out = prepare_channels(src, 2)
        assert out.shape == (2, 2)
        assert np.allclose(out[0], [0.5, 0.5])
        assert np.allclose(out[1], [0.5, 0.5])

    def test_rejects_zero_target(self):
        with pytest.raises(ValueError):
            prepare_channels(np.array([0.1], dtype=np.float32), 0)


# ── 引擎接收的数据 ──────────────────────────────────────────────────────

class TestEngineChannels:
    def test_stereo_kept_two_channels(self, monkeypatch):
        monkeypatch.setattr(pe, "sd", _fake_sd(channels=2))
        eng = pe.PlaybackEngine()
        eng.load(np.vstack([np.array([0.4, 0.1], np.float32),
                            np.array([-0.4, 0.2], np.float32)]), 48000)
        assert eng.channels == 2
        assert eng._audio.shape == (2, 2)
        assert not np.allclose(eng._audio[0], eng._audio[1])

    def test_mono_becomes_two_channels(self, monkeypatch):
        monkeypatch.setattr(pe, "sd", _fake_sd(channels=2))
        eng = pe.PlaybackEngine()
        eng.load(np.array([0.5, -0.25], dtype=np.float32), 48000)
        assert eng.channels == 2
        assert np.allclose(eng._audio[0], eng._audio[1])

    def test_engine_caps_to_device_channels(self, monkeypatch):
        monkeypatch.setattr(pe, "sd", _fake_sd(channels=1))
        eng = pe.PlaybackEngine()
        eng.load(np.vstack([np.array([0.2, 0.2], np.float32),
                            np.array([0.6, 0.6], np.float32)]), 48000)
        assert eng.channels == 1
        assert eng._audio.shape == (1, 2)
