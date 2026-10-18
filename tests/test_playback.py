"""播放路径测试 —— 合成信号 + sounddevice 替身，不碰真设备。

覆盖：声道布局、量化（int32 限幅）、直通判定、独占失败回退、切换时保住位置。
"""

import numpy as np
import pytest
from types import SimpleNamespace

import spectra.frontend.playback_engine as pe
from spectra.backend.output_devices import enumerate_channels
from spectra.backend.convert import (
    prepare_channels, quantize, INT32_MIN, INT32_MAX, INT16_MIN, INT16_MAX,
)


# ── 假设备表（照本机形状摆）────────────────────────────────────────────

DEVICES = [
    {"index": 0, "name": "MME default out", "hostapi": 0,
     "max_output_channels": 2, "max_input_channels": 0, "default_samplerate": 44100},
    {"index": 1, "name": "Mic only", "hostapi": 0,
     "max_output_channels": 0, "max_input_channels": 1, "default_samplerate": 44100},
    {"index": 2, "name": "YULONG DAC", "hostapi": 1,
     "max_output_channels": 2, "max_input_channels": 0, "default_samplerate": 88200},
]
HOSTAPIS = [
    {"name": "MME", "default_output_device": 0},
    {"name": "Windows WASAPI", "default_output_device": 2},
]
DEFAULT_OUTPUT = 0


def entries():
    return enumerate_channels(DEVICES, HOSTAPIS, DEFAULT_OUTPUT)


def pick(name: str, mode: str):
    for e in entries():
        if e.device_name == name and e.mode == mode:
            return e
    raise KeyError(name, mode)


# ── sounddevice 替身 ────────────────────────────────────────────────────

class FakeStream:
    def __init__(self, record):
        self.actual_samplerate = record["samplerate"]
        self.time = 0.0
        self.record = record

    def start(self):
        pass

    def stop(self):
        pass

    def close(self):
        pass


class FakeSD:
    """只实现引擎用到的那几个接口；fail(rec) 决定哪条路径报错。"""

    def __init__(self, fail=None, devices=None, hostapis=None, default_output=None):
        self._fail = fail or (lambda rec: False)
        self.opens = []
        self._devices = devices or DEVICES
        self._hostapis = hostapis or HOSTAPIS
        self._default_output = DEFAULT_OUTPUT if default_output is None else default_output
        self.default = SimpleNamespace(device=(-1, self._default_output))

    def query_devices(self, idx=None):
        if idx is None:
            return self._devices
        return next(d for d in self._devices if d["index"] == int(idx))

    def query_hostapis(self):
        return self._hostapis

    def WasapiSettings(self, exclusive=False):
        return SimpleNamespace(exclusive=bool(exclusive))

    def OutputStream(self, **kwargs):
        rec = {
            "device": kwargs.get("device"),
            "exclusive": bool(getattr(kwargs.get("extra_settings"), "exclusive", False)),
            "dtype": kwargs.get("dtype"),
            "samplerate": kwargs.get("samplerate"),
            "channels": kwargs.get("channels"),
        }
        self.opens.append(rec)
        if self._fail(rec):
            raise RuntimeError("PaErrorCode -9984 (mocked)")
        return FakeStream(rec)

    CallbackStop = RuntimeError
    CallbackFlags = object


def fake_engine(monkeypatch, fail=None, devices=None, hostapis=None, default_output=None):
    fake = FakeSD(fail, devices, hostapis, default_output)
    monkeypatch.setattr(pe, "sd", fake)
    return pe.PlaybackEngine(), fake


STEREO = np.vstack([np.array([0.4, 0.1, 0.2, -0.3], np.float32),
                    np.array([-0.4, 0.2, 0.3, 0.5], np.float32)])


# ── prepare_channels ────────────────────────────────────────────────────

class TestPrepareChannels:
    def test_stereo_kept_and_not_meaned(self):
        out = prepare_channels(STEREO, 2)
        assert out.shape == (2, 4)
        assert np.array_equal(out, STEREO)
        assert not np.allclose(out[0], out[1])

    def test_mono_duplicated(self):
        mono = np.array([0.5, -0.25], dtype=np.float32)
        out = prepare_channels(mono, 2)
        assert out.shape == (2, 2)
        assert np.allclose(out[0], mono) and np.allclose(out[1], mono)

    def test_pads_when_device_has_more_channels(self):
        out = prepare_channels(np.array([[0.3, 0.4]], dtype=np.float32), 2)
        assert out.shape == (2, 2)
        assert np.allclose(out[1], out[0])

    def test_folds_when_source_exceeds_device(self):
        """5.1 接立体声 DAC：平均折叠，不能截前 N 路（那会静默丢后环绕）。"""
        src = np.array([[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]], dtype=np.float32)
        out = prepare_channels(src, 2)
        assert np.allclose(out[0], [0.5, 0.5]) and np.allclose(out[1], [0.5, 0.5])

    def test_rejects_zero_target(self):
        with pytest.raises(ValueError):
            prepare_channels(np.array([0.1], dtype=np.float32), 0)


# ── 引擎声道 ────────────────────────────────────────────────────────────

class TestEngineChannels:
    def test_stereo_kept_two_channels(self, monkeypatch):
        eng, _ = fake_engine(monkeypatch)
        eng.load(STEREO, 48000)
        assert eng.channels == 2
        assert eng._audio.shape == (2, 4)
        assert not np.allclose(eng._audio[0], eng._audio[1])

    def test_mono_becomes_two_channels(self, monkeypatch):
        eng, _ = fake_engine(monkeypatch)
        eng.load(np.array([0.5, -0.25], dtype=np.float32), 48000)
        assert eng.channels == 2
        assert np.allclose(eng._audio[0], eng._audio[1])

    def test_caps_to_device_channels(self, monkeypatch):
        """5.1 源接到单声道 DAC：封顶后平均折叠，不能静默丢后面的声道。"""
        mono_table = [dict(d, max_output_channels=1) for d in DEVICES]
        eng, _ = fake_engine(monkeypatch, devices=mono_table)
        eng.load(np.array([[0.2, 0.2], [0.6, 0.6]], dtype=np.float32), 44100)
        assert eng.channels == 1
        assert np.allclose(eng._audio[0], [0.4, 0.4])


# ── 量化 ────────────────────────────────────────────────────────────────

class TestQuantize:
    @pytest.mark.filterwarnings("ignore::RuntimeWarning")
    def test_naive_cast_wraps(self):
        """记下这个 bug 的成因：1.0 × 2³¹ 直接 cast 会绕回 INT32_MIN。"""
        assert (np.array([1.0], dtype=np.float64) * 2.0 ** 31).astype(np.int32)[0] == INT32_MIN

    def test_int32_saturates_instead_of_wrapping(self):
        out = quantize(np.array([1.0, 1.2, -1.2, -1.0], dtype=np.float32), "int32")
        assert list(out) == [INT32_MAX, INT32_MAX, INT32_MIN, INT32_MIN]

    def test_int16_saturates(self):
        out = quantize(np.array([1.0, 1.3, -1.3], dtype=np.float32), "int16")
        assert list(out) == [INT16_MAX, INT16_MAX, INT16_MIN]

    def test_int24_source_to_int32_is_exact(self):
        """24-bit 源经 int32 目标不丢位：v/2²³ × 2³¹ = v·256，float64 下精确 → 不需要 dither。"""
        v = np.arange(-8000000, 8000001, 4096, dtype=np.int64)
        audio = (v / 2.0 ** 23).astype(np.float32)
        assert np.array_equal(quantize(audio, "int32"), (v * 256).astype(np.int32))

    def test_float32_unchanged(self):
        audio = np.array([0.25, -0.75], dtype=np.float32)
        assert np.array_equal(quantize(audio, "float32"), audio)

    def test_rejects_unknown_dtype(self):
        with pytest.raises(ValueError):
            quantize(np.array([0.1], dtype=np.float32), "int24")

    def test_callback_writes_quantized_data(self, monkeypatch):
        """int32 路径：回调写进 outdata 的必须已经是限幅后的整数，不能再绕回。"""
        eng, _ = fake_engine(monkeypatch)
        src = np.array([[1.2, -1.2]], dtype=np.float32)
        eng._audio = quantize(src, "int32")
        eng._source_audio = src
        eng._source_sr = 44100
        eng._total_frames = 2
        eng._sample_rate = 44100
        eng.channels = 1
        outdata = np.zeros((4, 1), dtype=np.int32)
        with pytest.raises(RuntimeError):        # 缓冲放完 → CallbackStop
            eng._callback(outdata, 4, None, 0)
        assert list(outdata[0]) == [INT32_MAX]
        assert list(outdata[1]) == [INT32_MIN]
        assert list(outdata[2:]) == [[0], [0]]

# ── 直通 / 回退 / 切换保住位置 ──────────────────────────────────────────

LONG = np.vstack([np.linspace(0.2, 1.2, 132300, dtype=np.float32),
                  np.linspace(-1.2, -0.2, 132300, dtype=np.float32)])   # 3 s @ 44100，顶到满刻度


class TestPassthroughAndFallback:
    def test_exclusive_requests_source_rate(self, monkeypatch):
        eng, fake = fake_engine(monkeypatch)
        eng.load(LONG, 44100)
        cap = eng.select_output(pick("YULONG DAC", "exclusive"))
        assert cap.ok and cap.channel.mode == "exclusive"
        assert eng.info.passthrough is True
        assert eng._sample_rate == 44100              # 直通 = 没重采样
        assert eng._total_frames == LONG.shape[1]
        assert fake.opens[-1]["exclusive"] is True
        assert fake.opens[-1]["samplerate"] == 44100

    def test_shared_uses_mix_format_rate_and_src(self, monkeypatch):
        eng, fake = fake_engine(monkeypatch)
        eng.load(LONG, 44100)
        cap = eng.select_output(pick("YULONG DAC", "shared"))
        assert cap.ok
        assert eng._sample_rate == 88200              # 共享必用 mix format 率
        assert eng._total_frames == LONG.shape[1] * 2  # 44100 → 88200 翻倍
        assert eng.info.passthrough is False
        assert fake.opens[-1]["exclusive"] is False

    def test_exclusive_prefers_float32(self, monkeypatch):
        eng, fake = fake_engine(monkeypatch)
        eng.load(LONG, 44100)
        eng.select_output(pick("YULONG DAC", "exclusive"))
        assert fake.opens[-1]["dtype"] == "float32"
        assert eng._audio.dtype == np.float32

    def test_dtype_falls_back_to_int32_when_float32_refused(self, monkeypatch):
        eng, fake = fake_engine(monkeypatch,
                                fail=lambda rec: rec["exclusive"] and rec["dtype"] == "float32")
        eng.load(LONG, 44100)
        cap = eng.select_output(pick("YULONG DAC", "exclusive"))
        assert cap.ok and cap.dtype == "int32"
        assert eng._audio.dtype == np.int32
        # 没绕回：正声道全正、负声道全负（1.2 顶到满刻度而不是翻成负数）
        assert int(eng._audio[0].max()) > 0 and int(eng._audio[1].min()) < 0

    def test_exclusive_failure_falls_back_to_shared(self, monkeypatch):
        """独占打不开 → 自动回退共享（同设备），并如实记录不是直通。"""
        eng, fake = fake_engine(monkeypatch, fail=lambda rec: rec["exclusive"])
        eng.load(LONG, 44100)
        target = pick("YULONG DAC", "exclusive")
        cap = eng.select_output(target)
        assert cap.ok
        assert cap.channel.mode == "shared"
        assert cap.fallback_from == target
        assert eng.info.channel.mode == "shared"       # UI 显示的是实际生效的那条
        assert eng.info.passthrough is False           # 回退后必然 SRC
        assert eng._sample_rate == 88200

    def test_all_paths_fail_reports_error(self, monkeypatch):
        eng, fake = fake_engine(monkeypatch, fail=lambda rec: True)
        eng.load(LONG, 44100)
        cap = eng.select_output(pick("YULONG DAC", "exclusive"))
        assert cap.ok is False
        assert cap.error
        assert eng.state == "stopped"

    def test_probe_cached_and_only_once_per_channel(self, monkeypatch):
        eng, fake = fake_engine(monkeypatch)
        eng.load(LONG, 44100)
        target = pick("YULONG DAC", "exclusive")
        eng.capability(target)
        first = len(fake.opens)
        eng.capability(target)
        assert len(fake.opens) == first                 # 每个通道只探一次

    def test_position_survives_channel_switch_while_stopped(self, monkeypatch):
        eng, fake = fake_engine(monkeypatch)
        eng.load(LONG, 44100)
        eng.track_position(2.0)
        assert eng.get_position() == pytest.approx(2.0)
        eng.select_output(pick("YULONG DAC", "shared"))
        assert eng._sample_rate == 88200
        assert eng.get_position() == pytest.approx(2.0, abs=1 / 88200)

    def test_position_survives_channel_switch_while_playing(self, monkeypatch):
        eng, fake = fake_engine(monkeypatch)
        eng.load(LONG, 44100)
        eng.play()
        assert eng.state == "playing"
        eng.track_position(1.5)
        eng.select_output(pick("YULONG DAC", "shared"))
        assert eng.state == "playing"                   # 关流再开流，仍在放
        assert eng.get_position() == pytest.approx(1.5, abs=1 / 88200)
        last = fake.opens[-1]
        assert last["device"] == 2 and last["exclusive"] is False
