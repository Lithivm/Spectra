"""Algorithm tests for Spectra analyzer — no audio files needed.

Tests clipping detection, RMS, peak, DR, flatten_analysis, and i18n.
Uses synthetic numpy signals exclusively.
"""

import numpy as np
import pytest
from pathlib import Path


# ── is_audio_file ──────────────────────────────────────────────────────

class TestIsAudioFile:
    def test_supported_extensions(self):
        from analyzer.load import is_audio_file
        for ext in [".flac", ".opus", ".wav", ".mp3", ".m4a", ".mp4",
                    ".aac", ".wma", ".ape", ".ogg", ".tta", ".aiff"]:
            assert is_audio_file(f"test{ext}") is True

    def test_unsupported_extensions(self):
        from analyzer.load import is_audio_file
        for name in ["readme.txt", "data.bin", "photo.jpg", "video.mkv", "doc.pdf"]:
            assert is_audio_file(name) is False

    def test_case_insensitive(self):
        from analyzer.load import is_audio_file
        assert is_audio_file("song.MP3") is True
        assert is_audio_file("song.WAV") is True
        assert is_audio_file("song.FlAc") is True


# ── Quality metrics with synthetic signals ─────────────────────────────

def _make_analyzer_with_audio(audio: np.ndarray, sr: int = 48000):
    """Create an AudioAnalyzer pre-loaded with synthetic audio."""
    from analyzer.core import AudioAnalyzer
    a = AudioAnalyzer()
    a.filepath = Path("/fake/test.wav")
    a.sample_rate = sr
    a.data = audio if audio.ndim > 1 else audio[np.newaxis, :]
    a.duration = float(audio.shape[-1]) / sr
    a.channels = a.data.shape[0]
    return a


class TestRMS:
    def test_sine_rms(self):
        # 1 kHz sine at -6 dBFS → RMS ≈ 0.5 / sqrt(2) ≈ 0.3536
        sr = 48000
        t = np.linspace(0, 1.0, sr, endpoint=False)
        sine = 0.5 * np.sin(2 * np.pi * 1000 * t)
        a = _make_analyzer_with_audio(sine.astype(np.float32))
        rms = a._compute_rms(a.data[0])
        assert abs(rms - 0.3536) < 1e-3

    def test_silence_rms(self):
        sr = 48000
        silence = np.zeros(sr, dtype=np.float32)
        a = _make_analyzer_with_audio(silence)
        rms = a._compute_rms(a.data[0])
        assert rms == 0.0


class TestPeak:
    def test_sine_peak(self):
        sr = 48000
        t = np.linspace(0, 1.0, sr, endpoint=False)
        sine = 0.75 * np.sin(2 * np.pi * 1000 * t)
        a = _make_analyzer_with_audio(sine.astype(np.float32))
        peak, _ = a._compute_peak(a.data[0])
        assert abs(peak - 0.75) < 1e-5

    def test_full_scale_peak(self):
        audio = np.array([0.0, 0.5, 1.0, 0.5, 0.0, -1.0, 0.0], dtype=np.float32)
        a = _make_analyzer_with_audio(audio)
        peak, idx = a._compute_peak(a.data[0])
        assert peak == 1.0
        assert idx in (2, 5)  # either +1.0 or -1.0


class TestClippingDetection:
    def test_clean_signal(self):
        sr = 48000
        t = np.linspace(0, 1.0, sr, endpoint=False)
        sine = 0.5 * np.sin(2 * np.pi * 1000 * t)
        a = _make_analyzer_with_audio(sine.astype(np.float32))
        result = a._detect_clipping(a.data[0], sr)
        assert result["ok"] is True
        assert result["count"] == 0

    def test_hard_clipped_signal(self):
        sr = 48000
        t = np.linspace(0, 0.1, int(0.1 * sr), endpoint=False)
        sine = 1.5 * np.sin(2 * np.pi * 1000 * t)
        clipped = np.clip(sine, -1.0, 1.0)
        a = _make_analyzer_with_audio(clipped.astype(np.float32))
        result = a._detect_clipping(a.data[0], sr)
        assert result["ok"] is False
        assert result["count"] > 0

    def test_silence_no_false_positive(self):
        sr = 48000
        silence = np.zeros(sr, dtype=np.float32)
        a = _make_analyzer_with_audio(silence)
        result = a._detect_clipping(a.data[0], sr)
        assert result["ok"] is True


class TestDynamicRange:
    def test_sine_dr_nonzero(self):
        # Signal with amplitude modulation → DR > 0
        sr = 48000
        t = np.linspace(0, 2.0, 2 * sr, endpoint=False)
        envelope = 0.1 + 0.9 * (0.5 + 0.5 * np.sin(2 * np.pi * 0.5 * t))  # 0.5 Hz AM
        sine = envelope * np.sin(2 * np.pi * 1000 * t)
        a = _make_analyzer_with_audio(sine.astype(np.float32))
        result = a._measure_dynamic_range(a.data[0])
        assert "dr" in result
        assert result["dr"] > 0

    def test_silence_dr_zero(self):
        sr = 48000
        silence = np.zeros(sr, dtype=np.float32)
        a = _make_analyzer_with_audio(silence)
        result = a._measure_dynamic_range(a.data[0])
        assert result["dr"] == 0.0


class TestZeroCrossingRate:
    def test_sine_zcr(self):
        sr = 48000
        t = np.linspace(0, 1.0, sr, endpoint=False)
        sine = np.sin(2 * np.pi * 1000 * t)
        a = _make_analyzer_with_audio(sine.astype(np.float32))
        zcr = a._compute_zcr(a.data[0])
        # 1 kHz sine → ~2000 zero crossings per second
        assert 1900 <= zcr <= 2100


# ── Batch / flatten_analysis ───────────────────────────────────────────

class TestFlattenAnalysis:
    def test_all_columns_present(self):
        from analyzer.batch import flatten_analysis, BATCH_COLUMNS

        md = {"format": "WAV", "duration": 10.0, "sample_rate": 48000,
              "channels": 2, "bitrate": 1536000,
              "标题": "Test", "艺术家": "Artist", "专辑": "Album",
              "年份": "2024", "流派": "Rock", "音轨": "1"}
        qa = {
            "peak_db": -0.5, "rms": 0.3,
            "clipping": {"ok": True, "count": 0, "longest_ms": 0},
            "upsampling": {"ok": True, "cutoff_hz": 22050},
            "dynamic_range": {"dr": 12.5},
            "loudness": {"integrated_lufs": -14.0, "short_term_lufs": -12.0,
                         "lra_lu": 4.0, "true_peak_db": -0.3},
        }

        row = flatten_analysis(md, qa, Path("/fake/test.wav"))

        for col in BATCH_COLUMNS:
            assert col in row, f"Missing column: {col}"

        assert row["filename"] == "test.wav"
        assert row["title"] == "Test"
        assert row["peak_db"] == -0.5
        assert row["dynamic_range_db"] == 12.5

    def test_no_quality_results(self):
        from analyzer.batch import flatten_analysis

        md = {"format": "MP3"}
        row = flatten_analysis(md, None, Path("/fake/test.mp3"))

        assert row["filename"] == "test.mp3"
        assert row["format"] == "MP3"
        # Quality fields should be defaults
        assert row["clipping_count"] == 0


# ── i18n ────────────────────────────────────────────────────────────────

class TestLang:
    def test_t_zh(self):
        from lang import t, LANG
        LANG_old = LANG
        try:
            import lang
            lang.LANG = "zh"
            assert t("中文", "English") == "中文"
        finally:
            lang.LANG = LANG_old

    def test_t_en(self):
        from lang import t, LANG
        LANG_old = LANG
        try:
            import lang
            lang.LANG = "en"
            assert t("中文", "English") == "English"
        finally:
            lang.LANG = LANG_old

    def test_toggle_lang(self):
        from lang import toggle_lang, LANG
        LANG_old = LANG
        try:
            import lang
            lang.LANG = "zh"
            new = toggle_lang()
            assert new == "en"
            new = toggle_lang()
            assert new == "zh"
        finally:
            lang.LANG = LANG_old

    def test_on_lang_change_returns_unsubscribe(self):
        from lang import on_lang_change

        calls = []
        unsub = on_lang_change(lambda lang_code: calls.append(lang_code))
        assert callable(unsub)

        unsub()  # should not raise
        assert len(calls) == 0  # no toggle happened yet


# ── Load module normalization ──────────────────────────────────────────

class TestLoadNormalization:
    def test_supported_extensions_set(self):
        from analyzer.load import SUPPORTED_EXTENSIONS
        assert ".wav" in SUPPORTED_EXTENSIONS
        assert ".flac" in SUPPORTED_EXTENSIONS
        assert ".mp3" in SUPPORTED_EXTENSIONS
        assert ".txt" not in SUPPORTED_EXTENSIONS


# ── AudioAnalyzer basic state ──────────────────────────────────────────

class TestAudioAnalyzerState:
    def test_default_state(self):
        from analyzer.core import AudioAnalyzer
        a = AudioAnalyzer()
        assert a.filepath is None
        assert a.data is None
        assert a.sample_rate == 0
        assert a.duration == 0.0

    def test_waveform_raises_without_data(self):
        from analyzer.core import AudioAnalyzer
        a = AudioAnalyzer()
        with pytest.raises(RuntimeError):
            _ = a.waveform

    def test_analyze_quality_raises_without_data(self):
        from analyzer.core import AudioAnalyzer
        a = AudioAnalyzer()
        with pytest.raises(RuntimeError):
            a.analyze_quality()

    def test_waveform_mono_downmix(self):
        sr = 48000
        stereo = np.random.randn(2, sr).astype(np.float32) * 0.1
        a = _make_analyzer_with_audio(stereo)
        wf = a.waveform
        assert wf.ndim == 1
        assert len(wf) == sr
