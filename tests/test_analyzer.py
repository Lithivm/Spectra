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
        result = a._detect_clipping(a.data, sr, a._source_format)
        assert result["ok"] is True
        assert result["count"] == 0

    def test_hard_clipped_signal(self):
        sr = 48000
        t = np.linspace(0, 0.1, int(0.1 * sr), endpoint=False)
        sine = 1.5 * np.sin(2 * np.pi * 1000 * t)
        clipped = np.clip(sine, -1.0, 1.0)
        a = _make_analyzer_with_audio(clipped.astype(np.float32))
        result = a._detect_clipping(a.data, sr, a._source_format)
        assert result["ok"] is False
        assert result["count"] > 0

    def test_silence_no_false_positive(self):
        sr = 48000
        silence = np.zeros(sr, dtype=np.float32)
        a = _make_analyzer_with_audio(silence)
        result = a._detect_clipping(a.data, sr, a._source_format)
        assert result["ok"] is True


class TestDynamicRange:
    """F4: DR = P95-P10 of R128 short-term loudness (LU), same family as LRA.

    Needs >= ~9s of audio for 3 x 3s short-term blocks; shorter → DR == 0.
    """

    def _dr(self, x):
        a = _make_analyzer_with_audio(np.asarray(x, dtype=np.float32))
        st = a._short_term_loudness_values(a.data.T, a.sample_rate)
        return a._measure_dynamic_range(st), a

    @staticmethod
    def _alternating_loudness(sr=48000, secs=12):
        # 4 x 3s sections alternating quiet/loud → short-term loudness swings.
        n = sr * secs
        t = np.arange(n) / sr
        tone = np.sin(2 * np.pi * 1000 * t)
        x = np.zeros(n)
        for i in range(secs // 3):
            amp = 0.05 if i % 2 == 0 else 0.8
            x[i * sr * 3:(i + 1) * sr * 3] = amp * tone[i * sr * 3:(i + 1) * sr * 3]
        return x

    def test_varying_loudness_dr_positive(self):
        res, _ = self._dr(self._alternating_loudness())
        assert "dr" in res
        assert res["dr"] > 0

    def test_steady_tone_dr_near_zero(self):
        sr = 48000
        n = sr * 12
        t = np.arange(n) / sr
        x = 0.5 * np.sin(2 * np.pi * 440 * t)  # constant loudness
        res, _ = self._dr(x)
        assert res["dr"] < 1.0

    def test_silence_dr_zero(self):
        x = np.zeros(48000 * 3, dtype=np.float32)  # -inf blocks → filtered → 0
        res, _ = self._dr(x)
        assert res["dr"] == 0.0

    def test_very_short_audio_has_no_loudness_range(self):
        a = _make_analyzer_with_audio(np.zeros(100, dtype=np.float32))
        assert a._short_term_loudness_values(a.data.T, a.sample_rate) == []
        assert a._measure_dynamic_range([]) == {"dr": 0.0}

    def test_single_sample_quality_analysis_is_safe(self):
        a = _make_analyzer_with_audio(np.zeros(1, dtype=np.float32))
        result = a.analyze_quality()
        assert result["dynamic_range"] == {"dr": 0.0}
        assert result["loudness"]["integrated_lufs"] == -np.inf

    def test_dr_equals_lra(self):
        # F4 (Option A): DR and LRA are the same R128 measurement.
        a = _make_analyzer_with_audio(self._alternating_loudness().astype(np.float32))
        q = a.analyze_quality()
        assert q["dynamic_range"]["dr"] == q["loudness"]["lra_lu"]


class TestHighFreqCutoff:
    """F3: cutoff_hz must land on the true content edge — not a floor-region
    PSD spike. ok=False means 'content bandwidth ceiling below Nyquist'.

    Signals are band-limited broadband content (FFT-masked → exact edge at fc)
    over a low full-band noise floor, kept SHORT (1 s) so the Welch PSD has few
    averages and high bin-to-bin variance — the condition under which the pre-fix
    high→low walk stopped at a random floor spike (cutoff 5.8k–23.9k instead of
    fc). Fixed seeds keep every case deterministic.
    """

    @staticmethod
    def _content_floor(fc=None, floor_db=-18.0, sr=48000, dur=1.0, seed=0):
        rng = np.random.default_rng(seed)
        n = int(sr * dur)
        x = rng.standard_normal(n)
        X = np.fft.rfft(x)
        if fc is not None:
            f = np.fft.rfftfreq(n, 1.0 / sr)
            X[f > fc] = 0
        content = np.fft.irfft(X, n=n)
        floor = rng.standard_normal(n) * 10 ** (floor_db / 20)
        return (content + floor).astype(np.float32)

    @staticmethod
    def _gentle_lp(fc=15000, order=4, sr=48000, dur=1.0, seed=0):
        import scipy.signal as sig
        rng = np.random.default_rng(seed)
        x = rng.standard_normal(int(sr * dur))
        b, a = sig.butter(order, fc / (sr / 2), "low")
        return sig.lfilter(b, a, x).astype(np.float32)

    def _run(self, x):
        a = _make_analyzer_with_audio(np.asarray(x, dtype=np.float32))
        return a._detect_high_freq_cutoff(a.data[0], a.sample_rate)

    def test_full_band_ok(self):
        r = self._run(self._content_floor(fc=None))  # white noise → flat to Nyquist
        assert r["ok"] is True
        assert r["cutoff_hz"] >= 0.85 * 24000

    def test_sharp_brickwall_15k(self):
        r = self._run(self._content_floor(fc=15000))
        assert r["ok"] is False
        assert abs(r["cutoff_hz"] - 15000) <= 1800

    def test_phone_band_3k4(self):
        # pre-fix: ~15768 (floor-region spike); must land near the true 3.4k edge.
        r = self._run(self._content_floor(fc=3400))
        assert r["ok"] is False
        assert abs(r["cutoff_hz"] - 3400) <= 900

    def test_bass_only_5k(self):
        # pre-fix: ~17666 (floor-region spike); must land near the true 5k edge.
        r = self._run(self._content_floor(fc=5000))
        assert r["ok"] is False
        assert abs(r["cutoff_hz"] - 5000) <= 1200

    def test_gentle_rolloff_not_flagged(self):
        # A gradual 4th-order rolloff is not an artificial hard edge.
        r = self._run(self._gentle_lp(fc=15000, order=4))
        assert r["ok"] is True


class TestReassignedSpectrogram:
    def test_heuristic_mode_returns_finite_spectrogram(self):
        sr = 48000
        t = np.arange(sr // 4) / sr
        audio = (0.5 * np.sin(2 * np.pi * 1000 * t)).astype(np.float32)
        a = _make_analyzer_with_audio(audio, sr)
        freqs, times, db = a._reassigned_spectrogram(
            n_fft=1024, hop_length=256
        )
        assert freqs.ndim == times.ndim == 1
        assert db.shape == (len(freqs), len(times))
        assert np.isfinite(db).all()
        assert "heuristic" in a._reassigned_spectrogram.__doc__.lower()


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


# ── F1: peak / true-peak / loudness must use ALL channels ─────────────

def _make_stereo(l, r, sr=48000):
    from analyzer.core import AudioAnalyzer
    a = AudioAnalyzer()
    a.filepath = Path("/fake/st.wav")
    a.sample_rate = sr
    data = np.stack([l, r]).astype(np.float32)
    a.data = data
    a.duration = len(l) / sr
    a.channels = 2
    a._mono = data[0]
    a._source_format = None
    return a


class TestChannelHandling:
    def test_peak_truepeak_loudness_use_all_channels(self):
        """R louder than L -> peak/true-peak/loudness must reflect R, not L."""
        sr = 48000
        t = np.arange(sr * 2) / sr
        L = (0.10 * np.sin(2 * np.pi * 1000 * t)).astype(np.float32)
        R = (0.90 * np.sin(2 * np.pi * 1000 * t)).astype(np.float32)
        a = _make_stereo(L, R, sr)
        qa = a.analyze_quality()
        assert qa["peak_db"] > -5, f"peak_db={qa['peak_db']} should reflect R (~-0.9), not L (-20)"
        assert qa["true_peak_db"] > -5, f"true_peak_db={qa['true_peak_db']} should reflect R"
        assert qa["loudness"]["integrated_lufs"] > -12, (
            f"integrated={qa['loudness']['integrated_lufs']} should be near R, not L"
        )

    def test_peak_mono_unchanged(self):
        """Regression guard: mono peak still correct after all-channel change."""
        sr = 48000
        t = np.arange(sr) / sr
        a = _make_analyzer_with_audio((0.5 * np.sin(2 * np.pi * 1000 * t)).astype(np.float32), sr)
        qa = a.analyze_quality()
        assert -7 < qa["peak_db"] < -5  # ~ -6 dBFS

    def test_channel_specific_metrics_include_right_channel(self):
        sr = 48000
        t = np.arange(sr * 2) / sr
        right = (0.8 * np.sin(2 * np.pi * 1000 * t)).astype(np.float32)
        a = _make_stereo(np.zeros_like(right), right, sr)
        qa = a.analyze_quality()
        assert qa["rms"] > 0.5
        assert qa["zero_crossing"] > 3000


class TestClippingIsolation:
    """F2: isolated full-scale peaks are NOT clips; flat-top runs ARE."""

    def test_isolated_single_samples_not_flagged(self):
        sr = 48000
        n = sr * 2
        clean = np.zeros(n, dtype=np.float32)
        for i in range(0, n, 500):
            clean[i] = 1.0  # isolated single-sample full-scale peaks
        a = _make_analyzer_with_audio(clean, sr)
        res = a._detect_clipping(a.data, sr, None)
        assert res["ok"] is True, f"isolated full-scale peaks should not be clips, got ok={res['ok']}"
        assert res["single_sample_count"] > 0

    def test_flat_top_run_flagged(self):
        sr = 48000
        t = np.arange(sr * 2) / sr
        clipped = np.clip(1.5 * np.sin(2 * np.pi * 1000 * t), -1, 1).astype(np.float32)
        a = _make_analyzer_with_audio(clipped, sr)
        res = a._detect_clipping(a.data, sr, None)
        assert res["ok"] is False
        assert res["count"] > 0

    def test_clean_fullscale_sine_not_flagged(self):
        """A clean sine peaking at exactly full scale (no flat-top) is not a clip."""
        sr = 48000
        t = np.arange(sr * 2) / sr
        s = np.sin(2 * np.pi * 1000 * t)
        s = s / np.max(np.abs(s))  # max sample == 1.0, but no run of full-scale samples
        a = _make_analyzer_with_audio(s.astype(np.float32), sr)
        res = a._detect_clipping(a.data, sr, None)
        assert res["ok"] is True, (
            f"clean full-scale sine should not be a clip, got ok={res['ok']} count={res.get('count')}"
        )
