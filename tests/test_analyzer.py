"""Algorithm tests for Spectra analyzer — no audio files needed.

Tests clipping detection, RMS, peak, DR, flatten_analysis, and i18n.
Uses synthetic numpy signals exclusively.
"""

import math

import numpy as np
import pytest
from pathlib import Path


# ── is_audio_file ──────────────────────────────────────────────────────

class TestIsAudioFile:
    def test_supported_extensions(self):
        from spectra.backend.load import is_audio_file
        for ext in [".flac", ".opus", ".wav", ".mp3", ".m4a", ".mp4",
                    ".aac", ".wma", ".ape", ".ogg", ".tta", ".aiff"]:
            assert is_audio_file(f"test{ext}") is True

    def test_unsupported_extensions(self):
        from spectra.backend.load import is_audio_file
        for name in ["readme.txt", "data.bin", "photo.jpg", "video.mkv", "doc.pdf"]:
            assert is_audio_file(name) is False

    def test_case_insensitive(self):
        from spectra.backend.load import is_audio_file
        assert is_audio_file("song.MP3") is True
        assert is_audio_file("song.WAV") is True
        assert is_audio_file("song.FlAc") is True


# ── Quality metrics with synthetic signals ─────────────────────────────

def _make_analyzer_with_audio(audio: np.ndarray, sr: int = 48000):
    """Create an AudioAnalyzer pre-loaded with synthetic audio."""
    from spectra.backend.core import AudioAnalyzer
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
    """DR = TT DR family: P95 - P10 over 300 ms K-weighted windows, ungated.

    Needs >= ~3 x 300 ms windows. LRA is a different reading (3 s windows + double
    gating) — the two must not be conflated.
    """

    def _dr(self, x):
        a = _make_analyzer_with_audio(np.asarray(x, dtype=np.float32))
        return a._measure_dynamic_range(a.data), a

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
        assert a._short_term_loudness_values(a.data, a.sample_rate) == []
        assert a._measure_dynamic_range(a.data) == {"dr": 0.0, "dr_basis": "none"}

    def test_single_sample_quality_analysis_is_safe(self):
        a = _make_analyzer_with_audio(np.zeros(1, dtype=np.float32))
        result = a.analyze_quality()
        assert result["dynamic_range"]["dr"] == 0.0
        assert result["loudness"]["integrated_lufs"] == -np.inf

    def test_dr_is_tt_dr_not_lra(self):
        """DR (TT, 300 ms 窗、无门控) 和 LRA (3 s 窗、门控) 是两个读数 —— 不能画等号。

        在明暗交替的信号上两者接近；在打击型信号上 TT DR 明显大于 LRA。
        """
        sr = 48000
        rng = np.random.default_rng(3)
        n = sr * 20
        x = rng.standard_normal(n) * 0.02
        for i in range(20):
            if i % 3 == 0:                      # 偶发强瞬态
                s = i * sr
                x[s:s + 2000] = rng.standard_normal(2000) * 0.9
        a = _make_analyzer_with_audio(x.astype(np.float32))
        q = a.analyze_quality()
        dr = q["dynamic_range"]
        assert dr["dr_basis"] == "TT_DR_300ms_ungated"
        assert dr["dr"] > q["loudness"]["lra_lu"] + 5.0

    def test_dr_matches_independent_reference(self):
        """DR 对照「整段一次性 K 滤波 + 逐 300 ms 窗」的直算参考。

        不能拿 BS.1770 附录的手写表值系数当参考：直接喂给 lfilter 时极点模 > 1（不稳定），
        结果是 nan。这里用同一套 DeMan 系数，但走一条完全独立的路径（不分块、不管状态），
        锁住分块实现的数值正确性。
        """
        from scipy.signal import lfilter
        from spectra.backend import loudness as r128

        sr = 48000
        x = self._alternating_loudness().astype(np.float64)
        y = x.copy()
        for b, a in r128.k_weight_coeffs(sr):
            y = lfilter(b, a, y)
        L = int(round(0.3 * sr))
        vals = [r128.K_CONSTANT + 10 * math.log10(max(float(np.mean(y[i:i + L] ** 2)), 1e-300))
                for i in range(0, len(x) - L + 1, L)]
        ref = float(np.percentile(vals, 95) - np.percentile(vals, 10))
        a = _make_analyzer_with_audio(x.astype(np.float32), sr)
        got = a._measure_dynamic_range(a.data)["dr"]
        assert abs(got - ref) < 0.05, (got, ref)


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
    def test_returns_finite_spectrogram(self):
        sr = 48000
        t = np.arange(sr // 4) / sr
        audio = (0.5 * np.sin(2 * np.pi * 1000 * t)).astype(np.float32)
        a = _make_analyzer_with_audio(audio, sr)
        freqs, times, db = a._reassigned_spectrogram(n_fft=2048, hop_length=512)
        assert freqs.ndim == times.ndim == 1
        assert db.shape == (len(freqs), len(times))
        assert np.isfinite(db).all()
        assert "Auger" in a._reassigned_spectrogram.__doc__

    def test_chirp_instantaneous_frequency(self):
        """Auger–Flandrin 重分配后，啁啓的瞬时频率（解析已知）必须被逐帧读出。"""
        sr = 48000
        T = 2.0
        t = np.arange(int(sr * T)) / sr
        x = np.sin(2 * np.pi * (1000 * t + 3000 * t * t / (2 * T)))
        a = _make_analyzer_with_audio(x.astype(np.float32), sr)
        freqs, times, db = a._reassigned_spectrogram(n_fft=2048, hop_length=512)
        bin_hz = sr / 2048
        for j in (30, 60, 112, 150):
            truth = 1000 + 3000 * (j * 512 / sr) / T
            row = int(np.argmax(db[:, j]))
            assert abs(freqs[row] - truth) <= bin_hz, (row, freqs[row], truth)

    def test_transient_moves_to_true_time(self):
        sr = 48000
        x = np.zeros(sr, dtype=np.float32)
        x[int(0.5 * sr)] = 1.0
        a = _make_analyzer_with_audio(x, sr)
        freqs, times, db = a._reassigned_spectrogram(n_fft=2048, hop_length=512)
        col = int(np.argmax(db[40, :]))
        assert abs(times[col] - 0.5) < 512 / sr, (times[col], col)


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
        from spectra.backend.batch import flatten_analysis, BATCH_COLUMNS

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
        from spectra.backend.batch import flatten_analysis

        md = {"format": "MP3"}
        row = flatten_analysis(md, None, Path("/fake/test.mp3"))

        assert row["filename"] == "test.mp3"
        assert row["format"] == "MP3"
        # Quality fields should be defaults
        assert row["clipping_count"] == 0


# ── i18n ────────────────────────────────────────────────────────────────

class TestLang:
    def test_t_zh(self):
        from spectra.lang import t, LANG
        LANG_old = LANG
        try:
            import spectra.lang as lang
            lang.LANG = "zh"
            assert t("中文", "English") == "中文"
        finally:
            lang.LANG = LANG_old

    def test_t_en(self):
        from spectra.lang import t, LANG
        LANG_old = LANG
        try:
            import spectra.lang as lang
            lang.LANG = "en"
            assert t("中文", "English") == "English"
        finally:
            lang.LANG = LANG_old

    def test_toggle_lang(self):
        from spectra.lang import toggle_lang, LANG
        LANG_old = LANG
        try:
            import spectra.lang as lang
            lang.LANG = "zh"
            new = toggle_lang()
            assert new == "en"
            new = toggle_lang()
            assert new == "zh"
        finally:
            lang.LANG = LANG_old

    def test_on_lang_change_returns_unsubscribe(self):
        from spectra.lang import on_lang_change

        calls = []
        unsub = on_lang_change(lambda lang_code: calls.append(lang_code))
        assert callable(unsub)

        unsub()  # should not raise
        assert len(calls) == 0  # no toggle happened yet


# ── Load module normalization ──────────────────────────────────────────

class TestLoadNormalization:
    def test_supported_extensions_set(self):
        from spectra.backend.load import SUPPORTED_EXTENSIONS
        assert ".wav" in SUPPORTED_EXTENSIONS
        assert ".flac" in SUPPORTED_EXTENSIONS
        assert ".mp3" in SUPPORTED_EXTENSIONS
        assert ".txt" not in SUPPORTED_EXTENSIONS


# ── AudioAnalyzer basic state ──────────────────────────────────────────

class TestAudioAnalyzerState:
    def test_default_state(self):
        from spectra.backend.core import AudioAnalyzer
        a = AudioAnalyzer()
        assert a.filepath is None
        assert a.data is None
        assert a.sample_rate == 0
        assert a.duration == 0.0

    def test_waveform_raises_without_data(self):
        from spectra.backend.core import AudioAnalyzer
        a = AudioAnalyzer()
        with pytest.raises(RuntimeError):
            _ = a.waveform

    def test_analyze_quality_raises_without_data(self):
        from spectra.backend.core import AudioAnalyzer
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

def _make_stereo(left, right, sr=48000):
    from spectra.backend.core import AudioAnalyzer
    a = AudioAnalyzer()
    a.filepath = Path("/fake/st.wav")
    a.sample_rate = sr
    data = np.stack([left, right]).astype(np.float32)
    a.data = data
    a.duration = len(left) / sr
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


# ── R128 在原生采样率上（不再降到 12 kHz）───────────────────────────────

class TestR128NativeRate:
    """BS.1770 要求 K 权重在原始采样率上设计；降到 12 kHz 的 Nyquist 只有 6 kHz，
    高架中心 16.8 kHz 整段被削掉。"""

    def _gain(self, f, sr):
        from spectra.backend import loudness as r128
        tot = 1.0
        for b, a in r128.k_weight_coeffs(int(sr)):
            z = np.exp(2j * np.pi * f / int(sr))
            tot *= np.polyval(b, z) / np.polyval(a, z)
        return 20 * math.log10(abs(tot))

    def test_k_weight_plateau_at_10khz(self):
        from spectra.backend import loudness as r128
        sr = 48000
        assert 3.5 <= self._gain(10000, sr) <= 4.5, self._gain(10000, sr)
        t = np.arange(3 * sr) / sr
        x = (0.3 * np.sin(2 * np.pi * 10000 * t)).astype(np.float32)
        a = _make_analyzer_with_audio(x, sr)
        measured = r128.gate_mean(r128.block_loudnesses(a.data, sr, 0.4, 0.1))
        expected = r128.K_CONSTANT + 10 * math.log10(0.3 ** 2 / 2) + self._gain(10000, sr)
        assert abs(measured - expected) < 0.3, (measured, expected)

    def test_chunked_equals_whole_file(self):
        from scipy.signal import lfilter
        from spectra.backend import loudness as r128
        sr = 44100
        rng = np.random.default_rng(5)
        x = rng.standard_normal(sr * 6) * 0.2
        y = x.copy()
        for b, a in r128.k_weight_coeffs(sr):
            y = lfilter(b, a, y)
        L, H = int(round(0.4 * sr)), int(round(0.1 * sr))
        ref = np.array([r128.K_CONSTANT + 10 * math.log10(max(float(np.mean(y[i:i + L] ** 2)), 1e-300))
                        for i in range(0, len(x) - L + 1, H)])
        got = r128.block_loudnesses(x[np.newaxis, :], sr, 0.4, 0.1, chunk_s=1.0)
        assert len(got) == len(ref)
        assert float(np.max(np.abs(got - ref))) < 1e-6, (got[:3], ref[:3])

    def test_true_peak_covers_full_band(self):
        from spectra.backend import loudness as r128
        sr = 48000
        t = np.arange(sr) / sr
        x = 0.9 * np.sin(2 * np.pi * 16000 * t + 0.31)      # 采样点错过波峰
        sample_peak = 20 * math.log10(float(np.max(np.abs(x))))
        tp = r128.true_peak_dbtp(x[np.newaxis, :], sr)
        assert tp is not None
        assert tp > sample_peak + 0.4, (tp, sample_peak)
        assert r128.true_peak_dbtp(x[np.newaxis, :], 11025) is None


# ── 位深感知的削波门限（2 LSB）────────────────────────────────────────

class TestClippingBitDepthThreshold:
    def test_quantized_near_ceiling_not_flagged(self):
        """16 bit 量化、峰 0.9994 的噪声离满刻度还差 >2 LSB —— 旧门限 0.999 会误报 160 次。"""
        rng = np.random.default_rng(6)
        r = rng.standard_normal(48000)
        r = r / np.max(np.abs(r)) * 0.9994
        x = (np.round(r * 32767) / 32767).astype(np.float32)
        a = _make_analyzer_with_audio(x)
        res = a._detect_clipping(a.data, a.sample_rate, "s16p")
        assert res["ok"] is True and res["count"] == 0, res

    def test_real_ceiling_run_still_flagged(self):
        rng = np.random.default_rng(7)
        x = np.round(rng.standard_normal(48000) * 1.3 * 32767) / 32767
        x = np.clip(x, -1.0 + 1 / 32767, 1.0 - 1 / 32767)
        a = _make_analyzer_with_audio(x.astype(np.float32))
        res = a._detect_clipping(a.data, a.sample_rate, "s16p")
        assert res["ok"] is False and res["count"] > 0, res


# ── 多分辨率：统一行网格 ──────────────────────────────────────────────

class TestMultiResolutionGrid:
    def test_freqs_monotonic_and_no_holes(self):
        import spectra.backend._state as st
        st._stft_cache.clear()
        sr = 44100
        t = np.arange(sr) / sr
        x = (0.2 * np.sin(2 * np.pi * 1000 * t)).astype(np.float32)
        a = _make_analyzer_with_audio(x, sr)
        freqs, times, db = a.spectrogram_db(mode="multi")
        assert np.all(np.diff(freqs) > 0), "freqs must be strictly monotonic"
        assert not np.all(db <= -120.0, axis=1).any(), "有行整行只有底噪（映射空洞）"

    def test_tone_lands_on_expected_grid_row(self):
        import spectra.backend._state as st
        st._stft_cache.clear()
        sr = 44100
        t = np.arange(sr) / sr
        x = (0.2 * np.sin(2 * np.pi * 1000 * t)).astype(np.float32)
        a = _make_analyzer_with_audio(x, sr)
        freqs, times, db = a.spectrogram_db(mode="multi")
        row = int(np.argmax(db.max(axis=1)))
        # 1 kHz 落在 mid 频段（n_fft=2048 → 21.5 Hz 一行）；最近映射会把这一行
        # 铺到它周围几个 grid 行，argmax 取第一个。
        assert abs(freqs[row] - 1000.0) <= sr / 2048, (row, freqs[row])


# ── dBFS 参考 ─────────────────────────────────────────────────────────

class TestDbfsReference:
    def test_absolute_reference_is_true_dbfs(self):
        import spectra.backend._state as st
        st._stft_cache.clear()
        sr = 48000
        t = np.arange(sr) / sr
        full = np.sin(2 * np.pi * 1000 * t).astype(np.float32)
        quiet = (0.01 * np.sin(2 * np.pi * 1000 * t)).astype(np.float32)
        a, b = _make_analyzer_with_audio(full, sr), _make_analyzer_with_audio(quiet, sr)
        p_full = float(a.spectrogram_db(n_fft=2048)[2].max())
        st._stft_cache.clear()          # 两个 analyzer 共用同一个假路径 → 缓存键会撞
        p_quiet = float(b.spectrogram_db(n_fft=2048)[2].max())
        assert abs((p_full - p_quiet) - 40.0) < 0.05, (p_full, p_quiet)
        assert -1.2 <= p_full <= -0.2, p_full      # 窗增益 + 半 bin 损失 ≈ −0.6 dB

    def test_ref_peak_is_a_display_option(self):
        import spectra.backend._state as st
        st._stft_cache.clear()
        sr = 48000
        t = np.arange(sr) / sr
        x = (0.01 * np.sin(2 * np.pi * 1000 * t)).astype(np.float32)
        a = _make_analyzer_with_audio(x, sr)
        assert abs(float(a.spectrogram_db(n_fft=2048, ref="peak")[2].max())) < 1e-4


# ── 缓存键（UI 快路径必须与分析器一致）───────────────────────────────

class TestCacheKey:
    def test_key_shape(self):
        a = _make_analyzer_with_audio(np.zeros(4800, dtype=np.float32))
        k = a._cache_key("standard", 2048, 256, "hann", "dbfs")
        assert k[2] == "standard" and k[3] == 2048 and k[4] == 256
        assert k[5] == "hann" and k[6] == "dbfs"
        for other in (("multi", 2048, 256, "hann", "dbfs"),
                      ("standard", 4096, 256, "hann", "dbfs"),
                      ("standard", 2048, 512, "hann", "dbfs"),
                      ("standard", 2048, 256, "blackman", "dbfs"),
                      ("standard", 2048, 256, "hann", "peak")):
            assert a._cache_key(*other) != k, other

    def test_cached_result_and_invalidate(self):
        import spectra.backend._state as st
        st._stft_cache.clear()
        rng = np.random.default_rng(2)
        a = _make_analyzer_with_audio((rng.standard_normal(48000) * 0.1).astype(np.float32))
        freqs, times, db = a.spectrogram_db(n_fft=2048)
        hit = a.cached_result("standard", 2048)
        assert hit is not None and np.array_equal(hit[2], db)
        assert a.cached_result("multi", 2048) is None
        assert a.invalidate_cache("standard", 2048) == 1
        assert a.cached_result("standard", 2048) is None
        st._stft_cache.clear()


# ── 流式分组：整组 max（含最后的半组）+ 组中心时间 ─────────────────────

class TestStreamingGrouping:
    def test_partial_last_group_and_centers(self):
        from scipy.signal import get_window
        from spectra.backend.spectrum import _amp_scale, _to_db
        import spectra.backend._state as st
        st._stft_cache.clear()

        sr = 48000
        n = sr * 6
        rng = np.random.default_rng(7)
        x = (rng.standard_normal(n) * 0.2).astype(np.float32)
        a = _make_analyzer_with_audio(x, sr)
        a.TARGET_FRAMES = 4                       # 强制分组
        freqs, times, acc = a.spectrogram_db_streaming(
            n_fft=2048, on_init=lambda f, t, h: None,
            on_block=lambda i, b: None, cancel_check=lambda: False)

        n_fft, n_freqs = 2048, 1025
        hop = a._resolve_hop(n_fft, None)
        padded = np.pad(x, n_fft // 2, mode="reflect")
        raw = 1 + (len(padded) - n_fft) // hop
        ds = raw // a.TARGET_FRAMES + 1
        win = get_window("hann", n_fft, fftbins=True)
        scale = _amp_scale(win, n_fft, n_freqs)
        cols = np.stack([_to_db(np.abs(np.fft.rfft(padded[c * hop:c * hop + n_fft] * win))
                                .astype(np.float32), scale) for c in range(raw)], axis=1)
        ref = np.stack([cols[:, k:k + ds].max(axis=1) for k in range(0, raw, ds)], axis=1)

        assert acc.shape == ref.shape
        assert np.allclose(acc, ref, atol=1e-4), (acc.shape, ref.shape)
        centers = [(k + (min(ds, raw - k) - 1) / 2) * hop / sr for k in range(0, raw, ds)]
        assert np.allclose(times, centers, atol=1e-9), (times, centers)
        st._stft_cache.clear()
