"""Quality analysis mixin — clipping, upsampling detection, DR, LUFS, true peak."""

from __future__ import annotations

import math

import numpy as np


class _QualityMixin:
    """Mixed into AudioAnalyzer.  Expects self.data, self.sample_rate, etc."""

    # ------------------------------------------------------------------
    # Quality analysis entry point
    # ------------------------------------------------------------------
    def analyze_quality(self, cancel_check=None) -> dict:
        if self.data is None:
            raise RuntimeError("未加载音频")
        audio = self._mono
        sr = self.sample_rate
        peak_val, peak_idx = self._compute_peak(audio)
        tp_audio = audio.astype(np.float64)
        tp_val = self._true_peak(np.column_stack([tp_audio, tp_audio]), sr)
        return {
            "clipping":      self._detect_clipping(audio, sr),
            "upsampling":    self._detect_high_freq_cutoff(audio, sr),
            "dynamic_range": self._measure_dynamic_range(audio),
            "peak_db":       round(20 * np.log10(peak_val + 1e-12), 1),
            "peak_sample":   peak_idx,
            "true_peak_db":  tp_val,
            "rms":           round(self._compute_rms(audio), 6),
            "zero_crossing": self._compute_zcr(audio),
            "loudness": self._measure_loudness(audio, sr, cancel_check),
        }

    # ------------------------------------------------------------------
    # Clipping detection
    # ------------------------------------------------------------------
    def _detect_clipping(self, audio: np.ndarray, sr: int) -> dict:
        """Flat-top + intersample-peak clipping detection."""
        eps = np.finfo(np.float32).eps * 10
        MIN_FLAT = 3
        abs_audio = np.abs(audio)
        over = abs_audio >= 0.999

        diff = np.abs(np.diff(audio))
        flat_mask = (diff < eps) & over[:-1] & over[1:]

        padded = np.pad(flat_mask, (1, 1), constant_values=False)
        edges = np.diff(padded.astype(np.int8))
        starts = np.where(edges == 1)[0]
        ends = np.where(edges == -1)[0] - 1
        long_enough = (ends - starts + 1) >= MIN_FLAT
        flat_starts = starts[long_enough].tolist()
        flat_ends = ends[long_enough].tolist()

        if not flat_starts:
            return {"ok": True, "count": 0, "longest_ms": 0, "method": "flat-top"}

        durations_ms = [int((e - s + 1) / sr * 1000) for s, e in zip(flat_starts, flat_ends)]
        longest_ms = max(durations_ms)

        hard_count = 0
        for s, e in zip(flat_starts, flat_ends):
            region = audio[s:e+1]
            if np.max(np.abs(np.diff(region))) < eps * 2:
                hard_count += 1

        return {
            "ok": False,
            "count": len(flat_starts),
            "longest_ms": longest_ms,
            "hard_clips": hard_count,
            "soft_clips": len(flat_starts) - hard_count,
            "method": "flat-top",
        }

    # ------------------------------------------------------------------
    # High-frequency cutoff detection
    # ------------------------------------------------------------------
    def _detect_high_freq_cutoff(self, audio: np.ndarray, sr: int) -> dict:
        """Multi-segment median-spectrum cutoff detection."""
        import scipy.ndimage as ndi

        nyq = sr / 2
        seg_dur = 1.5
        seg_len = int(seg_dur * sr)
        n_segs = min(8, max(3, len(audio) // seg_len))

        if n_segs <= 1 or len(audio) < seg_len:
            segments = [audio]
        else:
            max_start = len(audio) - seg_len
            rng = np.random.default_rng(42)
            starts = sorted(rng.integers(0, max(1, max_start), size=n_segs))
            segments = [audio[s:s + seg_len] for s in starts]

        spectra = []
        for seg in segments:
            S = np.abs(np.fft.rfft(seg * np.hanning(len(seg)))) ** 2
            spectra.append(S)

        min_len = min(len(s) for s in spectra)
        spectra = [s[:min_len] for s in spectra]

        med_spec = np.median(np.array(spectra), axis=0)
        freqs = np.fft.rfftfreq(len(segments[0]), 1.0 / sr)[:min_len]

        if np.max(med_spec) < 1e-12:
            return {"ok": True, "cutoff_hz": nyq, "confidence": 0.0, "method": "multi-seg median"}

        f_min = max(freqs[0], 100.0)
        f_max = nyq
        n_bins = 128
        log_edges = np.logspace(np.log10(f_min), np.log10(f_max), n_bins + 1)
        log_centers = np.sqrt(log_edges[:-1] * log_edges[1:])

        binned_energy = np.zeros(n_bins)
        bin_idx = np.clip(np.digitize(freqs, log_edges) - 1, 0, n_bins - 1)
        counts = np.bincount(bin_idx, minlength=n_bins)
        totals = np.bincount(bin_idx, weights=med_spec, minlength=n_bins)
        valid = counts > 0
        binned_energy[valid] = totals[valid] / counts[valid]

        binned_db = 10 * np.log10(binned_energy + 1e-12)
        binned_db = ndi.gaussian_filter1d(binned_db, sigma=1.5)

        hi_start = n_bins // 2
        hi_db = binned_db[hi_start:]
        hi_freqs = log_centers[hi_start:]

        slope = np.diff(hi_db)
        slope = np.append(slope, slope[-1])

        slope_mean = np.mean(slope)
        slope_std = np.std(slope)

        steep_drop = slope < (slope_mean - 2.5 * slope_std)

        cutoff_hz = nyq
        if np.any(steep_drop):
            drop_idx = int(np.argmax(steep_drop))
            cutoff_hz = float(hi_freqs[drop_idx])

        hi_energy = binned_energy[hi_start:]
        if np.mean(hi_energy) > 1e-12:
            energy_cv = float(np.std(hi_energy) / np.mean(hi_energy))
            cutoff_confidence = max(0.0, min(1.0, 1.0 - energy_cv * 2.0))
        else:
            cutoff_confidence = 1.0

        cutoff_significant = cutoff_hz < nyq * 0.9
        cutoff_confident = cutoff_confidence > 0.3
        is_cutoff = cutoff_significant and cutoff_confident

        return {
            "ok": not is_cutoff,
            "cutoff_hz": cutoff_hz,
            "nyq_hz": nyq,
            "method": "multi-seg median + slope",
        }

    # ------------------------------------------------------------------
    # Dynamic range
    # ------------------------------------------------------------------
    def _measure_dynamic_range(self, audio: np.ndarray) -> dict:
        import librosa
        rms = librosa.feature.rms(y=audio, frame_length=2048, hop_length=512, center=False)[0]
        frames_db = 20 * np.log10(rms[rms > 0])
        if len(frames_db) == 0:
            return {"dr": 0.0}
        top = float(np.max(frames_db))
        bottom = float(np.mean(frames_db))
        return {"dr": round(top - bottom, 1)}

    # ------------------------------------------------------------------
    # Basic metrics
    # ------------------------------------------------------------------
    def _compute_rms(self, audio: np.ndarray) -> float:
        return float(np.sqrt(np.mean(audio ** 2)))

    def _compute_peak(self, audio: np.ndarray) -> tuple[float, int]:
        idx = int(np.argmax(np.abs(audio)))
        return float(np.abs(audio[idx])), idx

    def _compute_zcr(self, audio: np.ndarray) -> int:
        return int(np.sum(np.abs(np.diff(np.signbit(audio)))))

    # ------------------------------------------------------------------
    # Loudness (EBU R128)
    # ------------------------------------------------------------------
    def _measure_loudness(self, audio: np.ndarray, sr: int, cancel_check=None) -> dict:
        """EBU R128 integrated loudness, short-term, LRA, true-peak (BS.1770-4)."""
        import pyloudnorm as pyln
        if audio.ndim == 1:
            audio_st = np.column_stack([audio, audio])
        else:
            audio_st = audio.T[:, :2]
        if audio_st.shape[1] == 1:
            audio_st = np.column_stack([audio_st[:, 0], audio_st[:, 0]])

        TARGET_SR = 12000
        if sr > TARGET_SR * 1.5:
            from scipy.signal import decimate
            factor = max(1, sr // TARGET_SR)
            meter_sr = sr // factor
            ch0 = decimate(audio_st[:, 0].astype(np.float64), factor, zero_phase=True)
            audio_meter = np.column_stack([ch0, ch0]).astype(np.float64)
        else:
            meter_sr = sr
            audio_meter = audio_st.astype(np.float64)

        meter = pyln.Meter(meter_sr)
        integrated = float(meter.integrated_loudness(audio_meter))

        block_s = 3
        hop = block_s * meter_sr
        n_blocks = max(1, len(audio_meter) // hop)
        st_vals: list[float] = []
        for i in range(n_blocks):
            if cancel_check is not None and cancel_check():
                break
            block = audio_meter[i * hop : (i + 1) * hop]
            if len(block) >= meter_sr:
                st_vals.append(float(meter.integrated_loudness(block)))
        short_term = max(st_vals) if st_vals else integrated

        if len(st_vals) >= 3:
            sv = sorted(st_vals)
            p10 = sv[int(len(sv) * 0.1)]
            p95 = sv[int(len(sv) * 0.95)]
            lra = round(p95 - p10, 1)
        else:
            lra = 0.0

        tp = self._true_peak(audio_st, sr)

        return {
            "integrated_lufs": round(integrated, 1),
            "short_term_lufs": round(short_term, 1),
            "lra_lu": lra,
            "true_peak_db": tp,
        }

    @staticmethod
    def _true_peak(audio_st: np.ndarray, sr: int) -> float:
        """BS.1770-4 true peak: 4× polyphase upsampling → max |sample| → dBTP."""
        from scipy import signal as scipy_signal

        oversample = 4
        peak = 0.0
        for ch in range(audio_st.shape[1]):
            upsampled = scipy_signal.resample_poly(
                audio_st[:, ch].astype(np.float64), oversample, 1)
            ch_peak = float(np.max(np.abs(upsampled)))
            if ch_peak > peak:
                peak = ch_peak
        if peak < 1e-12:
            return -120.0
        return round(20 * math.log10(peak), 1)
