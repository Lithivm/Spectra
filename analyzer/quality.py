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
        sr = self.sample_rate
        # Full-mix view (samples, channels) for peak/true-peak/loudness — must
        # reflect ALL channels, not just channel 0.
        data_t = self.data.T
        # Channel-0 view for spectral/single-channel metrics (upsampling/rms/zcr/DR).
        mono = self.data[0] if self.data.ndim > 1 else self.data

        # Peak reflects ALL channels, not just channel 0.
        abs_data = np.abs(self.data)
        peak_val = float(abs_data.max())
        _, sample_of_peak = np.unravel_index(int(np.argmax(abs_data)), self.data.shape)
        peak_idx = int(sample_of_peak)

        tp_val = self._true_peak(data_t, sr)
        return {
            "clipping":      self._detect_clipping(self.data, sr, self._source_format),
            "upsampling":    self._detect_high_freq_cutoff(mono, sr),
            "dynamic_range": self._measure_dynamic_range(mono),
            "peak_db":       round(20 * np.log10(peak_val + 1e-12), 1) if peak_val > 0 else -120.0,
            "peak_sample":   peak_idx,
            "true_peak_db":  tp_val,
            "rms":           round(self._compute_rms(mono), 6),
            "zero_crossing": self._compute_zcr(mono),
            "loudness": self._measure_loudness(data_t, sr, cancel_check, tp_val),
        }

    # ------------------------------------------------------------------
    # Clipping detection
    # ------------------------------------------------------------------
    def _detect_clipping(self, audio: np.ndarray, sr: int, source_format: str | None) -> dict:
        """Flat-top clipping detection with hard/soft classification.

        Detection:
          - Any sample >= 0.999 is a candidate clip.
          - Consecutive candidates form a clip region (a run).
          - Only runs of >= MIN_RUN consecutive full-scale samples count as clips;
            isolated single samples are informational (single_sample_count).

        Hard vs soft classification (for multi-sample regions):
          - Hard clip: signal is at the ceiling and flat (ptp < flat_thresh).
          - Soft clip: signal is at the ceiling but curved (ptp >= flat_thresh),
            e.g. tube/tape saturation.
        """
        CLIP_THRESH = 0.999
        MIN_RUN = 2  # a clip is >= this many consecutive full-scale samples

        # Flatness threshold: bit-depth-aware for integer formats
        _INT_FLAT = {
            's16': 1 / 16384,      # 2 quantization steps, 16-bit
            's16p': 1 / 16384,
            's32': 1 / (1 << 30),
            's32p': 1 / (1 << 30),
            's24': 1 / (1 << 22),
            's24p': 1 / (1 << 22),
        }
        flat_thresh = _INT_FLAT.get(source_format or '', 1e-6)

        # Ensure 2D: (channels, samples)
        if audio.ndim == 1:
            audio = audio[np.newaxis, :]

        n_channels = audio.shape[0]
        total_count = 0
        total_longest_ms = 0
        total_hard = 0
        total_soft = 0
        total_single = 0
        channels_affected: list[int] = []

        for ch in range(n_channels):
            ch_data = audio[ch]
            over = np.abs(ch_data) >= CLIP_THRESH
            if not np.any(over):
                continue

            # Find contiguous runs of over-threshold samples
            padded = np.empty(len(over) + 2, dtype=np.int8)
            padded[0] = 0
            padded[-1] = 0
            padded[1:-1] = over.astype(np.int8)
            edges = np.diff(padded)
            starts = np.where(edges == 1)[0]
            ends = np.where(edges == -1)[0] - 1
            lengths = ends - starts + 1

            # Dual-bucket: only runs of >= MIN_RUN consecutive full-scale samples
            # count as clips; shorter runs (isolated spikes / sine peaks) are
            # informational via single_sample_count.
            single_mask = lengths < MIN_RUN
            single_count = int(np.sum(single_mask))
            total_single += single_count

            multi_starts = starts[~single_mask]
            multi_ends = ends[~single_mask]
            multi_lengths = lengths[~single_mask]

            ch_count = len(multi_starts)   # clips = runs of >= MIN_RUN only
            if ch_count == 0:
                continue

            channels_affected.append(ch)

            # Longest duration (multi-sample only)
            ch_longest_ms = int((multi_lengths / sr * 1000).max())

            # Hard vs soft classification using ptp (peak-to-peak flatness)
            ch_hard = 0
            for s, e in zip(multi_starts, multi_ends):
                segment = ch_data[s:e + 1]
                if float(np.ptp(segment)) < flat_thresh:
                    ch_hard += 1
            ch_soft = ch_count - ch_hard

            total_count += ch_count
            total_longest_ms = max(total_longest_ms, ch_longest_ms)
            total_hard += ch_hard
            total_soft += ch_soft

        if total_count == 0:
            return {
                "ok": True,
                "count": 0,
                "longest_ms": 0,
                "single_sample_count": total_single,
                "channels_affected": channels_affected,
                "method": "flat-top",
            }

        return {
            "ok": False,
            "count": total_count,
            "longest_ms": total_longest_ms,
            "hard_clips": total_hard,
            "soft_clips": total_soft,
            "single_sample_count": total_single,
            "channels_affected": channels_affected,
            "method": "flat-top",
        }

    # ------------------------------------------------------------------
    # High-frequency cutoff detection
    # ------------------------------------------------------------------
    _SLOPE_NATURAL_MAX = -15.0   # dB/octave — natural rolloff limit
    _SLOPE_CUTOFF_MIN  = -25.0   # dB/octave — artificial cutoff threshold

    def _detect_high_freq_cutoff(self, audio: np.ndarray, sr: int) -> dict:
        """Detect spectral cutoff from upsampling or low-pass filtering.

        Algorithm:
          1. Welch PSD — deterministic, no random segments.
          2. Noise floor = P5 of full spectrum; signal ref = P90 of 2–12 kHz band.
          3. Walk high→low to find where PSD rises above noise floor by >6 dB.
          4. Spectral slope (dB/oct) on upper 1/3 band.
          5. Gibbs ringing detection near cutoff candidate.
          6. Multi-factor confidence = 0.4×contrast + 0.35×slope + 0.25×gibbs.
        """
        from scipy.signal import welch

        nyq = sr / 2

        # ── Step 1: Welch PSD ──
        freqs, psd = welch(audio, fs=sr, nperseg=8192, noverlap=4096, window='hann')
        psd_db = 10 * np.log10(psd + 1e-12)

        if np.max(psd_db) < -110:
            return {"ok": True, "cutoff_hz": nyq, "confidence": 0.0,
                    "slope_dboct": 0.0, "gibbs_detected": False,
                    "method": "welch+multifactor"}

        # ── Step 2: Noise floor & signal reference ──
        noise_floor_db = float(np.percentile(psd_db, 5))

        ref_mask = (freqs >= 2000) & (freqs <= 12000)
        if ref_mask.any():
            signal_ref_db = float(np.percentile(psd_db[ref_mask], 90))
        else:
            signal_ref_db = float(np.max(psd_db))

        contrast_db = signal_ref_db - noise_floor_db
        contrast_score = float(np.clip((contrast_db - 6) / 34, 0.0, 1.0))

        # ── Step 3: Find cutoff — walk high→low ──
        SHELF_THRESHOLD_DB = 6.0
        cutoff_hz = nyq
        for i in range(len(psd_db) - 1, -1, -1):
            if psd_db[i] > noise_floor_db + SHELF_THRESHOLD_DB:
                if i < len(freqs) - 1:
                    cutoff_hz = float(freqs[i + 1])
                else:
                    cutoff_hz = nyq
                break

        # ── Step 4: Spectral slope on upper 1/3 band ──
        slope_dboct = 0.0
        upper_mask = (freqs >= sr / 6) & (freqs > 0)
        if upper_mask.sum() >= 10:
            x = np.log2(freqs[upper_mask])
            y = psd_db[upper_mask]
            slope_dboct = float(np.polyfit(x, y, 1)[0])

        slope_score = float(np.clip(
            (-slope_dboct - (-self._SLOPE_NATURAL_MAX))
            / (-self._SLOPE_CUTOFF_MIN - (-self._SLOPE_NATURAL_MAX)),
            0.0, 1.0,
        ))

        # ── Step 5: Gibbs ringing detection ──
        gibbs_detected = False
        if cutoff_hz < nyq * 0.85:
            lo = cutoff_hz * 0.90
            hi = cutoff_hz * 1.10
            window_mask = (freqs >= lo) & (freqs <= hi)
            if window_mask.sum() >= 3:
                window_db = psd_db[window_mask]
                trend = np.linspace(window_db[0], window_db[-1], len(window_db))
                above_trend = window_db - trend
                if float(above_trend.max()) > 3.0:
                    gibbs_detected = True
        gibbs_score = 1.0 if gibbs_detected else 0.0

        # ── Step 6: Multi-factor confidence ──
        confidence = (
            0.40 * contrast_score
            + 0.35 * slope_score
            + 0.25 * gibbs_score
        )

        # ── Decision ──
        cutoff_significant = cutoff_hz < nyq * 0.85
        cutoff_confident = confidence > 0.3
        is_cutoff = cutoff_significant and cutoff_confident

        return {
            "ok": not is_cutoff,
            "cutoff_hz": round(cutoff_hz),
            "nyq_hz": nyq,
            "confidence": round(confidence, 2),
            "slope_dboct": round(slope_dboct, 1),
            "gibbs_detected": gibbs_detected,
            "method": "welch+multifactor",
        }

    # ------------------------------------------------------------------
    # Dynamic range
    # ------------------------------------------------------------------
    def _measure_dynamic_range(self, audio: np.ndarray) -> dict:
        """Dynamic range: P95 - P10 of frame RMS levels (dB).

        Uses ~100ms frames (4096 samples @ 44.1kHz) with 50% hop.
        Matches the DR meter convention used by TT DR Meter and similar tools.
        Measures each channel separately and returns the maximum DR.
        """
        frame_len = 4096
        hop = frame_len // 2

        def _dr_single_channel(ch: np.ndarray) -> float:
            n = len(ch)
            n_frames = max(1, (n - frame_len) // hop + 1)
            shape = (n_frames, frame_len)
            strides = (ch.strides[0] * hop, ch.strides[0])
            frames = np.lib.stride_tricks.as_strided(ch, shape=shape, strides=strides)
            rms = np.sqrt(np.mean(frames ** 2, axis=1))
            rms = rms[rms > 1e-10]  # exclude silence
            if len(rms) < 2:
                return 0.0
            frames_db = 20 * np.log10(rms)
            p95 = float(np.percentile(frames_db, 95))
            p10 = float(np.percentile(frames_db, 10))
            return p95 - p10

        # TT DR Meter: measure each channel, take the maximum
        if self.data is not None and self.data.ndim > 1:
            dr = max(_dr_single_channel(self.data[ch]) for ch in range(self.data.shape[0]))
        else:
            dr = _dr_single_channel(audio)

        return {"dr": round(dr, 1)}

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
    def _measure_loudness(self, audio_st: np.ndarray, sr: int, cancel_check=None, true_peak_val: float | None = None) -> dict:
        """EBU R128 integrated loudness, short-term, LRA, true-peak (BS.1770-4).

        audio_st: (samples, n_channels) — all channels; Meter applies R128 gains.
        """
        import pyloudnorm as pyln
        if audio_st.ndim == 1:
            audio_st = audio_st[:, np.newaxis]

        TARGET_SR = 12000
        if sr > TARGET_SR * 1.5:
            from scipy.signal import decimate
            factor = max(1, sr // TARGET_SR)
            meter_sr = sr // factor
            audio_meter = decimate(audio_st.astype(np.float64), factor, zero_phase=True, axis=0)
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
            sv = np.array(sorted(st_vals))
            p10, p95 = np.percentile(sv, [10, 95])
            lra = round(float(p95 - p10), 1)
        else:
            lra = 0.0

        tp = true_peak_val if true_peak_val is not None else self._true_peak(audio_st, sr)

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
        # Batch both channels in one resample_poly call (axis=0)
        upsampled = scipy_signal.resample_poly(
            audio_st.astype(np.float64), oversample, 1, axis=0)
        peak = float(np.max(np.abs(upsampled)))
        if peak < 1e-12:
            return -120.0
        return round(20 * math.log10(peak), 1)
