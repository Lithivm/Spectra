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
            "loudness": self._measure_loudness(audio, sr, cancel_check, tp_val),
        }

    # ------------------------------------------------------------------
    # Clipping detection
    # ------------------------------------------------------------------
    def _detect_clipping(self, audio: np.ndarray, sr: int) -> dict:
        """Flat-top clipping detection with hard/soft classification.

        Detection:
          - Any sample >= 0.999 is a candidate clip.
          - Consecutive candidates form a clip region.
          - Regions of length >= 2 are "flat-top" clips.
          - Single-sample peaks >= 0.999 are also reported as clips.

        Hard vs soft classification (for regions >= 3):
          - Hard clip: signal is at the ceiling and flat (2nd derivative ≈ 0).
          - Soft clip: signal is at the ceiling but curved (2nd derivative ≠ 0),
            e.g. tube/tape saturation.
        """
        CLIP_THRESH = 0.999
        eps = np.finfo(np.float32).eps * 10

        over = np.abs(audio) >= CLIP_THRESH
        if not np.any(over):
            return {"ok": True, "count": 0, "longest_ms": 0, "method": "flat-top"}

        # Find contiguous runs of over-threshold samples
        padded = np.empty(len(over) + 2, dtype=np.int8)
        padded[0] = 0
        padded[-1] = 0
        padded[1:-1] = over.astype(np.int8)
        edges = np.diff(padded)
        starts = np.where(edges == 1)[0]
        ends = np.where(edges == -1)[0] - 1
        lengths = ends - starts + 1

        # Filter: keep regions >= 1 sample (single peaks are valid clips)
        keep = lengths >= 1
        clip_starts = starts[keep]
        clip_ends = ends[keep]

        if len(clip_starts) == 0:
            return {"ok": True, "count": 0, "longest_ms": 0, "method": "flat-top"}

        durations_ms = ((clip_ends - clip_starts + 1) / sr * 1000).astype(int)
        longest_ms = int(durations_ms.max())

        # Hard vs soft classification using second derivative (curvature)
        # Hard clip: flat top → 2nd derivative ≈ 0
        # Soft clip: curved top → 2nd derivative ≠ 0
        lengths = clip_ends - clip_starts + 1
        hard_count = 0

        # length == 2: check if both samples are at the same level (vectorized)
        mask_len2 = lengths == 2
        if np.any(mask_len2):
            s2 = clip_starts[mask_len2]
            e2 = clip_ends[mask_len2]
            hard_count += int(np.sum(np.abs(audio[s2] - audio[e2]) < eps))

        # length >= 3: check curvature via 2nd derivative (regions vary in length)
        mask_len3 = lengths >= 3
        for s, e in zip(clip_starts[mask_len3], clip_ends[mask_len3]):
            region = audio[s:e + 1]
            d2 = np.abs(np.diff(region, n=2))
            if np.max(d2) < eps * 10:
                hard_count += 1

        return {
            "ok": False,
            "count": len(clip_starts),
            "longest_ms": longest_ms,
            "hard_clips": hard_count,
            "soft_clips": len(clip_starts) - hard_count,
            "method": "flat-top",
        }

    # ------------------------------------------------------------------
    # High-frequency cutoff detection
    # ------------------------------------------------------------------
    def _detect_high_freq_cutoff(self, audio: np.ndarray, sr: int) -> dict:
        """Detect spectral cutoff from upsampling or low-pass filtering.

        Algorithm:
          1. Compute median power spectrum across multiple random segments.
          2. Bin into 128 log-spaced frequency bins, convert to dB.
          3. Estimate the noise floor from the top 10% of bins.
          4. Walk from high→low frequency to find where the spectrum rises
             above the noise floor by >6 dB — that's the cutoff point.
          5. Confidence = contrast between the signal band and the noise shelf.
        """
        import scipy.ndimage as ndi

        nyq = sr / 2
        seg_dur = 1.5
        seg_len = int(seg_dur * sr)
        n_segs = min(8, max(3, len(audio) // seg_len))

        if n_segs <= 1 or len(audio) < seg_len:
            seg_starts = np.array([0])
        else:
            max_start = len(audio) - seg_len
            rng = np.random.default_rng(42)
            seg_starts = np.sort(rng.integers(0, max(1, max_start), size=n_segs))

        window = np.hanning(seg_len).astype(np.float32)
        n_rfft = seg_len // 2 + 1
        # Batch all segments into a 2D array for a single FFT call
        segments = np.empty((len(seg_starts), seg_len), dtype=np.float32)
        for i, s in enumerate(seg_starts):
            segments[i] = audio[s:s + seg_len]
        segments *= window  # broadcast window across all segments
        all_specs = np.abs(np.fft.rfft(segments, axis=1)) ** 2

        med_spec = np.median(all_specs, axis=0)
        freqs = np.fft.rfftfreq(seg_len, 1.0 / sr)

        if np.max(med_spec) < 1e-12:
            return {"ok": True, "cutoff_hz": nyq, "confidence": 0.0, "method": "multi-seg median"}

        # Log-spaced binning
        f_min = max(freqs[0], 100.0)
        n_bins = 128
        log_edges = np.logspace(np.log10(f_min), np.log10(nyq), n_bins + 1)
        log_centers = np.sqrt(log_edges[:-1] * log_edges[1:])

        binned_energy = np.zeros(n_bins)
        bin_idx = np.clip(np.digitize(freqs, log_edges) - 1, 0, n_bins - 1)
        counts = np.bincount(bin_idx, minlength=n_bins)
        totals = np.bincount(bin_idx, weights=med_spec, minlength=n_bins)
        valid = counts > 0
        binned_energy[valid] = totals[valid] / counts[valid]

        binned_db = 10 * np.log10(binned_energy + 1e-12)
        binned_db = ndi.gaussian_filter1d(binned_db, sigma=1.5)

        # ── Noise floor estimation ──
        # The top 10% of bins (highest frequencies) represent the noise shelf
        # if a cutoff exists, or natural rolloff if not.
        shelf_start = int(n_bins * 0.9)
        noise_floor_db = float(np.median(binned_db[shelf_start:]))

        # ── Find cutoff: walk high→low, find where spectrum rises above floor ──
        SHELF_THRESHOLD_DB = 6.0  # signal must be >6 dB above noise floor
        cutoff_hz = nyq
        signal_peak_db = float(np.max(binned_db[:shelf_start]))

        # Walk from top down to find the transition point
        for i in range(n_bins - 1, -1, -1):
            if binned_db[i] > noise_floor_db + SHELF_THRESHOLD_DB:
                # This bin is above the shelf — cutoff is between i and i+1
                if i < n_bins - 1:
                    cutoff_hz = float(log_centers[i + 1])
                else:
                    cutoff_hz = nyq
                break

        # ── Confidence: based on contrast between signal and shelf ──
        contrast_db = signal_peak_db - noise_floor_db
        if contrast_db > 40:
            confidence = 1.0
        elif contrast_db > 20:
            confidence = (contrast_db - 20) / 20
        else:
            confidence = 0.0

        # ── Decision ──
        cutoff_significant = cutoff_hz < nyq * 0.85
        cutoff_confident = confidence > 0.3
        is_cutoff = cutoff_significant and cutoff_confident

        return {
            "ok": not is_cutoff,
            "cutoff_hz": round(cutoff_hz),
            "nyq_hz": nyq,
            "method": "shelf detection",
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
    def _measure_loudness(self, audio: np.ndarray, sr: int, cancel_check=None, true_peak_val: float | None = None) -> dict:
        """EBU R128 integrated loudness, short-term, LRA, true-peak (BS.1770-4)."""
        import pyloudnorm as pyln
        if audio.ndim == 1:
            audio_st = np.repeat(audio[:, np.newaxis], 2, axis=1)
        else:
            audio_st = audio.T[:, :2]
        if audio_st.shape[1] == 1:
            audio_st = np.repeat(audio_st, 2, axis=1)

        TARGET_SR = 12000
        if sr > TARGET_SR * 1.5:
            from scipy.signal import decimate
            # Multi-stage cascaded decimation: factor=2 per stage
            # Better anti-aliasing than single large factor (e.g. 16)
            factor = max(1, sr // TARGET_SR)
            meter_sr = sr // factor
            ch = audio_st[:, 0].astype(np.float64)
            remaining = factor
            while remaining > 1:
                stage = min(remaining, 2)
                ch = decimate(ch, stage, zero_phase=True)
                remaining //= stage
            audio_meter = np.repeat(ch[:, np.newaxis], 2, axis=1)
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
