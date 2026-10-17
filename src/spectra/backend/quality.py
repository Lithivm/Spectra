"""Quality analysis mixin — clipping, upsampling detection, DR, LUFS, true peak."""

from __future__ import annotations

import math

import numpy as np

from . import loudness as r128


class _QualityMixin:
    """Mixed into AudioAnalyzer.  Expects self.data, self.sample_rate, etc."""

    # ------------------------------------------------------------------
    # Quality analysis entry point
    # ------------------------------------------------------------------
    def analyze_quality(self, cancel_check=None) -> dict:
        if self.data is None:
            raise RuntimeError("未加载音频")
        sr = self.sample_rate
        # Full-mix view (channels, samples) for peak / true peak / loudness —
        # must reflect ALL channels, not just channel 0.
        full_mix = self.data if self.data.ndim > 1 else self.data[np.newaxis, :]
        channels = full_mix

        # Peak reflects ALL channels, not just channel 0.
        abs_data = np.abs(self.data)
        peak_val = float(abs_data.max())
        _, sample_of_peak = np.unravel_index(int(np.argmax(abs_data)), self.data.shape)
        peak_idx = int(sample_of_peak)

        tp_val = self._true_peak(full_mix, sr)
        # R128 short-term loudness (all channels), computed once and shared by
        # LRA and dynamic-range (F4: DR = P95-P10 of the same array).
        st_vals = self._short_term_loudness_values(full_mix, sr, cancel_check)
        upsampling_by_channel = [
            self._detect_high_freq_cutoff(channel, sr) for channel in channels
        ]
        upsampling = min(upsampling_by_channel, key=lambda result: result["cutoff_hz"])
        return {
            "clipping":      self._detect_clipping(self.data, sr, self._source_format),
            "upsampling":    upsampling,
            "dynamic_range": self._measure_dynamic_range(full_mix, sr, cancel_check),
            "peak_db":       round(20 * np.log10(peak_val + 1e-12), 1) if peak_val > 0 else -120.0,
            "peak_sample":   peak_idx,
            "true_peak_db":  tp_val,
            "rms":           round(max(self._compute_rms(channel) for channel in channels), 6),
            "zero_crossing": max(self._compute_zcr(channel) for channel in channels),
            "loudness": self._measure_loudness(full_mix, sr, cancel_check, tp_val, st_vals=st_vals),
        }

    # ------------------------------------------------------------------
    # Clipping detection
    # ------------------------------------------------------------------
    def _detect_clipping(self, audio: np.ndarray, sr: int, source_format: str | None) -> dict:
        """Flat-top clipping detection with hard/soft classification.

        Detection:
          - Integer formats: only samples within 2 quantization steps of the code
            ceiling count (a sample merely rounded up to the ceiling is not a clip).
          - Float / unknown source: threshold 0.9999.
          - Consecutive candidates form a clip region (a run).
          - Only runs of >= MIN_RUN consecutive full-scale samples count as clips;
            isolated single samples are informational (single_sample_count).

        Hard vs soft classification (for multi-sample regions):
          - Hard clip: signal is at the ceiling and flat (ptp < flat_thresh).
          - Soft clip: signal is at the ceiling but curved (ptp >= flat_thresh),
            e.g. tube/tape saturation.
        """
        MIN_RUN = 2  # a clip is >= this many consecutive full-scale samples

        # Bit-depth aware: one LSB of the code range (normalised float view)
        _LSB = {
            's16': 1 / 32768, 's16p': 1 / 32768,
            's20': 1 / (1 << 19), 's20p': 1 / (1 << 19),
            's24': 1 / (1 << 23), 's24p': 1 / (1 << 23),
            's32': 1 / (1 << 31), 's32p': 1 / (1 << 31),
        }
        lsb = _LSB.get(source_format or '', None)
        if lsb is not None:
            clip_thresh = 1.0 - 2.0 * lsb      # 顶到码字天花板才算削波
            flat_thresh = 2.0 * lsb
        else:
            clip_thresh = 0.9999               # float / 未知来源
            flat_thresh = 1e-6

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
            over = np.abs(ch_data) >= clip_thresh
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
            ch_longest_ms = round(float((multi_lengths / sr * 1000).max()), 2)

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
        """Detect where the sustained content bandwidth ends (high-freq cutoff).

        ok=False means the content's bandwidth ceiling lies below Nyquist. This
        reports a FACT about the spectrum and does NOT distinguish an artificial
        low-pass filter from naturally bass-heavy / band-limited content — both
        present as "content stops, then a noise floor". See spec F3.

        Method:
          1. Welch PSD (deterministic).
          2. Median-smooth the PSD (~250 Hz) to suppress single-bin estimation
             spikes — the main fix; without it the high→low walk below stops at a
             random floor-region spike and reports a near-Nyquist cutoff.
          3. Walk high→low with hysteresis (K consecutive smoothed bins above the
             noise floor + 6 dB) to locate the edge robustly.
          4. confidence / slope_dboct / gibbs_detected are informational; ok is
             driven by the (now-robust) edge location.
        """
        from scipy.signal import welch, medfilt

        nyq = sr / 2
        if len(audio) < 8192:
            return {"ok": True, "cutoff_hz": nyq, "nyq_hz": nyq,
                    "confidence": 0.0, "slope_dboct": 0.0,
                    "gibbs_detected": False, "method": "welch+multifactor"}

        # ── Step 1: Welch PSD ──
        freqs, psd = welch(audio, fs=sr, nperseg=8192, noverlap=4096, window='hann')
        psd_db = 10 * np.log10(psd + 1e-12)

        if np.max(psd_db) < -110:
            return {"ok": True, "cutoff_hz": nyq, "nyq_hz": nyq,
                    "confidence": 0.0, "slope_dboct": 0.0,
                    "gibbs_detected": False, "method": "welch+multifactor"}

        # ── Step 2: Noise floor & global signal reference (contrast) ──
        noise_floor_db = float(np.percentile(psd_db, 5))
        ref_mask = (freqs >= 2000) & (freqs <= 12000)
        signal_ref_db = (float(np.percentile(psd_db[ref_mask], 90)) if ref_mask.any()
                         else float(np.max(psd_db)))
        contrast_db = signal_ref_db - noise_floor_db
        contrast_score = float(np.clip((contrast_db - 6) / 34, 0.0, 1.0))

        # ── Step 3: Robust edge — median-smooth PSD + hysteresis walk high→low ──
        bin_hz = sr / (len(freqs) - 1)
        k = max(3, int(round(250.0 / bin_hz))) | 1          # ~250 Hz median window
        psd_s = medfilt(psd_db, kernel_size=k)

        THR = 6.0
        K = max(3, int(round(150.0 / bin_hz)))              # ~150 Hz sustained run
        above = psd_s > (noise_floor_db + THR)
        cutoff_hz = nyq
        run = 0
        for i in range(len(above) - 1, -1, -1):
            if above[i]:
                run += 1
                if run >= K:
                    cutoff_hz = float(freqs[min(i + 1, len(freqs) - 1)])
                    break
            else:
                run = 0

        # ── Step 4: Local slope at the detected edge (±~1 octave) ──
        if cutoff_hz < nyq * 0.98:
            lo = max(20.0, cutoff_hz * 0.6)
            hi = min(nyq, cutoff_hz * 1.7)
            m = (freqs >= lo) & (freqs <= hi)
        else:
            m = (freqs >= sr / 6) & (freqs > 0)
        slope_dboct = float(np.polyfit(np.log2(freqs[m]), psd_db[m], 1)[0]) if m.sum() >= 8 else 0.0
        slope_score = float(np.clip(
            (-slope_dboct - (-self._SLOPE_NATURAL_MAX))
            / (-self._SLOPE_CUTOFF_MIN - (-self._SLOPE_NATURAL_MAX)),
            0.0, 1.0,
        ))

        # ── Step 5: Gibbs ringing near the edge (informational) ──
        gibbs_detected = False
        if cutoff_hz < nyq * 0.85:
            lo, hi = cutoff_hz * 0.90, cutoff_hz * 1.10
            window_mask = (freqs >= lo) & (freqs <= hi)
            if window_mask.sum() >= 3:
                window_db = psd_db[window_mask]
                trend = np.linspace(window_db[0], window_db[-1], len(window_db))
                gibbs_detected = float((window_db - trend).max()) > 3.0
        gibbs_score = 1.0 if gibbs_detected else 0.0

        # ── Step 6: confidence (informational) + decision ──
        confidence = (
            0.40 * contrast_score
            + 0.35 * slope_score
            + 0.25 * gibbs_score
        )

        # ok is driven by the robust edge location: content clearly stops more
        # than 15% below Nyquist → band-limited (a fact, not a defect judgement).
        is_cutoff = cutoff_hz < nyq * 0.85

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
    def _measure_dynamic_range(self, audio: np.ndarray, sr: int | None = None, cancel_check=None) -> dict:
        """TT DR: P95 - P10 over 300 ms K-weighted windows, **ungated**.

        This is the TT DR Meter family, not LRA: LRA gates and uses 3 s windows, so the
        two read differently on percussive material. Full-mix view, K-weighted.

        audio: (channels, samples) or (samples,).
        """
        audio = np.asarray(audio, dtype=np.float64)
        if audio.ndim == 1:
            audio = audio[np.newaxis, :]
        sr = int(sr or self.sample_rate)
        blocks = r128.block_loudnesses(audio, sr, 0.3, 0.3, cancel_check)
        if blocks.size == 0:
            return {"dr": 0.0, "dr_basis": "none"}
        return {"dr": self._loudness_percentile_range(blocks),
                "dr_basis": "TT_DR_300ms_ungated", "blocks": int(blocks.size)}

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
    @staticmethod
    def _loudness_percentile_range(st_vals: list[float] | np.ndarray, gate: bool = False) -> float:
        """P95 - P10 of finite K-weighted block loudness values (LU).

        0 when fewer than 3 finite samples. `gate=True` applies the R128 double
        gating first (that's how LRA is defined); DR (TT) uses the ungated values.
        """
        vals = np.array([v for v in st_vals if np.isfinite(v)], dtype=np.float64)
        if len(vals) < 3:
            return 0.0
        if gate:
            vals = r128.gate(vals)
            if len(vals) < 3:
                return 0.0
        p10, p95 = np.percentile(vals, [10, 95])
        return round(float(p95 - p10), 1)

    def _short_term_loudness_values(self, audio_st: np.ndarray, sr: int, cancel_check=None) -> list[float]:
        """R128 short-term loudness over the full mix → list of LUFS.

        3-second K-weighted windows advanced by 1 second (75% overlap), **ungated**
        — R128 defines short-term loudness without gating. Shared by the display
        bars and LRA (which gates them).

        audio: (channels, samples) or (samples,). Computed once over the whole
        file with bounded memory (no per-window re-filtering).
        """
        return [round(float(v), 1) for v in r128.block_loudnesses(audio_st, sr, 3.0, 1.0, cancel_check)]

    def _measure_loudness(self, audio_st: np.ndarray, sr: int, cancel_check=None, true_peak_val: float | None = None, st_vals: list[float] | None = None) -> dict:
        """R128 integrated / short-term loudness plus LRA and true peak.

        K 权重在**原始采样率**上计算。以前降到 12 kHz 分析会削掉 6 kHz 以上，而
        K 权重高架中心在 16.8 kHz —— 高频内容会读低几 dB 到几十 dB。
        积分响度按 R128 双重门控（绝对 -70 LU、相对 -10 LU）；LRA = 门控块的 P10-P95。

        audio: (channels, samples) or (samples,), all channels.
        """
        audio = np.asarray(audio_st, dtype=np.float64)
        if audio.ndim == 1:
            audio = audio[np.newaxis, :]

        blocks = r128.block_loudnesses(audio, sr, 0.4, 0.1, cancel_check)
        if cancel_check is not None and cancel_check():
            return {"error": "cancelled"}
        integrated = r128.gate_mean(blocks) if blocks.size else -math.inf
        if st_vals is None:
            st_vals = self._short_term_loudness_values(audio, sr, cancel_check)
        short_term = max((v for v in st_vals if np.isfinite(v)), default=integrated)
        tp = true_peak_val if true_peak_val is not None else self._true_peak(audio, sr)

        return {
            "integrated_lufs": round(integrated, 1),
            "short_term_lufs": round(short_term, 1),
            "lra_lu": self._loudness_percentile_range(st_vals, gate=True),
            "true_peak_db": tp,
        }

    @staticmethod
    def _true_peak(audio_st: np.ndarray, sr: int) -> float:
        """BS.1770 true peak: 4× oversampled polyphase FIR, chunked (memory O(chunk)).

        分析带覆盖到原始采样率的 Nyquist；以前降到 12 kHz 会削掉 6 kHz 以上。
        audio_st: (channels, samples) or (samples,).
        """
        tp = r128.true_peak_dbtp(audio_st, sr)
        if tp is None:
            return -120.0
        return round(tp, 1)
