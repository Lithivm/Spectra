from __future__ import annotations

import math
import os
import time
import warnings
from pathlib import Path

import pyfftw
import pyfftw.interfaces.scipy_fft as fftw_scipy
pyfftw.interfaces.cache.enable()
pyfftw.config.NUM_THREADS = os.cpu_count() or 4

import librosa
import numpy as np

librosa.set_fftlib(fftw_scipy)

from lang import t
from .load import load_audio
from .metadata import get_metadata

# Module-level STFT result cache — keyed by (filepath, mtime, mode, n_fft)
_stft_cache: dict = {}
_STFT_CACHE_MAX = 2


class AudioAnalyzer:
    """加载并分析音频文件。"""

    def __init__(self, filepath: str | Path | None = None):
        self.filepath: Path | None = None
        self.metadata: dict = {}
        self.data: np.ndarray | None = None
        self.sample_rate: int = 0
        self.channels: int = 0
        self.duration: float = 0.0

        if filepath:
            self.load(filepath)

    # ------------------------------------------------------------------
    # 加载
    # ------------------------------------------------------------------
    def load(self, filepath: str | Path) -> None:
        filepath = Path(filepath)
        self.data, self.sample_rate = load_audio(filepath)
        self.channels = self.data.shape[0] if self.data.ndim > 1 else 1
        self.duration = float(self.data.shape[-1]) / self.sample_rate
        self.filepath = filepath
        self.metadata = get_metadata(filepath)

    # ------------------------------------------------------------------
    # 波形
    # ------------------------------------------------------------------
    @property
    def waveform(self) -> np.ndarray:
        if self.data is None:
            raise RuntimeError("未加载音频")
        data = self.data
        if data.ndim > 1:
            return np.mean(data, axis=0)
        return data

    def get_waveform_range(self) -> tuple[float, float]:
        w = self.waveform
        return float(np.min(w)), float(np.max(w))

    # ------------------------------------------------------------------
    # 频谱 (STFT)
    # ------------------------------------------------------------------
    def stft(
        self,
        n_fft: int = 2048,
        hop_length: int | None = None,
        win_length: int | None = None,
        window: str = "hann",
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if self.data is None:
            raise RuntimeError("未加载音频")
        hop = hop_length or (n_fft // 4)   # 75% overlap
        if win_length is None:
            win_length = n_fft
        audio = self.data[0] if self.data.ndim > 1 else self.data
        S = librosa.stft(audio, n_fft=n_fft, hop_length=hop,
                         win_length=win_length, window=window)
        freqs = librosa.fft_frequencies(sr=self.sample_rate, n_fft=n_fft)
        times = librosa.frames_to_time(
            np.arange(S.shape[1]), sr=self.sample_rate, hop_length=hop)
        return freqs, times, S

    def spectrogram_db(
        self,
        n_fft: int = 2048,
        hop_length: int | None = None,
        win_length: int | None = None,
        window: str = "hann",
        mode: str = "standard",
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute dB spectrogram.

        Parameters
        ----------
        mode : str
            "standard" — single fixed-size STFT (default).
            "multi"    — multi-resolution: large FFT at lows, small at highs.
            "reassign" — phase-reassigned spectrogram for sharper harmonics.
        """
        print(f"[FFT backend] {librosa.get_fftlib()}")
        # Check cache — keyed by (filepath, mode, n_fft), single-entry
        if self.filepath and self.filepath.exists():
            cache_key = (str(self.filepath), mode, n_fft)
            if cache_key in _stft_cache:
                print(f"[PROFILE] STFT cache hit ({mode}, n_fft={n_fft})")
                return _stft_cache[cache_key]

        t0 = time.perf_counter()
        if mode == "multi":
            result = self._multi_resolution_stft(window)
        elif mode == "reassign":
            result = self._reassigned_spectrogram(n_fft, hop_length, win_length, window)
        else:
            freqs, times, S = self.stft(n_fft, hop_length, win_length, window)
            result = (freqs, times, librosa.amplitude_to_db(np.abs(S), ref=np.max))

        print(f"[TIMER] 计算结果: {time.perf_counter()-t0:.2f}s")

        # ── Downsample time axis ──
        TARGET_FRAMES = 16384
        freqs, times, db = result
        if db.shape[1] > TARGET_FRAMES:
            chunk = db.shape[1] // TARGET_FRAMES
            db = db[:, :chunk * TARGET_FRAMES].reshape(
                db.shape[0], TARGET_FRAMES, chunk
            ).max(axis=2)
            times = times[:chunk * TARGET_FRAMES].reshape(
                TARGET_FRAMES, chunk
            ).mean(axis=1)
            result = (freqs, times.astype(np.float64), db)

        print(f"[TIMER] 降采样: {time.perf_counter()-t0:.2f}s")
        print(f"[TIMER] 降采样后: {time.perf_counter()-t0:.2f}s")

        if self.filepath:
            _stft_cache.clear()
            _stft_cache[(str(self.filepath), mode, n_fft)] = result

        return result

    # ------------------------------------------------------------------
    # Multi-Resolution STFT
    # ------------------------------------------------------------------
    def _multi_resolution_stft(
        self,
        window: str = "hann",
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Three-band multi-resolution STFT.

        Low  (0–300 Hz)   : n_fft=8192 — fine frequency resolution for bass.
        Mid  (300–3k Hz)  : n_fft=2048 — balanced.
        High (3k–Nyquist) : n_fft=512  — sharp transients.
        """
        audio = self.data[0] if self.data.ndim > 1 else self.data
        sr = self.sample_rate
        nyq = sr / 2.0

        bands = [
            ("low",  8192, 0.0,   320.0),
            ("mid",  2048, 280.0, 3200.0),
            ("high",  512, 2800.0, nyq),
        ]

        hop = 512
        results: list[np.ndarray] = []
        band_freqs: list[np.ndarray] = []

        for _name, n_fft, f_lo, f_hi in bands:
            f_hi = min(f_hi, nyq)
            S = librosa.stft(audio, n_fft=n_fft, hop_length=hop,
                             win_length=n_fft, window=window, center=True)
            freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
            mag = np.abs(S)
            mask = (freqs >= f_lo) & (freqs < f_hi)
            if not np.any(mask):
                continue
            band_freqs.append(freqs[mask])
            results.append(mag[mask, :])
        if not results:
            raise RuntimeError("Multi-resolution STFT produced no bands")
        mag_combined = np.vstack(results).astype(np.float32)
        freqs_combined = np.concatenate(band_freqs)

        if not np.all(np.diff(freqs_combined) > 0):
            mono_freqs = [freqs_combined[0]]
            for f in freqs_combined[1:]:
                if f > mono_freqs[-1]:
                    mono_freqs.append(f)
            freqs_combined = np.array(mono_freqs, dtype=np.float64)
            mag_combined = mag_combined[:len(freqs_combined), :]

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            db = librosa.amplitude_to_db(mag_combined, ref=np.max(mag_combined), top_db=None)

        times = librosa.frames_to_time(
            np.arange(db.shape[1]), sr=sr, hop_length=hop)

        return freqs_combined, times, db

    # ------------------------------------------------------------------
    # Reassigned Spectrogram (iZotope RX style)
    # ------------------------------------------------------------------
    def _reassigned_spectrogram(
        self,
        n_fft: int = 4096,
        hop_length: int | None = None,
        win_length: int | None = None,
        window: str = "hann",
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Phase-reassigned spectrogram — ultra-sharp time-frequency rendering.

        Relocates STFT energy from grid centres to true instantaneous-frequency /
        group-delay coordinates using first-order phase derivatives (Auger-Flandrin
        method, IEEE TASSP 1995).
        """
        audio = self.data[0] if self.data.ndim > 1 else self.data
        sr = self.sample_rate
        hop = hop_length or (n_fft // 4)   # 75% overlap
        wlen = win_length or n_fft

        # ── STFT magnitude ──
        S = librosa.stft(audio, n_fft=n_fft, hop_length=hop,
                         win_length=wlen, window=window, center=True)

        # ── Time-derivative STFT: multiply signal by time ramp ──
        t_ramp = np.linspace(-1, 1, len(audio))
        S_t = librosa.stft(audio * t_ramp, n_fft=n_fft, hop_length=hop,
                           win_length=wlen, window=window, center=True)

        # ── Frequency-derivative STFT: multiply window by freq ramp ──
        win = librosa.filters.get_window(window, wlen)
        if wlen < n_fft:
            win = np.pad(win, (0, n_fft - wlen))
        w_ramp = (np.arange(n_fft) - n_fft / 2.0) / (n_fft / 2.0)
        win_f = win * w_ramp
        S_f = librosa.stft(audio, n_fft=n_fft, hop_length=hop,
                           win_length=n_fft, window=win_f, center=True)

        # ── Reassignment coordinates ──
        eps = 1e-10
        S_sq = np.abs(S) ** 2 + eps

        # Instantaneous frequency: ω̂ = ω + Im{S_t·S* / |S|²} / 2π
        omega_corr = np.imag(S_t * np.conj(S) / S_sq) / (2.0 * np.pi)

        # Group delay: τ̂ = t + Re{S_f·S* / |S|²}
        tau_corr = np.real(S_f * np.conj(S) / S_sq)

        freqs_base = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
        n_freqs = len(freqs_base)
        n_frames = S.shape[1]

        mag = np.abs(S).astype(np.float64)

        # Build reassigned spectrogram by accumulating magnitude into
        # corrected time-frequency bins
        mag_acc = np.zeros_like(mag)
        weight = np.zeros_like(mag)

        for k in range(n_freqs):
            # Target frequency indices for this bin's energy
            f_new = freqs_base[k] + omega_corr[k, :] * sr
            f_new = np.clip(f_new, freqs_base[0], freqs_base[-1])
            f_idx = np.searchsorted(freqs_base, f_new).clip(0, n_freqs - 1)

            # Target time indices
            t_new = np.arange(n_frames) + tau_corr[k, :] * (hop / sr)
            t_new = np.clip(t_new, 0, n_frames - 1)
            t_idx = np.round(t_new).astype(int).clip(0, n_frames - 1)

            np.add.at(mag_acc, (f_idx, t_idx), mag[k, :])
            np.add.at(weight, (f_idx, t_idx), 1.0)

        mag_acc /= np.maximum(weight, 1.0)

        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            db = librosa.amplitude_to_db(mag_acc, ref=np.max(mag_acc), top_db=None)

        times = librosa.frames_to_time(
            np.arange(n_frames), sr=sr, hop_length=hop)

        return freqs_base.astype(np.float64), times, db.astype(np.float32)

    def melspectrogram_db(
        self,
        n_mels: int = 512,
        n_fft: int = 4096,
        hop_length: int | None = None,
        fmin: float = 40.0,
        fmax: float | None = None,
        window: str = "hann",
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute mel-scale spectrogram → dB.  Returns (mel_freqs, times, db)."""
        if self.data is None:
            raise RuntimeError("未加载音频")
        hop = hop_length or (n_fft // 8)
        audio = self.data[0] if self.data.ndim > 1 else self.data
        fmax = fmax or (self.sample_rate / 2.0)

        S = librosa.feature.melspectrogram(
            y=audio, sr=self.sample_rate,
            n_mels=n_mels, n_fft=n_fft, hop_length=hop,
            fmin=fmin, fmax=fmax, window=window,
        )
        db = librosa.power_to_db(S, ref=np.max)
        mel_freqs = librosa.mel_frequencies(n_mels=n_mels, fmin=fmin, fmax=fmax)
        times = librosa.frames_to_time(
            np.arange(S.shape[1]), sr=self.sample_rate, hop_length=hop)
        return mel_freqs, times, db

    # ------------------------------------------------------------------
    # MFCC
    # ------------------------------------------------------------------
    def mfcc(self, n_mfcc: int = 13) -> np.ndarray:
        if self.data is None:
            raise RuntimeError("未加载音频")
        audio = self.data[0] if self.data.ndim > 1 else self.data
        return librosa.feature.mfcc(y=audio, sr=self.sample_rate, n_mfcc=n_mfcc)

    # ------------------------------------------------------------------
    # RMS
    # ------------------------------------------------------------------
    def rms(self, frame_length: int = 2048, hop_length: int | None = None) -> np.ndarray:
        hop = hop_length or (frame_length // 4)
        audio = self.data[0] if self.data.ndim > 1 else self.data
        return librosa.feature.rms(y=audio, frame_length=frame_length, hop_length=hop)[0]

    # ------------------------------------------------------------------
    # 频谱质心
    # ------------------------------------------------------------------
    def spectral_centroid(self) -> tuple[np.ndarray, np.ndarray]:
        if self.data is None:
            raise RuntimeError("未加载音频")
        audio = self.data[0] if self.data.ndim > 1 else self.data
        centroid = librosa.feature.spectral_centroid(y=audio, sr=self.sample_rate)[0]
        t = np.linspace(0, self.duration, len(centroid))
        return t, centroid

    # ------------------------------------------------------------------
    # Zero Crossing Rate
    # ------------------------------------------------------------------
    def zcr(self) -> tuple[np.ndarray, np.ndarray]:
        zcr = librosa.feature.zero_crossing_rate(self.waveform)[0]
        t = np.linspace(0, self.duration, len(zcr))
        return t, zcr

    # ------------------------------------------------------------------
    # 信息总结
    # ------------------------------------------------------------------
    def info(self) -> dict:
        info: dict = {
            t("文件名", "Filename"): str(self.filepath.name) if self.filepath else "—",
            t("时长", "Duration"): f"{self.duration:.2f} s",
            t("采样率", "Sample Rate"): f"{self.sample_rate} Hz",
            t("声道", "Channels"): self.channels,
        }
        if self.metadata:
            for k in ("标题", "艺术家", "专辑", "年份", "流派"):
                if self.metadata.get(k):
                    info[t(k, k)] = self.metadata[k]
        return info

    # ------------------------------------------------------------------
    # 质量分析
    # ------------------------------------------------------------------
    def analyze_quality(self) -> dict:
        if self.data is None:
            raise RuntimeError("未加载音频")
        audio = self.data[0] if self.data.ndim > 1 else self.data
        sr = self.sample_rate
        # 只取前60秒做分析，避免全曲FFT耗时过长
        max_samples = min(len(audio), 60 * sr)
        audio = audio[:max_samples]
        peak_val, peak_idx = self._compute_peak(audio)
        return {
            "clipping":      self._detect_clipping(audio, sr),
            "upsampling":    self._detect_high_freq_cutoff(audio, sr),
            "dynamic_range": self._measure_dynamic_range(audio),
            "peak_db":       round(20 * np.log10(peak_val + 1e-12), 1),
            "peak_sample":   peak_idx,
            "rms":           round(self._compute_rms(audio), 6),
            "zero_crossing": self._compute_zcr(audio),
            "freq_range":    self._compute_freq_range(audio),
            "bit_depth":     self._estimate_bit_depth(audio),
            "loudness_lufs": self._estimate_loudness(audio),
        }

    def _detect_clipping(self, audio: np.ndarray, sr: int) -> dict:
        threshold = 0.999
        clipped = np.abs(audio) >= threshold
        diff = np.diff(clipped.astype(int))
        clip_starts = np.where(diff > 0)[0] + 1
        clip_ends = np.where(diff < 0)[0]
        if len(clip_starts) > 0 and len(clip_ends) == 0:
            clip_ends = np.array([len(audio) - 1])
        elif len(clip_starts) > len(clip_ends):
            clip_ends = np.append(clip_ends, len(audio) - 1)
        long_clips = [(s, e) for s, e in zip(clip_starts, clip_ends) if (e - s) >= 3]
        if not long_clips:
            return {"ok": True, "count": 0, "longest_ms": 0}
        durations_ms = [int((e - s) / sr * 1000) for s, e in long_clips]
        return {"ok": False, "count": len(long_clips), "longest_ms": max(durations_ms)}

    def _detect_high_freq_cutoff(self, audio: np.ndarray, sr: int) -> dict:
        nyq = sr / 2
        n = len(audio)
        S = np.fft.rfft(audio)
        mag = np.abs(S) ** 2
        freqs = np.fft.rfftfreq(n, 1.0 / sr)

        # 只看70%~Nyquist的频段，索引统一在freqs/mag上
        low_bin = int(len(freqs) * 0.7)
        high_mag = mag[low_bin:]
        high_freqs = freqs[low_bin:]

        if len(high_mag) == 0 or np.max(high_mag) < 1e-12:
            return {"ok": True, "cutoff_hz": nyq}

        high_db = 10 * np.log10(high_mag + 1e-12)
        max_db = np.max(high_db)
        threshold_db = max_db - 25
        idx = np.where(high_db < threshold_db)[0]

        if len(idx) == 0:
            return {"ok": True, "cutoff_hz": nyq}

        # 直接用high_freqs索引，不加low_bin偏移
        cutoff_hz = float(high_freqs[idx[0]])

        if cutoff_hz < 0.9 * nyq:
            return {"ok": False, "cutoff_hz": cutoff_hz, "sr_hz": sr}
        return {"ok": True, "cutoff_hz": min(cutoff_hz, nyq)}

    def _measure_dynamic_range(self, audio: np.ndarray) -> dict:
        fft_size = 2048
        hop = fft_size // 4
        frame_count = int(np.ceil(len(audio) / hop))
        frame_start = np.arange(frame_count) * hop
        frame_end = np.minimum(frame_start + fft_size, len(audio))
        frames_db = []
        for i in range(frame_count):
            f = audio[frame_start[i]:frame_end[i]]
            rms = float(np.sqrt(np.mean(f ** 2)))
            if rms > 0:
                frames_db.append(20 * np.log10(rms))
        if not frames_db:
            return {"dr": 0.0}
        top = float(np.max(frames_db))
        bottom = float(np.mean(frames_db))
        return {"dr": round(top - bottom, 1)}

    def _compute_rms(self, audio: np.ndarray) -> float:
        return float(np.sqrt(np.mean(audio ** 2)))

    def _compute_peak(self, audio: np.ndarray) -> tuple[float, int]:
        idx = int(np.argmax(np.abs(audio)))
        return float(np.abs(audio[idx])), idx

    def _compute_zcr(self, audio: np.ndarray) -> int:
        return int(np.sum(np.abs(np.diff(np.signbit(audio)))))

    def _compute_freq_range(self, audio: np.ndarray) -> tuple[float, float]:
        S = np.fft.rfft(audio)
        mag = np.abs(S) ** 2
        mag_db = 10 * np.log10(mag + 1e-12)
        threshold = np.max(mag_db) - 20
        idx = np.where(mag_db >= threshold)[0]
        freqs = np.fft.rfftfreq(len(audio), 1.0 / self.sample_rate)
        if len(idx) == 0:
            return 0.0, 0.0
        return float(freqs[idx[0]]), float(freqs[idx[-1]])

    def _estimate_bit_depth(self, audio: np.ndarray) -> int:
        """粗略估计 bit depth，安全处理inf/overflow。"""
        peak = float(np.max(np.abs(audio)))
        if peak <= 0 or peak < 1e-10:
            return 32
        try:
            log_val = math.log2(1.0 / peak)
            depth = int(math.ceil(log_val))
            return min(max(depth, 0), 32)
        except (ValueError, OverflowError):
            return 32

    def _estimate_loudness(self, audio: np.ndarray) -> float:
        rms = self._compute_rms(audio)
        if rms <= 0:
            return -math.inf
        return round(20 * math.log10(rms), 1)
