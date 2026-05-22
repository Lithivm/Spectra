from __future__ import annotations

import atexit
import math
import os
import sys
import threading
import warnings
from pathlib import Path

import pyfftw
import pyfftw.interfaces.scipy_fft as fftw_scipy
pyfftw.interfaces.cache.enable()
pyfftw.config.NUM_THREADS = os.cpu_count() or 4

_wisdom_path = os.path.join(os.path.expanduser('~'), '.spectra', 'fftw_wisdom.pkl')
os.makedirs(os.path.dirname(_wisdom_path), exist_ok=True)

_wisdom_dirty = False
_wisdom_loaded = False
try:
    import pickle
    if os.path.exists(_wisdom_path):
        with open(_wisdom_path, "rb") as f:
            pyfftw.import_wisdom(pickle.load(f))
        _wisdom_loaded = True
except Exception:
    pass


def _flush_wisdom() -> None:
    """Persist FFTW wisdom to disk once at exit."""
    global _wisdom_dirty
    if not _wisdom_dirty:
        return
    try:
        with open(_wisdom_path, "wb") as f:
            pickle.dump(pyfftw.export_wisdom(), f)
    except Exception:
        pass
    _wisdom_dirty = False


atexit.register(_flush_wisdom)

if not _wisdom_loaded:
    try:
        _bundled = os.path.join(sys._MEIPASS, 'analyzer', 'fftw_wisdom.pkl')
        if os.path.exists(_bundled):
            with open(_bundled, "rb") as f:
                pyfftw.import_wisdom(pickle.load(f))
            import shutil
            shutil.copy2(_bundled, _wisdom_path)
    except Exception:
        pass

import librosa
import numpy as np

import pyloudnorm as pyln

librosa.set_fftlib(fftw_scipy)

from lang import t

_TAG_TR = {
    "标题": ("标题", "Title"),
    "艺术家": ("艺术家", "Artist"),
    "专辑": ("专辑", "Album"),
    "年份": ("年份", "Year"),
    "流派": ("流派", "Genre"),
    "音轨": ("音轨", "Track"),
    "碟片": ("碟片", "Disc"),
    "作曲": ("作曲", "Composer"),
    "格式": ("格式", "Format"),
}
from .load import load_audio
from .metadata import get_metadata

# Module-level STFT result cache — keyed by (filepath, mode, n_fft)
_stft_cache: dict = {}
_stft_lock = threading.Lock()


def _max_reduce_with_carry(block, factor, carry):
    """Collapse *factor* adjacent columns via element-wise max.

    Args:
        block: ``(n_freqs, cnt)`` float32 ndarray — newly arrived columns.
        factor: how many raw columns to collapse into one output column.
        carry: ``(n_freqs, leftover)`` or ``None`` — columns left over
               from the previous block that haven't yet filled a full group.

    Returns:
        ``(reduced, new_carry)`` where *reduced* is ``(n_freqs, out_cnt)``
        and *new_carry* is ``None`` or the residual columns.
    """
    n_freqs, cnt = block.shape

    if carry is not None:
        block = np.column_stack([carry, block])
        cnt += carry.shape[1]

    out_cnt = cnt // factor
    leftover = cnt % factor

    if out_cnt > 0:
        usable = cnt - leftover
        reshaped = block[:, :usable].reshape(n_freqs, out_cnt, factor)
        reduced = reshaped.max(axis=2).astype(np.float32)
        new_carry = block[:, usable:].copy() if leftover > 0 else None
    else:
        reduced = np.empty((n_freqs, 0), dtype=np.float32)
        new_carry = block.copy()

    return reduced, new_carry


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

    TARGET_FRAMES = 16384

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
        if self.filepath and self.filepath.exists():
            cache_key = (str(self.filepath), mode, n_fft)
            with _stft_lock:
                if cache_key in _stft_cache:
                    return _stft_cache[cache_key]

        # ── Adaptive hop: aim for TARGET_FRAMES directly, skip downsampling ──
        if hop_length is None:
            if self.duration > 0:
                target_hop = int(self.duration * self.sample_rate / self.TARGET_FRAMES)
                hop = max(n_fft // 8, min(n_fft, target_hop))
            else:
                hop = n_fft // 4
        else:
            hop = hop_length

        if mode == "multi":
            result = self._multi_resolution_stft(window)
        elif mode == "reassign":
            result = self._reassigned_spectrogram(n_fft, hop, win_length, window)
        else:
            freqs, times, S = self.stft(n_fft, hop, win_length, window)
            result = (freqs, times, librosa.amplitude_to_db(np.abs(S), ref=np.max, top_db=None))

        # ── Downsample time axis (only if adaptive hop overshot target) ──
        freqs, times, db = result
        if db.shape[1] > self.TARGET_FRAMES:
            chunk = db.shape[1] // self.TARGET_FRAMES
            db = db[:, :chunk * self.TARGET_FRAMES].reshape(
                db.shape[0], self.TARGET_FRAMES, chunk
            ).max(axis=2)
            times = times[:chunk * self.TARGET_FRAMES].reshape(
                self.TARGET_FRAMES, chunk
            ).mean(axis=1)
            result = (freqs, times.astype(np.float64), db)

        if self.filepath:
            with _stft_lock:
                _stft_cache[(str(self.filepath), mode, n_fft)] = result

        global _wisdom_dirty
        _wisdom_dirty = True

        return result

    # ------------------------------------------------------------------
    # Streaming Spectrogram (Spek-style scroll render)
    # ------------------------------------------------------------------
    def spectrogram_db_streaming(
        self, *,
        n_fft: int = 2048,
        hop_length: int | None = None,
        win_length: int | None = None,
        window: str = "hann",
        block_cols: int = 64,
        on_init,             # (freqs, total_cols, hop) -> None
        on_block,            # (start_col, block_db) -> None
        cancel_check=lambda: False,
    ):
        """Compute dB spectrogram in a streaming fashion.

        Returns ``(freqs, times, full_db)`` on success, ``None`` if cancelled
        or if the file is too short for meaningful streaming.

        Uses absolute dBFS reference (ref = 1.0) rather than per-file peak
        normalisation, because the global maximum is not known ahead of time.
        The shader's vmin/vmax range (-120..0 dB) absorbs the offset.
        """
        from scipy.signal import get_window

        audio = self.data[0] if self.data.ndim > 1 else self.data
        sr = self.sample_rate
        N_orig = len(audio)
        n_fft_w = win_length or n_fft
        n_freqs = n_fft // 2 + 1

        # ── Hop calculation (same as spectrogram_db) ──
        if hop_length is None:
            if self.duration > 0:
                target_hop = int(self.duration * sr / self.TARGET_FRAMES)
                hop = max(n_fft // 8, min(n_fft, target_hop))
            else:
                hop = n_fft // 4
        else:
            hop = hop_length

        pad = n_fft // 2
        audio_padded = np.pad(audio.astype(np.float32), pad, mode='reflect')
        N_pad = len(audio_padded)

        raw_cols = 1 + (N_pad - n_fft) // hop

        # ── Short file: signal caller to use non-streaming path ──
        if raw_cols < block_cols * 2:
            return None

        # ── Downsample factor (keep output cols ≤ TARGET_FRAMES) ──
        downsample = 1
        if raw_cols > self.TARGET_FRAMES:
            downsample = raw_cols // self.TARGET_FRAMES + 1
        total_cols = (raw_cols + downsample - 1) // downsample

        freqs = np.fft.rfftfreq(n_fft, 1.0 / sr).astype(np.float64)

        # ── Window (pre-compute, shared across all blocks) ──
        win = get_window(window, n_fft_w).astype(np.float32)
        if n_fft_w < n_fft:
            win = np.pad(win, (0, n_fft - n_fft_w))

        on_init(freqs, total_cols, hop)

        # ── FFTW batch plan ──
        buf_time = pyfftw.empty_aligned((n_fft, block_cols), dtype='float32')
        buf_freq = pyfftw.empty_aligned((n_freqs, block_cols), dtype='complex64')
        # Batch of 64 columns — 2–4 threads avoids oversubscription
        # when quality analysis (LUFS etc.) runs concurrently.
        _fft_threads = max(1, min(4, pyfftw.config.NUM_THREADS // 2))
        fft = pyfftw.FFTW(buf_time, buf_freq, axes=(0,), direction='FFTW_FORWARD',
                          flags=('FFTW_MEASURE',), threads=_fft_threads)

        # ── Streaming loop ──
        acc = np.zeros((n_freqs, total_cols), dtype=np.float32)
        carry = None
        out_col = 0
        eps = np.finfo(np.float32).eps

        for c0 in range(0, raw_cols, block_cols):
            if cancel_check():
                return None

            cnt = min(block_cols, raw_cols - c0)
            offset = c0 * hop

            # Fill time buffer (one frame per column)
            for j in range(cnt):
                s = offset + j * hop
                buf_time[:, j] = audio_padded[s:s + n_fft] * win
            if cnt < block_cols:
                buf_time[:, cnt:] = 0.0

            fft()  # GIL-released batch FFT

            mag = np.abs(buf_freq[:, :cnt])
            block_db = (20.0 * np.log10(np.maximum(mag / n_fft, eps))).astype(np.float32)

            if downsample > 1:
                block_db, carry = _max_reduce_with_carry(block_db, downsample, carry)

            bw = block_db.shape[1]
            if bw > 0:
                acc[:, out_col:out_col + bw] = block_db
                on_block(out_col, block_db)
                out_col += bw

        # Flush remaining carry
        if carry is not None and out_col < total_cols:
            carry_col = carry[:, 0:1] if carry.ndim > 1 else carry.reshape(-1, 1)
            acc[:, out_col] = carry_col[:, 0]
            on_block(out_col, carry_col)
            out_col += 1

        acc = acc[:, :out_col]
        times = librosa.frames_to_time(
            np.arange(out_col), sr=sr, hop_length=hop * downsample)

        # Write back to cache so second load is instant
        if self.filepath:
            with _stft_lock:
                _stft_cache[(str(self.filepath), "standard", n_fft)] = (freqs, times, acc)

        global _wisdom_dirty
        _wisdom_dirty = True

        return freqs, times, acc

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
        # corrected time-frequency bins — vectorised over all bins.
        mag_acc = np.zeros_like(mag)
        weight = np.zeros_like(mag)

        # Frequency corrections: shape (n_freqs, n_frames)
        f_new_all = freqs_base[:, None] + omega_corr * sr
        f_new_all = np.clip(f_new_all, freqs_base[0], freqs_base[-1])
        f_idx_all = np.searchsorted(freqs_base, f_new_all.ravel())
        f_idx_all = f_idx_all.clip(0, n_freqs - 1).reshape(n_freqs, n_frames)

        # Time corrections: shape (n_freqs, n_frames)
        t_new_all = np.arange(n_frames)[None, :] + tau_corr * (hop / sr)
        t_new_all = np.clip(t_new_all, 0, n_frames - 1)
        t_idx_all = np.round(t_new_all).astype(int).clip(0, n_frames - 1)

        np.add.at(mag_acc, (f_idx_all.ravel(), t_idx_all.ravel()), mag.ravel())
        np.add.at(weight, (f_idx_all.ravel(), t_idx_all.ravel()), 1.0)

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
        db = librosa.power_to_db(S, ref=np.max, top_db=None)
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
            t("时长", "Duration"): f"{int(self.duration // 60)}m {int(self.duration % 60)}s",
            t("采样率", "Sample Rate"): f"{self.sample_rate} Hz",
            t("声道", "Channels"): self.channels,
        }
        if self.data is not None:
            dtype = self.data.dtype
            if np.issubdtype(dtype, np.integer):
                info[t("位深", "Bit Depth")] = str(np.iinfo(dtype).bits)
            elif np.issubdtype(dtype, np.floating):
                info[t("位深", "Bit Depth")] = f"{np.finfo(dtype).bits}-bit float"
        if self.metadata:
            for k in ("标题", "艺术家", "专辑", "年份", "流派"):
                if self.metadata.get(k):
                    zh, en = _TAG_TR.get(k, (k, k))
                    info[t(zh, en)] = self.metadata[k]
        return info

    # ------------------------------------------------------------------
    # 质量分析
    # ------------------------------------------------------------------
    def analyze_quality(self, cancel_check=None) -> dict:
        if self.data is None:
            raise RuntimeError("未加载音频")
        audio = self.data[0] if self.data.ndim > 1 else self.data
        sr = self.sample_rate
        peak_val, peak_idx = self._compute_peak(audio)
        # True peak via 4x oversampling (independent of loudness measurement)
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

    def _detect_clipping(self, audio: np.ndarray, sr: int) -> dict:
        """Flat-top + intersample-peak clipping detection.

        Uses flat-top detection (consecutive identical samples near full-scale)
        as the primary criterion — this is more robust than a simple amplitude
        threshold and works across 16-bit / 24-bit / float sources.
        """
        eps = np.finfo(np.float32).eps * 10  # ~1.2e-6 — flat-top tolerance
        MIN_FLAT = 3  # minimum consecutive flat samples
        # Threshold for candidate samples — conservative for float audio
        abs_audio = np.abs(audio)
        over = abs_audio >= 0.999

        # ── Flat-top detection ──
        # Find runs where |diff| < eps AND both samples are over threshold
        diff = np.abs(np.diff(audio))
        flat_mask = (diff < eps) & over[:-1] & over[1:]

        # Find contiguous flat regions (run-length encoding)
        flat_starts: list[int] = []
        flat_ends: list[int] = []
        i = 0
        while i < len(flat_mask):
            if flat_mask[i]:
                start = i
                while i < len(flat_mask) and flat_mask[i]:
                    i += 1
                end = i  # inclusive end (last flat sample index)
                if end - start + 1 >= MIN_FLAT:
                    flat_starts.append(start)
                    flat_ends.append(end)
            else:
                i += 1

        if not flat_starts:
            return {"ok": True, "count": 0, "longest_ms": 0, "method": "flat-top"}

        durations_ms = [int((e - s + 1) / sr * 1000) for s, e in zip(flat_starts, flat_ends)]
        longest_ms = max(durations_ms)

        # ── Classify: hard vs soft clipping ──
        # Hard clipping: all samples in the flat region are within eps of each other
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

    def _detect_high_freq_cutoff(self, audio: np.ndarray, sr: int) -> dict:
        """Multi-segment median-spectrum cutoff detection.

        Uses 6 random 1.5s segments → median power spectrum → slope analysis
        on log-frequency bins. Much more robust than a single-segment FFT.
        """
        import scipy.ndimage as ndi

        nyq = sr / 2
        seg_dur = 1.5  # seconds per segment
        seg_len = int(seg_dur * sr)
        n_segs = min(8, max(3, len(audio) // seg_len))

        if n_segs <= 1 or len(audio) < seg_len:
            segments = [audio]
        else:
            # Random start positions, avoiding boundaries
            max_start = len(audio) - seg_len
            rng = np.random.default_rng(42)
            starts = sorted(rng.integers(0, max(1, max_start), size=n_segs))
            segments = [audio[s:s + seg_len] for s in starts]

        # Compute power spectra for each segment
        spectra = []
        for seg in segments:
            S = np.abs(np.fft.rfft(seg * np.hanning(len(seg)))) ** 2
            spectra.append(S)

        # Align to the shortest spectrum (in case of different segment lengths)
        min_len = min(len(s) for s in spectra)
        spectra = [s[:min_len] for s in spectra]

        # Median power spectrum — robust against outlier segments
        med_spec = np.median(np.array(spectra), axis=0)
        freqs = np.fft.rfftfreq(len(segments[0]), 1.0 / sr)[:min_len]

        if np.max(med_spec) < 1e-12:
            return {"ok": True, "cutoff_hz": nyq, "confidence": 0.0, "method": "multi-seg median"}

        # ── Log-frequency rebinning ──
        f_min = max(freqs[0], 100.0)
        f_max = nyq
        n_bins = 128
        log_edges = np.logspace(np.log10(f_min), np.log10(f_max), n_bins + 1)
        log_centers = np.sqrt(log_edges[:-1] * log_edges[1:])

        binned_energy = np.zeros(n_bins)
        for i in range(n_bins):
            mask = (freqs >= log_edges[i]) & (freqs < log_edges[i + 1])
            if np.any(mask):
                binned_energy[i] = np.mean(med_spec[mask])

        binned_db = 10 * np.log10(binned_energy + 1e-12)
        # Gaussian smooth on log axis
        binned_db = ndi.gaussian_filter1d(binned_db, sigma=1.5)

        # ── Slope-based cutoff detection ──
        # Look at upper half of frequency range
        hi_start = n_bins // 2
        hi_db = binned_db[hi_start:]
        hi_freqs = log_centers[hi_start:]

        # First derivative (slope) on log-frequency axis
        slope = np.diff(hi_db)
        slope = np.append(slope, slope[-1])  # pad to same length

        # Mean and std of slope for finding anomalies
        slope_mean = np.mean(slope)
        slope_std = np.std(slope)

        # A sudden negative slope (drop) indicates cutoff
        # Find where slope drops below mean - 2*std
        steep_drop = slope < (slope_mean - 2.5 * slope_std)

        cutoff_hz = nyq
        if np.any(steep_drop):
            # Find the first significant drop in the upper half
            drop_idx = int(np.argmax(steep_drop))
            cutoff_hz = float(hi_freqs[drop_idx])

        # ── Confidence: how likely this IS a real cutoff ──
        # Low energy variance in high freqs → flat spectrum → real cutoff (high confidence)
        # High variance → natural harmonic decay → not a cutoff (low confidence)
        hi_energy = binned_energy[hi_start:]
        if np.mean(hi_energy) > 1e-12:
            energy_cv = float(np.std(hi_energy) / np.mean(hi_energy))  # coefficient of variation
            cutoff_confidence = max(0.0, min(1.0, 1.0 - energy_cv * 2.0))
        else:
            cutoff_confidence = 1.0  # no energy at all → definitely cutoff

        # ── Decision ──
        # Require BOTH: cutoff significantly below Nyquist AND high confidence
        cutoff_significant = cutoff_hz < nyq * 0.9
        cutoff_confident = cutoff_confidence > 0.3
        is_cutoff = cutoff_significant and cutoff_confident

        return {
            "ok": not is_cutoff,
            "cutoff_hz": cutoff_hz,
            "nyq_hz": nyq,
            "method": "multi-seg median + slope",
        }

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

    def _measure_loudness(self, audio: np.ndarray, sr: int, cancel_check=None) -> dict:
        """EBU R128 integrated loudness, short-term, LRA, true-peak (BS.1770-4)."""
        if audio.ndim == 1:
            audio_st = np.column_stack([audio, audio])
        else:
            audio_st = audio.T[:, :2]
        if audio_st.shape[1] == 1:
            audio_st = np.column_stack([audio_st[:, 0], audio_st[:, 0]])

        # ── Decimate to ~12 kHz for LUFS (K-weighting passband ≪ 6 kHz) ──
        TARGET_SR = 12000
        if sr > TARGET_SR:
            from scipy.signal import decimate
            factor = sr // TARGET_SR
            meter_sr = sr // factor
            ch0 = decimate(audio_st[:, 0].astype(np.float64), factor, zero_phase=True)
            audio_meter = np.column_stack([ch0, ch0]).astype(np.float64)
        else:
            meter_sr = sr
            audio_meter = audio_st.astype(np.float64)

        meter = pyln.Meter(meter_sr)
        integrated = float(meter.integrated_loudness(audio_meter))

        # Short-term loudness: 3-second non-overlapping blocks
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

        # Loudness range (LRA): P10–P95 percentile spread
        if len(st_vals) >= 3:
            sv = sorted(st_vals)
            p10 = sv[int(len(sv) * 0.1)]
            p95 = sv[int(len(sv) * 0.95)]
            lra = round(p95 - p10, 1)
        else:
            lra = 0.0

        # True peak — measured on ORIGINAL full-bandwidth signal (BS.1770-4)
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
        # Resample each channel, take max across all
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
