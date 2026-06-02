"""Spectrum analysis mixin — STFT, multi-resolution, reassigned, mel, MFCC."""

from __future__ import annotations

import warnings

import numpy as np

from ._state import (
    _stft_cache,
    _stft_lock,
    _MAX_STFT_CACHE,
    _max_reduce_with_carry,
    _ensure_wisdom,
)


class _SpectrumMixin:
    """Mixed into AudioAnalyzer.  Expects self.data, self.sample_rate, etc."""

    TARGET_FRAMES = 16384

    # ------------------------------------------------------------------
    # STFT
    # ------------------------------------------------------------------
    def stft(
        self,
        n_fft: int = 2048,
        hop_length: int | None = None,
        win_length: int | None = None,
        window: str = "hann",
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        import librosa
        if self.data is None:
            raise RuntimeError("未加载音频")
        hop = hop_length or (n_fft // 4)
        if win_length is None:
            win_length = n_fft
        audio = self._mono
        S = librosa.stft(audio, n_fft=n_fft, hop_length=hop,
                         win_length=win_length, window=window)
        freqs = librosa.fft_frequencies(sr=self.sample_rate, n_fft=n_fft)
        times = librosa.frames_to_time(
            np.arange(S.shape[1]), sr=self.sample_rate, hop_length=hop)
        return freqs, times, S

    # ------------------------------------------------------------------
    # Spectrogram (dB)
    # ------------------------------------------------------------------
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
        import librosa
        _ensure_wisdom()
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
                while len(_stft_cache) > _MAX_STFT_CACHE:
                    _stft_cache.popitem(last=False)

        from . import _state
        _state._wisdom_dirty = True

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
        on_init,
        on_block,
        cancel_check=lambda: False,
    ):
        """Compute dB spectrogram in a streaming fashion.

        Returns ``(freqs, times, full_db)`` on success, ``None`` if cancelled
        or if the file is too short for meaningful streaming.

        Uses absolute dBFS reference (ref = 1.0) rather than per-file peak
        normalisation, because the global maximum is not known ahead of time.
        The shader's vmin/vmax range (-120..0 dB) absorbs the offset.
        """
        import librosa
        _ensure_wisdom()
        from scipy.signal import get_window

        import pyfftw

        audio = self._mono
        sr = self.sample_rate
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

            # Vectorized frame extraction via stride tricks (replaces per-column loop)
            # shape=(n_fft, cnt): rows are samples within a frame, columns are frames
            # stride[0]=1 element (adjacent samples), stride[1]=hop elements (next frame start)
            strides = (audio_padded.strides[0], audio_padded.strides[0] * hop)
            frames = np.lib.stride_tricks.as_strided(
                audio_padded[offset:], shape=(n_fft, cnt), strides=strides)
            buf_time[:, :cnt] = frames * win[:, None]
            if cnt < block_cols:
                buf_time[:, cnt:] = 0.0

            fft()

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
                while len(_stft_cache) > _MAX_STFT_CACHE:
                    _stft_cache.popitem(last=False)

        from . import _state
        _state._wisdom_dirty = True

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
        import librosa
        audio = self._mono
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
            mask = np.concatenate([[True], np.diff(freqs_combined) > 0])
            freqs_combined = freqs_combined[mask]
            mag_combined = mag_combined[mask]

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
        import librosa
        audio = self._mono
        sr = self.sample_rate
        hop = hop_length or (n_fft // 4)
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
        # Compute squared magnitude once in float64 — avoids a second np.abs call
        S_real = S.real.astype(np.float64)
        S_imag = S.imag.astype(np.float64)
        S_sq_f64 = S_real ** 2 + S_imag ** 2
        S_sq = S_sq_f64 + eps

        omega_corr = np.imag(S_t * np.conj(S) / S_sq) / (2.0 * np.pi)

        tau_corr = np.real(S_f * np.conj(S) / S_sq)

        freqs_base = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
        n_freqs = len(freqs_base)
        n_frames = S.shape[1]

        mag = np.sqrt(S_sq_f64)

        f_new_all = freqs_base[:, None] + omega_corr * sr
        f_new_all = np.clip(f_new_all, freqs_base[0], freqs_base[-1])
        f_idx_all = np.searchsorted(freqs_base, f_new_all.ravel())
        f_idx_all = f_idx_all.clip(0, n_freqs - 1).reshape(n_freqs, n_frames)

        t_new_all = np.arange(n_frames)[None, :] + tau_corr * (hop / sr)
        t_new_all = np.clip(t_new_all, 0, n_frames - 1)
        t_idx_all = np.round(t_new_all).astype(int).clip(0, n_frames - 1)

        # np.bincount is ~5-10x faster than np.add.at for large arrays
        flat_idx = (f_idx_all.ravel() * n_frames + t_idx_all.ravel()).astype(np.intp)
        n_total = n_freqs * n_frames
        mag_acc = np.bincount(flat_idx, weights=mag.ravel(), minlength=n_total).reshape(n_freqs, n_frames)
        weight = np.bincount(flat_idx, minlength=n_total).reshape(n_freqs, n_frames).astype(np.float64)

        mag_acc /= np.maximum(weight, 1.0)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            db = librosa.amplitude_to_db(mag_acc, ref=np.max(mag_acc), top_db=None)

        times = librosa.frames_to_time(
            np.arange(n_frames), sr=sr, hop_length=hop)

        return freqs_base.astype(np.float64), times, db.astype(np.float32)

    # ------------------------------------------------------------------
    # Mel Spectrogram
    # ------------------------------------------------------------------
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
        import librosa
        if self.data is None:
            raise RuntimeError("未加载音频")
        hop = hop_length or (n_fft // 8)
        audio = self._mono
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
        import librosa
        if self.data is None:
            raise RuntimeError("未加载音频")
        audio = self._mono
        return librosa.feature.mfcc(y=audio, sr=self.sample_rate, n_mfcc=n_mfcc)

    # ------------------------------------------------------------------
    # RMS
    # ------------------------------------------------------------------
    def rms(self, frame_length: int = 2048, hop_length: int | None = None) -> np.ndarray:
        import librosa
        hop = hop_length or (frame_length // 4)
        audio = self._mono
        return librosa.feature.rms(y=audio, frame_length=frame_length, hop_length=hop)[0]

    # ------------------------------------------------------------------
    # Spectral Centroid
    # ------------------------------------------------------------------
    def spectral_centroid(self) -> tuple[np.ndarray, np.ndarray]:
        import librosa
        if self.data is None:
            raise RuntimeError("未加载音频")
        audio = self._mono
        centroid = librosa.feature.spectral_centroid(y=audio, sr=self.sample_rate)[0]
        t = np.linspace(0, self.duration, len(centroid))
        return t, centroid

    # ------------------------------------------------------------------
    # Zero Crossing Rate
    # ------------------------------------------------------------------
    def zcr(self) -> tuple[np.ndarray, np.ndarray]:
        import librosa
        zcr = librosa.feature.zero_crossing_rate(self.waveform)[0]
        t = np.linspace(0, self.duration, len(zcr))
        return t, zcr
