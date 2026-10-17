"""Spectrum analysis mixin — STFT, multi-resolution, reassigned, mel, MFCC."""

from __future__ import annotations

import numpy as np

from ._state import (
    _stft_cache,
    _stft_lock,
    _MAX_STFT_CACHE,
    _max_reduce_with_carry,
    _ensure_wisdom,
)


_TINY = 1e-12


def _amp_scale(win: np.ndarray, n_fft: int, n_freqs: int) -> np.ndarray:
    """每 bin 的单边谱幅度归一化因子（把 STFT 幅度还原成相对满量程的幅度）。

    单边谱 ×2，再除以窗相干增益 mean(w)。DC 与 Nyquist bin 不加倍。
    hann 窗 mean = 0.5 → 因子 = 4/n_fft（比旧的 mag/n_fft 高 6.02 dB）。
    """
    cg = float(np.mean(win)) if len(win) else 1.0
    if cg <= 1e-12:
        cg = 1.0
    base = 2.0 / (n_fft * cg)
    scale = np.full(n_freqs, base, dtype=np.float32)
    scale[0] = base / 2.0
    if n_freqs > 1 and n_fft % 2 == 0:
        scale[-1] = base / 2.0
    return scale


def _to_db(mag: np.ndarray, scale: np.ndarray, ref: str = "dbfs") -> np.ndarray:
    """幅度 → dB。ref='dbfs'（默认，真 dBFS）或 ref='peak'（峰值归一化显示）。"""
    s = np.asarray(scale, dtype=np.float32)
    if mag.ndim > 1:
        s = s.reshape((-1,) + (1,) * (mag.ndim - 1))
    a = mag.astype(np.float32) * s
    if ref == "peak":
        r = float(np.max(a)) if a.size else 0.0
        if r <= 0.0:
            r = 1.0
    else:
        r = 1.0
    return (20.0 * np.log10(np.maximum(a / r, _TINY))).astype(np.float32)


class _SpectrumMixin:
    """Mixed into AudioAnalyzer.  Expects self.data, self.sample_rate, etc."""

    TARGET_FRAMES = 16384

    # ------------------------------------------------------------------
    # Cache key / hop
    # ------------------------------------------------------------------
    def _resolve_hop(self, n_fft: int, hop_length: int | None) -> int:
        if hop_length is not None:
            return int(hop_length)
        if self.duration > 0:
            target_hop = int(self.duration * self.sample_rate / self.TARGET_FRAMES)
            return max(n_fft // 8, min(n_fft, target_hop))
        return n_fft // 4

    def _cache_key(self, mode: str, n_fft: int, hop: int, window: str, ref: str):
        """规范化缓存键 — 两条路径共用同一形状。含文件指纹，防止文件被改后读到旧结果。"""
        if not self.filepath:
            return None
        try:
            st = self.filepath.stat()
            fingerprint = (int(st.st_size), int(st.st_mtime_ns))
        except OSError:
            fingerprint = (0, 0)
        return (str(self.filepath), fingerprint, mode, int(n_fft), int(hop), window, ref)

    def cached_result(self, mode: str, n_fft: int, window: str = "hann",
                      ref: str = "dbfs", hop: int | None = None):
        """UI 缓存快路径：用同一个规范化键查缓存。

        调用方自己拼键永远跟不上（hop 是文件长度决定的），所以由这里统一算。
        """
        key = self._cache_key(mode, n_fft, self._resolve_hop(n_fft, hop), window, ref)
        if key is None:
            return None
        with _stft_lock:
            return _stft_cache.get(key)

    def invalidate_cache(self, mode: str | None = None, n_fft: int | None = None) -> int:
        """Drop cached STFT results matching mode/n_fft. Returns number removed."""
        removed = 0
        with _stft_lock:
            for key in list(_stft_cache.keys()):
                if (mode is None or key[2] == mode) and (n_fft is None or key[3] == int(n_fft)):
                    _stft_cache.pop(key, None)
                    removed += 1
        return removed

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
        audio = self.waveform
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
        ref: str = "dbfs",
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute dB spectrogram.

        Parameters
        ----------
        mode : str
            "standard" — single fixed-size STFT (default).
            "multi"    — multi-resolution: large FFT at lows, small at highs.
            "reassign" — phase-reassigned spectrogram for sharper harmonics.
        ref : str
            "dbfs" (default) — absolute reference, values are true dBFS.
            "peak" — normalise to the file's own maximum (display option).
        """
        import librosa
        _ensure_wisdom()
        hop = self._resolve_hop(n_fft, hop_length)
        cache_key = self._cache_key(mode, n_fft, hop, window, ref)
        if cache_key:
            with _stft_lock:
                if cache_key in _stft_cache:
                    return _stft_cache[cache_key]

        win = np.asarray(librosa.filters.get_window(window, win_length or n_fft), dtype=np.float32)
        if len(win) < n_fft:
            win = np.pad(win, (0, n_fft - len(win)))
        scale = _amp_scale(win, n_fft, n_fft // 2 + 1)

        if mode == "multi":
            result = self._multi_resolution_stft(window, ref)
        elif mode == "reassign":
            result = self._reassigned_spectrogram(n_fft, hop, win_length, window, ref)
        else:
            freqs, times, S = self.stft(n_fft, hop, win_length, window)
            result = (freqs, times, _to_db(np.abs(S), scale, ref))

        # ── Downsample time axis (only if adaptive hop overshot target) ──
        freqs, times, db = result
        if db.shape[1] > self.TARGET_FRAMES:
            chunk = db.shape[1] // self.TARGET_FRAMES
            n_full = self.TARGET_FRAMES
            db_full = db[:, :chunk * n_full].reshape(
                db.shape[0], n_full, chunk
            ).max(axis=2)
            t_full = times[:chunk * n_full].reshape(n_full, chunk).mean(axis=1)
            tail = chunk * n_full
            if tail < db.shape[1]:                       # 尾部不足一整组：不能丢
                db_full = np.concatenate(
                    [db_full, db[:, tail:].max(axis=1, keepdims=True)], axis=1)
                t_full = np.concatenate([t_full, times[tail:].mean(axis=1, keepdims=True)], axis=1)
            result = (freqs, t_full.astype(np.float64), db_full)

        if cache_key:
            with _stft_lock:
                _stft_cache[cache_key] = result
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
        _ensure_wisdom()
        from scipy.signal import get_window

        import pyfftw

        audio = self.waveform
        sr = self.sample_rate
        n_fft_w = win_length or n_fft
        n_freqs = n_fft // 2 + 1

        # ── Hop calculation (same as spectrogram_db) ──
        hop = self._resolve_hop(n_fft, hop_length)

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
        scale = _amp_scale(win, n_fft, n_freqs)

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
            block_db = _to_db(mag, scale)

            if downsample > 1:
                block_db, carry = _max_reduce_with_carry(block_db, downsample, carry)

            bw = block_db.shape[1]
            if bw > 0:
                acc[:, out_col:out_col + bw] = block_db
                on_block(out_col, block_db)
                out_col += bw

        # Flush remaining carry: 部分组要取整组的 max，不能只取第一列
        if carry is not None and out_col < total_cols:
            carry_col = carry.max(axis=1, keepdims=True).astype(np.float32)
            acc[:, out_col] = carry_col[:, 0]
            on_block(out_col, carry_col)
            out_col += 1

        acc = acc[:, :out_col]

        if downsample > 1:
            # 每个输出列对应全局列序列的连续一组；取组中心而不是组起点
            g_full = raw_cols // downsample
            rem = raw_cols - g_full * downsample
            centers = np.arange(out_col, dtype=np.float64)
            full = min(g_full, out_col)
            centers[:full] = np.arange(full) * downsample + (downsample - 1) / 2.0
            if full < out_col:
                centers[full] = full * downsample + (rem - 1) / 2.0
            times = (centers * hop / sr).astype(np.float64)
        else:
            times = (np.arange(out_col) * hop / sr).astype(np.float64)

        # Write back to cache so second load is instant
        if self.filepath:
            with _stft_lock:
                _stft_cache[self._cache_key("standard", n_fft, hop, window, "dbfs")] = (freqs, times, acc)
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
        ref: str = "dbfs",
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Three-band multi-resolution STFT on a uniform row grid.

        Bands: low 0–320 Hz (n_fft 8192), mid 320–2800 Hz (2048), high 2800 Hz–Nyquist (512),
        fixed hop 512. Each band's rows are placed on the finest band's grid (spacing
        sr/8192 ≈ 5.86 Hz at 48 kHz) by nearest-row mapping; rows covered by two bands
        (the crossfade regions) take the max across bands.

        Why a grid instead of concatenated band rows: the concatenated version jumped
        backwards at band boundaries (`freqs` non-monotonic at rows 54 and 178 on a
        48 kHz file), so the Y axis and the cursor reported the wrong frequency for
        those rows; a plain grid keeps freqs strictly monotonic and needs no per-row
        LUT in the shader.
        """
        import librosa
        audio = self.waveform
        sr = self.sample_rate
        hop = 512
        nyq = sr / 2.0
        grid_n = 8192
        bands = ((8192, 0.0, 320.0), (2048, 320.0, 2800.0), (512, 2800.0, nyq))

        grid = librosa.fft_frequencies(sr=sr, n_fft=grid_n)
        acc: np.ndarray | None = None

        for n_fft, f_lo, f_hi in bands:
            f_hi = min(f_hi, nyq)
            S = librosa.stft(audio, n_fft=n_fft, hop_length=hop,
                             win_length=n_fft, window=window, center=True)
            freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
            mask = (freqs >= f_lo) & (freqs <= f_hi)
            if not np.any(mask):
                continue
            bf = freqs[mask]
            win_arr = np.asarray(librosa.filters.get_window(window, n_fft), dtype=np.float32)
            scale = _amp_scale(win_arr, n_fft, n_fft // 2 + 1).reshape(-1, 1)
            amp = np.abs(S) * scale                                # 真满刻度幅度
            band_rows = amp[mask, :]
            # 每个 grid 行取本频段内最近的行（重叠区跨频段取 max）
            idx = np.clip(np.searchsorted(bf, grid), 0, len(bf) - 1)
            lo = np.clip(idx - 1, 0, len(bf) - 1)
            nearest = np.where(np.abs(bf[lo] - grid) < np.abs(bf[idx] - grid), lo, idx)
            inside = (grid >= f_lo) & (grid <= f_hi)
            vals = band_rows[nearest, :]
            acc = vals if acc is None else np.maximum(acc, vals * inside[:, None])

        if acc is None:
            raise RuntimeError("Multi-resolution STFT produced no bands")
        db = _to_db(acc, 1.0, ref)
        times = librosa.frames_to_time(np.arange(db.shape[1]), sr=sr, hop_length=hop)
        return grid.astype(np.float64), times, db.astype(np.float32)

    # ------------------------------------------------------------------
    # Reassigned Spectrogram (iZotope RX style)
    # ------------------------------------------------------------------
    def _reassigned_spectrogram(
        self,
        n_fft: int = 4096,
        hop_length: int | None = None,
        win_length: int | None = None,
        window: str = "hann",
        ref: str = "dbfs",
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Auger–Flandrin phase reassignment (IEEE TASSP 1995).

        Energy in each window slot is moved to (instantaneous frequency, group delay)
        instead of staying at the bin centre:

            f̂(k,j) = f_k − Im(D/X) · sr / 2π       D = STFT with window g'(m)
            t̂(k,j) = j·hop/sr + Re(C/X) / sr       C = STFT with window m·g(m)

        Sign and scaling were checked against ground truth: a linear chirp (instantaneous
        frequency known analytically — matched within 1 Hz) and a single-sample click at a
        known time (reassigned onto that time). Empty cells keep 0 amplitude.
        """
        import librosa
        audio = self.waveform
        sr = self.sample_rate
        hop = self._resolve_hop(n_fft, hop_length)
        wlen = win_length or n_fft

        win = np.asarray(librosa.filters.get_window(window, wlen), dtype=np.float64)
        if len(win) < n_fft:
            win = np.pad(win, (0, n_fft - len(win)))
        gprime = np.gradient(win)                      # 窗导数
        ramp = np.arange(n_fft) - wlen / 2.0           # 窗内局部时间（相对窗中心）

        X = librosa.stft(audio, n_fft=n_fft, hop_length=hop, window=win, center=True)
        D = librosa.stft(audio, n_fft=n_fft, hop_length=hop, window=gprime, center=True)
        C = librosa.stft(audio, n_fft=n_fft, hop_length=hop, window=win * ramp, center=True)

        sq = np.abs(X).astype(np.float64) ** 2
        sq_safe = sq + np.finfo(np.float64).eps
        n_freqs, n_frames = X.shape
        freqs_base = librosa.fft_frequencies(sr=sr, n_fft=n_fft)

        f_hat = freqs_base[:, None] - (np.imag(D * np.conj(X)) / sq_safe) * sr / (2.0 * np.pi)
        t_hat = (np.arange(n_frames)[None, :] * hop + np.real(C * np.conj(X)) / sq_safe) / sr

        f_hat = np.clip(f_hat, freqs_base[0], freqs_base[-1])
        t_hat = np.clip(t_hat, 0.0, (n_frames - 1) * hop / sr)

        f_idx = np.searchsorted(freqs_base, f_hat.ravel()).clip(0, n_freqs - 1).reshape(n_freqs, n_frames)
        t_idx = np.rint(t_hat * sr / hop).astype(np.intp).clip(0, n_frames - 1)

        flat = (f_idx.ravel() * n_frames + t_idx.ravel()).astype(np.intp)
        n_total = n_freqs * n_frames
        mag = np.sqrt(sq)
        acc = np.bincount(flat, weights=mag.ravel(), minlength=n_total).reshape(n_freqs, n_frames)
        cnt = np.bincount(flat, minlength=n_total).reshape(n_freqs, n_frames)
        acc = acc / np.maximum(cnt, 1.0)

        db = _to_db(acc, _amp_scale(win, n_fft, n_freqs), ref)
        times = (np.arange(n_frames) * hop / sr).astype(np.float64)
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
        audio = self.waveform
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
        audio = self.waveform
        return librosa.feature.mfcc(y=audio, sr=self.sample_rate, n_mfcc=n_mfcc)

    # ------------------------------------------------------------------
    # RMS
    # ------------------------------------------------------------------
    def rms(self, frame_length: int = 2048, hop_length: int | None = None) -> np.ndarray:
        import librosa
        hop = hop_length or (frame_length // 4)
        audio = self.waveform
        return librosa.feature.rms(y=audio, frame_length=frame_length, hop_length=hop)[0]

    # ------------------------------------------------------------------
    # Spectral Centroid
    # ------------------------------------------------------------------
    def spectral_centroid(self) -> tuple[np.ndarray, np.ndarray]:
        import librosa
        if self.data is None:
            raise RuntimeError("未加载音频")
        audio = self.waveform
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
