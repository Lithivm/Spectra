"""Audio playback engine using sounddevice OutputStream.

Position tracking: simple frame counter in the audio callback (no DAC
time arithmetic in the hot path — avoids jitter and rounding noise).
UI position queries use stream.time from the main thread.
"""

from __future__ import annotations

import logging
import threading
import numpy as np
import sounddevice as sd

from spectra.backend.convert import prepare_channels
from PyQt6.QtCore import QObject, pyqtSignal

logger = logging.getLogger(__name__)


def _resample(audio: np.ndarray, src_sr: int, dst_sr: int) -> np.ndarray:
    """Resample (channels, samples) float32 array from src_sr to dst_sr.

    失败时抛 RuntimeError（只对 ImportError 做回退）——不能像以前那样静默返回
    原数组，否则调用方会把未重采样的数据标成设备采样率 → 变速变调。
    """
    if src_sr == dst_sr:
        return audio
    out: np.ndarray | None = None
    try:
        import soxr
        out = soxr.resample(audio.T, src_sr, dst_sr).T.astype(np.float32)
    except ImportError:
        out = None  # 无 soxr，试 scipy
    if out is None:
        try:
            from scipy.signal import resample_poly
            from math import gcd
            g = gcd(src_sr, dst_sr)
            up, down = dst_sr // g, src_sr // g
            out = resample_poly(audio, up, down, axis=1).astype(np.float32)
        except ImportError:
            out = None
    if out is None:
        raise RuntimeError("no resampler available (soxr/scipy)")
    # 长度校验：拦住任何「没真正重采样」的返回值
    expect = audio.shape[1] * dst_sr / src_sr
    if abs(out.shape[1] - expect) > max(2.0, expect * 1e-3):
        raise RuntimeError(
            f"resample length mismatch: got {out.shape[1]}, expected {expect:.0f}")
    return out


class PlaybackEngine(QObject):
    """Minimal audio player wrapping a sounddevice OutputStream.

    Signals
    -------
    state_changed(str)
        One of ``'playing'``, ``'paused'``, ``'stopped'``.
    """

    state_changed = pyqtSignal(str)

    def __init__(self, parent: QObject | None = None):
        super().__init__(parent)
        self._audio: np.ndarray | None = None       # (channels, samples) float32
        self._sample_rate: int = 0
        self._total_frames: int = 0
        self.channels: int = 1
        self._stream: sd.OutputStream | None = None
        self._stream_start_dac: float = 0.0
        # 流代数：每次关闭/新建流自增。sounddevice 的 finished_callback 是异步的，
        # 用代数识别「已废弃的流」的回调，避免它晚于 pause/seek 把计数器清零。
        self._stream_gen: int = 0
        self._start_frame: int = 0
        self._state: str = "stopped"
        self._device_sr: int = 0  # WASAPI device sample rate

        # Frame counter — only touched in the audio callback (real-time thread)
        self._cb_frame: int = 0
        self._cb_lock = threading.Lock()

        # Probe WASAPI device sample rate once
        self._probe_wasapi()

    def _probe_wasapi(self) -> None:
        """Find WASAPI default output device and its sample rate."""
        try:
            for api in sd.query_hostapis():
                if api["name"] == "Windows WASAPI":
                    dev_idx = api["default_output_device"]
                    if dev_idx >= 0:
                        self._wasapi_device = dev_idx
                        self._device_sr = int(sd.query_devices(dev_idx)["default_samplerate"])
                        return
        except Exception:
            pass
        self._wasapi_device = None
        self._device_sr = 0

    def _device_channels(self) -> int:
        """当前输出设备的输出声道数（拿不到就当 2）。"""
        try:
            idx = self._wasapi_device
            if idx is None:
                idx = sd.default.device[1]
            return max(1, int(sd.query_devices(idx)["max_output_channels"]))
        except Exception:
            logger.exception("读取输出设备声道数失败，按 2 声道处理")
            return 2

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def load(self, audio: np.ndarray, sample_rate: int) -> None:
        """Feed a new audio buffer.  Stops any current playback.

        audio 是 (channels, samples)：真立体声。传 mixdown 会丢声像和左右差异
        （不相关信号求和 ≈ −3 dB，反相内容相消），所以这里收多声道数据。
        mono 复制成设备声道数（通常 2），不依赖 Windows 引擎的 upmix。
        """
        self.stop()
        audio = prepare_channels(audio, self._device_channels())
        self.channels = int(audio.shape[0])
        audio = audio.astype(np.float32)
        # Resample to WASAPI device rate if needed
        if self._device_sr > 0 and sample_rate != self._device_sr:
            try:
                audio = _resample(audio, sample_rate, self._device_sr)
                sample_rate = self._device_sr
            except Exception:
                # 重采样不可用 —— 保留原采样率，交给 stream 自己去协商（可能失败但不会变速）
                logger.exception("Resample %s -> %s failed, keeping source rate",
                                 sample_rate, self._device_sr)
        self._audio = np.ascontiguousarray(audio)
        self._sample_rate = sample_rate
        self._total_frames = self._audio.shape[1]

    def play(self, start_seconds: float | None = None) -> None:
        """Start or resume playback from *start_seconds*."""
        if self._audio is None:
            return
        if start_seconds is not None:
            self._start_frame = max(0, min(
                int(start_seconds * self._sample_rate), self._total_frames - 1))
        with self._cb_lock:
            self._cb_frame = self._start_frame
        self._start_stream()

    def pause(self) -> None:
        """Pause.  Position is saved so play() will resume where we left off."""
        with self._cb_lock:
            saved = self._cb_frame
        # _close_stream 会推进 _stream_gen，旧流迟到的 _on_stream_finished 被忽略，
        # 所以下面的恢复不会被异步回调清零。
        self._close_stream()
        self._start_frame = saved
        with self._cb_lock:
            self._cb_frame = saved
        self._set_state("paused")

    def stop(self) -> None:
        """Stop and reset to beginning."""
        self._close_stream()  # may trigger _on_stream_finished
        with self._cb_lock:
            self._start_frame = 0
            self._cb_frame = 0
        self._set_state("stopped")

    def toggle(self) -> None:
        if self._state == "playing":
            self.pause()
        else:
            self.play()

    def seek(self, seconds: float) -> None:
        frame = max(0, min(int(seconds * self._sample_rate), self._total_frames - 1))
        was_playing = self._state == "playing"
        if was_playing:
            self._close_stream()
        with self._cb_lock:
            self._start_frame = frame
            self._cb_frame = frame
        if was_playing:
            self._start_stream()

    def track_position(self, seconds: float) -> None:
        """Update frame counters to match visual drag — no stream restart."""
        frame = max(0, min(int(seconds * self._sample_rate), self._total_frames - 1))
        with self._cb_lock:
            self._start_frame = frame
            self._cb_frame = frame

    def get_position(self) -> float:
        """Current playback position in seconds.

        回调游标 _cb_frame 是「已写入」的帧数，超前实际出声一个 block（latency='high'
        还要更多），直接用它进度条会跑在声音前面。这里用 stream.time 与
        _stream_start_dac 推算真正已播帧数，取两者较小值。
        """
        if self._sample_rate == 0:
            return 0.0
        if self._state != "playing":
            return self._start_frame / self._sample_rate
        with self._cb_lock:
            written = self._cb_frame
            start = self._start_frame
        played = written
        stream = self._stream
        if stream is not None:
            try:
                elapsed = stream.time - self._stream_start_dac
            except Exception:
                elapsed = 0.0
            if elapsed > 0.0:
                played = start + int(elapsed * self._sample_rate)
        frame = max(start, min(written, played, self._total_frames))
        return frame / self._sample_rate

    @property
    def is_playing(self) -> bool:
        return self._state == "playing"

    @property
    def state(self) -> str:
        return self._state

    @property
    def duration(self) -> float:
        if self._sample_rate == 0:
            return 0.0
        return self._total_frames / self._sample_rate

    @property
    def sample_rate(self) -> int:
        return self._sample_rate

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _start_stream(self) -> None:
        self._close_stream()
        try:
            kwargs: dict = {}
            if self._wasapi_device is not None:
                kwargs = {
                    "device": self._wasapi_device,
                    "extra_settings": sd.WasapiSettings(exclusive=False),
                }
            self._stream_gen += 1
            gen = self._stream_gen
            self._stream = sd.OutputStream(
                samplerate=self._sample_rate,
                channels=self.channels,
                dtype='float32',
                blocksize=0,
                latency='high',      # use larger buffer to avoid underruns
                callback=self._callback,
                # 带上代数：只有「当前流」的自然结束才改写计数器
                finished_callback=lambda g=gen: self._on_stream_finished(g),
                **kwargs,
            )
            self._stream.start()
            self._stream_start_dac = self._stream.time
            self._set_state("playing")
        except Exception:
            logger.exception("Failed to start audio stream")
            self._close_stream()
            self._set_state("stopped")

    def _close_stream(self) -> None:
        # 先推进代数（旧流的 finished 回调随后一律被忽略），再真正关闭
        self._stream_gen += 1
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:
                pass
            self._stream = None

    def _callback(self, outdata: np.ndarray, frames: int,
                  time_info, status: sd.CallbackFlags) -> None:
        """Real-time audio callback — must be fast and allocation-free."""
        if status:
            # Underflow/overflow — don't try to recover, just keep going
            pass

        with self._cb_lock:
            start = self._cb_frame
            end = min(start + frames, self._total_frames)
            self._cb_frame = end

        n_valid = end - start
        if n_valid > 0:
            # Audio stored as (channels, samples), outdata is (frames, channels)
            outdata[:n_valid] = self._audio[:, start:end].T

        if n_valid < frames:
            outdata[n_valid:] = 0.0
            raise sd.CallbackStop

    def _on_stream_finished(self, gen: int | None = None) -> None:
        """PortAudio 的 finished 回调（在音频线程里，异步到达）。

        只有当前流（gen 相同）的自然放完才清零；pause/seek/stop 主动关闭的旧流
        迟到回调直接忽略，否则暂停位置会被随机重置为 0。
        """
        if gen is not None and gen != self._stream_gen:
            return
        with self._cb_lock:
            self._start_frame = 0
            self._cb_frame = 0
        self._set_state("stopped")

    def _set_state(self, state: str) -> None:
        if state != self._state:
            self._state = state
            self.state_changed.emit(state)
