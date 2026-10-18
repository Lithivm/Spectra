"""Audio playback engine using sounddevice OutputStream.

Position tracking: simple frame counter in the audio callback (no DAC
time arithmetic in the hot path — avoids jitter and rounding noise).
UI position queries use stream.time from the main thread.

输出通道（系统 + WASAPI 共享/独占）由 ``backend.output_devices`` 枚举，本文件只负责：
按通道开流、按格式量化、独占失败自动回退共享，以及切换时保住播放位置。
"""

from __future__ import annotations

import logging
import threading
from dataclasses import replace

import numpy as np
import sounddevice as sd

from PyQt6.QtCore import QObject, pyqtSignal

from spectra.backend.convert import prepare_channels, quantize
from spectra.backend.output_devices import (
    ChannelCapability,
    OutputChannel,
    StreamInfo,
    WASAPI_HOST_API,
    enumerate_channels,
)

logger = logging.getLogger(__name__)

# 独占优先 float32（零量化，本机 DAC 可用），失败降级 int32，再降级 int16。
# 共享固定 float32 —— PortAudio 共享路径没有 int32 收益。
DTYPE_PREFERENCE_EXCLUSIVE = ("float32", "int32", "int16")
DTYPE_PREFERENCE_SHARED = ("float32",)


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
    output_ready(StreamInfo)
        实际生效的输出通道（设备名 / 格式 / 实际采样率 / 是否直通 / 是否回退）。
        UI 的状态栏与展开列表据此显示。
    """

    state_changed = pyqtSignal(str)
    output_ready = pyqtSignal(object)

    def __init__(self, parent: QObject | None = None,
                 channel: OutputChannel | None = None):
        super().__init__(parent)
        self._source_audio: np.ndarray | None = None  # float32 (channels, samples)，文件原始率
        self._source_sr: int = 0
        self._audio: np.ndarray | None = None         # 按目标格式/目标率排好
        self._sample_rate: int = 0                    # 流实际跑的率
        self._total_frames: int = 0
        self.channels: int = 1
        self._dtype: str = "float32"

        self._stream: sd.OutputStream | None = None
        self._stream_start_dac: float = 0.0
        # 流代数：每次关闭/新建流自增。sounddevice 的 finished_callback 是异步的，
        # 用代数识别「已废弃的流」的回调，避免它晚于 pause/seek 把计数器清零。
        self._stream_gen: int = 0
        self._start_frame: int = 0
        self._state: str = "stopped"

        # Frame counter — only touched in the audio callback (real-time thread)
        self._cb_frame: int = 0
        self._cb_lock = threading.Lock()

        self._channel: OutputChannel | None = channel
        self._capabilities: dict[str, ChannelCapability] = {}
        self._info: StreamInfo | None = None
        # 构造时不探测：独占能力只能真开一次流才知道，构造就开流会白抢独占，
        # 测试也没法脱机跑。UI 在启动时按需探一次（选中通道时），结果缓存。

    # ------------------------------------------------------------------
    # 输出通道
    # ------------------------------------------------------------------
    @property
    def channel(self) -> OutputChannel | None:
        return self._channel if self._channel is not None else self._default_channel()

    @property
    def info(self) -> StreamInfo | None:
        return self._info

    def available_channels(self) -> tuple[OutputChannel, ...]:
        """可选输出通道（UI 的展开列表就读这个）。"""
        try:
            return enumerate_channels(sd.query_devices(), sd.query_hostapis(),
                                      sd.default.device[1])
        except Exception:
            logger.exception("枚举输出通道失败")
            return ()

    def cached_capability(self, channel: OutputChannel) -> ChannelCapability | None:
        """已探测过的通道返回结果，没探测过的返回 None（不触发探测）。"""
        return self._capabilities.get(channel.key)

    def _default_channel(self) -> OutputChannel | None:
        """没选通道时的兜底：sounddevice 默认输出设备（系统那条）。"""
        entries = self.available_channels()
        return entries[0] if entries else None

    def dtype_preference(self, channel: OutputChannel) -> tuple[str, ...]:
        return DTYPE_PREFERENCE_EXCLUSIVE if channel.mode == "exclusive" else DTYPE_PREFERENCE_SHARED

    def requested_sr(self, channel: OutputChannel, source_sr: int | None = None) -> int:
        """向设备请求的采样率。

        独占按文件采样率直接请求（成功 = 直通，设备被切到文件原生率）；
        共享必须用 mix format 率（本机 88200）→ 文件率不同就必然 SRC。
        """
        if channel.mode == "exclusive" and source_sr:
            return int(source_sr)
        return int(channel.mix_sr)

    def capability(self, channel: OutputChannel) -> ChannelCapability:
        """这台设备在这个模式到底能不能开（缓存，每个通道只探一次）。"""
        key = channel.key
        if key in self._capabilities:
            return self._capabilities[key]
        cap = self._try_channel(channel)
        self._capabilities[key] = cap
        self._publish_info(self._info_from(cap))
        return cap

    def use_channel(self, channel: OutputChannel) -> None:
        """挂上通道但**不探测**（启动时先按持久化值挂上，探测由 UI 异步触发一次）。"""
        self._channel = channel
        self._publish_info(self._info_from(
            ChannelCapability(channel=channel, ok=True, dtype="float32", probed=False)))

    def select_output(self, channel: OutputChannel) -> ChannelCapability:
        """切输出通道：先关流再开流，播放位置按秒保住（换率后重算帧数）。"""
        cap = self.capability(channel)
        was_playing = self._state == "playing"
        position = self.get_position()
        self._channel = channel
        if self._source_audio is not None:
            self._apply(cap)
            self._set_frames_from_seconds(position)
        if was_playing:
            self._start_stream()
        else:
            self._publish_info(self._info_from(cap))
        return cap

    def _try_channel(self, channel: OutputChannel) -> ChannelCapability:
        """按格式偏好试开一次。独占失败 → 自动回退同设备的共享，并如实记录。

        回退不静默：``fallback_from`` 记着原请求，UI 的状态栏/列表据此显示「已回退」。
        """
        target = channel
        last_error: str | None = None
        for attempt in range(2):
            for dtype in self.dtype_preference(target):
                try:
                    stream = self._open_once(target, dtype, for_stream=False)
                except Exception as exc:
                    last_error = str(exc)
                    logger.warning("打不开 %s（%s/%s）: %s",
                                   target.device_name, target.mode, dtype, exc)
                    continue
                try:
                    stream.close()
                except Exception:
                    pass
                return ChannelCapability(
                    channel=target, ok=True, dtype=dtype,
                    fallback_from=None if attempt == 0 else channel,
                )
            if target.mode == "exclusive":
                target = replace(target, mode="shared")
                logger.warning("独占失败，自动回退共享：%s（非直通）", target.device_name)
                continue
            break
        return ChannelCapability(
            channel=target, ok=False, dtype=self.dtype_preference(target)[0],
            error=last_error or "无法打开输出通道",
            fallback_from=None if target.key == channel.key else channel,
        )

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
        channel = self.channel
        target_channels = int(channel.max_channels) if channel is not None else 2
        self._source_audio = prepare_channels(audio, target_channels)
        self._source_sr = int(sample_rate)
        self.channels = int(self._source_audio.shape[0])

        cap = self._capabilities.get(channel.key) if channel is not None else None
        if cap is None or not cap.ok:
            # load 时不探测（启动时 UI 已探过一次）；没结果就按 float32 + mix 率处理
            cap = ChannelCapability(channel=channel, ok=True, dtype="float32", probed=False)
        self._apply(cap)
        self._publish_info(self._info_from(cap))

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
    def _apply(self, cap: ChannelCapability) -> None:
        """把源数据按当前通道的目标率/目标格式排好：直通就不重采样。"""
        if self._source_audio is None:
            return
        target_sr = self.requested_sr(cap.channel, self._source_sr) if cap.channel else self._source_sr
        if target_sr != self._source_sr:
            try:
                self._audio = quantize(_resample(self._source_audio, self._source_sr, target_sr), cap.dtype)
            except Exception:
                logger.exception("Resample %s -> %s failed, keeping source rate",
                                 self._source_sr, target_sr)
                self._audio = quantize(self._source_audio, cap.dtype)
                target_sr = self._source_sr
        else:
            self._audio = quantize(self._source_audio, cap.dtype)
        self._sample_rate = target_sr
        self._total_frames = int(self._audio.shape[1])
        self.channels = int(self._audio.shape[0])
        self._dtype = cap.dtype

    def _set_frames_from_seconds(self, seconds: float) -> None:
        frame = max(0, min(int(round(seconds * self._sample_rate)), max(self._total_frames - 1, 0)))
        with self._cb_lock:
            self._start_frame = frame
            self._cb_frame = frame

    def _start_stream(self) -> None:
        self._close_stream()
        cap = self.channel and self.capability(self.channel)
        if cap is None or not cap.ok or self._audio is None:
            if cap is not None:
                self._publish_info(self._info_from(cap))
            self._set_state("stopped")
            return
        try:
            stream = self._open_once(cap.channel, cap.dtype, for_stream=True)
        except Exception as exc:
            # 真开流失败（设备被占/格式被拒）→ 重新判定一次并如实报告，不静默换设备
            cap = self._try_channel(cap.channel)
            self._apply(cap)
            if not cap.ok:
                self._publish_info(self._info_from(cap, error=str(exc)))
                self._set_state("stopped")
                return
            try:
                stream = self._open_once(cap.channel, cap.dtype, for_stream=True)
            except Exception as exc2:
                logger.exception("Failed to start audio stream")
                self._publish_info(self._info_from(cap, error=str(exc2)))
                self._set_state("stopped")
                return
        self._stream = stream
        self._stream_start_dac = stream.time
        self._publish_info(self._info_from(cap, actual_sr=int(stream.actual_samplerate)))
        self._set_state("playing")

    def _open_once(self, channel: OutputChannel, dtype: str, *, for_stream: bool):
        samplerate = self._sample_rate if for_stream else self.requested_sr(channel, self._source_sr)
        kwargs: dict = {
            "samplerate": samplerate,
            "channels": self.channels,
            "dtype": dtype,
            "blocksize": 0,
            "latency": "high",      # use larger buffer to avoid underruns
            "device": channel.device_index,
        }
        if channel.host_api == WASAPI_HOST_API:
            kwargs["extra_settings"] = sd.WasapiSettings(exclusive=channel.mode == "exclusive")
        if for_stream:
            gen = self._stream_gen
            kwargs["callback"] = self._callback
            kwargs["finished_callback"] = lambda g=gen: self._on_stream_finished(g)
        return sd.OutputStream(**kwargs)

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
            outdata[n_valid:] = 0
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

    def _info_from(self, cap: ChannelCapability, actual_sr: int | None = None,
                   error: str | None = None) -> StreamInfo:
        requested = self.requested_sr(cap.channel, self._source_sr) if cap.channel else 0
        return StreamInfo(
            channel=cap.channel,
            dtype=cap.dtype,
            source_sr=self._source_sr,
            requested_sr=requested,
            actual_sr=int(actual_sr if actual_sr is not None else requested),
            error=error if error is not None else cap.error,
            fallback_from=cap.fallback_from,
        )

    def _publish_info(self, info: StreamInfo) -> None:
        if info != self._info:
            self._info = info
            self.output_ready.emit(info)

    def _set_state(self, state: str) -> None:
        if state != self._state:
            self._state = state
            self.state_changed.emit(state)
