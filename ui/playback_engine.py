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

from PyQt6.QtCore import QObject, pyqtSignal

logger = logging.getLogger(__name__)


def _resample(audio: np.ndarray, src_sr: int, dst_sr: int) -> np.ndarray:
    """Resample (channels, samples) float32 array from src_sr to dst_sr."""
    if src_sr == dst_sr:
        return audio
    try:
        import soxr
        return soxr.resample(audio.T, src_sr, dst_sr).T.astype(np.float32)
    except ImportError:
        pass
    try:
        from scipy.signal import resample_poly
        from math import gcd
        g = gcd(src_sr, dst_sr)
        up, down = dst_sr // g, src_sr // g
        return resample_poly(audio, up, down, axis=1).astype(np.float32)
    except ImportError:
        pass
    logger.warning("No resampler available (soxr/scipy), playback may fail")
    return audio


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
        self._stream: sd.OutputStream | None = None
        self._stream_start_dac: float = 0.0
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

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def load(self, audio: np.ndarray, sample_rate: int) -> None:
        """Feed a new audio buffer.  Stops any current playback."""
        self.stop()
        if audio.ndim == 1:
            audio = audio[np.newaxis, :]
        audio = audio.astype(np.float32)
        # Resample to WASAPI device rate if needed
        if self._device_sr > 0 and sample_rate != self._device_sr:
            audio = _resample(audio, sample_rate, self._device_sr)
            sample_rate = self._device_sr
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
        self._close_stream()  # may trigger _on_stream_finished → resets counters
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
        """Current playback position in seconds (from callback frame counter)."""
        if self._state != "playing":
            return self._start_frame / self._sample_rate if self._sample_rate else 0.0
        with self._cb_lock:
            frame = self._cb_frame
        return frame / self._sample_rate if self._sample_rate else 0.0

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
            self._stream = sd.OutputStream(
                samplerate=self._sample_rate,
                channels=self._audio.shape[0],
                dtype='float32',
                blocksize=0,
                callback=self._callback,
                finished_callback=self._on_stream_finished,
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

    def _on_stream_finished(self) -> None:
        with self._cb_lock:
            self._start_frame = 0
            self._cb_frame = 0
        self._set_state("stopped")

    def _set_state(self, state: str) -> None:
        if state != self._state:
            self._state = state
            self.state_changed.emit(state)
