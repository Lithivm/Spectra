"""AudioAnalyzer — audio file loading, analysis, and metadata.

Facade class that composes spectrum analysis (spectrum.py) and quality
analysis (quality.py) mixins.  Module-level FFTW wisdom and STFT cache
live in _state.py.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from lang import t

from .load import load_audio
from .metadata import get_metadata

_librosa_ready = False


def _ensure_librosa() -> None:
    """One-time lazy setup: import librosa + pyfftw and wire them together."""
    global _librosa_ready
    if _librosa_ready:
        return
    import warnings
    import librosa
    import pyfftw.interfaces.scipy_fft as fftw_scipy
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=FutureWarning)
        librosa.set_fftlib(fftw_scipy)
    _librosa_ready = True


# Mixin imports are safe at module level — with their librosa/pyloudnorm
# imports moved to function-level, importing the modules is now lightweight.
from .spectrum import _SpectrumMixin  # noqa: E402
from .quality import _QualityMixin    # noqa: E402


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


class AudioAnalyzer(_SpectrumMixin, _QualityMixin):
    """Load and analyze audio files."""

    def __init__(self, filepath: str | Path | None = None):
        self.filepath: Path | None = None
        self.metadata: dict = {}
        self.data: np.ndarray | None = None
        self._mono: np.ndarray | None = None
        self.sample_rate: int = 0
        self.channels: int = 0
        self.duration: float = 0.0

        if filepath:
            self.load(filepath)

    # ------------------------------------------------------------------
    # Load
    # ------------------------------------------------------------------
    def load(self, filepath: str | Path) -> None:
        _ensure_librosa()
        filepath = Path(filepath)
        self.data, self.sample_rate = load_audio(filepath)
        self.channels = self.data.shape[0] if self.data.ndim > 1 else 1
        self.duration = float(self.data.shape[-1]) / self.sample_rate
        self.filepath = filepath
        self.metadata = get_metadata(filepath)
        self._mono = self.data[0] if self.data.ndim > 1 else self.data

    # ------------------------------------------------------------------
    # Waveform
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
    # Info
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
