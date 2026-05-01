"""音频加载模块 — 使用 PyAV (libav) 统一解码。

支持格式:
  FLAC, OPUS, WAV, MP3, M4A/ALAC, AAC, WMA,
  APE, OGG VORBIS, TTA, DSF/DSD

解码策略:
  所有格式 -> PyAV 解码到 float32 numpy 数组
  DSD (DSF/DFF) -> ffmpeg 子进程回退
  PyAV 失败 -> ffmpeg 子进程回退
"""

from __future__ import annotations

import logging
import subprocess
import tempfile
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = frozenset({
    ".flac", ".opus", ".wav", ".mp3", ".m4a", ".mp4",
    ".aac", ".wma", ".ape", ".ogg", ".tta", ".dsf", ".dff",
})


# ---------------------------------------------------------------------------
# 公共接口
# ---------------------------------------------------------------------------

def load_audio(filepath: str | Path) -> tuple[np.ndarray, int]:
    """加载音频文件。返回 (data: np.ndarray shape=(channels, samples), sample_rate)。"""
    filepath = Path(filepath)

    if not filepath.exists():
        raise FileNotFoundError(f"文件不存在: {filepath}")

    # DSD 格式直接走 ffmpeg 回退
    if filepath.suffix.lower() in (".dsf", ".dff"):
        result = _try_load_ffmpeg(filepath)
        if result is not None:
            return result
        raise ValueError(f"无法加载 DSD 文件: {filepath}")

    # PyAV 主解码
    result = _decode_with_av(filepath)
    if result is not None:
        return result

    # PyAV 失败 -> ffmpeg 回退
    result = _try_load_ffmpeg(filepath)
    if result is not None:
        return result

    raise ValueError(f"无法加载文件: {filepath}")


def _decode_with_av(filepath: Path) -> tuple[np.ndarray, int] | None:
    """使用 PyAV 解码音频到 float32 (channels, samples) 数组。"""
    try:
        import av
    except ImportError:
        return None

    try:
        container = av.open(str(filepath))
        try:
            audio_streams = [s for s in container.streams if s.type == 'audio']
            if not audio_streams:
                return None

            stream = audio_streams[0]
            sr = stream.sample_rate

            chunks = []
            for frame in container.decode(stream):
                # to_ndarray() 返回 (channels, samples) 对于 planar 格式
                # 或 (1, samples*channels) 对于 packed 格式
                arr = frame.to_ndarray()
                # 统一转为 float32
                if arr.dtype != np.float32:
                    arr = arr.astype(np.float32)
                    # 整数格式需要归一化到 [-1, 1]
                    if frame.format.name in ('s16', 's16p'):
                        arr /= 32768.0
                    elif frame.format.name in ('s32', 's32p'):
                        arr /= 2147483648.0
                    elif frame.format.name in ('s64', 's64p'):
                        arr /= 9223372036854775808.0
                chunks.append(arr)

            if not chunks:
                return None

            data = np.concatenate(chunks, axis=1).astype(np.float32)
            # 确保 shape 是 (channels, samples)
            if data.ndim == 1:
                data = data[np.newaxis, :]
            return data, sr

        finally:
            container.close()

    except Exception as e:
        logger.debug("PyAV decode failed for %s: %s", filepath, e)
        return None


def _try_load_ffmpeg(filepath: Path) -> tuple[np.ndarray, int] | None:
    """通过 ffmpeg 子进程解码到临时 wav，再用 soundfile 读取。"""
    try:
        import soundfile as sf
    except ImportError:
        return None

    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = tmp.name

        cmd = [
            "ffmpeg", "-y",
            "-i", str(filepath),
            "-ac", "2",
            "-ar", "48000",
            "-f", "wav",
            tmp_path,
        ]
        result = subprocess.run(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=120,
        )
        if result.returncode != 0:
            return None

        data, sr = sf.read(tmp_path, dtype="float32", always_2d=True)
        # soundfile返回(samples, channels)，转置为(channels, samples)
        return data.T, sr

    except Exception as e:
        logger.debug("ffmpeg backend failed for %s: %s", filepath, e)
        return None
    finally:
        if tmp_path:
            try:
                Path(tmp_path).unlink(missing_ok=True)
            except Exception:
                pass


def is_audio_file(filepath: str | Path) -> bool:
    return Path(filepath).suffix.lower() in SUPPORTED_EXTENSIONS


def is_dsd_file(filepath: str | Path) -> bool:
    return Path(filepath).suffix.lower() in (".dsf", ".dff")
