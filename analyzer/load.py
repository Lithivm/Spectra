"""音频加载模块 — 使用 PyAV (libav) 统一解码。

支持格式:
  FLAC, OPUS, WAV, MP3, M4A/ALAC, AAC, WMA,
  APE, OGG VORBIS, TTA, AIFF

解码策略:
  所有格式 -> PyAV 解码到 float32 numpy 数组
  PyAV 失败 -> ffmpeg 子进程回退（容错兜底）
"""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = frozenset({
    ".flac", ".opus", ".wav", ".mp3", ".m4a", ".mp4",
    ".aac", ".wma", ".ape", ".ogg", ".tta", ".aiff",
})


# ---------------------------------------------------------------------------
# 公共接口
# ---------------------------------------------------------------------------

def load_audio(filepath: str | Path) -> tuple[np.ndarray, int]:
    """加载音频文件。返回 (data: np.ndarray shape=(channels, samples), sample_rate)。"""
    filepath = Path(filepath)

    if not filepath.exists():
        raise FileNotFoundError(f"文件不存在: {filepath}")

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

            # 尝试从流元数据预估总采样数，减少内存重分配
            est_total = 0
            if stream.duration and stream.time_base:
                est_total = int(stream.duration * stream.time_base * sr)
            elif stream.frames > 0:
                est_total = stream.frames

            chunks: list[np.ndarray] = []
            total_samples = 0
            n_channels = 0

            for frame in container.decode(stream):
                arr = frame.to_ndarray()
                if arr.dtype != np.float32:
                    arr = arr.astype(np.float32)
                    fmt_name = frame.format.name
                    if not fmt_name.startswith(('flt', 'dbl')):
                        arr /= float(1 << (frame.format.bits - 1))

                # 从首帧确定通道数
                if n_channels == 0:
                    n_channels = arr.shape[0] if arr.ndim > 1 else 1

                chunks.append(arr)
                total_samples += arr.shape[1] if arr.ndim > 1 else arr.shape[0]

            if not chunks:
                return None

            # 预分配目标数组，逐块拷贝，避免 concatenate 的临时拷贝开销
            if n_channels > 1:
                data = np.empty((n_channels, total_samples), dtype=np.float32)
                offset = 0
                for blk in chunks:
                    n = blk.shape[1]
                    data[:, offset:offset + n] = blk
                    offset += n
            else:
                data = np.empty((1, total_samples), dtype=np.float32)
                offset = 0
                for blk in chunks:
                    flat = blk.ravel() if blk.ndim > 1 else blk
                    n = len(flat)
                    data[0, offset:offset + n] = flat
                    offset += n

            return data, sr

        finally:
            container.close()

    except Exception as e:
        logger.debug("PyAV decode failed for %s: %s", filepath, e)
        return None


def _try_load_ffmpeg(filepath: Path) -> tuple[np.ndarray, int] | None:
    """通过 ffmpeg 子进程解码 — 使用 stdout pipe 输出原始 f32le 数据，免临时文件。"""
    try:
        import json
    except ImportError:
        return None

    # 1. 用 ffprobe 探测原始参数
    try:
        probe_cmd = [
            "ffprobe", "-v", "quiet",
            "-select_streams", "a:0",
            "-show_entries", "stream=sample_rate,channels",
            "-of", "json",
            str(filepath),
        ]
        probe_result = subprocess.run(
            probe_cmd, capture_output=True, text=True, timeout=30,
        )
        if probe_result.returncode == 0:
            info = json.loads(probe_result.stdout)
            stream = info.get("streams", [{}])[0]
            sr = int(stream.get("sample_rate", 48000))
            ch = int(stream.get("channels", 2))
        else:
            sr, ch = 48000, 2
    except Exception:
        sr, ch = 48000, 2

    # 2. 用 ffmpeg 输出 raw f32le 到 stdout pipe
    try:
        cmd = [
            "ffmpeg", "-y",
            "-i", str(filepath),
            "-f", "f32le",
            "-ac", str(ch),
            "-ar", str(sr),
            "pipe:1",
        ]
        result = subprocess.run(
            cmd,
            capture_output=True,
            timeout=120,
        )
        if result.returncode != 0 or len(result.stdout) == 0:
            return None
        if len(result.stdout) % (4 * ch) != 0:
            logger.debug("ffmpeg pipe output size mismatch for %s", filepath)
            return None

        samples = len(result.stdout) // (4 * ch)
        data = np.frombuffer(result.stdout, dtype=np.float32).reshape((samples, ch))
        # 转为 (channels, samples)
        return data.T.copy(), sr

    except Exception as e:
        logger.debug("ffmpeg backend failed for %s: %s", filepath, e)
        return None


def is_audio_file(filepath: str | Path) -> bool:
    return Path(filepath).suffix.lower() in SUPPORTED_EXTENSIONS
