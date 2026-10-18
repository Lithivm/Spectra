"""播放前的数据整形：声道布局与量化。

纯 numpy，不碰 sounddevice / Qt —— 可以脱机单测（不需要真设备）。
"""

from __future__ import annotations

import numpy as np

INT32_MIN = -(2 ** 31)
INT32_MAX = 2 ** 31 - 1
INT16_MIN = -(2 ** 15)
INT16_MAX = 2 ** 15 - 1


def prepare_channels(audio: np.ndarray, target_channels: int) -> np.ndarray:
    """统一成 (target_channels, samples)。

    - mono（1-D，或单行 2-D）→ 复制成 target_channels。不依赖 Windows 引擎的
      upmix：两路内容完全相同，声像居中。
    - 源声道数 < target_channels → 用最后一路补齐。
    - 源声道数 > target_channels（5.1 接立体声 DAC）→ 平均折叠再铺开。
      DAC 装不下多声道，折叠至少不静默丢声道；这里不能「截前 N 路」，那会丢掉
      后环绕。

    不复制数据的情况：源本来就是 (target_channels, samples) 时原样返回（视图）。
    """
    if target_channels < 1:
        raise ValueError("target_channels 至少为 1")

    if audio.ndim == 1:
        return np.repeat(audio[np.newaxis, :], target_channels, axis=0)

    src = audio.shape[0]
    if src == target_channels:
        return audio
    if src < target_channels:
        pad = np.repeat(audio[-1:], target_channels - src, axis=0)
        return np.vstack([audio, pad])

    fold = np.mean(audio, axis=0)
    return np.repeat(fold[np.newaxis, :], target_channels, axis=0)


def quantize(audio: np.ndarray, dtype) -> np.ndarray:
    """按输出格式量化 float 采样。

    int32 必须**先限幅再乘**：``1.0 × 2³¹`` 在 float→int32 cast 时绕回成 ``INT32_MIN``
    （实测 1.0 和 1.2 都绕回），所以全程在 float64 上算、clip 到 int32 区间再 cast。

    24-bit 源 → int32 是 exact（``v/2²³ × 2³¹ = v·256``，float64 下精确）：目标格式是
    int32 时不需要 dither；只有硬件把独占格式定在 24/16-bit 才有截断问题。
    """
    try:
        dtype = np.dtype(dtype)
    except TypeError as exc:
        raise ValueError(f"不支持的输出格式: {dtype}") from exc

    if dtype == np.dtype(np.float32):
        return audio.astype(np.float32)

    if dtype not in (np.dtype(np.int32), np.dtype(np.int16)):
        raise ValueError(f"不支持的输出格式: {dtype}")

    bits = 31 if dtype == np.dtype(np.int32) else 15
    lo, hi = (INT32_MIN, INT32_MAX) if bits == 31 else (INT16_MIN, INT16_MAX)
    scaled = np.rint(np.clip(audio.astype(np.float64), -1.0, 1.0) * 2.0 ** bits)
    return np.clip(scaled, float(lo), float(hi)).astype(dtype)
