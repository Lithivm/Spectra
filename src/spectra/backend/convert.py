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
