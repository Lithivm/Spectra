"""EBU R128 loudness and true peak — computed at the native sample rate.

为什么不用 pyloudnorm 的 Meter：它把分析降采到 12 kHz。K 权重高架的中心在 16.8 kHz，
12 kHz 的 Nyquist 只有 6 kHz —— 高频内容整段被削掉。实测 48 kHz 文件里 8 kHz 单音读低
2.2 LU，16 kHz 读低 14 LU，高频占重的文件整段读低 40 LU。BS.1770 要求滤波器在原始
采样率上设计（De Man 双二阶），这里就是这么做。

系数用 pyloudnorm 的 DeMan 变体生成（与 ITU 表值对齐；pyloudnorm 默认的 K-weighting
变体差约 0.045 LU），计算路径是本地实现：分块流式 + 累积能量，内存 O(chunk)。

约定：音频一律是 (channels, samples) —— 与 `AudioAnalyzer.data` 一致；1-D 视为单声道。
"""
from __future__ import annotations

import math

import numpy as np
from scipy.signal import firwin, lfilter

_K_SHELF = (3.99984385397, 0.7071752369554193, 1681.9744509555319)   # (G, Q, fc)
_K_HP = (0.0, 0.5003270373253953, 38.13547087613982)
K_CONSTANT = -0.691          # BS.1770 标定常数
ABS_GATE_LU = -70.0
REL_GATE_LU = -10.0
CHANNEL_GAINS = (1.0, 1.0, 1.0, 1.41, 1.41)

_cache: dict[int, list[tuple[np.ndarray, np.ndarray]]] = {}


def k_weight_coeffs(sr: int) -> list[tuple[np.ndarray, np.ndarray]]:
    """K 权重双二阶级联（按原生采样率设计），缓存。"""
    sr = int(sr)
    hit = _cache.get(sr)
    if hit is not None:
        return hit
    from pyloudnorm.iirfilter import IIRfilter

    shelf = IIRfilter(_K_SHELF[0], _K_SHELF[1], _K_SHELF[2], sr, "high_shelf_DeMan")
    hp = IIRfilter(_K_HP[0], _K_HP[1], _K_HP[2], sr, "high_pass_DeMan")
    hit = [shelf.generate_coefficients(), hp.generate_coefficients()]
    _cache[sr] = hit
    return hit


def channel_gains(n_channels: int) -> np.ndarray:
    if n_channels <= len(CHANNEL_GAINS):
        return np.asarray(CHANNEL_GAINS[:n_channels], dtype=np.float64)
    return np.ones(n_channels, dtype=np.float64)


def block_loudnesses(audio: np.ndarray, sr: int, block_s: float, hop_s: float,
                     cancel_check=None, chunk_s: float = 5.0,
                     gains: np.ndarray | None = None) -> np.ndarray:
    """K 加权块响度（LU），分块流式处理，内存 O(chunk)。

    块起点在 hop 网格上（3 s / 1 s = 75% 重叠）。不做门控 —— 门控属于统计层
    （`gate` / `gate_mean`）。滤波器状态 zi 跨块连续，跨块窗口用上一块尾部的平方
    能量补齐，所以分块结果与整段一次性计算逐位相同。
    """
    audio = np.asarray(audio, dtype=np.float64)
    if audio.ndim == 1:
        audio = audio.reshape(1, -1)
    n_ch, n = audio.shape
    L = int(round(block_s * sr))
    H = int(round(hop_s * sr))
    if n < L or L < 1 or H < 1:
        return np.array([], dtype=np.float64)

    g = channel_gains(n_ch) if gains is None else np.asarray(gains, dtype=np.float64)
    coeffs = k_weight_coeffs(sr)
    # 从零状态起步（= 从文件头开始的真实滤波），跨块保持状态连续 → 分块结果与
    # 整段一次性滤波逐位相同。
    zis = [np.tile(np.zeros(max(len(a), len(b)) - 1), (n_ch, 1)) for b, a in coeffs]
    chunk = max(L, H, int(chunk_s * sr))
    out: list[np.ndarray] = []
    hist = None                       # 上一块尾部最多 L-1 行的平方能量
    processed = 0

    while processed < n:
        if cancel_check is not None and cancel_check():
            break
        end = min(processed + chunk, n)
        seg = audio[:, processed:end]
        y = seg
        for i, (b, a) in enumerate(coeffs):
            y, zis[i] = lfilter(b, a, y, zi=zis[i], axis=1)
        e = y * y
        m = e.shape[1]
        C = np.concatenate([np.zeros((n_ch, 1)), np.cumsum(e, axis=1)], axis=1)

        # 1) 跨块窗口：起点在上一块、结尾落在本块
        if hist is not None and hist.shape[1] > 0:
            h = hist.shape[1]
            g0 = max(0, processed - L + 1)
            s = -(-g0 // H) * H                       # ceil(g0 / H) * H
            while s < processed and s + L <= n:
                idx = s - (processed - h)                # 窗口起点在 hist 中的位置
                if idx >= 0:
                    acc = float(np.sum(g[:, None] * hist[:, idx:]) + np.sum(g[:, None] * e[:, :s + L - processed])) / L
                    out.append(np.array([K_CONSTANT + 10.0 * math.log10(max(acc, 1e-300))]))
                s += H

        # 2) 完整落在本块内的窗口（起点在 hop 网格上）
        if m >= L:
            s = np.arange((-processed) % H, m - L + 1, H)
            if s.size:
                energy = np.sum(g[:, None] * (C[:, s + L] - C[:, s]), axis=0) / L
                out.append(K_CONSTANT + 10.0 * np.log10(np.maximum(energy, 1e-300)))

        hist = e[:, -(min(L - 1, m)):] if L > 1 else None
        processed = end

    if not out:
        return np.array([], dtype=np.float64)
    vals = np.concatenate([np.asarray(v, dtype=np.float64) for v in out])
    return vals[np.isfinite(vals)]


def gate(vals: np.ndarray) -> np.ndarray:
    """R128 双重门控：绝对 -70 LU，再相对 -10 LU（相对第一级门控后的均值）。"""
    v = np.asarray(vals, dtype=np.float64)
    v = v[np.isfinite(v)]
    v = v[v > ABS_GATE_LU]
    if v.size == 0:
        return v
    return v[v > float(np.mean(v)) + REL_GATE_LU]


def gate_mean(vals: np.ndarray) -> float:
    g = gate(vals)
    return float(np.mean(g)) if g.size else -math.inf


def gated_percentiles(vals: np.ndarray, p_low: float, p_high: float) -> tuple[float, float, int]:
    g = gate(vals)
    if g.size == 0:
        return -math.inf, -math.inf, 0
    return float(np.percentile(g, p_low)), float(np.percentile(g, p_high)), int(g.size)


def true_peak_dbtp(audio: np.ndarray, sr: int, cancel_check=None,
                   chunk: int = 1 << 15, up: int = 4, taps: int = 72) -> float | None:
    """BS.1770 true peak：4× 过采样多相 FIR，分块（内存 O(chunk)），覆盖到 Nyquist。

    每个相位一组 taps（窗内自卷积），按相位组计算 —— 与整体 resample_poly 等价。
    sr < 16 kHz 返回 None（无法保证 10 kHz 分析带）。
    """
    audio = np.asarray(audio, dtype=np.float64)
    if audio.ndim == 1:
        audio = audio.reshape(1, -1)
    sr = int(sr)
    if sr < 16000:
        return None
    h = up * firwin(taps, 1.0 / (2 * up), fs=1.0)
    per = taps // up
    n_ch, n = audio.shape
    groups = [h[i::up] for i in range(up)]

    peak = 0.0
    hist = np.zeros((n_ch, per - 1))
    processed = 0
    while processed < n:
        if cancel_check is not None and cancel_check():
            break
        end = min(processed + chunk, n)
        seg = audio[:, processed:end]
        buf = np.concatenate([hist, seg], axis=1)
        for gp in groups:
            for c in range(n_ch):
                y = np.convolve(buf[c], gp, mode="valid")
                if y.size:
                    v = float(np.max(np.abs(y)))
                    if v > peak:
                        peak = v
        hist = seg[:, -(per - 1):]
        processed = end

    if peak <= 0.0:
        return None
    return float(20.0 * math.log10(peak))
