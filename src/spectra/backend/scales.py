"""Mel scale — Slaney convention, the single source of truth on the CPU side.

Slaney's fit is piecewise: linear below 1 kHz (one mel per critical band there)
and logarithmic above. The GPU mirror is `u_scale_mode == 2` in
`spectra/frontend/shaders/spectrogram.frag`; keep the constants in sync —
`tests/test_analyzer.py` pins both sides.

The other convention (HTK, `2595·log10(1+f/700)`, O'Shaughnessy 1987) is what
torchaudio defaults to; librosa defaults to Slaney. This project uses Slaney
everywhere so that the mel Y axis, the cursor readout and the mel filterbank
all describe the same mapping.
"""

from __future__ import annotations

import math

import numpy as np

F_SP = 200.0 / 3.0                 # Hz per mel in the linear region (66.67 Hz)
MIN_LOG_HZ = 1000.0                # start of the logarithmic region
MIN_LOG_MEL = MIN_LOG_HZ / F_SP    # mel value at that point (15.0)
LOGSTEP = math.log(6.4) / 27.0     # mel per natural-log ratio


def hz_to_mel(f: float | np.ndarray):
    """Hz → mel (Slaney). Scalar in → scalar out, array in → array out."""
    arr = np.asarray(f, dtype=float)
    if arr.ndim:
        return np.where(
            arr < MIN_LOG_HZ,
            arr / F_SP,
            MIN_LOG_MEL + np.log(arr / MIN_LOG_HZ) / LOGSTEP,
        )
    fv = float(arr)
    if fv < MIN_LOG_HZ:
        return fv / F_SP
    return MIN_LOG_MEL + math.log(fv / MIN_LOG_HZ) / LOGSTEP


def mel_to_hz(m: float) -> float:
    """mel → Hz, the inverse of `hz_to_mel`."""
    mv = float(m)
    if mv < MIN_LOG_MEL:
        return mv * F_SP
    return MIN_LOG_HZ * math.exp((mv - MIN_LOG_MEL) * LOGSTEP)


def mel_frac(f: float, f_min: float, f_max: float) -> float:
    """Normalized position of *f* on a mel axis spanning [f_min, f_max], 0..1."""
    m_min = hz_to_mel(f_min)
    m_max = hz_to_mel(f_max)
    if m_max <= m_min:
        return 0.5
    return (hz_to_mel(f) - m_min) / (m_max - m_min)
