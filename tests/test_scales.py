"""Mel convention pins — the project uses Slaney everywhere.

Three places must agree: `spectra.backend.scales` (CPU), the mel branch of
`spectra/frontend/shaders/spectrogram.frag` (GPU), and librosa's mel filterbank
default. These tests guard against the convention drifting back to HTK in one
place only, which is what produced the label/pixel mismatch.
"""

import math
from pathlib import Path

import numpy as np

SHADER = (Path(__file__).resolve().parent.parent
          / "src" / "spectra" / "frontend" / "shaders" / "spectrogram.frag")
WIDGET = (Path(__file__).resolve().parent.parent
          / "src" / "spectra" / "frontend" / "spectrogram_widget.py")


def glsl_mel(f):
    """Mirror of the GLSL helpers, written from the literal shader constants."""
    if f < 1000.0:
        return f / 66.6667
    return 15.0 + math.log(f / 1000.0) / 0.0687480


def glsl_hz(m):
    if m < 15.0:
        return m * 66.6667
    return 1000.0 * math.exp((m - 15.0) * 0.0687480)


class TestSlaneyConstants:
    def test_piecewise_constants(self):
        from spectra.backend import scales
        assert abs(scales.F_SP - 200.0 / 3.0) < 1e-12
        assert scales.MIN_LOG_HZ == 1000.0
        assert abs(scales.MIN_LOG_MEL - 15.0) < 1e-12
        assert abs(scales.LOGSTEP - math.log(6.4) / 27.0) < 1e-12

    def test_linear_region_below_1k(self):
        from spectra.backend import scales
        assert abs(scales.hz_to_mel(100.0) - 1.5) < 1e-9    # 1 mel per 66.7 Hz
        assert abs(scales.hz_to_mel(1000.0) - 15.0) < 1e-9

    def test_log_region_above_1k(self):
        from spectra.backend import scales
        assert abs(scales.hz_to_mel(2000.0) - 25.0819) < 0.0005
        assert abs(scales.hz_to_mel(10000.0) - 48.4913) < 0.0005

    def test_inverse_roundtrip(self):
        from spectra.backend import scales
        for f in [20.0, 100.0, 999.0, 1000.0, 1001.0, 3000.0, 20000.0]:
            assert abs(scales.mel_to_hz(scales.hz_to_mel(f)) - f) < 1e-6


class TestConventionAgreement:
    def test_cpu_matches_librosa_default(self):
        import librosa
        from spectra.backend import scales
        for f in [20, 100, 200, 500, 1000, 2000, 5000, 10000, 20000]:
            cpu = scales.hz_to_mel(float(f))
            lib = float(np.asarray(librosa.hz_to_mel(np.asarray([float(f)]))).item())
            assert abs(cpu - lib) < 1e-9, f"librosa default moved off Slaney at {f} Hz"

    def test_array_and_scalar_paths_agree(self):
        from spectra.backend import scales
        arr = np.array([20.0, 100.0, 1000.0, 5000.0])
        vals = scales.hz_to_mel(arr)
        for f, v in zip(arr, vals):
            assert abs(float(v) - scales.hz_to_mel(float(f))) < 1e-12

    def test_shader_uses_slaney_not_htk(self):
        text = SHADER.read_text(encoding="utf-8")
        assert "hz_to_mel" in text and "mel_to_hz" in text
        assert "2595.0" not in text, "shader still carries an HTK constant"
        assert "200.0 / 3.0" in text and "log(6.4) / 27.0" in text

    def test_widget_labels_use_the_same_convention(self):
        # The Y-axis labels/cursor must go through scales, not a second mel formula.
        text = WIDGET.read_text(encoding="utf-8")
        assert "librosa.hz_to_mel" not in text
        assert "scales.mel_frac" in text and "scales.mel_to_hz" in text

    def test_pixel_position_agrees_with_glsl_mirror(self):
        from spectra.backend import scales
        f_min, f_max = 20.0, 22050.0
        m_min, m_max = glsl_mel(f_min), glsl_mel(f_max)
        for f in [50, 100, 200, 500, 1000, 3000, 10000, 20000]:
            pos_cpu = scales.mel_frac(float(f), f_min, f_max)
            pos_gpu = (glsl_mel(float(f)) - m_min) / (m_max - m_min)
            assert abs(pos_cpu - pos_gpu) < 1e-4, f"position drift at {f} Hz"
            # and the GPU round trip lands back on the same frequency
            assert abs(glsl_hz(m_min + pos_gpu * (m_max - m_min)) - float(f)) < 0.1
