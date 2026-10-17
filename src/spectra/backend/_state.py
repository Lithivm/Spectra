"""Shared module-level state for the Spectra analyzer.

Contains FFTW wisdom management, STFT result cache, and utility functions
used across spectrum.py and core.py.  Imported by both modules — no circular
dependency because this module imports nothing from the rest of the package.
"""

from __future__ import annotations

import atexit
import os
import sys
import threading
from collections import OrderedDict

import numpy as np

# ---------------------------------------------------------------------------
# FFTW wisdom persistence (lazy — loaded on first STFT call)
# ---------------------------------------------------------------------------
_wisdom_path = os.path.join(os.path.expanduser('~'), '.spectra', 'fftw_wisdom.pkl')
_wisdom_dirty = False
_wisdom_loaded = False
_wisdom_lock = threading.Lock()


def _ensure_wisdom() -> None:
    """One-time setup: enable pyfftw cache, set thread count, load wisdom."""
    global _wisdom_loaded, _wisdom_dirty
    if _wisdom_loaded:
        return
    with _wisdom_lock:
        if _wisdom_loaded:
            return

        import pyfftw
        pyfftw.interfaces.cache.enable()
        pyfftw.config.NUM_THREADS = os.cpu_count() or 4

        os.makedirs(os.path.dirname(_wisdom_path), exist_ok=True)

        try:
            import pickle
            if os.path.exists(_wisdom_path):
                with open(_wisdom_path, "rb") as f:
                    pyfftw.import_wisdom(pickle.load(f))
                _wisdom_dirty = True   # mark so flush knows wisdom is populated
        except Exception:
            pass

        if not _wisdom_loaded:
            try:
                _bundled = os.path.join(sys._MEIPASS, 'analyzer', 'fftw_wisdom.pkl')
                if os.path.exists(_bundled):
                    with open(_bundled, "rb") as f:
                        pyfftw.import_wisdom(pickle.load(f))
                    import shutil
                    shutil.copy2(_bundled, _wisdom_path)
            except Exception:
                pass

        _wisdom_loaded = True


def _flush_wisdom() -> None:
    global _wisdom_dirty
    if not _wisdom_dirty:
        return
    try:
        import pickle
        import pyfftw
        with open(_wisdom_path, "wb") as f:
            pickle.dump(pyfftw.export_wisdom(), f)
    except Exception:
        pass
    _wisdom_dirty = False


atexit.register(_flush_wisdom)

# ---------------------------------------------------------------------------
# STFT result cache
# ---------------------------------------------------------------------------
_stft_cache: OrderedDict = OrderedDict()
_stft_lock = threading.Lock()
_MAX_STFT_CACHE = 8


# ---------------------------------------------------------------------------
# Streaming helper
# ---------------------------------------------------------------------------
def _max_reduce_with_carry(block, factor, carry):
    """Collapse *factor* adjacent columns via element-wise max.

    Args:
        block: ``(n_freqs, cnt)`` float32 ndarray — newly arrived columns.
        factor: how many raw columns to collapse into one output column.
        carry: ``(n_freqs, leftover)`` or ``None`` — columns left over
               from the previous block that haven't yet filled a full group.

    Returns:
        ``(reduced, new_carry)`` where *reduced* is ``(n_freqs, out_cnt)``
        and *new_carry* is ``None`` or the residual columns.
    """
    n_freqs, cnt = block.shape

    if carry is not None:
        block = np.column_stack([carry, block])
        cnt += carry.shape[1]

    out_cnt = cnt // factor
    leftover = cnt % factor

    if out_cnt > 0:
        usable = cnt - leftover
        reshaped = block[:, :usable].reshape(n_freqs, out_cnt, factor)
        reduced = reshaped.max(axis=2).astype(np.float32)
        new_carry = block[:, usable:].copy() if leftover > 0 else None
    else:
        reduced = np.empty((n_freqs, 0), dtype=np.float32)
        new_carry = block.copy()

    return reduced, new_carry
