"""Worker 生命周期回归测试。

CI 上的崩溃（Windows fatal exception: access violation）来自 QThread wrapper
被 GC 时线程还在跑 —— sip 析构一个正在运行的 QThread 是 UB，崩溃点在
`app.processEvents()`（GC 由事件循环里的分配触发）。这里把时序钉住：
窗口对象散掉时 wrapper 必须仍在注册表里，且线程被收尾。
"""

import gc
import sys
import time

import pytest
from PyQt6.QtWidgets import QApplication

from spectra.frontend import main_window as mwmod
from spectra.frontend.main_window import MainWindow, _RUNNING_WORKERS


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication(sys.argv)


class FakeSettings:
    def __init__(self, initial=None):
        self.data = dict(initial or {})

    def value(self, key, default=None):
        return self.data.get(key, default)

    def setValue(self, key, value):
        self.data[key] = value


class _NullNode:
    def __call__(self, *args, **kwargs):
        return None

    def connect(self, *args, **kwargs):
        return None

    def __getattr__(self, name):
        return _NullNode()


class FakeEngine:
    """没有真设备：entries 空 → 不建菜单，不探测。"""

    def __init__(self):
        self.entries = []
        self.info = None
        self.selected_key = None
        self.state = "stopped"
        self.is_playing = False
        self.duration = 1.0
        self.state_changed = _NullNode()
        self.output_ready = _NullNode()

    def available_channels(self):
        return ()

    def capability(self, channel):
        return None

    def cached_capability(self, channel):
        return None

    def get_position(self):
        return 0.0


def _slow_workers(monkeypatch, seconds=3.0):
    """模拟 CI 的冷 import 时序，并且让 run 响应 requestInterruption。"""
    def run(self):
        deadline = time.time() + seconds
        while time.time() < deadline and not self.isInterruptionRequested():
            time.sleep(0.05)
        self.ready.emit()
    monkeypatch.setattr(mwmod._PreloadWorker, "run", run)
    monkeypatch.setattr(mwmod._ProbeWorker, "run", run)


def _thread_done(worker) -> bool:
    """线程结束 = 没在跑，或者 wrapper 已被 sip 析构（finished 后释放注册表引用，
    GC 析构 wrapper 是安全的 —— 崩溃只发生在析构一个还在跑的 QThread）。"""
    try:
        return not worker.isRunning()
    except RuntimeError:
        return True


def test_window_gc_does_not_kill_running_thread(app, monkeypatch):
    """窗口对象散掉 + 强制 GC：wrapper 必须被钉住到线程结束，进程不能崩。"""
    _slow_workers(monkeypatch)
    workers = []
    for _ in range(4):
        mw = MainWindow(settings=FakeSettings(), playback=FakeEngine())
        mw.show()
        app.processEvents()
        workers.extend(w for w in (mw._preload_worker, mw._probe_worker) if w is not None)
        mw = None
        gc.collect()
        app.processEvents()
    for w in workers:
        assert _thread_done(w), "窗口析构时 destroyed 钩子应收尾线程，而不是让 GC 析构一个还在跑的 QThread"
    gc.collect()
    app.processEvents()
    assert not [w for w in _RUNNING_WORKERS if not _thread_done(w)], "注册表里不该留还在跑的线程"


def test_close_event_stops_workers(app, monkeypatch):
    _slow_workers(monkeypatch)
    mw = MainWindow(settings=FakeSettings(), playback=FakeEngine())
    mw.show()
    app.processEvents()
    preload = mw._preload_worker
    assert not _thread_done(preload), "未 close 时慢 preload 应还在跑"
    mw.close()
    app.processEvents()
    assert _thread_done(preload), "closeEvent 之后 preload 线程应结束"
