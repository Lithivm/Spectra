"""轻量浮点动画助手（Phase 4 动效统一入口）。

FloatAnim 封装 QVariantAnimation：
- ``start_from(v0, v1)`` — 从 v0 缓动到 v1（OutCubic），可重复调用实现"中途改目标"
- 每帧回调 ``on_tick(value)``（value 为缓动后的浮点值）
- 结束时保证 on_tick(end_value) 恰好落地，再调 ``on_finish()``

注意：动画对象必须被持有（挂在 widget 属性上），否则 GC 后动画停止。
"""
from PyQt6.QtCore import QEasingCurve, QVariantAnimation


class FloatAnim(QVariantAnimation):
    """0→1 风格浮点动画（默认 OutCubic 缓动）。"""

    def __init__(self, duration_ms: int, on_tick=None, on_finish=None):
        super().__init__()
        self._on_tick = on_tick
        self._on_finish = on_finish
        self.setDuration(duration_ms)
        self.setEasingCurve(QEasingCurve.Type.OutCubic)
        self.finished.connect(self._finished)

    def start_from(self, v0: float, v1: float = 1.0) -> None:
        """从当前值 v0 缓动到 v1（重启动画，不跳变：起点即当前显示值）。"""
        self.setStartValue(float(v0))
        self.setEndValue(float(v1))
        self.start()

    def rebind(self, on_tick=None, on_finish=None) -> None:
        """更新回调（复用同一动画对象时，闭包可能捕获了新的局部变量）。"""
        if on_tick is not None:
            self._on_tick = on_tick
        if on_finish is not None:
            self._on_finish = on_finish

    def updateCurrentValue(self, value):  # noqa: N802 (Qt 虚函数签名)
        if self._on_tick is not None:
            try:
                self._on_tick(value)
            except RuntimeError:
                # owner C++ object deleted (window closed mid-animation)
                self.stop()

    def _finished(self) -> None:
        # 保证末帧精确落在 end_value（避免浮点漂移）
        if self._on_tick is not None:
            try:
                self._on_tick(float(self.endValue()))
            except RuntimeError:
                pass
        if self._on_finish is not None:
            try:
                self._on_finish()
            except RuntimeError:
                pass  # owner widget already deleted
