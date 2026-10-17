"""Language switcher for the Spectrum UI.

Usage:
    from spectra.lang import t, LANG, toggle_lang, on_lang_change

    label = t("中文", "English")
    unsub = on_lang_change(lambda lang: some_widget.setText(t("中文", "English")))
    # When the widget is destroyed, call unsub() to prevent leaks
"""

import weakref

LANG = "zh"

_listeners: list = []


def t(zh: str, en: str) -> str:
    return zh if LANG == "zh" else en


def toggle_lang() -> str:
    global LANG
    LANG = "en" if LANG == "zh" else "zh"
    # 遍历副本：回调里可能调用 unsubscribe() 修改 _listeners（边迭代边改会崩）
    dead: list = []
    for entry in _listeners[:]:
        cb = entry() if isinstance(entry, weakref.ref) else entry
        if cb is None:
            dead.append(entry)  # 弱引用已失效
            continue
        try:
            cb(LANG)
        except Exception:
            # 控件已半销毁（RuntimeError）等 —— 丢掉这条，不影响其它监听者
            dead.append(entry)
    if dead:
        _listeners[:] = [e for e in _listeners if not any(e is d for d in dead)]
    return LANG


def on_lang_change(cb):
    """Register a callback for language changes.

    For bound methods, use weak references to avoid preventing GC:
        unsub = on_lang_change(widget.some_method)
    For lambdas and plain functions, a strong reference is kept.
    """
    # Use weak reference for bound methods to prevent preventing widget GC
    if hasattr(cb, '__self__'):
        ref = weakref.WeakMethod(cb, lambda r: _listeners.remove(r) if r in _listeners else None)
        _listeners.append(ref)
    else:
        _listeners.append(cb)

    def unsubscribe():
        # Try removing as weak ref first, then as strong ref
        for entry in _listeners[:]:
            if entry is cb:
                target = True
            elif isinstance(entry, weakref.ref):
                # 绑定方法每次取出的对象都不是同一个，只能比 ==（不能用 is）
                try:
                    alive_cb = entry()
                    target = alive_cb is not None and alive_cb == cb
                except Exception:
                    target = False
            else:
                target = False
            if target:
                try:
                    _listeners.remove(entry)
                except ValueError:
                    pass
                break

    return unsubscribe
