"""Language switcher for the Spectrum UI.

Usage:
    from lang import t, LANG, toggle_lang, on_lang_change

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
    # Prune dead weak references before dispatching
    alive: list = []
    for entry in _listeners:
        if isinstance(entry, weakref.ref):
            cb = entry()
            if cb is not None:
                alive.append(entry)
                cb(LANG)
            # else: dead ref, skip and don't re-add
        else:
            alive.append(entry)
            entry(LANG)
    if len(alive) < len(_listeners):
        _listeners[:] = alive
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
            if entry is cb or (isinstance(entry, weakref.ref) and entry() is cb):
                try:
                    _listeners.remove(entry)
                except ValueError:
                    pass
                break

    return unsubscribe
