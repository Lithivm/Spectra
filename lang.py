"""Language switcher for the Spectrum UI.

Usage:
    from lang import t, LANG, toggle_lang, on_lang_change

    label = t("中文", "English")
    on_lang_change(lambda lang: some_widget.setText(t("中文", "English")))
"""

LANG = "zh"

_listeners: list = []


def t(zh: str, en: str) -> str:
    return zh if LANG == "zh" else en


def toggle_lang() -> str:
    global LANG
    LANG = "en" if LANG == "zh" else "zh"
    for cb in _listeners:
        cb(LANG)
    return LANG


def on_lang_change(cb) -> None:
    _listeners.append(cb)
