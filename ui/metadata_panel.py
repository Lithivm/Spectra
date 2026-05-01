"""MetadataPanel — metadata display panel."""

from __future__ import annotations

import time

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QFrame, QScrollArea,
)
from PyQt6.QtCore import Qt

from analyzer.core import AudioAnalyzer, _TAG_TR
from lang import t, on_lang_change
from ui.styles import (
    BG_BASE, BG_SURFACE, BG_RAISED,
    BORDER_SUB, BORDER_MID,
    ACCENT, ACCENT_GRN, ACCENT_RED, ACCENT_AMB,
    TEXT_PRI, TEXT_SEC, TEXT_DIM,
)


def _section_label(text: str) -> QLabel:
    lbl = QLabel(text)
    lbl.setStyleSheet(f"""
        color: {TEXT_DIM};
        font-size: 9px;
        letter-spacing: 1.5px;
        font-weight: 600;
        padding: 14px 16px 6px 16px;
        background: transparent;
        border: none;
        font-family: 'Segoe UI', sans-serif;
    """)
    return lbl


def _divider() -> QFrame:
    d = QFrame()
    d.setFrameShape(QFrame.Shape.HLine)
    d.setFixedHeight(1)
    d.setStyleSheet(f"background: {BORDER_SUB}; border: none; margin: 0 16px;")
    return d


class _Row(QWidget):
    """单行 key/value。"""
    def __init__(self, key: str, value: str, value_color: str = TEXT_PRI) -> None:
        super().__init__()
        self.setStyleSheet(f"""
            QWidget {{
                background: transparent;
                border-radius: 6px;
            }}
            QWidget:hover {{
                background: {BG_RAISED};
            }}
        """)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(16, 5, 16, 5)
        layout.setSpacing(8)

        k = QLabel(key)
        k.setStyleSheet(f"""
            color: {TEXT_SEC};
            font-size: 11px;
            min-width: 85px;
            max-width: 85px;
            background: transparent;
            border: none;
        """)
        k.setWordWrap(False)

        v = QLabel(value)
        v.setStyleSheet(f"""
            color: {value_color};
            font-size: 11px;
            font-family: 'Consolas', monospace;
            background: transparent;
            border: none;
        """)
        v.setWordWrap(True)
        v.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)

        layout.addWidget(k)
        layout.addWidget(v, stretch=1)


class _AnalysisRow(QWidget):
    """质量分析行，带颜色指示点。"""
    def __init__(self, ok: bool, key: str, value: str, warn: bool = False) -> None:
        super().__init__()
        self.setStyleSheet(f"""
            QWidget {{
                background: transparent;
                border-radius: 6px;
            }}
            QWidget:hover {{
                background: {BG_RAISED};
            }}
        """)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(16, 6, 16, 6)
        layout.setSpacing(8)

        # 状态点
        dot = QLabel("●")
        if ok:
            color = ACCENT_GRN
        elif warn:
            color = ACCENT_AMB
        else:
            color = ACCENT_RED
        dot.setStyleSheet(f"color: {color}; font-size: 12px; background: transparent; border: none;")
        dot.setFixedWidth(16)
        layout.addWidget(dot)

        k = QLabel(key)
        k.setStyleSheet(f"""
            color: {TEXT_SEC};
            font-size: 11px;
            min-width: 75px;
            max-width: 75px;
            background: transparent;
            border: none;
        """)
        layout.addWidget(k)

        v = QLabel(value)
        v.setStyleSheet(f"""
            color: {TEXT_PRI if ok else (ACCENT_AMB if warn else ACCENT_RED)};
            font-size: 11px;
            font-family: 'Consolas', monospace;
            background: transparent;
            border: none;
        """)
        v.setWordWrap(True)
        layout.addWidget(v, stretch=1)


class MetadataPanel(QWidget):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._stored_analyzer: AudioAnalyzer | None = None
        self._stored_qa: dict | None = None
        self.setStyleSheet(f"""
            QWidget {{
                background-color: {BG_SURFACE};
                border: 1px solid {BORDER_SUB};
                border-radius: 12px;
            }}
        """)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        # 顶部标题
        header = QWidget()
        header.setFixedHeight(48)
        header.setStyleSheet(f"""
            background: transparent;
            border: none;
            border-bottom: 1px solid {BORDER_SUB};
            border-radius: 0;
        """)
        hl = QHBoxLayout(header)
        hl.setContentsMargins(16, 0, 16, 0)

        self._header_title = QLabel(t("文件信息", "File Info"))
        self._header_title.setStyleSheet(f"""
            color: {TEXT_PRI};
            font-size: 13px;
            font-weight: 600;
            background: transparent;
            border: none;
        """)
        hl.addWidget(self._header_title)
        hl.addStretch()

        self._indicator = QLabel("●")
        self._indicator.setStyleSheet(f"color: {TEXT_DIM}; font-size: 20px; background: transparent; border: none;")
        hl.addWidget(self._indicator)
        outer.addWidget(header)

        # 滚动区
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setStyleSheet(f"""
            QScrollArea {{
                border: none;
                background: transparent;
            }}
            QScrollBar:vertical {{
                background: transparent;
                width: 4px;
                border: none;
            }}
            QScrollBar::handle:vertical {{
                background: {BORDER_MID};
                border-radius: 2px;
            }}
        """)

        self._content = QWidget()
        self._content.setStyleSheet("background: transparent; border: none;")
        self._content_layout = QVBoxLayout(self._content)
        self._content_layout.setContentsMargins(0, 4, 0, 16)
        self._content_layout.setSpacing(0)

        self._show_empty()
        scroll.setWidget(self._content)
        outer.addWidget(scroll, stretch=1)

        on_lang_change(self._retranslate)

    def _show_empty(self) -> None:
        self._empty_label = QLabel(t("拖入文件\n进行分析", "Drop a file\nto analyze"))
        self._empty_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._empty_label.setStyleSheet(f"""
            color: {TEXT_DIM};
            font-size: 12px;
            padding: 48px 24px;
            background: transparent;
            border: none;
            line-height: 1.6;
        """)
        self._content_layout.addWidget(self._empty_label)
        self._content_layout.addStretch()

    def clear(self) -> None:
        while self._content_layout.count():
            item = self._content_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        self._empty_label = None
        self._analysis_placeholder = None
        self._indicator.setText("●")
        self._indicator.setStyleSheet(f"color: {TEXT_DIM}; font-size: 20px; background: transparent; border: none;")

    def load_metadata(self, analyzer: AudioAnalyzer) -> None:
        self.clear()
        self._stored_analyzer = analyzer
        self._stored_qa = None
        self._qa_analyzer = analyzer  # keep for later background analysis

        # TECHNICAL
        self._content_layout.addWidget(_section_label(t("技术信息", "TECHNICAL")))
        self._content_layout.addWidget(_divider())

        t_info = time.perf_counter()
        info = analyzer.info()
        print(f"[PROFILE] info(): {time.perf_counter() - t_info:.3f}s")

        hi_keys = {t("采样率", "Sample Rate"), t("声道", "Channels"), t("时长", "Duration")}
        for k, v in info.items():
            color = ACCENT if k in hi_keys else TEXT_PRI
            self._content_layout.addWidget(_Row(k, str(v), value_color=color))

        # TAGS
        if analyzer.metadata:
            self._content_layout.addWidget(_section_label(t("标签", "TAGS")))
            self._content_layout.addWidget(_divider())
            skip = {"filename", "filepath"}
            for k, v in analyzer.metadata.items():
                if k in skip:
                    continue
                zh, en = _TAG_TR.get(k, (k, k))
                self._content_layout.addWidget(_Row(t(zh, en), str(v)))

        # ANALYSIS — placeholder, filled by load_analysis()
        self._content_layout.addWidget(_section_label(t("分析", "ANALYSIS")))
        self._content_layout.addWidget(_divider())
        self._analysis_placeholder = QLabel(t("正在分析…", "Analyzing…"))
        self._analysis_placeholder.setStyleSheet(f"""
            color: {TEXT_DIM}; font-size: 11px; padding: 8px 16px;
            background: transparent; border: none;
        """)
        self._content_layout.addWidget(self._analysis_placeholder)

        self._content_layout.addStretch()

        self._indicator.setText("●")
        self._indicator.setStyleSheet(
            f"color: {ACCENT_GRN}; font-size: 20px; background: transparent; border: none;")

    def load_analysis(self, qa: dict | None) -> None:
        """Fill the ANALYSIS section — called from background thread result."""
        self._stored_qa = qa
        if not hasattr(self, '_analysis_placeholder') or self._analysis_placeholder is None:
            return
        # Remove placeholder
        self._content_layout.removeWidget(self._analysis_placeholder)
        self._analysis_placeholder.deleteLater()
        self._analysis_placeholder = None
        # Remove trailing stretch (it's the last item)
        item = self._content_layout.takeAt(self._content_layout.count() - 1)
        if item.spacerItem():
            del item

        if qa:
            clip = qa.get("clipping", {})
            if clip.get("ok"):
                self._content_layout.addWidget(
                    _AnalysisRow(True, t("削波", "Clipping"), t("未检测到削波", "No clipping detected")))
            else:
                n = clip.get("count", 0)
                ms = clip.get("longest_ms", 0)
                self._content_layout.addWidget(
                    _AnalysisRow(False, t("削波", "Clipping"), f"{n} events, max {ms}ms"))

            ups = qa.get("upsampling", {})
            if ups.get("ok"):
                self._content_layout.addWidget(
                    _AnalysisRow(True, t("高频", "Hi-freq"), t("正常", "Normal")))
            else:
                chz = ups.get("cutoff_hz", 0)
                sr_hz = ups.get("sr_hz", 0)
                self._content_layout.addWidget(
                    _AnalysisRow(False, t("高频", "Hi-freq"),
                                 t(f"截止于 {chz/1000:.1f} kHz（上限 {sr_hz/2000:.0f} kHz）",
                                   f"Cutoff at {chz/1000:.1f} kHz (limit {sr_hz/2000:.0f} kHz)")))

            dr = qa.get("dynamic_range", {})
            dr_val = dr.get("dr", 0)
            if dr_val > 14:
                label, ok, warn = t("优秀", "Excellent"), True, False
            elif dr_val >= 8:
                label, ok, warn = t("正常", "Normal"), True, False
            else:
                label, ok, warn = t("压缩", "Compressed"), False, True
            self._content_layout.addWidget(
                _AnalysisRow(ok, t("动态范围", "Dynamics"), f"DR {dr_val:.1f} — {label}", warn=warn))
        else:
            self._content_layout.addWidget(
                _Row("", t("分析不可用", "Analysis unavailable")))

        self._content_layout.addStretch()

    def _retranslate(self, _lang: str | None = None) -> None:
        self._header_title.setText(t("文件信息", "File Info"))
        if self._stored_analyzer:
            saved_qa = self._stored_qa
            self.load_metadata(self._stored_analyzer)
            self._stored_qa = saved_qa
            if saved_qa is not None:
                self.load_analysis(saved_qa)
        elif hasattr(self, '_empty_label') and self._empty_label:
            self._empty_label.setText(t("拖入文件\n进行分析", "Drop a file\nto analyze"))
