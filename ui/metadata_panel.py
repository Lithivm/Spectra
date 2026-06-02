"""MetadataPanel — metadata display panel."""

from __future__ import annotations

import logging
import time

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QFrame, QScrollArea,
)
from PyQt6.QtCore import Qt

from analyzer.core import AudioAnalyzer, _TAG_TR
from lang import t, on_lang_change

logger = logging.getLogger(__name__)
from ui.styles import (
    BG_SURFACE, BG_RAISED,
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

        self._key_label = QLabel(key)
        self._key_label.setStyleSheet(f"""
            color: {TEXT_SEC};
            font-size: 11px;
            min-width: 75px;
            max-width: 75px;
            background: transparent;
            border: none;
        """)
        self._key_label.setWordWrap(False)

        self._value_label = QLabel(value)
        self._value_label.setStyleSheet(f"""
            color: {value_color};
            font-size: 11px;
            font-family: 'Consolas', monospace;
            background: transparent;
            border: none;
        """)
        self._value_label.setWordWrap(True)
        self._value_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)

        layout.addWidget(self._key_label)
        layout.addWidget(self._value_label, stretch=1)
        # 占位 — 与 _AnalysisRow 的指示点对齐
        layout.addSpacing(20)

    def set_texts(self, key: str, value: str) -> None:
        """Update labels in-place (for language switch)."""
        self._key_label.setText(key)
        self._value_label.setText(value)


class _AnalysisRow(QWidget):
    """质量分析行，带颜色指示点。"""
    def __init__(self, ok: bool, key: str, value: str, warn: bool = False) -> None:
        super().__init__()
        self._ok = ok
        self._warn = warn
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

        self._key_label = QLabel(key)
        self._key_label.setStyleSheet(f"""
            color: {TEXT_SEC};
            font-size: 11px;
            min-width: 75px;
            max-width: 75px;
            background: transparent;
            border: none;
        """)
        layout.addWidget(self._key_label)

        self._value_label = QLabel(value)
        self._value_label.setStyleSheet(f"""
            color: {TEXT_PRI if ok else (ACCENT_AMB if warn else ACCENT_RED)};
            font-size: 11px;
            font-family: 'Consolas', monospace;
            background: transparent;
            border: none;
        """)
        self._value_label.setWordWrap(True)
        layout.addWidget(self._value_label, stretch=1)

        # 状态点 — 靠右垂直居中对齐
        dot = QLabel("●")
        if ok:
            color = ACCENT_GRN
        elif warn:
            color = ACCENT_AMB
        else:
            color = ACCENT_RED
        dot.setStyleSheet(f"color: {color}; font-size: 12px; background: transparent; border: none;")
        dot.setFixedWidth(20)
        dot.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        layout.addWidget(dot)

    def set_texts(self, key: str, value: str) -> None:
        """Update labels in-place (for language switch)."""
        self._key_label.setText(key)
        self._value_label.setText(value)


class MetadataPanel(QWidget):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._stored_analyzer: AudioAnalyzer | None = None
        self._stored_qa: dict | None = None
        # Widget reference caches for fast _retranslate_with_data
        self._section_labels: list[QLabel] = []
        self._info_rows: list[_Row] = []
        self._tag_rows: list[_Row] = []
        self._analysis_rows: list[_AnalysisRow] = []
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
        self._section_labels.clear()
        self._info_rows.clear()
        self._tag_rows.clear()
        self._analysis_rows.clear()
        self._indicator.setText("●")
        self._indicator.setStyleSheet(f"color: {TEXT_DIM}; font-size: 20px; background: transparent; border: none;")

    def load_metadata(self, analyzer: AudioAnalyzer) -> None:
        self.clear()
        self._stored_analyzer = analyzer
        self._stored_qa = None
        self._qa_analyzer = analyzer  # keep for later background analysis

        # TECHNICAL
        sec = _section_label(t("技术信息", "TECHNICAL"))
        self._section_labels.append(sec)
        self._content_layout.addWidget(sec)
        self._content_layout.addWidget(_divider())

        t_info = time.perf_counter()
        info = analyzer.info()
        logger.debug("info(): %.3fs", time.perf_counter() - t_info)

        hi_keys = {t("采样率", "Sample Rate"), t("声道", "Channels"), t("时长", "Duration")}
        for k, v in info.items():
            color = ACCENT if k in hi_keys else TEXT_PRI
            row = _Row(k, str(v), value_color=color)
            self._info_rows.append(row)
            self._content_layout.addWidget(row)

        # TAGS
        if analyzer.metadata:
            sec = _section_label(t("标签", "TAGS"))
            self._section_labels.append(sec)
            self._content_layout.addWidget(sec)
            self._content_layout.addWidget(_divider())
            skip = {"filename", "filepath", "sample_rate", "bitrate", "channels", "mime_type", "duration"}
            _GENERIC = {"format": ("格式", "Format")}
            for k, v in analyzer.metadata.items():
                if k in skip:
                    continue
                zh, en = _GENERIC.get(k) or _TAG_TR.get(k, (k, k))
                row = _Row(t(zh, en), str(v))
                self._tag_rows.append(row)
                self._content_layout.addWidget(row)

        # ANALYSIS — placeholder, filled by load_analysis()
        sec = _section_label(t("分析", "ANALYSIS"))
        self._section_labels.append(sec)
        self._content_layout.addWidget(sec)
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
                row = _AnalysisRow(True, t("削波", "Clipping"), t("未检测到削波", "No clipping detected"))
            else:
                n = clip.get("count", 0)
                ms = clip.get("longest_ms", 0)
                hard = clip.get("hard_clips", 0)
                soft = clip.get("soft_clips", 0)
                detail = f"{n} events, max {ms}ms"
                if hard > 0 or soft > 0:
                    detail += f"  ({hard} hard / {soft} soft)"
                row = _AnalysisRow(False, t("削波", "Clipping"), detail)
            self._analysis_rows.append(row)
            self._content_layout.addWidget(row)

            ups = qa.get("upsampling", {})
            chz = ups.get("cutoff_hz", 0)
            if ups.get("ok"):
                row = _AnalysisRow(True, t("高频", "Hi-freq"), t("正常", "Normal"))
            else:
                row = _AnalysisRow(False, t("高频", "Hi-freq"), f"{chz/1000:.1f} kHz")
            self._analysis_rows.append(row)
            self._content_layout.addWidget(row)

            dr = qa.get("dynamic_range", {})
            dr_val = dr.get("dr", 0)
            if dr_val > 14:
                label, ok, warn = t("优秀", "Excellent"), True, False
            elif dr_val >= 8:
                label, ok, warn = t("正常", "Normal"), True, False
            else:
                label, ok, warn = t("压缩", "Compressed"), False, True
            row = _AnalysisRow(ok, t("动态范围", "Dynamics"), f"DR {dr_val:.1f} — {label}", warn=warn)
            self._analysis_rows.append(row)
            self._content_layout.addWidget(row)

            # ── LUFS (EBU R128) ──
            loud = qa.get("loudness", {})
            if loud:
                il = loud.get("integrated_lufs", 0)
                row = _Row("LUFS (I)", f"{il:.1f} LUFS")
                self._info_rows.append(row)
                self._content_layout.addWidget(row)

                stl = loud.get("short_term_lufs", 0)
                row = _Row("LUFS (S)", f"{stl:.1f} LUFS")
                self._info_rows.append(row)
                self._content_layout.addWidget(row)

                lra = loud.get("lra_lu", 0)
                row = _Row("LRA", f"{lra:.1f} LU" if lra > 0 else "N/A")
                self._info_rows.append(row)
                self._content_layout.addWidget(row)

            # ── Additional metrics ──
            peak_db = qa.get("peak_db", None)
            tp_db = qa.get("true_peak_db", None)
            if peak_db is not None:
                label = f"{peak_db:.1f} dBFS"
                if tp_db is not None and tp_db > peak_db:
                    label += f"  (TP {tp_db:.1f} dBTP)"
                row = _Row(t("峰值", "Peak"), label)
                self._info_rows.append(row)
                self._content_layout.addWidget(row)
            rms_val = qa.get("rms", None)
            if rms_val is not None:
                import math
                rms_db = 20 * math.log10(max(rms_val, 1e-10))
                row = _Row("RMS", f"{rms_db:.1f} dBFS")
                self._info_rows.append(row)
                self._content_layout.addWidget(row)
        else:
            self._content_layout.addWidget(
                _Row("", t("分析不可用", "Analysis unavailable")))

        self._content_layout.addStretch()

    def _retranslate(self, _lang: str | None = None) -> None:
        self._header_title.setText(t("文件信息", "File Info"))
        if self._stored_analyzer:
            self._retranslate_with_data()
        elif hasattr(self, '_empty_label') and self._empty_label:
            self._empty_label.setText(t("拖入文件\n进行分析", "Drop a file\nto analyze"))

    def _retranslate_with_data(self) -> None:
        """Update all row texts in-place — no widget destruction."""
        analyzer = self._stored_analyzer
        if analyzer is None:
            return
        qa = self._stored_qa

        info = analyzer.info()
        info_items = list(info.items())

        _GENERIC = {"format": ("格式", "Format")}
        skip = {"filename", "filepath", "sample_rate", "bitrate", "channels", "mime_type", "duration"}
        tag_items = []
        if analyzer.metadata:
            for k, v in analyzer.metadata.items():
                if k in skip:
                    continue
                zh, en = _GENERIC.get(k) or _TAG_TR.get(k, (k, k))
                tag_items.append((t(zh, en), str(v)))

        # Update section labels directly from cached references
        for i, lbl in enumerate(self._section_labels):
            if i == 0:
                lbl.setText(t("技术信息", "TECHNICAL"))
            elif i == 1 and len(self._section_labels) >= 3:
                lbl.setText(t("标签", "TAGS"))
            elif i == len(self._section_labels) - 1:
                lbl.setText(t("分析", "ANALYSIS"))

        # Update info rows
        for i, row in enumerate(self._info_rows):
            if i < len(info_items):
                k, v = info_items[i]
                row.set_texts(k, str(v))

        # Update tag rows
        for i, row in enumerate(self._tag_rows):
            if i < len(tag_items):
                row.set_texts(*tag_items[i])

        # Update analysis rows
        if qa:
            for i, row in enumerate(self._analysis_rows):
                self._update_analysis_row(row, i, qa)

    @staticmethod
    def _update_analysis_row(row: _AnalysisRow, idx: int, qa: dict) -> None:
        """Update a single _AnalysisRow with current language texts."""
        clip = qa.get("clipping", {})
        ups = qa.get("upsampling", {})
        dr = qa.get("dynamic_range", {})

        if idx == 0:
            if clip.get("ok"):
                row.set_texts(t("削波", "Clipping"), t("未检测到削波", "No clipping detected"))
            else:
                n = clip.get("count", 0)
                ms = clip.get("longest_ms", 0)
                hard = clip.get("hard_clips", 0)
                soft = clip.get("soft_clips", 0)
                detail = f"{n} events, max {ms}ms"
                if hard > 0 or soft > 0:
                    detail += f"  ({hard} hard / {soft} soft)"
                row.set_texts(t("削波", "Clipping"), detail)
        elif idx == 1:
            chz = ups.get("cutoff_hz", 0)
            if ups.get("ok"):
                row.set_texts(t("高频", "Hi-freq"), t("正常", "Normal"))
            else:
                row.set_texts(t("高频", "Hi-freq"), f"{chz/1000:.1f} kHz")
        elif idx == 2:
            dr_val = dr.get("dr", 0)
            if dr_val > 14:
                label = t("优秀", "Excellent")
            elif dr_val >= 8:
                label = t("正常", "Normal")
            else:
                label = t("压缩", "Compressed")
            row.set_texts(t("动态范围", "Dynamics"), f"DR {dr_val:.1f} — {label}")
