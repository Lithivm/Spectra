"""Batch analysis progress dialog."""

from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QLabel, QProgressBar,
    QPushButton, QHBoxLayout, QTextEdit,
)
from lang import t
from ui.styles import (
    BG_SURFACE, BG_RAISED, BG_WELL, BORDER_SUB, BORDER_MID,
    TEXT_PRI, TEXT_SEC, ACCENT,
    PRIMARY_HOVER_A, PRIMARY_HOVER_B,
    CORNER_SM, CORNER_LG, FS_SM, FS_BODY, FS_LG,
)


class BatchProgressDialog(QDialog):
    cancelled = pyqtSignal()

    def __init__(self, total: int, parent=None):
        super().__init__(parent)
        self._total = total
        self._cancelled = False
        self.setWindowTitle(t("批量分析", "Batch Analysis"))
        self.setMinimumSize(480, 340)
        self.setModal(True)
        self.setStyleSheet(f"""
            QDialog {{
                background-color: {BG_SURFACE};
                border: 1px solid {BORDER_SUB};
                border-radius: {CORNER_LG}px;
            }}
            QLabel {{
                color: {TEXT_PRI};
                background: transparent;
                border: none;
            }}
            QProgressBar {{
                background-color: {BG_WELL};
                border: 1px solid {BORDER_MID};
                border-radius: {CORNER_SM}px;
                height: 24px;
                text-align: center;
                color: {TEXT_PRI};
                font-size: {FS_BODY}px;
            }}
            QProgressBar::chunk {{
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 {PRIMARY_HOVER_A}, stop:1 {PRIMARY_HOVER_B});
                border-radius: {CORNER_SM - 1}px;
            }}
            QPushButton {{
                background-color: {BG_RAISED};
                border: 1px solid {BORDER_MID};
                border-radius: {CORNER_SM}px;
                color: {TEXT_SEC};
                padding: 6px 16px;
                font-size: {FS_BODY}px;
            }}
            QPushButton:hover {{
                border-color: {ACCENT};
                color: {TEXT_PRI};
            }}
            QTextEdit {{
                background-color: {BG_WELL};
                border: 1px solid {BORDER_MID};
                border-radius: {CORNER_SM}px;
                color: {TEXT_SEC};
                font-family: 'Consolas', monospace;
                font-size: {FS_SM}px;
            }}
            QScrollBar:vertical {{
                background: transparent;
                width: 6px;
                border: none;
            }}
            QScrollBar::handle:vertical {{
                background: {BORDER_MID};
                border-radius: 3px;
            }}
        """)
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(12)

        self._status_label = QLabel(t("正在分析...", "Analyzing..."))
        self._status_label.setStyleSheet(f"font-size: {FS_LG}px; font-weight: 600;")
        layout.addWidget(self._status_label)

        self._file_label = QLabel(f"0 / {self._total}")
        self._file_label.setStyleSheet(f"color: {TEXT_SEC}; font-size: {FS_BODY}px;")
        layout.addWidget(self._file_label)

        self._progress = QProgressBar()
        self._progress.setMaximum(self._total)
        self._progress.setValue(0)
        layout.addWidget(self._progress)

        self._log = QTextEdit()
        self._log.setReadOnly(True)
        self._log.setMaximumHeight(150)
        layout.addWidget(self._log)

        btn_layout = QHBoxLayout()
        btn_layout.addStretch()
        self._cancel_btn = QPushButton(t("取消", "Cancel"))
        self._cancel_btn.clicked.connect(self._on_cancel)
        btn_layout.addWidget(self._cancel_btn)
        self._export_btn = QPushButton(t("导出CSV", "Export CSV"))
        self._export_btn.setEnabled(False)
        self._export_btn.setObjectName("primary")
        self._export_btn.setStyleSheet(f"""
            QPushButton#primary {{
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 {PRIMARY_HOVER_A}, stop:1 {PRIMARY_HOVER_B});
                border: none;
                color: white;
                font-weight: 600;
                border-radius: {CORNER_SM}px;
                padding: 6px 16px;
            }}
        """)
        btn_layout.addWidget(self._export_btn)
        layout.addLayout(btn_layout)

    def update_progress(self, n: int, filename: str, ok: bool = True) -> None:
        self._progress.setValue(n)
        self._file_label.setText(f"{n} / {self._total}")
        icon = "+" if ok else "!"
        self._log.append(f"[{icon}] {filename}")

    def finish(self) -> None:
        self._status_label.setText(t("分析完成 — 选择一个目标文件以导出 CSV",
                                     "Analysis complete — select a destination to export CSV"))
        self._export_btn.setEnabled(True)
        self._cancel_btn.setText(t("关闭", "Close"))

    def _on_cancel(self):
        self._cancelled = True
        self.cancelled.emit()

    @property
    def is_cancelled(self) -> bool:
        return self._cancelled
