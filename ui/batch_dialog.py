"""Batch analysis progress dialog."""

from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QLabel, QProgressBar,
    QPushButton, QHBoxLayout, QTextEdit,
)
from lang import t
from ui.styles import (
    BG_SURFACE, BG_RAISED, BORDER_SUB, BORDER_MID,
    TEXT_PRI, TEXT_SEC, TEXT_DIM, ACCENT,
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
                border-radius: 12px;
            }}
            QLabel {{
                color: {TEXT_PRI};
                background: transparent;
                border: none;
            }}
            QProgressBar {{
                background-color: #1a1d1f;
                border: 1px solid {BORDER_MID};
                border-radius: 6px;
                height: 24px;
                text-align: center;
                color: {TEXT_PRI};
                font-size: 11px;
            }}
            QProgressBar::chunk {{
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #E0B55A, stop:1 #C89A3A);
                border-radius: 5px;
            }}
            QPushButton {{
                background-color: {BG_RAISED};
                border: 1px solid {BORDER_MID};
                border-radius: 6px;
                color: {TEXT_SEC};
                padding: 6px 16px;
                font-size: 11px;
            }}
            QPushButton:hover {{
                border-color: {ACCENT};
                color: {TEXT_PRI};
            }}
            QTextEdit {{
                background-color: #1a1d1f;
                border: 1px solid {BORDER_MID};
                border-radius: 6px;
                color: {TEXT_SEC};
                font-family: 'Consolas', monospace;
                font-size: 10px;
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
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(12)

        self._status_label = QLabel(t("正在分析...", "Analyzing..."))
        self._status_label.setStyleSheet("font-size: 13px; font-weight: 600;")
        layout.addWidget(self._status_label)

        self._file_label = QLabel(f"0 / {self._total}")
        self._file_label.setStyleSheet(f"color: {TEXT_SEC}; font-size: 11px;")
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
                    stop:0 #E0B55A, stop:1 #C89A3A);
                border: none;
                color: white;
                font-weight: 600;
                border-radius: 6px;
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
