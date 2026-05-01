"""Audio Analyzer entry point."""

import sys
import traceback
from pathlib import Path

from PyQt6.QtWidgets import QApplication

from ui.main_window import MainWindow


def main() -> None:
    print("[main.py] Creating QApplication...")
    try:
        app = QApplication.instance() or QApplication([])
        print("[main.py] QApplication created OK")
        app.setStyle("Fusion")
        print("[main.py] Creating MainWindow()...")
        window = MainWindow()
        print("[main.py] MainWindow created OK")
        print("[main.py] Showing window...")
        window.show()
        print("[main.py] Entering event loop...")
        sys.exit(app.exec())
    except Exception:
        print("[main.py] EXCEPTION:")
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
