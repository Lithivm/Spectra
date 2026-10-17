"""Spectra entry point."""

import logging
import logging.handlers
import sys
import threading
import traceback
from pathlib import Path

from PyQt6.QtGui import QIcon
from PyQt6.QtWidgets import QApplication

from spectra.frontend.main_window import MainWindow


def _setup_logging() -> None:
    log_dir = Path.home() / ".spectra"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "crash.log"

    handler = logging.handlers.RotatingFileHandler(
        str(log_path), maxBytes=5 * 1024 * 1024, backupCount=3, encoding="utf-8"
    )
    handler.setFormatter(logging.Formatter(
        "%(asctime)s | %(threadName)s | %(name)s | %(levelname)s | %(message)s"
    ))

    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    root.addHandler(handler)

    sys.excepthook = lambda et, ev, tb: (
        logging.getLogger("excepthook").critical("Uncaught exception", exc_info=(et, ev, tb)),
        handler.flush(),
    )

    def _thread_excepthook(args: threading.ExceptHookArgs) -> None:
        logging.getLogger("thread_hook").critical(
            f"Uncaught in thread {args.thread.name if args.thread else '?'}",
            exc_info=(args.exc_type, args.exc_value, args.exc_traceback),
        )
        handler.flush()
    threading.excepthook = _thread_excepthook


def main() -> None:
    _setup_logging()
    logger = logging.getLogger("main")
    logger.info("Starting Spectra — log session begin")
    print("[spectra] Creating QApplication...")
    try:
        app = QApplication.instance() or QApplication([])
        print("[spectra] QApplication created OK")
        base = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent))
        icon_path = str(base / "assets" / "logo.png")
        app.setWindowIcon(QIcon(icon_path))
        app.setStyle("Fusion")
        print("[spectra] Creating MainWindow()...")
        window = MainWindow()
        print("[spectra] MainWindow created OK")
        print("[spectra] Showing window...")
        window.show()
        print("[spectra] Entering event loop...")
        sys.exit(app.exec())
    except Exception:
        print("[spectra] EXCEPTION:")
        traceback.print_exc()
        logger.exception("Fatal startup error")
        raise


if __name__ == "__main__":
    main()
