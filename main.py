"""Spectra 入口薄壳。

开发期直接 `python main.py`；pip install 后走 `spectra` 命令（spectra.__main__:main）。
"""

import sys
from pathlib import Path

try:
    from spectra.__main__ import main
except ImportError:  # 未安装成包：把 src 挂上 sys.path
    sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
    from spectra.__main__ import main

if __name__ == "__main__":
    main()
