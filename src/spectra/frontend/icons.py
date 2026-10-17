"""SVG icon set — rendered at runtime via QSvgRenderer (no external files).

Icons are 24x24 viewBox templates with a single {c} color placeholder.
Rendering to QPixmap keeps PyInstaller packaging trivial (no data files).
"""

from __future__ import annotations

from PyQt6.QtCore import QByteArray, Qt
from PyQt6.QtGui import QIcon, QImage, QPainter, QPixmap
from PyQt6.QtSvg import QSvgRenderer

# 24x24 viewBox templates — fill/stroke color injected via .format(c=...)
_ICONS: dict[str, str] = {
    "play": (
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24">'
        '<path d="M8 5.5v13l11-6.5z" fill="{c}"/>'
        '</svg>'
    ),
    "pause": (
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24">'
        '<rect x="7" y="5" width="3.6" height="14" rx="1" fill="{c}"/>'
        '<rect x="13.4" y="5" width="3.6" height="14" rx="1" fill="{c}"/>'
        '</svg>'
    ),
    "open": (
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" '
        'stroke="{c}" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round">'
        '<path d="M3 7a2 2 0 0 1 2-2h4l2 2h8a2 2 0 0 1 2 2v9a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z"/>'
        '</svg>'
    ),
    "save": (
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" '
        'stroke="{c}" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round">'
        '<path d="M12 4v10m0 0l-4-4m4 4l4-4"/>'
        '<path d="M5 17v2a2 2 0 0 0 2 2h10a2 2 0 0 0 2-2v-2"/>'
        '</svg>'
    ),
    "minimize": (
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" '
        'stroke="{c}" stroke-width="1.8" stroke-linecap="round">'
        '<path d="M5 12h14"/>'
        '</svg>'
    ),
    "maximize": (
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" '
        'stroke="{c}" stroke-width="1.8" stroke-linejoin="round">'
        '<rect x="5.5" y="5.5" width="13" height="13" rx="1.5"/>'
        '</svg>'
    ),
    "close": (
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" '
        'stroke="{c}" stroke-width="1.8" stroke-linecap="round">'
        '<path d="M6 6l12 12M18 6L6 18"/>'
        '</svg>'
    ),
}


def render_icon(name: str, size: int = 16, color: str = "#F0EDE8", dpr: float = 1.0) -> QIcon:
    """Render an SVG icon template to a QIcon at the given pixel size.

    Pass dpr (device pixel ratio) so the pixmap is rendered at `size * dpr`
    physical pixels — crisp on HiDPI/scaled displays instead of stretched.

    The SVG is rasterized into a plain QImage (pure device pixels — QPainter
    auto-applies a dpr transform on high-dpr pixmaps, which shifts/clips the
    render), and only then is the dpr stamped on the resulting QPixmap.
    """
    svg = _ICONS[name].format(c=color)
    renderer = QSvgRenderer(QByteArray(svg.encode("utf-8")))
    phys = max(1, int(round(size * dpr)))
    img = QImage(phys, phys, QImage.Format.Format_ARGB32_Premultiplied)
    img.fill(Qt.GlobalColor.transparent)
    p = QPainter(img)
    renderer.render(p)
    p.end()
    pm = QPixmap.fromImage(img)
    if dpr != 1.0:
        pm.setDevicePixelRatio(dpr)
    return QIcon(pm)
