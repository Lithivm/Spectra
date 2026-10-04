"""SVG icon set — rendered at runtime via QSvgRenderer (no external files).

Icons are 24x24 viewBox templates with a single {c} color placeholder.
Rendering to QPixmap keeps PyInstaller packaging trivial (no data files).
"""

from __future__ import annotations

from PyQt6.QtCore import QByteArray, Qt
from PyQt6.QtGui import QIcon, QPainter, QPixmap
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
    "logo": (
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24">'
        '<rect x="2"    y="8"   width="3" height="8"  rx="1.5" fill="{c}"/>'
        '<rect x="7.7"  y="5"   width="3" height="14" rx="1.5" fill="{c}"/>'
        '<rect x="13.3" y="7"   width="3" height="10" rx="1.5" fill="{c}"/>'
        '<rect x="19"   y="6"   width="3" height="12" rx="1.5" fill="{c}"/>'
        '</svg>'
    ),
}


def render_icon(name: str, size: int = 16, color: str = "#F0EDE8") -> QIcon:
    """Render an SVG icon template to a QIcon at the given pixel size."""
    svg = _ICONS[name].format(c=color)
    renderer = QSvgRenderer(QByteArray(svg.encode("utf-8")))
    pm = QPixmap(size, size)
    pm.fill(Qt.GlobalColor.transparent)
    p = QPainter(pm)
    renderer.render(p)
    p.end()
    return QIcon(pm)
