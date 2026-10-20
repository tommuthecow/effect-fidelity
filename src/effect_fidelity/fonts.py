"""DM Sans everywhere: reportlab (embedded in PDFs), matplotlib (charts), and HTML (@font-face, base64-embedded)."""
from __future__ import annotations
import base64
from pathlib import Path

DIR = Path(__file__).resolve().parents[2] / "assets" / "fonts"
FILES = {"DMSans": "DMSans-Regular.ttf", "DMSans-Bold": "DMSans-Bold.ttf", "DMSans-Italic": "DMSans-Italic.ttf"}
REG, BOLD = "DMSans", "DMSans-Bold"


def reportlab():
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from reportlab.lib.fonts import addMapping
    for name, f in FILES.items():
        if name not in pdfmetrics.getRegisteredFontNames():
            pdfmetrics.registerFont(TTFont(name, str(DIR / f)))
    addMapping("DMSans", 0, 0, "DMSans"); addMapping("DMSans", 1, 0, "DMSans-Bold")
    addMapping("DMSans", 0, 1, "DMSans-Italic"); addMapping("DMSans", 1, 1, "DMSans-Bold")


def matplotlib():
    import matplotlib as mpl
    from matplotlib import font_manager
    for f in FILES.values():
        font_manager.fontManager.addfont(str(DIR / f))
    mpl.rcParams["font.family"] = "DM Sans"


def css() -> str:
    def face(f, w, s):
        b = base64.b64encode((DIR / f).read_bytes()).decode()
        return f"@font-face{{font-family:'DM Sans';font-weight:{w};font-style:{s};src:url(data:font/ttf;base64,{b}) format('truetype')}}"
    return face(FILES["DMSans"], 400, "normal") + face(FILES["DMSans-Bold"], 700, "normal") + face(FILES["DMSans-Italic"], 400, "italic")
