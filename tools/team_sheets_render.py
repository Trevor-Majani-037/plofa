"""
Render the 2026/27 squad-depth deck.

Two entry points:

  * :func:`render_deck` — native PowerPoint (editable shapes, real crest images)
  * :func:`export_pngs` — PowerPoint COM export of that deck to PNG

Called by ``python tools/team_sheets_2627.py``.
"""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Emu, Inches, Pt

from team_sheets_2627 import TeamSheet, display_names

# ── canvas ────────────────────────────────────────────────────────────
SLIDE_W = 13.333
SLIDE_H = 7.5

PITCH_L, PITCH_T = 3.62, 1.22
PITCH_W, PITCH_H = 9.30, 6.10

INK = RGBColor(0x1A, 0x1A, 0x1A)
LINE = RGBColor(0x3C, 0x3C, 0x3C)
MUTED = RGBColor(0x6B, 0x6B, 0x6B)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)

#: 2026/27 deck typeface.
FONT = "Consolas"

TOKEN = 0.42          # crest token size, inches
PLATE_W = 1.42
LINE_H = 0.17


def _rgb(hexstr: str) -> RGBColor:
    h = hexstr.lstrip("#")
    return RGBColor(int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16))


def _luminance(hexstr: str) -> float:
    h = hexstr.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) / 255 for i in (0, 2, 4))
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def _text_on(hexstr: str) -> RGBColor:
    return INK if _luminance(hexstr) > 0.62 else WHITE


def _no_shadow(shape) -> None:
    try:
        shape.shadow.inherit = False
    except Exception:
        pass


def _pic_fit(slide, path, left, top, box_w, box_h):
    """Place an image inside a box, preserving its aspect ratio and centring it."""
    from PIL import Image

    with Image.open(path) as im:
        iw, ih = im.size
    scale = min(box_w / iw, box_h / ih)
    w, h = iw * scale, ih * scale
    return slide.shapes.add_picture(
        str(path),
        Inches(left + (box_w - w) / 2), Inches(top + (box_h - h) / 2),
        width=Inches(w), height=Inches(h),
    )


def _knockout_white(path, thresh: int = 238):
    """Return a PNG copy of `path` with its white page knocked out.

    The league crest ships as a JPEG photographed on white paper. On the title
    slide's grey band that white page reads as an ugly pasted rectangle, so the
    near-white pixels become transparent and only the badge remains.
    """
    from PIL import Image

    path = Path(path)
    cache = Path(__file__).resolve().parent.parent / "artifacts" / "crests"
    cache.mkdir(parents=True, exist_ok=True)
    out = cache / (path.stem.replace(" ", "_") + "_knockout.png")
    if out.exists() and out.stat().st_mtime >= path.stat().st_mtime:
        return out

    with Image.open(path) as im:
        rgba = im.convert("RGBA")
    px = rgba.load()
    w, h = rgba.size
    for y in range(h):
        for x in range(w):
            r, g, b, a = px[x, y]
            if r >= thresh and g >= thresh and b >= thresh:
                px[x, y] = (r, g, b, 0)
    rgba.save(out)
    return out


def _rect(slide, l, t, w, h, *, fill=None, line=None, width=1.25,
          shape=MSO_SHAPE.RECTANGLE, radius=None):
    shp = slide.shapes.add_shape(shape, Inches(l), Inches(t), Inches(w), Inches(h))
    _no_shadow(shp)
    if fill is None:
        shp.fill.background()
    else:
        shp.fill.solid()
        shp.fill.fore_color.rgb = fill
    if line is None:
        shp.line.fill.background()
    else:
        shp.line.color.rgb = line
        shp.line.width = Pt(width)
    shp.text_frame.word_wrap = False
    return shp


def _line(slide, x1, y1, x2, y2, color=LINE, width=1.25):
    from pptx.enum.shapes import MSO_CONNECTOR

    conn = slide.shapes.add_connector(
        MSO_CONNECTOR.STRAIGHT, Inches(x1), Inches(y1), Inches(x2), Inches(y2)
    )
    conn.line.color.rgb = color
    conn.line.width = Pt(width)
    _no_shadow(conn)
    return conn


def _text(slide, l, t, w, h, runs, *, size=14, color=INK, bold=True,
          align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.TOP, font=FONT,
          spacing=None):
    tb = slide.shapes.add_textbox(Inches(l), Inches(t), Inches(w), Inches(h))
    tf = tb.text_frame
    tf.word_wrap = False
    tf.margin_left = tf.margin_right = Emu(0)
    tf.margin_top = tf.margin_bottom = Emu(0)
    tf.vertical_anchor = anchor
    if isinstance(runs, str):
        runs = [runs]
    for i, line in enumerate(runs):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = align
        if spacing:
            p.line_spacing = spacing
        r = p.add_run()
        r.text = line
        r.font.size = Pt(size)
        r.font.bold = bold
        r.font.name = font
        r.font.color.rgb = color
    return tb


def _pitch(slide, accent: RGBColor) -> None:
    _rect(slide, PITCH_L, PITCH_T, PITCH_W, PITCH_H, fill=WHITE, line=LINE, width=1.25)

    mid_y = PITCH_T + PITCH_H / 2
    _line(slide, PITCH_L, mid_y, PITCH_L + PITCH_W, mid_y)

    d = 0.155 * PITCH_W
    _rect(slide, PITCH_L + PITCH_W / 2 - d / 2, mid_y - d / 2, d, d,
          fill=None, line=LINE, width=1.25, shape=MSO_SHAPE.OVAL)

    # penalty + six-yard boxes, top and bottom
    for top in (PITCH_T, PITCH_T + PITCH_H):
        sgn = 1 if top == PITCH_T else -1
        bw, bh = 0.60 * PITCH_W, 0.155 * PITCH_H
        _rect(slide, PITCH_L + (PITCH_W - bw) / 2,
              top if sgn > 0 else top - bh, bw, bh, fill=None, line=LINE)
        sw, sh = 0.30 * PITCH_W, 0.058 * PITCH_H
        _rect(slide, PITCH_L + (PITCH_W - sw) / 2,
              top if sgn > 0 else top - sh, sw, sh, fill=None, line=LINE)
        # penalty spot — kept clear of the striker's crest token at the top of
        # the box and of the GK token at the bottom
        dot = 0.055
        spot_y = top + sgn * 0.082 * PITCH_H
        _rect(slide, PITCH_L + PITCH_W / 2 - dot / 2, spot_y - dot / 2, dot, dot,
              fill=LINE, line=None, shape=MSO_SHAPE.OVAL)


def _plate_lines(slot) -> list[str]:
    names = display_names(slot.players)
    out = []
    for p, nm in zip(slot.players, names):
        no = "" if p.number is None else str(int(p.number))
        out.append(f"{no} {nm}".strip())
    return out


def _plate_rect(slot, lines, side: str):
    """(left, top, width, height) for a slot's name plate on the given side."""
    px = PITCH_L + slot.x * PITCH_W
    py = PITCH_T + slot.y * PITCH_H
    h = 0.05 + LINE_H * max(1, len(lines))
    gap = 0.03
    if side == "above":
        return px - PLATE_W / 2, py - TOKEN / 2 - gap - h, PLATE_W, h
    if side == "below":
        return px - PLATE_W / 2, py + TOKEN / 2 + gap, PLATE_W, h
    if side == "right":
        return px + TOKEN / 2 + gap, py - h / 2, PLATE_W, h
    return px - TOKEN / 2 - gap - PLATE_W, py - h / 2, PLATE_W, h


def _overlap(a, b) -> bool:
    al, at, aw, ah = a
    bl, bt, bw, bh = b
    return (al < bl + bw and bl < al + aw and at < bt + bh and bt < at + ah)


def _layout(sheet: TeamSheet):
    """Place every name plate without collisions.

    A plate prefers to sit below its crest token. It moves above, then to the
    side, only when that spot is taken by another plate or by a crest token.
    Slots are resolved from the back of the pitch forward, because the GK and
    the centre-backs are the ones squeezed against the goal line.
    """
    tokens = [
        (PITCH_L + s.x * PITCH_W - TOKEN / 2,
         PITCH_T + s.y * PITCH_H - TOKEN / 2,
         TOKEN, TOKEN)
        for s in sheet.slots
    ]
    placed: list = []
    plan = []

    for idx, slot in enumerate(sorted(range(len(sheet.slots)),
                                      key=lambda i: -sheet.slots[i].y)):
        sl = sheet.slots[idx]
        lines = _plate_lines(sl)
        best = None
        for side in ("below", "above", "right", "left"):
            rect = _plate_rect(sl, lines, side)
            l, t, w, h = rect
            if l < PITCH_L or l + w > PITCH_L + PITCH_W:
                continue
            if t < PITCH_T - 0.10 or t + h > PITCH_T + PITCH_H + 0.10:
                continue
            clashes = sum(1 for tk in tokens if _overlap(rect, tk))
            clashes += sum(1 for other in placed if _overlap(rect, other))
            if clashes == 0:
                best = rect
                break
            if best is None:
                best = rect
        placed.append(best)
        plan.append((sl, lines, best))

    plan.sort(key=lambda item: sheet.slots.index(item[0]))
    return plan



def _slot(slide, sheet: TeamSheet, slot, lines, rect) -> None:
    px = PITCH_L + slot.x * PITCH_W
    py = PITCH_T + slot.y * PITCH_H

    # crest token
    if sheet.crest and Path(sheet.crest).exists():
        _pic_fit(slide, sheet.crest,
                 px - TOKEN / 2, py - TOKEN / 2, TOKEN, TOKEN)
    else:
        badge = _rect(slide, px - TOKEN / 2, py - TOKEN / 2, TOKEN, TOKEN,
                      fill=_rgb(sheet.accent), line=None, shape=MSO_SHAPE.OVAL)
        badge.text_frame.text = ""

    # name plate
    longest = max((len(s) for s in lines), default=0)
    # Consolas is monospaced and wider per character than Arial
    size = 10 if longest <= 10 else 9 if longest <= 13 else 8 if longest <= 16 else 7
    plate_l, plate_t, plate_w, plate_h = rect
    plate = _rect(slide, plate_l, plate_t, plate_w, plate_h,
                  fill=_rgb(sheet.accent), line=None,
                  shape=MSO_SHAPE.ROUNDED_RECTANGLE)
    plate.adjustments[0] = 0.14
    tf = plate.text_frame
    tf.word_wrap = False
    tf.margin_left = tf.margin_right = Emu(0)
    tf.margin_top = Emu(0)
    tf.margin_bottom = Emu(0)
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    for i, line in enumerate(lines):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = PP_ALIGN.CENTER
        p.line_spacing = 0.95
        r = p.add_run()
        r.text = line
        r.font.size = Pt(size)
        r.font.bold = True
        r.font.name = FONT
        r.font.color.rgb = _text_on(sheet.accent)


def _club_slide(prs: Presentation, sheet: TeamSheet) -> None:
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    accent = _rgb(sheet.accent)

    _pitch(slide, accent)

    # title band
    band = _rect(slide, PITCH_L, 0.34, PITCH_W, 0.70, fill=accent, line=None,
                 shape=MSO_SHAPE.ROUNDED_RECTANGLE)
    band.adjustments[0] = 0.18
    tf = band.text_frame
    tf.word_wrap = False
    tf.margin_top = Emu(0)
    tf.margin_bottom = Emu(0)
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    p = tf.paragraphs[0]
    p.alignment = PP_ALIGN.CENTER
    r = p.add_run()
    r.text = f"SQUAD DEPTH 26/27  ·  {sheet.club.upper()}"
    r.font.size = Pt(26)
    r.font.bold = True
    r.font.name = FONT
    r.font.color.rgb = _text_on(sheet.accent)

    # left column — crest, club, formation
    if sheet.crest and Path(sheet.crest).exists():
        _pic_fit(slide, sheet.crest, 0.35, 1.45, 2.90, 2.85)
    else:
        _rect(slide, 0.42, 1.55, 2.75, 2.75, fill=accent, line=None,
              shape=MSO_SHAPE.OVAL)

    _text(slide, 0.20, 4.52, 3.20, 0.55, sheet.club.upper(), size=24, color=accent)
    _text(slide, 0.20, 5.10, 3.20, 0.40,
          f"FORMATION  {sheet.formation}", size=13, color=MUTED)
    _text(slide, 0.20, 5.48, 3.20, 0.40,
          f"SQUAD  {sheet.squad_size} PLAYERS", size=13, color=MUTED)
    _text(slide, 0.20, 6.55, 3.20, 0.40,
          "PLOFA · 2026/27", size=11, color=MUTED, bold=False)

    for slot, lines, rect in _layout(sheet):
        _slot(slide, sheet, slot, lines, rect)


def _title_slide(prs: Presentation, league_crest: Path | None, n_clubs: int,
                 n_players: int, out_of: Sequence[str], in_to: Sequence[str]) -> None:
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    red = RGBColor(0xA6, 0x19, 0x2E)
    band = RGBColor(0xEC, 0xEC, 0xEC)

    _rect(slide, 0.85, 1.45, 11.6, 3.55, fill=band, line=None)

    if league_crest and Path(league_crest).exists():
        _pic_fit(slide, _knockout_white(league_crest), 1.05, 1.85, 2.40, 2.75)

    _text(slide, 3.65, 1.75, 8.5, 0.9, "PLOFA", size=60, color=red,
          align=PP_ALIGN.LEFT)
    _text(slide, 3.65, 2.65, 8.5, 0.9, "PLAYERS PER TEAM 26/27", size=40,
          color=red, align=PP_ALIGN.LEFT)
    _text(slide, 3.65, 3.65, 8.5, 0.45, "Squads   ·   Formations", size=20,
          color=red, align=PP_ALIGN.LEFT, bold=False)

    _text(slide, 0.85, 5.35, 11.6, 0.45,
          f"{n_clubs} clubs   ·   {n_players} players   ·   every club's depth chart "
          f"in the shape it plays", size=15, color=INK, bold=False)
    _text(slide, 0.85, 5.85, 11.6, 0.45,
          "Promoted in:  " + ", ".join(in_to) + "      Relegated out:  " + ", ".join(out_of),
          size=13, color=MUTED, bold=False)
    _text(slide, 0.85, 6.30, 11.6, 0.45,
          "Built from PLOFA-2026-2027.xlsx (PLAYERS sheet) — the same workbook the "
          "match engine loads.", size=12, color=MUTED, bold=False)


def render_deck(sheets: Sequence[TeamSheet], out_pptx: Path,
                league_crest: Path | None = None,
                out_of: Sequence[str] = (), in_to: Sequence[str] = ()) -> Path:
    prs = Presentation()
    prs.slide_width = Inches(SLIDE_W)
    prs.slide_height = Inches(SLIDE_H)

    _title_slide(prs, league_crest, len(sheets),
                 sum(s.squad_size for s in sheets), out_of, in_to)
    for sheet in sheets:
        _club_slide(prs, sheet)

    out_pptx.parent.mkdir(parents=True, exist_ok=True)
    prs.save(str(out_pptx))
    return out_pptx


def export_pngs(pptx_path: Path, png_dir: Path, width: int = 1920,
                height: int = 1080) -> Path:
    """Export every slide to PNG through the installed PowerPoint."""
    import win32com.client as win32

    png_dir.mkdir(parents=True, exist_ok=True)
    app = win32.Dispatch("PowerPoint.Application")
    pres = None
    try:
        pres = app.Presentations.Open(str(pptx_path), WithWindow=False)
        pres.Export(str(png_dir), "PNG", width, height)
    finally:
        if pres is not None:
            pres.Close()
        app.Quit()
    return png_dir
