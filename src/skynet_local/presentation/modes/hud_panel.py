"""Side-panel HUD drawn from SceneTracker data — one card per person on scene."""

from __future__ import annotations
import time
import cv2
import numpy as np
from skynet_local.domain.scene_tracker import SceneTracker
from skynet_local.infrastructure.vision.attributes.emotion_analyzer import emotion_to_emoticon

# Layout constants
_PANEL_W       = 280
_CARD_PAD      = 10
_CARD_MARGIN   = 8
_BAR_H         = 8
_BAR_W         = 160
_FONT          = cv2.FONT_HERSHEY_SIMPLEX
_FONT_SM       = 0.38
_FONT_MD       = 0.48
_FONT_LG       = 0.55
_THICK_THIN    = 1
_THICK_MED     = 1

# Colors (BGR)
_BG            = (18,  18,  18)
_CARD_BG       = (32,  32,  32)
_CARD_BORDER   = (60,  60,  60)
_TEXT_PRIMARY  = (220, 220, 220)
_TEXT_MUTED    = (130, 130, 130)
_GREEN         = (60,  210, 100)
_YELLOW        = (40,  210, 220)
_RED           = (70,   70, 220)
_CYAN          = (200, 200,  50)
_WHITE         = (255, 255, 255)

# Emotion bar colours (BGR) — per class
_EMOTION_COLORS: dict[str, tuple] = {
    "happiness": (50,  200,  80),
    "neutral":   (150, 150, 150),
    "surprise":  (50,  200, 220),
    "sadness":   (200,  80,  50),
    "anger":     (50,   50, 220),
    "disgust":   (50,  150,  50),
    "fear":      (180, 100, 200),
    "contempt":  (100, 100, 180),
}

_EMOTION_ORDER = [
    "happiness", "sadness", "anger", "surprise",
    "neutral",   "fear",    "disgust", "contempt",
]


def _bar(img, x, y, w, h, frac, color, bg=(50, 50, 50)):
    frac = max(0.0, min(1.0, frac))
    cv2.rectangle(img, (x, y), (x + w, y + h), bg, -1)
    if frac > 0:
        cv2.rectangle(img, (x, y), (x + int(w * frac), y + h), color, -1)


def _text(img, txt, x, y, scale, color, thick=_THICK_THIN):
    cv2.putText(img, txt, (x, y), _FONT, scale, color, thick, cv2.LINE_AA)


def draw_hud_panel(frame: np.ndarray) -> np.ndarray:
    """Attach a right-side HUD panel to frame and return the combined image."""
    tracker = SceneTracker.instance()
    persons = tracker.persons

    h, w = frame.shape[:2]
    panel = np.full((h, _PANEL_W, 3), _BG, dtype=np.uint8)

    # Header
    _text(panel, "SCENE TRACKER", _CARD_PAD, 22, _FONT_MD, _CYAN, _THICK_MED)
    cv2.line(panel, (_CARD_PAD, 28), (_PANEL_W - _CARD_PAD, 28), _CARD_BORDER, 1)

    y = 36

    if not persons:
        _text(panel, "no persons on scene", _CARD_PAD, y + 16, _FONT_SM, _TEXT_MUTED)
    else:
        for person in persons.values():
            y = _draw_person_card(panel, person, y, h)
            if y >= h - 10:
                break

    return np.hstack([frame, panel])


def _draw_person_card(panel, person, y_start: int, panel_h: int) -> int:
    """Draw one person card. Returns y position after the card."""
    # Estimate card height: header + id bar + emotion bars * 8 + chewing
    n_emotions = len(_EMOTION_ORDER)
    card_h = 28 + 20 + n_emotions * 14 + 22 + _CARD_PAD * 2
    x0, x1 = _CARD_PAD, _PANEL_W - _CARD_PAD
    y0, y1 = y_start + _CARD_MARGIN, y_start + _CARD_MARGIN + card_h

    if y1 > panel_h - 4:
        return panel_h  # no space

    # Card background
    cv2.rectangle(panel, (x0, y0), (x1, y1), _CARD_BG, -1)
    cv2.rectangle(panel, (x0, y0), (x1, y1), _CARD_BORDER, 1)

    cx = x0 + _CARD_PAD
    cy = y0 + _CARD_PAD + 12

    # Name + emoticon
    emo_icon = emotion_to_emoticon(person.dominant_emotion)
    name_str = f"{person.label}  {emo_icon}".strip()
    _text(panel, name_str, cx, cy, _FONT_LG, _GREEN, _THICK_MED)
    cy += 4

    # Identity confidence bar
    cy += 14
    _text(panel, "ID CONF", cx, cy, _FONT_SM, _TEXT_MUTED)
    id_pct = f"{person.identity_prob * 100:.0f}%"
    _text(panel, id_pct, cx + _BAR_W + 4, cy, _FONT_SM, _TEXT_PRIMARY)
    cy += 3
    _bar(panel, cx, cy, _BAR_W, _BAR_H, person.identity_prob, _GREEN)
    cy += _BAR_H + 8

    # Emotion bars
    _text(panel, "EMOTION", cx, cy, _FONT_SM, _TEXT_MUTED)
    cy += 3
    for emo in _EMOTION_ORDER:
        prob = person.emotion_probs.get(emo, 0.0)
        color = _EMOTION_COLORS.get(emo, _TEXT_MUTED)
        # Dim all bars except dominant
        is_dominant = (emo == person.dominant_emotion)
        bar_color = color if is_dominant else tuple(max(0, c - 80) for c in color)
        label_color = _TEXT_PRIMARY if is_dominant else _TEXT_MUTED
        short = emo[:3].upper()
        _text(panel, short, cx, cy + 7, _FONT_SM, label_color)
        pct = f"{prob * 100:.0f}%"
        _bar(panel, cx + 28, cy, _BAR_W - 28, _BAR_H, prob, bar_color)
        _text(panel, pct, cx + _BAR_W + 4, cy + 7, _FONT_SM, label_color)
        cy += 14

    # Chewing / eating
    cy += 4
    if person.is_chewing and person.chewing_duration_sec > 0:
        chew_str = f"CHEWING  {person.chewing_duration_sec:.1f}s"
        _text(panel, chew_str, cx, cy, _FONT_SM, _YELLOW, _THICK_MED)
    else:
        _text(panel, "chewing: -", cx, cy, _FONT_SM, _TEXT_MUTED)

    return y1 + 2