import math
import random
import time
import numpy as np
import cv2

GESTURE_COLORS = {
    "fist":   (80, 80, 255),
    "spread": (80, 255, 160),
    "ok":     (255, 200, 80),
    "bad":    (255, 80, 200),
    "pinch":  (200, 80, 255),
    "great":  (80, 255, 255),
    None:     (200, 200, 200),
}

DECK_A_COLOR = (80, 255, 160)
DECK_B_COLOR = (200, 80, 255)

NUM_BANDS = 32
_smoothed_bands = np.zeros(NUM_BANDS, dtype=np.float32)


def compute_rms(block: np.ndarray) -> float:
    if block.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(block.astype(np.float32) ** 2)))


def compute_bands(block: np.ndarray, num_bands: int = NUM_BANDS) -> np.ndarray:
    global _smoothed_bands
    if block.size < 64:
        return _smoothed_bands

    window = np.hanning(len(block))
    spectrum = np.abs(np.fft.rfft(block * window))
    edges = np.logspace(0, np.log10(len(spectrum) - 1), num_bands + 1).astype(int)

    bands = np.zeros(num_bands, dtype=np.float32)
    for i in range(num_bands):
        lo, hi = edges[i], max(edges[i] + 1, edges[i + 1])
        bands[i] = spectrum[lo:hi].mean() if hi > lo else 0.0

    max_val = bands.max()
    if max_val > 0:
        bands = bands / max_val

    _smoothed_bands = _smoothed_bands * 0.6 + bands * 0.4
    return _smoothed_bands


class Particle:
    __slots__ = ("x", "y", "vx", "vy", "life", "max_life", "color", "size")

    def __init__(self, x, y, vx, vy, life, color, size):
        self.x = x
        self.y = y
        self.vx = vx
        self.vy = vy
        self.life = life
        self.max_life = life
        self.color = color
        self.size = size

    def step(self):
        self.x += self.vx
        self.y += self.vy
        self.vy += 0.12
        self.life -= 1

    def draw(self, frame):
        if self.life <= 0:
            return
        alpha = self.life / self.max_life
        c = tuple(int(ch * alpha) for ch in self.color)
        cv2.circle(frame, (int(self.x), int(self.y)), max(1, int(self.size * alpha)), c, -1, cv2.LINE_AA)


class VisualsEngine:
    def __init__(self):
        self.particles: list[Particle] = []
        self.beat_threshold = 0.12
        self._last_rms = 0.0

    def spawn_particles(self, center, color, count, intensity):
        cx, cy = center
        for _ in range(count):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(2, 8) * (0.5 + intensity)
            vx = math.cos(angle) * speed
            vy = math.sin(angle) * speed - 2
            life = random.randint(20, 50)
            size = random.uniform(3, 7) * (0.7 + intensity)
            self.particles.append(Particle(cx, cy, vx, vy, life, color, size))

    def emit_on_gesture(self, hand_centers, gesture, rms):
        if not hand_centers:
            return
        color = GESTURE_COLORS.get(gesture, GESTURE_COLORS[None])
        count = int(2 + rms * 20)
        for c in hand_centers:
            self.spawn_particles(c, color, count, rms)

    def emit_on_beat(self, rms, frame_shape):
        if rms > self.beat_threshold and rms > self._last_rms * 1.4:
            h, w = frame_shape[:2]
            color = (255, 255, 255)
            for _ in range(12):
                x = random.uniform(0, w)
                y = random.uniform(h * 0.5, h)
                angle = random.uniform(-math.pi, 0)
                speed = random.uniform(6, 14)
                vx = math.cos(angle) * speed
                vy = math.sin(angle) * speed
                self.particles.append(Particle(x, y, vx, vy, 40, color, 4))
        self._last_rms = rms

    def update_and_draw(self, frame):
        alive = []
        for p in self.particles:
            p.step()
            if p.life > 0:
                p.draw(frame)
                alive.append(p)
        self.particles = alive[-500:]


def draw_frequency_bars(frame, bands, color=(0, 255, 200), baseline_y=None, max_height=100):
    h, w = frame.shape[:2]
    if baseline_y is None:
        baseline_y = h - 80
    n = len(bands)
    usable_w = int(w * 0.6)
    offset_x = (w - usable_w) // 2
    bar_w = max(2, usable_w // n)
    for i, amp in enumerate(bands):
        bar_h = int(amp * max_height)
        x1 = offset_x + i * bar_w
        x2 = x1 + bar_w - 1
        y1 = baseline_y - bar_h
        y2 = baseline_y
        # gradient from blue-green at bottom to magenta at top
        mix = amp
        c = (int(80 + mix * 120), int(255 - mix * 100), int(160 + mix * 60))
        cv2.rectangle(frame, (x1, y1), (x2, y2), c, -1)


def draw_hud(frame, gesture_a, gesture_b=None, dual=False, bpm_indicator=False, recording=False):
    h, w = frame.shape[:2]

    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 70), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

    cv2.putText(frame, "DJ TURBO", (20, 45),
                cv2.FONT_HERSHEY_DUPLEX, 1.2, (255, 255, 255), 2, cv2.LINE_AA)

    if bpm_indicator:
        cv2.circle(frame, (w // 2, 40), 14, (0, 255, 255), -1, cv2.LINE_AA)

    if recording:
        cv2.circle(frame, (w - 90, 40), 10, (0, 0, 255), -1, cv2.LINE_AA)
        cv2.putText(frame, "REC", (w - 75, 47),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)

    help_text = (
        "q:quit  r:rec  |  peace=next track  rock=swap A<->B  |  hand height = crossfader"
        if dual else "q:quit  r:rec  |  peace=next track"
    )
    cv2.putText(frame, help_text,
                (20, h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1, cv2.LINE_AA)


def draw_crossfader(frame, value: float):
    h, w = frame.shape[:2]
    bar_w = 280
    bar_h = 12
    x0 = (w - bar_w) // 2
    y0 = h - 40
    cv2.rectangle(frame, (x0, y0), (x0 + bar_w, y0 + bar_h), (40, 40, 50), -1)
    mid_x = x0 + int(value * bar_w)
    cv2.rectangle(frame, (x0, y0), (mid_x, y0 + bar_h), DECK_A_COLOR, -1)
    cv2.rectangle(frame, (mid_x, y0), (x0 + bar_w, y0 + bar_h), DECK_B_COLOR, -1)
    cv2.circle(frame, (mid_x, y0 + bar_h // 2), 9, (255, 255, 255), -1, cv2.LINE_AA)
    cv2.circle(frame, (mid_x, y0 + bar_h // 2), 9, (0, 0, 0), 2, cv2.LINE_AA)


def draw_jog_wheel(frame, center, radius, color, progress_angle_rad: float, rms: float, label: str):
    """Spinning jog wheel with progress ring and label."""
    cx, cy = center
    # outer ring
    cv2.circle(frame, (cx, cy), radius, (40, 40, 48), -1, cv2.LINE_AA)
    cv2.circle(frame, (cx, cy), radius, color, 2, cv2.LINE_AA)
    # inner disc
    inner_r = int(radius * 0.65)
    cv2.circle(frame, (cx, cy), inner_r, (18, 18, 22), -1, cv2.LINE_AA)

    # Moving indicator — tick that rotates with the track (simulating jog wheel turning)
    tick_len = radius - 4
    tx = int(cx + tick_len * math.cos(progress_angle_rad))
    ty = int(cy + tick_len * math.sin(progress_angle_rad))
    cv2.line(frame, (cx, cy), (tx, ty), color, 3, cv2.LINE_AA)

    # Pulse on beat
    pulse_r = int(inner_r * (0.6 + rms * 0.4))
    pulse_color = tuple(int(c * (0.3 + rms * 0.7)) for c in color)
    cv2.circle(frame, (cx, cy), pulse_r, pulse_color, -1, cv2.LINE_AA)

    # Label
    text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_DUPLEX, 0.7, 2)[0]
    cv2.putText(frame, label, (cx - text_size[0] // 2, cy + text_size[1] // 2),
                cv2.FONT_HERSHEY_DUPLEX, 0.7, (240, 240, 240), 2, cv2.LINE_AA)


def draw_vu_meter(frame, rect, rms: float, color):
    """Vertical VU meter inside rect (x, y, w, h)."""
    x, y, ww, hh = rect
    cv2.rectangle(frame, (x, y), (x + ww, y + hh), (30, 30, 38), -1)
    cv2.rectangle(frame, (x, y), (x + ww, y + hh), (80, 80, 88), 1)
    level_h = int(hh * min(1.0, rms * 4))
    y_top = y + hh - level_h
    # segmented
    seg = 6
    for i in range(seg):
        seg_y1 = y + hh - int(hh * (i + 1) / seg)
        seg_y2 = y + hh - int(hh * i / seg) - 2
        if seg_y2 > y_top:
            fade = 1 - (i / seg) * 0.6
            c = tuple(int(ch * fade) for ch in color)
            if i >= seg - 1:
                c = (80, 80, 255)  # red zone
            cv2.rectangle(frame, (x + 2, seg_y1), (x + ww - 2, seg_y2), c, -1)


def _truncate(text: str, max_chars: int) -> str:
    text = text.strip()
    if len(text) <= max_chars:
        return text
    return text[:max_chars - 1].rstrip() + "…"


def draw_track_card(frame, side: str, meta, gesture: str | None, progress: float, rms: float, active: bool, volume: float = 0.8):
    """Minimalist sidebar card. `meta` is a dict {'artist', 'title'} or a string."""
    if isinstance(meta, str):
        artist, title = "", meta
    else:
        artist = meta.get("artist", "") or ""
        title = meta.get("title", "") or ""

    h, w = frame.shape[:2]
    card_w = 220
    card_h = 150
    margin = 18
    pad = 12

    if side == "left":
        x0 = margin
        color = DECK_A_COLOR
        label = "A"
    else:
        x0 = w - card_w - margin
        color = DECK_B_COLOR
        label = "B"
    y0 = 86

    # Background strip
    overlay = frame.copy()
    cv2.rectangle(overlay, (x0, y0), (x0 + card_w, y0 + card_h), (10, 10, 14), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

    # Left accent bar
    accent_w = 4 if active else 1
    cv2.rectangle(frame, (x0, y0), (x0 + accent_w, y0 + card_h), color, -1)

    # Deck label (big letter top-left)
    cv2.putText(frame, label, (x0 + pad, y0 + 30),
                cv2.FONT_HERSHEY_DUPLEX, 1.0, color, 2, cv2.LINE_AA)

    # Artist (small, muted) and title (larger, white)
    max_chars = 24
    artist_trunc = _truncate(artist.upper(), max_chars)
    title_trunc = _truncate(title, max_chars)

    text_x = x0 + pad + 32
    cv2.putText(frame, artist_trunc, (text_x, y0 + 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (160, 160, 160), 1, cv2.LINE_AA)
    cv2.putText(frame, title_trunc, (text_x, y0 + 42),
                cv2.FONT_HERSHEY_DUPLEX, 0.5, (240, 240, 240), 1, cv2.LINE_AA)

    # Slim VU strip under title
    vu_y = y0 + 58
    draw_vu_meter(frame, (x0 + pad, vu_y, card_w - pad * 2, 6), rms, color)

    # FX + VOL on one line
    gcolor = GESTURE_COLORS.get(gesture, GESTURE_COLORS[None])
    cv2.putText(frame, f"FX {gesture or '-'}", (x0 + pad, y0 + 92),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, gcolor, 1, cv2.LINE_AA)
    cv2.putText(frame, f"VOL {int(volume * 100)}%", (x0 + card_w - 96, y0 + 92),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, (220, 220, 220), 1, cv2.LINE_AA)

    # Progress bar
    pb_y = y0 + card_h - 26
    pb_w = card_w - pad * 2
    cv2.rectangle(frame, (x0 + pad, pb_y), (x0 + pad + pb_w, pb_y + 4), (40, 40, 50), -1)
    cv2.rectangle(frame, (x0 + pad, pb_y), (x0 + pad + int(pb_w * progress), pb_y + 4), color, -1)
    cv2.putText(frame, f"{int(progress * 100)}%", (x0 + pad, pb_y + 16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (160, 160, 160), 1, cv2.LINE_AA)


def draw_side_legend(frame, side: str):
    """Minimal gesture legend on the side, below the track card. Stays visible always."""
    h, w = frame.shape[:2]
    items_fx = [
        ("fist", "fist mute"),
        ("spread", "spread +6dB"),
        ("ok", "ok high-pass"),
        ("bad", "bad low-pass"),
        ("pinch", "pinch -10dB"),
        ("great", "great pitch+"),
    ]
    items_action = [
        (None, "peace next track"),
        (None, "rock swap A<->B"),
        (None, "3 dedos vol +"),
        (None, "1 dedo (point) vol -"),
        (None, "altura crossfader"),
    ]

    card_w = 220
    margin = 18
    pad = 12
    x0 = margin if side == "left" else w - card_w - margin
    y0 = 86 + 150 + 10  # below track card

    overlay = frame.copy()
    legend_h = (len(items_fx) + len(items_action) + 2) * 18 + 12
    cv2.rectangle(overlay, (x0, y0), (x0 + card_w, y0 + legend_h), (10, 10, 14), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

    # "FX" heading
    cv2.putText(frame, "FX", (x0 + pad, y0 + 16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (140, 200, 160), 1, cv2.LINE_AA)
    for i, (key, label) in enumerate(items_fx):
        yy = y0 + 32 + i * 16
        c = GESTURE_COLORS.get(key, (180, 180, 180))
        cv2.circle(frame, (x0 + pad + 4, yy - 4), 3, c, -1, cv2.LINE_AA)
        cv2.putText(frame, label, (x0 + pad + 14, yy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (210, 210, 210), 1, cv2.LINE_AA)

    # "ACTIONS" heading
    action_base = y0 + 32 + len(items_fx) * 16 + 8
    cv2.putText(frame, "ACCIONES", (x0 + pad, action_base),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 180, 80), 1, cv2.LINE_AA)
    for i, (_, label) in enumerate(items_action):
        yy = action_base + 16 + i * 16
        cv2.putText(frame, label, (x0 + pad + 14, yy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (210, 210, 210), 1, cv2.LINE_AA)


def draw_gesture_help(frame, alpha: float = 0.85):
    """Translucent panel listing gestures and their effects. Pops up when toggled."""
    if alpha <= 0:
        return
    h, w = frame.shape[:2]

    panel_w = 520
    panel_h = 360
    x0 = (w - panel_w) // 2
    y0 = (h - panel_h) // 2

    # Background with blur-ish feel
    overlay = frame.copy()
    cv2.rectangle(overlay, (x0, y0), (x0 + panel_w, y0 + panel_h), (12, 12, 15), -1)
    cv2.rectangle(overlay, (x0, y0), (x0 + panel_w, y0 + panel_h), (255, 255, 255), 1, cv2.LINE_AA)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

    # Title
    cv2.putText(frame, "GESTOS", (x0 + 20, y0 + 38),
                cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)

    # Section FX
    cv2.putText(frame, "EFECTOS (cada mano sobre su deck)", (x0 + 20, y0 + 72),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (120, 255, 180), 1, cv2.LINE_AA)

    fx_items = [
        ("fist", "fist", "mute (-100 dB)"),
        ("spread", "spread", "boost (+6 dB)"),
        ("ok", "ok", "high-pass (quita graves)"),
        ("bad", "bad (pulgar abajo)", "low-pass (quita agudos)"),
        ("pinch", "pinch", "vol -10 dB"),
        ("great", "great (pinch muy cerrado)", "pitch up +7 semitonos"),
    ]
    base_y = y0 + 92
    for i, (key, label, desc) in enumerate(fx_items):
        color = GESTURE_COLORS.get(key, (200, 200, 200))
        yy = base_y + i * 22
        cv2.circle(frame, (x0 + 26, yy - 5), 5, color, -1, cv2.LINE_AA)
        cv2.putText(frame, f"{label:<28}{desc}", (x0 + 40, yy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (230, 230, 230), 1, cv2.LINE_AA)

    # Section ACTIONS
    action_y = base_y + len(fx_items) * 22 + 18
    cv2.putText(frame, "ACCIONES", (x0 + 20, action_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 180, 80), 1, cv2.LINE_AA)

    action_items = [
        ("peace (V sign, indice+medio)", "siguiente track del deck"),
        ("rock (indice+menique)", "swap A <-> B"),
        ("altura de la mano", "crossfader (mas alta = mas fuerte)"),
    ]
    for i, (label, desc) in enumerate(action_items):
        yy = action_y + 20 + i * 22
        cv2.putText(frame, f"{label:<28}{desc}", (x0 + 40, yy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (230, 230, 230), 1, cv2.LINE_AA)

    # Section KEYS
    keys_y = action_y + 20 + len(action_items) * 22 + 16
    cv2.putText(frame, "TECLAS:  q = salir   r = grabar   h = ocultar ayuda",
                (x0 + 20, keys_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, (180, 180, 180), 1, cv2.LINE_AA)


def draw_hand_glow(frame, hand_centers, gesture_map: dict):
    """Soft glow around each hand, color depends on its deck's gesture."""
    for idx, center in enumerate(hand_centers):
        gesture = gesture_map.get(idx)
        color = GESTURE_COLORS.get(gesture, (180, 180, 180))
        overlay = frame.copy()
        for r, alpha in [(90, 0.12), (60, 0.18), (35, 0.25)]:
            cv2.circle(overlay, center, r, color, -1, cv2.LINE_AA)
        cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)
