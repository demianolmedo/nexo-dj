import os
import sys
import math
import re
import time
import cv2
import mediapipe as mp

from funcionesHand import _detect_gesture, _detect_rotation
from audioResponses import (
    process_gesture,
    set_volume,
    set_crossfader,
    get_crossfader,
    start_audio,
    get_last_block,
    get_deck_block,
    get_deck_progress,
    get_num_decks,
    get_deck_gesture,
    get_deck_volume,
    get_deck_path,
    nudge_volume,
    swap_decks,
    next_track_in_folder,
)
from visuals import (
    VisualsEngine,
    compute_rms,
    compute_bands,
    draw_frequency_bars,
    draw_hud,
    draw_crossfader,
    draw_track_card,
    draw_hand_glow,
    draw_side_legend,
)
from background import segment_person, make_dj_background, composite_person_on_bg
from recorder import SessionRecorder

if len(sys.argv) < 2:
    print("Usage: python main.py <trackA.mp3> [trackB.mp3]")
    sys.exit(1)

tracks = sys.argv[1:3]
start_audio(tracks)
dual = get_num_decks() == 2


def _clean_part(s: str) -> str:
    """Remove ft./feat. clauses, bracket/paren content, collapse whitespace."""
    s = re.sub(r"\([^)]*\)", "", s)
    s = re.sub(r"\[[^\]]*\]", "", s)
    s = re.sub(r"\s+ft\.?\s.*$", "", s, flags=re.I)
    s = re.sub(r"\s+feat\.?\s.*$", "", s, flags=re.I)
    return re.sub(r"\s+", " ", s).strip(" -.,")


def _compact_artist(s: str) -> str:
    """Shorten long artist names: strip honorifics, keep first+last if >2 words."""
    # Remove honorifics and stage-name tokens
    s = re.sub(r"\b(Jr|Sr|Dr|Mr|Mrs|Ft|Feat)\.?\b", "", s, flags=re.I)
    s = re.sub(r"\s+", " ", s).strip(" -.,")
    words = s.split()
    if len(words) > 2:
        return f"{words[0]} {words[-1]}"
    return s


def _compact_title(s: str) -> str:
    """Trim title noise: 'Remix', 'Bootleg', 'Version', etc. as trailing words."""
    s = re.sub(r"\b(Official|Audio|Video|Lyrics?|Lyric|HQ|HD|4K|Remix|Bootleg|Version|Edit|Mix|Extended)\b",
               "", s, flags=re.I)
    return re.sub(r"\s+", " ", s).strip(" -.,")


def prettify_title(path: str) -> dict:
    """Return {'artist': str, 'title': str} from a file path. Compact + human-friendly."""
    base = os.path.basename(path)
    name = os.path.splitext(base)[0]

    if " - " in name:
        artist_raw, title_raw = name.split(" - ", 1)
        artist = _compact_artist(_clean_part(artist_raw))
        title = _compact_title(_clean_part(title_raw))
        if not artist and not title:
            return {"artist": "", "title": "Untitled"}
        return {"artist": artist, "title": title or "Untitled"}

    name = _compact_title(_clean_part(name))
    return {"artist": "", "title": name or "Untitled"}


track_titles = [prettify_title(p) for p in tracks]

videoCamera = cv2.VideoCapture(0)
if not videoCamera.isOpened():
    print("Error: Could not open video capture device")
    sys.exit(1)

cv2.namedWindow('DJ Turbo', cv2.WINDOW_NORMAL)

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5,
)

visuals = VisualsEngine()
recorder = SessionRecorder()

ACTION_GESTURES = {"peace", "rock", "three", "point"}
CONTINUOUS_ACTIONS = {"three", "point"}
FX_GESTURES = {"fist", "spread", "ok", "bad", "pinch", "great"}
ACTION_COOLDOWN_SEC = 1.0
VOLUME_NUDGE_PER_FRAME = 0.015  # ~0.45/sec at 30fps
_last_action_ts = {"peace_a": 0.0, "peace_b": 0.0, "rock": 0.0}
_last_action_label = ""
_last_action_label_until = 0.0


def trigger_action(key: str, do: callable, label: str):
    global _last_action_label, _last_action_label_until
    now = time.time()
    if now - _last_action_ts.get(key, 0.0) < ACTION_COOLDOWN_SEC:
        return
    _last_action_ts[key] = now
    do()
    _last_action_label = label
    _last_action_label_until = now + 1.2

while True:
    retVal, frame = videoCamera.read()
    if not retVal:
        print("Error en videoCamera.read()")
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    # --- Background segmentation: keep person, replace bg with DJ scene ---
    person_mask = segment_person(frame)
    mixed_block = get_last_block()
    rms = compute_rms(mixed_block)
    bg = make_dj_background(frame.shape, rms, get_deck_gesture(0), get_deck_gesture(1) if dual else None)
    frame = composite_person_on_bg(frame, person_mask, bg)

    hand_centers = []
    hand_gesture_map = {}
    left_hand = None
    right_hand = None

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            wrist = hand_landmarks.landmark[0]
            wrist_px = (int(wrist.x * w), int(wrist.y * h))
            hand_centers.append(wrist_px)
            if wrist.x <= 0.5:
                left_hand = (hand_landmarks, wrist)
            else:
                right_hand = (hand_landmarks, wrist)

    def _handle_action(raw: str, deck_idx: int):
        if raw == "peace":
            def _load():
                p = next_track_in_folder(deck_idx)
                if p:
                    track_titles[deck_idx] = prettify_title(p)
            trigger_action(f"peace_{deck_idx}", _load, f"NEXT TRACK {chr(65 + deck_idx)}")
        elif raw == "rock" and dual:
            def _swap():
                swap_decks()
                track_titles[0], track_titles[1] = track_titles[1], track_titles[0]
            trigger_action("rock", _swap, "SWAP A <-> B")
        elif raw == "three":
            nudge_volume(deck_idx, VOLUME_NUDGE_PER_FRAME)
        elif raw == "point":
            nudge_volume(deck_idx, -VOLUME_NUDGE_PER_FRAME)

    gesture_a = None
    if left_hand is not None:
        raw = _detect_gesture(left_hand[0])
        if raw in ACTION_GESTURES:
            _handle_action(raw, 0)
        elif raw in FX_GESTURES:
            gesture_a = raw
            process_gesture(gesture_a, 0)

    gesture_b = None
    if dual and right_hand is not None:
        raw = _detect_gesture(right_hand[0])
        if raw in ACTION_GESTURES:
            _handle_action(raw, 1)
        elif raw in FX_GESTURES:
            gesture_b = raw
            process_gesture(gesture_b, 1)

    # Crossfader from hand heights
    if dual and left_hand is not None and right_hand is not None:
        diff = right_hand[1].y - left_hand[1].y
        xfade = max(0.0, min(1.0, 0.5 - diff))
        set_crossfader(xfade)

    # Single-track legacy: right hand rotation = volume
    if not dual and right_hand is not None:
        gesture_r = _detect_gesture(right_hand[0])
        if gesture_r == 'spread':
            angle = _detect_rotation(right_hand[0])
            volume = (angle + math.pi / 6) / (2 * math.pi / 6)
            set_volume(volume, 0)

    # Build hand-gesture map for glow (index = position in hand_centers list)
    for idx, center in enumerate(hand_centers):
        cx_norm = center[0] / w
        if cx_norm <= 0.5:
            hand_gesture_map[idx] = gesture_a
        else:
            hand_gesture_map[idx] = gesture_b if dual else gesture_a

    # Hand glow
    draw_hand_glow(frame, hand_centers, hand_gesture_map)

    # Visuals (particles)
    bands = compute_bands(mixed_block)
    visuals.emit_on_beat(rms, frame.shape)
    visuals.emit_on_gesture(hand_centers, get_deck_gesture(0), rms)
    visuals.update_and_draw(frame)

    # Frequency bars mixed
    draw_frequency_bars(frame, bands)

    # Track cards (sidebars)
    deck_a_rms = compute_rms(get_deck_block(0))
    progress_a = get_deck_progress(0)
    draw_track_card(frame, "left", track_titles[0], gesture_a, progress_a, deck_a_rms,
                    active=(gesture_a is not None), volume=get_deck_volume(0))

    if dual:
        deck_b_rms = compute_rms(get_deck_block(1))
        progress_b = get_deck_progress(1)
        draw_track_card(frame, "right", track_titles[1], gesture_b, progress_b, deck_b_rms,
                        active=(gesture_b is not None), volume=get_deck_volume(1))

    # Crossfader
    if dual:
        draw_crossfader(frame, get_crossfader())

    # Top HUD
    draw_hud(frame, gesture_a, gesture_b, dual=dual, bpm_indicator=(rms > 0.12), recording=recorder.active)

    # Side legends (always visible, minimal)
    draw_side_legend(frame, "left")
    if dual:
        draw_side_legend(frame, "right")

    # Action flash label
    if time.time() < _last_action_label_until and _last_action_label:
        remaining = _last_action_label_until - time.time()
        alpha = min(1.0, remaining / 0.4)
        label_color = (255, 255, 255)
        font_scale = 1.3
        text_size = cv2.getTextSize(_last_action_label, cv2.FONT_HERSHEY_DUPLEX, font_scale, 2)[0]
        tx = (w - text_size[0]) // 2
        ty = h // 2
        shadow = frame.copy()
        cv2.putText(shadow, _last_action_label, (tx, ty),
                    cv2.FONT_HERSHEY_DUPLEX, font_scale, (0, 0, 0), 6, cv2.LINE_AA)
        cv2.putText(shadow, _last_action_label, (tx, ty),
                    cv2.FONT_HERSHEY_DUPLEX, font_scale, label_color, 2, cv2.LINE_AA)
        cv2.addWeighted(shadow, alpha, frame, 1 - alpha, 0, frame)

    recorder.write_frame(frame)
    cv2.imshow('DJ Turbo', frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    if key == ord('r'):
        recorder.toggle(frame.shape)
    # Volume controls
    if key == ord('1'):
        nudge_volume(0, 0.05)
    if key == ord('z'):
        nudge_volume(0, -0.05)
    if key == ord('2') and dual:
        nudge_volume(1, 0.05)
    if key == ord('x') and dual:
        # Swap decks A <-> B
        swap_decks()
        track_titles[0], track_titles[1] = track_titles[1], track_titles[0]
    if key == ord('c') and dual:
        nudge_volume(1, -0.05)
    # Load next track in folder
    if key == ord('n'):
        new_path = next_track_in_folder(0)
        if new_path:
            track_titles[0] = prettify_title(new_path)
    if key == ord('m') and dual:
        new_path = next_track_in_folder(1)
        if new_path:
            track_titles[1] = prettify_title(new_path)

if recorder.active:
    recorder.stop()

videoCamera.release()
cv2.destroyAllWindows()
