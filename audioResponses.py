import sounddevice as sd
import numpy as np
import threading
from pedalboard import Pedalboard, Gain, HighpassFilter, LowpassFilter, PitchShift
import librosa

blocksize = 1024

stream = None
_decks: list["Deck"] = []
_last_block = np.zeros(blocksize, dtype=np.float32)
_block_lock = threading.Lock()
_recorded_chunks = []
_recording = False
_record_lock = threading.Lock()
_crossfader = 0.5

boards = {
    "spread": Pedalboard([Gain(gain_db=6)]),
    "pinch":  Pedalboard([Gain(gain_db=-10)]),
    "ok":     Pedalboard([HighpassFilter(cutoff_frequency_hz=3000)]),
    "bad":    Pedalboard([LowpassFilter(cutoff_frequency_hz=1000)]),
    "fist":   Pedalboard([Gain(gain_db=-100)]),
    "great":  Pedalboard([PitchShift(semitones=7)]),
}


class Deck:
    def __init__(self, name: str, path: str):
        self.name = name
        self.path = path
        self.y, self.sr = librosa.load(path, sr=None)
        self.ptr = 0
        self.gesture: str | None = None
        self.volume = 0.8
        self.last_block = np.zeros(blocksize, dtype=np.float32)

    def read_block(self, frames: int) -> np.ndarray:
        block = self.y[self.ptr:self.ptr + frames]
        if len(block) < frames:
            block = np.pad(block, (0, frames - len(block)))
        self.ptr = (self.ptr + frames) % len(self.y)

        if self.gesture in boards:
            block = boards[self.gesture](block, self.sr)

        block = block * self.volume
        self.last_block = block.astype(np.float32, copy=True)
        return block

    def load(self, path: str):
        new_y, new_sr = librosa.load(path, sr=None)
        self.path = path
        self.y = new_y
        self.sr = new_sr
        self.ptr = 0
        self.last_block = np.zeros(blocksize, dtype=np.float32)


def audio_callback(outdata, frames, time_info, status):
    global _last_block

    if len(_decks) == 0:
        outdata.fill(0.0)
        return

    if len(_decks) == 1:
        mixed = _decks[0].read_block(frames)
    else:
        a = _decks[0].read_block(frames)
        b = _decks[1].read_block(frames)
        # crossfader: 0.0 = full A, 1.0 = full B, 0.5 = 50/50
        gain_a = np.cos(_crossfader * np.pi / 2)
        gain_b = np.sin(_crossfader * np.pi / 2)
        mixed = a * gain_a + b * gain_b

    outdata[:] = mixed.reshape(-1, 1)

    with _block_lock:
        _last_block = mixed.astype(np.float32, copy=True)
    if _recording:
        with _record_lock:
            _recorded_chunks.append(mixed.astype(np.float32, copy=True))


def start_audio(paths: list[str]):
    global stream, _decks
    _decks = [Deck(f"Deck {chr(65+i)}", p) for i, p in enumerate(paths)]
    sr = _decks[0].sr
    if len(_decks) > 1 and _decks[1].sr != sr:
        # Resample deck B to match A
        _decks[1].y = librosa.resample(_decks[1].y, orig_sr=_decks[1].sr, target_sr=sr)
        _decks[1].sr = sr
    stream = sd.OutputStream(callback=audio_callback, samplerate=sr, channels=1, blocksize=blocksize)
    stream.start()


def get_last_block():
    with _block_lock:
        return _last_block.copy()


def get_deck_block(index: int) -> np.ndarray:
    if 0 <= index < len(_decks):
        return _decks[index].last_block.copy()
    return np.zeros(blocksize, dtype=np.float32)


def get_deck_progress(index: int) -> float:
    if 0 <= index < len(_decks):
        deck = _decks[index]
        if len(deck.y) > 0:
            return deck.ptr / len(deck.y)
    return 0.0


def get_sample_rate():
    return _decks[0].sr if _decks else None


def get_num_decks() -> int:
    return len(_decks)


def get_crossfader() -> float:
    return _crossfader


def set_crossfader(value: float):
    global _crossfader
    _crossfader = max(0.0, min(1.0, float(value)))


def process_gesture(gesture: str, deck_index: int = 0):
    if 0 <= deck_index < len(_decks):
        deck = _decks[deck_index]
        if gesture != deck.gesture:
            print(f"[{deck.name}] Gesto: {deck.gesture} → {gesture}")
        deck.gesture = gesture


def set_volume(volume: float, deck_index: int = 0):
    if 0 <= deck_index < len(_decks):
        _decks[deck_index].volume = max(0.0, min(1.0, float(volume)))


def get_deck_gesture(deck_index: int) -> str | None:
    if 0 <= deck_index < len(_decks):
        return _decks[deck_index].gesture
    return None


def get_deck_volume(deck_index: int) -> float:
    if 0 <= deck_index < len(_decks):
        return _decks[deck_index].volume
    return 0.0


def get_deck_path(deck_index: int) -> str | None:
    if 0 <= deck_index < len(_decks):
        return _decks[deck_index].path
    return None


def nudge_volume(deck_index: int, delta: float):
    if 0 <= deck_index < len(_decks):
        d = _decks[deck_index]
        d.volume = max(0.0, min(1.0, d.volume + delta))


def swap_decks():
    if len(_decks) == 2:
        a, b = _decks[0], _decks[1]
        # Swap path/y/sr/ptr/volume/gesture; keep name labels
        a.y, b.y = b.y, a.y
        a.sr, b.sr = b.sr, a.sr
        a.ptr, b.ptr = b.ptr, a.ptr
        a.path, b.path = b.path, a.path
        a.volume, b.volume = b.volume, a.volume
        a.gesture, b.gesture = b.gesture, a.gesture
        a.last_block, b.last_block = b.last_block, a.last_block


def load_track(deck_index: int, path: str):
    if 0 <= deck_index < len(_decks):
        _decks[deck_index].load(path)


def next_track_in_folder(deck_index: int) -> str | None:
    """Cycle to the next audio file in the current track's folder. Returns new path."""
    import os
    from pathlib import Path
    if not (0 <= deck_index < len(_decks)):
        return None
    current = Path(_decks[deck_index].path)
    folder = current.parent
    exts = {".mp3", ".wav", ".m4a", ".flac", ".ogg"}
    candidates = sorted([p for p in folder.iterdir() if p.suffix.lower() in exts])
    if not candidates:
        return None
    try:
        idx = candidates.index(current)
    except ValueError:
        idx = -1
    new_path = str(candidates[(idx + 1) % len(candidates)])
    _decks[deck_index].load(new_path)
    return new_path


def start_recording():
    global _recording, _recorded_chunks
    with _record_lock:
        _recorded_chunks = []
    _recording = True


def stop_recording():
    global _recording
    _recording = False
    with _record_lock:
        if not _recorded_chunks:
            return None
        return np.concatenate(_recorded_chunks)
