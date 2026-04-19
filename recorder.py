import os
import subprocess
import time
from pathlib import Path
import cv2
import numpy as np
import soundfile as sf

import audioResponses


class SessionRecorder:
    def __init__(self, output_dir: str = "recordings"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.writer = None
        self.video_path = None
        self.audio_path = None
        self.final_path = None
        self.active = False
        self.start_ts = None
        self.fps = 30

    def toggle(self, frame_shape):
        if self.active:
            return self.stop()
        self.start(frame_shape)
        return None

    def start(self, frame_shape):
        h, w = frame_shape[:2]
        ts = time.strftime("%Y%m%d-%H%M%S")
        self.video_path = self.output_dir / f"dj-turbo-{ts}.video.mp4"
        self.audio_path = self.output_dir / f"dj-turbo-{ts}.audio.wav"
        self.final_path = self.output_dir / f"dj-turbo-{ts}.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self.writer = cv2.VideoWriter(str(self.video_path), fourcc, self.fps, (w, h))
        self.start_ts = time.time()
        audioResponses.start_recording()
        self.active = True
        print(f"REC started → {self.final_path}")

    def write_frame(self, frame):
        if self.active and self.writer is not None:
            self.writer.write(frame)

    def stop(self):
        if not self.active:
            return None
        self.active = False
        if self.writer is not None:
            self.writer.release()
            self.writer = None

        audio = audioResponses.stop_recording()
        sr = audioResponses.get_sample_rate()
        if audio is not None and sr:
            sf.write(str(self.audio_path), audio, sr)

        merged = self._merge_av()
        if merged:
            print(f"REC saved → {self.final_path}")
            return str(self.final_path)

        print(f"REC saved (video-only) → {self.video_path}")
        return str(self.video_path)

    def _merge_av(self) -> bool:
        if not self.audio_path or not self.audio_path.exists():
            return False
        if not self.video_path or not self.video_path.exists():
            return False
        try:
            subprocess.run([
                "ffmpeg", "-y",
                "-i", str(self.video_path),
                "-i", str(self.audio_path),
                "-c:v", "copy",
                "-c:a", "aac",
                "-shortest",
                str(self.final_path),
            ], check=True, capture_output=True)
            os.remove(self.video_path)
            os.remove(self.audio_path)
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            return False
