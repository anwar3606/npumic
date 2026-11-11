"""Voice isolation: removing other people's voices after noise removal.

A Selector (without latency padding, so "off" adds no delay) chooses between
off and VoxLock. VoxLock needs a sample of the user's voice; it is
recorded here from this step's input (after noise removal and level control):
RECORD_SECONDS of speech, of which the loudest 3 s are used. One sample is
kept per microphone in data/voices/; a microphone without its own sample uses
the most recent one.
"""

import json
import re
import threading
import time

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

from npumic.engine import SAMPLE_RATE, Selector, Stage
from npumic.runtime import DATA_DIR
from npumic.stages.rate16k import MODEL_RATE, RATIO
from npumic.stages.voxlock import Voice, loudest

RECORD_SECONDS = 10
MIN_SAMPLE_DBFS = -50.0
VOICES_DIR = DATA_DIR / "voices"


def _slug(mic_name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", mic_name)[:120]


class VoiceIsolation(Stage):
    name = "isolation"

    def __init__(self, selector: Selector, voice: Voice, voice_options: set[str], voices_dir=VOICES_DIR,
                 on_recorded=lambda error: None):
        """`voice_options` are the selector options that need the voice sample. `on_recorded` is
        called from a background thread when a recording ends, with an error message or None."""
        self.selector = selector
        self.on_recorded = on_recorded
        self.voice = voice
        self.voice_options = voice_options
        self.voices_dir = voices_dir
        self.voice_mic: str | None = None  # label of the microphone the sample was recorded with
        self.voice_recorded: str | None = None
        self.voice_for_this_mic = False
        self.record_error: str | None = None
        self._mic: tuple[str, str] | None = None
        self._recording: list[np.ndarray] | None = None
        self._recorded = 0
        self._processing = False

    @property
    def latency(self) -> int:
        return self.selector.latency

    def reset(self) -> None:
        self.selector.reset()

    def process_block(self, block: np.ndarray) -> np.ndarray:
        recording = self._recording
        if recording is not None and self._recorded < RECORD_SECONDS * SAMPLE_RATE:
            recording.append(block.copy())
            self._recorded += len(block)
        return self.selector.process(block)

    # ---- voice sample ------------------------------------------------------------

    @property
    def record_progress(self) -> float | None:
        """Fraction recorded while recording, 1.0 while the sample is being prepared, else None."""
        if self._processing:
            return 1.0
        if self._recording is None:
            return None
        return min(1.0, self._recorded / (RECORD_SECONDS * SAMPLE_RATE))

    def record(self) -> None:
        """Records a new voice sample in the background for the current microphone."""
        if self._recording is not None or self._processing:
            raise ValueError("Already recording")
        if self._mic is None:
            raise ValueError("No microphone")
        self.record_error = None
        self._recorded = 0
        self._recording = []
        threading.Thread(target=self._finish_recording, args=(self._mic,), name="npumic-voice", daemon=True).start()

    def _finish_recording(self, mic: tuple[str, str]) -> None:
        while self._recorded < RECORD_SECONDS * SAMPLE_RATE:
            time.sleep(0.05)
        self._processing = True
        audio = np.concatenate(self._recording)
        self._recording = None
        try:
            sample = loudest(resample_poly(audio, 1, RATIO).astype(np.float32))
            level = 10 * np.log10(np.mean(sample.astype(np.float64) ** 2) + 1e-20)
            if level < MIN_SAMPLE_DBFS:
                raise ValueError("Did not hear your voice; check the microphone and record again")
            self.voice.set(sample)
            self._save(mic, sample)
            self._show(mic[1], time.strftime("%Y-%m-%d %H:%M"), mic == self._mic)
        except Exception as exc:  # reported on the control page
            self.record_error = str(exc)
        finally:
            self._processing = False
        self.on_recorded(self.record_error)

    def _save(self, mic: tuple[str, str], sample: np.ndarray) -> None:
        self.voices_dir.mkdir(parents=True, exist_ok=True)
        path = self.voices_dir / f"{_slug(mic[0])}.wav"
        sf.write(path, sample, MODEL_RATE)
        path.with_suffix(".json").write_text(json.dumps({"mic": mic[1], "recorded": time.strftime("%Y-%m-%d %H:%M")}))

    def _show(self, mic_label: str, recorded: str, this_mic: bool) -> None:
        self.voice_mic, self.voice_recorded, self.voice_for_this_mic = mic_label, recorded, this_mic

    def use_mic(self, name: str, label: str) -> None:
        """Switches to the microphone's own voice sample, or keeps the current one when it has none."""
        self._mic = (name, label)
        path = self.voices_dir / f"{_slug(name)}.wav"
        if not path.exists():
            if self.voice.ready:
                self.voice_for_this_mic = False
                return
            saved = sorted(self.voices_dir.glob("*.wav"), key=lambda p: p.stat().st_mtime) if self.voices_dir.exists() else []
            if not saved:
                return
            path = saved[-1]
        sample, rate = sf.read(path, dtype="float32")
        if rate != MODEL_RATE:
            return
        self.voice.set(sample)
        meta = path.with_suffix(".json")
        info = json.loads(meta.read_text()) if meta.exists() else {}
        self._show(info.get("mic", path.stem), info.get("recorded", ""), path.stem == _slug(name))
