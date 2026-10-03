"""
Local, free, open-source voice models for the email summarizer:

    Speech-to-text: Whisper large-v3-turbo (OpenAI, MIT license)    models/whisper-large-v3-turbo
    Text-to-speech: Kokoro-82M (hexgrad, Apache 2.0 license)        models/Kokoro-82M

Both run on this computer and are loaded the first time they are used.

Setup:
    pip install kokoro soundfile
    (if Windows blocks spaCy, which Kokoro uses for pronunciation, spacy_lite.py is used instead)
    (the model files are already in models/; to re-download them see README of each model on Hugging Face)

Quick test:
    python voice.py "Hello, this is a test."      # writes voice_test.wav
"""

import io
import re
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

ROOT = Path(__file__).parent
WHISPER_DIR = ROOT / "models" / "whisper-large-v3-turbo"
KOKORO_DIR = ROOT / "models" / "Kokoro-82M"
KOKORO_REPO = "hexgrad/Kokoro-82M"
TTS_SAMPLE_RATE = 24000
STT_SAMPLE_RATE = 16000

VOICES = {  # voice id -> label; the first letter is the accent (a = American, b = British)
    "af_heart": "Heart · US female",
    "af_bella": "Bella · US female",
    "am_michael": "Michael · US male",
    "bf_emma": "Emma · UK female",
    "bm_george": "George · UK male",
}
DEFAULT_VOICE = "af_heart"


def _import_kokoro():
    """Import Kokoro, using spacy_lite if real spaCy is blocked or missing on this PC."""
    try:
        import spacy  # noqa: F401
    except (ImportError, OSError) as e:
        print(f"[..] spaCy unavailable ({e.__class__.__name__}); using the built-in lightweight tokenizer")
        import spacy_lite
        spacy_lite.install()
    import kokoro
    return kokoro


def speakable(text):
    """Turn a markdown-ish summary into plain sentences that read well aloud."""
    text = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", text)  # [link text](url) -> link text
    dotted = lambda s: s.replace(".", " dot ")
    text = re.sub(r"\b([\w.+-]+)@([\w-]+(?:\.[\w-]+)+)", lambda m: f"{dotted(m[1])} at {dotted(m[2])}", text)
    text = re.sub(r"\bhttps?://(?:www\.)?([\w-]+(?:\.[\w-]+)+)\S*", lambda m: dotted(m[1]), text)
    text = re.sub(r"\bwww\.([\w-]+(?:\.[\w-]+)+)\S*", lambda m: dotted(m[1]), text)
    text = re.sub(r"(?:₹|\bRs\.?|\bINR)\s?(\d[\d,]*(?:\.\d+)?)", r"\1 rupees", text)
    lines = []
    for line in text.splitlines():
        line = re.sub(r"^\s*(?:[-*•]|\d+[.)])\s+", "", line)  # bullet and number markers
        line = re.sub(r"[*_`#>|]+", "", line).strip()
        if line:
            lines.append(line if line[-1] in ".!?:;" else line + ".")
    return "\n".join(lines)


class Voice:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.stt = None
        self.tts_model = None
        self.tts_pipelines = {}

    # ------------------------------------------------------------ speech-to-text
    def _load_stt(self):
        if self.stt is None:
            from transformers import pipeline
            print(f"[..] Loading Whisper speech-to-text on {self.device}...")
            self.stt = pipeline(
                "automatic-speech-recognition", model=str(WHISPER_DIR), device=self.device,
                dtype=torch.float16 if self.device == "cuda" else torch.float32,
            )
            print("[ok] Whisper ready")
        return self.stt

    def transcribe(self, audio: np.ndarray) -> str:
        """audio: mono float32 samples at 16 kHz. Any language; recordings over 30 s use Whisper's long-form mode."""
        stt = self._load_stt()
        # Trailing silence stops Whisper from dropping a short last sentence that ends right at the cut.
        audio = np.concatenate([audio, np.zeros(STT_SAMPLE_RATE, dtype=np.float32)])
        long_form = len(audio) > 30 * STT_SAMPLE_RATE
        result = stt({"raw": audio, "sampling_rate": STT_SAMPLE_RATE},
                     return_timestamps=long_form, generate_kwargs={"task": "transcribe"})
        return result["text"].strip()

    # ------------------------------------------------------------ text-to-speech
    def _load_tts(self, lang):
        kokoro = _import_kokoro()
        if self.tts_model is None:
            print(f"[..] Loading Kokoro text-to-speech on {self.device}...")
            self.tts_model = kokoro.KModel(repo_id=KOKORO_REPO, config=str(KOKORO_DIR / "config.json"),
                                           model=str(KOKORO_DIR / "kokoro-v1_0.pth")).to(self.device).eval()
            print("[ok] Kokoro ready")
        if lang not in self.tts_pipelines:
            self.tts_pipelines[lang] = kokoro.KPipeline(lang_code=lang, repo_id=KOKORO_REPO, model=self.tts_model)
        return self.tts_pipelines[lang]

    def speak(self, text: str, voice: str = DEFAULT_VOICE, speed: float = 1.0) -> bytes:
        """Return a WAV file of the text read aloud."""
        if voice not in VOICES:
            voice = DEFAULT_VOICE
        pipeline = self._load_tts(voice[0])
        chunks = []
        with torch.no_grad():
            for result in pipeline(speakable(text), voice=str(KOKORO_DIR / "voices" / f"{voice}.pt"), speed=speed):
                if result.audio is not None:
                    chunks.append(result.audio.cpu().numpy())
                    chunks.append(np.zeros(int(TTS_SAMPLE_RATE * 0.25), dtype=np.float32))  # short pause between lines
        audio = np.concatenate(chunks) if chunks else np.zeros(TTS_SAMPLE_RATE // 2, dtype=np.float32)
        buf = io.BytesIO()
        sf.write(buf, audio, TTS_SAMPLE_RATE, format="WAV", subtype="PCM_16")
        return buf.getvalue()

    def unload(self):
        """Free GPU memory (used before training)."""
        self.stt = None
        self.tts_model = None
        self.tts_pipelines = {}


if __name__ == "__main__":
    text = " ".join(sys.argv[1:]) or "Hello! This is your email summarizer speaking."
    Path("voice_test.wav").write_bytes(Voice().speak(text))
    print("[ok] Wrote voice_test.wav")
