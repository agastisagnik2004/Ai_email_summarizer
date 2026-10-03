"""
Download any model the app needs that isn't in the models/ folder yet (first run on a new PC).
Run by "Start Embrify.bat"; safe to run again - models that are already there are skipped.

Public models (downloaded from Hugging Face if missing):
    models/Qwen2.5-0.5B-Instruct        original model (intent, tone, Compare mode)       ~1 GB
    models/whisper-large-v3-turbo       speech-to-text for DICTATE / audio import         ~1.6 GB
    models/Kokoro-82M                   text-to-speech for LISTEN                         ~330 MB

Your trained models can't be downloaded - they only exist on the PC where you trained them. Copy them
with the project folder to keep them. Without them the app still works:
    models/Qwen2.5-0.5B-Email-Summarizer   (Custom-vN)   -> the original model writes the summaries
    models/Qwen2.5-0.5B-Email-Reply        (reply model) -> replies come from the other models / template
"""

import sys
from pathlib import Path

from huggingface_hub import snapshot_download

ROOT = Path(__file__).parent
MODELS = ROOT / "models"
KOKORO_VOICES = ["af_heart", "af_bella", "am_michael", "bf_emma", "bm_george"]

PUBLIC = [
    # folder, Hugging Face repo, files to fetch (None = all), size, file that proves it's complete
    ("Qwen2.5-0.5B-Instruct", "Qwen/Qwen2.5-0.5B-Instruct", None, "~1 GB", "model.safetensors"),
    ("whisper-large-v3-turbo", "openai/whisper-large-v3-turbo",
     ["*.json", "*.txt", "model.safetensors"], "~1.6 GB", "model.safetensors"),
    ("Kokoro-82M", "hexgrad/Kokoro-82M",
     ["config.json", "kokoro-v1_0.pth", "LICENSE*", "README.md"] + [f"voices/{v}.pt" for v in KOKORO_VOICES],
     "~330 MB", "kokoro-v1_0.pth"),
]
TRAINED = [
    ("Qwen2.5-0.5B-Email-Summarizer", "your fine-tuned summarizer (Custom-vN)", "the original model will write the summaries"),
    ("Qwen2.5-0.5B-Email-Reply", "your DPO reply model", "replies will come from the other models or the template"),
]


def main():
    MODELS.mkdir(exist_ok=True)
    failed = False
    for folder, repo, patterns, size, marker in PUBLIC:
        target = MODELS / folder
        if (target / marker).exists():
            print(f"[ok] {folder}")
            continue
        print(f"[..] Downloading {repo} ({size}) - first run only, please wait...")
        try:
            snapshot_download(repo, local_dir=target, allow_patterns=patterns)
            print(f"[ok] {folder} downloaded")
        except Exception as e:
            failed = True
            print(f"[!!] Could not download {repo}: {e}\n     Check the internet connection and run Start Embrify.bat again.")

    for folder, what, fallback in TRAINED:
        if (MODELS / folder / "config.json").exists():
            print(f"[ok] {folder} ({what})")
        else:
            print(f"[--] {folder} not found: {what} isn't on this PC, so {fallback}.\n"
                  f"     To use it, copy models\\{folder} from the PC where you trained it.")

    # the original model is required; voice models only affect DICTATE / LISTEN
    if not (MODELS / "Qwen2.5-0.5B-Instruct" / "model.safetensors").exists():
        sys.exit(1)
    if failed:
        print("[!!] Some voice models are missing; DICTATE / LISTEN won't work until they are downloaded.")


if __name__ == "__main__":
    main()
