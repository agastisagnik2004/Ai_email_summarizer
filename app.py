"""
AI Email Summarizer backend: a local Hugging Face model plus the web API. It also serves the
frontend page, which lives in frontend/index.html. No Ollama or API key needed.

Setup:
    pip install torch transformers accelerate fastapi uvicorn peft kokoro soundfile

The default model is loaded from the local folder models/Qwen2.5-0.5B-Email-Summarizer, which is
Qwen2.5-0.5B-Instruct fine-tuned on email summaries by train.py (no internet needed).
Falls back to models/Qwen2.5-0.5B-Instruct if the fine-tuned model hasn't been made yet.

Two modes in the web interface:
    Single   - paste an email and get a summary from the fine-tuned model, streamed as it's written.
    Compare  - see two summaries (Option Alpha / Beta: original vs fine-tuned model, in random order)
               and tap the heart on the one you prefer. Votes are saved to feedback/preferences.json
                 (one vote per email + style). With 30 new votes you can train; at 40 training
                 starts automatically (train_dpo.py) and the fine-tuned model is updated.
               Every version is kept (Custom-v1, v2, ...); pick one with the radio buttons in the header.

Voice (voice.py, free open-source models that run locally):
    DICTATE / audio import  - Whisper large-v3-turbo turns speech into the source email text.
    LISTEN                  - Kokoro-82M reads the source email or any summary aloud.

Usage:
    python app.py                                      # web interface at http://127.0.0.1:7860
    python app.py --model Qwen/Qwen2.5-1.5B-Instruct   # use a bigger, more accurate model (downloads it)
    python app.py --cli                                # chat with the model in the terminal
    python app.py --question "Explain JPEG compression in 2 lines"
    python app.py --analyze email.txt                  # email assistant analysis in the terminal

Small instruct models (downloaded to ~/.cache/huggingface on first run):
    Qwen/Qwen2.5-0.5B-Instruct        (~1 GB)
    Qwen/Qwen2.5-1.5B-Instruct        (~3 GB)
    Qwen/Qwen3-0.6B
    HuggingFaceTB/SmolLM2-360M-Instruct
    microsoft/Phi-3-mini-4k-instruct  (~7.6 GB)
    meta-llama/Llama-3.2-1B-Instruct  (gated: accept licence + `huggingface-cli login`)
"""

import argparse
import asyncio
import gc
import hashlib
import json
import mimetypes
import queue
import random
import re
import threading
import traceback
import uuid
import webbrowser
from collections import OrderedDict
from datetime import datetime
from pathlib import Path

import torch
import uvicorn
import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from starlette.concurrency import run_in_threadpool
from pydantic import BaseModel
import assistant
from voice import DEFAULT_VOICE, STT_SAMPLE_RATE, VOICES, Voice
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    StoppingCriteria,
    StoppingCriteriaList,
    TextIteratorStreamer,
)

# ---------------------------------------------------------------- backend: model ----

ROOT = Path(__file__).parent
CHAT_SYSTEM_PROMPT = "You are a helpful, concise assistant."
BASE_MODEL = str(ROOT / "models" / "Qwen2.5-0.5B-Instruct")
DEFAULT_MODEL = str(ROOT / "models" / "Qwen2.5-0.5B-Email-Summarizer")  # made by train.py, updated by train_dpo.py
REPLY_MODEL = str(ROOT / "models" / "Qwen2.5-0.5B-Email-Reply")  # made by train_reply_dpo.py (suggested responses)


def load(model_id: str):
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    dtype = torch.float16 if device != "cpu" else torch.float32
    print(f"[..] Loading {model_id} on {device}...")

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype).to(device)
    model.eval()
    print("[ok] Model ready\n")
    return tokenizer, model, device


class StopOnEvent(StoppingCriteria):
    """Ends generation early once the event is set (e.g. the user pressed Stop)."""

    def __init__(self, event: threading.Event):
        self.event = event

    def __call__(self, input_ids, scores, **kwargs):
        return torch.full((input_ids.shape[0],), self.event.is_set(), dtype=torch.bool, device=input_ids.device)


class LoopGuard:
    """Detects when the model starts repeating itself instead of finishing.

    Text is passed on one line or sentence at a time. As soon as a line or sentence repeats one
    already written (e.g. "NSE Alert:" for the third time), the repeat is dropped and generation
    is stopped, so the reply ends where the real content ended instead of at max_new_tokens.
    """

    BOUNDARY = re.compile(r"\n|(?<=[.!?])\s")

    def __init__(self):
        self.seen, self.buffer, self.looping = set(), "", False

    def _repeats(self, segment):
        key = re.sub(r"[\W_]+", " ", segment).strip().lower()
        if len(key) < 8:  # ignore blank lines and very short fragments like "- None"
            return False
        if key in self.seen:
            return True
        self.seen.add(key)
        return False

    def feed(self, piece):
        """Add new text; return the part that is safe to show."""
        self.buffer += piece
        out = ""
        while (m := self.BOUNDARY.search(self.buffer)):
            segment, self.buffer = self.buffer[:m.end()], self.buffer[m.end():]
            if self._repeats(segment):
                self.looping, self.buffer = True, ""
                break
            out += segment
        return out

    def flush(self):
        rest, self.buffer = self.buffer, ""
        return "" if self.looping or self._repeats(rest) else rest


def stream(tokenizer, model, device, history, max_new_tokens=512, temperature=0.7, stop_event=None):
    """Yield the model's reply piece by piece, stopping early if it starts looping."""
    prompt = tokenizer.apply_chat_template(history, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    stop_event = stop_event or threading.Event()

    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    kwargs = dict(**inputs, streamer=streamer, max_new_tokens=max_new_tokens, repetition_penalty=1.1,
                  stopping_criteria=StoppingCriteriaList([StopOnEvent(stop_event)]))
    if temperature > 0:
        kwargs.update(do_sample=True, temperature=temperature, top_p=0.9)
    else:
        kwargs.update(do_sample=False)
    threading.Thread(target=model.generate, kwargs=kwargs).start()

    guard = LoopGuard()
    try:
        for piece in streamer:
            text = guard.feed(piece)
            if text:
                yield text
            if guard.looping:
                print("[..] Model started repeating itself; stopped generation early.")
                break
        if tail := guard.flush():
            yield tail
    finally:
        stop_event.set()  # end generation on loop, on finish, or if the caller stops reading


def ask(tokenizer, model, device, history, max_new_tokens=512) -> str:
    answer = ""
    for piece in stream(tokenizer, model, device, history, max_new_tokens):
        print(piece, end="", flush=True)
        answer += piece
    print()
    return answer.strip()


def run_cli(tokenizer, model, device, question=None, max_tokens=512):
    history = [{"role": "system", "content": CHAT_SYSTEM_PROMPT}]

    if question:
        history.append({"role": "user", "content": question})
        ask(tokenizer, model, device, history, max_tokens)
        return

    print("Type 'exit' to quit, 'reset' to clear history.\n")
    while True:
        try:
            q = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not q:
            continue
        if q.lower() in {"exit", "quit"}:
            break
        if q.lower() == "reset":
            history = history[:1]
            print("[history cleared]")
            continue

        history.append({"role": "user", "content": q})
        print("AI: ", end="")
        history.append({"role": "assistant", "content": ask(tokenizer, model, device, history, max_tokens)})


# ------------------------------------------------------- backend: summarization ----

SUMMARY_SYSTEM_PROMPT = (
    "You are an assistant that summarizes emails. Use only information stated in the email. "
    "Never invent reasons, names, dates or details that are not in the text."
)

STYLES = {
    "Short summary": "Summarize this email in 2-3 sentences.",
    "Bullet points": "Summarize this email as a short list of bullet points covering the key information.",
    "Action items": (
        "List the action items, deadlines and requests in this email as bullet points. "
        "If there are none, say 'No action items.'"
    ),
    "One line (TL;DR)": "Summarize this email in a single sentence.",
}


def summary_messages(email, style):
    return [
        {"role": "system", "content": SUMMARY_SYSTEM_PROMPT},
        {"role": "user", "content": f"{STYLES[style]}\n\nEmail:\n\"\"\"\n{email}\n\"\"\""},
    ]


def summarize_stream(tokenizer, model, device, email, style, max_tokens=256, temperature=0.2, stop_event=None):
    """Yield pieces of the summary for an email as they are generated."""
    yield from stream(tokenizer, model, device, summary_messages(email, style), int(max_tokens), temperature, stop_event)


# ------------------------------------------------ backend: human feedback (RLHF) ----

FEEDBACK_FILE = ROOT / "feedback" / "preferences.json"
MIN_PREFS = 30         # new votes needed before training can start
AUTO_TRAIN_PREFS = 40  # new votes at which training starts on its own


def normalize(text):
    return " ".join(text.split())


def pref_key(email, style):
    """Same email (ignoring spacing) + same style = duplicate vote."""
    return hashlib.sha1(f"{style}\n{normalize(email)}".encode("utf-8")).hexdigest()[:16]


def load_preferences():
    if not FEEDBACK_FILE.exists():
        return []
    return json.loads(FEEDBACK_FILE.read_text(encoding="utf-8"))


def save_preferences(prefs):
    FEEDBACK_FILE.parent.mkdir(parents=True, exist_ok=True)
    tmp = FEEDBACK_FILE.with_suffix(".tmp")
    tmp.write_text(json.dumps(prefs, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(FEEDBACK_FILE)


def trained_rounds(prefs=None):
    prefs = load_preferences() if prefs is None else prefs
    return max((p.get("trained_round") or 0 for p in prefs), default=0)


# Model versions: Custom-v1 is the model made by train.py; each training round adds one.
# The newest version always lives in DEFAULT_MODEL; older ones are kept as DEFAULT_MODEL-v1, -v2, ...
def version_dir(version):
    return Path(f"{DEFAULT_MODEL}-v{version}")


def model_versions():
    """{version number: folder}, newest first, for every version saved on disk."""
    latest = trained_rounds() + 1
    found = {latest: Path(DEFAULT_MODEL)} if Path(DEFAULT_MODEL).exists() else {}
    for folder in Path(DEFAULT_MODEL).parent.glob(Path(DEFAULT_MODEL).name + "-v*"):
        suffix = folder.name.rsplit("-v", 1)[1]
        if suffix.isdigit() and int(suffix) < latest and folder.is_dir():
            found[int(suffix)] = folder
    return dict(sorted(found.items(), reverse=True))


def migrate_previous_version():
    """Older builds kept a single backup named ...-previous; give it its version number."""
    previous = Path(f"{DEFAULT_MODEL}-previous")
    latest = trained_rounds() + 1
    if previous.is_dir() and latest > 1 and not version_dir(latest - 1).exists():
        previous.rename(version_dir(latest - 1))
        print(f"[ok] Kept the previous model as Custom-v{latest - 1}")


class ChildStop:
    """Stop signal for one generation inside a multi-step request: it stops when the request is stopped
    (Stop button / closed tab), but finishing this generation doesn't stop the rest of the request."""

    def __init__(self, parent):
        self.parent, self.own = parent, threading.Event()

    def is_set(self):
        return self.own.is_set() or (self.parent is not None and self.parent.is_set())

    def set(self):
        self.own.set()


class Engine:
    """Holds the models and the training state shared by all requests."""

    def __init__(self, model_path):
        self.model_path = model_path
        self.tokenizer, self.model, self.device = load(model_path)
        self.active_version = next((v for v, d in model_versions().items()
                                    if Path(d).resolve() == Path(model_path).resolve()), None)
        self.base = None  # original model, loaded the first time the Review tab is used
        self.reply = None  # reply model (train_reply_dpo.py), loaded the first time a suggested response is written
        self.voice = Voice()  # speech-to-text + text-to-speech, loaded the first time they are used
        self.gpu_lock = threading.Lock()   # one generation (or training run) at a time
        self.prefs_lock = threading.Lock()
        self.training = False
        self.status = {"state": "idle", "message": ""}

    def base_model(self):
        if self.base is None:
            _, self.base, _ = load(BASE_MODEL)
        return self.base

    def reply_model(self):
        """The trained reply model, or None if train_reply_dpo.py hasn't been run."""
        if self.reply is None and Path(REPLY_MODEL, "config.json").exists():
            _, self.reply, _ = load(REPLY_MODEL)
        return self.reply

    def summarize(self, model, email, style, max_tokens, temperature):
        return "".join(summarize_stream(self.tokenizer, model, self.device, email, style,
                                        max_tokens, temperature)).strip()

    def analyze(self, email, stop_event=None):
        """Yield (section, text) for the email assistant format. Caller must hold gpu_lock."""
        # Summary, Action Items: your fine-tuned model (the selected Custom-vN version).
        # Suggested Response: the reply model (train_reply_dpo.py), checked, with fallbacks.
        # Intent, Sentiment: the original model. Deadline and Urgency: rules.
        tuned, original, reply = self.model, self.base_model(), self.reply_model()

        def summarize(text):
            return "".join(summarize_stream(self.tokenizer, tuned, self.device, text, "Short summary",
                                            200, 0.0, ChildStop(stop_event)))

        def with_model(model):
            return lambda messages, max_tokens: "".join(
                stream(self.tokenizer, model, self.device, messages, max_tokens, 0.0, ChildStop(stop_event)))

        for section in assistant.analyze(email, summarize, with_model(original), with_model(tuned),
                                         with_model(reply) if reply is not None else None):
            if stop_event is not None and stop_event.is_set():
                return
            yield section

    def stats(self):
        prefs = load_preferences()
        new = [p for p in prefs if not p.get("trained_round")]
        wins = sum(p["chosen_source"] == "fine-tuned" for p in prefs)
        return {
            "total": len(prefs), "new": len(new), "min": MIN_PREFS, "auto": AUTO_TRAIN_PREFS,
            "rounds": max((p.get("trained_round") or 0 for p in prefs), default=0),
            "finetuned_win_rate": wins / len(prefs) if prefs else None,
            "training": self.training, "status": self.status,
            "active_version": self.active_version,
            "versions": [{"version": v, "label": f"Custom-v{v}", "latest": i == 0}
                         for i, v in enumerate(model_versions())],
        }

    def select_version(self, version):
        """Switch the model that writes summaries to another saved version."""
        versions = model_versions()
        if version not in versions:
            raise KeyError(version)
        if version == self.active_version:
            return
        with self.gpu_lock:
            self.model = None  # free the current one first; two copies may not fit next to the voice models
            gc.collect()
            torch.cuda.empty_cache()
            try:
                self.tokenizer, self.model, self.device = load(str(versions[version]))
                self.model_path, self.active_version = str(versions[version]), version
            except Exception:
                self.tokenizer, self.model, self.device = load(self.model_path)  # stay on the old version
                raise

    def start_training(self):
        if self.training:
            return False
        self.training = True
        self.status = {"state": "training", "message": "Starting training..."}
        threading.Thread(target=self._train_round, daemon=True).start()
        return True

    def _log(self, message):
        print(f"[rlhf] {message}")
        self.status = {"state": "training", "message": message}

    def _train_round(self):
        from train_dpo import TMP_DIR, install_model, next_round, train_dpo

        try:
            with self.prefs_lock:
                prefs = load_preferences()
                pairs = [p for p in prefs if not p.get("trained_round")]
                round_no = next_round(prefs)
            with self.gpu_lock:
                self.base = None  # free GPU memory for training
                self.reply = None
                self.voice.unload()
                gc.collect()
                torch.cuda.empty_cache()
                stats = train_dpo(pairs, self.model_path, TMP_DIR, log=self._log)
                self.model = None
                gc.collect()
                torch.cuda.empty_cache()
                # the current newest version (Custom-v{round_no}) is kept as its own folder
                install_model(TMP_DIR, DEFAULT_MODEL, version_dir(round_no))
                self.model_path, self.active_version = DEFAULT_MODEL, round_no + 1
                self._log("Loading the updated model...")
                self.tokenizer, self.model, self.device = load(DEFAULT_MODEL)

            ids = {p["id"] for p in pairs}
            with self.prefs_lock:
                prefs = load_preferences()
                for p in prefs:
                    if p["id"] in ids:
                        p["trained_round"] = round_no
                save_preferences(prefs)
            self.status = {"state": "done", "message": (
                f"Round {round_no} complete. The model learned from {stats['pairs']} of your preferences "
                f"and now agrees with your choice {stats['accuracy']:.0%} of the time.")}
            print(f"[rlhf] {self.status['message']}")
        except Exception as e:
            traceback.print_exc()
            if self.model is None:
                self.tokenizer, self.model, self.device = load(self.model_path)
            self.status = {"state": "error", "message": f"Training failed: {e}"}
        finally:
            self.training = False


# ------------------------------------------------------------ backend: web API ----

class SummarizeRequest(BaseModel):
    email: str
    style: str = "Short summary"
    max_tokens: int = 256
    temperature: float = 0.2


class SpeakRequest(BaseModel):
    text: str
    voice: str = DEFAULT_VOICE
    speed: float = 1.0


MAX_AUDIO_SECONDS = 15 * 60


class AnalyzeRequest(BaseModel):
    email: str


class VersionRequest(BaseModel):
    version: int


class VoteRequest(BaseModel):
    pair_id: str
    choice: int


def check_request(req: SummarizeRequest):
    email = req.email.strip()
    if not email:
        raise HTTPException(400, "The email is empty.")
    if req.style not in STYLES:
        raise HTTPException(400, f"Unknown style: {req.style}")
    return email, min(max(req.max_tokens, 16), 1024), min(max(req.temperature, 0.0), 1.5)


def create_app(engine: Engine):
    app = FastAPI(title="Email Summarizer")
    current = {"stop": None}
    pending = OrderedDict()  # pair_id -> comparison waiting for a vote

    def relay(work, stop):
        """Run work() (a generator) in a worker thread that holds the GPU lock, and stream its chunks.

        If the browser goes away mid-stream (tab closed, page refreshed), the relay is cancelled, which
        sets stop: the worker stops generating and always releases the GPU lock. Holding the lock inside
        the response generator itself could leave it locked forever after a disconnect.
        """
        chunks = queue.Queue()
        end = object()

        def run():
            try:
                with engine.gpu_lock:
                    current["stop"] = stop
                    for chunk in work():
                        chunks.put(chunk)
                        if stop.is_set():
                            break
            except Exception as e:  # shown to the caller instead of a silent cut-off
                traceback.print_exc()
                chunks.put(e)
            finally:
                stop.set()
                chunks.put(end)

        threading.Thread(target=run, daemon=True).start()

        async def stream_out():
            try:
                while True:
                    item = await asyncio.to_thread(chunks.get)
                    if item is end:
                        break
                    if isinstance(item, Exception):
                        raise item
                    yield item
            finally:
                stop.set()

        return stream_out()

    def not_while_training():
        if engine.training:
            raise HTTPException(503, "The model is training on your feedback. Try again in a minute.")

    @app.get("/", response_class=HTMLResponse)
    def index():
        return HTMLResponse(FRONTEND.read_text(encoding="utf-8"), headers={"Cache-Control": "no-cache"})

    @app.get("/api/info")
    def info():
        return {"styles": list(STYLES)}

    @app.post("/api/summarize")
    def summarize(req: SummarizeRequest):
        not_while_training()
        email, max_tokens, temperature = check_request(req)

        stop = threading.Event()

        def work():
            yield from summarize_stream(engine.tokenizer, engine.model, engine.device, email, req.style,
                                        max_tokens, temperature, ChildStop(stop))

        return StreamingResponse(relay(work, stop), media_type="text/plain; charset=utf-8")

    @app.post("/api/analyze")
    def analyze(req: AnalyzeRequest):
        """Email assistant. Streams one JSON object per line: {"section", "title", "text"}, then {"done", "text"}."""
        not_while_training()
        email = req.email.strip()
        if not email:
            raise HTTPException(400, "The email is empty.")
        titles = dict(assistant.SECTIONS)

        stop = threading.Event()

        def work():
            done = {}
            for key, text in engine.analyze(email, stop):
                done[key] = text
                yield json.dumps({"section": key, "title": titles[key], "text": text}) + "\n"
            if not stop.is_set():
                yield json.dumps({"done": True, "text": assistant.as_text(done)}) + "\n"

        return StreamingResponse(relay(work, stop), media_type="application/x-ndjson")

    @app.post("/api/stop")
    def stop():
        if current["stop"] is not None:
            current["stop"].set()
        return {"ok": True}

    @app.post("/api/compare")
    def compare(req: SummarizeRequest):
        not_while_training()
        email, max_tokens, temperature = check_request(req)
        if Path(engine.model_path).resolve() == Path(BASE_MODEL).resolve():
            raise HTTPException(400, "Review needs the fine-tuned model. Run train.py first.")

        with engine.gpu_lock:
            finetuned = engine.summarize(engine.model, email, req.style, max_tokens, temperature)
            original = engine.summarize(engine.base_model(), email, req.style, max_tokens, temperature)

        responses = [{"source": "fine-tuned", "text": finetuned}, {"source": "original", "text": original}]
        random.shuffle(responses)  # blind test: the page never learns which model wrote which
        key = pref_key(email, req.style)
        duplicate = any(p["id"] == key for p in load_preferences())
        identical = normalize(finetuned) == normalize(original)

        pair_id = uuid.uuid4().hex
        pending[pair_id] = {"id": key, "email": email, "style": req.style, "temperature": temperature,
                            "responses": responses, "locked": duplicate or identical}
        while len(pending) > 100:
            pending.popitem(last=False)
        return {"pair_id": pair_id, "responses": [r["text"] for r in responses],
                "duplicate": duplicate, "identical": identical}

    @app.post("/api/vote")
    def vote(req: VoteRequest):
        pair = pending.get(req.pair_id)
        if pair is None:
            raise HTTPException(404, "This comparison has expired. Generate the responses again.")
        if pair["locked"]:
            raise HTTPException(409, "This comparison can't be rated (duplicate or identical responses).")
        if req.choice not in (0, 1):
            raise HTTPException(400, "Choice must be 0 or 1.")

        chosen, rejected = pair["responses"][req.choice], pair["responses"][1 - req.choice]
        with engine.prefs_lock:
            prefs = load_preferences()
            if any(p["id"] == pair["id"] for p in prefs):
                raise HTTPException(409, "You've already rated this email in this style.")
            prefs.append({
                "id": pair["id"],
                "created_at": datetime.now().isoformat(timespec="seconds"),
                "style": pair["style"],
                "email": pair["email"],
                "chosen": chosen["text"],
                "rejected": rejected["text"],
                "chosen_source": chosen["source"],
                "rejected_source": rejected["source"],
                "temperature": pair["temperature"],
                "trained_round": None,
            })
            save_preferences(prefs)
        del pending[req.pair_id]

        stats = engine.stats()
        stats["auto_training"] = stats["new"] >= AUTO_TRAIN_PREFS and engine.start_training()
        if stats["auto_training"]:
            stats = {**engine.stats(), "auto_training": True}
        return stats

    @app.get("/api/voices")
    def voices():
        return {"voices": [{"id": k, "label": v} for k, v in VOICES.items()], "default": DEFAULT_VOICE}

    @app.post("/api/transcribe")
    async def transcribe(request: Request):
        """Body: mono float32 little-endian samples at 16 kHz (the page decodes and resamples the audio)."""
        not_while_training()
        body = await request.body()
        if len(body) % 4 or len(body) < 4 * STT_SAMPLE_RATE // 2:
            raise HTTPException(400, "The recording is empty or too short.")
        audio = np.frombuffer(body, dtype="<f4").astype(np.float32)
        if len(audio) > MAX_AUDIO_SECONDS * STT_SAMPLE_RATE:
            raise HTTPException(413, f"Audio is longer than {MAX_AUDIO_SECONDS // 60} minutes.")

        def run():
            with engine.gpu_lock:
                return engine.voice.transcribe(audio)
        text = await run_in_threadpool(run)
        return {"text": text, "seconds": round(len(audio) / STT_SAMPLE_RATE, 1)}

    @app.post("/api/speak")
    def speak(req: SpeakRequest):
        not_while_training()
        text = req.text.strip()
        if not text:
            raise HTTPException(400, "Nothing to read aloud.")
        if len(text) > 20000:
            raise HTTPException(413, "That text is too long to read aloud (20,000 characters max).")
        with engine.gpu_lock:
            wav = engine.voice.speak(text, req.voice, min(max(req.speed, 0.5), 2.0))
        return Response(wav, media_type="audio/wav")

    @app.post("/api/model")
    def select_model(req: VersionRequest):
        not_while_training()
        try:
            engine.select_version(req.version)
        except KeyError:
            raise HTTPException(404, f"Custom-v{req.version} isn't saved on this computer.")
        return engine.stats()

    @app.get("/api/feedback")
    def feedback():
        return engine.stats()

    @app.post("/api/train")
    def train():
        stats = engine.stats()
        if stats["new"] < MIN_PREFS:
            raise HTTPException(400, f"Collect at least {MIN_PREFS} new preferences first ({stats['new']} so far).")
        engine.start_training()
        return engine.stats()

    # Files the page loads: the loading animation and its player (served locally, no CDN).
    mimetypes.add_type("application/wasm", ".wasm")
    mimetypes.add_type("text/javascript", ".js")
    app.mount("/static", StaticFiles(directory=FRONTEND.parent), name="static")

    return app


# ------------------------------------------------------------ frontend: web page ----

FRONTEND = ROOT / "frontend" / "index.html"  # the page lives in its own file; edits show on refresh


# ------------------------------------------------------------------ entry point ----

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL, help="local folder or Hugging Face model id")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--no-browser", action="store_true", help="don't open the browser automatically")
    parser.add_argument("--cli", action="store_true", help="chat in the terminal instead of the web UI")
    parser.add_argument("--question", help="ask one question in the terminal and exit")
    parser.add_argument("--analyze", metavar="FILE", help="print the email assistant analysis of an email file ('-' = stdin)")
    parser.add_argument("--max-tokens", type=int, default=512, help="max reply length in terminal mode")
    args = parser.parse_args()

    migrate_previous_version()
    if args.model == DEFAULT_MODEL and not Path(DEFAULT_MODEL).exists():
        print("[!!] Fine-tuned model not found (run train.py); using the base model.")
        args.model = BASE_MODEL

    if args.analyze:
        import sys
        text = sys.stdin.read() if args.analyze == "-" else Path(args.analyze).read_text(encoding="utf-8", errors="replace")
        engine = Engine(args.model)
        titles = dict(assistant.SECTIONS)
        for key, value in engine.analyze(text.strip()):
            print(f"{titles[key]}:\n{value}\n", flush=True)
        return

    if args.cli or args.question:
        run_cli(*load(args.model), args.question, args.max_tokens)
        return

    engine = Engine(args.model)
    url = f"http://127.0.0.1:{args.port}"
    print(f"[ok] Open {url} in your browser (Ctrl+C to quit)")
    if not args.no_browser:
        threading.Timer(1.5, webbrowser.open, [url]).start()
    uvicorn.run(create_app(engine), host="127.0.0.1", port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
