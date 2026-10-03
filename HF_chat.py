"""
Download a small LLM from Hugging Face and chat with it locally (no Ollama).
 
Setup:
    pip install torch transformers accelerate
 
Usage:
    python hf_chat.py
    python hf_chat.py --model Qwen/Qwen2.5-1.5B-Instruct
    python hf_chat.py --question "Explain JPEG compression in 2 lines"
 
Small instruct models (downloaded to ~/.cache/huggingface on first run):
    Qwen/Qwen2.5-0.5B-Instruct        (~1 GB)
    Qwen/Qwen2.5-1.5B-Instruct        (~3 GB)
    Qwen/Qwen3-0.6B
    HuggingFaceTB/SmolLM2-360M-Instruct
    microsoft/Phi-3-mini-4k-instruct  (~7.6 GB)
    meta-llama/Llama-3.2-1B-Instruct  (gated: accept licence + `huggingface-cli login`)
"""
 
import argparse
from threading import Thread
 
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
 
SYSTEM_PROMPT = "You are a helpful, concise assistant."
 
 
def load(model_id: str):
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    dtype = torch.float16 if device != "cpu" else torch.float32
    print(f"[..] Loading {model_id} on {device} (downloads on first run)...")
 
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype).to(device)
    model.eval()
    print("[ok] Model ready\n")
    return tokenizer, model, device
 
 
def stream(tokenizer, model, device, history, max_new_tokens=512, temperature=0.7):
    """Yield the model's reply piece by piece as it is generated."""
    prompt = tokenizer.apply_chat_template(history, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    kwargs = dict(**inputs, streamer=streamer, max_new_tokens=max_new_tokens)
    if temperature > 0:
        kwargs.update(do_sample=True, temperature=temperature, top_p=0.9)
    else:
        kwargs.update(do_sample=False)
    Thread(target=model.generate, kwargs=kwargs).start()
    yield from streamer


def ask(tokenizer, model, device, history, max_new_tokens=512) -> str:
    answer = ""
    for piece in stream(tokenizer, model, device, history, max_new_tokens):
        print(piece, end="", flush=True)
        answer += piece
    print()
    return answer.strip()
 
 
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--question")
    parser.add_argument("--max-tokens", type=int, default=512)
    args = parser.parse_args()
 
    tokenizer, model, device = load(args.model)
    history = [{"role": "system", "content": SYSTEM_PROMPT}]
 
    if args.question:
        history.append({"role": "user", "content": args.question})
        ask(tokenizer, model, device, history, args.max_tokens)
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
        history.append({"role": "assistant", "content": ask(tokenizer, model, device, history, args.max_tokens)})
 
 
if __name__ == "__main__":
    main()