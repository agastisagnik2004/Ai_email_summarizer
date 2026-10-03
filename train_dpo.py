"""
Reinforcement learning from human feedback for the email summarizer, using DPO
(Direct Preference Optimization).

The app's Review tab shows two summaries of the same email, one from the original model and one
from the fine-tuned model, in random order. The one you tap the heart on is stored as "chosen"
and the other as "rejected" in feedback/preferences.json. This script trains the fine-tuned
model to prefer the chosen summaries, then replaces models/Qwen2.5-0.5B-Email-Summarizer.
Every earlier version is kept as models/Qwen2.5-0.5B-Email-Summarizer-v1, -v2, ...

The app runs this automatically; you can also run it by hand:
    python train_dpo.py            # train on votes not used in an earlier round (needs 30+)
    python train_dpo.py --all      # train on every vote collected so far
"""

import argparse
import gc
import random
import shutil
from pathlib import Path

import torch
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

from app import (BASE_MODEL, DEFAULT_MODEL, MIN_PREFS, ROOT, load_preferences, migrate_previous_version,
                 save_preferences, summary_messages, version_dir)

TMP_DIR = ROOT / "models" / "_rlhf_tmp"


def encode(tokenizer, pair, max_len):
    """Token ids for prompt + chosen and prompt + rejected, and where the answer starts."""
    prompt = tokenizer.apply_chat_template(summary_messages(pair["email"], pair["style"]),
                                           tokenize=False, add_generation_prompt=True)
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]

    def with_answer(answer):
        answer_ids = tokenizer(answer.strip() + "<|im_end|>\n", add_special_tokens=False)["input_ids"]
        return torch.tensor([(prompt_ids + answer_ids)[:max_len]])

    return with_answer(pair["chosen"]), with_answer(pair["rejected"]), len(prompt_ids)


def answer_logp(model, ids, prompt_len):
    """Total log-probability the model gives to the answer tokens (everything after the prompt)."""
    ids = ids.to(model.device)
    n = ids.shape[1] - prompt_len
    logits = model(input_ids=ids, logits_to_keep=n + 1).logits[0, :-1].float()
    return logits.log_softmax(-1).gather(-1, ids[0, prompt_len:, None]).sum()


def train_dpo(pairs, model_path, out_dir, epochs=3, lr=5e-5, beta=0.1, accum=4, max_len=1024, log=print, seed=42):
    """Train model_path on preference pairs with LoRA + DPO and save the merged model to out_dir."""
    random.seed(seed)
    torch.manual_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32

    log(f"Loading model for training ({len(pairs)} preferences)...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path, dtype=dtype).to(device)
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    model = get_peft_model(model, LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05, task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]))
    params = [p for p in model.parameters() if p.requires_grad]
    for p in params:
        p.data = p.data.float()  # keep the small trainable LoRA weights in full precision

    data = [encode(tokenizer, p, max_len) for p in pairs]
    data = [d for d in data if d[0].shape[1] > d[2] and d[1].shape[1] > d[2]]
    if not data:
        raise ValueError("No usable preference pairs (emails too long?).")

    # The reference model is the starting model: same weights with the LoRA adapter switched off.
    log("Scoring the summaries with the current model...")
    model.eval()
    with torch.no_grad(), model.disable_adapter(), \
            torch.autocast(device, dtype=torch.bfloat16, enabled=device == "cuda"):
        ref = [(answer_logp(model, c, n).item(), answer_logp(model, r, n).item()) for c, r, n in data]
    model.train()

    optimizer = torch.optim.AdamW(params, lr=lr)
    order = list(range(len(data)))
    stats = {}
    for epoch in range(1, epochs + 1):
        random.shuffle(order)
        total_loss, correct = 0.0, 0
        for step, i in enumerate(order, 1):
            chosen, rejected, n = data[i]
            ref_chosen, ref_rejected = ref[i]
            with torch.autocast(device, dtype=torch.bfloat16, enabled=device == "cuda"):
                pol_chosen = answer_logp(model, chosen, n)
                pol_rejected = answer_logp(model, rejected, n)
            # DPO: raise the chosen summary's probability relative to the rejected one,
            # measured against the reference model so the model can't drift too far.
            margin = beta * ((pol_chosen - ref_chosen) - (pol_rejected - ref_rejected))
            loss = -F.logsigmoid(margin)
            (loss / accum).backward()
            total_loss += loss.item()
            correct += margin.item() > 0
            if step % accum == 0 or step == len(order):
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                optimizer.step()
                optimizer.zero_grad()
        stats = {"loss": total_loss / len(order), "accuracy": correct / len(order), "pairs": len(data)}
        log(f"Epoch {epoch}/{epochs}: loss {stats['loss']:.3f}, prefers your choice {stats['accuracy']:.0%} of the time")

    log("Saving the updated model...")
    model.eval()
    merged = model.merge_and_unload()
    out_dir = Path(out_dir)
    if out_dir.exists():
        shutil.rmtree(out_dir)
    merged.to(torch.float16).save_pretrained(out_dir, safe_serialization=True)
    tokenizer.save_pretrained(out_dir)
    del model, merged
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()
    return stats


def install_model(new_dir, target, backup):
    """Replace target with new_dir, keeping the old version in the folder backup."""
    new_dir, target, backup = Path(new_dir), Path(target), Path(backup)
    if backup.exists():
        shutil.rmtree(backup)
    if target.exists():
        target.rename(backup)
    try:
        new_dir.rename(target)
    except Exception:
        if backup.exists() and not target.exists():
            backup.rename(target)
        raise


def next_round(prefs):
    return 1 + max((p.get("trained_round") or 0 for p in prefs), default=0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--all", action="store_true", help="train on every vote, not just new ones")
    parser.add_argument("--min", type=int, default=MIN_PREFS, help="minimum number of votes needed")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--beta", type=float, default=0.1)
    args = parser.parse_args()

    prefs = load_preferences()
    pairs = prefs if args.all else [p for p in prefs if not p.get("trained_round")]
    if len(pairs) < args.min:
        print(f"[!!] Only {len(pairs)} new preferences; need at least {args.min}. Collect more in the Review tab.")
        return

    migrate_previous_version()
    start = DEFAULT_MODEL if Path(DEFAULT_MODEL).exists() else BASE_MODEL
    round_no = next_round(prefs)
    stats = train_dpo(pairs, start, TMP_DIR, args.epochs, args.lr, args.beta, log=lambda m: print(f"[..] {m}"))
    install_model(TMP_DIR, DEFAULT_MODEL, version_dir(round_no))

    ids = {p["id"] for p in pairs}
    for p in prefs:
        if p["id"] in ids:
            p["trained_round"] = round_no
    save_preferences(prefs)
    print(f"[ok] Round {round_no} done on {stats['pairs']} preferences. Updated model saved to {DEFAULT_MODEL}")


if __name__ == "__main__":
    main()
