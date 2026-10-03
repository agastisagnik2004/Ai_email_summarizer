"""
Train an email-REPLY model: the original Qwen2.5-0.5B-Instruct + SFT + DPO on email_dpo.jsonl.

email_dpo.jsonl has one JSON object per line:
    {"prompt": "...Email:\\n<email>", "chosen": "...Response: <good reply>", "rejected": "...Response: <bad reply>"}

Two stages, both with LoRA (a small trainable add-on), merged into a standalone model at the end:
    1. SFT  - learn the reply format from the chosen replies (DPO alone teaches format poorly with ~100 pairs)
    2. DPO  - learn to prefer the chosen reply over the rejected one, relative to the SFT model,
              plus a likelihood term on the chosen reply (RPO) so the model keeps ending replies cleanly

The model sees the same prompt here as in the app (assistant.reply_messages), and is saved to
models/Qwen2.5-0.5B-Email-Reply. The summarizer (Custom-vN) is not changed.

Usage:
    python train_reply_dpo.py
    python train_reply_dpo.py --data email_dpo.jsonl --sft-epochs 2 --dpo-epochs 2 --nll-weight 1.0
"""

import argparse
import json
import random
import re
import shutil
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

from assistant import reply_messages

ROOT = Path(__file__).parent
LORA = dict(r=16, lora_alpha=32, lora_dropout=0.05, task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])


# ------------------------------------------------------------------ data
def extract_reply(text):
    """The reply after 'Response:' / 'Suggested Response:' in an analysis, as greeting / body / sign-off."""
    m = re.search(r"(?:Suggested )?Response:\s*(.*)$", text, re.S)
    if not m:
        return None
    reply = m.group(1).strip()
    if "\n" not in reply:  # single-line replies: put greeting and sign-off on their own lines, like real emails
        reply = re.sub(r"^((?:hi|hello|dear)\b[^,\n]{0,40},)\s*", r"\1\n\n", reply, flags=re.I)
        reply = re.sub(r"\s*\b((?:best regards|kind regards|regards|best|thanks|sincerely),)\s*([^,\n]{1,40})$",
                       r"\n\n\1\n\2", reply, flags=re.I)
    return reply


def load_pairs(path):
    pairs = []
    for i, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        email = row["prompt"].split("Email:", 1)[-1].strip()
        chosen, rejected = extract_reply(row["chosen"]), extract_reply(row["rejected"])
        if not email or not chosen or not rejected or chosen == rejected:
            print(f"[!!] line {i}: skipped (missing email or response)")
            continue
        pairs.append({"email": email, "chosen": chosen, "rejected": rejected})
    return pairs


def encode(tokenizer, email, answer, max_len):
    """(ids, prompt_len) for chat prompt + answer; the loss is computed on the answer only."""
    prompt = tokenizer.apply_chat_template(reply_messages(email), tokenize=False, add_generation_prompt=True)
    p = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    a = tokenizer(answer.strip() + "<|im_end|>\n", add_special_tokens=False)["input_ids"]
    return torch.tensor([(p + a)[:max_len]]), len(p)


def answer_logp(model, ids, prompt_len, mean=False):
    """Sum (or mean) log-probability of the answer tokens."""
    ids = ids.to(model.device)
    n = ids.shape[1] - prompt_len
    logits = model(input_ids=ids, logits_to_keep=n + 1).logits[0, :-1].float()
    lp = logits.log_softmax(-1).gather(-1, ids[0, prompt_len:, None]).squeeze(-1)
    return lp.mean() if mean else lp.sum()


def add_lora(model):
    model = get_peft_model(model, LoraConfig(**LORA))
    params = [p for p in model.parameters() if p.requires_grad]
    for p in params:
        p.data = p.data.float()  # small trainable weights in full precision
    return model, params


@torch.no_grad()
def generate(model, tokenizer, email, max_new_tokens=220):
    model.eval()
    prompt = tokenizer.apply_chat_template(reply_messages(email), tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=model.device.type == "cuda"):
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False, repetition_penalty=1.1)
    return tokenizer.decode(out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()


# ------------------------------------------------------------------ training stages
def run_sft(model, data, epochs, lr, accum, log):
    """Stage 1: maximise the likelihood of the chosen replies."""
    model, params = add_lora(model)
    model.train()
    opt = torch.optim.AdamW(params, lr=lr)
    order = list(range(len(data)))
    for epoch in range(1, epochs + 1):
        random.shuffle(order)
        total = 0.0
        for step, i in enumerate(order, 1):
            ids, n = data[i]["chosen"]
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=model.device.type == "cuda"):
                loss = -answer_logp(model, ids, n, mean=True)
            (loss / accum).backward()
            total += loss.item()
            if step % accum == 0 or step == len(order):
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                opt.step()
                opt.zero_grad()
        log(f"SFT epoch {epoch}/{epochs}: loss {total / len(order):.3f}")
    return model.merge_and_unload()


def preference_stats(model, data, ref, beta):
    """Share of pairs where the model prefers chosen over rejected more than the reference does, and mean margin."""
    model.eval()
    correct, margins = 0, []
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=model.device.type == "cuda"):
        for d, (rc, rr) in zip(data, ref):
            pc = answer_logp(model, *d["chosen"]).item()
            pr = answer_logp(model, *d["rejected"]).item()
            m = beta * ((pc - rc) - (pr - rr))
            margins.append(m)
            correct += (pc - pr) > 0  # absolute: chosen more likely than rejected
    model.train()
    return correct / max(len(data), 1), sum(margins) / max(len(margins), 1)


def reference_logps(model, data):
    with torch.no_grad(), model.disable_adapter(), \
            torch.autocast("cuda", dtype=torch.bfloat16, enabled=model.device.type == "cuda"):
        return [(answer_logp(model, *d["chosen"]).item(), answer_logp(model, *d["rejected"]).item()) for d in data]


def run_dpo(model, train, evals, epochs, lr, beta, accum, log, nll_weight=1.0):
    """Stage 2: DPO against the SFT model (the same weights with the new LoRA switched off).

    Plain DPO only cares that chosen beats rejected, so on ~100 pairs it can push the chosen reply's own
    probability down too, and the model forgets how to end a reply. Adding the chosen reply's average
    negative log-likelihood (nll_weight, as in RPO) keeps good replies, and their ending, likely.
    """
    model, params = add_lora(model)
    model.eval()
    ref_train, ref_eval = reference_logps(model, train), reference_logps(model, evals)
    acc0, _ = preference_stats(model, evals, ref_eval, beta)
    log(f"held-out: chosen reply preferred {acc0:.0%} before DPO")
    model.train()
    opt = torch.optim.AdamW(params, lr=lr)
    order = list(range(len(train)))
    stats = {}
    for epoch in range(1, epochs + 1):
        random.shuffle(order)
        total, correct = 0.0, 0
        for step, i in enumerate(order, 1):
            d, (rc, rr) = train[i], ref_train[i]
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=model.device.type == "cuda"):
                pc = answer_logp(model, *d["chosen"])
                pr = answer_logp(model, *d["rejected"])
            margin = beta * ((pc - rc) - (pr - rr))
            n_chosen = d["chosen"][0].shape[1] - d["chosen"][1]
            loss = -F.logsigmoid(margin) + nll_weight * (-pc / n_chosen)
            (loss / accum).backward()
            total += loss.item()
            correct += margin.item() > 0
            if step % accum == 0 or step == len(order):
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                opt.step()
                opt.zero_grad()
        acc, margin = preference_stats(model, evals, ref_eval, beta)
        stats = {"train_loss": total / len(order), "train_accuracy": correct / len(order),
                 "eval_accuracy": acc, "eval_margin": margin, "eval_accuracy_before": acc0}
        log(f"DPO epoch {epoch}/{epochs}: loss {stats['train_loss']:.3f}, train {stats['train_accuracy']:.0%}, "
            f"held-out chosen preferred {acc:.0%} (margin {margin:+.2f})")
    return model.merge_and_unload(), stats


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(ROOT / "email_dpo.jsonl"))
    ap.add_argument("--base", default=str(ROOT / "models" / "Qwen2.5-0.5B-Instruct"))
    ap.add_argument("--out", default=str(ROOT / "models" / "Qwen2.5-0.5B-Email-Reply"))
    ap.add_argument("--sft-epochs", type=int, default=2)
    ap.add_argument("--dpo-epochs", type=int, default=2)
    ap.add_argument("--sft-lr", type=float, default=1e-4)
    ap.add_argument("--dpo-lr", type=float, default=2e-5)
    ap.add_argument("--nll-weight", type=float, default=1.0, help="weight of the chosen-reply likelihood term (0 = plain DPO)")
    ap.add_argument("--beta", type=float, default=0.1)
    ap.add_argument("--accum", type=int, default=4, help="gradient accumulation steps (effective batch size)")
    ap.add_argument("--eval-frac", type=float, default=0.1)
    ap.add_argument("--max-len", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    log = lambda m: print(f"[..] {m}", flush=True)
    t0 = time.time()

    pairs = load_pairs(args.data)
    random.shuffle(pairs)
    n_eval = max(1, round(len(pairs) * args.eval_frac))
    eval_pairs, train_pairs = pairs[:n_eval], pairs[n_eval:]
    log(f"{len(pairs)} pairs from {args.data}: {len(train_pairs)} train / {len(eval_pairs)} held out")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(args.base)
    model = AutoModelForCausalLM.from_pretrained(args.base, dtype=torch.bfloat16 if device == "cuda" else torch.float32).to(device)
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()

    enc = lambda ps: [{"chosen": encode(tokenizer, p["email"], p["chosen"], args.max_len),
                       "rejected": encode(tokenizer, p["email"], p["rejected"], args.max_len)} for p in ps]
    train, evals = enc(train_pairs), enc(eval_pairs)
    samples = eval_pairs[:3]
    before = [generate(model, tokenizer, p["email"]) for p in samples]

    log("Stage 1: SFT on the chosen replies")
    model = run_sft(model, train, args.sft_epochs, args.sft_lr, args.accum, log)
    log("Stage 2: DPO (chosen vs rejected)")
    model, stats = run_dpo(model, train, evals, args.dpo_epochs, args.dpo_lr, args.beta, args.accum, log, args.nll_weight)

    after = [generate(model, tokenizer, p["email"]) for p in samples]
    out = Path(args.out)
    if out.exists():
        shutil.rmtree(out)
    model.to(torch.float16).save_pretrained(out, safe_serialization=True)
    tokenizer.save_pretrained(out)
    info = {"base": args.base, "data": args.data, "pairs": len(pairs), "train": len(train_pairs), "held_out": len(eval_pairs),
            "args": vars(args), "stats": stats, "minutes": round((time.time() - t0) / 60, 1)}
    (out / "training_info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    log(f"Saved the reply model to {out} ({info['minutes']} min)")

    for p, b, a in zip(samples, before, after):
        print(f"\n======== held-out email ========\n{p['email'][:500]}\n\n--- reference (chosen) reply ---\n{p['chosen']}"
              f"\n\n--- original model, before training ---\n{b}\n\n--- reply model, after SFT + DPO ---\n{a}")


if __name__ == "__main__":
    main()
