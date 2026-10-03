"""
Fine-tune the local Qwen2.5-0.5B-Instruct model on email/summary pairs (LoRA), then merge the
adapter into a new standalone model folder that app.py can load.

Setup:
    pip install torch transformers peft pandas pyarrow

Usage:
    python train.py
    python train.py --data train-00000-of-00001.parquet --epochs 2 --lr 1e-4

Each row of the dataset gives two training examples, using the same prompts as app.py:
    email -> summary                  ("Short summary" style)
    email -> maximum_brevity_summary  ("One line (TL;DR)" style)
"""

import argparse
import math
import random
from pathlib import Path

import pandas as pd
import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup

from app import BASE_MODEL, STYLES, SUMMARY_SYSTEM_PROMPT

ROOT = Path(__file__).parent
TARGETS = {"summary": "Short summary", "maximum_brevity_summary": "One line (TL;DR)"}


def make_examples(df):
    examples = []
    for _, row in df.iterrows():
        for column, style in TARGETS.items():
            target = row[column]
            if not isinstance(target, str) or not target.strip():
                continue
            examples.append({"email": row["email"].strip(), "style": style, "target": target.strip()})
    return examples


def messages_for(email, style):
    return [
        {"role": "system", "content": SUMMARY_SYSTEM_PROMPT},
        {"role": "user", "content": f"{STYLES[style]}\n\nEmail:\n\"\"\"\n{email}\n\"\"\""},
    ]


def tokenize(tokenizer, example, max_len):
    """Token ids for prompt + answer, with the loss only on the answer tokens."""
    prompt = tokenizer.apply_chat_template(messages_for(example["email"], example["style"]),
                                           tokenize=False, add_generation_prompt=True)
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    answer_ids = tokenizer(example["target"] + "<|im_end|>\n", add_special_tokens=False)["input_ids"]
    ids = (prompt_ids + answer_ids)[:max_len]
    labels = ([-100] * len(prompt_ids) + answer_ids)[:max_len]
    return ids, labels


def collate(batch, pad_id, device):
    width = max(len(ids) for ids, _ in batch)
    input_ids = torch.full((len(batch), width), pad_id)
    labels = torch.full((len(batch), width), -100)
    mask = torch.zeros((len(batch), width), dtype=torch.long)
    for i, (ids, lab) in enumerate(batch):
        input_ids[i, :len(ids)] = torch.tensor(ids)
        labels[i, :len(lab)] = torch.tensor(lab)
        mask[i, :len(ids)] = 1
    return input_ids.to(device), labels.to(device), mask.to(device)


@torch.no_grad()
def evaluate(model, data, pad_id, device, batch_size):
    model.eval()
    total_loss, total_tokens = 0.0, 0
    for i in range(0, len(data), batch_size):
        input_ids, labels, mask = collate(data[i:i + batch_size], pad_id, device)
        with torch.autocast(device, dtype=torch.bfloat16, enabled=device == "cuda"):
            out = model(input_ids=input_ids, attention_mask=mask, labels=labels)
        n = (labels[:, 1:] != -100).sum().item()
        total_loss += out.loss.item() * n
        total_tokens += n
    model.train()
    return total_loss / max(total_tokens, 1)


@torch.no_grad()
def generate(model, tokenizer, email, style, device, max_new_tokens=160):
    model.eval()
    prompt = tokenizer.apply_chat_template(messages_for(email, style), tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.autocast(device, dtype=torch.bfloat16, enabled=device == "cuda"):
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    model.train()
    return tokenizer.decode(out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=str(ROOT / "train-00000-of-00001.parquet"))
    parser.add_argument("--base", default=BASE_MODEL)
    parser.add_argument("--out", default=str(ROOT / "models" / "Qwen2.5-0.5B-Email-Summarizer"))
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-len", type=int, default=1024)
    parser.add_argument("--eval-frac", type=float, default=0.1, help="share of conversations held out")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ---- data: hold out whole conversations so replies to the same thread don't leak into eval
    df = pd.read_parquet(args.data)
    conv_ids = sorted(df["conversation_id"].unique())
    random.shuffle(conv_ids)
    eval_ids = set(conv_ids[:max(1, round(len(conv_ids) * args.eval_frac))])
    train_df, eval_df = df[~df["conversation_id"].isin(eval_ids)], df[df["conversation_id"].isin(eval_ids)]
    train_ex, eval_ex = make_examples(train_df), make_examples(eval_df)
    print(f"[..] {len(df)} emails -> {len(train_ex)} training / {len(eval_ex)} held-out examples")

    # ---- model + LoRA
    tokenizer = AutoTokenizer.from_pretrained(args.base)
    model = AutoModelForCausalLM.from_pretrained(args.base, dtype=torch.float32).to(device)
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    lora = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, task_type="CAUSAL_LM",
                      target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    train_data = [tokenize(tokenizer, ex, args.max_len) for ex in train_ex]
    eval_data = [tokenize(tokenizer, ex, args.max_len) for ex in eval_ex]
    sample = eval_ex[0]
    before = generate(model, tokenizer, sample["email"], sample["style"], device)
    base_loss = evaluate(model, eval_data, pad_id, device, args.batch_size)
    print(f"[..] held-out loss before training: {base_loss:.4f}")

    # ---- train
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=0.0)
    steps_per_epoch = math.ceil(len(train_data) / args.batch_size)
    total_steps = steps_per_epoch * args.epochs
    scheduler = get_cosine_schedule_with_warmup(optimizer, max(1, total_steps // 10), total_steps)
    model.train()

    best_loss, best_state = base_loss, None
    for epoch in range(1, args.epochs + 1):
        random.shuffle(train_data)
        running = 0.0
        for step in range(steps_per_epoch):
            input_ids, labels, mask = collate(train_data[step * args.batch_size:(step + 1) * args.batch_size], pad_id, device)
            with torch.autocast(device, dtype=torch.bfloat16, enabled=device == "cuda"):
                loss = model(input_ids=input_ids, attention_mask=mask, labels=labels).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            running += loss.item()
            if (step + 1) % 10 == 0 or step + 1 == steps_per_epoch:
                print(f"    epoch {epoch} step {step + 1}/{steps_per_epoch}  train loss {running / (step + 1):.4f}")
        eval_loss = evaluate(model, eval_data, pad_id, device, args.batch_size)
        print(f"[ok] epoch {epoch}: held-out loss {eval_loss:.4f}")
        if eval_loss < best_loss:
            best_loss = eval_loss
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items() if "lora_" in k}

    if best_state is None:
        print("[!!] Training did not improve held-out loss; no new model saved.")
        return
    model.load_state_dict(best_state, strict=False)
    after = generate(model, tokenizer, sample["email"], sample["style"], device)

    # ---- merge LoRA into the base weights and save a standalone model
    out = Path(args.out)
    merged = model.merge_and_unload()
    merged.to(torch.float16).save_pretrained(out, safe_serialization=True)
    tokenizer.save_pretrained(out)
    print(f"\n[ok] Saved new model to {out}")
    print(f"     held-out loss {base_loss:.4f} -> {best_loss:.4f}")

    print(f"\n--- held-out example ({sample['style']}) ---\nEMAIL:\n{sample['email'][:800]}")
    print(f"\nREFERENCE:\n{sample['target']}\n\nBEFORE:\n{before}\n\nAFTER:\n{after}")


if __name__ == "__main__":
    main()
