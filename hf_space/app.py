"""
Embrify on Hugging Face Spaces (ZeroGPU): the six-section email analysis from assistant.py.

Models (downloaded from the Hub at startup):
    Qwen/Qwen2.5-0.5B-Instruct      original model: intent, sentiment, reply fallback
    Sagnik345/embrify-summarizer    fine-tuned summarizer (Custom-vN): summary, action items
    Sagnik345/embrify-reply         reply model (SFT + DPO): suggested response
"""

import gradio as gr
import spaces
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import assistant

BASE = "Qwen/Qwen2.5-0.5B-Instruct"
TUNED = "Sagnik345/embrify-summarizer"
REPLY = "Sagnik345/embrify-reply"

# the summarizer was trained with exactly this prompt (app.py / train.py in the main repo)
SUMMARY_SYSTEM_PROMPT = (
    "You are an assistant that summarizes emails. Use only information stated in the email. "
    "Never invent reasons, names, dates or details that are not in the text."
)

tokenizer = AutoTokenizer.from_pretrained(BASE)
# ZeroGPU: models go on cuda at startup; the real GPU is attached only inside @spaces.GPU functions
models = {name: AutoModelForCausalLM.from_pretrained(path, dtype=torch.float16).to("cuda").eval()
          for name, path in [("base", BASE), ("tuned", TUNED), ("reply", REPLY)]}


def generate(model, messages, max_tokens):
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_tokens, do_sample=False, repetition_penalty=1.1)
    return tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()


def summarize(text):
    messages = [{"role": "system", "content": SUMMARY_SYSTEM_PROMPT},
                {"role": "user", "content": f"Summarize this email in 2-3 sentences.\n\nEmail:\n\"\"\"\n{text}\n\"\"\""}]
    return generate(models["tuned"], messages, 200)


@spaces.GPU(duration=90)
def analyze(email):
    email = (email or "").strip()
    if not email:
        yield "Paste an email first."
        return
    titles = dict(assistant.SECTIONS)
    out = ""
    for key, text in assistant.analyze(
            email, summarize,
            lambda m, n: generate(models["base"], m, n),
            lambda m, n: generate(models["tuned"], m, n),
            lambda m, n: generate(models["reply"], m, n)):
        body = f"```\n{text}\n```" if key == "response" else text.replace("\n", "  \n")
        out += f"### {titles[key]}\n{body}\n\n"
        yield out  # each section appears as soon as it's ready


EXAMPLE = """Subject: Project Testing and Documentation Update

Hi Sagnik,

Please complete the remaining testing for the document-processing module by October 12, 2026. The testing should cover OCR accuracy, document tampering detection, font analysis, and processing time for multi-page PDFs.

Please also update the project documentation with the latest test results and clearly mention any issues identified during testing.

If you encounter any blocker that may delay the testing, please inform me as soon as possible so that we can discuss the next steps.

We will review the testing results during the project meeting on October 13, 2026, at 3:00 PM.

Best regards,
Rahul"""

with gr.Blocks(title="Embrify") as demo:
    gr.Markdown(
        "# Embrify: AI email assistant\n"
        "Paste an email and get its **summary, intent, action items, deadlines, sentiment / urgency** and a "
        "**suggested reply**, written by small fine-tuned Qwen2.5-0.5B models.\n\n"
        "_Demo only: on this hosted version your email is processed on Hugging Face's servers, so please don't "
        "paste private emails. The [full app](https://github.com/agastisagnik2004/Ai_email_summarizer) runs "
        "entirely on your own PC._")
    with gr.Row():
        with gr.Column():
            email = gr.Textbox(lines=18, label="Email", placeholder="Paste an email here…")
            btn = gr.Button("Analyze Email", variant="primary")
            gr.Examples([[EXAMPLE]], inputs=email, label="Try an example")
        with gr.Column():
            result = gr.Markdown("The analysis appears here.")
    btn.click(analyze, email, result)

demo.launch()
