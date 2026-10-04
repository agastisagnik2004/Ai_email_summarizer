# Embrify — Video Walkthrough Script

**Length:** about 19 minutes · **Format:** screen recording + voice-over (face cam optional)
**Audience:** everyone — Part A is for non-technical viewers, Part B goes under the hood.

How to read this script:

- **SHOW** — what is on screen
- **SAY** — what you say (read it naturally; you don't have to be word-perfect)
- **TIP** — recording advice for that moment

---

## Before you record — checklist

1. **Start the app** with `Start Embrify.bat` and wait for the browser to open.
2. **Warm up the models.** Run one Full analysis and press LISTEN once *before* recording.
   The first run loads the models onto the GPU and is slow; after that it takes a few seconds.
3. **Prepare the sample emails** from the [Appendix](#appendix-a--sample-emails-to-use) in a text editor, ready to copy.
4. Save one of them as `sample.eml` or `sample.txt` on the Desktop for the Import demo.
5. Open `docs/architecture.png` in a separate window (full screen, zoomable).
6. Browser: zoom **110–125 %**, hide bookmarks bar, close other tabs, turn on Do Not Disturb.
7. Record at **1920 × 1080**. Mic test: speak one sentence, check the level.
8. Have the GitHub page open: `github.com/agastisagnik2004/Ai_email_summarizer`.

---

## Chapter overview

| # | Chapter | Time | View |
|---|---|---|---|
| 0 | Cold open | 0:00 – 0:30 | Non-technical |
| 1 | The problem | 0:30 – 1:30 | Non-technical |
| 2 | Demo: the Full analysis | 1:30 – 5:00 | Non-technical |
| 3 | More ways to use it: styles, voice, import | 5:00 – 6:30 | Non-technical |
| 4 | Private by design | 6:30 – 7:15 | Non-technical |
| 5 | It learns from you: Compare, ♥ and versions | 7:15 – 9:00 | Both |
| 6 | Architecture | 9:00 – 11:15 | Technical |
| 7 | How the six sections are produced | 11:15 – 13:30 | Technical |
| 8 | Training the models | 13:30 – 16:00 | Technical |
| 9 | Engineering details | 16:00 – 17:30 | Technical |
| 10 | Honest limitations and what's next | 17:30 – 18:30 | Both |
| 11 | One-click install, GitHub and outro | 18:30 – 19:30 | Both |

---

# PART A — The non-technical view

## 0 · Cold open (0:00 – 0:30)

**SHOW:** The app with the long project email already pasted. Click **Analyze Email**. The robot loader plays,
then the six cards fill in one by one.

**SAY:**
> This is a nine-hundred-word project email with three deadlines and thirteen deliverables hidden inside it.
> Watch what happens when I click one button.
> …
> In about twenty seconds I have a summary, every action item, every deadline with its exact time, how urgent
> it is, and a reply I can send. And all of this ran on my own laptop — nothing went to the cloud.
> I'm Sagnik, and this is **Embrify**.

**TIP:** Speed up the waiting part 2× in editing, but keep the cards appearing at normal speed.

---

## 1 · The problem (0:30 – 1:30)

**SHOW:** The Embrify logo (header of the page), then a busy inbox screenshot or just your face cam.

**SAY:**
> Most of us spend hours a week reading emails just to answer three questions:
> *What do they want from me? By when? And how do I reply?*
> Long emails bury the important parts — a deadline in paragraph seven, a meeting time in the last line.
> Online AI tools can help, but that means pasting your work emails into someone else's server.
> So I built an assistant that reads the email for you, finds everything that matters, drafts a reply —
> and does it completely offline, on a normal laptop.

---

## 2 · Demo: the Full analysis (1:30 – 5:00)

**SHOW:** Click **CLEAR**, paste the *Project Testing* email (Appendix, email 1). Point out **Single** mode and
the **Full analysis** style are selected. Click **Analyze Email**.

**SAY:**
> Let me use a normal email from my manager, Rahul. I paste it here, keep "Full analysis" selected,
> and click Analyze.

**SHOW:** The loader, then the cards appearing. Walk through each card with the mouse.

**SAY — card by card:**

1. **Summary**
   > First, a short summary — the whole email in two or three sentences.
2. **Intent**
   > Then the intent: what kind of email this is. Here it's a *request for action* and it also involves a
   > *meeting*, so it gets both tags.
3. **Action items**
   > The action items — every task the email asks me to do, written as a clear to-do list.
4. **Deadline**
   > The deadlines, with the exact date and time on the left and what's due on the right.
   > October 12 for the testing, and the review meeting on October 13 at 3 PM.
5. **Sentiment / Urgency**
   > The tone of the email, and how urgent it is — and it tells me *why* it decided that,
   > so I never have to just trust a label.
6. **Suggested response**
   > And finally a reply I can send, confirming the work and the dates.
   > I can copy just the reply, or copy the whole analysis in one click.

**SHOW:** Click **COPY ALL**, paste it into Notepad to show the plain-text format.

**SAY:**
> Copy-all gives me the analysis as clean text — Summary, Intent, Action Items, Deadline, Urgency,
> Suggested Response — ready to paste into notes or a task tracker.

**SHOW:** Now paste the long *AI Document Processing Project* email (Appendix, email 2) and analyze it.
Scroll slowly through Action Items (25 tasks + 13 deliverables) and the three deadlines.

**SAY:**
> It also works on really long emails. This one has twenty-five separate requests, a numbered list of thirteen
> deliverables and three deadlines — one of them tomorrow, which is why urgency is *High*.
> Look at the reply: it confirms every deadline and the review meeting, one by one.

**TIP:** Zoom the browser to 125 % for this part so the text is readable on a phone.

---

## 3 · More ways to use it: styles, voice, import (5:00 – 6:30)

**SHOW:** Click the **Short summary**, **Bullet points**, **Action items** and **One line (TL;DR)** style buttons
in turn, generating one each. The summary streams in word by word with the glowing border.

**SAY:**
> If I only want a quick summary, I pick a style: short, bullet points, just the action items,
> or a one-line TL;DR. It writes the answer live, word by word.

**SHOW:** Click **LISTEN** under the email, then **LISTEN** on the suggested reply.

**SAY:**
> I can have the email or the reply read aloud — there are five voices in the Advanced settings.

**SHOW:** Click **CLEAR**, then **DICTATE**, speak a short email, press **STOP**. The text appears.

**SAY:**
> And I can dictate. I'll speak an email… and it's transcribed right here — again, on my laptop.

**SHOW:** Click **IMPORT FILE**, choose `sample.eml`.

**SAY:**
> I can import a saved email file, or even an audio recording, and it fills the box for me.

**TIP:** Allow microphone access *before* recording so the browser pop-up doesn't appear on camera.

---

## 4 · Private by design (6:30 – 7:15)

**SHOW:** Turn Wi-Fi off (show the taskbar icon), then run another analysis — it still works.

**SAY:**
> Here's the part I care about most. I've just turned off Wi-Fi… and it still works.
> Every model runs on my laptop's graphics card. Emails, votes and voice recordings never leave this PC.
> No account, no subscription, no API key.

**TIP:** Remember to turn Wi-Fi back on before Chapter 11.

---

## 5 · It learns from you: Compare, ♥ and versions (7:15 – 9:00)

**SHOW:** Switch to **COMPARE**. Paste an email, click **Generate Summary Options**. Two cards appear:
Option Alpha and Option Beta. Hover the ♥ buttons (rainbow border), click one → *Preference recorded*.

**SAY:**
> Embrify can also learn what *I* like. In Compare mode it writes two summaries side by side.
> I don't know which model wrote which — it's a blind test. I tap the heart on the one I prefer.

**SHOW:** The **Preference profile** bar at the bottom of the right panel.

**SAY:**
> Every heart is saved. After thirty votes I can press Train, and at forty it trains by itself.
> The model learns from my choices and becomes a new version.

**SHOW:** The header radio buttons **Custom-v2 LATEST** and **Custom-v1**. Click v1, then back to v2.

**SAY:**
> Every version is kept. Right now I'm on Custom-v2, trained on my votes. If I ever prefer the old one,
> I just switch back here.

---

# PART B — The technical view

## 6 · Architecture (9:00 – 11:15)

**SHOW:** `docs/architecture.png` full screen. Zoom into each numbered part as you talk about it.

**SAY:**
> Now let's go under the hood. The whole system fits on this one sheet, in five parts.

**SHOW:** Zoom to **01 The page you use**.
> **One** — the web page. It's plain HTML, CSS and JavaScript in a single file, with no build step.
> The loading animation, icons and fonts are bundled locally, so the page works offline too.

**SHOW:** Zoom to **02 The server**.
> **Two** — the server, `app.py`, built with FastAPI and uvicorn. It only listens on localhost,
> port 7860. Each feature is a small API: `/api/summarize` streams text, `/api/analyze` streams one JSON line
> per section, there are endpoints for voting, training, switching model versions, and for voice.

**SHOW:** Zoom to **03 The assistant**.
> **Three** — the assistant, `assistant.py`. This is the brain of the Full analysis, and I'll come back to it
> in a minute. Below it, `voice.py` handles speech.

**SHOW:** Zoom to **04 Models on disk**, then the GPU bar.
> **Four** — six local models, about 5.8 gigabytes on disk. Everything loaded together uses about
> 5.2 of my laptop's 6 gigabytes of GPU memory — an RTX 3050.

**SHOW:** Zoom to **05 How it keeps learning**.
> **Five** — the learning loop: a dataset trains the first summarizer, my heart votes train the next
> versions, and a separate dataset trains the reply model.

---

## 7 · How the six sections are produced (11:15 – 13:30)

**SHOW:** The diagram zoomed on part 03, or the "How it works" table in `README.md` on GitHub.

**SAY:**
> The models here are small — half a billion parameters, Qwen 2.5. A model that small can't reliably do six
> different jobs in one answer: it mixes sections up and invents details.
> So I split the analysis into six small jobs, and each job uses the most reliable tool for it.

| Section | SAY |
|---|---|
| **Summary** | "My fine-tuned summarizer — the Custom-v2 model." |
| **Intent** | "The original Qwen model picks a category; for long emails the subject line becomes the description." |
| **Action items** | "My fine-tuned model, *merged with* every request sentence in the email and any numbered list of deliverables, so nothing is missed." |
| **Deadline** | "No AI at all — rules. A deadline is a word like *by*, *due* or *scheduled for* followed by a date. That's why it never invents a date." |
| **Urgency** | "Also rules, with a points system: urgent words like *ASAP* or *priority*, a deadline today or tomorrow, lots of exclamation marks. It shows its reasons." |
| **Suggested response** | "My DPO-trained reply model writes first. If its draft fails the checks, the next model tries, and the last resort is a template built from the tasks and deadlines." |

**SAY:**
> The key idea is: *every model answer is checked by rules.* A reply that asks the sender questions back,
> describes the sender in the third person, or forgets a deadline is thrown away.
> For example, one draft said "your order has been shipped" — that was never in the email, so it was rejected.
> And to handle long emails, the rules read the *whole* email, while the models get a condensed version
> with the opening and every sentence that contains a request, a date or an urgent word.

---

## 8 · Training the models (13:30 – 16:00)

**SHOW:** `train.py` in the editor (scroll the docstring), then `train_dpo.py`, then `train_reply_dpo.py`.

**SAY — the summarizer:**
> The first summarizer comes from `train.py`. I fine-tuned Qwen 2.5-0.5B-Instruct on a dataset of 100 emails
> with reference summaries using **LoRA** — instead of retraining all 500 million weights, it trains a small
> add-on of about 9 million, which fits on a 6 GB GPU. Held-out loss went from 1.21 to 0.91.
> That became **Custom-v1**.

**SAY — learning from votes:**
> The heart votes from Compare mode are saved as *chosen* and *rejected* pairs in `preferences.json`.
> `train_dpo.py` trains on them with **DPO — Direct Preference Optimization.** It teaches the model to make the
> summary I preferred more likely than the one I rejected, compared to the previous version, so it can't
> drift too far. My first forty votes produced **Custom-v2**. Every old version is kept on disk.

**SAY — the reply model:**
> The reply model is trained with `train_reply_dpo.py` on my own dataset, `email_dpo.jsonl` — 100 emails, each
> with a good reply and a bad reply. It starts from the original Qwen model and trains in two stages:
> first plain fine-tuning on the good replies, so it learns the reply format, then DPO on good versus bad.
> My first run over-trained — the loss hit zero and the model started inventing lists after the reply.
> So I added a likelihood term, sometimes called RPO, which keeps the good replies — including how they end —
> likely. On ten emails it never saw, it now prefers the good reply 80 % of the time, and it trains in about
> eight minutes.

**TIP:** Show `models/` in File Explorer: `Email-Summarizer`, `Email-Summarizer-v1`, `Email-Reply`.

---

## 9 · Engineering details (16:00 – 17:30)

**SHOW:** `app.py` in the editor — scroll to `relay()`, `LoopGuard`, `select_version()`.

**SAY:**
> A few engineering problems I had to solve along the way:

1. > **Streaming.** Results stream to the page as they're generated — text for summaries, one JSON line per
   > section for the analysis — so you see progress instead of a spinner.
2. > **One GPU, many requests.** A single lock lets one job use the GPU at a time. The work runs in a separate
   > thread, so if you close the tab mid-answer, the job stops and the GPU is freed. Before that fix, closing
   > the tab could freeze the server.
3. > **Loop guard.** Small models sometimes get stuck repeating a line. The loop guard watches the output and
   > stops generation the moment a sentence repeats.
4. > **Voice.** The browser converts any audio to 16 kHz itself, then Whisper large-v3-turbo transcribes it.
   > Read-aloud uses Kokoro-82M. Windows Smart App Control blocked a library Kokoro needs, so I wrote a small
   > pure-Python replacement, `spacy_lite.py`, instead of disabling a security feature.
5. > **Frontend.** One HTML file, responsive for phones and tablets, with the Embrify brand colours,
   > glossy buttons and a Lottie loading animation — all served locally.

---

## 10 · Honest limitations and what's next (17:30 – 18:30)

**SHOW:** Face cam, or the app idle.

**SAY:**
> It isn't perfect, and I want to be honest about where it struggles.
> The models are tiny, so summaries can occasionally add a detail that isn't in the email — that's exactly why
> deadlines and urgency come from rules, and why every reply is checked.
> The reply dataset is very repetitive, so the reply model learned one style very well and others less.
> Next steps: more varied training data, a 1.5-billion-parameter model — which still fits my GPU —
> and connecting it directly to an inbox.

---

## 11 · One-click install, GitHub and outro (18:30 – 19:30)

**SHOW:** File Explorer → double-click **Start Embrify.bat**. The console shows the `[ok]` lines and the browser opens.

**SAY:**
> To run it yourself on Windows, there's one file: *Start Embrify.bat*. Double-click it — it finds or installs
> Python, sets up everything, downloads the models it needs and opens the app. The first run takes a few
> minutes; after that it starts in seconds.

**SHOW:** The GitHub repository page, scroll the README with the architecture diagram.

**SAY:**
> The full code, the training scripts, the datasets and this architecture diagram are on my GitHub —
> the link is in the description. Thanks for watching. If you have ideas, open an issue or reach out.
> This was Embrify.

**SHOW:** End card: Embrify logo + `github.com/agastisagnik2004/Ai_email_summarizer`.

---

## Appendix A — Sample emails to use

**Email 1 — Project Testing (Chapter 2)**

```
Subject: Project Testing and Documentation Update

Hi Sagnik,

Please complete the remaining testing for the document-processing module by October 12, 2026. The testing should cover OCR accuracy, document tampering detection, font analysis, and processing time for multi-page PDFs.

Please also update the project documentation with the latest test results and clearly mention any issues identified during testing.

If you encounter any blocker that may delay the testing, please inform me as soon as possible so that we can discuss the next steps.

We will review the testing results during the project meeting on October 13, 2026, at 3:00 PM.

Best regards,
Rahul
```

**Email 2 — Long project email (Chapter 2):** the *"Action Required: AI Document Processing Project – Final
Deliverables, Review Meeting, and Deployment Preparation"* email (from Rahul Sen to Amit).

**Email 3 — Complaint (good for Compare mode and the reply checks)**

```
From: Rahul Mehta <rahul.mehta@example.com>
Subject: Order #48213 still not delivered

Hello,

I ordered a standing desk (order #48213) on September 12 and was promised delivery within 7 days. It is now September 30 and it still hasn't arrived. This is really frustrating.

Please tell me where my order is and arrange delivery by this Friday, or I want a full refund.

Rahul
```

**Email 4 — Short request (good for the one-line TL;DR)**

```
From: Priya Sharma <priya@acme.com>
Subject: Budget numbers needed ASAP

Hi Arjun,

Could you please send me the final Q4 budget numbers by tomorrow 5pm? Also, can you confirm whether the marketing spend includes the October campaign?

Thanks!
Priya
```

---

## Appendix B — Plain-English glossary (for on-screen captions)

| Term | One-line explanation |
|---|---|
| **Model** | The AI "brain" — a file of learned numbers that turns text into text. |
| **0.5B parameters** | The model's size: half a billion learned numbers. Big chat AIs have hundreds of billions. |
| **Fine-tuning** | Extra training that teaches a general model one specific job. |
| **LoRA** | Fine-tuning a small add-on instead of the whole model — cheap enough for a laptop. |
| **DPO** | Training from "this answer is better than that one" pairs — like the ♥ votes. |
| **RPO / likelihood term** | An extra rule during DPO that stops the model forgetting how good answers look. |
| **Held-out data** | Examples the model never trained on, used to test it fairly. |
| **GPU** | The graphics card; it runs AI models much faster than the processor. |
| **Streaming** | Showing the answer as it's being written instead of waiting for the end. |
| **Rules** | Plain code with no AI — used where accuracy matters most (dates, urgency). |

---

## Appendix C — 60-second short version (Reels / Shorts / LinkedIn)

1. **(0–5 s)** SHOW: long email → click Analyze. SAY: "This email hides three deadlines. Watch."
2. **(5–20 s)** SHOW: the six cards appearing. SAY: "Summary, intent, every task, every deadline, urgency —
   and a reply ready to send."
3. **(20–30 s)** SHOW: Wi-Fi off, it still works. SAY: "All on my laptop. Nothing goes to the cloud."
4. **(30–45 s)** SHOW: Compare mode, tap ♥, version radio buttons. SAY: "And it learns my style from what
   I tap — every version is kept."
5. **(45–60 s)** SHOW: architecture diagram, then the GitHub page. SAY: "Small open models, fine-tuned with
   LoRA and DPO, checked by rules. It's called Embrify — the code is on my GitHub."

---

## Appendix D — Editing notes

- Add **chapter markers** using the times in the overview table (YouTube reads them from the description).
- Speed up model loading and long waits 2–4×; never speed up the moment results appear.
- Add captions for the glossary terms the first time they are spoken.
- Suggested description: one-paragraph summary, chapter list, GitHub link, models used
  (Qwen2.5-0.5B-Instruct, Whisper large-v3-turbo, Kokoro-82M).
