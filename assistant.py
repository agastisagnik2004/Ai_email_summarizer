"""
AI Email Assistant: turns one email into a structured analysis.

    Summary:              written by the fine-tuned summarizer model
    Intent:               what the sender wants (category tags + a short description)
    Action Items:         tasks for the recipient (fine-tuned model + the email's own requests and deliverables)
    Deadline:             every date/time the email ties to a task or a meeting
    Sentiment / Urgency:  the sender's tone, plus how urgent the email is and why
    Suggested Response:   a draft reply that confirms the requests and deadlines (reply model from
                          train_reply_dpo.py when trained; checked, with fallbacks)

Each section uses its own small, focused prompt, because a 0.5B model follows one simple instruction
much more reliably than six at once. Deadlines, urgency, requests and deliverables are found with rules
on the WHOLE email (so nothing at the end is missed), and every model answer is checked and replaced
by a rule-based answer when it fails the check, so the output stays grounded in the email.

analyze() yields (section, text) pairs as each one finishes, so the page can show them one by one.
"""

import re

SECTIONS = [
    ("summary", "Summary"),
    ("intent", "Intent"),
    ("actions", "Action Items"),
    ("deadline", "Deadline"),
    ("sentiment", "Sentiment / Urgency"),
    ("response", "Suggested Response"),
]

INTENTS = [
    "Request for action", "Question", "Information / update", "Meeting or scheduling", "Complaint",
    "Approval request", "Follow-up", "Feedback", "Thank you", "Sales or marketing", "Notification or alert",
]
SENTIMENTS = ["Positive", "Neutral", "Concerned", "Frustrated", "Angry", "Excited", "Appreciative"]

SYSTEM = ("You are an assistant that reads emails for a busy professional. Use only information stated in "
          "the email. Never invent names, dates, numbers or commitments.")
FOCUS_CHARS = 6000   # how much text the 0.5B model sees; long emails are condensed to this (rules still read everything)
MAX_ACTIONS = 25   # long project emails can ask for many things; list them all rather than silently dropping some
DEADLINE_SEP = " — "  # "<when> — <task>" (not ':', because times like 5:00 PM contain one)

# ------------------------------------------------------------------ date / deadline rules
_DAY = r"(?:mon|tues|wednes|thurs|fri|satur|sun)day"
_MONTH = r"(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|june?|july?|aug(?:ust)?|sept?(?:ember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\.?"
_DATE = (
    rf"(?:(?:next |this |coming )?{_DAY}(?:\s+(?:morning|afternoon|evening|night|noon|eod))?"
    rf"|{_MONTH}\s+\d{{1,2}}(?:st|nd|rd|th)?(?:,?\s+\d{{4}})?"
    rf"|\d{{1,2}}(?:st|nd|rd|th)?\s+(?:of\s+)?{_MONTH}(?:,?\s+\d{{4}})?"
    r"|\d{4}-\d{2}-\d{2}|\d{1,2}[/-]\d{1,2}(?:[/-]\d{2,4})?|\d{1,2}\.\d{1,2}\.\d{2,4}"
    r"|the \d{1,2}(?:st|nd|rd|th)"  # "before the 20th"
    r"|(?:\d+|one|two|three|four|five|seven|ten)\s+(?:business\s+|working\s+)?(?:hours?|days?|weeks?)"  # "within 3 days"
    r"|today|tonight|tomorrow|end of (?:the |this |next )?(?:day|week|month)|(?:this|next) (?:week|month)|eod|eow|cob"
    r"|\d{1,2}(?::\d{2})?\s?(?:am|pm)|noon|midnight)"
)
# a full "when": up to three date/time parts, e.g. "5:00 PM on October 7, 2026" or "4:00 PM tomorrow, October 2"
_WHEN = rf"{_DATE}(?:,?\s+(?:on\s+|at\s+)?{_DATE}){{0,2}}"
DATE_RE = re.compile(rf"\b{_DATE}\b", re.I)
# a deadline = a deadline word followed (within a few words) by a date: "by Tuesday noon", "scheduled for Oct 8"
DEADLINE_RE = re.compile(
    rf"\b(?:by|before|until|no later than|due(?: on| by)?|deadline(?: is)?|latest by|scheduled (?:for|on)|within"
    rf"|last (?:date|day)\b[^.]{{0,60}}?\bis|(?:closes?|ends?|expires?) on|on|at)\b"
    rf"[\s*_,:\-–—]+(?:[\w-]+[\s*_,]+){{0,3}}?({_WHEN})\b", re.I)
STRONG_CUE = re.compile(r"\b(by|before|until|no later than|due|deadline|latest|last date|within|closes?|expires?|scheduled"
                        r"|ready|submit|send|meeting|call|review|demo|presentation|interview)\b", re.I)
DURATION = re.compile(r"(?:\d+|one|two|three|four|five|seven|ten)\s+(?:business\s+|working\s+)?(?:hours?|days?|weeks?)", re.I)
URGENT_WORDS = re.compile(r"\b(urgent(?:ly)?|asap|immediately|right away|critical|emergency|time[- ]sensitive|time[- ]critical"
                          r"|high priority|as a priority|top priority|priority|as soon as possible|at the earliest|action required)\b", re.I)
SOON = re.compile(r"\b(today|tonight|eod|cob|end of (?:the )?day|tomorrow|within 24 hours|this morning|this afternoon)\b", re.I)
COMPLAINT = re.compile(r"\b(refund|i (?:want|wish) to complain|not (?:yet )?(?:received|delivered|arrived|working)|still (?:has ?n[o’']t|not|hasn[’']t)|disappointed|unacceptable|frustrat\w*|poor service|nobody (?:has )?repl)", re.I)
REQUEST = re.compile(r"\b(please|kindly|can you|could you|would you|need you to|would like you to|let'?s|make sure|ensure|"
                     r"don'?t forget|remember to|i want|we (?:also )?need to|you should|has requested)\b", re.I)
MEETING = re.compile(r"\b(meeting|call|review meeting|demo|presentation|interview|workshop|session)\b[^.]{0,60}\b(scheduled|on|at)\b", re.I)
SIGNOFF = re.compile(r"^(?:(?:best|kind|warm|many)\s+)?(?:regards|wishes)|^(?:thanks|thank you|cheers|sincerely|yours|best)\b", re.I)
ROLE_WORDS = {"manager", "team", "engineer", "director", "lead", "officer", "department", "head", "research", "engineering",
              "project", "ceo", "cto", "founder", "support", "sales", "marketing", "hr", "admin", "operations", "group"}


def _clean(text):
    """Drop markdown emphasis and squeeze whitespace."""
    return re.sub(r"\s+", " ", re.sub(r"\*\*|__|`", "", text)).strip()


def _sentences(text):
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n+", text) if s.strip()]


def _is_header(line):
    return bool(re.match(r"^(from|to|cc|bcc|date|sent|subject|re|fw|fwd)\s*:", line, re.I))


def subject_line(email):
    """The Subject: line, or the first line when the email starts with a bare subject."""
    m = re.search(r"^subject:\s*(.+)$", email, re.I | re.M)
    if m:
        return m.group(1).strip()
    first = next((l.strip() for l in email.splitlines() if l.strip()), "")
    if len(first) < 200 and not re.match(r"(?i)(dear|hi|hello|hey|good (morning|afternoon|evening))\b", first) and not first.endswith("."):
        return first
    return ""


def topic(subject):
    """'Action Required: AI Project – Final Deliverables' -> 'AI Project – Final Deliverables'."""
    return re.sub(r"^(?:(?:action required|urgent|important|reminder|fw|fwd|re)\s*[:\-–]\s*)+", "", subject, flags=re.I).strip()


def _short(text, limit=190):
    text = _clean(text).rstrip(" ,;:")
    return text if len(text) <= limit else text[:limit].rsplit(" ", 1)[0].rstrip(" ,;:") + "…"


def focus_text(email, limit=FOCUS_CHARS):
    """For the model: the whole email if it fits, else its opening plus every sentence with a request,
    date or urgency word from the rest (so details near the end are not lost)."""
    if len(email) <= limit:
        return email
    head = email[: limit // 2].rsplit("\n", 1)[0]
    out = head
    for s in _sentences(email[len(head):]):
        if DEADLINE_RE.search(s) or URGENT_WORDS.search(s) or REQUEST.search(s) or re.match(r"\d+[.)]\s", s):
            if len(out) + len(s) + 1 > limit:
                break
            out += "\n" + s
    return out


def rule_deadlines(email):
    """'<when> — <task>' for each date/time the email ties to a task or meeting (first mention of each date)."""
    found, seen = [], set()
    for s in _sentences(email):
        if _is_header(s) or len(s) > 600:
            continue
        for m in DEADLINE_RE.finditer(s):
            cue = s[max(0, m.start() - 1): m.start(1)]
            if not STRONG_CUE.search(s[: m.end()]) and not re.search(r"\b(by|before|until|due)\b", cue, re.I):
                continue  # "on Monday" alone is just a date, not a deadline
            when = _clean(m.group(1))
            if DURATION.match(when) and not re.search(r"\bwithin\W*$", s[: m.start(1)], re.I):
                continue  # "3 days" is a deadline only as "within 3 days"
            if DURATION.match(when):
                when = "Within " + when
            key = re.sub(r"[^a-z0-9]", "", (re.search(rf"{_MONTH}\s+\d{{1,2}}|\d{{1,2}}\s+{_MONTH}|{_DAY}|today|tomorrow|tonight",
                                                      when, re.I) or [when])[0].lower())
            if key in seen:
                continue
            seen.add(key)
            before = _clean(s[: m.start(1)])
            before = re.sub(r"(?i)[\s,:\-–—]*\b(?:(?:on or )?(?:by|before)(?: the)?|until|no later than|due(?: on| by)?|on|at|for|within"
                            r"|(?:latest )?by(?: the)? end of|latest(?: by)?)\s*$", "", before)
            before = re.sub(r"(?i)\s+latest$", "", before).rstrip(" :-–—")
            if re.fullmatch(r"(?i)(?:the\s+)?(?:deadline|due date|last date)", before):  # "Deadline: 20/10 for the bank details"
                after = re.sub(r"(?i)^[\s,:\-–—]*(?:for|to)?\s*", "", _clean(s[m.end():])).rstrip(".")
                before = f"Deadline for {after}" if after else before
            task = _short(before) or _short(s)
            found.append(f"{when[0].upper() + when[1:]}{DEADLINE_SEP}{task}")
    return found


def _imperative(sentence):
    """'Mark, can you please also confirm X?' -> 'Confirm X'  ·  "Let's decide by Friday." -> 'Decide by Friday'."""
    s = _clean(sentence).rstrip(".?!").strip()
    s = re.sub(r"^(?:also|and|so|but|in addition|finally|first|second|third),?\s+", "", s, flags=re.I)
    s = re.sub(r"^[A-Z][a-z]+,\s+(?=(?:can|could|would|will|please|kindly)\b)", "", s)  # leading "Mark, "
    s = re.sub(r"^(?:could|can|would|will) you (?:please |kindly )?", "", s, flags=re.I)
    s = re.sub(r"^(?:the team|we|i) (?:would )?(?:also )?(?:like|need|want|would like) you to\s+", "", s, flags=re.I)
    s = re.sub(r"^(?:we|i) (?:also )?need to\s+", "", s, flags=re.I)
    s = re.sub(r"^the [\w\s-]{1,40}? (?:has|have) requested (an?|the) ", r"Prepare \1 ", s, flags=re.I)  # "The deployment team has requested a checklist"
    # "In addition to the technical work, please prepare X" -> "Prepare X" (but keep conditions: "If you ..., please")
    if not re.match(r"(?i)(if|when|whenever|unless|because|as soon as)\b", s):
        s = re.sub(r"^[^,]{3,70},\s*(?:please|kindly)\s+(?:also\s+)?", "", s, flags=re.I)
    else:
        s = re.sub(r",\s*(?:please|kindly)\s+", ", ", s, count=1, flags=re.I)
    s = re.sub(r"^(?:please|kindly)\s+(?:also\s+)?", "", s, flags=re.I)
    s = re.sub(r"^let'?s\s+", "", s, flags=re.I)
    return s[:1].upper() + s[1:] if s else ""


def rule_requests(email):
    """Instructions taken from the sentences where the sender asks for something ('please send…')."""
    out = []
    for s in _sentences(email):
        if REQUEST.search(s) and not _is_header(s) and 12 <= len(s) <= 600 and not re.match(r"(?i)please note|thank", s):
            task = _short(_imperative(s))
            # "I want a refund" is a demand and "The goal is..." a statement, not tasks
            if task and not re.match(r"(?i)(i|we|it|this|that|the|our|these|those|because)\b", task):
                out.append(task)
    return out


KEY_TASK = re.compile(r"\b(send|submit|upload|share|present|presentation|demo|demonstration|meeting|review|report|deadline|"
                      r"deliver|prepare|complete|update)\b", re.I)


def pick_tasks(tasks, limit=MAX_ACTIONS):
    """Keep at most `limit` tasks in email order, preferring ones with a date or a key deliverable."""
    if len(tasks) <= limit:
        return tasks
    rank = sorted(range(len(tasks)), key=lambda i: (not DATE_RE.search(tasks[i]), not KEY_TASK.search(tasks[i]), i))
    keep = sorted(rank[:limit])
    return [tasks[i] for i in keep]


def rule_deliverables(email):
    """Items of a numbered list in the email (e.g. '1. Final source code...'), when it has 3 or more."""
    items = [_short(m.group(1), 140).rstrip(".") for m in re.finditer(r"^\s*\d{1,2}[.)]\s+(.{3,200})$", email, re.M)]
    return items if len(items) >= 3 else []


def _content_words(text):
    stop = {"that", "this", "with", "from", "they", "their", "should", "will", "email", "please", "could", "would",
            "whether", "also", "make", "sure", "the", "and", "for", "all", "any"}
    return {w for w in re.findall(r"[a-z0-9]{3,}", text.lower()) if w not in stop}


def _overlap(a, b):
    wa, wb = _content_words(a), _content_words(b)
    return len(wa & wb) / min(len(wa), len(wb)) if wa and wb else 0.0


def merge_tasks(primary, extra, limit=MAX_ACTIONS):
    """primary first; add items from extra that aren't already covered (shared words < 50% of the shorter one)."""
    out = []
    for item in list(primary) + list(extra):
        if all(_overlap(item, o) < 0.5 for o in out):
            out.append(item)
    return out[:limit]


def for_reader(task, name):
    """Rewrite a task from the sender's words for the reader: 'Send me X' -> 'Send Priya X'."""
    who = name or "the sender"
    task = re.sub(r"\bme\b", who, task)
    return re.sub(r"\bmy\b", f"{who}'s" if name else "the sender's", task)


def commitment(task):
    """'Send me the Q4 numbers by tomorrow' -> 'send you the Q4 numbers by tomorrow' (for the reply)."""
    c = _imperative(task)
    c = re.split(r",?\s+(?:or|otherwise|else)\s+(?:i|we)\b", c, flags=re.I)[0]  # drop "..., or I want a refund"
    c = re.sub(r"\bme\b", "you", c)
    c = re.sub(r"\bmy\b", "your", c)
    return c[:1].lower() + c[1:]


def urgency(email, deadlines):
    """(level, reasons) from clear signals instead of the model's guess."""
    score, reasons = 0, []
    subject = subject_line(email)
    words = {m.group(0).lower() for m in URGENT_WORDS.finditer(email)}
    if words:
        score += 2 if URGENT_WORDS.search(subject) or len(words) > 1 else 1
        reasons.append("says " + ", ".join(f'"{w}"' for w in sorted(words)))
    if any(SOON.search(d.split(DEADLINE_SEP)[0]) for d in deadlines):
        score += 2
        reasons.append("a deadline is today or tomorrow")
    elif deadlines:
        score += 1
        reasons.append("has a deadline" if len(deadlines) == 1 else f"{len(deadlines)} deadlines")
    if email.count("!") >= 3:
        score += 1
        reasons.append("several exclamation marks")
    level = "High" if score >= 3 else "Medium" if score >= 1 else "Low"
    return level, reasons or ["no deadline or urgent wording"]


def _name_like(line):
    words = line.split()
    return (bool(re.fullmatch(r"[A-Z][a-zA-Z'’-]+(?: [A-Z][a-zA-Z'’-]+){0,2}", line))
            and not any(w.lower() in ROLE_WORDS for w in words))


def sender_name(email):
    """First name of the sender: From: line, else the name under the sign-off ('Best regards,\\n\\nRahul Sen')."""
    m = re.search(r"^from:\s*\"?([A-Za-z][\w'-]*)[^<\n]*<", email, re.I | re.M)
    if m and m.group(1).lower() not in {"the", "no", "noreply", "info", "support"}:
        return m.group(1)
    lines = [l.strip() for l in email.strip().splitlines() if l.strip()]
    for i in range(len(lines) - 1, -1, -1):
        if SIGNOFF.match(lines[i]) and len(lines[i]) <= 30:
            for cand in lines[i + 1: i + 3]:
                if _name_like(cand):
                    return cand.split()[0]
            break
    for line in reversed(lines[-4:]):
        if _name_like(line):
            return line.split()[0]
    return ""


# ------------------------------------------------------------------ cleaning model output
def _closest(text, options, default):
    low = text.lower()
    for opt in options:
        if opt.lower() in low or opt.split()[0].lower() in low.split()[:3]:
            return opt
    return default


def _list_items(text, limit=8):
    items = []
    for line in text.splitlines():
        line = _clean(re.sub(r"^\s*(?:[-*•]|\d+[.)])\s*", "", line)).strip("*").strip()
        if not line or re.fullmatch(r"(?i)none\.?|no action items\.?|n/?a", line) or line.endswith(":"):
            continue
        if line.lower() not in (i.lower() for i in items):
            items.append(line)
    return items[:limit]


def _grounded(item, email):
    """At least half of an item's content words must appear in the email (filters invented tasks)."""
    words = [w for w in re.findall(r"[a-z]{4,}", item.lower()) if w not in {"that", "this", "with", "from", "they", "their", "should", "will", "email"}]
    if not words:
        return False
    low = email.lower()
    return sum(w in low for w in words) / len(words) >= 0.5


# The reply model (train_reply_dpo.py) is trained and used with exactly this prompt.
REPLY_SYSTEM = "You write short, polite, professional email replies on behalf of the person who received the email."
REPLY_TASK = ("Write a reply to this email. Confirm each request and every deadline with its exact date and time, "
              "write in the first person, and do not invent facts, dates or promises.")


def reply_messages(email):
    return [{"role": "system", "content": REPLY_SYSTEM},
            {"role": "user", "content": f"{REPLY_TASK}\n\nEmail:\n\"\"\"\n{email.strip()}\n\"\"\""}]


def _messages(task, email, system=SYSTEM):
    return [{"role": "system", "content": system},
            {"role": "user", "content": f"{task}\n\nEmail:\n\"\"\"\n{email}\n\"\"\""}]


def _when_phrase(when):
    """'This Friday' -> 'this Friday' (weekday and month names keep their capital)."""
    if when and when.split()[0].lower().rstrip(",") in {"this", "next", "coming", "end", "today", "tomorrow", "tonight", "eod", "cob"}:
        return when[0].lower() + when[1:]
    return when


def _deadline_promise(line):
    """'5:00 PM on October 7, 2026 — The final report should be ready' -> "I'll have the final report ready by 5:00 PM on October 7, 2026"."""
    when, task = (line.split(DEADLINE_SEP, 1) + [""])[:2]
    when = _when_phrase(when)
    m = re.match(r"(?i)(?:the\s+)?(.+?)\s+(?:should|must|needs? to|will|has to)\s+be\s+(ready|completed|submitted|done|delivered|finali[sz]ed|shared|sent)$", task)
    if m:
        return f"I'll have the {m.group(1)} {m.group(2)} by {when}"
    m = re.match(r"(?i)(?:the\s+)?(.+\b(?:meeting|call|review|demo|presentation|interview|session|workshop))\b.*?\b(?:is|has been)\s+scheduled", task)
    if m:
        return f"I'll be prepared for the {m.group(1)} on {when}"
    c = commitment(task)
    return f"I'll {c} by {when}" if c and not re.match(r"(?i)(the|a|an|our|your)\b", c) else f"Noted: {task} ({when})"


def _template_reply(name, intent, actions, deadlines, deliverables=(), email=""):
    hi = f"Hi {name}," if name else "Hi,"
    if len(actions) >= 5 or len(deadlines) >= 2 or deliverables:  # a big request: confirm everything clearly
        scope = f"the remaining tasks and the {len(deliverables)} deliverables" if deliverables else "all the tasks you listed"
        lines = [f"Thank you for the detailed update. I've noted {scope}, and I'll treat them as a priority."]
        if deadlines:
            lines += ["", "To confirm the timeline:"] + [f"- {_deadline_promise(d)}." for d in deadlines[:5]]
        closing = ("I'll keep you posted on progress and let you know right away if I run into any blockers."
                   if re.search(r"\bblocker", email, re.I) else "I'll keep you posted on progress.")
        return f"{hi}\n\n" + "\n".join(lines) + f"\n\n{closing}"
    when = _when_phrase(deadlines[0].split(DEADLINE_SEP, 1)[0]) if deadlines else ""
    when = f" by {when}" if when and not when.lower().startswith(("by ", "before ")) else (f" {when}" if when else "")
    if intent.startswith("Complaint"):
        body = ("Thank you for letting us know, and I'm sorry for the trouble. I'm looking into this now "
                f"and will get back to you{when or ' shortly'} with an update.")
    elif intent.startswith(("Information / update", "Notification or alert", "Thank you")):
        body = "Thanks for the update — noted. I'll reach out if I have any questions."
    elif intent.startswith("Meeting or scheduling"):
        body = "Thanks for the note about the schedule. I'll check my calendar and confirm shortly."
    elif actions:
        promises = [commitment(a) for a in actions[:3] if commitment(a)]
        body = ("Thanks for your email. I'll " + ", and I'll ".join(promises) + "." if promises
                else f"Thanks for your email. I'll take care of this{when}, and I'll let you know once it's done.")
    else:
        body = "Thanks for your email. I'll look into it and get back to you soon."
    return f"{hi}\n\n{body}"


# ------------------------------------------------------------------ the assistant
def analyze(email, summarize, generate, generate_tuned=None, generate_reply=None):
    """
    summarize(email) -> str                   summary from the fine-tuned model
    generate(messages, max_tokens) -> str     greedy answer from the original instruct model
                                              (intent, sentiment)
    generate_tuned(messages, max_tokens)      the fine-tuned summarizer (Custom-vN), used for the action
                                              items (and as a reply fallback); defaults to generate
    generate_reply(messages, max_tokens)      the reply model from train_reply_dpo.py, tried first for the
                                              suggested response; None if it hasn't been trained
    Yields (section_key, text) in the order of SECTIONS.
    """
    focus = focus_text(email)
    long_email = len(email) > 2500
    # rule-based facts come from the whole email
    deadlines = rule_deadlines(email)
    requests = rule_requests(email)
    deliverables = rule_deliverables(email)
    name = sender_name(email)
    subject = subject_line(email)

    yield "summary", re.sub(r"\*\*|__", "", summarize(focus)).strip()

    # ---- intent: category tags + description
    raw = generate(_messages(
        "What is the main intent of this email? Reply in exactly this format:\n"
        "<category>: <everything the sender wants, in at most 20 words>\n"
        f"The category must be one of: {', '.join(INTENTS)}.", focus), 70)
    labels = ["Complaint"] if COMPLAINT.search(email) else [_closest(raw, INTENTS, "Information / update")]
    if len(requests) >= 3 and labels[0] not in ("Request for action", "Complaint"):
        labels.insert(0, "Request for action")
    if MEETING.search(email) and "Meeting or scheduling" not in labels:
        labels.append("Meeting or scheduling")
    detail = raw.split(":", 1)[1] if ":" in raw else raw
    detail = _clean((detail.strip().splitlines() or [""])[0]).strip("-–— ")
    if (len(requests) >= 4 or not detail or not _grounded(detail, focus)) and topic(subject):
        detail = topic(subject)  # many requests: the subject line describes the whole email better than one of them
    elif detail and not _grounded(detail, focus):
        detail = ""
    label = " + ".join(dict.fromkeys(labels))
    yield "intent", f"{label}: {detail}" if detail and detail.lower() != label.lower() else label

    # ---- action items: fine-tuned model + every request in the email; plus its numbered deliverables
    raw = (generate_tuned or generate)(_messages(
        "What does the sender ask the reader of this email to do? Write each request as a short instruction "
        "that starts with a verb (for example: Send the report by Friday). Use a numbered list. "
        "Leave out things that are already done. If nothing is asked, reply exactly: None", focus), 200)
    model_tasks = [a for a in _list_items(raw, 8) if _grounded(a, focus)]
    merged = (merge_tasks(requests, model_tasks, limit=99) if long_email
              else merge_tasks(model_tasks, requests, limit=99))
    actions = pick_tasks(merged)
    shown = [for_reader(a, name) for a in actions]
    text = "\n".join(f"{i}. {a}" for i, a in enumerate(shown, 1)) if shown else "None"
    if deliverables:
        text += "\n\nDeliverables listed in the email:\n" + "\n".join(f"- {d}" for d in deliverables)
    yield "actions", text

    # ---- deadline: every date the email ties to a task or meeting
    def readable(line):  # "Please send me X" -> "Send Rahul X"
        when, task = line.split(DEADLINE_SEP, 1)
        if REQUEST.match(task) or re.match(r"(?i)(could|can|would) you\b", task):
            task = for_reader(_imperative(task), name)
        return f"{when}{DEADLINE_SEP}{task}"
    yield "deadline", "\n".join(readable(d) for d in deadlines) if deadlines else "No deadline mentioned."

    # ---- sentiment + urgency
    raw = generate(_messages(
        f"What is the sender's tone in this email? Reply with one word from: {', '.join(SENTIMENTS)}.", focus), 8)
    sentiment = _closest(raw, SENTIMENTS, "Neutral")
    if label.startswith("Complaint") and sentiment in ("Neutral", "Positive", "Excited", "Appreciative"):
        sentiment = "Frustrated"
    level, reasons = urgency(email, deadlines)
    yield "sentiment", f"Sentiment: {sentiment}\nUrgency: {level} ({'; '.join(reasons)})"

    # ---- suggested response
    greeting = f"Hi {name}," if name else "Hi,"
    complex_email = len(actions) >= 5 or len(deadlines) >= 2 or bool(deliverables)
    third_person = rf"^(?:{re.escape(name)}|the sender|he|she|they)\b|\b(?:{re.escape(name) if name else 'the sender'}|she|he) (?:is|has|will|was|wants|requests|asks)\b"

    def checked(raw):
        """The reply body if it reads like a real reply from the reader, else None."""
        body = re.sub(r"^(?:hi|hello|dear)\b[^\n,]*,?\s*", "", raw.strip(), flags=re.I)
        body = re.split(r"\n\s*(?:best|kind|warm)?\s*regards|\n\s*(?:thanks|sincerely|cheers|best),?\s*\n", body, flags=re.I)[0].strip()
        ok = (len(body.split()) >= 12 and "**" not in body and not re.search(r"^\s*\d+[.)]", body, re.M)
              and "?" not in body                                            # a reply shouldn't ask the sender's questions back
              # written by the reader, in the first person ("I will…" or "We will…")
              and (re.search(r"\b(I|I'll|I'm|I've)\b", body) or re.search(r"\b(we'll|we will|we are|we have|we've)\b", body, re.I))
              and not re.search(third_person, body, re.I)                     # not a summary about the sender ("Priya is...")
              # written from the sender's side, or giving the sender instructions, instead of replying to them
              and not re.search(r"\b(we've been|we have been|our board|your prompt (?:response|attention)|my request"
                                r"|please (?:refer|notify|deal|follow|check|contact|note))\b", body, re.I)
              and _grounded(body, focus + " thank thanks reply look into update received request requirements")
              and not re.search(r"\b(as an ai|i am unable|i cannot|general advice)\b", body, re.I))
        return body if ok else None

    def confirms_deadlines(body):
        """Every deadline's date must appear in the reply (e.g. 'October 7' or 'Oct 7')."""
        low = body.lower()
        for d in deadlines:
            when = d.split(DEADLINE_SEP, 1)[0]
            m = re.search(rf"({_MONTH})\s+(\d{{1,2}})|(\d{{1,2}})\s+({_MONTH})", when, re.I)
            if m:
                month, day = (m.group(1), m.group(2)) if m.group(1) else (m.group(4), m.group(3))
                if not re.search(rf"\b{month[:3].lower()}\w*\.?\s+{day}\b|\b{day}(?:st|nd|rd|th)?\s+(?:of\s+)?{month[:3].lower()}", low):
                    return False
            else:
                key = re.search(rf"{_DAY}|today|tomorrow|tonight|noon", when, re.I)
                if key and key.group(0).lower() not in low:
                    return False
        return True

    body, writer = None, ""
    # 1. the reply model (Qwen2.5-0.5B-Instruct + SFT + DPO on email_dpo.jsonl), when it has been trained
    if generate_reply is not None:
        body = checked(generate_reply(reply_messages(focus), 260))
        if body and complex_email and not confirms_deadlines(body):
            body = None  # a big request must confirm every deadline; otherwise the timeline template is safer
        writer = "the reply model" if body else ""
    if body is None and not complex_email:
        promises = [commitment(a) for a in actions[:4] if commitment(a)]
        plan = ("say that you will: " + "; ".join(promises) if promises
                else "acknowledge the email" if not label.startswith("Complaint") else "apologise and say you are looking into it")
        prompt = _messages(
            f"You are the person who received this email. Write the body of your reply to the sender in 2 to 3 "
            f"sentences of plain text: thank them, then {plan}. Write in the first person (I will...). Do not ask "
            f"questions. Do not repeat details from the email that you are not replying to. No lists, no greeting, "
            f"no sign-off.", focus,
            system="You write short, polite, professional email replies. Never invent facts, dates or promises.")
        # 2. your fine-tuned summarizer, 3. the original model
        body, writer = checked((generate_tuned or generate)(prompt, 160)), "your fine-tuned model"
        if body is None and generate_tuned is not None:
            body, writer = checked(generate(prompt, 160)), "the original model"
    if body is None:  # 4. the template, built from the tasks and deadlines found
        writer = "the template (no draft passed the reply check)"
        reply = (_template_reply(name, label, actions, deadlines, deliverables, email) if complex_email
                 else _template_reply(name, label, actions, deadlines))
    else:
        reply = f"{greeting}\n\n{body}"
    print(f"[assistant] suggested response written by {writer}")
    yield "response", reply + "\n\nBest regards,\n[Your name]"


def as_text(sections):
    """The analysis in the plain-text format: 'Summary:\\n...\\n\\nIntent:\\n...'."""
    titles = dict(SECTIONS)
    return "\n\n".join(f"{titles[k]}:\n{v}" for k, v in sections.items() if k in titles)
