"""Understand the spoken command with a small language model running locally.

Talks to any Ollama-compatible server (Ollama on the laptop or Pi, hailo-ollama on an AI HAT+ 2).
The model answers in a fixed JSON shape: whether this is a request for the yogurt, and the word the
speaker used for who should get it ("me", "him", ...). Mapping that word to "the speaker" or
"someone else" is then a plain lookup, which small models get wrong far less often than deciding it themselves.
"""

import json
import re
import urllib.request
from dataclasses import dataclass

SYSTEM = """You read one sentence that a person said to a robot arm named Gurt. The robot can do \
exactly one thing: pick up the yogurt cup and hand it to a person. The sentence comes from a speech \
recogniser, so it may contain small mistakes, and the robot's name ("Gurt", "Yo Gurt", "Yogurt") may \
appear at the start.

Reply with JSON:
- "request": "yogurt" if the person wants the yogurt given, passed, handed, brought or fetched to \
somebody, or says somebody wants it. "nothing" for anything else: other requests, greetings, thanks, \
small talk, unclear words, cancelling, or only the robot's name.
- "recipient": the word or words in the sentence that say who gets the yogurt, copied exactly, for \
example "me", "him", "her", "us", "my friend", "Sarah". If the sentence does not say who, write "me"."""

EXAMPLES = [
    ("give me the yogurt", "yogurt", "me"),
    ("gurt, give him the yogurt", "yogurt", "him"),
    ("can I have the yogurt please", "yogurt", "I"),
    ("hand the yogurt to her", "yogurt", "her"),
    ("what time is it", "nothing", "me"),
    ("yogurt over here", "yogurt", "me"),
    ("the yogurt is for my sister", "yogurt", "my sister"),
    ("hello there", "nothing", "me"),
]

SCHEMA = {
    "type": "object",
    "properties": {
        "request": {"type": "string", "enum": ["yogurt", "nothing"]},
        "recipient": {"type": "string"},
    },
    "required": ["request", "recipient"],
}

# Recipient words that mean "the person who spoke". Anything else is somebody else.
SELF_WORDS = {"", "me", "i", "us", "we", "myself", "ourselves", "my", "mine", "here", "over here", "my hand"}


@dataclass
class Command:
    text: str
    give_yogurt: bool
    recipient: str        # "speaker" or "someone_else"
    recipient_words: str  # what the model picked out of the sentence, e.g. "him"


def parse_command(text: str, url: str, model: str, timeout_s: float = 60) -> Command:
    """Blocking; call it from a worker thread. Raises RuntimeError if the model server is unreachable."""
    if not text.strip():
        return Command(text, False, "speaker", "")
    messages = [{"role": "system", "content": SYSTEM}]
    for said, request, recipient in EXAMPLES:
        messages.append({"role": "user", "content": said})
        messages.append({"role": "assistant", "content": json.dumps({"request": request, "recipient": recipient})})
    messages.append({"role": "user", "content": text})
    body = {"model": model, "messages": messages, "stream": False, "format": SCHEMA,
            "options": {"temperature": 0, "num_predict": 40}, "keep_alive": "1h"}
    request = urllib.request.Request(f"{url}/api/chat", json.dumps(body).encode(), {"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=timeout_s) as response:
            answer = json.loads(json.loads(response.read())["message"]["content"])
    except Exception as e:
        raise RuntimeError(f"language model '{model}' at {url} did not answer ({e}) - is Ollama running "
                           f"and has `ollama pull {model}` been done?") from e
    words = str(answer.get("recipient", "")).strip()
    if words.lower() not in text.lower():  # the model must copy the word from the sentence, not invent one
        words = ""
    is_self = re.sub(r"[^a-z ]", "", words.lower()).strip() in SELF_WORDS
    return Command(text, answer.get("request") == "yogurt", "speaker" if is_self else "someone_else", words)
