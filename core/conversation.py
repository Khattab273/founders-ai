"""One conversation, typed or spoken: intake until the profile is confirmed, then the matchmaker.

Both the text chat and the voice chat call turn(), so they can't drift apart. A typed message
gets a text reply; a spoken message (spoken=True) also yields the reply as sentences to say.

    chat = start()
    for kind, data in turn(chat.id, text, file=None, spoken=False): ...

Events:
    ("history", list)     the full chat so far, whenever it changes
    ("say", str)          a sentence to speak (only when spoken=True)
    ("profile", dict)     the member's profile as the agent understands it
    ("matches", list)     recommendations (dicts from matchmaker.as_dicts)
    ("mode", str)         "intake" or "match"
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from pathlib import Path

from core import intake, matchmaker, voice

STEP_TEXT = {
    "scan": lambda d: f"🔎 Looking through {d['total']} people…",
    "filter": lambda d: f"✓ {d['label']}: {d['kept']} left",
    "narrow": lambda d: f"✓ Narrowed {d['considered']} to the closest {d['kept']}",
    "rerank": lambda d: f"✓ Scored {d['scored']} on fit, kept the top {d['kept']}",
    "fit": lambda d: f"· {d['name']}: {d['level']}",
}
FOLLOW_UP = "Ask me to refine the list, why someone isn't on it, for more people like someone, or what to ask them."


@dataclass
class Chat:
    id: str
    mode: str = "intake"
    history: list = field(default_factory=list)
    matches: list = field(default_factory=list)
    profile: dict = field(default_factory=dict)
    lock: threading.Lock = field(default_factory=threading.Lock, repr=False)


_chats: dict[str, Chat] = {}


def start() -> Chat:
    st = intake.start()
    chat = Chat(id=st["session_id"], history=[{"role": "assistant", "content": st["reply"]}])
    _chats[chat.id] = chat
    return chat


def get(chat_id: str) -> Chat:
    if chat_id not in _chats:
        raise KeyError("Unknown or expired chat. Start a new one.")
    return _chats[chat_id]


def _search(chat: Chat, gen, spoken: bool):
    """Show search steps live in one bubble, then any reply; speak the outcome if spoken."""
    lines, bubble, reply = [], None, ""
    for kind, data in gen:
        if kind in STEP_TEXT:
            lines.append(STEP_TEXT[kind](data))
            if bubble is None:
                chat.history.append({"role": "assistant", "content": ""})
                bubble = len(chat.history) - 1
            chat.history[bubble]["content"] = "\n".join(lines)
            yield ("history", chat.history)
        elif kind == "done":
            chat.matches = matchmaker.as_dicts(data["matches"])
            yield ("matches", chat.matches)
            if data.get("message"):
                reply = data["message"]
                chat.history.append({"role": "assistant", "content": reply})
                yield ("history", chat.history)
            elif spoken:
                reply = voice.say_matches(chat.matches)
        elif kind == "reply":
            chat.matches = data["matches"]
            reply = data["text"]
            chat.history.append({"role": "assistant", "content": reply})
            yield ("matches", chat.matches)
            yield ("history", chat.history)
    if spoken and reply:
        for sentence in voice.sentences(voice.speakable(reply)):
            yield ("say", sentence)


def turn(chat_id: str, text: str = "", file: str | Path | None = None, spoken: bool = False):
    chat = get(chat_id)
    with chat.lock:   # one turn at a time per chat (e.g. a voice turn and a typed turn can't interleave)
        text = (text or "").strip()
        label = ("🎤 " if spoken else "") + (text or ("📎 " + Path(file).name if file else ""))
        chat.history.append({"role": "user", "content": label})
        yield ("history", chat.history)
        if chat.mode == "intake":
            chat.history.append({"role": "assistant", "content": ""})
            said, st = [], None
            for kind, data in intake.send_stream(chat.id, text, file=file):
                if kind == "say":
                    said.append(data)
                    chat.history[-1]["content"] = " ".join(said)
                    yield ("history", chat.history)
                    if spoken:
                        yield ("say", data)
                else:
                    st = data
            chat.history[-1]["content"] = st["reply"]
            chat.profile = {**st["profile"], "_missing": st["missing"], "_done": st["done"]}
            yield ("history", chat.history)
            yield ("profile", chat.profile)
            if not st["done"]:
                return
            chat.mode = "match"
            yield ("mode", chat.mode)
            yield from _search(chat, matchmaker.find(chat.id), spoken)
            if chat.matches:
                chat.history.append({"role": "assistant", "content": FOLLOW_UP})
                yield ("history", chat.history)
            return
        yield from _search(chat, matchmaker.chat(chat.id, text), spoken)


def spoken_turn(chat_id: str, audio_path: str):
    """A voice message: transcribe, then the same turn as text. Yields ("heard", text) or ("error", text) first."""
    heard = voice.transcribe(audio_path)
    if heard.error:
        yield ("error", heard.error)
        return
    yield ("heard", heard.text)
    yield from turn(chat_id, heard.text, spoken=True)
