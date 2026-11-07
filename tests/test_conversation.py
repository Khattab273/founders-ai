"""Shared conversation turn: typed messages never speak; spoken ones do, with the same replies."""

from core import conversation, intake, llm


def fake_llm(monkeypatch, turn):
    def stream(messages, schema, **k):
        for s in turn["reply"].split(". "):
            yield ("sentence", s)
        yield ("result", (schema.model_validate(turn), "fake"))
    monkeypatch.setattr(llm, "stream_json", stream)
    monkeypatch.setattr(llm, "chat_json", lambda m, schema, **k: (schema.model_validate(turn), "fake"))


TURN = {"reply": "Hi Maha. What are you raising?", "role": "founder", "profile": {"display_name": "Maha"}}


def test_typed_turn_never_speaks(monkeypatch):
    fake_llm(monkeypatch, TURN)
    chat = conversation.start()
    events = list(conversation.turn(chat.id, "I'm Maha", spoken=False))
    assert not [e for e in events if e[0] == "say"]
    assert chat.history[-1]["content"] == "Hi Maha. What are you raising?"
    assert chat.history[-2]["content"] == "I'm Maha"


def test_spoken_turn_speaks_the_same_reply(monkeypatch):
    fake_llm(monkeypatch, TURN)
    chat = conversation.start()
    events = list(conversation.turn(chat.id, "I'm Maha", spoken=True))
    said = [d for k, d in events if k == "say"]
    assert said and " ".join(said).startswith("Hi Maha")
    assert chat.history[-2]["content"] == "🎤 I'm Maha" and chat.history[-1]["content"] == TURN["reply"]


def test_spoken_turn_reports_transcription_problems(monkeypatch):
    from core import voice
    monkeypatch.setattr(voice, "transcribe", lambda path: voice.Transcript(error=voice.NOTHING_HEARD))
    chat = conversation.start()
    assert list(conversation.spoken_turn(chat.id, "x.wav")) == [("error", voice.NOTHING_HEARD)]
    assert len(chat.history) == 1          # nothing added to the chat for an empty recording


def test_new_message_cancels_the_reply_in_progress(monkeypatch):
    import threading
    fake_llm(monkeypatch, TURN)
    chat = conversation.start()
    first = conversation.turn(chat.id, "I'm Maha", spoken=True)
    next(first)                                   # first turn is running (holding the chat)
    done = []
    t = threading.Thread(target=lambda: done.append(list(conversation.turn(chat.id, "raising 1.5M", spoken=True))))
    t.start()
    import time; time.sleep(0.2)                  # the second turn has asked the first to stop
    rest = list(first)                            # first turn stops at its next step
    t.join(timeout=5)
    assert not t.is_alive() and done and rest == []
    assert chat.history[-2]["content"] == "🎤 raising 1.5M"


def test_cancelled_turn_leaves_no_trace(monkeypatch):
    fake_llm(monkeypatch, TURN)
    chat = conversation.start()
    before = list(chat.history)
    s = intake.get_session(chat.id)
    n = len(s.history)
    g = conversation.turn(chat.id, "Hi I'm Maha", spoken=True)
    next(g); next(g); next(g)            # part-way through the reply
    g.close()                            # the member interrupted / the browser dropped it
    assert chat.history == before and len(s.history) == n and s.user_turns == 0
