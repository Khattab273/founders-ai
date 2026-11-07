"""Test page for the AI layer (core/conversation.py). Not the real frontend.

Run:  .venv/bin/python scripts/chat_demo.py   ->  http://127.0.0.1:7870
      AI_CACHE_DIR=./ai_cache .venv/bin/python scripts/chat_demo.py   (rehearsals replay instantly)

Text: type (or attach a CV / deck) and get text replies.
Voice: press "Start voice chat" once, then just talk. The browser notices when you stop, sends what you
said to /voice/<chat id>, plays each reply sentence as soon as it's ready, and listens again. Talking
over the agent interrupts it. Both paths run the same conversation (core.conversation.turn).
"""

from __future__ import annotations

import base64
import json
import os
import sys
import tempfile
import threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import gradio as gr
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

from fastapi import FastAPI, Request  # noqa: E402
from fastapi.responses import JSONResponse, StreamingResponse  # noqa: E402

from core import conversation, voice  # noqa: E402


def cards(matches: list[dict]) -> str:
    if not matches:
        return "_No recommendations yet._"
    icons = ("✓", "?", "→")
    return "\n\n".join(f"**{m['name']}** · {m['role']} · {m['score']} ({m['fit']})  \n{m['headline']}  \n" +
                         "  \n".join(f"{icons[i] if i < 3 else '·'} {line}" for i, line in enumerate(m["summary"])) +
                         f"  \n`{m['id']}`" for m in matches)


def begin():
    chat = conversation.start()
    return chat.id, chat.history, {}, cards([])


def send_text(message: dict, chat_id: str):
    """Typed message (and optional upload): text replies only. Outputs: chat, input box, profile, cards."""
    text, files = message.get("text", ""), message.get("files", [])
    skip = gr.skip()
    for kind, data in conversation.turn(chat_id, text, file=files[0] if files else None, spoken=False):
        if kind == "history":
            yield list(data), None, skip, skip
        elif kind == "profile":
            yield skip, None, data, skip
        elif kind == "matches":
            yield skip, None, skip, cards(data)


def refresh(chat_id: str):
    """The voice chat updates the conversation on the server; this redraws it."""
    try:
        chat = conversation.get(chat_id)
    except KeyError:
        return gr.skip(), gr.skip(), gr.skip()
    return list(chat.history), chat.profile, cards(chat.matches)


def intro(person_id: str, chat_id: str, history: list):
    from core import matchmaker
    try:
        text = matchmaker.draft_intro(chat_id, person_id.strip())
    except (ValueError, KeyError) as exc:
        text = str(exc)
    return history + [{"role": "assistant", "content": "✉️ Intro draft:\n\n" + text}]


# ---------------- voice endpoint ----------------

api = FastAPI()


def _audio_b64(sentence: str) -> str | None:
    out = voice.speech(sentence)
    if out is None:
        return None
    rate, pcm = out
    return base64.b64encode(voice._wav_bytes(pcm, rate)).decode()


@api.post("/voice/{chat_id}")
async def voice_turn(chat_id: str, request: Request):
    """One spoken message in (WAV), a stream of NDJSON events out: heard, audio (per sentence), error, done."""
    try:
        conversation.get(chat_id)
    except KeyError:
        return JSONResponse({"error": "Unknown chat. Reload the page."}, status_code=404)
    body = await request.body()
    if not body or len(body) > 8_000_000:
        return JSONResponse({"error": "Empty or too long recording."}, status_code=400)
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp.write(body)
    tmp.close()

    def events():
        try:
            for kind, data in conversation.spoken_turn(chat_id, tmp.name):
                if kind == "heard":
                    yield json.dumps({"type": "heard", "text": data}) + "\n"
                elif kind == "error":
                    yield json.dumps({"type": "error", "text": data, "audio": _audio_b64(data)}) + "\n"
                elif kind == "say":
                    yield json.dumps({"type": "audio", "text": data, "audio": _audio_b64(data)}) + "\n"
                elif kind in ("history", "matches"):
                    yield json.dumps({"type": "update"}) + "\n"
        except Exception as exc:   # never leave the browser hanging
            yield json.dumps({"type": "error", "text": "Something went wrong. Please try again.",
                              "detail": str(exc)[:200]}) + "\n"
        finally:
            Path(tmp.name).unlink(missing_ok=True)
            yield json.dumps({"type": "done"}) + "\n"

    return StreamingResponse(events(), media_type="application/x-ndjson")


# ---------------- hands-free voice client (runs in the browser) ----------------

VOICE_JS = r"""
<script>
(() => {
  const SPEECH = 0.018, SPEECH_WHILE_PLAYING = 0.07, END_SILENCE_MS = 1300, MIN_SPEECH_MS = 350, PREROLL_S = 0.5;
  const S = {on: false, stream: null, ctx: null, proc: null, rate: 48000, pre: [], cur: null, inSpeech: false,
             speechMs: 0, silenceMs: 0, abort: null, queue: [], playing: null, busy: false,
             lastFrames: null, gotAudio: false, carry: null};
  const el = id => document.getElementById(id);
  const status = t => { const e = el('voice-status'); if (e) e.textContent = t; };
  const chatId = () => { const e = document.querySelector('#chat-id textarea, #chat-id input'); return e ? e.value : ''; };
  const refresh = () => { const b = document.querySelector('#voice-refresh'); if (b) b.click(); };
  const idle = () => status(S.on ? 'Listening… just talk' : 'Voice chat is off');

  function wav(chunks, rate) {                      // Float32 frames -> 16 kHz mono WAV
    const n = chunks.reduce((a, c) => a + c.length, 0), all = new Float32Array(n);
    let o = 0; for (const c of chunks) { all.set(c, o); o += c.length; }
    const ratio = rate / 16000, len = Math.floor(n / ratio), pcm = new Int16Array(len);
    for (let i = 0; i < len; i++) {
      let s = 0, k = Math.floor(i * ratio), e = Math.min(n, Math.floor((i + 1) * ratio));
      for (let j = k; j < e; j++) s += all[j];
      const v = Math.max(-1, Math.min(1, s / Math.max(1, e - k))); pcm[i] = v * 32767;
    }
    const buf = new ArrayBuffer(44 + pcm.length * 2), d = new DataView(buf);
    const w = (p, s) => { for (let i = 0; i < s.length; i++) d.setUint8(p + i, s.charCodeAt(i)); };
    w(0, 'RIFF'); d.setUint32(4, 36 + pcm.length * 2, true); w(8, 'WAVE'); w(12, 'fmt ');
    d.setUint32(16, 16, true); d.setUint16(20, 1, true); d.setUint16(22, 1, true); d.setUint32(24, 16000, true);
    d.setUint32(28, 32000, true); d.setUint16(32, 2, true); d.setUint16(34, 16, true); w(36, 'data');
    d.setUint32(40, pcm.length * 2, true); new Int16Array(buf, 44).set(pcm);
    return new Blob([buf], {type: 'audio/wav'});
  }

  function stopPlayback() {
    if (S.playing) { S.playing.pause(); S.playing = null; }
    S.queue = [];
  }
  function playNext() {
    if (S.playing || !S.queue.length) { if (!S.playing && !S.busy) idle(); return; }
    const a = new Audio('data:audio/wav;base64,' + S.queue.shift());
    S.playing = a; status('Speaking… (talk to interrupt)');
    a.onended = a.onerror = () => { S.playing = null; playNext(); };
    a.play().catch(() => { S.playing = null; playNext(); });
  }

  async function send(frames) {
    if (S.abort) S.abort.abort();                   // a new message replaces one still being answered
    S.lastFrames = frames; S.gotAudio = false;
    const blob = wav(frames, S.rate);
    S.abort = new AbortController(); S.busy = true; status('Thinking…');
    try {
      const r = await fetch('voice/' + encodeURIComponent(chatId()), {method: 'POST', body: blob,
                            headers: {'Content-Type': 'audio/wav'}, signal: S.abort.signal});
      if (!r.ok) { status('Voice error: ' + ((await r.json()).error || r.status)); S.busy = false; return; }
      const reader = r.body.getReader(), dec = new TextDecoder(); let buf = '';
      while (true) {
        const {value, done} = await reader.read(); if (done) break;
        buf += dec.decode(value, {stream: true});
        let i; while ((i = buf.indexOf('\n')) >= 0) {
          const line = buf.slice(0, i).trim(); buf = buf.slice(i + 1); if (!line) continue;
          const ev = JSON.parse(line);
          if (ev.type === 'heard' || ev.type === 'update') refresh();
          if ((ev.type === 'audio' || ev.type === 'error') && ev.audio) { S.gotAudio = true; S.queue.push(ev.audio); playNext(); }
          if (ev.type === 'error' && !ev.audio) status(ev.text);
          if (ev.type === 'done') refresh();
        }
      }
    } catch (e) { if (e.name !== 'AbortError') status('Voice connection problem. Try again.'); }
    S.busy = false; if (!S.playing) idle();
  }

  function onAudio(frame, ms) {
    let sum = 0; for (let i = 0; i < frame.length; i++) sum += frame[i] * frame[i];
    const rms = Math.sqrt(sum / frame.length), thr = S.playing ? SPEECH_WHILE_PLAYING : SPEECH;
    S.pre.push(frame); while (S.pre.length * frame.length > PREROLL_S * S.rate) S.pre.shift();
    if (rms > thr) {
      if (!S.inSpeech) {
        S.inSpeech = true; S.speechMs = 0; S.cur = S.pre.slice(); stopPlayback();
        if (S.abort) {                              // they're talking again: drop the old reply
          if (!S.gotAudio && S.lastFrames) S.carry = S.lastFrames;   // ...but keep what they said, if it wasn't answered
          S.abort.abort(); S.abort = null; S.busy = false;
        }
        status('Listening…');
      }
      S.silenceMs = 0;
    } else if (S.inSpeech) { S.silenceMs += ms; }
    if (S.inSpeech) {
      S.cur.push(frame); S.speechMs += ms;
      if (S.silenceMs > END_SILENCE_MS) {
        S.inSpeech = false;
        if (S.speechMs - S.silenceMs >= MIN_SPEECH_MS) {
          const gap = new Float32Array(Math.floor(0.3 * S.rate));
          send(S.carry ? [...S.carry, gap, ...S.cur] : S.cur);  // one message: the cut-off half + the rest
          S.carry = null;
        } else idle();
        S.cur = null;
      }
    }
  }

  async function start() {
    try {
      S.stream = await navigator.mediaDevices.getUserMedia({audio: {echoCancellation: true, noiseSuppression: true,
                                                                    autoGainControl: true}});
    } catch (e) { status('Microphone blocked. Allow microphone access for this site and try again.'); return; }
    S.ctx = new (window.AudioContext || window.webkitAudioContext)(); S.rate = S.ctx.sampleRate;
    const src = S.ctx.createMediaStreamSource(S.stream);
    S.proc = S.ctx.createScriptProcessor(2048, 1, 1);
    S.proc.onaudioprocess = e => { if (S.on) onAudio(new Float32Array(e.inputBuffer.getChannelData(0)),
                                                     2048 / S.rate * 1000); };
    const mute = S.ctx.createGain(); mute.gain.value = 0;
    src.connect(S.proc); S.proc.connect(mute); mute.connect(S.ctx.destination);
    S.on = true; el('voice-btn').textContent = '⏹ Stop voice chat'; idle();
  }
  function stop() {
    S.on = false; stopPlayback(); if (S.abort) S.abort.abort();
    if (S.stream) S.stream.getTracks().forEach(t => t.stop());
    if (S.ctx) S.ctx.close(); S.stream = S.ctx = S.proc = null; S.inSpeech = false;
    el('voice-btn').textContent = '🎙 Start voice chat'; idle();
  }
  window.handshakeVoice = {start, stop, state: S};
  document.addEventListener('click', e => {
    if (e.target && e.target.id === 'voice-btn') { S.on ? stop() : start(); }
  });
})();
</script>
"""

VOICE_HTML = """<div class="voice-box">
<button id="voice-btn" type="button">🎙 Start voice chat</button>
<span id="voice-status">Voice chat is off</span>
<div class="voice-tip">Talk naturally; it replies when you pause, and you can talk over it to interrupt.
Headphones work best.</div></div>"""

CSS = """
#chat-id, #voice-refresh {display: none !important}
.voice-box {display: flex; flex-wrap: wrap; align-items: center; gap: 12px; padding: 4px 2px}
#voice-btn {padding: 10px 18px; border-radius: 999px; border: none; font-weight: 600; cursor: pointer;
            background: #4f46e5; color: white}
#voice-status {font-size: 14px; opacity: .8}
.voice-tip {width: 100%; font-size: 12px; opacity: .6}
"""

with gr.Blocks(title="Handshake AI layer test") as demo:
    chat_id = gr.Textbox(elem_id="chat-id", show_label=False, container=False)
    with gr.Row():
        with gr.Column(scale=3):
            bot = gr.Chatbot(height=560, label="Handshake agent")
            gr.HTML(VOICE_HTML)
            box = gr.MultimodalTextbox(file_types=[".pdf", ".docx", ".pptx", ".txt", ".png", ".jpg"],
                                       placeholder="Type, or attach a CV / pitch deck / fund deck", show_label=False)
        with gr.Column(scale=2):
            found = gr.Markdown(label="Recommendations")
            with gr.Row():
                pid = gr.Textbox(placeholder="person id, e.g. i-ali", show_label=False, scale=3)
                gr.Button("Draft intro", scale=1).click(intro, [pid, chat_id, bot], bot)
            with gr.Accordion("Profile (live)", open=False):
                prof = gr.JSON()
            gr.Button("Restart").click(begin, outputs=[chat_id, bot, prof, found])
    voice_refresh = gr.Button(elem_id="voice-refresh")
    voice_refresh.click(refresh, [chat_id], [bot, prof, found], show_progress="hidden", queue=False)
    demo.load(begin, outputs=[chat_id, bot, prof, found])
    box.submit(send_text, [box, chat_id], [bot, box, prof, found])

demo.queue()
app = gr.mount_gradio_app(api, demo, path="/", head=VOICE_JS, css=CSS, ssr_mode=False)

if __name__ == "__main__":
    import uvicorn
    threading.Thread(target=voice.warm, daemon=True).start()   # load voice models on the GPU in the background
    on_space = bool(os.getenv("SPACE_ID"))
    uvicorn.run(app, host="0.0.0.0" if on_space else "127.0.0.1", port=7860 if on_space else 7870)
