#!/usr/bin/env python3
"""
Text-to-speech engine for the website (Azure AI Speech REST API + ffmpeg).

Turns a two-person dialogue script into one audio file:

    Host: Welcome back to the show.
    Guest (excited): Thanks for having me!
    Host: Sure. {pause 0.6} {thoughtful} Let's start at the beginning.

  * Exactly two speakers: labels are matched to the speaker names you set
    (default "Host" and "Guest") or to "Speaker 1" / "Speaker 2".
  * A line without a label continues the previous speaker.
  * Text in [square brackets] is ignored (notes such as [laughs]).
  * (emotion) after a name applies to the whole turn; {emotion} switches
    mid-turn; {pause} / {pause 1.5} inserts silence in seconds.
  * Emotions only change speed, pitch and volume (see EMOTIONS below).

Environment variables:
  AZURE_SPEECH_KEY       key of your Azure Speech resource (required)
  AZURE_SPEECH_REGION    region of that resource, e.g. eastus (required)
  AZURE_SPEECH_ENDPOINT  optional full base URL override (used for testing)
  TTS_CONCURRENCY        parallel requests to the voice service (default 2)
  MONTHLY_CHAR_LIMIT     characters this site may send per month (default 450000,
                         safely under the 500,000 free characters; 0 = no limit)
"""
import asyncio
import json
import os
import random
import re
import shutil
import subprocess
import threading
import time
import urllib.error
import urllib.request
import wave
from xml.sax.saxutils import escape as _xml_escape

import numpy as np

SR = 24000                      # we request 24 kHz mono MP3 from Azure
MAX_SPEAKERS = 2
OUT_DIR = os.path.join(os.getcwd(), "tts_output")

PREFERRED = {
    "en": ["en-US-AndrewMultilingualNeural", "en-US-AvaMultilingualNeural",
           "en-US-BrianMultilingualNeural", "en-US-EmmaMultilingualNeural",
           "en-US-GuyNeural", "en-US-JennyNeural", "en-US-ChristopherNeural", "en-US-AriaNeural"],
    "ar": ["ar-EG-ShakirNeural", "ar-EG-SalmaNeural", "ar-SA-HamedNeural", "ar-SA-ZariyahNeural"],
    "ja": ["ja-JP-KeitaNeural", "ja-JP-NanamiNeural"],
}
LANGS = {"English": "en", "Arabic": "ar", "Japanese": "ja", "French": "fr",
         "Spanish": "es", "German": "de", "All languages": ""}
DEFAULT_NAMES = ["Host", "Guest"]

# Emotion presets: (speed %, pitch Hz, volume %) ADDED to each speaker's own settings.
# Only delivery changes (prosody). Edit these numbers to taste; add your own rows freely.
EMOTIONS = {
    "neutral":    (0, 0, 0),
    "happy":      (8, 3, 5),
    "excited":    (14, 5, 8),
    "curious":    (4, 4, 0),
    "surprised":  (10, 7, 8),
    "serious":    (-8, -3, 3),
    "thoughtful": (-12, -2, -4),
    "calm":       (-10, -1, -6),
    "sad":        (-16, -5, -10),
}
EMOTION_ALIASES = {"enthusiastic": "excited", "cheerful": "happy", "shocked": "surprised",
                   "gentle": "calm", "reflective": "thoughtful"}

EXAMPLE_SCRIPT = """Host: Welcome back to the show. Today we are talking about why we sleep.
Guest (excited): Thanks for having me! It is honestly a stranger story than most people expect.
Host (curious): Let's start with the basics. What actually happens to the body during sleep?
Guest: Quite a lot. {pause 0.6} {thoughtful} Your brain replays the day, clears waste, and files memories away.
"""


# --------------------------------------------------------------------------- ffmpeg
def find_ffmpeg():
    path = shutil.which("ffmpeg")
    if path:
        return path
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        raise RuntimeError("ffmpeg was not found. Install it or run:  pip install imageio-ffmpeg")


def mp3_to_pcm(mp3, ffmpeg):
    r = subprocess.run([ffmpeg, "-loglevel", "error", "-i", "pipe:0", "-f", "s16le",
                        "-ar", str(SR), "-ac", "1", "pipe:1"],
                       input=mp3, capture_output=True)
    if r.returncode != 0 or not r.stdout:
        raise RuntimeError("ffmpeg could not decode a chunk: " + r.stderr.decode(errors="ignore")[:300])
    return np.frombuffer(r.stdout, dtype=np.int16)


def trim_silence(a, thr=250, keep=0.05):
    """Cut leading/trailing silence so the pause between turns is exactly what you set."""
    idx = np.flatnonzero(np.abs(a) > thr)
    if idx.size == 0:
        return a
    s = max(0, idx[0] - int(SR * keep))
    e = min(len(a), idx[-1] + int(SR * keep))
    return a[s:e]


# --------------------------------------------------------------------------- voices
FALLBACK_VOICES = [
    {"name": n, "gender": g, "locale": n[:5]} for n, g in [
        ("en-US-AndrewMultilingualNeural", "Male"), ("en-US-AvaMultilingualNeural", "Female"),
        ("en-US-BrianMultilingualNeural", "Male"), ("en-US-EmmaMultilingualNeural", "Female"),
        ("en-US-GuyNeural", "Male"), ("en-US-JennyNeural", "Female"),
        ("ar-EG-ShakirNeural", "Male"), ("ar-EG-SalmaNeural", "Female"),
        ("ja-JP-KeitaNeural", "Male"), ("ja-JP-NanamiNeural", "Female")]
]


def _azure_cfg():
    key = os.environ.get("AZURE_SPEECH_KEY", "").strip()
    region = os.environ.get("AZURE_SPEECH_REGION", "").strip()
    if not key or not (region or os.environ.get("AZURE_SPEECH_ENDPOINT", "").strip()):
        raise RuntimeError("Azure Speech is not configured: set AZURE_SPEECH_KEY and AZURE_SPEECH_REGION "
                           "in the server's environment variables.")
    return key, region


def _azure_base(region):
    override = os.environ.get("AZURE_SPEECH_ENDPOINT", "").strip().rstrip("/")
    return override or f"https://{region}.tts.speech.microsoft.com"


def load_voices():
    """Ask Azure which voices this region offers; fall back to a short built-in list."""
    try:
        key, region = _azure_cfg()
        req = urllib.request.Request(_azure_base(region) + "/cognitiveservices/voices/list",
                                     headers={"Ocp-Apim-Subscription-Key": key})
        with urllib.request.urlopen(req, timeout=20) as r:
            vs = json.loads(r.read().decode("utf-8"))
        out = [{"name": v["ShortName"], "gender": v["Gender"], "locale": v["Locale"]} for v in vs]
        if not out:
            raise RuntimeError("empty voice list")
        return sorted(out, key=lambda v: v["name"])
    except Exception as e:
        print(f"[warn] Could not fetch the voice list ({e}). Using a short built-in list.")
        return FALLBACK_VOICES


def default_voices(voices, prefix, n=MAX_SPEAKERS):
    """Pick n different-sounding defaults for a language, alternating male/female when possible."""
    names = [v["name"] for v in voices if v["name"].lower().startswith(prefix.lower())] if prefix else \
            [v["name"] for v in voices]
    picked = [p for p in PREFERRED.get(prefix, []) if p in names]
    picked += [x for x in names if x not in picked]
    return (picked or [v["name"] for v in voices])[:n] + [picked[0] if picked else voices[0]["name"]] * n


# --------------------------------------------------------------------------- text
def clean_text(text):
    text = text.replace("\u2019", "'").replace("\u2018", "'").replace("\u201c", '"').replace("\u201d", '"')
    text = re.sub(r"\[[^\]]*\]", " ", text)          # [stage directions] are not spoken
    text = re.sub(r"[*_#`~^|<>]+", " ", text)         # markdown symbols
    text = re.sub(r"[ \t]+", " ", text)
    return re.sub(r" ([.,!?;:])", r"\1", text).strip()


_LABEL = re.compile(r"^\s*([^\W\d_][\w \-]{0,29}?)\s*(?:\(([^()]{1,30})\))?\s*[:\uff1a]\s*(.*)$", re.UNICODE)


def parse_script(text):
    """-> [[label or None, emotion or None, text], ...]. Lines without a label continue the previous turn."""
    turns = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        m = _LABEL.match(line)
        if m and not re.match(r"^\s*https?$", m.group(1), re.I):
            turns.append([m.group(1).strip(), m.group(2), m.group(3)])
        elif turns:
            turns[-1][2] += "\n" + line
        else:
            turns.append([None, None, line])
    return turns


def resolve_emotion(name):
    key = re.sub(r"\s+", " ", (name or "neutral").strip().lower())
    key = EMOTION_ALIASES.get(key, key)
    if key not in EMOTIONS:
        raise ValueError(f"Unknown emotion '{name}'. Available: {', '.join(EMOTIONS)} (or {{pause 1.5}})")
    return key


def map_labels(turns, slot_names):
    """-> [(slot 0-3, emotion, text)]. Named slots first, then unknown labels in order."""
    norm = lambda s: re.sub(r"\s+", " ", (s or "").strip().lower())
    names = [norm(n) for n in slot_names]
    mapping = {}
    labels = []
    for lab, _, _ in turns:
        if lab is not None and norm(lab) not in labels:
            labels.append(norm(lab))
    for lab in labels:
        for i, n in enumerate(names):
            if n and (lab == n or lab == f"speaker {i + 1}"):
                mapping[lab] = i
                break
    for lab in labels:
        if lab not in mapping:
            free = [i for i in range(MAX_SPEAKERS) if i not in mapping.values()]
            if not free:
                raise ValueError(f"This studio supports {MAX_SPEAKERS} speakers (Host and Guest), but the script uses "
                                 f"{len(labels)} different labels: {', '.join(labels)}")
            mapping[lab] = free[0]
    return [(mapping[norm(lab)] if lab is not None else 0, resolve_emotion(emo), t) for lab, emo, t in turns]


_TAG = re.compile(r"\{([^{}]{1,30})\}")


def split_directives(text, emotion):
    """Split a turn into ('say', emotion, text) and ('pause', seconds, None) items.
    {excited} changes the emotion from that point on; {pause} / {pause 1.5} adds silence."""
    items, pos = [], 0

    def add_text(t):
        if t.strip():
            items.append(("say", emotion, t))

    for m in _TAG.finditer(text):
        add_text(text[pos:m.start()])
        pos = m.end()
        tag = m.group(1).strip().lower()
        pm = re.fullmatch(r"pause(?:\s+(\d+(?:\.\d+)?))?\s*s?", tag)
        if pm:
            items.append(("pause", min(float(pm.group(1) or 0.8), 10.0), None))
        else:
            emotion = resolve_emotion(tag)
    add_text(text[pos:])
    return items


def _num(s):
    return int(re.match(r"[+-]?\d+", s).group())


def voice_params(slot, emotion, strength=1.0):
    """Speaker settings + emotion preset (scaled by strength) -> rate, pitch, volume strings."""
    dr, dp, dv = EMOTIONS[emotion]
    rate = max(-60, min(80, _num(slot["rate"]) + round(dr * strength)))
    pitch = max(-40, min(40, _num(slot["pitch"]) + round(dp * strength)))
    vol = max(-60, min(30, _num(slot.get("volume", "+0%")) + round(dv * strength)))
    return f"{rate:+d}%", f"{pitch:+d}Hz", f"{vol:+d}%"


def _fit(sentence, n):
    if len(sentence) <= n:
        return [sentence]
    out = []
    for piece in re.split(r"(?<=[,;:\u060c\u061b])\s+", sentence):
        if len(piece) <= n:
            out.append(piece)
            continue
        cur = ""
        for w in piece.split():
            if cur and len(cur) + 1 + len(w) > n:
                out.append(cur)
                cur = w
            else:
                cur = f"{cur} {w}".strip()
        if cur:
            out.append(cur)
    return out


def split_chunks(text, max_chars):
    chunks = []
    for para in [p.strip() for p in text.split("\n") if p.strip()]:
        cur = ""
        for sent in re.split(r"(?<=[.!?\u2026\u061f\u3002])\s+", para):
            for part in _fit(sent, max_chars):
                if cur and len(cur) + 1 + len(part) > max_chars:
                    chunks.append(cur)
                    cur = part
                else:
                    cur = f"{cur} {part}".strip()
        if cur:
            chunks.append(cur)
    return [c for c in chunks if re.search(r"\w", c)]


# --------------------------------------------------------------------------- synthesis
class AzureError(RuntimeError):
    """A problem retrying will not fix (bad key, quota used up, unreadable text)."""


_BAD_XML = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")


def build_ssml(text, voice, rate, pitch, volume):
    lang = "-".join(voice.split("-")[:2]) or "en-US"
    attr = {"'": "&apos;", '"': "&quot;"}
    body = _xml_escape(_BAD_XML.sub(" ", text))
    return (f"<speak version='1.0' xmlns='http://www.w3.org/2001/10/synthesis' xml:lang='{_xml_escape(lang, attr)}'>"
            f"<voice name='{_xml_escape(voice, attr)}'>"
            f"<prosody rate='{_xml_escape(rate, attr)}' pitch='{_xml_escape(pitch, attr)}' "
            f"volume='{_xml_escape(volume, attr)}'>{body}</prosody></voice></speak>")


def _azure_synth(ssml, key, region, timeout=60):
    req = urllib.request.Request(
        _azure_base(region) + "/cognitiveservices/v1",
        data=ssml.encode("utf-8"), method="POST",
        headers={"Ocp-Apim-Subscription-Key": key,
                 "Content-Type": "application/ssml+xml",
                 "X-Microsoft-OutputFormat": "audio-24khz-48kbitrate-mono-mp3",
                 "User-Agent": "DuoVoiceStudio"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


async def tts_bytes(text, voice, rate, pitch, volume, retries=6):
    key, region = _azure_cfg()
    ssml = build_ssml(text, voice, rate, pitch, volume)
    last = None
    for attempt in range(retries):
        wait = 2.0 * (2 ** min(attempt, 4)) + random.uniform(0, 1.0)   # exponential backoff + jitter
        try:
            data = await asyncio.to_thread(_azure_synth, ssml, key, region)
            if data:
                return data
            last = RuntimeError("empty audio")
        except urllib.error.HTTPError as e:
            if e.code in (401, 403):
                raise AzureError(f"Azure refused the request (HTTP {e.code}). Check AZURE_SPEECH_KEY and "
                                 f"AZURE_SPEECH_REGION, and that the free monthly quota is not used up.")
            if e.code == 400:
                raise AzureError("Azure could not read part of the text or settings (HTTP 400). "
                                 "Check the voice name and the script.")
            last = e
            ra = e.headers.get("Retry-After") if e.headers else None
            if ra and ra.isdigit():
                wait = max(wait, min(int(ra), 30))
        except Exception as e:
            last = e
        await asyncio.sleep(wait)
    raise RuntimeError(f"Azure TTS failed for voice {voice} after {retries} attempts: {last}")


# --------------------------------------------------------------------------- monthly quota
MONTHLY_CHAR_LIMIT = int(os.environ.get("MONTHLY_CHAR_LIMIT", 450_000))
_USAGE_FILE = os.environ.get("USAGE_FILE", os.path.join(os.path.dirname(os.path.abspath(__file__)), "usage.json"))
_USAGE_LOCK = threading.Lock()


class QuotaError(RuntimeError):
    pass


def _month():
    return time.strftime("%Y-%m", time.gmtime())


def _read_usage():
    try:
        with open(_USAGE_FILE, encoding="utf-8") as f:
            return int(json.load(f).get(_month(), 0))
    except Exception:
        return 0


def usage_status():
    """-> (characters used this month, monthly limit). Best effort: resets if the server's disk is wiped."""
    with _USAGE_LOCK:
        return _read_usage(), MONTHLY_CHAR_LIMIT


def reserve_chars(n):
    with _USAGE_LOCK:
        used = _read_usage()
        if MONTHLY_CHAR_LIMIT > 0 and used + n > MONTHLY_CHAR_LIMIT:
            raise QuotaError(f"This episode needs about {n:,} characters but only "
                             f"{max(MONTHLY_CHAR_LIMIT - used, 0):,} are left in this month's voice quota.")
        try:
            tmp = _USAGE_FILE + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump({_month(): used + n}, f)
            os.replace(tmp, _USAGE_FILE)
        except OSError:
            pass


async def _run_jobs(jobs, slots, ffmpeg, progress, strength=1.0):
    jobs = [j for j in jobs if "pause" not in j]
    # Few parallel connections keep us under the free tier's request-rate limit.
    # Override with the TTS_CONCURRENCY environment variable if needed.
    sem = asyncio.Semaphore(max(1, int(os.environ.get("TTS_CONCURRENCY", "2"))))
    done = 0

    async def one(job):
        nonlocal done
        s = slots[job["slot"]]
        async with sem:
            rate, pitch, vol = voice_params(s, job["emotion"], strength)
            mp3 = await tts_bytes(job["text"], s["voice"], rate, pitch, vol)
            job["pcm"] = trim_silence(await asyncio.to_thread(mp3_to_pcm, mp3, ffmpeg))
        done += 1
        if progress:
            progress((done, len(jobs)), desc=f"Generated {done}/{len(jobs)} chunks")

    await asyncio.gather(*(one(j) for j in jobs))


def srt_time(sec):
    ms = int(round(sec * 1000))
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02}:{m:02}:{s:02},{ms:03}"


def assemble(jobs, turn_gap, jitter, chunk_gap=0.15):
    pieces, cues, total = [], [], 0
    first, after_pause = True, False
    for j in jobs:
        if "pause" in j:                      # explicit {pause}: silence, no extra gap around it
            sil = np.zeros(int(SR * j["pause"]), dtype=np.int16)
            pieces.append(sil)
            total += len(sil)
            after_pause = True
            continue
        if not first and not after_pause:
            gap = turn_gap + random.uniform(-jitter, jitter) if j["new_turn"] else chunk_gap
            sil = np.zeros(int(SR * max(0.08, gap)), dtype=np.int16)
            pieces.append(sil)
            total += len(sil)
        first = after_pause = False
        start = total / SR
        pieces.append(j["pcm"])
        total += len(j["pcm"])
        cues.append((start, total / SR, j["text"]))
    return np.concatenate(pieces), cues


def generate(script, slots, turn_gap=0.45, max_chars=600, make_srt=False, jitter=0.1,
             name="episode", progress=None, emotion_strength=1.0, out_dir=None):
    """slots: list of 4 dicts {name, voice, rate '+0%', pitch '+0Hz'}. Returns (files, status)."""
    ffmpeg = find_ffmpeg()
    turns = map_labels(parse_script(script), [s["name"] for s in slots])
    jobs = []
    for slot, emo, text in turns:
        first = True
        for kind, val, t in split_directives(text, emo):
            if kind == "pause":
                jobs.append({"pause": val})
                continue
            for c in split_chunks(clean_text(t), 140 if make_srt else max_chars):
                jobs.append({"slot": slot, "text": c, "emotion": val, "new_turn": first})
                first = False
    speech = [j for j in jobs if "pause" not in j]
    if not speech:
        raise ValueError("Nothing to read: the script is empty.")

    reserve_chars(sum(len(j["text"]) for j in speech))
    asyncio.run(_run_jobs(jobs, slots, ffmpeg, progress, emotion_strength))
    audio, cues = assemble(jobs, turn_gap, jitter)

    target_dir = out_dir or OUT_DIR
    os.makedirs(target_dir, exist_ok=True)
    base = os.path.join(target_dir, f"{name}_{time.strftime('%Y%m%d_%H%M%S')}")
    with wave.open(base + ".wav", "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(audio.tobytes())
    subprocess.run([ffmpeg, "-y", "-loglevel", "error", "-f", "s16le", "-ar", str(SR), "-ac", "1",
                    "-i", "pipe:0", "-b:a", "192k", base + ".mp3"], input=audio.tobytes(), check=True)
    files = [base + ".mp3", base + ".wav"]
    if make_srt:
        with open(base + ".srt", "w", encoding="utf-8") as f:
            for n, (a, b, t) in enumerate(cues, 1):
                f.write(f"{n}\n{srt_time(a)} --> {srt_time(b)}\n{t}\n\n")
        files.append(base + ".srt")
    used = sorted({j["slot"] for j in speech})
    emos = sorted({j["emotion"] for j in speech} - {"neutral"})
    status = (f"Done: {len(speech)} chunks, {len(audio) / SR / 60:.1f} min, speakers used: "
              + ", ".join(slots[i]["name"] for i in used)
              + (f", emotions used: {', '.join(emos)}" if emos else ""))
    return files, status


# --------------------------------------------------------------------------- helpers
def make_slots(names, voices, rates, pitches):
    return [{"name": (n or "").strip(), "voice": v, "rate": f"{int(r):+d}%", "pitch": f"{int(p):+d}Hz"}
            for n, v, r, p in zip(names, voices, rates, pitches)]
