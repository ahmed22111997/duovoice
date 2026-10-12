#!/usr/bin/env python3
"""
FastAPI website around tts_engine.py (Azure AI Speech).

Run locally:
    pip install -r requirements.txt
    uvicorn app:app --host 0.0.0.0 --port 8000
    open http://127.0.0.1:8000

Deploy: see README.md
"""
import json
import os
import re
import secrets
import shutil
import subprocess
import threading
import time
import traceback
import uuid
import wave

from fastapi import FastAPI, BackgroundTasks, HTTPException, Request, UploadFile, File, Form
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import tts_engine as ptts

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
JOBS_DIR = os.path.join(BASE_DIR, "jobs")
MUSIC_DIR = os.path.join(BASE_DIR, "static", "music")
os.makedirs(JOBS_DIR, exist_ok=True)

app = FastAPI(title="DuoVoice Studio")

# In-memory job store: {job_id: {status, message, files, error, created}}
# status: "queued" -> "running" -> "done" | "error"
JOBS: dict[str, dict] = {}

# Voice list is fetched once at startup (needs internet) and cached in memory.
VOICES: list[dict] = []
VOICE_NAMES: set[str] = set()

# --------------------------------------------------------------------------- limits
MAX_SCRIPT_CHARS = int(os.environ.get("MAX_SCRIPT_CHARS", 20_000))
MAX_SPEAKER_NAME_CHARS = 30
GENERATE_LIMIT_PER_HOUR = int(os.environ.get("GENERATE_LIMIT_PER_HOUR", 12))
MAX_CONCURRENT_JOBS_PER_IP = 1

# --------------------------------------------------------------------------- access password
# Generating audio/video is locked behind a password that lives ONLY in the server's
# environment (APP_PASSWORD on Render), never in the code or the repository.
APP_PASSWORD = os.environ.get("APP_PASSWORD", "").strip()
MAX_BAD_PASSWORDS_PER_HOUR = 8
_AUTH_LOCK = threading.Lock()
_BAD_AUTH: dict[str, list[float]] = {}


def _require_password(request: Request):
    ip = _client_ip(request)
    if not APP_PASSWORD:
        raise HTTPException(503, "Generation is locked: APP_PASSWORD is not set on the server.")
    now = time.time()
    with _AUTH_LOCK:
        bad = [t for t in _BAD_AUTH.get(ip, []) if now - t < 3600]
        _BAD_AUTH[ip] = bad
        if len(bad) >= MAX_BAD_PASSWORDS_PER_HOUR:
            raise HTTPException(429, "Too many wrong passwords. Try again in an hour.")
    given = request.headers.get("x-access-password", "")
    if not secrets.compare_digest(given.encode("utf-8"), APP_PASSWORD.encode("utf-8")):
        with _AUTH_LOCK:
            _BAD_AUTH.setdefault(ip, []).append(now)
        raise HTTPException(401, "Wrong or missing password.")

# {ip: [timestamps of /api/generate calls in the last hour]}
_RATE_LOCK = threading.Lock()
_RATE_HITS: dict[str, list[float]] = {}
# {ip: number of jobs currently queued/running for that ip}
_INFLIGHT_LOCK = threading.Lock()
_INFLIGHT: dict[str, int] = {}


def _client_ip(request: Request) -> str:
    # Render (and most PaaS) sit behind a proxy; the real visitor IP is the
    # first entry of X-Forwarded-For when present.
    fwd = request.headers.get("x-forwarded-for")
    if fwd:
        return fwd.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


def _check_rate_limit(ip: str):
    now = time.time()
    with _RATE_LOCK:
        hits = [t for t in _RATE_HITS.get(ip, []) if now - t < 3600]
        if len(hits) >= GENERATE_LIMIT_PER_HOUR:
            retry_in = int(3600 - (now - hits[0]))
            raise HTTPException(
                429,
                f"Rate limit reached ({GENERATE_LIMIT_PER_HOUR} generations/hour). "
                f"Try again in about {max(retry_in, 1)} seconds.",
            )
        hits.append(now)
        _RATE_HITS[ip] = hits


def _check_inflight(ip: str):
    with _INFLIGHT_LOCK:
        if _INFLIGHT.get(ip, 0) >= MAX_CONCURRENT_JOBS_PER_IP:
            raise HTTPException(429, "You already have a generation in progress. Please wait for it to finish.")
        _INFLIGHT[ip] = _INFLIGHT.get(ip, 0) + 1


def _release_inflight(ip: str):
    with _INFLIGHT_LOCK:
        _INFLIGHT[ip] = max(0, _INFLIGHT.get(ip, 1) - 1)


@app.on_event("startup")
def _load_voices():
    global VOICES, VOICE_NAMES
    VOICES = ptts.load_voices()
    VOICE_NAMES = {v["name"] for v in VOICES}


# --------------------------------------------------------------------------- music library
def _load_music_manifest() -> list[dict]:
    path = os.path.join(MUSIC_DIR, "manifest.json")
    if not os.path.isfile(path):
        return []
    with open(path, encoding="utf-8") as f:
        return json.load(f)


MUSIC_LIBRARY = _load_music_manifest()
MUSIC_BY_ID = {t["id"]: t for t in MUSIC_LIBRARY}


def _wav_duration(path: str) -> float:
    with wave.open(path, "rb") as w:
        return w.getnframes() / w.getframerate()


def _mix_with_music(speech_wav: str, music_mp3: str, out_mp3: str, duration: float,
                    volume_db: float = -18.0, fade: float = 2.0):
    ffmpeg = ptts.find_ffmpeg()
    fade = max(0.1, min(fade, duration / 2))
    filt = (
        f"[1:a]atrim=0:{duration:.3f},"
        f"afade=t=in:st=0:d={fade:.2f},"
        f"afade=t=out:st={max(duration - fade, 0):.3f}:d={fade:.2f},"
        f"volume={volume_db}dB[music];"
        f"[0:a][music]amix=inputs=2:duration=first:dropout_transition=0[out]"
    )
    cmd = [ffmpeg, "-y", "-loglevel", "error",
          "-i", speech_wav,
          "-stream_loop", "-1", "-i", music_mp3,
          "-filter_complex", filt,
          "-map", "[out]", "-b:a", "192k", out_mp3]
    subprocess.run(cmd, check=True)


# --------------------------------------------------------------------------- image -> video
ALLOWED_IMAGE_TYPES = {"image/jpeg": ".jpg", "image/png": ".png", "image/webp": ".webp"}
MAX_IMAGES = 30
MAX_IMAGE_MB = 10
VIDEO_FORMATS = {"mp4", "webm"}
VIDEO_W, VIDEO_H = 1280, 720


def _render_video(image_paths: list[str], audio_path: str, duration: float,
                  out_path: str, srt_path: str | None, fmt: str):
    ffmpeg = ptts.find_ffmpeg()
    base_vf = (f"scale={VIDEO_W}:{VIDEO_H}:force_original_aspect_ratio=decrease,"
              f"pad={VIDEO_W}:{VIDEO_H}:(ow-iw)/2:(oh-ih)/2,format=yuv420p")
    sub_style = "FontName=DejaVu Sans,FontSize=22,BorderStyle=3,Outline=1,Shadow=0"

    codec = {
        "mp4": ["-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k"],
        "webm": ["-c:v", "libvpx-vp9", "-c:a", "libopus", "-b:a", "160k"],
    }[fmt]

    if len(image_paths) == 1:
        vf = base_vf + (f",subtitles={srt_path}:force_style='{sub_style}'" if srt_path else "")
        cmd = [ffmpeg, "-y", "-loglevel", "error", "-loop", "1", "-i", image_paths[0],
              "-i", audio_path, "-vf", vf, *codec, "-r", "15", "-shortest", out_path]
    else:
        # One looped input per image, scaled/padded individually, then joined with the
        # concat *filter* (not the concat demuxer, which chokes when images are in
        # different formats like a mix of JPG and PNG).
        per = duration / len(image_paths)
        inputs = []
        filter_parts = []
        for idx, img in enumerate(image_paths):
            inputs += ["-loop", "1", "-t", f"{per:.3f}", "-i", img]
            filter_parts.append(f"[{idx}:v]{base_vf}[v{idx}]")
        concat_refs = "".join(f"[v{idx}]" for idx in range(len(image_paths)))
        filter_complex = ";".join(filter_parts) + f";{concat_refs}concat=n={len(image_paths)}:v=1:a=0[vcat]"
        if srt_path:
            filter_complex += f";[vcat]subtitles={srt_path}:force_style='{sub_style}'[vout]"
            vmap = "[vout]"
        else:
            vmap = "[vcat]"
        audio_idx = len(image_paths)
        cmd = [ffmpeg, "-y", "-loglevel", "error", *inputs, "-i", audio_path,
              "-filter_complex", filter_complex, "-map", vmap, "-map", f"{audio_idx}:a",
              *codec, "-r", "24", "-shortest", out_path]

    subprocess.run(cmd, check=True)


def _build_video(job_id: str, image_paths: list[str], burn_subtitles: bool, video_format: str):
    job = JOBS[job_id]
    job["video"]["status"] = "running"
    try:
        job_dir = os.path.join(JOBS_DIR, job_id)
        wav_name = next(f for f in job["files"] if f.endswith(".wav"))
        duration = _wav_duration(os.path.join(job_dir, wav_name))

        audio_name = (next((f for f in job["files"] if f.endswith("_with_music.mp3")), None)
                     or next(f for f in job["files"] if f.endswith(".mp3")))
        audio_path = os.path.join(job_dir, audio_name)

        srt_path = None
        if burn_subtitles:
            srt_name = next((f for f in job["files"] if f.endswith(".srt")), None)
            if srt_name:
                srt_path = os.path.join(job_dir, srt_name)

        out_name = f"episode_video.{video_format}"
        out_path = os.path.join(job_dir, out_name)
        _render_video(image_paths, audio_path, duration, out_path, srt_path, video_format)

        job["files"].append(out_name)
        job["video"].update(status="done", file=out_name)
    except Exception as e:
        job["video"].update(status="error", error=str(e))
        traceback.print_exc()


# --------------------------------------------------------------------------- models
class SpeakerIn(BaseModel):
    name: str
    voice: str
    rate: int = 0     # percent, e.g. -10
    pitch: int = 0    # Hz, e.g. -2


class GenerateIn(BaseModel):
    script: str
    speakers: list[SpeakerIn]
    turn_gap: float = 0.45
    max_chars: int = 600
    srt: bool = False
    jitter: bool = True
    emotion_strength: float = 1.0
    name: str = "episode"
    music_track: str | None = None       # id from /api/music, or None for no music
    music_volume_db: float = -18.0       # how far below the speech level the music sits


def _validate_generate(payload: GenerateIn):
    if not payload.script.strip():
        raise HTTPException(400, "Script is empty.")
    if len(payload.script) > MAX_SCRIPT_CHARS:
        raise HTTPException(400, f"Script is too long ({len(payload.script)} chars). Limit is {MAX_SCRIPT_CHARS}.")
    if not (1 <= len(payload.speakers) <= ptts.MAX_SPEAKERS):
        raise HTTPException(400, f"Provide between 1 and {ptts.MAX_SPEAKERS} speakers.")
    for s in payload.speakers:
        if not s.name.strip():
            raise HTTPException(400, "Every speaker needs a name.")
        if len(s.name) > MAX_SPEAKER_NAME_CHARS:
            raise HTTPException(400, f"Speaker name '{s.name[:20]}...' is too long.")
        if VOICE_NAMES and s.voice not in VOICE_NAMES:
            raise HTTPException(400, f"Unknown voice '{s.voice}'.")
        if not (-60 <= s.rate <= 80):
            raise HTTPException(400, "Speed must be between -60 and 80 percent.")
        if not (-40 <= s.pitch <= 40):
            raise HTTPException(400, "Pitch must be between -40 and 40 Hz.")
    if not (0.0 <= payload.turn_gap <= 5.0):
        raise HTTPException(400, "Pause between turns must be between 0 and 5 seconds.")
    if not (100 <= payload.max_chars <= 2000):
        raise HTTPException(400, "Max characters per request must be between 100 and 2000.")
    if not (0.0 <= payload.emotion_strength <= 3.0):
        raise HTTPException(400, "Emotion strength must be between 0 and 3.")
    if not re.match(r"^[\w \-]{1,60}$", payload.name or ""):
        raise HTTPException(400, "Episode name may only contain letters, numbers, spaces, - and _.")
    if payload.music_track is not None and payload.music_track not in MUSIC_BY_ID:
        raise HTTPException(400, f"Unknown music track '{payload.music_track}'.")
    if not (-40.0 <= payload.music_volume_db <= 0.0):
        raise HTTPException(400, "Music volume must be between -40 and 0 dB.")


# --------------------------------------------------------------------------- job worker
def _friendly_error(e: Exception) -> str:
    msg = str(e)
    if "ffmpeg" in msg.lower():
        return "Audio processing failed on the server. Please try again."
    return msg


def _run_job(job_id: str, payload: GenerateIn, ip: str):
    JOBS[job_id]["status"] = "running"
    try:
        slots = ptts.make_slots(
            [s.name for s in payload.speakers],
            [s.voice for s in payload.speakers],
            [s.rate for s in payload.speakers],
            [s.pitch for s in payload.speakers],
        )
        # pad to MAX_SPEAKERS so map_labels() never runs out of slots
        while len(slots) < ptts.MAX_SPEAKERS:
            slots.append({"name": f"Speaker {len(slots) + 1}", "voice": slots[0]["voice"],
                          "rate": "+0%", "pitch": "+0Hz"})

        job_dir = os.path.join(JOBS_DIR, job_id)
        files, status = ptts.generate(
            payload.script, slots,
            turn_gap=payload.turn_gap,
            max_chars=payload.max_chars,
            make_srt=payload.srt,
            jitter=0.1 if payload.jitter else 0.0,
            name=payload.name,
            emotion_strength=payload.emotion_strength,
            out_dir=job_dir,
        )

        if payload.music_track:
            track = MUSIC_BY_ID[payload.music_track]
            wav_path = next(f for f in files if f.endswith(".wav"))
            music_path = os.path.join(MUSIC_DIR, track["file"])
            mixed_path = wav_path.replace(".wav", "_with_music.mp3")
            duration = _wav_duration(wav_path)
            _mix_with_music(wav_path, music_path, mixed_path, duration, payload.music_volume_db)
            files.append(mixed_path)
            status += f", music: {track['name']}"

        JOBS[job_id].update(
            status="done",
            message=status,
            files=[os.path.basename(f) for f in files],
        )
    except Exception as e:
        traceback.print_exc()  # full details stay in the server log
        JOBS[job_id].update(status="error", error=_friendly_error(e))
    finally:
        _release_inflight(ip)


# --------------------------------------------------------------------------- API
@app.get("/api/voices")
def api_voices(lang: str = ""):
    if lang:
        return [v for v in VOICES if v["name"].lower().startswith(lang.lower())]
    return VOICES


@app.get("/api/music")
def api_music():
    return [{"id": t["id"], "name": t["name"], "mood": t["mood"]} for t in MUSIC_LIBRARY]


@app.get("/api/defaults")
def api_defaults():
    return {
        "langs": ptts.LANGS,
        "default_names": ptts.DEFAULT_NAMES,
        "emotions": list(ptts.EMOTIONS.keys()),
        "example_script": ptts.EXAMPLE_SCRIPT,
        "max_speakers": ptts.MAX_SPEAKERS,
        "default_voices_en": ptts.default_voices(VOICES, "en"),
        "max_script_chars": MAX_SCRIPT_CHARS,
        "generate_limit_per_hour": GENERATE_LIMIT_PER_HOUR,
        "quota_used": ptts.usage_status()[0],
        "quota_limit": ptts.usage_status()[1],
    }


@app.post("/api/generate")
def api_generate(payload: GenerateIn, background_tasks: BackgroundTasks, request: Request):
    _require_password(request)
    _validate_generate(payload)
    ip = _client_ip(request)
    _check_inflight(ip)
    try:
        _check_rate_limit(ip)
    except HTTPException:
        _release_inflight(ip)
        raise
    job_id = uuid.uuid4().hex[:12]
    JOBS[job_id] = {"status": "queued", "message": "", "files": [], "error": None, "created": time.time()}
    background_tasks.add_task(_run_job, job_id, payload, ip)
    return {"job_id": job_id}


@app.post("/api/attach-video/{job_id}")
async def api_attach_video(
    job_id: str,
    request: Request,
    background_tasks: BackgroundTasks,
    images: list[UploadFile] = File(...),
    burn_subtitles: bool = Form(False),
    video_format: str = Form("mp4"),
):
    _require_password(request)
    job = JOBS.get(job_id)
    if not job or job["status"] != "done":
        raise HTTPException(404, "Job not found, or the audio isn't ready yet.")
    if video_format not in VIDEO_FORMATS:
        raise HTTPException(400, f"video_format must be one of {sorted(VIDEO_FORMATS)}.")
    if not images:
        raise HTTPException(400, "Upload at least one image.")
    if len(images) > MAX_IMAGES:
        raise HTTPException(400, f"Upload at most {MAX_IMAGES} images.")
    if burn_subtitles and not any(f.endswith(".srt") for f in job["files"]):
        raise HTTPException(
            400,
            "This episode wasn't generated with subtitles. Regenerate the audio with "
            "'Also create subtitles' checked first if you want burned-in captions.",
        )

    job_dir = os.path.join(JOBS_DIR, job_id)
    img_dir = os.path.join(job_dir, "images")
    os.makedirs(img_dir, exist_ok=True)
    saved_paths = []
    for i, up in enumerate(images):
        ext = ALLOWED_IMAGE_TYPES.get(up.content_type)
        if not ext:
            raise HTTPException(400, f"Unsupported image type: {up.content_type}")
        data = await up.read()
        if len(data) > MAX_IMAGE_MB * 1024 * 1024:
            raise HTTPException(400, f"Each image must be under {MAX_IMAGE_MB} MB.")
        path = os.path.join(img_dir, f"{i:03d}{ext}")
        with open(path, "wb") as f:
            f.write(data)
        saved_paths.append(path)

    job["video"] = {"status": "queued", "error": None, "file": None}
    background_tasks.add_task(_build_video, job_id, saved_paths, burn_subtitles, video_format)
    return {"status": "queued"}


@app.get("/api/video-status/{job_id}")
def api_video_status(job_id: str):
    job = JOBS.get(job_id)
    if not job or "video" not in job:
        raise HTTPException(404, "No video job found for this episode yet.")
    return job["video"]


@app.get("/api/status/{job_id}")
def api_status(job_id: str):
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(404, "Unknown job id.")
    return job


@app.get("/api/download/{job_id}/{filename}")
def api_download(job_id: str, filename: str):
    job = JOBS.get(job_id)
    if not job or job["status"] != "done":
        raise HTTPException(404, "File not ready.")
    if filename not in job["files"]:
        raise HTTPException(404, "Unknown file.")
    path = os.path.join(JOBS_DIR, job_id, filename)
    if not os.path.isfile(path):
        raise HTTPException(404, "File missing on disk.")
    media = "audio/mpeg" if filename.endswith(".mp3") else \
            "audio/wav" if filename.endswith(".wav") else \
            "video/mp4" if filename.endswith(".mp4") else \
            "video/webm" if filename.endswith(".webm") else "text/plain"
    return FileResponse(path, media_type=media, filename=filename)


# Housekeeping: delete job folders older than 2 hours so a free-tier disk doesn't fill up.
@app.on_event("startup")
def _cleanup_old_jobs():
    cutoff = time.time() - 2 * 3600
    if not os.path.isdir(JOBS_DIR):
        return
    for name in os.listdir(JOBS_DIR):
        p = os.path.join(JOBS_DIR, name)
        try:
            if os.path.isdir(p) and os.path.getmtime(p) < cutoff:
                shutil.rmtree(p, ignore_errors=True)
        except OSError:
            pass


# --------------------------------------------------------------------------- site files
def _site_base(request: Request) -> str:
    env = os.environ.get("SITE_URL", "").strip().rstrip("/")
    if env:
        return env
    proto = request.headers.get("x-forwarded-proto", request.url.scheme).split(",")[0].strip()
    host = (request.headers.get("x-forwarded-host") or request.headers.get("host") or "").split(",")[0].strip()
    return f"{proto}://{host}"


def _site_paths() -> list[str]:
    """URL paths of every public HTML page under static/ (index.html -> '/')."""
    paths = []
    root = os.path.join(BASE_DIR, "static")
    for dirpath, _dirs, names in os.walk(root):
        rel_dir = os.path.relpath(dirpath, root).replace(os.sep, "/")
        if rel_dir.startswith("music"):
            continue
        for n in sorted(names):
            if not n.endswith(".html"):
                continue
            prefix = "" if rel_dir == "." else rel_dir + "/"
            if n == "index.html":
                paths.append("/" + prefix)
            else:
                paths.append("/" + prefix + n)
    return sorted(paths, key=lambda x: (x != "/", x))


@app.get("/health", include_in_schema=False)
def health():
    return {"ok": True}


@app.get("/robots.txt", include_in_schema=False)
def robots(request: Request):
    body = f"User-agent: *\nAllow: /\nDisallow: /api/\n\nSitemap: {_site_base(request)}/sitemap.xml\n"
    return Response(body, media_type="text/plain")


@app.get("/sitemap.xml", include_in_schema=False)
def sitemap(request: Request):
    base = _site_base(request)
    urls = "".join(f"  <url><loc>{base}{p}</loc></url>\n" for p in _site_paths())
    xml = ('<?xml version="1.0" encoding="UTF-8"?>\n'
           '<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">\n' + urls + "</urlset>\n")
    return Response(xml, media_type="application/xml")


# --------------------------------------------------------------------------- frontend
STATIC_DIR = os.path.join(BASE_DIR, "static")
if not os.path.isdir(STATIC_DIR):
    # Guard against a deploy where static/index.html wasn't committed to git
    # (e.g. missing from the repo / excluded by .gitignore) — build a minimal
    # fallback instead of crashing the whole server on startup.
    os.makedirs(STATIC_DIR, exist_ok=True)
    with open(os.path.join(STATIC_DIR, "index.html"), "w", encoding="utf-8") as f:
        f.write(
            "<!doctype html><html><body style='font-family:sans-serif;padding:40px'>"
            "<h2>static/index.html is missing from this deploy</h2>"
            "<p>The API is running (try <code>/api/voices</code>), but the real frontend "
            "file was not found on disk. Make sure <code>static/index.html</code> is "
            "committed to git and re-deploy.</p></body></html>"
        )

app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")
