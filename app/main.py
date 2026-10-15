"""خادم الويب — كل المعالجة تتم على هذا السيرفر، بدون أي خدمة خارجية."""
from __future__ import annotations

import json
import subprocess
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from starlette.background import BackgroundTask

from . import config, jobs

manager = jobs.JobManager()

ALLOWED_EXT = {".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v", ".3gp",
               ".mp3", ".m4a", ".wav", ".aac", ".ogg", ".opus", ".flac"}


@asynccontextmanager
async def lifespan(_app):
    manager.start()
    yield


app = FastAPI(title="مُصفّي", lifespan=lifespan, docs_url=None, redoc_url=None)


@app.middleware("http")
async def security_headers(request: Request, call_next):
    resp = await call_next(request)
    resp.headers["X-Content-Type-Options"] = "nosniff"
    resp.headers["Referrer-Policy"] = "no-referrer"
    resp.headers["X-Frame-Options"] = "DENY"
    # لا تحميل لأي مورد من خارج الموقع
    resp.headers["Content-Security-Policy"] = (
        "default-src 'self'; img-src 'self' blob: data:; media-src 'self' blob:; "
        "style-src 'self' 'unsafe-inline'; script-src 'self'; connect-src 'self'")
    if request.url.path.startswith("/api/"):
        resp.headers["Cache-Control"] = "no-store"
    return resp


def _dir(job_id: str) -> Path:
    try:
        d = jobs.job_dir(job_id)
    except ValueError:
        raise HTTPException(404, "غير موجود")
    if not d.exists():
        raise HTTPException(404, "انتهت صلاحية الملف أو تم حذفه")
    return d


def _probe(path: Path) -> dict:
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries",
                          "stream=codec_type:format=duration", "-of", "json", str(path)],
                         capture_output=True, text=True)
    if out.returncode != 0:
        return {}
    d = json.loads(out.stdout or "{}")
    types = {s.get("codec_type") for s in d.get("streams", [])}
    return {"duration": float(d.get("format", {}).get("duration") or 0),
            "has_audio": "audio" in types, "has_video": "video" in types}


@app.get("/api/limits")
def limits():
    return {"max_upload_mb": config.MAX_UPLOAD_MB, "max_duration_min": config.MAX_DURATION_SEC // 60,
            "ttl_min": config.TTL_AFTER_DONE_MIN, "queue": manager.busy()}


@app.post("/api/jobs")
async def create_job(
    file: UploadFile = File(...),
    separate_audio: bool = Form(True),
    blur_mode: str = Form("none"),          # none | women | all
    sensitivity: str = Form("cautious"),    # cautious | balanced
    quality: str = Form("fast"),            # fast | hq
):
    if blur_mode not in ("none", "women", "all") or sensitivity not in ("cautious", "balanced") \
            or quality not in ("fast", "hq"):
        raise HTTPException(400, "خيارات غير صحيحة")
    ext = Path(file.filename or "").suffix.lower()
    if ext not in ALLOWED_EXT:
        raise HTTPException(400, "صيغة الملف غير مدعومة")
    if manager.busy() >= config.MAX_QUEUE:
        raise HTTPException(503, "الخادم مشغول حاليًا، حاول بعد قليل")

    jid, d = jobs.new_job()
    src = d / f"source{ext}"
    limit = config.MAX_UPLOAD_MB * 1024 * 1024
    size = 0
    try:
        with open(src, "wb") as fh:
            while chunk := await file.read(1024 * 1024):
                size += len(chunk)
                if size > limit:
                    raise HTTPException(413, f"حجم الملف أكبر من {config.MAX_UPLOAD_MB} ميجابايت")
                fh.write(chunk)
        info = _probe(src)
        if not info or not (info["has_audio"] or info["has_video"]):
            raise HTTPException(400, "تعذّر قراءة الملف")
        if info["duration"] > config.MAX_DURATION_SEC:
            raise HTTPException(400, f"مدة المقطع أطول من {config.MAX_DURATION_SEC // 60} دقيقة")
        if not (separate_audio and info["has_audio"]) and not (blur_mode != "none" and info["has_video"]):
            raise HTTPException(400, "لا يوجد ما يمكن معالجته بهذه الخيارات")
    except HTTPException:
        jobs.delete_job(d)
        raise
    except Exception:
        jobs.delete_job(d)
        raise HTTPException(400, "تعذّر رفع الملف")

    jobs.write_meta(d, {
        "source": src.name, "original_name": Path(file.filename or "file").name,
        "separate_audio": separate_audio, "blur_mode": blur_mode, "sensitivity": sensitivity,
        "quality": quality, **info,
    })
    manager.submit("process", jid)
    return {"id": jid}


@app.get("/api/jobs/{job_id}")
def job_status(job_id: str):
    d = _dir(job_id)
    st = jobs.read_json(d, jobs.STATUS)
    meta = jobs.read_json(d, jobs.META)
    if st.get("state") in ("ready", "exported"):
        jobs.write_status(d)  # المستخدم ما زال على الصفحة: تمديد المهلة
    st["queue_position"] = manager.queue_position(job_id) if st.get("state") == "queued" else None
    st["meta"] = {k: meta.get(k) for k in ("has_audio", "has_video", "blur_mode", "sensitivity",
                                              "separate_audio", "duration", "original_name")}
    return st


@app.get("/api/jobs/{job_id}/stem/{name}")
def stem(job_id: str, name: str):
    if name not in ("speech", "music", "noise", "ambient"):
        raise HTTPException(404)
    p = _dir(job_id) / "stems" / f"{name}.webm"
    if not p.exists():
        raise HTTPException(404)
    return FileResponse(p, media_type="audio/webm")


@app.get("/api/jobs/{job_id}/source")
def source(job_id: str):
    d = _dir(job_id)
    preview = d / "preview.mp4"
    if preview.exists():
        return FileResponse(preview, media_type="video/mp4")
    meta = jobs.read_json(d, jobs.META)
    return FileResponse(d / meta["source"])


@app.get("/api/jobs/{job_id}/tracks")
def tracks(job_id: str):
    p = _dir(job_id) / "tracks.json"
    if not p.exists():
        raise HTTPException(404)
    return FileResponse(p, media_type="application/json")


@app.get("/api/jobs/{job_id}/thumb/{tid}")
def thumb(job_id: str, tid: int):
    p = _dir(job_id) / "thumbs" / f"{int(tid)}.jpg"
    if not p.exists():
        raise HTTPException(404)
    return FileResponse(p, media_type="image/jpeg")


@app.post("/api/jobs/{job_id}/export")
async def export(job_id: str, request: Request):
    d = _dir(job_id)
    st = jobs.read_json(d, jobs.STATUS)
    retry_after_error = st.get("state") == "error" and st.get("result") is not None
    if st.get("state") not in ("ready", "exported") and not retry_after_error:
        raise HTTPException(409, "المعالجة لم تنتهِ بعد")
    body = await request.json()
    fmt = body.get("format", "video")
    if fmt not in ("video", "audio", "stems"):
        raise HTTPException(400)
    gains = {k: max(0.0, min(1.5, float(v))) for k, v in (body.get("gains") or {}).items()
             if k in ("speech", "music", "noise", "ambient")}
    decisions = {str(int(k)): bool(v) for k, v in (body.get("decisions") or {}).items()}
    region = body.get("region", "body") if body.get("region") in ("body", "face") else "body"
    manager.submit("export", job_id, {"format": fmt, "gains": gains, "decisions": decisions,
                                      "region": region, "blur": bool(body.get("blur", True))})
    return {"ok": True}


@app.get("/api/jobs/{job_id}/download")
def download(job_id: str):
    d = _dir(job_id)
    st = jobs.read_json(d, jobs.STATUS)
    name = st.get("download")
    if st.get("state") != "exported" or not name:
        raise HTTPException(409, "الملف غير جاهز")
    path = d / "out" / name
    jobs.write_status(d, state="downloading", message="جارٍ التحميل")
    # الحذف الفعلي لكل ملفات المستخدم فور انتهاء إرسال الملف
    return FileResponse(path, filename=name, background=BackgroundTask(jobs.delete_job, d))


@app.delete("/api/jobs/{job_id}")
def delete(job_id: str):
    try:
        d = jobs.job_dir(job_id)
    except ValueError:
        raise HTTPException(404)
    jobs.delete_job(d)
    return {"deleted": True}


@app.get("/api/health")
def health():
    return {"ok": True}


app.mount("/", StaticFiles(directory=Path(__file__).parent / "static", html=True), name="static")


@app.exception_handler(404)
async def not_found(request: Request, exc):
    if request.url.path.startswith("/api/"):
        detail = getattr(exc, "detail", "غير موجود")
        return JSONResponse({"detail": detail}, status_code=404)
    return Response("Not found", status_code=404)
