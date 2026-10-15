"""إدارة المهام والحذف التلقائي.

- كل مهمة لها مجلد واحد باسم عشوائي لا يمكن تخمينه، وحذفه يحذف كل شيء يخص المستخدم.
- المعالجة تتم في عملية مستقلة واحدة (worker) بالترتيب، لأن المعالج هو المورد المحدود.
- الحذف يتم في أربع حالات:
  1. فور انتهاء تحميل المستخدم للنتيجة.
  2. عند ضغط المستخدم زر «احذف الآن».
  3. تلقائيًا إذا لم يُحمّل خلال TTL_AFTER_DONE_MIN دقيقة، أو تجاوزت المهمة TTL_MAX_MIN.
  4. عند كل تشغيل للخادم يُمسح مجلد البيانات بالكامل.
"""
from __future__ import annotations

import json
import multiprocessing as mp
import os
import secrets
import shutil
import threading
import time
import traceback
from pathlib import Path

from . import config

STATUS = "status.json"
META = "meta.json"


# ---------- ملفات الحالة ----------

def job_dir(job_id: str) -> Path:
    if not job_id or not all(c.isalnum() or c in "-_" for c in job_id):
        raise ValueError("bad id")
    return config.DATA_DIR / job_id


def new_job() -> tuple[str, Path]:
    jid = secrets.token_urlsafe(18)
    d = config.DATA_DIR / jid
    d.mkdir(parents=True)
    write_status(d, state="uploading", progress=0, message="جارٍ الرفع")
    return jid, d


def _atomic_write(path: Path, data: dict) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False))
    os.replace(tmp, path)


def read_json(d: Path, name: str) -> dict:
    try:
        return json.loads((d / name).read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def write_status(d: Path, **kw) -> None:
    if not d.exists():
        return
    st = read_json(d, STATUS)
    st.setdefault("created", time.time())
    st.update(kw)
    st["updated"] = time.time()
    _atomic_write(d / STATUS, st)


def write_meta(d: Path, meta: dict) -> None:
    _atomic_write(d / META, meta)


def delete_job(d: Path) -> None:
    shutil.rmtree(d, ignore_errors=True)


# ---------- العامل (worker) ----------

def _process(d: Path) -> None:
    from . import audio, video

    meta = read_json(d, META)
    src = d / meta["source"]
    result: dict = {}
    steps = []
    if meta["has_audio"] and meta["separate_audio"]:
        steps.append("audio")
    if meta["has_video"] and meta["blur_mode"] != "none":
        steps.append("video")
    n = len(steps) or 1

    for i, step in enumerate(steps):
        base, span = 100 * i / n, 100 / n

        def prog(p, msg, base=base, span=span):
            if not d.exists():
                raise InterruptedError("deleted")
            write_status(d, state="processing", progress=round(base + span * p / 100), message=msg)

        if step == "audio":
            result["audio"] = audio.run(src, d, meta["quality"] == "hq", prog)
        else:
            res = video.analyze(src, d, prog)
            res["decisions"] = video.decide(res["tracks"], meta["blur_mode"], meta["sensitivity"])
            result["video"] = res
    write_status(d, state="ready", progress=100, message="جاهز للمعاينة", result=result,
                 done_at=time.time())


def _export(d: Path, opts: dict) -> None:
    from . import audio, video

    meta = read_json(d, META)
    src = d / meta["source"]
    out_dir = d / "out"
    out_dir.mkdir(exist_ok=True)
    fmt = opts["format"]
    base_name = Path(meta["original_name"]).stem[:60] or "musaffi"

    def prog(p, msg):
        if not d.exists():
            raise InterruptedError("deleted")
        write_status(d, state="exporting", progress=p, message=msg)

    has_stems = (d / "stems").exists()
    mixed = d / "mixed.wav"
    if has_stems and fmt in ("video", "audio"):
        prog(2, "مزج المسارات")
        audio.render_mix(d, opts.get("gains", {}), mixed)

    if fmt == "stems":
        prog(10, "تجهيز المسارات المفصولة")
        out = out_dir / f"{base_name}-stems.zip"
        import zipfile
        with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
            for name in audio.STEMS:
                z.write(d / "stems" / f"{name}.wav", f"{name}.wav")
    elif fmt == "audio":
        out = out_dir / f"{base_name}-clean.m4a"
        a_in = mixed if has_stems else src
        _ff("-i", str(a_in), "-vn", "-c:a", "aac", "-b:a", "256k", str(out))
    else:
        out = out_dir / f"{base_name}-clean.mp4"
        vsrc = src
        blur = meta["has_video"] and (d / "tracks.json").exists() and opts.get("blur", True)
        if blur:
            vsrc = d / "blurred.mp4"
            video.render(src, d, opts.get("decisions", {}), opts.get("region", "body"),
                         meta["blur_mode"], meta["sensitivity"], vsrc,
                         lambda p, m: prog(5 + int(p * 0.9), m))
        prog(96, "دمج الصوت والصورة")
        args = ["-i", str(vsrc)]
        if has_stems:
            args += ["-i", str(mixed), "-map", "0:v:0", "-map", "1:a:0"]
        else:
            args += ["-i", str(src), "-map", "0:v:0", "-map", "1:a:0?"]
        vcodec = ["-c:v", "copy"]
        _ff(*args, *vcodec, "-c:a", "aac", "-b:a", "256k", "-shortest", "-movflags", "+faststart",
            str(out))
        if blur:
            vsrc.unlink(missing_ok=True)
    mixed.unlink(missing_ok=True)
    write_status(d, state="exported", progress=100, message="الملف جاهز للتحميل",
                 download=out.name, exported_at=time.time())


def _ff(*args: str) -> None:
    import subprocess
    subprocess.run(["ffmpeg", "-v", "error", "-y", *args], check=True)


def worker_loop(q: mp.Queue) -> None:
    os.environ.setdefault("OMP_NUM_THREADS", str(config.THREADS))
    while True:
        kind, jid, payload = q.get()
        d = config.DATA_DIR / jid
        if not d.exists():
            continue
        try:
            write_status(d, state="processing" if kind == "process" else "exporting",
                         progress=0, message="بدأت المعالجة")
            if kind == "process":
                _process(d)
            else:
                _export(d, payload)
        except InterruptedError:
            delete_job(d)
        except Exception as e:  # noqa: BLE001
            traceback.print_exc()
            write_status(d, state="error", message=f"حدث خطأ أثناء المعالجة: {type(e).__name__}",
                         done_at=time.time())


class JobManager:
    def __init__(self) -> None:
        ctx = mp.get_context("spawn")
        self.q: mp.Queue = ctx.Queue()
        self.pending: list[str] = []
        self.lock = threading.Lock()
        self.worker = ctx.Process(target=worker_loop, args=(self.q,), daemon=True)

    def start(self) -> None:
        # الخصوصية: مسح أي بقايا من تشغيل سابق
        for p in config.DATA_DIR.iterdir():
            shutil.rmtree(p, ignore_errors=True) if p.is_dir() else p.unlink(missing_ok=True)
        self.worker.start()
        threading.Thread(target=self._sweeper, daemon=True).start()

    def submit(self, kind: str, jid: str, payload: dict | None = None) -> None:
        with self.lock:
            self.pending.append(jid)
        write_status(config.DATA_DIR / jid, state="queued", progress=0, message="في الانتظار")
        self.q.put((kind, jid, payload or {}))

    def queue_position(self, jid: str) -> int | None:
        with self.lock:
            self.pending = [j for j in self.pending
                            if read_json(config.DATA_DIR / j, STATUS).get("state") == "queued"]
            return self.pending.index(jid) + 1 if jid in self.pending else None

    def busy(self) -> int:
        with self.lock:
            return len(self.pending)

    def _sweeper(self) -> None:
        while True:
            now = time.time()
            for d in list(config.DATA_DIR.iterdir()):
                if not d.is_dir():
                    continue
                st = read_json(d, STATUS)
                created = st.get("created") or d.stat().st_mtime
                finished = st.get("exported_at") or st.get("done_at")
                expired = now - created > config.TTL_MAX_MIN * 60
                idle = finished and st.get("state") in ("ready", "exported", "downloading", "error") and \
                    now - st.get("updated", finished) > config.TTL_AFTER_DONE_MIN * 60
                if expired or idle:
                    delete_job(d)
            time.sleep(config.SWEEP_EVERY_SEC)
