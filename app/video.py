"""تمويه الأشخاص في الفيديو: الجميع، أو النساء فقط.

المرحلة الأولى (تحليل) — تُحفظ النتائج في tracks.json:
  - YOLO11-seg: يكتشف كل شخص ويرسم حدود جسمه بدقة.
  - ByteTrack: يتتبع كل شخص عبر الإطارات برقم ثابت، فلا يومض التمويه.
  - SCRFD (insightface det_10g): يكتشف الوجوه.
  - genderage (insightface): يقدّر الجنس من الوجه، وتُجمع التقديرات لكل شخص عبر كل ظهوره.
المرحلة الثانية (تصدير): يُطبَّق التمويه حسب قرار كل شخص، ويستطيع المستخدم تعديل القرار قبل التصدير.

قاعدة الحيطة: في وضع «النساء فقط» يُموَّه كل شخص لم يتأكد النظام أنه رجل
(مثل من يظهر من الخلف أو المنقبة التي لا يظهر وجهها).
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

import cv2
import numpy as np

from . import config

FACE_STRIDE = 3        # كشف الوجوه كل 3 إطارات (توفير على المعالج)
HOLD_FRAMES = 8        # إبقاء التمويه بضعة إطارات إذا فقد النظام الشخص لحظيًا
MIN_FACE_PX = 20       # الوجوه الأصغر من هذا لا تُستخدم لتقدير الجنس

_models: dict = {}


def _load():
    if not _models:
        import torch
        from insightface.model_zoo import get_model
        from ultralytics import YOLO

        torch.set_num_threads(config.THREADS)
        cv2.setNumThreads(config.THREADS)
        _models["yolo"] = YOLO(str(config.MODELS_DIR / config.PERSON_SEG_MODEL))
        face_dir = config.MODELS_DIR / "insightface" / "models" / config.FACE_PACK
        det = get_model(str(face_dir / "det_10g.onnx"), providers=["CPUExecutionProvider"])
        det.prepare(ctx_id=-1, input_size=(640, 640), det_thresh=0.45)
        ga = get_model(str(face_dir / "genderage.onnx"), providers=["CPUExecutionProvider"])
        ga.prepare(ctx_id=-1)
        _models["det"], _models["ga"] = det, ga
    return _models


def _p_female(ga, img, bbox) -> float:
    """احتمال أن الوجه لامرأة (0..1) من مخرجات نموذج genderage."""
    from insightface.utils import face_align

    w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
    center = (bbox[2] + bbox[0]) / 2, (bbox[3] + bbox[1]) / 2
    scale = ga.input_size[0] / (max(w, h) * 1.5)
    aimg, _ = face_align.transform(img, center, ga.input_size[0], scale, 0)
    blob = cv2.dnn.blobFromImage(aimg, 1.0 / ga.input_std, tuple(aimg.shape[0:2][::-1]),
                                 (ga.input_mean,) * 3, swapRB=True)
    pred = ga.session.run(ga.output_names, {ga.input_name: blob})[0][0]
    z = pred[:2] - pred[:2].max()
    e = np.exp(z)
    return float(e[0] / e.sum())  # الفهرس 0 = أنثى، 1 = ذكر


def probe(src: Path) -> dict:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
         "stream=width,height,r_frame_rate,nb_frames:format=duration", "-of", "json", str(src)],
        capture_output=True, text=True, check=True).stdout
    d = json.loads(out)
    if not d.get("streams"):
        return {}
    s = d["streams"][0]
    num, den = s["r_frame_rate"].split("/")
    fps = float(num) / float(den or 1)
    dur = float(d.get("format", {}).get("duration", 0))
    return {"width": s["width"], "height": s["height"], "fps": fps,
            "frames": int(s.get("nb_frames") or round(fps * dur))}


def _assign_face(face_box, people):
    """يربط الوجه بالشخص الذي يقع مركز الوجه داخل الجزء العلوي من جسمه."""
    cx, cy = (face_box[0] + face_box[2]) / 2, (face_box[1] + face_box[3]) / 2
    best, best_area = None, None
    for tid, (x1, y1, x2, y2) in people.items():
        if x1 <= cx <= x2 and y1 <= cy <= y1 + 0.6 * (y2 - y1):
            area = (x2 - x1) * (y2 - y1)
            if best_area is None or area < best_area:
                best, best_area = tid, area
    return best


def analyze(src: Path, work: Path, progress) -> dict:
    m = _load()
    info = probe(src)
    total = max(1, info["frames"])
    cap = cv2.VideoCapture(str(src))
    thumbs = work / "thumbs"
    thumbs.mkdir(exist_ok=True)

    frames: list[list] = []        # لكل إطار: [[tid, x1,y1,x2,y2, poly], ...]
    face_obs: dict[int, list] = {}  # لكل شخص: [(frame, rel_face_box)]
    gender: dict[int, list] = {}    # لكل شخص: [(p_female, weight)]
    orphans: list[list] = []        # وجوه بلا جسم مكتشف: [frame, x1,y1,x2,y2, p_female]
    best_thumb: dict[int, float] = {}

    m["yolo"].predictor = None  # تصفير التتبع لكل فيديو جديد
    f = 0
    while True:
        ok, img = cap.read()
        if not ok:
            break
        res = m["yolo"].track(img, persist=True, classes=[0], conf=0.25, iou=0.5, imgsz=640,
                              tracker="bytetrack.yaml", verbose=False)[0]
        people, entries = {}, []
        if res.boxes is not None and res.boxes.id is not None:
            ids = res.boxes.id.int().tolist()
            boxes = res.boxes.xyxy.cpu().numpy()
            polys = res.masks.xy if res.masks is not None else [None] * len(ids)
            for tid, b, poly in zip(ids, boxes, polys):
                b = [int(round(v)) for v in b]
                people[tid] = b
                flat = []
                if poly is not None and len(poly) >= 3:
                    p = cv2.approxPolyDP(poly.astype(np.float32).reshape(-1, 1, 2), 1.5, True)
                    flat = p.reshape(-1).round().astype(int).tolist()
                entries.append([tid, *b, flat])
                area = (b[2] - b[0]) * (b[3] - b[1])
                if area > best_thumb.get(tid, 0) * 1.15:
                    best_thumb[tid] = area
                    crop = img[max(0, b[1]):b[3], max(0, b[0]):b[2]]
                    if crop.size:
                        s = 160 / max(crop.shape[:2])
                        cv2.imwrite(str(thumbs / f"{tid}.jpg"),
                                    cv2.resize(crop, None, fx=s, fy=s), [cv2.IMWRITE_JPEG_QUALITY, 80])
        frames.append(entries)

        if f % FACE_STRIDE == 0:
            dets, _ = m["det"].detect(img, max_num=0)
            for d in dets:
                fb = [float(v) for v in d[:4]]
                size = min(fb[2] - fb[0], fb[3] - fb[1])
                pf = _p_female(m["ga"], img, fb) if size >= MIN_FACE_PX else None
                tid = _assign_face(fb, people)
                if tid is None:
                    orphans.append([f, *[int(v) for v in fb], pf])
                    continue
                x1, y1, x2, y2 = people[tid]
                w, h = max(1, x2 - x1), max(1, y2 - y1)
                rel = [(fb[0] - x1) / w, (fb[1] - y1) / h, (fb[2] - x1) / w, (fb[3] - y1) / h]
                face_obs.setdefault(tid, []).append((f, rel))
                if pf is not None:
                    weight = float(d[4]) * min(1.0, size / 60)
                    gender.setdefault(tid, []).append((pf, weight))
        f += 1
        if f % 15 == 0:
            progress(int(100 * f / total), f"تحليل الصورة: الإطار {f} من {total}")
    cap.release()

    tracks = _summarize(frames, face_obs, gender)
    data = {"info": {**info, "frames": len(frames)}, "tracks": tracks,
            "frames": _fill_gaps(frames, face_obs), "orphans": orphans}
    (work / "tracks.json").write_text(json.dumps(data, separators=(",", ":")))
    return {"info": data["info"], "tracks": tracks}


def _summarize(frames, face_obs, gender) -> dict:
    seen: dict[int, list] = {}
    for fi, entries in enumerate(frames):
        for e in entries:
            seen.setdefault(e[0], []).append(fi)
    out = {}
    for tid, fl in seen.items():
        g = gender.get(tid, [])
        wsum = sum(w for _, w in g)
        pf = sum(p * w for p, w in g) / wsum if wsum > 0 else None
        out[str(tid)] = {
            "first": fl[0], "last": fl[-1], "count": len(fl),
            "faces": len(face_obs.get(tid, [])), "votes": len(g),
            "p_female": None if pf is None else round(pf, 3),
        }
    return out


def _fill_gaps(frames, face_obs) -> list:
    """يضيف لكل إطار مكان الوجه، ويسد الفجوات القصيرة حتى لا يظهر الشخص لحظة فقدانه."""
    face_by_tid = {tid: sorted(obs) for tid, obs in face_obs.items()}

    def face_rel(tid, fi):
        obs = face_by_tid.get(tid)
        if not obs:
            return None
        return min(obs, key=lambda o: abs(o[0] - fi))[1]

    by_tid: dict[int, dict[int, list]] = {}
    for fi, entries in enumerate(frames):
        for e in entries:
            by_tid.setdefault(e[0], {})[fi] = e
    out = [[] for _ in frames]
    for tid, fmap in by_tid.items():
        present = sorted(fmap)
        for i, fi in enumerate(present):
            out[fi].append(fmap[fi])
            nxt = present[i + 1] if i + 1 < len(present) else None
            gap_end = min(nxt, fi + HOLD_FRAMES + 1) if nxt is not None and nxt - fi <= HOLD_FRAMES * 3 \
                else min(len(frames), fi + HOLD_FRAMES + 1)
            for g in range(fi + 1, gap_end):
                out[g].append(fmap[fi])
        # إطارات قبل أول ظهور
        for g in range(max(0, present[0] - HOLD_FRAMES), present[0]):
            out[g].append(fmap[present[0]])
    final = []
    for fi, entries in enumerate(out):
        row = []
        for tid, x1, y1, x2, y2, poly in entries:
            rel = face_rel(tid, fi)
            w, h = x2 - x1, y2 - y1
            if rel is not None:
                fb = [x1 + rel[0] * w, y1 + rel[1] * h, x1 + rel[2] * w, y1 + rel[3] * h]
            else:  # تقدير موضع الرأس إذا لم يظهر الوجه أبدًا
                hw = min(w, h * 0.3) * 0.8
                cx = (x1 + x2) / 2
                fb = [cx - hw / 2, y1, cx + hw / 2, y1 + hw * 1.25]
            row.append([tid, x1, y1, x2, y2, [int(round(v)) for v in fb], poly])
        final.append(row)
    return final


def decide(tracks: dict, mode: str, sensitivity: str = "cautious") -> dict[str, bool]:
    """القرار المبدئي لكل شخص: هل يُموَّه؟"""
    if mode == "all":
        return {t: True for t in tracks}
    if mode == "none":
        return {t: False for t in tracks}
    thr = 0.35 if sensitivity == "cautious" else 0.5
    out = {}
    for t, s in tracks.items():
        pf = s["p_female"]
        if pf is None or s["votes"] < 2:
            out[t] = sensitivity == "cautious"  # غير معروف ← يُموَّه احتياطًا
        else:
            out[t] = pf >= thr
    return out


def _blur_image(img: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]
    block = max(8, min(h, w) // 28)
    small = cv2.resize(img, (max(1, w // block), max(1, h // block)), interpolation=cv2.INTER_AREA)
    pix = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
    k = (block * 2) | 1
    return cv2.GaussianBlur(pix, (k, k), 0)


def render(src: Path, work: Path, decisions: dict[str, bool], region: str, mode: str,
           sensitivity: str, dst: Path, progress) -> None:
    """يكتب فيديو جديدًا (بدون صوت) مع تطبيق التمويه."""
    data = json.loads((work / "tracks.json").read_text())
    info = data["info"]
    w, h, fps = info["width"], info["height"], info["fps"]
    blur_ids = {int(t) for t, v in decisions.items() if v}
    thr = 0.35 if sensitivity == "cautious" else 0.5
    orphans_by_f: dict[int, list] = {}
    for o in data["orphans"]:
        fi, x1, y1, x2, y2, pf = o
        if mode == "all" or (mode == "women" and (pf is None or pf >= thr)):
            for g in range(fi, fi + FACE_STRIDE + 2):
                orphans_by_f.setdefault(g, []).append((x1, y1, x2, y2))

    pad = max(4, int(0.012 * max(w, h)))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (pad * 2 + 1, pad * 2 + 1))
    cap = cv2.VideoCapture(str(src))
    enc = subprocess.Popen(
        ["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{w}x{h}",
         "-r", f"{fps}", "-i", "-", "-c:v", "libx264", "-preset", "medium", "-crf", "18",
         "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(dst)],
        stdin=subprocess.PIPE)
    fi = 0
    total = len(data["frames"])
    try:
        while True:
            ok, img = cap.read()
            if not ok:
                break
            if img.shape[1] != w or img.shape[0] != h:
                img = cv2.resize(img, (w, h))
            entries = data["frames"][fi] if fi < total else []
            mask = np.zeros((h, w), np.uint8)
            for tid, x1, y1, x2, y2, fb, poly in entries:
                if tid not in blur_ids:
                    continue
                if region == "face":
                    cx, cy = (fb[0] + fb[2]) / 2, (fb[1] + fb[3]) / 2
                    ax, ay = (fb[2] - fb[0]) * 0.75, (fb[3] - fb[1]) * 0.8
                    cv2.ellipse(mask, (int(cx), int(cy)), (max(2, int(ax)), max(2, int(ay))),
                                0, 0, 360, 255, -1)
                elif len(poly) >= 6:
                    cv2.fillPoly(mask, [np.array(poly, np.int32).reshape(-1, 2)], 255)
                else:
                    cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)
            for x1, y1, x2, y2 in orphans_by_f.get(fi, []):
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                cv2.ellipse(mask, (cx, cy), (int((x2 - x1) * 0.75), int((y2 - y1) * 0.8)),
                            0, 0, 360, 255, -1)
            if mask.any():
                mask = cv2.dilate(mask, kernel)
                soft = cv2.GaussianBlur(mask, (0, 0), pad / 2).astype(np.float32)[..., None] / 255
                soft = np.maximum(soft, (mask[..., None] > 0).astype(np.float32))
                img = (_blur_image(img) * soft + img * (1 - soft)).astype(np.uint8)
            enc.stdin.write(img.tobytes())
            fi += 1
            if fi % 30 == 0:
                progress(int(100 * fi / max(1, total)), f"تطبيق التمويه: الإطار {fi} من {total}")
    finally:
        cap.release()
        enc.stdin.close()
        enc.wait()
    if enc.returncode != 0:
        raise RuntimeError("فشل ترميز الفيديو")
