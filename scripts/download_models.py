"""يحمّل كل النماذج مرة واحدة إلى مجلد models/ — بعدها يعمل الموقع بدون أي اتصال خارجي.

الاستخدام:
    python scripts/download_models.py          # النماذج الأساسية
    python scripts/download_models.py --hq     # إضافة نموذج الجودة القصوى (640 ميجابايت)
"""
from __future__ import annotations

import io
import shutil
import subprocess
import sys
import tempfile
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODELS = ROOT / "models"

FILES = {
    # اكتشاف وتقسيم الأشخاص (Ultralytics YOLO11، رخصة AGPL-3.0)
    "yolo11s-seg.pt": "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11s-seg.pt",
}
# فصل الصوت البشري عن الموسيقى (MDX-Net وBS-Roformer من مشروع UVR المجاني)
AUDIO_MODELS = ["UVR-MDX-NET-Voc_FT.onnx"]
HQ_AUDIO_MODELS = ["model_bs_roformer_ep_317_sdr_12.9755.ckpt"]


def fetch(url: str, dst: Path) -> None:
    if dst.exists() and dst.stat().st_size > 0:
        print(f"✓ موجود: {dst.relative_to(ROOT)}")
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    print(f"↓ تحميل: {dst.relative_to(ROOT)}")
    tmp = dst.with_suffix(dst.suffix + ".part")
    with urllib.request.urlopen(url) as r, open(tmp, "wb") as fh:
        shutil.copyfileobj(r, fh, 1024 * 1024)
    tmp.rename(dst)


def fetch_insightface() -> None:
    """كاشف الوجوه SCRFD ونموذج الجنس — للاستخدام غير التجاري فقط."""
    out = MODELS / "insightface" / "models" / "buffalo_l"
    if (out / "det_10g.onnx").exists() and (out / "genderage.onnx").exists():
        print("✓ موجود: نماذج الوجوه")
        return
    print("↓ تحميل: نماذج الوجوه (buffalo_l)")
    url = "https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip"
    with urllib.request.urlopen(url) as r:
        z = zipfile.ZipFile(io.BytesIO(r.read()))
    out.mkdir(parents=True, exist_ok=True)
    for name in ("det_10g.onnx", "genderage.onnx"):  # نحتاج نموذجين فقط من الحزمة
        with z.open(name) as src, open(out / name, "wb") as dst:
            shutil.copyfileobj(src, dst)


def fetch_deepfilternet() -> None:
    """DeepFilterNet3 لعزل الكلام (رخصة MIT / Apache-2.0)."""
    out = MODELS / "DeepFilterNet3"
    if (out / "config.ini").exists():
        print("✓ موجود: DeepFilterNet3")
        return
    print("↓ تحميل: DeepFilterNet3")
    with tempfile.TemporaryDirectory() as tmp:
        subprocess.run(["git", "clone", "-q", "--depth", "1", "--filter=blob:none", "--sparse",
                        "https://github.com/Rikorose/DeepFilterNet.git", tmp], check=True)
        subprocess.run(["git", "-C", tmp, "sparse-checkout", "set", "models"], check=True)
        with zipfile.ZipFile(Path(tmp) / "models" / "DeepFilterNet3.zip") as z:
            z.extractall(MODELS)


def fetch_audio_models(names: list[str]) -> None:
    """نماذج فصل الصوت: تتركها مكتبة audio-separator تحمّل النموذج وملفات وصفه مرة واحدة."""
    from audio_separator.separator import Separator

    for name in names:
        print(f"↓ تجهيز: {name}")
        Separator(model_file_dir=str(MODELS / "audio"), log_level=40).load_model(model_filename=name)


def main() -> None:
    sys.path.insert(0, str(ROOT))
    for rel, url in FILES.items():
        fetch(url, MODELS / rel)
    fetch_audio_models(AUDIO_MODELS + (HQ_AUDIO_MODELS if "--hq" in sys.argv else []))
    fetch_insightface()
    fetch_deepfilternet()
    print("\nتم. حجم مجلد النماذج:")
    subprocess.run(["du", "-sh", str(MODELS)])
    print("\nتشغيل تجريبي للنماذج للتأكد أنها تعمل بدون إنترنت...")
    subprocess.run([sys.executable, "-m", "app.selftest"], cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
