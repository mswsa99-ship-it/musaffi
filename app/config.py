"""إعدادات المشروع — كلها قابلة للتغيير من متغيرات البيئة."""
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

# مكان ملفات المستخدمين المؤقتة (يُمسح بالكامل عند كل تشغيل للخادم)
DATA_DIR = Path(os.getenv("MUSAFFI_DATA_DIR", BASE_DIR / "data"))
# مكان النماذج (تُحمَّل مرة واحدة بالسكربت scripts/download_models.py)
MODELS_DIR = Path(os.getenv("MUSAFFI_MODELS_DIR", BASE_DIR / "models"))

# حدود لحماية السيرفر
MAX_UPLOAD_MB = int(os.getenv("MUSAFFI_MAX_UPLOAD_MB", "300"))
MAX_DURATION_SEC = int(os.getenv("MUSAFFI_MAX_DURATION_SEC", "900"))  # 15 دقيقة
MAX_QUEUE = int(os.getenv("MUSAFFI_MAX_QUEUE", "10"))

# الحذف التلقائي
# إذا لم يحمّل المستخدم النتيجة خلال هذه المدة بعد انتهاء المعالجة تُحذف الملفات
TTL_AFTER_DONE_MIN = int(os.getenv("MUSAFFI_TTL_AFTER_DONE_MIN", "30"))
# أقصى عمر لأي مهمة مهما كانت حالتها (احتياط)
TTL_MAX_MIN = int(os.getenv("MUSAFFI_TTL_MAX_MIN", "240"))
SWEEP_EVERY_SEC = 60

# نماذج الصوت
# سريع على المعالج العادي: UVR-MDX-NET-Voc_FT.onnx
# أعلى جودة (بطيء جدًا بدون GPU): model_bs_roformer_ep_317_sdr_12.9755.ckpt
VOCAL_MODEL_FAST = os.getenv("MUSAFFI_VOCAL_MODEL_FAST", "UVR-MDX-NET-Voc_FT.onnx")
# الجودة القصوى على المعالج العادي: نفس النموذج بإعدادات أدق (ضعف الوقت).
# إذا توفر كرت شاشة: MUSAFFI_VOCAL_MODEL_HQ=model_bs_roformer_ep_317_sdr_12.9755.ckpt
# (حمّله أولًا: python scripts/download_models.py --hq)
VOCAL_MODEL_HQ = os.getenv("MUSAFFI_VOCAL_MODEL_HQ", "UVR-MDX-NET-Voc_FT.onnx")

# نماذج الصورة
PERSON_SEG_MODEL = os.getenv("MUSAFFI_PERSON_MODEL", "yolo11s-seg.pt")
FACE_PACK = os.getenv("MUSAFFI_FACE_PACK", "buffalo_l")

# عدد أنوية المعالج المستخدمة
THREADS = int(os.getenv("MUSAFFI_THREADS", str(os.cpu_count() or 2)))

DATA_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)
