"""فصل الصوت إلى أربعة مسارات: كلام، موسيقى، ضوضاء ثابتة، أصوات محيطة وأخرى.

السلسلة (كل النماذج مجانية وتعمل محليًا):
1. MDX-Net / BS-Roformer (audio-separator): يفصل الصوت البشري عن الموسيقى.
2. DeepFilterNet3: يعزل الكلام النظيف من الصوت البشري.
3. الباقي (الصوت البشري - الكلام) يُقسَّم بـ noisereduce إلى ضوضاء ثابتة وأصوات أخرى.

مجموع المسارات الأربعة = الصوت الأصلي تقريبًا، لذلك وضع كل الأشرطة على 100% يعيد الصوت كما هو.
"""
from __future__ import annotations

import subprocess
import sys
import types
from pathlib import Path

import numpy as np
import soundfile as sf

from . import config

SR = 44100
STEMS = ["speech", "music", "noise", "ambient"]

_separators: dict[str, object] = {}
_df = None


def _ffmpeg(*args: str) -> None:
    subprocess.run(["ffmpeg", "-v", "error", "-y", *args], check=True)


def extract_audio(src: Path, dst: Path) -> None:
    _ffmpeg("-i", str(src), "-vn", "-ac", "2", "-ar", str(SR), "-c:a", "pcm_f32le", str(dst))


def _get_separator(model: str, hq: bool):
    key = f"{model}|{hq}"
    if key not in _separators:
        from audio_separator.separator import Separator

        sep = Separator(
            model_file_dir=str(config.MODELS_DIR / "audio"),
            output_dir=None,
            output_format="WAV",
            normalization_threshold=1.0,
            output_single_stem="Vocals",
            # الجودة القصوى: تداخل أكبر + إزالة تشويش النموذج (ضعف الوقت تقريبًا)
            mdx_params={"hop_length": 1024, "segment_size": 256, "overlap": 0.5 if hq else 0.25,
                        "batch_size": 1, "enable_denoise": hq},
            mdxc_params={"segment_size": 256, "override_model_segment_size": False,
                         "batch_size": 1, "overlap": 4, "pitch_shift": 0},
        )
        sep.load_model(model_filename=model)
        _separators[key] = sep
    return _separators[key]


def _get_df():
    """DeepFilterNet3 — مع تجاوز استيراد قديم في مكتبة torchaudio."""
    global _df
    if _df is None:
        if "torchaudio.backend.common" not in sys.modules:
            common = types.ModuleType("torchaudio.backend.common")
            common.AudioMetaData = object
            sys.modules.setdefault("torchaudio.backend", types.ModuleType("torchaudio.backend"))
            sys.modules["torchaudio.backend.common"] = common
        from df.enhance import enhance, init_df

        model, state, _ = init_df(str(config.MODELS_DIR / "DeepFilterNet3"), log_level="ERROR")
        _df = (model, state, enhance)
    return _df


def _resample(x: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    if sr_in == sr_out:
        return x
    import soxr

    return soxr.resample(x, sr_in, sr_out, quality="VHQ").astype(np.float32)


def _match_len(x: np.ndarray, n: int) -> np.ndarray:
    if len(x) >= n:
        return x[:n]
    return np.pad(x, ((0, n - len(x)), (0, 0)))


def separate_vocals(mix_path: Path, work: Path, hq: bool) -> np.ndarray:
    sep = _get_separator(config.VOCAL_MODEL_HQ if hq else config.VOCAL_MODEL_FAST, hq)
    sep.output_dir = str(work)
    sep.model_instance.output_dir = str(work)
    out = sep.separate(str(mix_path), {"Vocals": "vocals_raw"})
    path = work / Path(out[0]).name
    voc, sr = sf.read(path, dtype="float32", always_2d=True)
    path.unlink(missing_ok=True)
    # إصلاح خلل في audio-separator: يضرب الناتج في أعلى قيمة للصوت حتى لو لم يُطبَّع الصوت،
    # فيخرج الصوت البشري أخفض من حقيقته ويتسرب جزء منه إلى مسار الموسيقى.
    mix, _ = sf.read(mix_path, dtype="float32", always_2d=True)
    peak = float(np.abs(mix).max())
    if 0 < peak <= 1.0:
        voc = voc / peak
    return _resample(voc, sr, SR)


def enhance_speech(vocals: np.ndarray) -> np.ndarray:
    import torch

    try:
        model, state, enhance = _get_df()
    except ImportError:
        # احتياط إذا تعذّر تثبيت DeepFilterNet على السيرفر: تنظيف أبسط بخوارزمية noisereduce
        import noisereduce as nr

        out = nr.reduce_noise(y=vocals.T, sr=SR, stationary=False, prop_decrease=0.9, n_jobs=1)
        return _match_len(out.T.astype(np.float32), len(vocals))
    df_sr = state.sr()
    x = _resample(vocals, SR, df_sr)
    # المعالجة على أجزاء (30 ثانية) مع تداخل ودمج ناعم، لتقليل استهلاك الذاكرة في المقاطع الطويلة
    chunk, ov = 30 * df_sr, df_sr // 2
    y = np.zeros_like(x)
    weight = np.zeros((len(x), 1), np.float32)
    start = 0
    while start < len(x):
        end = min(len(x), start + chunk + ov)
        with torch.no_grad():
            part = enhance(model, state, torch.from_numpy(np.ascontiguousarray(x[start:end].T)))
        part = _match_len(part.numpy().T.astype(np.float32), end - start)
        w = np.ones((end - start, 1), np.float32)
        fade = min(ov, end - start)
        if start > 0:
            w[:fade, 0] = np.linspace(0, 1, fade)
        if end < len(x):
            w[-fade:, 0] = np.minimum(w[-fade:, 0], np.linspace(1, 0, fade))
        y[start:end] += part * w
        weight[start:end] += w
        if end >= len(x):
            break
        start += chunk
    y /= np.maximum(weight, 1e-6)
    return _match_len(_resample(y, df_sr, SR), len(vocals))


def split_noise(residual: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """يفصل الضوضاء الثابتة (طنين، مكيف، رياح) عن بقية الأصوات المتغيرة."""
    import noisereduce as nr

    cleaned = nr.reduce_noise(y=residual.T, sr=SR, stationary=True, prop_decrease=1.0,
                              n_jobs=1).T.astype(np.float32)
    cleaned = _match_len(cleaned, len(residual))
    noise = residual - cleaned
    return noise, cleaned


def _rms_db(x: np.ndarray) -> float:
    r = float(np.sqrt(np.mean(np.square(x)) + 1e-12))
    return round(20 * np.log10(r + 1e-12), 1)


def _envelope(x: np.ndarray, bins: int = 400) -> list[float]:
    mono = np.abs(x).mean(axis=1)
    n = max(1, len(mono) // bins)
    m = mono[: n * bins].reshape(-1, n)
    env = np.sqrt(np.mean(np.square(m), axis=1))
    return [round(float(v), 5) for v in env]


def analyze(stems: dict[str, np.ndarray]) -> dict:
    """مؤشرات القوة: مستوى كل مسار بالديسيبل، ونسبته من الطاقة الكلية، ومنحنى زمني."""
    energy = {k: float(np.sum(np.square(v))) for k, v in stems.items()}
    total = sum(energy.values()) or 1.0
    peak = max(max(_envelope(v)) for v in stems.values()) or 1.0
    return {
        k: {
            "db": _rms_db(v),
            "share": round(100 * energy[k] / total, 1),
            "envelope": [round(e / peak, 4) for e in _envelope(v)],
        }
        for k, v in stems.items()
    }


def run(src: Path, work: Path, hq: bool, progress) -> dict:
    """يشغّل السلسلة كاملة ويحفظ المسارات في work/stems."""
    stems_dir = work / "stems"
    stems_dir.mkdir(exist_ok=True)
    mix_path = work / "mix.wav"
    extract_audio(src, mix_path)
    mix, _ = sf.read(mix_path, dtype="float32", always_2d=True)

    progress(5, "فصل الصوت البشري عن الموسيقى")
    vocals = _match_len(separate_vocals(mix_path, work, hq), len(mix))
    music = mix - vocals

    progress(60, "عزل الكلام عن بقية الأصوات")
    speech = enhance_speech(vocals)
    residual = vocals - speech

    progress(80, "فصل الضوضاء عن الأصوات المحيطة")
    noise, ambient = split_noise(residual)

    stems = {"speech": speech, "music": music, "noise": noise, "ambient": ambient}
    for name, data in stems.items():
        sf.write(stems_dir / f"{name}.wav", data, SR, subtype="FLOAT")
        # نسخة مضغوطة صغيرة للمعاينة داخل المتصفح
        _ffmpeg("-i", str(stems_dir / f"{name}.wav"), "-c:a", "libopus", "-b:a", "96k",
                str(stems_dir / f"{name}.webm"))
    mix_path.unlink(missing_ok=True)
    progress(95, "حساب مؤشرات القوة")
    return analyze(stems)


def render_mix(work: Path, gains: dict[str, float], dst: Path) -> None:
    """يمزج المسارات بالقوة التي اختارها المستخدم (0 إلى 1.5)."""
    out = None
    for name in STEMS:
        g = float(gains.get(name, 0.0))
        if g <= 0:
            continue
        data, _ = sf.read(work / "stems" / f"{name}.wav", dtype="float32", always_2d=True)
        out = data * g if out is None else out + data * g
    if out is None:
        data, _ = sf.read(work / "stems" / "speech.wav", dtype="float32", always_2d=True)
        out = np.zeros_like(data)
    peak = float(np.abs(out).max())
    if peak > 0.99:  # منع التشويه عند رفع الأشرطة فوق 100%
        out *= 0.99 / peak
    sf.write(dst, out, SR, subtype="PCM_24")
