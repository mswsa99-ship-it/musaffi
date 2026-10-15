"""اختبار ذاتي: يتأكد أن كل النماذج تعمل محليًا وبدون أي اتصال بالإنترنت.

    python -m app.selftest
"""
from __future__ import annotations

import socket
import tempfile
import time
from pathlib import Path

import numpy as np
import soundfile as sf


def _block_network() -> None:
    def deny(*_a, **_k):
        raise OSError("الاتصال الخارجي ممنوع أثناء الاختبار الذاتي")

    socket.socket.connect = deny  # type: ignore[assignment]
    socket.create_connection = deny  # type: ignore[assignment]


def main() -> None:
    _block_network()
    from . import audio, config, video

    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        t = np.linspace(0, 3, 3 * audio.SR, endpoint=False)
        tone = 0.2 * np.sin(2 * np.pi * 220 * t) + 0.02 * np.random.randn(len(t))
        src = work / "test.wav"
        sf.write(src, np.stack([tone, tone], 1).astype(np.float32), audio.SR)

        t0 = time.time()
        res = audio.run(src, work, False, lambda p, m: None)
        print(f"✓ فصل الصوت يعمل ({time.time() - t0:.0f} ثانية) — المسارات: {', '.join(res)}")

        m = video._load()
        img = np.full((480, 640, 3), 200, np.uint8)
        m["yolo"].predict(img, verbose=False)
        m["det"].detect(img, max_num=0)
        print("✓ نماذج الصورة تعمل")
    print(f"✓ كل النماذج تعمل محليًا من: {config.MODELS_DIR}")


if __name__ == "__main__":
    main()
