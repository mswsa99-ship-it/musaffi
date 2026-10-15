"""يبني نسخة المتصفح: صفحة واحدة + ملفات النماذج مقسّمة (كل جزء أقل من 14 ميجابايت).

    python web/build.py --vocft models/audio/UVR-MDX-NET-Voc_FT.onnx --yolo yolo11n-seg.onnx \
        --det det_500m.onnx --ga genderage.onnx --ortwasm ort-wasm-simd-threaded.jsep.wasm
"""
import argparse
import json
from pathlib import Path

HERE = Path(__file__).parent
PART = 14 * 1024 * 1024


def main() -> None:
    ap = argparse.ArgumentParser()
    for k in ("vocft", "yolo", "det", "ga", "ortwasm"):
        ap.add_argument(f"--{k}", required=True)
    ap.add_argument("--out", default=str(HERE / "dist"))
    a = ap.parse_args()
    out = Path(a.out)
    (out / "models").mkdir(parents=True, exist_ok=True)

    manifest = {}
    for name in ("vocft", "yolo", "det", "ga", "ortwasm"):
        data = Path(getattr(a, name)).read_bytes()
        parts = [data[i:i + PART] for i in range(0, len(data), PART)]
        for i, p in enumerate(parts):
            (out / "models" / f"{name}.{i}.bin").write_bytes(p)
        manifest[name] = {"parts": len(parts), "size": len(data)}

    def module(path: str, glob: str) -> str:
        code = (HERE / "src" / path).read_text()
        return f"(function(){{var module={{exports:{{}}}};\n{code}\nwindow.{glob}=module.exports;}})();\n"

    js = (f"window.MUSAFFI_MANIFEST={json.dumps(manifest)};\n"
          + module("dsp.js", "MUSAFFI_DSP") + module("separate.js", "MUSAFFI_SEP")
          + module("vision.js", "MUSAFFI_VISION") + (HERE / "src" / "app.js").read_text())
    html = (HERE / "src" / "index.html").read_text()
    html = html.replace("/*CSS*/", (HERE / "src" / "style.css").read_text())
    html = html.replace("/*JS*/", js.replace("</script", "<\\/script"))
    (out / "index.html").write_text(html)
    print("built", out, {k: v["parts"] for k, v in manifest.items()})


if __name__ == "__main__":
    main()
