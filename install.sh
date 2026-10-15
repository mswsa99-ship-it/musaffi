#!/usr/bin/env bash
# مثبّت مُصفّي — أمر واحد يجهّز كل شيء على سيرفر Ubuntu جديد (يدعم ARM وx86):
#
#   curl -fsSL https://raw.githubusercontent.com/mswsa99-ship-it/musaffi/main/install.sh | sudo bash
#
# ما يفعله:
#   1. يثبّت الأدوات (Python, ffmpeg, Caddy) ويضيف ذاكرة احتياطية.
#   2. يفتح منفذي 80 و443 في جدار حماية السيرفر.
#   3. يثبّت الموقع والنماذج في /opt/musaffi ويشغّل اختبارًا ذاتيًا بدون إنترنت.
#   4. يشغّل الموقع كخدمة تعمل تلقائيًا بعد إعادة التشغيل.
#   5. يجهّز رابط HTTPS مجاني وآمن تلقائيًا (عبر sslip.io + Let's Encrypt).
# يمكن تشغيله أكثر من مرة لتحديث الموقع.
set -euo pipefail

REPO="${MUSAFFI_REPO:-https://github.com/mswsa99-ship-it/musaffi.git}"
APP=/opt/musaffi
DATA=/var/lib/musaffi
LOG=/var/log/musaffi-install.log

say()  { printf '\n\033[1;32m==> %s\033[0m\n' "$*"; }
fail() { printf '\n\033[1;31m✗ %s\033[0m\nالتفاصيل في %s\n' "$*" "$LOG"; exit 1; }
trap 'fail "توقف التثبيت عند السطر $LINENO"' ERR

[ "$(id -u)" = 0 ] || { echo "شغّل الأمر مع sudo"; exit 1; }
export HOME="${HOME:-/root}"
# عند أول تشغيل للسيرفر قد يكون نظام التحديث التلقائي شغالًا؛ ننتظره بدل الفشل
echo 'DPkg::Lock::Timeout "900";' > /etc/apt/apt.conf.d/90musaffi-lock-wait
exec > >(tee -a "$LOG") 2>&1
ARCH=$(uname -m)
say "بدء تثبيت مُصفّي على معالج $ARCH — يستغرق 20 إلى 40 دقيقة، لا تغلق النافذة"

# ---------- 1. الأدوات ----------
say "1/6 تثبيت الأدوات الأساسية"
export DEBIAN_FRONTEND=noninteractive
apt-get update -y
apt-get install -y python3 python3-venv python3-dev build-essential ffmpeg git curl \
    libgl1 libglib2.0-0 libsndfile1 caddy iptables-persistent pkg-config

if ! swapon --show | grep -q /swapfile; then
  say "إضافة ذاكرة احتياطية 4 جيجابايت (تمنع توقف المعالجة في المقاطع الطويلة)"
  fallocate -l 4G /swapfile && chmod 600 /swapfile && mkswap /swapfile && swapon /swapfile
  grep -q /swapfile /etc/fstab || echo '/swapfile none swap sw 0 0' >> /etc/fstab
fi

# ---------- 2. جدار الحماية ----------
say "2/6 فتح منفذي الموقع 80 و443"
for p in 80 443; do
  iptables -C INPUT -p tcp --dport $p -j ACCEPT 2>/dev/null || iptables -I INPUT 1 -p tcp --dport $p -j ACCEPT
done
netfilter-persistent save || true

# ---------- 3. الكود ----------
say "3/6 تحميل كود الموقع"
id musaffi >/dev/null 2>&1 || useradd --system --create-home --home-dir /home/musaffi musaffi
if [ -d "$APP/.git" ]; then git -C "$APP" pull --ff-only; else git clone --depth 1 "$REPO" "$APP"; fi

# ---------- 4. المكتبات ----------
say "4/6 تثبيت مكتبات الذكاء الاصطناعي (أطول خطوة)"
[ -d "$APP/.venv" ] || python3 -m venv "$APP/.venv"
PIP="$APP/.venv/bin/pip"
"$PIP" install -q --upgrade pip wheel setuptools
if [ "$ARCH" = "x86_64" ]; then
  "$PIP" install -q torch torchaudio --index-url https://download.pytorch.org/whl/cpu
else
  "$PIP" install -q torch torchaudio   # نسخ ARM على PyPI تعمل على المعالج مباشرة
fi
grep -v -i '^DeepFilterLib' "$APP/requirements.txt" > /tmp/musaffi-req.txt
"$PIP" install -q -r /tmp/musaffi-req.txt
"$PIP" uninstall -y -q opencv-python 2>/dev/null || true
"$PIP" install -q --force-reinstall --no-deps opencv-python-headless

# DeepFilterNet (عزل الكلام): على ARM لا توجد نسخة جاهزة فنبنيها بلغة Rust
if ! "$PIP" install -q --only-binary=:all: DeepFilterLib==0.5.6 2>/dev/null; then
  say "بناء مكتبة عزل الكلام من المصدر (قد يستغرق 10 دقائق)"
  if [ ! -x /root/.cargo/bin/cargo ]; then
    curl -fsSL https://sh.rustup.rs | sh -s -- -y --profile minimal
  fi
  export PATH="/root/.cargo/bin:$PATH"
  "$PIP" install -q maturin
  "$PIP" install -q --no-build-isolation DeepFilterLib==0.5.6 \
    || echo "تنبيه: تعذّر بناء DeepFilterNet — سيعمل الموقع بطريقة تنظيف بديلة أبسط"
fi
"$PIP" install -q --no-deps deepfilternet==0.5.6 || true

# ---------- 5. النماذج ----------
say "5/6 تحميل النماذج واختبارها بدون إنترنت"
cd "$APP"
YOLO_CONFIG_DIR=/tmp/yolo MPLCONFIGDIR=/tmp/mpl "$APP/.venv/bin/python" scripts/download_models.py
mkdir -p "$DATA" && chown -R musaffi:musaffi "$DATA" "$APP/models"

cat > /etc/systemd/system/musaffi.service <<EOF
[Unit]
Description=Musaffi
After=network.target

[Service]
User=musaffi
WorkingDirectory=$APP
Environment=MUSAFFI_DATA_DIR=$DATA
Environment=YOLO_OFFLINE=1
Environment=HF_HUB_OFFLINE=1
Environment=YOLO_CONFIG_DIR=/tmp/musaffi-yolo
Environment=MPLCONFIGDIR=/tmp/musaffi-mpl
ExecStart=$APP/.venv/bin/uvicorn app.main:app --host 127.0.0.1 --port 8000 --workers 1 --timeout-keep-alive 30
Restart=always
RestartSec=5
PrivateTmp=true

[Install]
WantedBy=multi-user.target
EOF
systemctl daemon-reload
systemctl enable --now musaffi
systemctl restart musaffi

# ---------- 6. الرابط الآمن ----------
say "6/6 تجهيز الرابط الآمن (HTTPS)"
IP=$(curl -fsS https://api.ipify.org || curl -fsS https://ifconfig.me)
HOST="${MUSAFFI_DOMAIN:-${IP//./-}.sslip.io}"
cat > /etc/caddy/Caddyfile <<EOF
$HOST {
    request_body {
        max_size 320MB
    }
    reverse_proxy 127.0.0.1:8000 {
        transport http {
            read_timeout 10m
            write_timeout 10m
        }
    }
    # لا نسجّل الطلبات حتى لا تبقى روابط المقاطع في السجلات
    log {
        output discard
    }
}
EOF
systemctl enable caddy
systemctl restart caddy

for i in $(seq 1 30); do
  curl -fsS http://127.0.0.1:8000/api/health >/dev/null 2>&1 && break
  sleep 3
done

trap - ERR
printf '\n\033[1;32m%s\033[0m\n' "════════════════════════════════════════════"
printf '\033[1;32m  تم التثبيت بنجاح ✓\033[0m\n'
printf '  رابط موقعك:  \033[1mhttps://%s\033[0m\n' "$HOST"
printf '  (قد يحتاج الرابط دقيقة أو دقيقتين لتجهيز شهادة الأمان أول مرة)\n'
printf '\033[1;32m%s\033[0m\n' "════════════════════════════════════════════"
