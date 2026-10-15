#!/usr/bin/env bash
# إعداد كامل لمُصفّي على Oracle Cloud المجاني — يُشغَّل من Cloud Shell داخل موقع Oracle:
#
#   bash <(curl -fsSL https://raw.githubusercontent.com/mswsa99-ship-it/musaffi/main/oracle_setup.sh)
#
# ينشئ: شبكة + جدار حماية يفتح 80/443 + سيرفر Ampere A1 (معالجان، 12 جيجا — ضمن المجاني دائمًا)
# ثم يثبّت الموقع تلقائيًا على السيرفر، ويطبع رابط الموقع.
set -uo pipefail

NAME=musaffi
OCPUS=2
MEM=12
INSTALL_URL="https://raw.githubusercontent.com/mswsa99-ship-it/musaffi/main/install.sh"

say()  { printf '\n\033[1;32m==> %s\033[0m\n' "$*"; }
warn() { printf '\033[1;33m%s\033[0m\n' "$*"; }
die()  { printf '\n\033[1;31m✗ %s\033[0m\n' "$*"; exit 1; }

command -v oci >/dev/null || die "شغّل هذا الأمر من Cloud Shell داخل موقع Oracle (أيقونة >_ أعلى الصفحة)."
T="${OCI_TENANCY:-}"
[ -n "$T" ] || T=$(oci iam compartment list --all --query 'data[0]."compartment-id"' --raw-output 2>/dev/null)
[ -n "$T" ] || die "تعذّر تحديد الحساب. أعد فتح Cloud Shell وحاول مرة ثانية."

q() { oci "$@" --raw-output 2>/dev/null | tr -d '\r'; }

# ---------- 1. الشبكة ----------
say "1/4 تجهيز الشبكة وجدار الحماية"
VCN=$(q network vcn list -c "$T" --display-name "$NAME-vcn" --lifecycle-state AVAILABLE --query 'data[0].id')
if [ -z "$VCN" ] || [ "$VCN" = "null" ]; then
  VCN=$(q network vcn create -c "$T" --cidr-blocks '["10.0.0.0/16"]' --display-name "$NAME-vcn" \
        --dns-label musaffi --wait-for-state AVAILABLE --query 'data.id')
fi
[ -n "$VCN" ] && [ "$VCN" != "null" ] || die "تعذّر إنشاء الشبكة."

IG=$(q network internet-gateway list -c "$T" --vcn-id "$VCN" --query 'data[0].id')
if [ -z "$IG" ] || [ "$IG" = "null" ]; then
  IG=$(q network internet-gateway create -c "$T" --vcn-id "$VCN" --is-enabled true \
       --display-name "$NAME-ig" --wait-for-state AVAILABLE --query 'data.id')
fi

RT=$(q network vcn get --vcn-id "$VCN" --query 'data."default-route-table-id"')
oci network route-table update --rt-id "$RT" --force \
  --route-rules "[{\"destination\":\"0.0.0.0/0\",\"destinationType\":\"CIDR_BLOCK\",\"networkEntityId\":\"$IG\"}]" >/dev/null \
  || die "تعذّر ضبط اتصال الإنترنت للشبكة."

SL=$(q network vcn get --vcn-id "$VCN" --query 'data."default-security-list-id"')
rule() { printf '{"source":"0.0.0.0/0","protocol":"6","isStateless":false,"tcpOptions":{"destinationPortRange":{"min":%s,"max":%s}}}' "$1" "$1"; }
oci network security-list update --security-list-id "$SL" --force \
  --ingress-security-rules "[$(rule 22),$(rule 80),$(rule 443)]" \
  --egress-security-rules '[{"destination":"0.0.0.0/0","protocol":"all","isStateless":false}]' >/dev/null \
  || die "تعذّر فتح منافذ الموقع."

SUB=$(q network subnet list -c "$T" --vcn-id "$VCN" --display-name "$NAME-subnet" --query 'data[0].id')
if [ -z "$SUB" ] || [ "$SUB" = "null" ]; then
  SUB=$(q network subnet create -c "$T" --vcn-id "$VCN" --cidr-block 10.0.0.0/24 \
        --display-name "$NAME-subnet" --dns-label web --wait-for-state AVAILABLE --query 'data.id')
fi
[ -n "$SUB" ] && [ "$SUB" != "null" ] || die "تعذّر إنشاء الشبكة الفرعية."

# ---------- 2. نظام التشغيل ----------
say "2/4 اختيار نظام Ubuntu 24.04"
IMG=$(q compute image list -c "$T" --operating-system "Canonical Ubuntu" --operating-system-version "24.04" \
      --shape VM.Standard.A1.Flex --sort-by TIMECREATED --sort-order DESC --all --query 'data[0].id')
[ -n "$IMG" ] && [ "$IMG" != "null" ] || die "لم أجد نظام Ubuntu 24.04 لهذا النوع من السيرفرات."

[ -f ~/.ssh/id_rsa.pub ] || ssh-keygen -t rsa -b 4096 -N "" -f ~/.ssh/id_rsa -q

USERDATA=$(mktemp)
cat > "$USERDATA" <<EOF
#!/bin/bash
export HOME=/root
curl -fsSL $INSTALL_URL -o /root/musaffi-install.sh
bash /root/musaffi-install.sh
EOF

# ---------- 3. السيرفر ----------
say "3/4 إنشاء السيرفر (معالجان، 12 جيجا — مجاني دائمًا)"
INST=$(q compute instance list -c "$T" --display-name "$NAME" --lifecycle-state RUNNING --query 'data[0].id')
if [ -z "$INST" ] || [ "$INST" = "null" ]; then
  INST=""
  for AD in $(q iam availability-domain list -c "$T" --query 'data[].name' | tr -d '[]",'); do
    echo "   محاولة في المنطقة: $AD"
    OUT=$(oci compute instance launch -c "$T" --availability-domain "$AD" \
          --shape VM.Standard.A1.Flex --shape-config "{\"ocpus\":$OCPUS,\"memoryInGBs\":$MEM}" \
          --image-id "$IMG" --subnet-id "$SUB" --assign-public-ip true --display-name "$NAME" \
          --boot-volume-size-in-gbs 50 --user-data-file "$USERDATA" \
          --ssh-authorized-keys-file ~/.ssh/id_rsa.pub \
          --wait-for-state RUNNING --max-wait-seconds 900 --query 'data.id' --raw-output 2>&1)
    if [[ "$OUT" == ocid1.instance* ]]; then INST="$OUT"; break; fi
    if echo "$OUT" | grep -qi "capacity"; then warn "   لا توجد سعة متاحة هنا حاليًا، أجرّب مكانًا آخر..."; continue; fi
    if echo "$OUT" | grep -qi "limit"; then die "حسابك وصل للحد المجاني. احذف أي سيرفر Ampere آخر ثم أعد المحاولة."; fi
    echo "$OUT" | tail -5
  done
  [ -n "$INST" ] || die "لا توجد سعة مجانية متاحة الآن في منطقتك (وضع شائع في Oracle). أعد تشغيل نفس الأمر بعد ساعة أو ساعتين."
else
  warn "   السيرفر موجود مسبقًا، سأستخدمه."
fi

IP=$(q compute instance list-vnics --instance-id "$INST" --query 'data[0]."public-ip"')
[ -n "$IP" ] && [ "$IP" != "null" ] || die "تعذّر الحصول على عنوان السيرفر."
URL="https://${IP//./-}.sslip.io"

# ---------- 4. التثبيت ----------
say "4/4 السيرفر يعمل الآن ويثبّت الموقع تلقائيًا (20 إلى 40 دقيقة)"
echo "   سأتحقق كل دقيقة، ويمكنك إغلاق هذه النافذة إن أردت — التثبيت يكمل وحده."
for i in $(seq 1 60); do
  if curl -fsS -m 10 "$URL/api/health" >/dev/null 2>&1; then
    printf '\n\033[1;32m%s\033[0m\n' "════════════════════════════════════════════"
    printf '\033[1;32m  موقعك جاهز ✓\033[0m\n  \033[1m%s\033[0m\n' "$URL"
    printf '\033[1;32m%s\033[0m\n' "════════════════════════════════════════════"
    exit 0
  fi
  printf '.'
  sleep 60
done
printf '\n\nالتثبيت ما زال يعمل. افتح هذا الرابط بعد قليل:\n  \033[1m%s\033[0m\n' "$URL"
