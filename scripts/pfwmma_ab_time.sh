#!/bin/bash
# pfwmma_ab_time.sh <0|1|2> : 同 pfwmma_ab.sh, 但额外开 STRATA_PREFILL_TIMING=1 相位表
# 注意: 重启后第一次请求含内核模块冷加载, 必须先发一个 warmup 再读相位表
set -u
VAL="${1:?usage: pfwmma_ab_time.sh <0|1|2>}"
: "${STRATA_KEY:?STRATA_KEY env required (this repo ships no default key)}"
KEY="$STRATA_KEY"
echo "--- arm STRATA_PF_WMMA=$VAL (+PREFILL_TIMING) ---"
printf '[Service]\nEnvironment="STRATA_PF_WMMA=%s"\nEnvironment="STRATA_PREFILL_TIMING=1"\n' "$VAL" | sudo -n tee /etc/systemd/system/strata-sc117.service.d/zz-pfwmma.conf > /dev/null
sudo -n systemctl daemon-reload
sudo -n systemctl restart strata-sc117
for i in $(seq 1 60); do
  sleep 5
  code=$(curl -s -o /dev/null -w '%{http_code}' -m 5 -H "Authorization: Bearer $KEY" http://127.0.0.1:8080/v1/models 2>/dev/null)
  if [ "$code" = "200" ]; then
    echo "READY after ~$((i*5))s (STRATA_PF_WMMA=$VAL)"
    exit 0
  fi
done
echo "TIMEOUT waiting for engine"
exit 1
