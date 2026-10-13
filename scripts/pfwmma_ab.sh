#!/bin/bash
# pfwmma_ab.sh <0|1|2> : 同二进制 A/B — 只切 STRATA_PF_WMMA env, 重启 strata-sc117, 等就绪
#   0 = MMQ (上游默认路径; 生产回滚位, 输出与旧生产二进制逐字节一致)
#   1 = 全 WMMA (上游 sm_70 默认; 我们的包净打平, 见 README 台账)
#   2 = 自研 down-only 混合 (gate/up 走 MMQ, 只 down 上 WMMA; 生产在跑)
set -u
VAL="${1:?usage: pfwmma_ab.sh <0|1|2>}"
: "${STRATA_KEY:?STRATA_KEY env required (this repo ships no default key)}"
KEY="$STRATA_KEY"
echo "--- arm STRATA_PF_WMMA=$VAL ---"
echo "[Service]" | sudo -n tee /etc/systemd/system/strata-sc117.service.d/zz-pfwmma.conf > /dev/null
echo "Environment=\"STRATA_PF_WMMA=$VAL\"" | sudo -n tee -a /etc/systemd/system/strata-sc117.service.d/zz-pfwmma.conf > /dev/null
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
