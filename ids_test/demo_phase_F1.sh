#!/usr/bin/env bash
# demo_phase_F1.sh
# Phase F.1 — Demonstrate IPS reaction to a controlled "attack" CSV.
#
# What it does:
#   1. Backs up current iptables rules
#   2. Starts ips_hailo.py with --dry-run=false (REAL blocking)
#   3. Builds a CSV with a single flow marked as DDoS from $TARGET_IP
#   4. Appends it to the live_sample.csv that IPS is tailing
#   5. Waits, then checks iptables for the new DROP rule
#   6. Reads back the action log
#
# Usage:
#   bash demo_phase_F1.sh <TARGET_IP>
#
# Default TARGET_IP: 10.99.99.42

set -u
TARGET_IP="${1:-10.99.99.42}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TS="$(date +%Y%m%d_%H%M%S)"
OUT_DIR="$HOME/thesis_screenshots/phase_F"
mkdir -p "$OUT_DIR"
LOG="$OUT_DIR/demo_F1_${TS}.log"

ATTACK_CSV="$SCRIPT_DIR/sample_labeled.csv"   # used for flow template
LIVE_CSV="$SCRIPT_DIR/live_sample.csv"        # IPS tails this
WHITELIST="$SCRIPT_DIR/whitelist.txt"

# === Helpers ===
log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }
hdr() { echo ""; echo "=== $* ==="; } 2>&1 | tee -a "$LOG"

# === Pre-flight ===
log "================================================================"
log "Phase F.1 — IPS reaction demo"
log "================================================================"
log "Target IP (will be blocked):  $TARGET_IP"
log "Whitelist file: $WHITELIST"
log "Log: $LOG"
log "================================================================"

if [[ ! -f "$ATTACK_CSV" ]]; then
    log "[ERROR] Source CSV not found: $ATTACK_CSV"; exit 1
fi
if [[ ! -f "$WHITELIST" ]]; then
    log "[WARN] No whitelist.txt — recommended for safety"
fi

# Refuse to run if TARGET_IP is in your LAN subnet
LOCAL_NET=$(ip route | awk '/proto kernel/ {print $1; exit}')
if [[ -n "$LOCAL_NET" ]] && python3 -c "
import ipaddress, sys
try:
    if ipaddress.ip_address('$TARGET_IP') in ipaddress.ip_network('$LOCAL_NET'):
        sys.exit(0)
    sys.exit(1)
except Exception:
    sys.exit(1)
"; then
    log "[ERROR] $TARGET_IP is INSIDE your LAN ($LOCAL_NET) — refusing to proceed"
    log "        Choose an IP outside your LAN, e.g. 10.99.99.42"
    exit 1
fi
log "[ok] Target IP $TARGET_IP is OUTSIDE LAN $LOCAL_NET — safe to proceed"

# === Step 1: backup iptables ===
hdr "Step 1: Backup current iptables"
sudo iptables-save > "$OUT_DIR/iptables_before_${TS}.txt"
log "Saved current rules to iptables_before_${TS}.txt"
log ""
log "Current INPUT chain (first 10 lines):"
sudo iptables -L INPUT -v -n --line-numbers | head -10 | tee -a "$LOG"

# === Step 2: build a controlled "attack" CSV row ===
hdr "Step 2: Build attack injection CSV"
HEADER=$(head -1 "$ATTACK_CSV")
ATTACK_TEMPLATE=$(grep -m1 ",DDoS$" "$ATTACK_CSV" || grep -m1 ",Bot$" "$ATTACK_CSV" || \
                  grep -m1 ",DoS Hulk$" "$ATTACK_CSV")
if [[ -z "$ATTACK_TEMPLATE" ]]; then
    log "[ERROR] No attack row found in $ATTACK_CSV"; exit 1
fi
log "Found attack template (truncated): ${ATTACK_TEMPLATE:0:100}..."

# Identify Source IP column index (1-based)
SRC_IP_COL=$(echo "$HEADER" | tr ',' '\n' | grep -nxi "source ip" | head -1 | cut -d: -f1)
if [[ -z "$SRC_IP_COL" ]]; then
    log "[ERROR] Couldn't find 'Source IP' column"; exit 1
fi
log "Source IP column index: $SRC_IP_COL"

# Replace the source IP in the template with TARGET_IP
INJECTED=$(echo "$ATTACK_TEMPLATE" | awk -F',' -v col="$SRC_IP_COL" -v ip="$TARGET_IP" \
            'BEGIN {OFS=","} { $col = ip; print }')
log "Injected row source IP: $(echo "$INJECTED" | cut -d',' -f$SRC_IP_COL)"

# Save injection sample
INJECTION_CSV="$OUT_DIR/injection_${TS}.csv"
echo "$HEADER" > "$INJECTION_CSV"
echo "$INJECTED" >> "$INJECTION_CSV"
log "Saved injection sample to $INJECTION_CSV"

# === Step 3: start IPS in background ===
hdr "Step 3: Start ips_hailo.py with REAL blocking enabled"

# Initialize live_sample.csv with the header only
echo "$HEADER" > "$LIVE_CSV"

ALERT_LOG="$OUT_DIR/alerts_F1_${TS}.log"
ACTION_LOG="$OUT_DIR/actions_F1_${TS}.log"

# Activate venv if needed (caller should have done this already)
IPS_CMD=(python3 "$SCRIPT_DIR/ips_hailo.py"
         --input "$LIVE_CSV"
         --hef "$SCRIPT_DIR/ids_mlp_binary_logits.hef"
         --scaler "$SCRIPT_DIR/scaler_params.npz"
         --features "$SCRIPT_DIR/feature_names.npy"
         --alert-log "$ALERT_LOG"
         --action-log "$ACTION_LOG"
         --batch 32
         --poll 1)
if [[ -f "$WHITELIST" ]]; then
    IPS_CMD+=(--whitelist "$WHITELIST")
fi
# CRITICAL: no --dry-run here — we want REAL iptables actions

log "Command: ${IPS_CMD[*]}"
log "[STARTING IPS] You will see its output below (waiting 5s for startup)..."

"${IPS_CMD[@]}" > "$OUT_DIR/ips_F1_${TS}.log" 2>&1 &
IPS_PID=$!
log "IPS PID: $IPS_PID"

sleep 5
if ! kill -0 "$IPS_PID" 2>/dev/null; then
    log "[FATAL] IPS died within 5s. Last 20 lines:"
    tail -20 "$OUT_DIR/ips_F1_${TS}.log" | tee -a "$LOG"
    exit 1
fi
log "[ok] IPS alive after 5s"

# === Step 4: inject the attack ===
hdr "Step 4: Inject the attack row into live CSV"
echo "$INJECTED" >> "$LIVE_CSV"
log "Appended 1 attack row to $LIVE_CSV"
log "Waiting 10s for IPS to detect + react..."
sleep 10

# === Step 5: verify ===
hdr "Step 5: Verify iptables contains a DROP rule for $TARGET_IP"

sudo iptables -L INPUT -v -n --line-numbers > "$OUT_DIR/iptables_after_${TS}.txt"
log ""
log "iptables INPUT chain AFTER injection:"
cat "$OUT_DIR/iptables_after_${TS}.txt" | head -20 | tee -a "$LOG"

if sudo iptables -C INPUT -s "$TARGET_IP" -j DROP 2>/dev/null; then
    log ""
    log "[SUCCESS] ✓ DROP rule for $TARGET_IP IS PRESENT in iptables"
else
    log ""
    log "[FAIL] ✗ No DROP rule for $TARGET_IP found"
fi

# === Step 6: read action log ===
hdr "Step 6: Action log from IPS (proves it knew about the IP)"
if [[ -f "$ACTION_LOG" ]]; then
    log "Last 5 lines of action log:"
    tail -5 "$ACTION_LOG" | tee -a "$LOG"
else
    log "[WARN] no action log produced"
fi

# === Step 7: cleanup ===
hdr "Step 7: Cleanup (stop IPS, remove injected rule)"
kill -TERM "$IPS_PID" 2>/dev/null || true
sleep 2
kill -KILL "$IPS_PID" 2>/dev/null || true
log "IPS stopped"

log ""
log "To manually remove the test DROP rule later, run:"
log "  sudo iptables -D INPUT -s $TARGET_IP -j DROP"
log ""
log "Or to restore original ruleset:"
log "  sudo iptables-restore < $OUT_DIR/iptables_before_${TS}.txt"

log ""
log "================================================================"
log "Demo F.1 complete."
log "Artifacts in: $OUT_DIR"
log "================================================================"
ls -la "$OUT_DIR" | grep "$TS" | tee -a "$LOG"
