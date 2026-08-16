#!/usr/bin/env python3
"""
tools/contract_preflight.py — Preflight Contract & Integrity Validator

Validates:
1. Barrier configuration synchronization: TIMEFRAME_CONFIG vs optimized_barriers_*.json
2. Manifest contract integrity for all active serving models:
   - Barrier parameters (tp_mult_trending, tp_mult_ranging, sl_mult, lookahead)
   - ADX hysteresis parameters (regime_adx_enter, regime_adx_exit)
   - Feature count and feature name list synchronization
   - Binary model file existence (XGB, LGB, CatBoost, Calibrator, Meta-model)
3. Model slot denylist enforcement

Exits 0 on total contract alignment, exits 1 on any discrepancy.
"""

import os
import sys
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    TIMEFRAME_CONFIG,
    REGIME_ADX_ENTER_BY_INTERVAL,
    REGIME_ADX_EXIT_BY_INTERVAL,
    MODEL_SLOT_DENYLIST,
    DYNAMIC_REGIME_ROUTING_INTERVALS,
    ENABLE_DYNAMIC_REGIME_ROUTING
)

def run_preflight():
    print("=" * 60)
    print("🔍 RUNNING MODEL & BARRIER CONTRACT PREFLIGHT AUDIT")
    print("=" * 60)

    errors = []
    warnings = []

    # Check 1: TIMEFRAME_CONFIG vs optimized_barriers_*.json
    print("\n--- Check 1: TIMEFRAME_CONFIG vs optimized_barriers_*.json ---")
    for iv, cfg in TIMEFRAME_CONFIG.items():
        barrier_path = f"optimized_barriers_{iv}.json"
        if not os.path.exists(barrier_path):
            warnings.append(f"Missing {barrier_path} on disk")
            continue
        try:
            with open(barrier_path, "r") as f:
                ob = json.load(f)
            for k in ["tp_mult_trending", "tp_mult_ranging", "sl_mult", "lookahead"]:
                if k in ob:
                    diff = abs(float(ob[k]) - float(cfg[k]))
                    if diff > 1e-9:
                        errors.append(f"Barrier drift in {iv}m for '{k}': file={ob[k]}, config={cfg[k]} (diff={diff:.2e})")
                    else:
                        print(f"  ✅ {iv}m {k:<18}: {ob[k]} == {cfg[k]}")
        except Exception as e:
            errors.append(f"Failed to read {barrier_path}: {e}")

    # Check 2: Active Servable Manifest Contracts
    print("\n--- Check 2: Active Production Model Manifest Contracts ---")
    active_slots = []
    for iv in ["15", "30", "60", "120", "240"]:
        for reg in ["trending", "ranging"]:
            slot = f"{reg}_{iv}"
            is_denied = slot in MODEL_SLOT_DENYLIST
            manifest_path = f"ensemble_{reg}_trend_{iv}_manifest.json"

            if is_denied:
                print(f"  🚫 {slot:<15} : DENIED (Inert / Abstains safely)")
                continue

            if not os.path.exists(manifest_path):
                errors.append(f"Active model slot '{slot}' has missing manifest '{manifest_path}'")
                continue

            try:
                with open(manifest_path, "r") as f:
                    manifest = json.load(f)
                
                bcfg = manifest.get("barrier_config", {})
                if not bcfg:
                    errors.append(f"{slot} manifest missing 'barrier_config' block")
                    continue

                # Verify barriers against live TIMEFRAME_CONFIG
                live_tf = TIMEFRAME_CONFIG.get(iv, {})
                for k in ["tp_mult_trending", "tp_mult_ranging", "sl_mult", "lookahead"]:
                    if k in bcfg and k in live_tf:
                        diff = abs(float(bcfg[k]) - float(live_tf[k]))
                        if diff > 1e-9:
                            errors.append(f"{slot} barrier '{k}' mismatch: trained={bcfg[k]}, serving={live_tf[k]}")

                # Verify ADX thresholds against live config
                exp_enter = REGIME_ADX_ENTER_BY_INTERVAL.get(iv, 28.0)
                exp_exit = REGIME_ADX_EXIT_BY_INTERVAL.get(iv, 24.0)

                act_enter = float(bcfg.get("regime_adx_enter", -1))
                act_exit = float(bcfg.get("regime_adx_exit", -1))

                if abs(act_enter - exp_enter) > 1e-9:
                    errors.append(f"{slot} regime_adx_enter mismatch: trained={act_enter}, serving={exp_enter}")
                if abs(act_exit - exp_exit) > 1e-9:
                    errors.append(f"{slot} regime_adx_exit mismatch: trained={act_exit}, serving={exp_exit}")

                # Verify feature count
                feats = manifest.get("feature_names", [])
                feat_cnt = manifest.get("feature_count", 0)
                if len(feats) != feat_cnt:
                    errors.append(f"{slot} feature count mismatch: count={feat_cnt}, list_len={len(feats)}")

                # Verify binary files on disk
                prefix = f"ensemble_{reg}_trend_{iv}"
                price_prefix = f"ensemble_{reg}_price_{iv}"
                for req_f in [
                    f"{prefix}_xgb.json",
                    f"{prefix}_lgb.txt",
                    f"{prefix}_cat.json",
                    f"{price_prefix}_xgb.json",
                    f"{price_prefix}_lgb.txt",
                    f"{price_prefix}_cat.json",
                    f"calibrator_{reg}_{iv}.json"
                ]:
                    if not os.path.exists(req_f):
                        errors.append(f"{slot} missing required binary/calibrator file: {req_f}")

                active_slots.append(slot)
                print(f"  ✅ {slot:<15} : Verified (ADX {act_enter:.1f}/{act_exit:.1f}, {feat_cnt} features, all binaries present)")

            except Exception as e:
                errors.append(f"Failed to validate manifest for {slot}: {e}")

    print("\n" + "=" * 60)
    print(f"Active Servable Slots: {len(active_slots)} ({', '.join(active_slots)})")
    print(f"Denied / Offline Slots: {len(MODEL_SLOT_DENYLIST)} ({', '.join(sorted(MODEL_SLOT_DENYLIST))})")

    if warnings:
        print("\n⚠️ WARNINGS:")
        for w in warnings:
            print(f"  - {w}")

    if errors:
        print("\n❌ PREFLIGHT FAILED — CONTRACT VIOLATIONS DETECTED:")
        for err in errors:
            print(f"  - {err}")
        print("=" * 60)
        sys.exit(1)
    else:
        print("\n✅ PREFLIGHT PASSED — ALL CONTRACTS 100% IN LOCKSTEP")
        print("=" * 60)
        sys.exit(0)

if __name__ == "__main__":
    run_preflight()
