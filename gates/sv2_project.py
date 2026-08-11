#!/usr/bin/env python3
"""SV2.4 gate — the honest Llama-1B serving projection (σ-labeled).

tok/s = min( BW_eff / bytes_per_token , MAC_rate / MACs_per_token )

Every input is FROZEN and printed: the DDR budget and TA clock are
model inputs (docs/SOFTWARE_AND_VALIDATION.md#serving-engine), the streaming efficiency is
MEASURED by tb/tensor_array smoke_eff (ci/logs/sv2/eff.json), the
model card is Llama-3.2-1B-class. Bars: W8 >= 8 tok/s, W4 >= 16.
"""
import json
import pathlib
import sys

MK = pathlib.Path(__file__).resolve().parent.parent
SV2 = MK / "ci/logs/sv2"

# frozen model inputs (spec §bars + §σ-scope)
DDR_BUDGET = 12e9            # B/s effective (frozen model input)
TA_CLK = 250e6               # Hz (frozen target; OOC bar SV2.3)
BEAT_BYTES = 64              # spec v1 (512b)
LANES = {"w8": 64, "w4": 128}

# Llama-3.2-1B-class card (frozen): params dominate; KV at ctx 1024
PARAMS = 1.236e9
KV_BYTES_TOK = 16 * 2 * 8 * 64 * 1024   # layers x 2 x kv_heads x hd x ctx
MACS_TOK = PARAMS * 1.02                 # matvec + attention margin
BARS = {"w8": 8.0, "w4": 16.0}


def main():
    eff = json.loads((SV2 / "eff.json").read_text())
    rep = {"inputs": {"ddr_budget_Bps": DDR_BUDGET, "ta_clk_Hz": TA_CLK,
                      "beat_bytes": BEAT_BYTES, "params": PARAMS,
                      "kv_bytes_per_tok": KV_BYTES_TOK,
                      "macs_per_tok": MACS_TOK,
                      "measured_eff": {f: eff[f]["eff"]
                                       for f in ("w8", "w4")}}}
    ok = True
    for fmt in ("w8", "w4"):
        e = eff[fmt]["eff"]
        port_bw = e * BEAT_BYTES * TA_CLK
        bw = min(port_bw, DDR_BUDGET)
        wbytes = PARAMS * (1.0 if fmt == "w8" else 0.5)
        bytes_tok = wbytes + KV_BYTES_TOK
        mac_rate = e * LANES[fmt] * TA_CLK
        toks_bw = bw / bytes_tok
        toks_mac = mac_rate / MACS_TOK
        toks = min(toks_bw, toks_mac)
        rep[fmt] = {"eff": e, "port_GBps": port_bw / 1e9,
                    "bw_GBps": bw / 1e9, "bytes_per_tok": bytes_tok,
                    "toks_bw_bound": toks_bw, "toks_mac_bound": toks_mac,
                    "toks_projected": toks, "bar": BARS[fmt],
                    "pass": toks >= BARS[fmt]}
        ok &= toks >= BARS[fmt]
        print(f"  {fmt.upper()}: eff {e:.3f}  port {port_bw / 1e9:.1f} GB/s"
              f"  BW-bound {toks_bw:.1f} tok/s  MAC-bound {toks_mac:.1f}"
              f"  -> {toks:.1f} tok/s (bar {BARS[fmt]}) "
              f"{'PASS' if toks >= BARS[fmt] else 'FAIL'}")
    rep["pass"] = bool(ok)
    (SV2 / "sv2_report.json").write_text(json.dumps(rep, indent=1))
    print(f"SV2.4 (σ-projection): {'PASS' if ok else 'FAIL'} -> "
          f"{SV2 / 'sv2_report.json'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
