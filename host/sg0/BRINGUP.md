# SG0 board-day runbook (KV260/KR260, K26 SOM)

Everything below assumes the artifacts from the sg0bit sitting:
`sg0_p<N>.bit` (per profile), `host/sg0/sg0.dtsi`, and the repo on
the board (or gates rsync'd).

## 1. Package the app (once per profile, on any Linux box or the board)

    bootgen -arch zynqmp -image bit.bif -process_bitstream bin
      # bit.bif:  all: { [destination_device=pl] sg0_p0.bit }
      # yields sg0_p0.bit.bin
    dtc -O dtb -o sg0.dtbo -b 0 -@ sg0.dtsi
    # shell.json: {"shell_type": "XRT_FLAT", "num_slots": "1"}
    sudo mkdir -p /lib/firmware/xilinx/sg0-p0
    sudo cp sg0_p0.bit.bin sg0.dtbo shell.json /lib/firmware/xilinx/sg0-p0/

## 2. Boot-time carveout check

`sg0.dtsi` reserves 1 GB at 0x4000_0000 (no-map) and binds the
bridge at 0xA000_0000 to generic-uio. Confirm after loadapp:

    sudo xmutil unloadapp
    time sudo xmutil loadapp sg0-p0        # <- the PR3 number
    ls /dev/uio*                            # bridge should enumerate
    cat /proc/device-tree/reserved-memory/sg0carve@40000000/reg | xxd

If generic-uio doesn't bind, add `uio_pdrv_genirq.of_id=generic-uio`
to bootargs (the classic Kria uio gotcha).

## 3. SG0 bars (freeze at the sitting; candidates from the spec)

    export MK_TRANSPORT=uio
    python - <<'EOF'          # SG0.1: bridge echo
    import sys; sys.path.insert(0, "host")
    import mkcuda
    rt = mkcuda.Runtime()     # asserts STATUS magic 0x05D0
    print("magic ok, mcycle:", rt.mcycle())
    EOF

    # SG0.2: carveout r/w — pattern test through rt.write/rt.read
    # SG0.3: run the chip-level ISA/profile gates unchanged on silicon
    python gates/s7_isa.py
    python gates/s8_isa.py
    python gates/pr4_absence.py
    # Then run the PR2 drain/swap/reload procedure across a real xmutil
    # swap; PR3 is the `time` measurement above.

## Known deltas vs sim (disclosed, not surprises)

- mk_mcycle() on silicon counts wall cycles of a free-running clock;
  cycle NUMBERS differ from sim (D-026 — values, not cycles, are
  the architectural contract).
- The uio transport maps RESET->run=0, RUN->run=1+poll; PSTAT/SCHED
  instrument verbs have no hardware mapping at SG0 (raise cleanly).
- Gates that pass --pstat/--sched flags must run without them on
  the board.
