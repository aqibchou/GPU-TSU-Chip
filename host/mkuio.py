"""MK_TRANSPORT=uio — mkcuda's silicon transport (docs/FPGA_IMPLEMENTATION.md#sg0-fpga-bridge).

The same Runtime surface the gates already use, retargeted from the
sim harness's line protocol to the SG0 bridge: registers via the
generic-uio mmap, device memory via /dev/mem at CARVE_BASE. The
gates run UNCHANGED — that is the SG0 thesis (the certified gate
suite IS the bring-up suite).

Semantics mapping (see sg0_bridge as-built regmap):
  RESET  -> CTRL.run=0 (SoC held in reset; DRAM untouched — the PR2
            physics). launch() calls RESET before image load and
            again before RUN; both land as run=0.
  RUN n  -> first call releases run=1; the chip free-runs (cycle
            stepping is a sim concept). Subsequent RUN chunks are a
            poll pacing sleep. DONE detection stays what it always
            was: the host reads the DONE words from device memory.
  LOAD/PEEK -> direct mmap of the carveout. Writes below IMEM_BYTES
            are MIRRORED into the imem BRAM through the AXI-Lite
            window (the image must live in both: fetch reads BRAM,
            dmem data reads DRAM).

UNTESTED UNTIL BOARD DAY — written against the unit-certified
bridge; first exercised at SG0.1/SG0.2.
"""
import mmap
import os
import struct
import time

import mkcuda
from mkcuda import Buffer

CARVE_BASE = 0x4000_0000
CARVE_SIZE = 0x4000_0000          # 1 GB
IMEM_BYTES = 32768 * 4
R_CTRL, R_STATUS, R_MCLO, R_MCHI, R_IMEM_ADDR, R_IMEM_DATA = \
    0x00, 0x04, 0x08, 0x0C, 0x10, 0x14
MAGIC = 0x05D0


class UioRuntime(mkcuda.Runtime):
    """Runtime against the SG0 bridge instead of the sim harness."""

    def __init__(self, uio_dev=None, mem_dev="/dev/mem"):
        # no harness process, no single-sim rule — this is hardware
        uio_dev = uio_dev or os.environ.get("MK_UIO", "/dev/uio0")
        self._uio_f = os.open(uio_dev, os.O_RDWR | os.O_SYNC)
        self._reg = mmap.mmap(self._uio_f, 0x1000)
        self._mem_f = os.open(mem_dev, os.O_RDWR | os.O_SYNC)
        self._mem = mmap.mmap(self._mem_f, CARVE_SIZE, offset=CARVE_BASE)
        self.proc = None
        self._heap = mkcuda.HEAP_BASE
        self._running = False
        st = self._rr(R_STATUS)
        assert (st >> 16) == MAGIC, \
            f"SG0 bridge not answering: STATUS={st:#x}"

    # ---- register access ----
    def _rr(self, off):
        return struct.unpack_from("<I", self._reg, off)[0]

    def _rw(self, off, val):
        struct.pack_into("<I", self._reg, off, val & 0xFFFFFFFF)

    def mcycle(self):
        lo = self._rr(R_MCLO)
        return (self._rr(R_MCHI) << 32) | lo

    # ---- the protocol verbs the gates actually use ----
    def _cmd(self, line):
        toks = line.split()
        verb = toks[0]
        if verb == "RESET":
            self._rw(R_CTRL, 0)
            self._running = False
            return "OK"
        if verb == "RUN":
            if not self._running:
                self._rw(R_CTRL, 1)
                self._running = True
            else:
                time.sleep(0.001)      # poll pacing; the chip free-runs
            return "OK"
        if verb == "QUIT":
            return "OK"
        raise NotImplementedError(
            f"uio transport: no hardware mapping for '{verb}' "
            "(sim-only instrument?)")

    def close(self):
        try:
            self._rw(R_CTRL, 0)
        finally:
            for m in (self._mem, self._reg):
                try:
                    m.close()
                except Exception:
                    pass
            os.close(self._mem_f)
            os.close(self._uio_f)

    # ---- memory ----
    def write(self, buf, data: bytes, offset=0):
        a = buf.addr + offset
        self._mem[a:a + len(data)] = data
        if a < IMEM_BYTES:                 # mirror .text into imem BRAM
            end = min(a + len(data), IMEM_BYTES)
            chunk = data[:end - a]
            pad = (-len(chunk)) % 4
            chunk += b"\x00" * pad
            self._rw(R_IMEM_ADDR, a >> 2)
            for i in range(0, len(chunk), 4):
                self._rw(R_IMEM_DATA,
                         struct.unpack_from("<I", chunk, i)[0])

    def read(self, buf, nbytes=None, offset=0):
        n = buf.nbytes if nbytes is None else nbytes
        a = buf.addr + offset
        return bytes(self._mem[a:a + n])
