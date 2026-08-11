#!/usr/bin/env python3
"""mkcuda — the tiny-CUDA host runtime (M17, runtime_spec).

The CUDA-shaped API, honestly scoped to the v1 hardware (see
docs/SOFTWARE_AND_VALIDATION.md#cuda-shaped-programming-model for the full concept map):

    rt = Runtime(backend="sim")          # context (persistent sim proc)
    k  = rt.compile("sw/kernels/vec_add.c")      # nvcc-alike, cached
    a  = rt.to_device(np_array)          # cudaMalloc + H2D
    c  = rt.alloc(nbytes)                # cudaMalloc
    st = rt.launch(k, grid_n=N, args=[a, b, c, N])   # <<<...>>> + sync
    out = rt.from_device(c, dtype, count)            # D2H
    st.cycles, st.ipc, st.dbeats, ...    # nvprof-lite per-launch stats
    rt.device_log()                      # mk_log() ring buffers, per tid
    (mk_assert failures raise KernelAssert with tid + code)

Backends: "sim" = persistent Verilated simt_core (tb/simt_soc);
"hw" = same interface, NotImplementedError until the Bring-up Campaign.
"""
import hashlib
import os
import pathlib
import subprocess
import sys
import time

import numpy as np

# The tree this module lives in. Gates put their own tree's host/ first on
# sys.path, so a worktree gate simulates that worktree's RTL.
MK = pathlib.Path(__file__).resolve().parent.parent
# Toolchains default to this checkout but can live in a shared location.
TOOL_ROOT = pathlib.Path(os.environ.get("GPU_TSU_TOOL_ROOT", MK))
GCC = TOOL_ROOT / "tools/xpack-riscv-none-elf-gcc/bin/riscv-none-elf-gcc"
OBJCOPY = TOOL_ROOT / "tools/xpack-riscv-none-elf-gcc/bin/riscv-none-elf-objcopy"
OBJDUMP = TOOL_ROOT / "tools/xpack-riscv-none-elf-gcc/bin/riscv-none-elf-objdump"
RT_DIR = MK / "sw/rt"
CACHE = pathlib.Path(os.environ.get("GPU_TSU_CACHE_DIR",
                                    MK / "data/kernel_cache"))
# §1c profiles: MK_PROFILE selects the harness build (0/unset = union,
# the unchanged default path; N != 0 = build_pN with -GPROFILE=N)
_PROFILE = os.environ.get("MK_PROFILE", "0")
_BUILD = "build" if _PROFILE == "0" else f"build_p{_PROFILE}"
HARNESS = MK / f"tb/simt_soc/{_BUILD}/soc_harness"

# runtime_spec §2 (frozen)
PARAMS_BASE = 0x0002_0000
DONE_BASE = 0x0002_0100
LOG_BASE = 0x0002_0200          # mk_log rings: 64 tids x 32 words
LOG_SLOTS = 32
ASSERT_BASE = 0x0002_2200       # mk_assert: word per tid (0 ok, else code)
HEAP_BASE = 0x0020_0000    # above the stack region (gpt2_spec §1)
HEAP_END = 0x1000_0000
NTHREADS = 64
# -O0 is the SIMT-SAFETY default, not laziness: unoptimized GCC never
# rotates loops, so every backedge is an unconditional jump (warp-uniform
# by construction) and every conditional branch is forward — arbitrary C
# is divergence-legal. -O1+ rotates guards into lane-divergent BACKWARD
# branches (observed twice; runtime_spec §4). Opt in per kernel via
# extra_cflags after inspecting the disasm.
CFLAGS = ["-march=rv32i_zicsr", "-mabi=ilp32", "-O0", "-nostdlib",
          "-nostartfiles", "-ffreestanding", "-fno-builtin",
          "-Wall", "-Werror"]


CAUSES = {0: "misaligned-fetch", 2: "illegal-instruction (v1 limits: "
          "divergent backward branch / divergent JALR / stack overflow)",
          3: "ebreak", 4: "load-misaligned", 6: "store-misaligned",
          11: "ecall"}


class KernelAssert(Exception):
    def __init__(self, failures):
        self.failures = failures            # [(tid, code), ...]
        msgs = []
        for tid, code in failures[:4]:
            if code & 0x80000000:
                cause = code & 0x7FFFFFFF
                msgs.append(f"tid {tid} TRAPPED: mcause {cause} "
                            f"({CAUSES.get(cause, '?')})")
            else:
                msgs.append(f"tid {tid} mk_assert code {code}")
        super().__init__("; ".join(msgs) +
                         (" ..." if len(failures) > 4 else ""))


class LaunchStats:
    def __init__(self, cycles, commits, wb, memc, dbeats, wall_s):
        self.cycles, self.commits = cycles, commits
        self.wb_commits, self.mem_commits = wb, memc
        self.dbeats, self.wall_s = dbeats, wall_s
        self.ipc = commits / cycles if cycles else 0.0

    def __repr__(self):
        return (f"<LaunchStats cycles={self.cycles} commits={self.commits} "
                f"ipc={self.ipc:.3f} dbeats={self.dbeats}>")


class Kernel:
    def __init__(self, name, image, elf, disasm_path):
        self.name = name
        self.image = image                  # bytes, load at 0
        self.elf = elf
        self.disasm_path = disasm_path


class Buffer:
    def __init__(self, addr, nbytes):
        self.addr, self.nbytes = addr, nbytes

    def __index__(self):                    # lets launch args take Buffers
        return self.addr


class Runtime:
    def __init__(self, backend="sim", quiet=True):
        if backend == "hw":
            raise NotImplementedError(
                "--backend hw is the Bring-up Campaign's flag: same "
                "interface over the UART bridge (runtime_spec §6)")
        assert backend == "sim", backend
        self.quiet = quiet
        # single-device-sim rule (2026-07-14): two concurrent Verilated
        # soc_harness sims (~16 GB pressure each) drove the box to
        # jetsam and killed the first T5σ cert at iter 8575/12000.
        # Refuse to start a second one unless explicitly overridden.
        others = subprocess.run(["pgrep", "-x", "soc_harness"],
                                capture_output=True, text=True).stdout.split()
        if others and not os.environ.get("MK_ALLOW_CONCURRENT_SIM"):
            raise RuntimeError(
                f"another soc_harness is running (pids {others}) — the "
                "single-device-sim rule (jetsam killed the 2026-07-14 T5 "
                "cert). Set MK_ALLOW_CONCURRENT_SIM=1 to override.")
        CACHE.mkdir(parents=True, exist_ok=True)
        # always invoke make: the Makefile's dependency tracking decides
        # whether to rebuild (2026-07-11: the exists()-only guard ran a
        # stale soc_harness through the S7 gate and voided a bisect)
        subprocess.run(["make", "-s", "-C", MK / "tb/simt_soc", "soc",
                        f"PROFILE={_PROFILE}"],
                       check=True)
        # instrument identity: every gate log names the tree it simmed
        print(f"[mkcuda] tree {MK}", file=sys.stderr)
        self.proc = subprocess.Popen(
            [str(HARNESS)], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            text=True, bufsize=1, cwd=MK)
        self._heap = HEAP_BASE

    # ---------------- protocol ----------------
    def _cmd(self, line):
        self.proc.stdin.write(line + "\n")
        self.proc.stdin.flush()
        return self.proc.stdout.readline().strip()

    def close(self):
        if self.proc.poll() is None:
            try:
                self._cmd("QUIT")
            except Exception:
                pass
            self.proc.wait(timeout=10)

    # ---------------- nvcc-alike (with content-hash cache) --------------
    def compile(self, src_path, extra_cflags=()):
        src_path = pathlib.Path(src_path)
        src = src_path.read_bytes()
        rt_bits = (RT_DIR / "crt0.S").read_bytes() + \
            (RT_DIR / "kernel.ld").read_bytes()
        for hdr in sorted(RT_DIR.glob("*.h")):    # ALL runtime headers
            rt_bits += hdr.read_bytes()
        h = hashlib.sha256(src + rt_bits +
                           " ".join(CFLAGS + list(extra_cflags)).encode()
                           ).hexdigest()[:16]
        name = src_path.stem
        img_p = CACHE / f"{name}-{h}.bin"
        elf_p = CACHE / f"{name}-{h}.elf"
        dis_p = CACHE / f"{name}-{h}.dis"
        if not img_p.exists():
            subprocess.run(
                [str(GCC), *CFLAGS, *extra_cflags, "-I", str(RT_DIR),
                 "-T", str(RT_DIR / "kernel.ld"),
                 str(RT_DIR / "crt0.S"), str(src_path), "-o", str(elf_p)],
                check=True, capture_output=True, text=True)
            subprocess.run([str(OBJCOPY), "-O", "binary", str(elf_p),
                            str(img_p)], check=True)
            with open(dis_p, "w") as f:
                subprocess.run([str(OBJDUMP), "-d", str(elf_p)], stdout=f,
                               check=True)
        return Kernel(name, img_p.read_bytes(), elf_p, dis_p)

    # ---------------- memory (cudaMalloc / Memcpy / Memset) -------------
    def alloc(self, nbytes, align=64):
        a = (self._heap + align - 1) & ~(align - 1)
        assert a + nbytes <= HEAP_END, "device heap exhausted"
        self._heap = a + nbytes
        return Buffer(a, nbytes)

    def free_all(self):
        """Arena reset (v1 allocator is a bump arena)."""
        self._heap = HEAP_BASE

    def write(self, buf, data: bytes, offset=0):
        step = 2048
        for i in range(0, len(data), step):
            chunk = data[i:i + step]
            r = self._cmd(f"LOAD {buf.addr + offset + i:x} {chunk.hex()}")
            assert r == "OK", r

    def read(self, buf, nbytes=None, offset=0):
        n = buf.nbytes if nbytes is None else nbytes
        out = b""
        step = 2048
        for i in range(0, n, step):
            k = min(step, n - i)
            r = self._cmd(f"PEEK {buf.addr + offset + i:x} {k}")
            out += bytes.fromhex(r)
        return out

    def to_device(self, arr: np.ndarray):
        b = self.alloc(arr.nbytes)
        self.write(b, arr.tobytes())
        return b

    def from_device(self, buf, dtype, count):
        return np.frombuffer(self.read(buf, np.dtype(dtype).itemsize * count),
                             dtype=dtype)

    def memset(self, buf, byte=0):
        self.write(buf, bytes([byte]) * buf.nbytes)

    # ---------------- launch (<<<grid>>> + synchronize) -----------------
    def launch(self, kernel, grid_n, args=(), max_cycles=50_000_000,
               chunk=200_000):
        t0 = time.time()
        self._cmd("RESET")
        # image + param block + cleared mailboxes/log/assert
        self.write(Buffer(0, len(kernel.image)), kernel.image)
        grid_pad = ((grid_n + NTHREADS - 1) // NTHREADS) * NTHREADS
        pb = [len(args), grid_pad] + [int(a) & 0xFFFFFFFF for a in args]
        self.write(Buffer(PARAMS_BASE, 0),
                   b"".join(v.to_bytes(4, "little") for v in pb))
        self.write(Buffer(DONE_BASE, 0), bytes(4 * NTHREADS))
        self.write(Buffer(ASSERT_BASE, 0), bytes(4 * NTHREADS))
        self.write(Buffer(LOG_BASE, 0),
                   bytes(4 * NTHREADS * (LOG_SLOTS + 1)))
        c_before = self._counters()
        self._cmd("RESET")
        ran = 0
        while ran < max_cycles:
            self._cmd(f"RUN {chunk}")
            ran += chunk
            done = np.frombuffer(
                self.read(Buffer(DONE_BASE, 4 * NTHREADS)), dtype="<u4")
            if int(done.sum()) == NTHREADS:
                break
        else:
            fails = self._assert_failures()
            if fails:
                raise KernelAssert(fails)
            raise TimeoutError(
                f"{kernel.name}: not all threads done after {ran} cycles "
                f"({int(done.sum())}/{NTHREADS})")
        c_after = self._counters()
        fails = self._assert_failures()
        if fails:
            raise KernelAssert(fails)
        d = {k: c_after[k] - c_before[k] for k in c_after}
        return LaunchStats(d["CYCLES"], d["COMMITS"], d["WBCOMMITS"],
                           d["MEMCOMMITS"], d["DBEATS"], time.time() - t0)

    def _counters(self):
        toks = self._cmd("COUNTERS").split()
        return {toks[i]: int(toks[i + 1]) for i in range(0, len(toks), 2)}

    def set_latency(self, lat):
        """G1 knob: memory service latency in cycles (sim backend)."""
        assert self._cmd(f"LAT {int(lat)}") == "OK"

    def _assert_failures(self):
        a = np.frombuffer(self.read(Buffer(ASSERT_BASE, 4 * NTHREADS)),
                          dtype="<u4")
        return [(int(t), int(c)) for t, c in enumerate(a) if c]

    # ---------------- device log (printf-lite) ----------------
    def device_log(self):
        """mk_log() ring buffers -> {tid: [values]}."""
        raw = np.frombuffer(
            self.read(Buffer(LOG_BASE, 4 * NTHREADS * (LOG_SLOTS + 1))),
            dtype="<u4").reshape(NTHREADS, LOG_SLOTS + 1)
        out = {}
        for tid in range(NTHREADS):
            n = min(int(raw[tid, 0]), LOG_SLOTS)
            if n:
                out[tid] = [int(v) for v in raw[tid, 1:1 + n]]
        return out

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()

# ---- MK_TRANSPORT=uio: swap in the silicon transport (SG0) ----
if os.environ.get("MK_TRANSPORT") == "uio":
    from mkuio import UioRuntime as Runtime      # noqa: F811,E402
