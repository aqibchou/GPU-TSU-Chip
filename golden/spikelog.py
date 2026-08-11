"""Parse `spike -l --log-commits` stderr into the canonical commit format
(see golden/iss.py). Rules learned empirically (M2):

- commit lines carry a privilege digit:  core   0: 3 0xPC (0xINSTR) ...
- disasm lines don't:                    core   0: 0xPC (0xINSTR) mnemonic
- loads:   `xN 0xVAL mem 0xADDR`   (mem token = address only -> ignored)
- stores:  `mem 0xADDR 0xVAL`      (two hex after mem -> a store)
- CSR effects appear as `cNNN_name 0xVAL` tokens -> ignored
- exception lines (`core   0: exception ...`) -> no commit, skipped
- spike's 0x1000 bootrom runs first -> caller drops commits until entry pc
"""
import re

_COMMIT = re.compile(r"^core\s+\d+:\s+\d+\s+0x([0-9a-f]+)\s+\(0x([0-9a-f]+)\)(.*)$")
_XTOK = re.compile(r"\bx(\d+)\s+0x([0-9a-f]+)")
_MEMST = re.compile(r"\bmem\s+0x([0-9a-f]+)\s+0x([0-9a-f]+)")

_SIZE_BY_F3 = {0: 0, 1: 1, 2: 2}


def parse(text: str, entry: int, tohost: int | None = None):
    """Returns (lines, tohost_value). Stops at (and includes) the first store
    to tohost, mirroring the ISS/RTL halt convention."""
    out = []
    started = False
    for raw in text.splitlines():
        m = _COMMIT.match(raw)
        if not m:
            continue
        pc = int(m.group(1), 16)
        instr = int(m.group(2), 16)
        rest = m.group(3)
        if not started:
            if pc == entry:
                started = True
            else:
                continue
        s = f"{pc:08x}:{instr:08x}"
        xm = _XTOK.search(rest)
        if xm and int(xm.group(1)) != 0:
            s += f":x{int(xm.group(1))}={int(xm.group(2), 16):08x}"
        st = _MEMST.search(rest)
        val = None
        if st:
            addr = int(st.group(1), 16)
            sval = int(st.group(2), 16)
            f3 = (instr >> 12) & 7
            sz = _SIZE_BY_F3.get(f3, 2)
            s += f":S@{addr:08x}={sval & ((1 << (8 << sz)) - 1):08x}.{sz}"
            if tohost is not None and addr == tohost:
                out.append(s)
                return out, sval
        out.append(s)
    return out, None
