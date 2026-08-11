"""Minimal ELF32 loader: PT_LOAD segments + symbol lookup (tohost etc.).
Enough for riscv-tests and rvfuzz binaries; no relocation, no dynamic."""
import struct


def load(path: str):
    """Returns (entry, segments, symbols) where segments is a list of
    (paddr, bytes) and symbols maps name -> value."""
    with open(path, "rb") as f:
        data = f.read()
    if data[:4] != b"\x7fELF" or data[4] != 1:
        raise ValueError(f"{path}: not a 32-bit ELF")
    (e_entry,) = struct.unpack_from("<I", data, 24)
    (e_phoff,) = struct.unpack_from("<I", data, 28)
    (e_shoff,) = struct.unpack_from("<I", data, 32)
    (e_phentsize, e_phnum) = struct.unpack_from("<HH", data, 42)
    (e_shentsize, e_shnum) = struct.unpack_from("<HH", data, 46)

    segs = []
    for i in range(e_phnum):
        off = e_phoff + i * e_phentsize
        p_type, p_offset, p_vaddr, p_paddr, p_filesz, p_memsz = \
            struct.unpack_from("<IIIIII", data, off)
        if p_type == 1 and p_memsz:  # PT_LOAD
            blob = bytearray(data[p_offset:p_offset + p_filesz])
            blob += bytes(p_memsz - p_filesz)
            segs.append((p_paddr, bytes(blob)))

    # symbols: find .symtab (type 2) + its strtab
    syms = {}
    shdrs = []
    for i in range(e_shnum):
        off = e_shoff + i * e_shentsize
        sh_name, sh_type, _f, _a, sh_offset, sh_size, sh_link, _i2, _al, sh_entsize = \
            struct.unpack_from("<IIIIIIIIII", data, off)
        shdrs.append((sh_type, sh_offset, sh_size, sh_link, sh_entsize))
    for sh_type, sh_offset, sh_size, sh_link, sh_entsize in shdrs:
        if sh_type == 2 and sh_entsize:
            str_off = shdrs[sh_link][1]
            for j in range(sh_size // sh_entsize):
                st_name, st_value = struct.unpack_from("<II", data, sh_offset + j * sh_entsize)
                if st_name:
                    end = data.index(b"\0", str_off + st_name)
                    syms[data[str_off + st_name:end].decode()] = st_value
    return e_entry, segs, syms


if __name__ == "__main__":
    import sys
    entry, segs, syms = load(sys.argv[1])
    print(f"entry {entry:#x}")
    for pa, blob in segs:
        print(f"  seg {pa:#x} {len(blob)} bytes")
    for k in ("tohost", "fromhost", "_start"):
        if k in syms:
            print(f"  {k} = {syms[k]:#x}")
