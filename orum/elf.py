"""ELF64 emitter for Orum.

Produces a runnable static Linux x86-64 binary: PT_LOAD segments
(R+X for headers+text+rodata, RW for .data), loaded at a fixed
address. No libc — the generated code uses direct syscalls.

v0.006 freestanding: no-syscall kernels. Adds .symtab (with UND
__orum_heap_start / __orum_heap_end and the SHN_ABS entry symbol),
.strtab, .shstrtab and .rela.text (R_X86_64_64 against the UND
symbols). The binary must be linked (resolving the two heap symbols)
before boot.
"""
from __future__ import annotations


VBASE = 0x400000  # load address
TEXT_OFF = 192  # file offset of .text (after ehdr+2*phdr, 16-aligned)

# Section / symbol constants
SHT_SYMTAB = 2
SHT_STRTAB = 3
SHT_RELA = 4
SHN_UNDEF = 0
SHN_ABS = 0xFFF1
STB_GLOBAL = 1
STT_NOTYPE = 0
STT_FUNC = 2
R_X86_64_64 = 1

HEAP_START_SYM = "__orum_heap_start"
HEAP_END_SYM = "__orum_heap_end"


def write_elf(text: bytes, rodata: bytes,
              rodata_refs: list[tuple[int, int]],
              entry_off: int = 0,
              data: bytes = b"",
              data_refs: list[tuple[int, int]] = (),
              freestanding: bool = False,
              entry_symbol: str = "_start",
              sym_relocs: list[tuple[int, str]] = ()) -> bytes:
    """Assemble the ELF file.

    text: machine code; rodata: read-only data; rodata_refs: list of
    (text_pos, rodata_off) pairs whose 8 bytes get patched with the
    absolute virtual address of the rodata entry; entry_off: offset of
    the entry point within text; data: writable data (second PT_LOAD,
    PF_R|PF_W); data_refs: list of (text_pos, data_off) pairs patched
    with the absolute virtual address of the data entry (same mechanism
    as rodata_refs).

    Freestanding extras: sym_relocs is a list of (text_pos,
    symbol_name) pairs recorded as R_X86_64_64 relocations against
    UND symbols; the entry symbol is recorded SHN_ABS so its address
    is inspectable (readelf -s) and comparable to e_entry.
    """
    text = bytearray(text)
    rodata_off = TEXT_OFF + len(text)
    while rodata_off % 8:
        rodata_off += 1
    rodata_vaddr = VBASE + rodata_off
    for pos, roff in rodata_refs:
        addr = rodata_vaddr + roff
        text[pos : pos + 8] = addr.to_bytes(8, "little")

    data_off = rodata_off + len(rodata)
    # The data segment must start on its own page: if it shared a page
    # with the R+X text segment, the RW mapping would overlay it and
    # strip execute permission from the whole page (SEGV_ACCERR).
    while data_off % 0x1000:
        data_off += 1
    data_vaddr = VBASE + data_off
    for pos, doff in data_refs:
        addr = data_vaddr + doff
        text[pos : pos + 8] = addr.to_bytes(8, "little")

    entry_vaddr = VBASE + TEXT_OFF + entry_off

    # Segment 1 (R+X) covers headers+text+rodata, i.e. everything before
    # the data page; segment 2 (RW) covers the data page. No overlap.
    seg1_filesz = data_off
    filesz = data_off + len(data)
    out = bytearray()

    # --- ELF header (64 bytes) ---
    out += b"\x7fELF"
    out += bytes([2, 1, 1, 0])  # 64-bit, little-endian, v1, SysV ABI
    out += b"\x00" * 8  # padding / OSABI version
    out += (2).to_bytes(2, "little")  # ET_EXEC
    out += (62).to_bytes(2, "little")  # EM_X86_64
    out += (1).to_bytes(4, "little")  # EV_CURRENT
    out += entry_vaddr.to_bytes(8, "little")  # e_entry
    out += (64).to_bytes(8, "little")  # e_phoff
    out += (0).to_bytes(8, "little")  # e_shoff (patched below if needed)
    out += (0).to_bytes(4, "little")  # e_flags
    out += (64).to_bytes(2, "little")  # e_ehsize
    out += (56).to_bytes(2, "little")  # e_phentsize
    out += (2).to_bytes(2, "little")  # e_phnum
    out += (0).to_bytes(2, "little")  # e_shentsize (patched below)
    out += (0).to_bytes(2, "little")  # e_shnum (patched below)
    out += (0).to_bytes(2, "little")  # e_shstrndx (patched below)
    assert len(out) == 64

    # --- Program header 1: PT_LOAD, read+execute (56 bytes) ---
    out += (1).to_bytes(4, "little")  # PT_LOAD
    out += (5).to_bytes(4, "little")  # PF_R | PF_X
    out += (0).to_bytes(8, "little")  # p_offset
    out += VBASE.to_bytes(8, "little")  # p_vaddr
    out += VBASE.to_bytes(8, "little")  # p_paddr
    out += seg1_filesz.to_bytes(8, "little")  # p_filesz
    out += seg1_filesz.to_bytes(8, "little")  # p_memsz
    out += (0x1000).to_bytes(8, "little")  # p_align
    assert len(out) == 120

    # --- Program header 2: PT_LOAD, read+write (.data) ---
    out += (1).to_bytes(4, "little")  # PT_LOAD
    out += (6).to_bytes(4, "little")  # PF_R | PF_W
    out += data_off.to_bytes(8, "little")  # p_offset
    out += data_vaddr.to_bytes(8, "little")  # p_vaddr
    out += data_vaddr.to_bytes(8, "little")  # p_paddr
    out += len(data).to_bytes(8, "little")  # p_filesz
    out += len(data).to_bytes(8, "little")  # p_memsz
    out += (0x1000).to_bytes(8, "little")  # p_align
    assert len(out) == 176

    out += b"\x00" * (TEXT_OFF - len(out))
    out += text
    while len(out) < rodata_off:
        out.append(0)
    out += rodata
    while len(out) < data_off:
        out.append(0)
    out += data

    if freestanding:
        _append_freestanding_sections(out, entry_vaddr, entry_symbol,
                                     sym_relocs)
    return bytes(out)


def _append_freestanding_sections(out: bytearray, entry_vaddr: int,
                                 entry_symbol: str,
                                 sym_relocs: list[tuple[int, str]]):
    """Append .symtab/.strtab/.shstrtab/.rela.text + section headers.

    Symbols: null, __orum_heap_start (UND), __orum_heap_end (UND),
    entry_symbol (SHN_ABS = entry_vaddr). Relocations: R_X86_64_64 at
    the vaddr of each recorded movabs immediate, against the matching
    UND symbol.
    """
    while len(out) % 8:
        out.append(0)

    # --- .strtab ---
    strtab = b"\x00"
    heap_start_name = len(strtab)
    strtab += HEAP_START_SYM.encode() + b"\x00"
    heap_end_name = len(strtab)
    strtab += HEAP_END_SYM.encode() + b"\x00"
    entry_name = len(strtab)
    strtab += entry_symbol.encode() + b"\x00"
    strtab_off = len(out)
    out += strtab

    # --- .symtab (24 bytes per entry) ---
    def sym(name_off, info, shndx, value):
        e = bytearray()
        e += name_off.to_bytes(4, "little")  # st_name
        e += bytes([info, 0])  # st_info, st_other
        e += shndx.to_bytes(2, "little")  # st_shndx
        e += value.to_bytes(8, "little")  # st_value
        e += (0).to_bytes(8, "little")  # st_size
        return bytes(e)

    symtab = b"\x00" * 24  # null entry
    symtab += sym(heap_start_name, (STB_GLOBAL << 4) | STT_NOTYPE,
                  SHN_UNDEF, 0)
    symtab += sym(heap_end_name, (STB_GLOBAL << 4) | STT_NOTYPE,
                  SHN_UNDEF, 0)
    symtab += sym(entry_name, (STB_GLOBAL << 4) | STT_FUNC,
                  SHN_ABS, entry_vaddr)
    symtab_off = len(out)
    out += symtab
    # symbol indices: 1 = heap_start, 2 = heap_end, 3 = entry
    sym_index = {HEAP_START_SYM: 1, HEAP_END_SYM: 2}

    # --- .rela.text (24 bytes per entry) ---
    rela = bytearray()
    for text_pos, name in sym_relocs:
        idx = sym_index[name]  # KeyError = compiler bug, loud
        rela += (VBASE + TEXT_OFF + text_pos).to_bytes(8, "little")
        rela += ((idx << 32) | R_X86_64_64).to_bytes(8, "little")
        rela += (0).to_bytes(8, "little", signed=True)  # r_addend
    rela_off = len(out)
    out += bytes(rela)

    # --- .shstrtab ---
    shstrtab = b"\x00"
    sh_symtab = len(shstrtab)
    shstrtab += b".symtab\x00"
    sh_strtab = len(shstrtab)
    shstrtab += b".strtab\x00"
    sh_shstrtab = len(shstrtab)
    shstrtab += b".shstrtab\x00"
    sh_rela = len(shstrtab)
    shstrtab += b".rela.text\x00"
    shstrtab_off = len(out)
    out += shstrtab

    # --- section headers (64 bytes each) ---
    while len(out) % 8:
        out.append(0)
    shoff = len(out)

    def shdr(name, shtype, flags, addr, off, size, link, info, align,
             entsize):
        h = bytearray()
        h += name.to_bytes(4, "little")
        h += shtype.to_bytes(4, "little")
        h += flags.to_bytes(8, "little")
        h += addr.to_bytes(8, "little")
        h += off.to_bytes(8, "little")
        h += size.to_bytes(8, "little")
        h += link.to_bytes(4, "little")
        h += info.to_bytes(4, "little")
        h += align.to_bytes(8, "little")
        h += entsize.to_bytes(8, "little")
        return bytes(h)

    out += shdr(0, 0, 0, 0, 0, 0, 0, 0, 0, 0)  # null
    out += shdr(sh_symtab, SHT_SYMTAB, 0, 0, symtab_off, len(symtab),
                2, 1, 8, 24)  # link=.strtab, info=1 local
    out += shdr(sh_strtab, SHT_STRTAB, 0, 0, strtab_off, len(strtab),
                0, 0, 1, 0)
    out += shdr(sh_shstrtab, SHT_STRTAB, 0, 0, shstrtab_off,
                len(shstrtab), 0, 0, 1, 0)
    out += shdr(sh_rela, SHT_RELA, 0, 0, rela_off, len(rela),
                1, 0, 8, 24)  # link=.symtab

    # --- patch the ELF header ---
    out[40:48] = shoff.to_bytes(8, "little")  # e_shoff
    out[58:60] = (64).to_bytes(2, "little")  # e_shentsize
    out[60:62] = (5).to_bytes(2, "little")  # e_shnum
    out[62:64] = (3).to_bytes(2, "little")  # e_shstrndx
