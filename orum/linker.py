"""Orum linker (v0.013 phase 2, extended v0.015).

Links multiple Orum object files (.o) into an executable ELF.

v0.015 additions:
- rodata_refs: cross-object string/data references are patched.
- Shared heap: all objects share one data section (heap_ptr at +0,
  heap_end at +8), so lists and heap code work across module
  boundaries. Function entries get a stub that runs heap init before
  calling the entry function.
- Function entries: jumping directly to a function segfaults on
  return, so the linker appends a call-and-exit stub.
"""
import struct
import sys
from . import objfile
from . import elf
from .x86 import Emitter, RAX, RCX, RDI

HEAP_SIZE = 1 << 20  # 1 MiB, must match codegen.HEAP_SIZE


def _build_entry_stub(fn_addr, stub_pos, data_vaddr, freestanding):
    """Build the entry stub bytes for a function entry.

    Hosted: heap_init (brk) + call fn + exit(return value).
    Freestanding: call fn + halt (heap comes from extern symbols;
    freestanding linked heap init is the kernel's job via the
    __orum_heap_start/__orum_heap_end symbols, resolved by the
    runtime, not by this stub).
    Returns bytes. Length is deterministic for a given mode.
    """
    e = Emitter()
    if not freestanding:
        # heap_init (hosted, mirrors codegen.gen_heap_init):
        #   rax = brk(0); heap_ptr = rax
        #   heap_end = rax + 1MiB; brk(heap_end)
        e.mov_ri64(RAX, 12)  # sys_brk
        e.xor_rr(RDI, RDI)
        e.syscall()
        # movabs rcx, data_vaddr; mov [rcx], rax
        e.mov_ri64(RCX, data_vaddr)
        e.mov_mr(RCX, 0, RAX)
        e.mov_rr(RDI, RAX)
        e.add_ri(RDI, HEAP_SIZE)
        e.mov_mr(RCX, 8, RDI)  # heap_end at +8
        e.mov_ri64(RAX, 12)
        e.syscall()
        # (skip the brk-failure check here; heap ops will fault loudly
        # if brk failed, same as a failed allocation.)
    # call fn_addr: E8 + rel32, rel32 = target - (call_pos + 5).
    # NOTE: call_pos for the rel32 math is the ABSOLUTE position in
    # combined text, but the patch index into e.code is RELATIVE.
    rel_call_pos = len(e.code)
    e.code += b"\xE8\x00\x00\x00\x00"
    abs_call_pos = stub_pos + rel_call_pos
    rel32 = fn_addr - (abs_call_pos + 5)
    e.code[rel_call_pos + 1:rel_call_pos + 5] = struct.pack('<i', rel32)
    if freestanding:
        e.code += b"\xFA"  # cli
        e.code += b"\xF4"  # hlt
        e.code += b"\xEB\xFE"  # jmp hlt
    else:
        # exit(return value): rdi = rax, rax = 60, syscall
        e.mov_rr(RDI, RAX)
        e.mov_ri64(RAX, 60)
        e.syscall()
    return bytes(e.code)


def link(objects, entry_symbol="_start", freestanding=False):
    """objects: list of (symbols, relocs, text, rodata, rodata_refs,
    data_refs) tuples. Returns ELF bytes.
    """
    # Normalize: accept 4-tuples (v1), 5-tuples (v2 rodata only),
    # and 6-tuples (v2 full).
    normed = []
    for obj in objects:
        if len(obj) == 4:
            symbols, relocs, text, rodata = obj
            normed.append((symbols, relocs, text, rodata, [], []))
        elif len(obj) == 5:
            symbols, relocs, text, rodata, rodata_refs = obj
            normed.append((symbols, relocs, text, rodata, rodata_refs, []))
        else:
            normed.append(obj)
    objects = normed

    # Build global symbol table: name -> absolute offset in combined .text
    text_base = 0
    obj_text_bases = []
    global_syms = {}  # name -> absolute offset in combined .text
    global_kinds = {}  # name -> kind (0 = func, from objfile)

    for idx, (symbols, relocs, text, rodata, rodata_refs, data_refs) in enumerate(objects):
        obj_text_bases.append(text_base)
        for name, offset, kind in symbols:
            if offset != objfile.UNDEF:
                if name in global_syms:
                    raise ValueError(f"duplicate symbol: {name}")
                global_syms[name] = text_base + offset
                global_kinds[name] = kind
        text_base += len(text)
        while text_base % 16:
            text_base += 1

    # Combine .text and .rodata
    combined_text = bytearray()
    combined_rodata = bytearray()
    obj_rodata_bases = []
    for idx, (symbols, relocs, text, rodata, rodata_refs, data_refs) in enumerate(objects):
        while len(combined_text) < obj_text_bases[idx]:
            combined_text.append(0x90)  # nop
        combined_text += text
        obj_rodata_bases.append(len(combined_rodata))
        combined_rodata += rodata

    # Apply function-call relocations
    for idx, (symbols, relocs, text, rodata, rodata_refs, data_refs) in enumerate(objects):
        base = obj_text_bases[idx]
        for rel_offset, sym_name in relocs:
            if sym_name not in global_syms:
                raise ValueError(f"undefined symbol: {sym_name}")
            target = global_syms[sym_name]
            patch_pos = base + rel_offset
            rel32 = target - (patch_pos + 4)
            combined_text[patch_pos:patch_pos+4] = struct.pack('<i', rel32)

    # Collect rodata refs, adjusted for each object's bases
    combined_rodata_refs = []
    for idx, (symbols, relocs, text, rodata, rodata_refs, data_refs) in enumerate(objects):
        tbase = obj_text_bases[idx]
        rbase = obj_rodata_bases[idx]
        for text_pos, rodata_off in rodata_refs:
            combined_rodata_refs.append(
                (tbase + text_pos, rbase + rodata_off))

    # Collect data refs (heap_ptr/heap_end). These point into the
    # SHARED data section, not per-object data: every object's
    # heap operations use the same heap_ptr/heap_end slots.
    combined_data_refs = []
    for idx, (symbols, relocs, text, rodata, rodata_refs, data_refs) in enumerate(objects):
        tbase = obj_text_bases[idx]
        for text_pos, data_off in data_refs:
            combined_data_refs.append((tbase + text_pos, data_off))

    # Find entry point
    entry_is_fn = False
    if entry_symbol not in global_syms:
        mangled = f"fn_{entry_symbol}" if not entry_symbol.startswith("fn_") else entry_symbol
        if mangled in global_syms:
            entry_symbol = mangled
            entry_is_fn = True
        else:
            raise ValueError(f"entry symbol not found: {entry_symbol}")
    elif global_kinds.get(entry_symbol) == 0:
        entry_is_fn = True
    entry_off = global_syms[entry_symbol]

    # Shared data section: 16 bytes (heap_ptr at +0, heap_end at +8).
    # The ELF writer places data on its own page after rodata.
    shared_data = bytes(16)

    if entry_is_fn:
        # Append a call-and-exit stub (with heap init for hosted).
        # data_vaddr depends on final text length including the stub,
        # so build once with a dummy to measure, then rebuild real.
        fn_addr = entry_off
        stub_pos = len(combined_text)
        dummy = _build_entry_stub(fn_addr, stub_pos, 0, freestanding)
        # Compute data_vaddr the same way elf.write_elf does:
        #   rodata_off = TEXT_OFF + len(text), aligned to 8
        #   data_off = rodata_off + len(rodata), aligned to page
        #   data_vaddr = VBASE + data_off
        text_len = len(combined_text) + len(dummy)
        rodata_off = elf.TEXT_OFF + text_len
        while rodata_off % 8:
            rodata_off += 1
        data_off = rodata_off + len(combined_rodata)
        while data_off % 0x1000:
            data_off += 1
        data_vaddr = elf.VBASE + data_off
        stub = _build_entry_stub(fn_addr, stub_pos, data_vaddr, freestanding)
        assert len(stub) == len(dummy), "stub length must be stable"
        combined_text += stub
        entry_off = stub_pos

    return elf.write_elf(
        bytes(combined_text),
        bytes(combined_rodata),
        combined_rodata_refs,
        entry_off,
        shared_data,
        combined_data_refs,
        freestanding=freestanding,
        entry_symbol=entry_symbol,
        sym_relocs=[],
    )


def link_files(obj_paths, output_path, entry="_start", freestanding=False):
    """Link .o files to an executable."""
    objects = []
    for p in obj_paths:
        with open(p, 'rb') as f:
            data = f.read()
        objects.append(objfile.read_object(data))
    elf_data = link(objects, entry_symbol=entry, freestanding=freestanding)
    with open(output_path, 'wb') as f:
        f.write(elf_data)
    import os, stat
    mode = os.stat(output_path).st_mode
    os.chmod(output_path, mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    return output_path
