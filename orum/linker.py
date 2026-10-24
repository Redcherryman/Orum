"""Orum linker (v0.013 phase 2).

Links multiple Orum object files (.o) into an executable ELF.
"""
import struct
import sys
from . import objfile
from . import elf


def link(objects, entry_symbol="_start", freestanding=False):
    """objects: list of (symbols, relocs, text, rodata) tuples.
    Returns ELF bytes.
    """
    # Build global symbol table: name -> (text_offset, obj_idx)
    # First, layout the .text sections
    text_base = 0
    obj_text_bases = []
    global_syms = {}  # name -> absolute offset in combined .text
    
    for idx, (symbols, relocs, text, rodata) in enumerate(objects):
        obj_text_bases.append(text_base)
        for name, offset, kind in symbols:
            if offset != objfile.UNDEF:
                # Defined symbol
                if name in global_syms:
                    raise ValueError(f"duplicate symbol: {name}")
                global_syms[name] = text_base + offset
        # Align to 16 bytes for next object
        text_base += len(text)
        while text_base % 16:
            text_base += 1
    
    # Combine .text and .rodata
    combined_text = bytearray()
    combined_rodata = bytearray()
    rodata_base = text_base  # rodata follows text
    
    # Track rodata offsets per object
    obj_rodata_bases = []
    for idx, (symbols, relocs, text, rodata) in enumerate(objects):
        # Pad text to alignment
        while len(combined_text) < obj_text_bases[idx]:
            combined_text.append(0x90)  # nop
        combined_text += text
        obj_rodata_bases.append(len(combined_rodata))
        combined_rodata += rodata
    
    # Apply relocations
    # For each reloc (offset, name) in each object:
    # - Find the target symbol address
    # - Patch the rel32 at (obj_text_base + offset)
    for idx, (symbols, relocs, text, rodata) in enumerate(objects):
        base = obj_text_bases[idx]
        for rel_offset, sym_name in relocs:
            if sym_name not in global_syms:
                raise ValueError(f"undefined symbol: {sym_name}")
            target = global_syms[sym_name]
            patch_pos = base + rel_offset
            # rel32 = target - (patch_pos + 4)
            rel32 = target - (patch_pos + 4)
            combined_text[patch_pos:patch_pos+4] = struct.pack('<i', rel32)
    
    # Find entry point
    if entry_symbol not in global_syms:
        # Try with fn_ prefix (Orum mangles function names)
        mangled = f"fn_{entry_symbol}" if not entry_symbol.startswith("fn_") else entry_symbol
        if mangled in global_syms:
            entry_symbol = mangled
        else:
            raise ValueError(f"entry symbol not found: {entry_symbol}")
    entry_off = global_syms[entry_symbol]
    
    # Output ELF using the existing writer
    # For simplicity, use the freestanding ELF writer with our combined sections
    return elf.write_elf(
        bytes(combined_text),
        bytes(combined_rodata),
        [],  # rodata_refs (TODO: handle)
        entry_off,
        b"",  # data
        [],
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
