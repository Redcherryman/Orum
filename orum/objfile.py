"""Orum object file format (v0.013 phase 2, v2 in v0.015).

Simple relocatable object format for separate compilation.
Not ELF — Orum-specific, designed for our simple linker.

Format v2:
- Magic: b'ORUMOBJ\x00'
- u32: version (2)
- u32: num_symbols
- For each symbol:
  - u8: name_len, bytes: name
  - u64: offset in .text (or 0xFFFFFFFFFFFFFFFF for UNDEF)
  - u8: kind (0=func, 1=data)
- u32: num_relocs
- For each reloc:
  - u64: offset in .text (where the 32-bit rel32 is)
  - u8: name_len, bytes: symbol name
- u32: num_rodata_refs (v2 addition: cross-object string/data refs)
- For each rodata_ref:
  - u64: offset in .text (where the 64-bit address placeholder is)
  - u64: offset in this object's .rodata
- u32: num_data_refs (v2 addition: heap_ptr/heap_end refs for shared heap)
- For each data_ref:
  - u64: offset in .text (where the 64-bit address placeholder is)
  - u64: offset in the shared .data section (heap_ptr=+0, heap_end=+8)
- u64: text_size, bytes: .text
- u64: rodata_size, bytes: .rodata

v1 files (without rodata_refs/data_refs) are rejected: recompile with
the current compiler. The format is internal and not yet stable.
"""
import struct

MAGIC = b'ORUMOBJ\x00'
VERSION = 2
UNDEF = 0xFFFFFFFFFFFFFFFF


def write_object(symbols, relocs, text, rodata, rodata_refs=(), data_refs=()):
    """symbols: list of (name, offset, kind). relocs: list of (offset, name).
    rodata_refs: list of (text_offset, rodata_offset) pairs.
    data_refs: list of (text_offset, data_offset) pairs."""
    out = bytearray()
    out += MAGIC
    out += struct.pack('<I', VERSION)
    out += struct.pack('<I', len(symbols))
    for name, offset, kind in symbols:
        nb = name.encode()
        out += struct.pack('<B', len(nb))
        out += nb
        out += struct.pack('<Q', offset)
        out += struct.pack('<B', kind)
    out += struct.pack('<I', len(relocs))
    for offset, name in relocs:
        nb = name.encode()
        out += struct.pack('<Q', offset)
        out += struct.pack('<B', len(nb))
        out += nb
    out += struct.pack('<I', len(rodata_refs))
    for text_off, rodata_off in rodata_refs:
        out += struct.pack('<Q', text_off)
        out += struct.pack('<Q', rodata_off)
    out += struct.pack('<I', len(data_refs))
    for text_off, data_off in data_refs:
        out += struct.pack('<Q', text_off)
        out += struct.pack('<Q', data_off)
    out += struct.pack('<Q', len(text))
    out += text
    out += struct.pack('<Q', len(rodata))
    out += rodata
    return bytes(out)


def read_object(data):
    """Returns (symbols, relocs, text, rodata, rodata_refs, data_refs)."""
    pos = 0
    assert data[pos:pos+8] == MAGIC, "not an Orum object"
    pos += 8
    ver, = struct.unpack('<I', data[pos:pos+4])
    pos += 4
    if ver != VERSION:
        raise ValueError(
            f"unsupported Orum object version {ver} (want {VERSION}); "
            "recompile with the current compiler")
    num_sym, = struct.unpack('<I', data[pos:pos+4])
    pos += 4
    symbols = []
    for _ in range(num_sym):
        nl = data[pos]
        pos += 1
        name = data[pos:pos+nl].decode()
        pos += nl
        offset, = struct.unpack('<Q', data[pos:pos+8])
        pos += 8
        kind = data[pos]
        pos += 1
        symbols.append((name, offset, kind))
    num_reloc, = struct.unpack('<I', data[pos:pos+4])
    pos += 4
    relocs = []
    for _ in range(num_reloc):
        offset, = struct.unpack('<Q', data[pos:pos+8])
        pos += 8
        nl = data[pos]
        pos += 1
        name = data[pos:pos+nl].decode()
        pos += nl
        relocs.append((offset, name))
    num_rrefs, = struct.unpack('<I', data[pos:pos+4])
    pos += 4
    rodata_refs = []
    for _ in range(num_rrefs):
        text_off, = struct.unpack('<Q', data[pos:pos+8])
        pos += 8
        rodata_off, = struct.unpack('<Q', data[pos:pos+8])
        pos += 8
        rodata_refs.append((text_off, rodata_off))
    num_drefs, = struct.unpack('<I', data[pos:pos+4])
    pos += 4
    data_refs = []
    for _ in range(num_drefs):
        text_off, = struct.unpack('<Q', data[pos:pos+8])
        pos += 8
        data_off, = struct.unpack('<Q', data[pos:pos+8])
        pos += 8
        data_refs.append((text_off, data_off))
    text_size, = struct.unpack('<Q', data[pos:pos+8])
    pos += 8
    text = data[pos:pos+text_size]
    pos += text_size
    rodata_size, = struct.unpack('<Q', data[pos:pos+8])
    pos += 8
    rodata = data[pos:pos+rodata_size]
    return symbols, relocs, text, rodata, rodata_refs, data_refs
