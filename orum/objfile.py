"""Orum object file format (v0.013 phase 2).

Simple relocatable object format for separate compilation.
Not ELF — Orum-specific, designed for our simple linker.

Format:
- Magic: b'ORUMOBJ\x00'
- u32: version (1)
- u32: num_symbols
- For each symbol:
  - u8: name_len, bytes: name
  - u64: offset in .text (or 0xFFFFFFFFFFFFFFFF for UNDEF)
  - u8: kind (0=func, 1=data)
- u32: num_relocs
- For each reloc:
  - u64: offset in .text (where the 32-bit rel32 is)
  - u8: name_len, bytes: symbol name
- u64: text_size, bytes: .text
- u64: rodata_size, bytes: .rodata
"""
import struct

MAGIC = b'ORUMOBJ\x00'
VERSION = 1
UNDEF = 0xFFFFFFFFFFFFFFFF


def write_object(symbols, relocs, text, rodata):
    """symbols: list of (name, offset, kind). relocs: list of (offset, name)."""
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
    out += struct.pack('<Q', len(text))
    out += text
    out += struct.pack('<Q', len(rodata))
    out += rodata
    return bytes(out)


def read_object(data):
    """Returns (symbols, relocs, text, rodata)."""
    pos = 0
    assert data[pos:pos+8] == MAGIC, "not an Orum object"
    pos += 8
    ver, = struct.unpack('<I', data[pos:pos+4])
    pos += 4
    assert ver == VERSION
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
    text_size, = struct.unpack('<Q', data[pos:pos+8])
    pos += 8
    text = data[pos:pos+text_size]
    pos += text_size
    rodata_size, = struct.unpack('<Q', data[pos:pos+8])
    pos += 8
    rodata = data[pos:pos+rodata_size]
    return symbols, relocs, text, rodata
