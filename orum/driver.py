"""Driver: the full Orum pipeline in one call.

    source text
      -> lexer -> parser -> AST
      -> typechecker (annotates + enforces)
      -> codegen (x86-64 machine code + .rodata)
      -> ELF writer (runnable binary bytes)

v0.006: freestanding mode (no syscalls, no brk heap, configurable
entry symbol) for kernel work.
"""
from __future__ import annotations

import os
import stat

from . import elf
from .codegen import Codegen
from .lexer import Lexer
from .parser import Parser
from .typecheck import typecheck


def compile_source(source: str, freestanding: bool = False,
                   entry: str = "_start") -> bytes:
    """Compile Orum source to an ELF64 binary (as bytes).

    freestanding: no syscalls, heap via __orum_heap_start /
    __orum_heap_end externs, entry symbol configurable via `entry`.
    """
    prog = Parser(Lexer(source).tokenize()).parse_program()
    # v0.012: constant folding optimizer
    from .optimizer import fold_constants
    prog = fold_constants(prog)
    typecheck(prog, freestanding=freestanding)
    cg = Codegen(prog, freestanding=freestanding, entry_symbol=entry)
    (text, rodata, refs, entry_off, data, data_refs,
     sym_refs) = cg.compile()
    return elf.write_elf(text, rodata, refs, entry_off, data, data_refs,
                         freestanding=freestanding, entry_symbol=entry,
                         sym_relocs=sym_refs)


def build_binary(source: str, path: str, freestanding: bool = False,
                 entry: str = "_start") -> str:
    """Compile Orum source and write an executable file. Returns path."""
    data = compile_source(source, freestanding=freestanding, entry=entry)
    with open(path, "wb") as f:
        f.write(data)
    mode = os.stat(path).st_mode
    os.chmod(path, mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    return path
