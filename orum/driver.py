"""Driver: the full Orum pipeline in one call.

    source text
      -> lexer -> parser -> AST
      -> typechecker (annotates + enforces)
      -> codegen (x86-64 machine code + .rodata)
      -> ELF writer (runnable binary bytes)

v0.006: freestanding mode (no syscalls, no brk heap, configurable
entry symbol) for kernel work.
v0.013: separate compilation phase 1 — `take` imports via file inclusion.
"""
from __future__ import annotations

import os
import stat

from . import ast_nodes as A
from . import elf
from .codegen import Codegen
from .lexer import Lexer
from .parser import Parser


def resolve_takes(prog: A.Program, base_dir: str, seen: set = None,
                  stack: list = None) -> A.Program:
    """Resolve `take x from mod.path` imports by loading module files.

    v0.013 phase 1: file inclusion (not true .o linking yet).
    For `take x from foo.bar`, loads `foo/bar.orum` relative to base_dir,
    parses it, and prepends its statements (excluding `take`s, which are
    resolved recursively). The `x` names are not filtered yet — whole
    module is included (like C #include).

    v0.015: import cycles are REPORTED, not silently skipped. `seen`
    tracks fully-loaded modules (diamond imports skip safely); `stack`
    tracks modules currently being loaded — a take of a file on the
    stack is a cycle and raises an error.
    """
    if seen is None:
        seen = set()
    if stack is None:
        stack = []
    new_stmts = []
    for stmt in prog.statements:
        if isinstance(stmt, A.Take):
            # Resolve module path to file
            mod_path = os.path.join(base_dir, *stmt.module) + ".orum"
            if mod_path in stack:
                cycle = " -> ".join(stack + [mod_path])
                raise ValueError(f"import cycle detected: {cycle}")
            if mod_path in seen:
                continue  # already fully loaded (diamond import), skip
            if not os.path.exists(mod_path):
                raise FileNotFoundError(f"module not found: {mod_path} (for `take` from {stmt.module})")
            seen.add(mod_path)
            stack.append(mod_path)
            try:
                with open(mod_path) as f:
                    mod_source = f.read()
                mod_prog = Parser(Lexer(mod_source).tokenize()).parse_program()
                # Recursively resolve takes in the module
                mod_prog = resolve_takes(mod_prog, base_dir, seen, stack)
            finally:
                stack.pop()
            # Prepend the module's statements (excluding its takes, already resolved)
            for ms in mod_prog.statements:
                if not isinstance(ms, A.Take):
                    new_stmts.append(ms)
        else:
            new_stmts.append(stmt)
    prog.statements = new_stmts
    return prog
from .typecheck import typecheck


def compile_source(source: str, freestanding: bool = False,
                   entry: str = "_start", base_dir: str = None) -> bytes:
    """Compile Orum source to an ELF64 binary (as bytes).

    freestanding: no syscalls, heap via __orum_heap_start /
    __orum_heap_end externs, entry symbol configurable via `entry`.
    base_dir: directory for resolving `take` module imports.
    """
    prog = Parser(Lexer(source).tokenize()).parse_program()
    # v0.013: resolve `take` imports (separate compilation, phase 1: file inclusion)
    if base_dir:
        prog = resolve_takes(prog, base_dir)
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


def compile_object(source: str, freestanding: bool = False) -> bytes:
    """Compile Orum source to an Orum object file (.o) for linking.

    v0.013 phase 2: true separate compilation.
    Returns object file bytes (Orum-specific format, see objfile.py).
    """
    from . import objfile
    prog = Parser(Lexer(source).tokenize()).parse_program()
    from .optimizer import fold_constants
    prog = fold_constants(prog)
    typecheck(prog, freestanding=freestanding, allow_undefined=True)
    cg = Codegen(prog, freestanding=freestanding, entry_symbol="_start")
    # Enable object mode (allow undefined functions)
    cg.object_mode = True
    (text, rodata, refs, entry_off, data, data_refs, sym_refs) = cg.compile()
    
    # Build symbol table: function names -> offsets
    symbols = []
    for fname, label in cg.fn_labels.items():
        if label in cg.e.labels:
            offset = cg.e.labels[label]
            symbols.append((fname, offset, 0))  # 0 = func kind
    
    # Get relocations (undefined function calls)
    relocs = getattr(cg, 'obj_relocs', [])

    # rodata_refs: (text_pos, rodata_off) pairs from codegen — positions
    # in .text where a 64-bit rodata address must be patched at link time.
    # v0.015: stored in the object so the linker can resolve cross-object
    # string/data references.
    # data_refs: (text_pos, data_off) pairs — heap_ptr/heap_end references
    # for the shared heap. v0.015: stored so the linker can point all
    # objects at one shared data section.
    return objfile.write_object(symbols, relocs, text, rodata, refs, data_refs)


def build_binary(source: str, path: str, freestanding: bool = False,
                 entry: str = "_start") -> str:
    """Compile Orum source and write an executable file. Returns path."""
    data = compile_source(source, freestanding=freestanding, entry=entry)
    with open(path, "wb") as f:
        f.write(data)
    mode = os.stat(path).st_mode
    os.chmod(path, mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    return path
