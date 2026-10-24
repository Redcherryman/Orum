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

import dataclasses
import os
import stat

from . import ast_nodes as A
from . import elf
from .codegen import Codegen
from .lexer import Lexer
from .parser import Parser


def _decl_name(stmt) -> str | None:
    """Declaration name for top-level named statements, else None."""
    if isinstance(stmt, (A.FnDef, A.StructDef, A.Binding, A.Share)):
        return stmt.name
    return None


def _collect_refs(node, out: set):
    """Collect identifier references from an AST node (Var names, type
    names, struct literal names). Generic walk over dataclass fields."""
    if node is None:
        return
    if isinstance(node, A.Var):
        out.add(node.name)
        return
    if isinstance(node, A.NamedType):
        out.add(node.name)
        return
    if isinstance(node, A.StructLit):
        out.add(node.name)
    if dataclasses.is_dataclass(node) and not isinstance(node, type):
        for f in dataclasses.fields(node):
            _collect_refs(getattr(node, f.name), out)
        return
    if isinstance(node, (list, tuple)):
        for item in node:
            _collect_refs(item, out)


def _decl_refs(stmt) -> set:
    """Names a declaration references (body, value, fields, signature)."""
    refs: set = set()
    if isinstance(stmt, A.FnDef):
        for p in stmt.params:
            _collect_refs(p.type, refs)
        _collect_refs(stmt.ret, refs)
        for s in stmt.body:
            _collect_refs(s, refs)
    elif isinstance(stmt, A.StructDef):
        for f in stmt.fields:
            _collect_refs(f.type, refs)
    elif isinstance(stmt, A.Binding):
        _collect_refs(stmt.type, refs)
        _collect_refs(stmt.value, refs)
    elif isinstance(stmt, A.Share):
        _collect_refs(stmt.value, refs)
    return refs


def resolve_takes(prog: A.Program, base_dir: str, seen: dict = None,
                  stack: list = None) -> A.Program:
    """Resolve `take x from mod.path` imports by loading module files.

    v0.016: takes are FILTERED, not whole-module inclusion. For
    `take x from foo.bar`, loads `foo/bar.orum` relative to base_dir and
    includes only x plus the declarations x transitively references
    (within that module). A module's own `take`s are resolved recursively
    first, so cross-module dependencies still arrive. Diamond imports
    (two files taking different names from one module) include each
    declaration once: `seen` maps module path -> included decl names.
    Taking a name the module does not declare is an error.
    """
    if seen is None:
        seen = {}
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
            if not os.path.exists(mod_path):
                raise FileNotFoundError(f"module not found: {mod_path} (for `take` from {stmt.module})")
            stack.append(mod_path)
            try:
                with open(mod_path) as f:
                    mod_source = f.read()
                mod_prog = Parser(Lexer(mod_source).tokenize()).parse_program()
                mod_takes = [s for s in mod_prog.statements if isinstance(s, A.Take)]
                mod_own = [s for s in mod_prog.statements if not isinstance(s, A.Take)]
                # Resolve the module's own takes first (deps before dependents).
                if mod_takes:
                    dep_prog = A.Program(statements=mod_takes)
                    dep_prog = resolve_takes(dep_prog, base_dir, seen, stack)
                    new_stmts.extend(dep_prog.statements)
                # Filter the module's own declarations to the taken names
                # plus their transitive dependencies within this module.
                own_decls = {}
                for s in mod_own:
                    n = _decl_name(s)
                    if n is not None:
                        own_decls[n] = s
                for name in stmt.names:
                    if name not in own_decls:
                        raise ValueError(
                            f"module {'.'.join(stmt.module)} does not define '{name}'"
                        )
                closure = set(stmt.names)
                work = list(stmt.names)
                while work:
                    cur = work.pop()
                    for ref in _decl_refs(own_decls[cur]):
                        if ref in own_decls and ref not in closure:
                            closure.add(ref)
                            work.append(ref)
                already = seen.setdefault(mod_path, set())
                for s in mod_own:
                    n = _decl_name(s)
                    if n is not None and n in closure and n not in already:
                        new_stmts.append(s)
                        already.add(n)
            finally:
                stack.pop()
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
