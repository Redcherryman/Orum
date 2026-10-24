#!/usr/bin/env python3
"""orumc — the Orum compiler command line.

Usage:
    orumc program.orum [-o OUTPUT] [--freestanding] [--entry SYMBOL]
    orumc -c program.orum [-o program.o] [--freestanding]
    orumc --link a.o b.o [-o OUTPUT] [--freestanding] [--entry SYMBOL]

--freestanding: kernel mode. No syscalls, heap via the extern symbols
__orum_heap_start / __orum_heap_end (to be resolved at link time),
`say`/`exit` rejected, `crash` halts the CPU.
--entry SYMBOL: ELF entry symbol (default _start).
-c / --compile-only: compile a single .orum source to an Orum object
file (.o) for later linking, instead of a full executable.
--link: link one or more Orum object files (.o) into an executable.
"""
from __future__ import annotations

import argparse
import os
import sys

from .driver import build_binary, compile_object
from .linker import link_files


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        prog="orumc", description="Compile Orum source to an ELF64 binary.")
    ap.add_argument("source", nargs="+",
                    help="Orum source file (.orum), or object files (.o) with --link")
    ap.add_argument("-o", "--output", default=None,
                    help="output path (default: input file without .orum, or a.out for --link)")
    ap.add_argument("--freestanding", action="store_true",
                    help="freestanding (kernel) mode: no syscalls")
    ap.add_argument("--entry", default="_start",
                    help="ELF entry symbol (default: _start)")
    ap.add_argument("-c", "--compile-only", action="store_true",
                    help="compile to an Orum object (.o) instead of an executable")
    ap.add_argument("--link", action="store_true",
                    help="link Orum object files (.o) into an executable")
    args = ap.parse_args(argv)

    if args.link:
        # Link mode: all positionals are .o files.
        out = args.output or "a.out"
        try:
            link_files(args.source, out, entry=args.entry,
                       freestanding=args.freestanding)
        except Exception as exc:
            print(f"orumc: error: {exc}", file=sys.stderr)
            return 1
        mode = "freestanding" if args.freestanding else "hosted"
        print(f"orumc: linked {out} ({mode}, entry {args.entry})")
        return 0

    if len(args.source) != 1:
        print("orumc: error: exactly one source file required "
              "(use --link for multiple objects)", file=sys.stderr)
        return 1
    src_path = args.source[0]

    with open(src_path, "r", encoding="utf-8") as f:
        source = f.read()

    if args.compile_only:
        # Object mode: compile to .o, do not link.
        out = args.output
        if out is None:
            base = src_path
            if base.endswith(".orum"):
                base = base[: -len(".orum")]
            out = base + ".o" if base else "a.o"
        try:
            obj_data = compile_object(source, freestanding=args.freestanding)
        except Exception as exc:
            print(f"orumc: error: {exc}", file=sys.stderr)
            return 1
        with open(out, "wb") as f:
            f.write(obj_data)
        mode = "freestanding" if args.freestanding else "hosted"
        print(f"orumc: wrote {out} (object, {mode})")
        return 0

    # Normal mode: compile to executable.
    out = args.output
    if out is None:
        base = src_path
        if base.endswith(".orum"):
            base = base[: -len(".orum")]
        out = base if base else "a.out"
    try:
        build_binary(source, out, freestanding=args.freestanding,
                     entry=args.entry)
    except Exception as exc:  # lexer/parser/typecheck/codegen errors
        print(f"orumc: error: {exc}", file=sys.stderr)
        return 1
    mode = "freestanding" if args.freestanding else "hosted"
    print(f"orumc: wrote {out} ({mode}, entry {args.entry})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
