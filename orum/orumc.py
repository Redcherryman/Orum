#!/usr/bin/env python3
"""orumc — the Orum compiler command line.

Usage:
    orumc program.orum [-o OUTPUT] [--freestanding] [--entry SYMBOL]

--freestanding: kernel mode. No syscalls, heap via the extern symbols
__orum_heap_start / __orum_heap_end (to be resolved at link time),
`say`/`exit` rejected, `crash` halts the CPU.
--entry SYMBOL: ELF entry symbol (default _start).
"""
from __future__ import annotations

import argparse
import os
import sys

from .driver import build_binary


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        prog="orumc", description="Compile Orum source to an ELF64 binary.")
    ap.add_argument("source", help="Orum source file (.orum)")
    ap.add_argument("-o", "--output", default=None,
                    help="output path (default: input file without .orum)")
    ap.add_argument("--freestanding", action="store_true",
                    help="freestanding (kernel) mode: no syscalls")
    ap.add_argument("--entry", default="_start",
                    help="ELF entry symbol (default: _start)")
    args = ap.parse_args(argv)

    with open(args.source, "r", encoding="utf-8") as f:
        source = f.read()
    out = args.output
    if out is None:
        base = args.source
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
