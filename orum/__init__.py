"""Orum v0.003 — the working base of the Orum language.

This package is the bootstrap: a lexer, parser, typechecker, and
x86-64/ELF backend that turn Orum source text into runnable Linux
binaries. It is intentionally local-only and will be rewritten in Orum
itself piece by piece until Orum self-hosts.
"""

from .ast_nodes import (
    Program,
    FnDef, Param, Binding, Assign, Share, Take,
    StructDef, StructField, If, While, For,
    Match, MatchArm, Return, Crash, ExprStmt,
    BinOp, UnaryOp, Call, StructLit, FieldAccess, Index, ListNew,
    Var, IntLit, StrLit, InterpString,
    NamedType, RawType, ListType, ArrayType, UnionType, FnType,
)
from .lexer import Lexer, Token, LexError
from .parser import Parser, ParseError, parse
from .typecheck import TypeChecker, TypeCheckError, typecheck
from .codegen import Codegen, CodegenError
from . import x86, elf
from .driver import compile_source, build_binary

__version__ = "0.006"

__all__ = [
    "__version__",
    "parse", "typecheck", "compile_source", "build_binary",
    "Lexer", "Token", "LexError",
    "Parser", "ParseError",
    "TypeChecker", "TypeCheckError",
    "Codegen", "CodegenError",
    "x86", "elf",
    "Program",
    "FnDef", "Param", "Binding", "Assign", "Share", "Take",
    "StructDef", "StructField", "If", "While", "For",
    "Match", "MatchArm", "Return", "Crash", "ExprStmt",
    "BinOp", "UnaryOp", "Call", "StructLit", "FieldAccess", "Index", "ListNew",
    "Var", "IntLit", "StrLit", "InterpString",
    "NamedType", "RawType", "ListType", "ArrayType", "UnionType", "FnType",
]
