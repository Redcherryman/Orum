"""AST nodes for Orum v0.001.

The tree is produced by orum.parser and is deliberately simple:
statements, expressions, and types as plain dataclasses.
Dataclass equality lets tests compare whole programs directly.
"""
from __future__ import annotations

from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Top level
# ---------------------------------------------------------------------------

@dataclass
class Program:
    statements: list  # list[statement]


# ---------------------------------------------------------------------------
# Statements
# ---------------------------------------------------------------------------

@dataclass
class FnDef:
    name: str
    params: list  # list[Param]
    ret: object  # type node, or None
    body: list  # list[statement]


@dataclass
class Param:
    name: str
    type: object  # type node
    mutable: bool = False  # v0.003: `mut` params opt into change


@dataclass
class Binding:
    """A declaration: `name = expr`, `mut name = expr`,
    `name: type = expr`, `mut name: type = expr`."""
    name: str
    type: object  # type node, or None
    value: object  # expression
    mutable: bool = False


@dataclass
class Assign:
    """Re-assignment to an already-declared name, field, or index."""
    target: object  # Var | FieldAccess | Index
    value: object  # expression


@dataclass
class Share:
    """`share name = expr` — export a name from this module."""
    name: str
    value: object  # expression


@dataclass
class Take:
    """`take a, b from mod.path` — import names from another module."""
    names: list  # list[str]
    module: list  # list[str], dotted path parts


@dataclass
class StructDef:
    name: str
    fields: list  # list[StructField]


@dataclass
class StructField:
    name: str
    type: object  # type node


@dataclass
class If:
    cond: object  # expression
    then_body: list  # list[statement]
    else_body: list  # list[statement]


@dataclass
class While:
    cond: object  # expression
    body: list  # list[statement]


@dataclass
class For:
    var: str
    iter: object  # expression
    body: list  # list[statement]


@dataclass
class MatchArm:
    kind: str  # 'ok' | 'err'
    binding: str
    body: object  # a single statement


@dataclass
class Match:
    subject: object  # expression
    arms: list  # list[MatchArm]


@dataclass
class Return:
    value: object  # expression, or None


@dataclass
class Crash:
    message: object  # expression


@dataclass
class AsmIn:
    """`in <var> = <reg>` — copy Orum integer variable into a register."""
    var: str
    reg: str  # register name as written (e.g. "rax", "edx")


@dataclass
class AsmOut:
    """`out <var> = <reg>` — copy a register into a `mut` Orum variable."""
    var: str
    reg: str


@dataclass
class AsmClobber:
    """`clobber "<reg>"` — declared-trashed register (trusted, not verified)."""
    reg: str


@dataclass
class AsmBlock:
    """`asm ... end` — inline machine code.

    Emission order is ALWAYS: in-moves, raw bytes verbatim, out-moves,
    regardless of source order of the directives. The block may trash any
    register except rsp/rbp; values cross the boundary only via in/out.
    """
    ins: list      # list[AsmIn]
    outs: list     # list[AsmOut]
    clobbers: list  # list[AsmClobber]
    raw_bytes: list  # list[int], each 0-255, emitted verbatim


@dataclass
class ExprStmt:
    expr: object  # expression


# ---------------------------------------------------------------------------
# Expressions
# ---------------------------------------------------------------------------

@dataclass
class BinOp:
    op: str  # '+', '-', '*', '/', '%', '==', '!=', '<', '<=', '>', '>=',
             # '&', '|', '^', '<<', '>>' (v0.004: bitwise, C-like precedence)
    left: object
    right: object


@dataclass
class UnaryOp:
    op: str  # '-', '~' (v0.004: '~' is bitwise not)
    operand: object


@dataclass
class Call:
    func: object  # expression (Var, or FieldAccess for method calls)
    args: list  # list[expression]


@dataclass
class StructLit:
    """`Point(x: 10, y: 20)` — all-named-argument call form."""
    name: str
    fields: list  # list[tuple[str, expression]]


@dataclass
class FieldAccess:
    obj: object  # expression
    field: str


@dataclass
class Index:
    obj: object  # expression
    index: object  # expression
    # v0.004 VOLATILITY: set by the typechecker when `obj` is `raw *T`.
    # A volatile access must NEVER be elided, merged with another access,
    # or reordered — raw pointers exist for hardware/MMIO, and the
    # hardware can change the value (or care about the write) at any time.
    # The codegen emits exactly one load/store per volatile access and no
    # current or future pass may touch them. (Plain attribute — dataclass
    # equality is unaffected.)
    volatile: bool = False


@dataclass
class ListNew:
    """`list[T]()` — construct a new empty growable list of T."""
    elem: object  # type node


@dataclass
class Var:
    name: str


@dataclass
class IntLit:
    value: int


@dataclass
class StrLit:
    value: str


@dataclass
class InterpString:
    """`"hello $name"` — parts are StrLit or Var."""
    parts: list  # list[StrLit | Var]


# ---------------------------------------------------------------------------
# Types (parsed, not checked, in v0.001)
# ---------------------------------------------------------------------------

@dataclass
class NamedType:
    name: str  # 'int', 'u32', 'str', user struct names, ...
    args: list = field(default_factory=list)  # type/int args for other generics


@dataclass
class RawType:
    """`raw *T` — the exempt raw pointer type."""
    inner: object  # type node


@dataclass
class ListType:
    """`list[T]` — growable heap list."""
    elem: object  # type node


@dataclass
class ArrayType:
    """`array[T, N]` — fixed-size inline array."""
    elem: object  # type node
    size: int


@dataclass
class UnionType:
    """`A | B` — e.g. `string | error`."""
    options: list  # list[type node]


@dataclass
class FnType:
    """`fn(T, U) -> R` — first-class function type."""
    params: list  # list[type node]
    ret: object  # type node
