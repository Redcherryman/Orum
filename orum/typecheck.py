"""Typechecker for Orum v0.001.

Enforces the decided type system:
- Integer sizes: i8-i64 / u8-u64, `int` = 64-bit signed. Literals adapt to
  an expected type and are range-checked (`x: u8 = 300` is an error).
- `mut` enforcement: immutable by default; assigning to a non-`mut`
  binding is an error. Stores through raw pointers/indexes are exempt.
- Function signatures: arity and argument types.
- Union types `A | B` for error returns; `ok(e)` / `err(e)` construct
  them and are checked against an expected union type (from a return
  annotation or binding annotation). v0.001: union payloads must be
  integer-like.
- `+` on strings is an error (interpolation is the only way to build
  strings); comparisons produce `bool`; `if`/`while` need `bool`.

Every expression node is annotated with `.inferred` (its type) for the
code generator. `ok(...)` / `err(...)` calls are rewritten to Ok / Err
nodes. All errors carry line numbers.
"""
from __future__ import annotations

from . import ast_nodes as A


class TypeCheckError(Exception):
    """A type error, always with a line number."""


INT_TYPES = {
    "i8", "i16", "i32", "i64",
    "u8", "u16", "u32", "u64",
    "int",
}
SIGNED_TYPES = {"i8", "i16", "i32", "i64", "int"}
UNSIGNED_TYPES = {"u8", "u16", "u32", "u64"}

INT_RANGES = {
    "i8": (-128, 127),
    "i16": (-32768, 32767),
    "i32": (-2147483648, 2147483647),
    "i64": (-9223372036854775808, 9223372036854775807),
    "u8": (0, 255),
    "u16": (0, 65535),
    "u32": (0, 4294967295),
    "u64": (0, 18446744073709551615),
    "int": (-9223372036854775808, 9223372036854775807),
}

UNIT = A.NamedType("unit")  # internal: result of `say`, `exit`


# -- inline asm (v0.005) ---------------------------------------------------
# Register numbers follow orum.x86: rax=0 rcx=1 rdx=2 rbx=3 rsp=4 rbp=5
# rsi=6 rdi=7 r8=8 ... r15=15. 32-bit aliases zero-extend on use.
ASM_REG64 = {
    "rax": 0, "rcx": 1, "rdx": 2, "rbx": 3,
    "rsi": 6, "rdi": 7,
    "r8": 8, "r9": 9, "r10": 10, "r11": 11,
    "r12": 12, "r13": 13, "r14": 14, "r15": 15,
}
ASM_REG32 = {
    "eax": 0, "ecx": 1, "edx": 2, "ebx": 3,
    "esi": 6, "edi": 7,
    "r8d": 8, "r9d": 9, "r10d": 10, "r11d": 11,
    "r12d": 12, "r13d": 13, "r14d": 14, "r15d": 15,
}
# Frame registers: the compiler owns the stack frame. Never touchable.
ASM_FORBIDDEN_REGS = {"rsp", "rbp", "esp", "ebp"}


def asm_reg_info(name: str):
    """-> (number, bits) or None for unknown names."""
    if name in ASM_REG64:
        return (ASM_REG64[name], 64)
    if name in ASM_REG32:
        return (ASM_REG32[name], 32)
    return None


def is_int_like(t) -> bool:
    return isinstance(t, A.NamedType) and t.name in INT_TYPES


def is_bool(t) -> bool:
    return isinstance(t, A.NamedType) and t.name == "bool"


def is_str(t) -> bool:
    return isinstance(t, A.NamedType) and t.name == "str"


def types_equal(a, b) -> bool:
    """Structural equality for types."""
    if type(a) is not type(b):
        return False
    if isinstance(a, A.NamedType):
        return a.name == b.name and len(a.args) == len(b.args) and all(
            types_equal(x, y) for x, y in zip(a.args, b.args)
        )
    if isinstance(a, A.RawType):
        return types_equal(a.inner, b.inner)
    if isinstance(a, A.ListType):
        return types_equal(a.elem, b.elem)
    if isinstance(a, A.ArrayType):
        return types_equal(a.elem, b.elem) and a.size == b.size
    if isinstance(a, A.UnionType):
        return len(a.options) == len(b.options) and all(
            types_equal(x, y) for x, y in zip(a.options, b.options)
        )
    if isinstance(a, A.FnType):
        return len(a.params) == len(b.params) and all(
            types_equal(x, y) for x, y in zip(a.params, b.params)
        ) and types_equal(a.ret, b.ret)
    return False


def compatible(param_t, arg_t) -> bool:
    """Can `arg_t` be used where `param_t` is expected?
    Integer-like types are mutually compatible (widths are enforced
    separately on literals; at runtime everything is a 64-bit word)."""
    if is_int_like(param_t) and is_int_like(arg_t):
        return True
    return types_equal(param_t, arg_t)


def const_value(node):
    """The integer value of a compile-time-constant expression,
    or None if it isn't one."""
    if isinstance(node, A.IntLit):
        return node.value
def _to_signed64(u):
    """Interpret a 64-bit word as signed (two's complement), matching
    the runtime: everything is a 64-bit word, `say` prints signed."""
    u &= 0xFFFFFFFFFFFFFFFF
    return u - 0x10000000000000000 if u >= 0x8000000000000000 else u


def const_value(node):
    """The integer value of a compile-time-constant expression,
    or None if it isn't one."""
    if isinstance(node, A.IntLit):
        return node.value
    if isinstance(node, A.UnaryOp) and node.op in ("-", "~"):
        v = const_value(node.operand)
        if v is None:
            return None
        if node.op == "-":
            return -v
        # '~': bitwise not on the 64-bit word, signed interpretation.
        return _to_signed64(~v)
    if isinstance(node, A.BinOp) and node.op in (
            "+", "-", "*", "/", "%", "&", "|", "^", "<<", ">>"):
        l = const_value(node.left)
        r = const_value(node.right)
        if l is None or r is None:
            return None
        if node.op == "+":
            return l + r
        if node.op == "-":
            return l - r
        if node.op == "*":
            return l * r
        if node.op in ("&", "|", "^", "<<", ">>"):
            # Bitwise ops fold on 64-bit words, exactly like the hardware
            # (and the runtime): the shift count is masked to 6 bits, the
            # result is a 64-bit word read back as signed.
            mask = 0xFFFFFFFFFFFFFFFF
            if node.op == "&":
                return _to_signed64(l & r)
            if node.op == "|":
                return _to_signed64(l | r)
            if node.op == "^":
                return _to_signed64(l ^ r)
            n = r & 63
            if node.op == "<<":
                return _to_signed64((l << n) & mask)
            # '>>': arithmetic (sar) when signed, logical (shr) otherwise —
            # same rule the codegen uses (from node.signed).
            if getattr(node, "signed", True):
                # Python >> is arithmetic; work on the signed word.
                return _to_signed64(l) >> n
            return _to_signed64((l & mask) >> n)
        if r == 0:
            return None  # div by zero: reported separately
        if node.op == "/":
            # Orum `/` truncates toward zero, like the hardware idiv
            q = abs(l) // abs(r)
            return q if (l < 0) == (r < 0) else -q
        return l % r if l >= 0 else -(-l % r)
    return None


def type_name(t) -> str:
    if isinstance(t, A.NamedType):
        return t.name
    if isinstance(t, A.RawType):
        return f"raw *{type_name(t.inner)}"
    if isinstance(t, A.ListType):
        return f"list[{type_name(t.elem)}]"
    if isinstance(t, A.ArrayType):
        return f"array[{type_name(t.elem)}, {t.size}]"
    if isinstance(t, A.UnionType):
        return " | ".join(type_name(o) for o in t.options)
    if isinstance(t, A.FnType):
        ps = ", ".join(type_name(p) for p in t.params)
        return f"fn({ps}) -> {type_name(t.ret)}"
    return "?"


class TypeChecker:
    def __init__(self, freestanding: bool = False):
        self.freestanding = freestanding
        self.structs: dict[str, A.StructDef] = {}
        self.functions: dict[str, tuple[list, object]] = {}
        self.var_scopes: list[dict] = [{}]  # name -> (type, mut)
        self.current_fn_ret = None
        self.in_function = False

    # -- errors ---------------------------------------------------------

    def err(self, line, msg) -> TypeCheckError:
        return TypeCheckError(f"{line}: type error: {msg}")

    # -- scopes ----------------------------------------------------------

    def push_scope(self):
        self.var_scopes.append({})

    def pop_scope(self):
        self.var_scopes.pop()

    def declare_var(self, name, type_, mut):
        self.var_scopes[-1][name] = (type_, mut)

    def lookup_var(self, name):
        for scope in reversed(self.var_scopes):
            if name in scope:
                return scope[name]
        return None

    def _is_mut_var(self, name):
        """Is `name` a declared `mut` variable? (Params included.)"""
        found = self.lookup_var(name)
        return found is not None and found[1]

    # -- entry ------------------------------------------------------------

    def check_program(self, prog: A.Program) -> A.Program:
        # Pass 1: register structs and function signatures (order-free).
        for stmt in prog.statements:
            if isinstance(stmt, A.StructDef):
                self.register_struct(stmt)
            elif isinstance(stmt, A.FnDef):
                self.register_function(stmt)
        # Pass 2: check everything.
        for stmt in prog.statements:
            self.check_stmt(stmt)
        return prog

    def register_struct(self, node: A.StructDef):
        if node.name in self.structs:
            raise self.err(node.line, f"duplicate struct '{node.name}'")
        seen = set()
        for f in node.fields:
            if f.name in seen:
                raise self.err(node.line, f"duplicate field '{f.name}'")
            seen.add(f.name)
            self.check_type_valid(f.type, node.line)
        self.structs[node.name] = node

    def register_function(self, node: A.FnDef):
        if node.name in self.functions:
            raise self.err(node.line, f"duplicate function '{node.name}'")
        for p in node.params:
            self.check_type_valid(p.type, node.line)
        if node.ret is not None:
            self.check_type_valid(node.ret, node.line)
        self.functions[node.name] = (
            [(p.name, p.type, p.mutable) for p in node.params],
            node.ret,
        )

    def check_type_valid(self, t, line):
        """Is this type name usable? (Parsed types, not checked, in v0.1.)"""
        if isinstance(t, A.NamedType):
            if t.name in INT_TYPES or t.name in (
                "f32", "f64", "bool", "str", "bytes", "unit",
            ):
                return
            if t.name in self.structs:
                # generic args on structs are not supported yet
                if t.args:
                    raise self.err(line, "generic structs not supported yet")
                return
            # builtin generic containers may appear with any args
            if t.name in ("list", "array"):
                return
            raise self.err(line, f"unknown type '{t.name}'")
        elif isinstance(t, A.RawType):
            self.check_type_valid(t.inner, line)
        elif isinstance(t, A.ListType):
            self.check_type_valid(t.elem, line)
        elif isinstance(t, A.ArrayType):
            self.check_type_valid(t.elem, line)
        elif isinstance(t, A.UnionType):
            for o in t.options:
                self.check_type_valid(o, line)
        elif isinstance(t, A.FnType):
            for p in t.params:
                self.check_type_valid(p, line)
            self.check_type_valid(t.ret, line)

    # -- statements --------------------------------------------------------

    def check_stmt(self, stmt):
        if isinstance(stmt, A.FnDef):
            self.check_fn_def(stmt)
        elif isinstance(stmt, A.StructDef):
            pass  # registered in pass 1
        elif isinstance(stmt, A.Binding):
            self.check_binding(stmt)
        elif isinstance(stmt, A.Assign):
            self.check_assign(stmt)
        elif isinstance(stmt, A.Share):
            t = self.check_expr(stmt.value)
            stmt.inferred = t
            self.declare_var(stmt.name, t, mut=False)
        elif isinstance(stmt, A.Take):
            # v0.001: single-file compilation; taken names are placeholders.
            for n in stmt.names:
                self.declare_var(n, A.NamedType("int"), mut=False)
        elif isinstance(stmt, A.If):
            ct = self.check_expr(stmt.cond)
            if not is_bool(ct):
                raise self.err(stmt.line,
                               f"'if' needs bool, got {type_name(ct)}")
            stmt.inferred = UNIT
            for s in stmt.then_body:
                self.check_stmt(s)
            for s in stmt.else_body:
                self.check_stmt(s)
        elif isinstance(stmt, A.While):
            ct = self.check_expr(stmt.cond)
            if not is_bool(ct):
                raise self.err(stmt.line,
                               f"'while' needs bool, got {type_name(ct)}")
            stmt.inferred = UNIT
            for s in stmt.body:
                self.check_stmt(s)
        elif isinstance(stmt, A.For):
            it = self.check_expr(stmt.iter)
            if isinstance(it, A.ListType):
                # v0.003: `for x in list` — x is a fresh immutable binding
                # per iteration (share-to-read); the length is snapshotted
                # so the loop always terminates.
                stmt.for_kind = "list"
                stmt.inferred = UNIT
                self.declare_var(stmt.var, it.elem, mut=False)
                self._check_for_list_no_mutation(stmt)
                for s in stmt.body:
                    self.check_stmt(s)
            elif is_int_like(it):
                stmt.for_kind = "int"
                stmt.inferred = UNIT
                self.declare_var(stmt.var, A.NamedType("int"), mut=False)
                for s in stmt.body:
                    self.check_stmt(s)
            else:
                raise self.err(
                    stmt.line,
                    f"'for x in <expr>' needs an integer or a list, "
                    f"got {type_name(it)}",
                )
        elif isinstance(stmt, A.Match):
            self.check_match(stmt)
        elif isinstance(stmt, A.Return):
            self.check_return(stmt)
        elif isinstance(stmt, A.Crash):
            mt = self.check_expr(stmt.message)
            if not is_str(mt):
                raise self.err(stmt.line,
                               f"'crash' needs a str, got {type_name(mt)}")
            stmt.inferred = UNIT
        elif isinstance(stmt, A.AsmBlock):
            self.check_asm_block(stmt)
        elif isinstance(stmt, A.ExprStmt):
            stmt.inferred = self.check_expr(stmt.expr)
        else:
            raise self.err(getattr(stmt, "line", "?"),
                           f"unsupported statement {type(stmt).__name__}")

    def _check_asm_reg(self, name: str, line, what: str) -> tuple:
        """Validate an asm register name. Returns (number, bits)."""
        if name in ASM_FORBIDDEN_REGS:
            raise self.err(
                line,
                f"asm {what}: cannot use '{name}' — the compiler owns "
                f"the stack frame (rsp/rbp)",
            )
        info = asm_reg_info(name)
        if info is None:
            raise self.err(
                line,
                f"asm {what}: unknown register '{name}' "
                f"(allowed: {', '.join(sorted(ASM_REG64))}, "
                f"{', '.join(sorted(ASM_REG32))})",
            )
        return info

    def check_asm_block(self, node: A.AsmBlock):
        """Typecheck an `asm ... end` block.

        - `in <var> = <reg>`: var must be a declared integer variable.
        - `out <var> = <reg>`: var must be a declared MUT integer variable
          (it's a write — the `mut` rule applies).
        - `clobber "<reg>"`: register must be known and not rsp/rbp.
          The declaration is TRUSTED, not verified (v0.005 has no
          instruction decoder — see docs/v0.005.md).
        - `raw` bytes were range-checked by the parser.
        Registers/vars are annotated onto the node for codegen.
        """
        line = getattr(node, "line", "?")
        for d in node.ins:
            found = self.lookup_var(d.var)
            if found is None:
                raise self.err(line,
                               f"asm in: undeclared variable '{d.var}'")
            t, _mut = found
            if not is_int_like(t):
                raise self.err(
                    line,
                    f"asm in: '{d.var}' must be an integer variable, "
                    f"got {type_name(t)}",
                )
            d.reg_num, d.reg_bits = self._check_asm_reg(d.reg, line, "in")
        for d in node.outs:
            found = self.lookup_var(d.var)
            if found is None:
                raise self.err(line,
                               f"asm out: undeclared variable '{d.var}'")
            t, mut = found
            if not is_int_like(t):
                raise self.err(
                    line,
                    f"asm out: '{d.var}' must be an integer variable, "
                    f"got {type_name(t)}",
                )
            if not mut:
                raise self.err(
                    line,
                    f"asm out: cannot write to immutable '{d.var}' "
                    f"(declare it with `mut`)",
                )
            d.reg_num, d.reg_bits = self._check_asm_reg(d.reg, line, "out")
        for d in node.clobbers:
            d.reg_num, d.reg_bits = self._check_asm_reg(d.reg, line,
                                                        "clobber")
        node.inferred = UNIT

    def _check_for_list_no_mutation(self, stmt):
        """Reject detectable mutation of the iterated list inside the loop.

        Orum's rule is share-to-read/own-to-change: the loop holds a
        read view, so `l.push(x)`, `l[i] = v`, or handing `l` to a `mut`
        parameter inside the body is a compile error. Only direct uses
        of the iterated variable are detectable; aliasing through
        another name is documented behavior (the length is snapshotted,
        so the loop still terminates).
        """
        if not isinstance(stmt.iter, A.Var):
            return  # only a plain variable can be tracked
        lname = stmt.iter.name
        line = getattr(stmt, "line", "?")

        def walk(s):
            # A bare `words.push(x)` statement is ExprStmt(Call(...)).
            node = s.expr if isinstance(s, A.ExprStmt) else s
            if isinstance(node, A.Call):
                f = node.func
                if (isinstance(f, A.FieldAccess)
                        and isinstance(f.obj, A.Var)
                        and f.obj.name == lname):
                    raise self.err(
                        line,
                        f"cannot mutate '{lname}' while iterating it "
                        f"with 'for'",
                    )
                if isinstance(f, A.Var) and f.name in self.functions:
                    params, _ = self.functions[f.name]
                    for (pname, ptype, pmut), arg in zip(params, node.args):
                        if (pmut and isinstance(arg, A.Var)
                                and arg.name == lname):
                            raise self.err(
                                line,
                                f"cannot pass '{lname}' to mut parameter "
                                f"'{pname}' while iterating it with 'for'",
                            )
            elif isinstance(node, A.Assign):
                t = node.target
                if (isinstance(t, A.Index) and isinstance(t.obj, A.Var)
                        and t.obj.name == lname):
                    raise self.err(
                        line,
                        f"cannot mutate '{lname}' while iterating it "
                        f"with 'for'",
                    )
            for c in self._child_stmts(s):
                walk(c)

        for s in stmt.body:
            walk(s)

    @staticmethod
    def _child_stmts(s):
        if isinstance(s, A.If):
            return s.then_body + s.else_body
        if isinstance(s, (A.While, A.For)):
            return s.body
        if isinstance(s, A.Match):
            return [a.body for a in s.arms]
        return []

    def check_fn_def(self, node: A.FnDef):
        if self.in_function:
            raise self.err(node.line,
                           "nested functions are not supported in v0.001")
        params, ret = self.functions[node.name]
        self.in_function = True
        self.push_scope()
        for pname, ptype, pmut in params:
            self.declare_var(pname, ptype, mut=pmut)
        self.current_fn_ret = ret
        for s in node.body:
            self.check_stmt(s)
        self.pop_scope()
        self.in_function = False
        self.current_fn_ret = None
        node.inferred = UNIT

    def check_binding(self, node: A.Binding):
        t = self.check_expr(node.value, expected=node.type)
        if node.type is not None and not compatible(node.type, t):
            raise self.err(
                node.line,
                f"cannot bind {type_name(t)} to '{node.name}: "
                f"{type_name(node.type)}'",
            )
        final = node.type if node.type is not None else t
        if node.type is not None:
            self.check_const_range(node.type, node.value, node.line)
        node.inferred = final
        self.declare_var(node.name, final, mut=node.mutable)

    def check_assign(self, node: A.Assign):
        v = self.check_expr(node.value)
        tgt = node.target
        if isinstance(tgt, A.Var):
            found = self.lookup_var(tgt.name)
            if found is None:
                raise self.err(node.line,
                               f"assign to undeclared '{tgt.name}'")
            ttype, mut = found
            if not mut:
                raise self.err(node.line,
                               f"cannot assign to immutable '{tgt.name}' "
                               "(declare it with `mut`)")
            tgt.inferred = ttype
            if not compatible(ttype, v):
                raise self.err(
                    node.line,
                    f"cannot assign {type_name(v)} to '{tgt.name}: "
                    f"{type_name(ttype)}'",
                )
        elif isinstance(tgt, A.FieldAccess):
            # root must be a declared mut struct variable
            root = tgt
            while isinstance(root, A.FieldAccess):
                root = root.obj
            if not isinstance(root, A.Var):
                raise self.err(node.line,
                               "field assignment needs a variable root")
            found = self.lookup_var(root.name)
            if found is None:
                raise self.err(node.line,
                               f"assign to undeclared '{root.name}'")
            _, mut = found
            if not mut:
                raise self.err(
                    node.line,
                    f"cannot mutate field of immutable '{root.name}'",
                )
            ftype = self.check_expr(tgt)  # sets inferred along the way
            if not compatible(ftype, v):
                raise self.err(
                    node.line,
                    f"cannot assign {type_name(v)} to field of type "
                    f"{type_name(ftype)}",
                )
        elif isinstance(tgt, A.Index):
            # raw-pointer stores are exempt from `mut` (kernel/hardware);
            # list stores are not: the list variable must be `mut`.
            obj_t = self.check_expr(tgt.obj)
            if isinstance(obj_t, A.ListType):
                if not isinstance(tgt.obj, A.Var):
                    raise self.err(
                        node.line,
                        "list index assignment needs a plain variable "
                        "in v0.002",
                    )
                found = self.lookup_var(tgt.obj.name)
                if found is None or not found[1]:
                    raise self.err(
                        node.line,
                        f"cannot assign into immutable "
                        f"'{tgt.obj.name}' (declare it with `mut`)",
                    )
            it = self.check_expr(tgt)
            if not compatible(it, v):
                raise self.err(
                    node.line,
                    f"cannot store {type_name(v)} into {type_name(it)}",
                )
        node.inferred = UNIT

    def check_return(self, node: A.Return):
        if not self.in_function:
            raise self.err(node.line, "'return' outside a function")
        ret = self.current_fn_ret
        if node.value is None:
            node.inferred = UNIT
            return
        t = self.check_expr(node.value, expected=ret)
        if ret is not None and not compatible(ret, t):
            raise self.err(
                node.line,
                f"cannot return {type_name(t)} from function "
                f"returning {type_name(ret)}",
            )
        if ret is not None:
            self.check_const_range(ret, node.value, node.line)
        node.inferred = UNIT

    def check_match(self, node: A.Match):
        st = self.check_expr(node.subject)
        if not (
            isinstance(st, A.UnionType) and len(st.options) == 2
        ):
            raise self.err(
                node.line,
                f"'match' needs a 2-option union, got {type_name(st)}",
            )
        kinds = sorted(a.kind for a in node.arms)
        if kinds != ["err", "ok"]:
            raise self.err(
                node.line,
                "'match' needs exactly one 'ok' and one 'err' arm "
                "in v0.001",
            )
        for arm in node.arms:
            side = st.options[0] if arm.kind == "ok" else st.options[1]
            self.declare_var(arm.binding, side, mut=False)
            self.check_stmt(arm.body)
        node.inferred = UNIT

    # -- expressions (returns the type; sets .inferred) ----------------------

    def check_expr(self, node, expected=None):
        if isinstance(node, A.IntLit):
            return self.check_int_lit(node, expected)
        if isinstance(node, A.StrLit):
            node.inferred = A.NamedType("str")
            return node.inferred
        if isinstance(node, A.InterpString):
            for p in node.parts:
                if isinstance(p, A.Var):
                    pt = self.check_expr(p)
                    if not (is_int_like(pt) or is_str(pt) or is_bool(pt)):
                        line = getattr(node, "line", "?")
                        raise self.err(
                            line,
                            "can only interpolate int/str/bool, got "
                            f"{type_name(pt)}",
                        )
                # StrLit parts are fine
            node.inferred = A.NamedType("str")
            return node.inferred
        if isinstance(node, A.Var):
            found = self.lookup_var(node.name)
            if found is None:
                line = getattr(node, "line", "?")
                raise self.err(line, f"undeclared name '{node.name}'")
            node.inferred = found[0]
            return node.inferred
        if isinstance(node, A.ArrayLit):
            # `[1, 2, 3]` — byte array literal. All elements must be
            # integer literals in 0..255. Decays to `raw *u8` (pointer
            # to rodata), like C array decay.
            line = getattr(node, "line", "?")
            if not node.elems:
                raise self.err(line, "empty array literal not allowed")
            for e in node.elems:
                if not isinstance(e, A.IntLit):
                    raise self.err(
                        line,
                        "array literal elements must be integer literals",
                    )
                if not 0 <= e.value <= 255:
                    raise self.err(
                        line,
                        f"array literal byte out of range: {e.value}",
                    )
            node.inferred = A.RawType(A.NamedType("u8"))
            return node.inferred
        if isinstance(node, A.BinOp):
            return self.check_binop(node)
        if isinstance(node, A.UnaryOp):
            ot = self.check_expr(node.operand)
            if not is_int_like(ot):
                line = getattr(node, "line", "?")
                raise self.err(line,
                               f"unary '{node.op}' needs an integer, "
                               f"got {type_name(ot)}")
            # '-' and '~' both preserve the operand's type.
            node.inferred = ot
            return ot
        if isinstance(node, A.Call):
            return self.check_call(node, expected)
        if isinstance(node, A.ListNew):
            self.check_type_valid(node.elem,
                                  getattr(node, "line", "?"))
            node.inferred = A.ListType(node.elem)
            return node.inferred
        if isinstance(node, A.StructLit):
            return self.check_struct_lit(node, expected)
        if isinstance(node, A.FieldAccess):
            return self.check_field(node)
        if isinstance(node, A.Index):
            return self.check_index(node)
        line = getattr(node, "line", "?")
        raise self.err(line, f"unsupported expression {type(node).__name__}")

    def check_int_lit(self, node: A.IntLit, expected):
        v = node.value
        if expected is not None and is_int_like(expected):
            lo, hi = INT_RANGES[expected.name]
            if not (lo <= v <= hi):
                line = getattr(node, "line", "?")
                raise self.err(
                    line,
                    f"literal {v} does not fit in {expected.name} "
                    f"(range {lo}..{hi})",
                )
            node.inferred = expected
            return expected
        if not (INT_RANGES["i64"][0] <= v <= INT_RANGES["u64"][1]):
            line = getattr(node, "line", "?")
            raise self.err(line, f"literal {v} does not fit in 64 bits")
        node.inferred = A.NamedType("int")
        return node.inferred

    def check_const_range(self, type_, node, line):
        """If `node` is a compile-time constant and `type_` is an integer
        type, enforce the range (`x: u8 = 0 - 129` is an error)."""
        if not is_int_like(type_):
            return
        v = const_value(node)
        if v is None:
            return
        lo, hi = INT_RANGES[type_.name]
        # v0.004: bitwise ops yield 64-bit patterns. const_value returns
        # the signed interpretation, but a negative signed value may still
        # be a valid unsigned pattern (e.g. `x: u64 = 1 << 63` is the
        # pattern 0x8000...0, a fine u64). Reinterpret as unsigned before
        # giving up — this only turns previous errors into acceptance,
        # never the reverse.
        if not (lo <= v <= hi) and v < 0 and \
                type_.name in UNSIGNED_TYPES:
            v = v & 0xFFFFFFFFFFFFFFFF
        if not (lo <= v <= hi):
            raise self.err(
                line,
                f"constant {v} does not fit in {type_.name} "
                f"(range {lo}..{hi})",
            )

    def check_binop(self, node: A.BinOp):
        line = getattr(node, "line", "?")
        lt = self.check_expr(node.left)
        rt = self.check_expr(node.right)
        op = node.op
        if op == "+":
            if is_str(lt) or is_str(rt):
                raise self.err(
                    line,
                    "`+` on strings is not allowed; interpolation "
                    '`"hi $name"` is the only way to build strings',
                )
            # v0.010: pointer arithmetic — `raw *T + int` or `int + raw *T`
            if isinstance(lt, A.RawType) and is_int_like(rt):
                node.inferred = lt
                return node.inferred
            if isinstance(rt, A.RawType) and is_int_like(lt):
                node.inferred = rt
                return node.inferred
        if op == "-":
            # v0.010: `raw *T - int` -> `raw *T`
            # (pointer-pointer subtraction not supported yet)
            if isinstance(lt, A.RawType) and is_int_like(rt):
                node.inferred = lt
                return node.inferred
        if op in ("+", "-", "*", "/", "%"):
            if not (is_int_like(lt) and is_int_like(rt)):
                raise self.err(
                    line,
                    f"'{op}' needs integers, got {type_name(lt)} "
                    f"and {type_name(rt)}",
                )
            if op == "/" or op == "%":
                if const_value(node.right) == 0:
                    raise self.err(line, "division by zero")
            # result: keep the left type when both are the same flavor,
            # else plain `int` (everything is a 64-bit word at runtime)
            node.inferred = lt if types_equal(lt, rt) else A.NamedType("int")
            node.signed = self._signed(lt, rt)
            return node.inferred
        if op in ("&", "|", "^", "<<", ">>"):
            # v0.004 bitwise operators. Integer operands only.
            # Result type: for `& | ^`, same rule as arithmetic (left type
            # when both sides match, else `int`). For shifts, the result
            # has the VALUE's type — the shift amount's type never leaks
            # into the result.
            if not (is_int_like(lt) and is_int_like(rt)):
                raise self.err(
                    line,
                    f"'{op}' needs integers, got {type_name(lt)} "
                    f"and {type_name(rt)}",
                )
            if op in ("<<", ">>"):
                node.inferred = lt
            else:
                node.inferred = lt if types_equal(lt, rt) \
                    else A.NamedType("int")
            # Shift semantics follow signedness (documented in codegen):
            # `>>` on signed is arithmetic (sar), on unsigned is logical
            # (shr). `<<` is shl either way.
            node.signed = self._signed(lt, rt)
            return node.inferred
        if op in ("==", "!=", "<", "<=", ">", ">="):
            if op in ("<", "<=", ">", ">="):
                if not (is_int_like(lt) and is_int_like(rt)):
                    raise self.err(
                        line,
                        f"'{op}' needs integers, got {type_name(lt)} "
                        f"and {type_name(rt)}",
                    )
                node.signed = self._signed(lt, rt)
            else:
                ok = (is_int_like(lt) and is_int_like(rt)) or (
                    is_bool(lt) and is_bool(rt)
                )
                if not ok:
                    raise self.err(
                        line,
                        f"'{op}' needs matching integers/bools, got "
                        f"{type_name(lt)} and {type_name(rt)}",
                    )
                node.signed = True
            node.inferred = A.NamedType("bool")
            return node.inferred
        raise self.err(line, f"unknown operator '{op}'")

    @staticmethod
    def _signed(lt, rt):
        """Signedness for codegen: unsigned if either side is unsigned."""
        if isinstance(lt, A.NamedType) and lt.name in UNSIGNED_TYPES:
            return False
        if isinstance(rt, A.NamedType) and rt.name in UNSIGNED_TYPES:
            return False
        return True

    def check_call(self, node: A.Call, expected):
        line = getattr(node, "line", "?")
        if isinstance(node.func, A.FieldAccess):
            return self.check_method_call(node, expected, line)
        if not isinstance(node.func, A.Var):
            raise self.err(line, "unsupported call target")
        name = node.func.name
        if name == "say":
            return self.check_say(node, line)
        if name == "exit":
            return self.check_exit(node, line)
        if name == "len":
            return self.check_len(node, line)
        if name == "rawptr":
            return self.check_rawptr(node, expected, line)
        if name == "addr":
            return self.check_addr(node, expected, line)
        if name in ("ok", "err"):
            return self.check_ok_err(node, name, expected, line)
        # v0.011: function pointer call — variable with FnType
        var_info = self.lookup_var(name)
        if var_info is not None:
            vtype, _ = var_info
            if isinstance(vtype, A.FnType):
                # Check args against FnType params
                if len(node.args) != len(vtype.params):
                    raise self.err(
                        line,
                        f"function pointer takes {len(vtype.params)} argument(s), "
                        f"got {len(node.args)}",
                    )
                for ptype, arg in zip(vtype.params, node.args):
                    at = self.check_expr(arg, expected=ptype)
                    # TODO: type compatibility check
                node.inferred = vtype.ret
                node.builtin = "fn_ptr_call"
                node.fn_ptr_name = name
                return node.inferred
        if name not in self.functions:
            raise self.err(line, f"call to undeclared function '{name}'")
        params, ret = self.functions[name]
        if len(node.args) != len(params):
            raise self.err(
                line,
                f"'{name}' takes {len(params)} argument(s), "
                f"got {len(node.args)}",
            )
        for (pname, ptype, pmut), arg in zip(params, node.args):
            at = self.check_expr(arg, expected=ptype)
            if not compatible(ptype, at):
                raise self.err(
                    line,
                    f"'{name}': argument '{pname}' needs "
                    f"{type_name(ptype)}, got {type_name(at)}",
                )
            if pmut and isinstance(ptype, A.ListType):
                # A `mut` list parameter aliases the caller's object, so
                # the caller must hand over something it may be mutated:
                # a `mut` variable. (Value-typed `mut` params are private
                # copies and need no such rule.)
                if not (isinstance(arg, A.Var)
                        and self._is_mut_var(arg.name)):
                    raise self.err(
                        line,
                        f"'{name}': mut parameter '{pname}' needs a `mut` "
                        f"list variable argument",
                    )
            self.check_const_range(ptype, arg, line)
        node.inferred = ret if ret is not None else UNIT
        return node.inferred

    def check_method_call(self, node: A.Call, expected, line):
        """`receiver.method(args)` — v0.002: only `list.push(x)`.

        Syntax choice: dot-call, `l.push(x)`. The receiver must be a
        `mut` variable (push mutates, and Orum's share-to-read/own-to-
        change rule applies)."""
        recv = node.func.obj
        method = node.func.field
        obj_t = self.check_expr(recv)
        if not isinstance(obj_t, A.ListType):
            raise self.err(
                line,
                f"method '{method}' is not supported on {type_name(obj_t)} "
                "(only lists have methods in v0.002)",
            )
        if not isinstance(recv, A.Var):
            raise self.err(
                line,
                f"list method '{method}' needs a plain variable receiver "
                "in v0.002",
            )
        found = self.lookup_var(recv.name)
        if found is None:  # cannot happen: check_expr would have failed
            raise self.err(line, f"undeclared name '{recv.name}'")
        _, mut = found
        if method == "push":
            if not mut:
                raise self.err(
                    line,
                    f"cannot call mutating 'push' on immutable "
                    f"'{recv.name}' (declare it with `mut`)",
                )
            if len(node.args) != 1:
                raise self.err(line, "'push' takes exactly 1 argument")
            at = self.check_expr(node.args[0])
            if not compatible(obj_t.elem, at):
                raise self.err(
                    line,
                    f"'push' needs {type_name(obj_t.elem)}, "
                    f"got {type_name(at)}",
                )
            node.inferred = UNIT
            node.builtin = "list_push"
            return UNIT
        raise self.err(
            line,
            f"unknown list method '{method}' (v0.002 knows: push)",
        )

    def check_len(self, node: A.Call, line):
        if len(node.args) != 1:
            raise self.err(line, "'len' takes exactly 1 argument")
        at = self.check_expr(node.args[0])
        if not isinstance(at, A.ListType):
            raise self.err(
                line, f"'len' needs a list, got {type_name(at)}"
            )
        node.inferred = A.NamedType("int")
        node.builtin = "len"
        return node.inferred

    def check_rawptr(self, node: A.Call, expected, line):
        """`rawptr(addr)` — v0.004: materialize a `raw *T` from an integer
        address. The element type comes from the expected type, so the
        call site must be annotated: `p: raw *u32 = rawptr(0xF000)`.

        v0.011: also accepts `fn` types for function pointers:
        `f: fn(u64) -> u64 = rawptr(0x1000)`.

        This is deliberately explicit and dangerous-looking: converting
        an integer to a pointer is exactly the kind of thing Orum wants
        visible. On hosted builds the address must be valid (real MMIO
        addresses arrive with freestanding mode, a later version)."""
        if len(node.args) != 1:
            raise self.err(line, "'rawptr' takes exactly 1 argument")
        at = self.check_expr(node.args[0])
        if not is_int_like(at):
            raise self.err(
                line, f"'rawptr' needs an integer address, "
                f"got {type_name(at)}"
            )
        if not isinstance(expected, (A.RawType, A.FnType)):
            raise self.err(
                line, "'rawptr' needs a `raw *T` or `fn` annotation at the "
                "call site, e.g. `p: raw *u32 = rawptr(0xF000)`"
            )
        node.inferred = expected
        node.builtin = "rawptr"
        return node.inferred

    def check_addr(self, node: A.Call, expected, line):
        """`addr(var)` — v0.009: address of a local variable as u64.
        For FFI OUT parameters (e.g. UEFI LocateProtocol's `&gop`).
        Returns the address as a plain u64; use `rawptr()` to get a
        typed `raw *T` if needed."""
        if len(node.args) != 1:
            raise self.err(line, "'addr' takes exactly 1 argument")
        arg = node.args[0]
        if not isinstance(arg, A.Var):
            raise self.err(
                line, f"'addr' needs a variable name, got {type(arg).__name__}"
            )
        found = self.lookup_var(arg.name)
        if found is None:
            raise self.err(line, f"'addr' of undeclared variable '{arg.name}'")
        node.inferred = A.NamedType("u64")
        node.builtin = "addr"
        node.addr_var = arg.name
        return node.inferred

    def check_say(self, node: A.Call, line):
        if self.freestanding:
            raise self.err(
                line, "'say' requires hosted mode — it needs syscalls")
        if len(node.args) != 1:
            raise self.err(line, "'say' takes exactly 1 argument")
        at = self.check_expr(node.args[0])
        if isinstance(at, A.ListType):
            el = at.elem
            if not (is_int_like(el) or is_str(el) or is_bool(el)):
                raise self.err(
                    line,
                    f"'say' on list[{type_name(el)}] is not supported "
                    "in v0.002 (element must be int/str/bool)",
                )
            node.inferred = UNIT
            node.builtin = "say"
            return UNIT
        if not (is_int_like(at) or is_str(at) or is_bool(at)):
            raise self.err(
                line,
                f"'say' needs int/str/bool, got {type_name(at)}",
            )
        node.inferred = UNIT
        node.builtin = "say"
        return UNIT

    def check_exit(self, node: A.Call, line):
        if self.freestanding:
            raise self.err(
                line, "'exit' requires hosted mode — it needs syscalls")
        if len(node.args) != 1:
            raise self.err(line, "'exit' takes exactly 1 argument")
        at = self.check_expr(node.args[0])
        if not is_int_like(at):
            raise self.err(line,
                           f"'exit' needs an integer, got {type_name(at)}")
        node.inferred = UNIT
        node.builtin = "exit"
        return UNIT

    def check_ok_err(self, node: A.Call, name, expected, line):
        if len(node.args) != 1:
            raise self.err(line, f"'{name}' takes exactly 1 argument")
        if not (
            isinstance(expected, A.UnionType) and len(expected.options) == 2
        ):
            raise self.err(
                line,
                f"'{name}(...)' needs an expected `T | E` union type "
                "(from a return or binding annotation)",
            )
        side = expected.options[0] if name == "ok" else expected.options[1]
        if not is_int_like(side):
            raise self.err(
                line,
                "v0.001: union payloads must be integer-like, "
                f"got {type_name(side)}",
            )
        pt = self.check_expr(node.args[0])
        if not is_int_like(pt):
            raise self.err(
                line,
                f"'{name}' payload must be integer-like, "
                f"got {type_name(pt)}",
            )
        # Mark the call so codegen emits a tagged-union construction
        # (same pattern as the say/exit builtins).
        node.inferred = expected
        node.builtin = name  # 'ok' or 'err'
        return expected

    def check_struct_lit(self, node: A.StructLit, expected):
        line = getattr(node, "line", "?")
        if node.name not in self.structs:
            raise self.err(line, f"unknown struct '{node.name}'")
        sdef = self.structs[node.name]
        if len(node.fields) != len(sdef.fields):
            raise self.err(
                line,
                f"'{node.name}' needs {len(sdef.fields)} field(s), "
                f"got {len(node.fields)}",
            )
        given = [n for n, _ in node.fields]
        want = [f.name for f in sdef.fields]
        if sorted(given) != sorted(want):
            raise self.err(
                line,
                f"'{node.name}' fields are {want}, got {given}",
            )
        ftypes = {f.name: f.type for f in sdef.fields}
        for fname, fexpr in node.fields:
            ft = self.check_expr(fexpr, expected=ftypes[fname])
            if not compatible(ftypes[fname], ft):
                raise self.err(
                    line,
                    f"field '{fname}' needs {type_name(ftypes[fname])}, "
                    f"got {type_name(ft)}",
                )
        node.inferred = A.NamedType(node.name)
        return node.inferred

    def check_field(self, node: A.FieldAccess):
        line = getattr(node, "line", "?")
        ot = self.check_expr(node.obj)
        if not (isinstance(ot, A.NamedType) and ot.name in self.structs):
            raise self.err(
                line,
                f"field access needs a struct, got {type_name(ot)}",
            )
        sdef = self.structs[ot.name]
        for f in sdef.fields:
            if f.name == node.field:
                node.inferred = f.type
                return f.type
        raise self.err(
            line, f"struct '{ot.name}' has no field '{node.field}'"
        )

    def check_index(self, node: A.Index):
        line = getattr(node, "line", "?")
        ot = self.check_expr(node.obj)
        if isinstance(ot, A.ListType):
            it = self.check_expr(node.index)
            if not is_int_like(it):
                raise self.err(
                    line,
                    f"list index needs an integer, got {type_name(it)}",
                )
            node.inferred = ot.elem
            return ot.elem
        if not isinstance(ot, A.RawType):
            raise self.err(
                line,
                f"indexing needs a raw pointer or list, "
                f"got {type_name(ot)}",
            )
        it = self.check_expr(node.index)
        if not is_int_like(it):
            raise self.err(
                line,
                f"index needs an integer, got {type_name(it)}",
            )
        # v0.004 VOLATILITY (decided, no opt-out): every `raw *T` access
        # is volatile. Raw pointers exist for hardware/MMIO — the compiler
        # must never elide, merge, or reorder these accesses. The flag is
        # stamped here so every current and future pass can see it; the
        # codegen emits exactly one load/store per volatile access.
        node.volatile = True
        node.inferred = ot.inner
        return ot.inner


def typecheck(prog: A.Program, freestanding: bool = False) -> A.Program:
    """Run the typechecker over `prog` (in place). Returns `prog`."""
    return TypeChecker(freestanding=freestanding).check_program(prog)
