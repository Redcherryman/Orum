"""Tests for Orum v0.004: bitwise operators and volatile raw pointers.

Bitwise: `& | ^ << >>` (binary, C-like precedence) and `~` (unary).
Volatile: EVERY `raw *T` load/store is volatile — never elided, merged,
or reordered. The volatility tests prove this two ways: by counting
emitter calls during codegen, and by disassembling the final binary.
"""
import os
import re
import subprocess
import tempfile
import unittest

from orum import ast_nodes as A
from orum.driver import build_binary, compile_source
from orum.lexer import Lexer
from orum.parser import Parser, ParseError
from orum.typecheck import TypeCheckError

from tests.test_compiler import CompilerTestBase


def parse(src):
    return Parser(Lexer(src).tokenize()).parse_program()


def toks(src):
    return [(t.type, t.value) for t in Lexer(src).tokenize()
            if t.type not in ("NEWLINE", "EOF")]


def disassemble(src):
    """Compile src and return objdump disassembly of the code bytes."""
    from orum.codegen import Codegen
    from orum.typecheck import typecheck
    prog = Parser(Lexer(src).tokenize()).parse_program()
    typecheck(prog)
    text, _, _, _, _, _, _ = Codegen(prog).compile()
    with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as f:
        f.write(bytes(text))
        path = f.name
    try:
        out = subprocess.run(
            ["objdump", "-b", "binary", "-m", "i386:x86-64", "-D", path],
            capture_output=True, text=True, timeout=30).stdout
    finally:
        os.unlink(path)
    return out


class TestBitwiseLexer(unittest.TestCase):
    def test_bitand(self):
        self.assertEqual(toks("a & b"),
                         [("IDENT", "a"), ("BITAND", "&"), ("IDENT", "b")])

    def test_bitxor(self):
        self.assertEqual(toks("a ^ b"),
                         [("IDENT", "a"), ("BITXOR", "^"), ("IDENT", "b")])

    def test_bitnot(self):
        self.assertEqual(toks("~a"), [("BITNOT", "~"), ("IDENT", "a")])

    def test_shl_shr(self):
        self.assertEqual(toks("a << b"), [("IDENT", "a"), ("SHL", "<<"),
                                          ("IDENT", "b")])
        self.assertEqual(toks("a >> b"), [("IDENT", "a"), ("SHR", ">>"),
                                          ("IDENT", "b")])

    def test_pipe_still_pipe(self):
        # `|` lexes as PIPE; the parser decides union-vs-bitor by context.
        self.assertEqual(toks("a | b"),
                         [("IDENT", "a"), ("PIPE", "|"), ("IDENT", "b")])

    def test_shr_vs_gt(self):
        # `>>` is one token, not two `>`; `>=` still lexes as GTE.
        self.assertEqual([t for t, _ in toks("a >> b")],
                         ["IDENT", "SHR", "IDENT"])
        self.assertEqual([t for t, _ in toks("a >= b")],
                         ["IDENT", "GTE", "IDENT"])
        self.assertEqual([t for t, _ in toks("a > b")],
                         ["IDENT", "GT", "IDENT"])


class TestBitwiseParser(unittest.TestCase):
    def test_shift_looser_than_additive(self):
        # C-like: `1 << 2 + 3` is `1 << (2 + 3)`.
        prog = parse("say(1 << 2 + 3)\n")
        call = prog.statements[0].expr
        arg = call.args[0]
        self.assertEqual(arg,
                         A.BinOp("<<", A.IntLit(1),
                                 A.BinOp("+", A.IntLit(2), A.IntLit(3))))

    def test_multiplicative_tighter_than_shift(self):
        # `2 * 3 << 2` is `(2 * 3) << 2`.
        prog = parse("say(2 * 3 << 2)\n")
        arg = prog.statements[0].expr.args[0]
        self.assertEqual(arg,
                         A.BinOp("<<", A.BinOp("*", A.IntLit(2), A.IntLit(3)),
                                 A.IntLit(2)))

    def test_bitwise_relative_order(self):
        # `|` loosest, then `^`, then `&`: `1 | 2 ^ 3 & 4`
        prog = parse("say(1 | 2 ^ 3 & 4)\n")
        arg = prog.statements[0].expr.args[0]
        self.assertEqual(arg,
                         A.BinOp("|", A.IntLit(1),
                                 A.BinOp("^", A.IntLit(2),
                                         A.BinOp("&", A.IntLit(3),
                                                 A.IntLit(4)))))

    def test_c_gotcha_preserved(self):
        # `&` binds looser than `==`: `a & mask == 0` is `a & (mask == 0)`.
        # Documented; parenthesize your masks.
        prog = parse("say(a & mask == 0)\n")
        arg = prog.statements[0].expr.args[0]
        self.assertEqual(arg.op, "&")
        self.assertEqual(arg.right.op, "==")

    def test_bitnot_unary(self):
        prog = parse("say(~x)\n")
        arg = prog.statements[0].expr.args[0]
        self.assertEqual(arg, A.UnaryOp("~", A.Var("x")))

    def test_double_bitnot(self):
        prog = parse("say(~~x)\n")
        arg = prog.statements[0].expr.args[0]
        self.assertEqual(arg, A.UnaryOp("~", A.UnaryOp("~", A.Var("x"))))

    def test_deref_desugar(self):
        # `*p` desugars to `p[0]`.
        prog = parse("say(*p)\n")
        arg = prog.statements[0].expr.args[0]
        self.assertEqual(arg, A.Index(A.Var("p"), A.IntLit(0)))

    def test_deref_in_multiply(self):
        # `a * *p` — binary `*` vs deref `*` disambiguate by position.
        prog = parse("say(a * *p)\n")
        arg = prog.statements[0].expr.args[0]
        self.assertEqual(arg.op, "*")
        self.assertEqual(arg.right, A.Index(A.Var("p"), A.IntLit(0)))

    def test_deref_assignable(self):
        # `*p = v` parses as an assignment (desugared Index is assignable).
        prog = parse("*p = 42\n")
        stmt = prog.statements[0]
        self.assertIsInstance(stmt, A.Assign)
        self.assertEqual(stmt.target,
                         A.Index(A.Var("p"), A.IntLit(0)))

    def test_union_still_parses(self):
        # `|` in type position is still union, not bitor.
        prog = parse("fn f(x: int | str) -> int\nreturn 1\nend\n")
        fn = prog.statements[0]
        self.assertIsInstance(fn.params[0].type, A.UnionType)


class TestBitwiseTypecheck(unittest.TestCase):
    def test_bitwise_needs_ints(self):
        with self.assertRaises(TypeCheckError):
            compile_source('say(1 & "a")\n')
        with self.assertRaises(TypeCheckError):
            compile_source('say(1 | "s")\n')
        with self.assertRaises(TypeCheckError):
            compile_source('say(1 << "a")\n')

    def test_bitnot_needs_int(self):
        with self.assertRaises(TypeCheckError):
            compile_source('say(~(1 == 1))\n')

    def test_shift_result_type(self):
        # shifts keep the value's type; `& | ^` follow arithmetic rules
        from orum.typecheck import typecheck
        prog = parse("x = 1 << 2\n")
        typecheck(prog)
        self.assertEqual(prog.statements[0].inferred, A.NamedType("int"))

    def test_const_range_bitwise(self):
        # `x: u8 = 0xF0 & 0x0F` is 0, fits; `0xFF | 0x100` is 511, doesn't
        compile_source("x: u8 = 0xF0 & 0x0F\n")
        with self.assertRaises(TypeCheckError):
            compile_source("x: u8 = 0xFF | 0x100\n")

    def test_u64_shift_const(self):
        # `x: u64 = 1 << 63` — the pattern fits u64 (unsigned reinterp.)
        compile_source("x: u64 = 1 << 63\n")

    def test_rawptr_needs_annotation(self):
        with self.assertRaises(TypeCheckError):
            compile_source("p = rawptr(0x1000)\n")

    def test_rawptr_needs_int_addr(self):
        with self.assertRaises(TypeCheckError):
            compile_source('p: raw *u32 = rawptr("nope")\n')

    def test_rawptr_ok(self):
        compile_source("p: raw *u32 = rawptr(0x1000)\n")

    def test_deref_non_raw_rejected(self):
        with self.assertRaises(TypeCheckError):
            compile_source("x = 5\nsay(*x)\n")


class TestBitwiseRuntime(CompilerTestBase):
    def test_and(self):
        self.assert_stdout("say(0xF0 & 0x3C)\n", "48\n")

    def test_or(self):
        self.assert_stdout("say(0xF0 | 0x3C)\n", "252\n")

    def test_xor(self):
        self.assert_stdout("say(0xFF ^ 0x3C)\n", "195\n")

    def test_shl(self):
        self.assert_stdout("say(1 << 10)\n", "1024\n")

    def test_shr(self):
        self.assert_stdout("say(1024 >> 3)\n", "128\n")

    def test_bitnot_zero(self):
        self.assert_stdout("say(~0)\n", "-1\n")

    def test_bitnot_ff(self):
        self.assert_stdout("say(~0xFF)\n", "-256\n")

    def test_shl_63(self):
        # 1 << 63 as signed 64-bit is the most negative int64
        self.assert_stdout("say(1 << 63)\n", "-9223372036854775808\n")

    def test_arithmetic_shift(self):
        # `>>` on signed is arithmetic (sar): sign propagates
        self.assert_stdout("say((0 - 8) >> 2)\n", "-2\n")

    def test_logical_shift(self):
        # `>>` on unsigned is logical (shr): zeros fill
        self.assert_stdout("x: u64 = 0x8000000000000000\nsay(x >> 1)\n",
                           "4611686018427387904\n")

    def test_variable_shift_count(self):
        # non-constant counts go through CL
        self.assert_stdout("mut n = 3\nsay(1 << n)\nsay(64 >> n)\n",
                           "8\n8\n")

    def test_precedence_mixed(self):
        self.assert_stdout("say(1 | 2 ^ 3 & 6)\n", "1\n")
        self.assert_stdout("say(1 << 2 + 3)\n", "32\n")

    def test_chained_ops(self):
        # 0xFF & 0xF0 = 0xF0; 0x0F ^ 0x33 = 0x3C; 0xF0 | 0x3C = 0xFC
        self.assert_stdout("say(0xFF & 0xF0 | 0x0F ^ 0x33)\n", "252\n")

    def test_mask_idiom(self):
        self.assert_stdout(
            "mut reg = 0xDEADBEEF\n"
            "reg = reg & ~0xFF\n"
            "say(reg)\n",
            "3735928320\n")


class TestRawPointers(CompilerTestBase):
    def test_volatile_flag_stamped(self):
        from orum.typecheck import typecheck
        prog = parse("p: raw *u32 = rawptr(0x1000)\nx = p[0]\n")
        typecheck(prog)
        idx = prog.statements[1].value
        self.assertIsInstance(idx, A.Index)
        self.assertTrue(idx.volatile)

    def test_list_index_not_volatile(self):
        from orum.typecheck import typecheck
        prog = parse("mut l = list[int]()\nx = l[0]\n")
        typecheck(prog)
        idx = prog.statements[1].value
        self.assertIsInstance(idx, A.Index)
        self.assertFalse(idx.volatile)

    def test_deref_volatile(self):
        # `*p` desugars to `p[0]` — still stamped volatile
        from orum.typecheck import typecheck
        prog = parse("p: raw *u32 = rawptr(0x1000)\nx = *p\n")
        typecheck(prog)
        idx = prog.statements[1].value
        self.assertTrue(idx.volatile)

    def test_raw_store_exempt_from_mut(self):
        # raw stores don't need `mut` (decided; hardware has no `mut`)
        compile_source("p: raw *u32 = rawptr(0x1000)\np[0] = 42\n")

    def test_rawptr_wrong_argc(self):
        with self.assertRaises(TypeCheckError):
            compile_source("p: raw *u32 = rawptr()\n")
        with self.assertRaises(TypeCheckError):
            compile_source("p: raw *u32 = rawptr(1, 2)\n")

    def test_index_size_variants(self):
        for t in ("u8", "u16", "u32", "u64", "i8", "i16", "i32", "i64"):
            compile_source(f"p: raw *{t} = rawptr(0x1000)\nx = p[2]\n"
                           f"p[2] = x\n")


class TestVolatility(unittest.TestCase):
    """Prove the volatility guarantee: every raw access emits exactly
    one load/store — never elided, merged, or reordered."""

    def _count_emitter_calls(self, src):
        """Compile src, counting load_sized/store_sized emitter calls."""
        from orum import x86
        from orum.codegen import Codegen
        from orum.typecheck import typecheck
        loads, stores = [], []
        orig_load = x86.Emitter.load_sized
        orig_store = x86.Emitter.store_sized

        def counting_load(self, dst, base, disp, size, signed):
            loads.append((dst, base, disp, size, signed))
            return orig_load(self, dst, base, disp, size, signed)

        def counting_store(self, base, disp, size, src):
            stores.append((base, disp, size, src))
            return orig_store(self, base, disp, size, src)

        x86.Emitter.load_sized = counting_load
        x86.Emitter.store_sized = counting_store
        try:
            prog = Parser(Lexer(src).tokenize()).parse_program()
            typecheck(prog)
            Codegen(prog).compile()
        finally:
            x86.Emitter.load_sized = orig_load
            x86.Emitter.store_sized = orig_store
        return loads, stores

    def test_lexical_reads_emit_n_loads(self):
        # 10 distinct source reads -> exactly 10 loads. No merging of
        # separate reads into one (codegen is static; loop hoisting is
        # covered by test_load_inside_loop_in_binary).
        src = ("p: raw *u32 = rawptr(0x10000)\n"
               "a = p[0]\nb = p[1]\nc = p[2]\nd = p[3]\ne = p[4]\n"
               "f = p[5]\ng = p[6]\nh = p[7]\ni = p[8]\nj = p[9]\n")
        loads, _ = self._count_emitter_calls(src)
        self.assertEqual(len(loads), 10,
                         f"expected 10 volatile loads, got {len(loads)}")

    def test_store_then_load_not_folded(self):
        # store followed by load of the same address: both emitted,
        # no store-to-load forwarding.
        src = ("p: raw *u32 = rawptr(0x10000)\n"
               "p[0] = 42\nx = p[0]\n")
        loads, stores = self._count_emitter_calls(src)
        self.assertEqual(len(stores), 1)
        self.assertEqual(len(loads), 1)

    def test_double_store_not_eliminated(self):
        # two stores to the same address: both emitted, no dead-store
        # elimination (hardware may care about every write).
        src = ("p: raw *u32 = rawptr(0x10000)\n"
               "p[0] = 1\np[0] = 2\n")
        _, stores = self._count_emitter_calls(src)
        self.assertEqual(len(stores), 2)

    def test_deref_store_load(self):
        # `*p` form gets the same treatment
        src = ("p: raw *u32 = rawptr(0x10000)\n"
               "*p = 7\nx = *p\n")
        loads, stores = self._count_emitter_calls(src)
        self.assertEqual(len(stores), 1)
        self.assertEqual(len(loads), 1)

    def test_load_inside_loop_in_binary(self):
        # Disassemble the final binary: the volatile load must sit
        # INSIDE the loop (between the back-edge target and the jump),
        # proving no hoisting happened anywhere in the pipeline.
        src = ("p: raw *u32 = rawptr(0x10000)\n"
               "mut i = 0\nmut x = 0\n"
               "while i < 10\nx = p[0]\ni = i + 1\nend\n")
        dis = disassemble(src)
        # parse into (addr, mnemonic, operands)
        insns = []
        for m in re.finditer(
                r"^\s*([0-9a-f]+):\s+(?:[0-9a-f]{2} )+\s+(\w+)\s*(.*)$",
                dis, re.M):
            insns.append((int(m.group(1), 16), m.group(2), m.group(3)))
        # find backward jumps (loop back-edges)
        back_edges = []
        for addr, mn, ops in insns:
            if mn == "jmp":
                tgt = re.search(r"0x([0-9a-f]+)", ops)
                if tgt and int(tgt.group(1), 16) < addr:
                    back_edges.append((int(tgt.group(1), 16), addr))
        self.assertTrue(back_edges, "expected a loop back-edge")
        # the u32 volatile load is `mov (%rax),%eax` — a 32-bit mov, which
        # the CPU zero-extends into the full 64-bit register. (There is
        # no `movzx r64, r/m32`: 0x0F 0xB7 is a 16-bit load.)
        loads = [(a, mn, ops) for a, mn, ops in insns
                 if mn == "mov" and "(%rax)" in ops and "%eax" in ops]
        self.assertTrue(loads, "expected a volatile u32 load in binary")
        # at least one load must be strictly inside a loop region
        inside = any(tgt < a < jmp
                     for a, _, _ in loads for tgt, jmp in back_edges)
        self.assertTrue(inside,
                        "volatile load was hoisted out of the loop")
