"""Tests for Orum v0.005: inline `asm ... end` blocks.

Syntax:
    asm
    in <var> = <reg>       # copy Orum int var into register (before)
    out <var> = <reg>      # copy register into `mut` Orum int var (after)
    clobber "<reg>"        # declared-trashed register (trusted, not verified)
    raw <byte>, <byte>...  # machine-code bytes emitted verbatim
    end

Emission order is ALWAYS: in-moves, raw bytes, out-moves.
Allowed registers: rax rbx rcx rdx rsi rdi r8-r15 (+ 32-bit aliases
eax ebx ecx edx esi edi r8d-r15d). rsp/rbp/esp/ebp are compile errors.
There is no assembler in the pipeline: `raw` bytes go out verbatim.
"""
import subprocess
import tempfile
import unittest

from orum import ast_nodes as A
from orum.driver import build_binary, compile_source
from orum.lexer import Lexer
from orum.parser import Parser, ParseError
from orum.typecheck import TypeCheckError, typecheck

from tests.test_compiler import CompilerTestBase


def parse(src):
    return Parser(Lexer(src).tokenize()).parse_program()


def toks(src):
    return [(t.type, t.value) for t in Lexer(src).tokenize()
            if t.type not in ("NEWLINE", "EOF")]


def check(src):
    """Parse + typecheck; returns the program."""
    prog = parse(src)
    typecheck(prog)
    return prog


class TestAsmLexer(unittest.TestCase):
    def test_keywords(self):
        ts = toks("asm\nout x = rax\nclobber \"rcx\"\nend\n")
        types = [t for t, _ in ts]
        self.assertIn("ASM", types)
        self.assertIn("OUT", types)
        self.assertIn("CLOBBER", types)

    def test_in_still_in(self):
        # `in` was already a keyword (for x in ...); still lexes as IN
        ts = toks("for i in x\nend\n")
        self.assertIn("IN", [t for t, _ in ts])

    def test_reg_names_are_idents(self):
        ts = toks("asm\nin x = r12\nend\n")
        vals = [v for _, v in ts]
        self.assertIn("r12", vals)


class TestAsmParser(unittest.TestCase):
    def test_full_block_shape(self):
        prog = parse(
            "fn f() -> int\n"
            "mut y: int = 0\n"
            "asm\n"
            "in y = rax\n"
            "out y = rbx\n"
            "clobber \"rcx\"\n"
            "raw 0x90, 0x31\n"
            "end\n"
            "return y\n"
            "end\n"
        )
        blk = prog.statements[0].body[1]
        self.assertIsInstance(blk, A.AsmBlock)
        self.assertEqual([(d.var, d.reg) for d in blk.ins], [("y", "rax")])
        self.assertEqual([(d.var, d.reg) for d in blk.outs], [("y", "rbx")])
        self.assertEqual([d.reg for d in blk.clobbers], ["rcx"])
        self.assertEqual(blk.raw_bytes, [0x90, 0x31])

    def test_directives_any_order(self):
        prog = parse(
            "fn f() -> int\n"
            "mut y: int = 0\n"
            "asm\n"
            "raw 0x90\n"
            "out y = rax\n"
            "in y = rbx\n"
            "end\n"
            "return y\n"
            "end\n"
        )
        blk = prog.statements[0].body[1]
        self.assertEqual(len(blk.ins), 1)
        self.assertEqual(len(blk.outs), 1)
        self.assertEqual(blk.raw_bytes, [0x90])

    def test_empty_block(self):
        prog = parse("fn f() -> int\nasm\nend\nreturn 0\nend\n")
        blk = prog.statements[0].body[0]
        self.assertIsInstance(blk, A.AsmBlock)
        self.assertEqual(blk.raw_bytes, [])

    def test_raw_byte_range(self):
        with self.assertRaises(ParseError):
            parse("fn f() -> int\nasm\nraw 256\nend\nreturn 0\nend\n")

    def test_raw_needs_int(self):
        with self.assertRaises(ParseError):
            parse('fn f() -> int\nasm\nraw "nope"\nend\nreturn 0\nend\n')

    def test_bad_directive(self):
        with self.assertRaises(ParseError):
            parse("fn f() -> int\nasm\nfrobnicate\nend\nreturn 0\nend\n")

    def test_missing_end(self):
        with self.assertRaises(ParseError):
            parse("fn f() -> int\nasm\nraw 0x90\n")

    def test_multiple_raw_lines(self):
        prog = parse(
            "fn f() -> int\nasm\nraw 0x0F\nraw 0x31\nend\nreturn 0\nend\n"
        )
        blk = prog.statements[0].body[0]
        self.assertEqual(blk.raw_bytes, [0x0F, 0x31])


class TestAsmTypecheck(unittest.TestCase):
    def test_valid_block(self):
        check(
            "fn f(x: int) -> int\n"
            "mut y: int = 0\n"
            "asm\n"
            "in x = rcx\n"
            "out y = rdx\n"
            "clobber \"r11\"\n"
            "raw 0x90\n"
            "end\n"
            "return y\n"
            "end\n"
        )

    def test_all_64bit_regs_ok(self):
        regs = " ".join(
            ["rax", "rbx", "rcx", "rdx", "rsi", "rdi"] +
            [f"r{i}" for i in range(8, 16)])
        lines = "\n".join(f'clobber "{r}"' for r in regs.split())
        check(f"fn f() -> int\nasm\n{lines}\nend\nreturn 0\nend\n")

    def test_all_32bit_regs_ok(self):
        regs = ["eax", "ebx", "ecx", "edx", "esi", "edi"] + \
            [f"r{i}d" for i in range(8, 16)]
        lines = "\n".join(f'clobber "{r}"' for r in regs)
        check(f"fn f() -> int\nasm\n{lines}\nend\nreturn 0\nend\n")

    def test_unknown_register(self):
        with self.assertRaises(TypeCheckError):
            check("fn f() -> int\nmut y: int = 0\n"
                  "asm\nout y = r16\nend\nreturn y\nend\n")

    def test_unknown_register_in(self):
        with self.assertRaises(TypeCheckError):
            check("fn f(x: int) -> int\n"
                  "asm\nin x = rbx2\nend\nreturn x\nend\n")

    def test_rsp_forbidden(self):
        with self.assertRaises(TypeCheckError):
            check("fn f() -> int\nmut y: int = 0\n"
                  "asm\nout y = rsp\nend\nreturn y\nend\n")

    def test_rbp_forbidden(self):
        with self.assertRaises(TypeCheckError):
            check("fn f(x: int) -> int\n"
                  "asm\nin x = rbp\nend\nreturn x\nend\n")

    def test_esp_ebp_forbidden(self):
        with self.assertRaises(TypeCheckError):
            check("fn f() -> int\n"
                  'asm\nclobber "esp"\nend\nreturn 0\nend\n')
        with self.assertRaises(TypeCheckError):
            check("fn f() -> int\n"
                  'asm\nclobber "ebp"\nend\nreturn 0\nend\n')

    def test_out_needs_mut(self):
        with self.assertRaises(TypeCheckError):
            check("fn f() -> int\ny: int = 0\n"
                  "asm\nout y = rax\nend\nreturn y\nend\n")

    def test_out_undeclared(self):
        with self.assertRaises(TypeCheckError):
            check("fn f() -> int\n"
                  "asm\nout nope = rax\nend\nreturn 0\nend\n")

    def test_in_undeclared(self):
        with self.assertRaises(TypeCheckError):
            check("fn f() -> int\n"
                  "asm\nin nope = rax\nend\nreturn 0\nend\n")

    def test_in_needs_int(self):
        with self.assertRaises(TypeCheckError):
            check('fn f(s: str) -> int\n'
                  "asm\nin s = rax\nend\nreturn 0\nend\n")

    def test_out_needs_int(self):
        with self.assertRaises(TypeCheckError):
            check('fn f() -> int\nmut s: str = "x"\n'
                  "asm\nout s = rax\nend\nreturn 0\nend\n")

    def test_clobber_unknown_reg(self):
        with self.assertRaises(TypeCheckError):
            check("fn f() -> int\n"
                  'asm\nclobber "rax2"\nend\nreturn 0\nend\n')

    def test_in_param_ok(self):
        # params are readable bindings — usable in `in`
        check("fn f(x: int) -> int\n"
              "asm\nin x = r8\nraw 0x90\nend\nreturn x\nend\n")


class TestAsmCodegen(unittest.TestCase):
    def text_of(self, src):
        from orum.codegen import Codegen
        prog = check(src)
        text, _, _, _, _, _, _ = Codegen(prog).compile()
        return bytes(text)

    def test_emission_order(self):
        # in-move, then raw bytes verbatim, then out-move
        text = self.text_of(
            "fn f(x: int) -> int\n"
            "mut y: int = 0\n"
            "asm\n"
            "in x = rcx\n"      # 48 8b 4d f8  mov rcx,[rbp-8]
            "out y = rdx\n"     # 48 89 55 f0  mov [rbp-16],rdx
            "raw 0x90, 0x90\n"
            "end\n"
            "return y\n"
            "end\n"
            "say(f(1))\n"
        )
        i = text.find(b"\x90\x90")
        self.assertNotEqual(i, -1, "raw bytes missing")
        # in-move immediately before: 48 8b 4d f8
        self.assertEqual(text[i - 4:i], b"\x48\x8b\x4d\xf8")
        # out-move immediately after: 48 89 55 f0
        self.assertEqual(text[i + 2:i + 6], b"\x48\x89\x55\xf0")

    def test_out32_zero_extends(self):
        # out y = eax -> 8b c0 (mov eax,eax) then 48 89 .. (mov [rbp],rax)
        text = self.text_of(
            "fn f() -> int\n"
            "mut y: int = 0\n"
            "asm\n"
            "out y = eax\n"
            "raw 0x90\n"
            "end\n"
            "return y\n"
            "end\n"
            "say(f())\n"
        )
        i = text.find(b"\x90")
        self.assertEqual(text[i + 1:i + 3], b"\x8b\xc0")
        self.assertEqual(text[i + 3:i + 4], b"\x48")  # REX.W store follows

    def test_in32_load(self):
        # in x = edx -> 8b 55 f8 (mov edx,[rbp-8]), no REX.W
        text = self.text_of(
            "fn f(x: int) -> int\n"
            "asm\n"
            "in x = edx\n"
            "raw 0x90\n"
            "end\n"
            "return x\n"
            "end\n"
            "say(f(7))\n"
        )
        i = text.find(b"\x90")
        self.assertEqual(text[i - 3:i], b"\x8b\x55\xf8")

    def test_r8_uses_rex(self):
        # high registers need REX.R/B: in x = r12 -> 4c 8b 65 f8
        text = self.text_of(
            "fn f(x: int) -> int\n"
            "asm\n"
            "in x = r12\n"
            "raw 0x90\n"
            "end\n"
            "return x\n"
            "end\n"
            "say(f(7))\n"
        )
        i = text.find(b"\x90")
        self.assertEqual(text[i - 4:i], b"\x4c\x8b\x65\xf8")


class TestAsmEndToEnd(CompilerTestBase):
    RDTSC = (
        "fn rdtsc() -> u64\n"
        "mut lo: u64 = 0\n"
        "mut hi: u64 = 0\n"
        "asm\n"
        "out lo = eax\n"
        "out hi = edx\n"
        "raw 0x0F, 0x31\n"
        "end\n"
        "return (hi << 32) | lo\n"
        "end\n"
    )

    def test_rdtsc_nonzero_and_increasing(self):
        src = self.RDTSC + (
            "mut a = rdtsc()\n"
            "mut b = rdtsc()\n"
            "say(a)\n"
            "say(b)\n"
            "if a > 0\n"
            "if b > a\n"
            "say(1)\n"
            "end\n"
            "end\n"
        )
        p = self.run_orum(src)
        self.assertEqual(p.returncode, 0, f"stderr: {p.stderr!r}")
        lines = p.stdout.decode().strip().split("\n")
        self.assertEqual(lines[-1], "1")  # b > a > 0

    def test_in_round_trip(self):
        src = (
            "fn ident(x: int) -> int\n"
            "mut y: int = 0\n"
            "asm\n"
            "in x = r10\n"
            "out y = r10\n"
            "raw 0x90\n"
            "end\n"
            "return y\n"
            "end\n"
            "say(ident(12345))\n"
        )
        self.assert_stdout(src, "12345\n")

    def test_in32_round_trip(self):
        src = (
            "fn ident32(x: u32) -> u32\n"
            "mut y: u32 = 0\n"
            "asm\n"
            "in x = ecx\n"
            "out y = ecx\n"
            "raw 0x90\n"
            "end\n"
            "return y\n"
            "end\n"
            "say(ident32(4000000000))\n"
        )
        self.assert_stdout(src, "4000000000\n")

    def test_cpuid_multi_out(self):
        # cpuid is unprivileged: leaf 0 -> eax = max basic leaf (> 0)
        src = (
            "fn cpuid_max() -> u32\n"
            "mut leaf: u32 = 0\n"
            "mut a: u32 = 0\n"
            "mut b: u32 = 0\n"
            "mut c: u32 = 0\n"
            "mut d: u32 = 0\n"
            "asm\n"
            "in leaf = eax\n"
            "out a = eax\n"
            "out b = ebx\n"
            "out c = ecx\n"
            "out d = edx\n"
            "raw 0x0F, 0xA2\n"
            "end\n"
            "return a\n"
            "end\n"
            "mut m = cpuid_max()\n"
            "if m > 0\n"
            "say(1)\n"
            "else\n"
            "say(0)\n"
            "end\n"
        )
        self.assert_stdout(src, "1\n")

    def test_clobber_declared(self):
        # raw bytes trash r11 (declared); program must stay correct
        src = (
            "fn f(x: int) -> int\n"
            "mut y: int = 0\n"
            "asm\n"
            "in x = rax\n"
            "out y = rax\n"
            'clobber "r11"\n'
            "raw 0x4D, 0x31, 0xDB\n"  # xor r11, r11
            "end\n"
            "return y\n"
            "end\n"
            "say(f(99))\n"
        )
        self.assert_stdout(src, "99\n")

    def test_asm_in_loop(self):
        src = self.RDTSC + (
            "mut i = 0\n"
            "mut last: u64 = 0\n"
            "mut ok = 1\n"
            "while i < 5\n"
            "mut t = rdtsc()\n"
            "if t < last\n"
            "ok = 0\n"
            "end\n"
            "last = t\n"
            "i = i + 1\n"
            "end\n"
            "say(ok)\n"
        )
        self.assert_stdout(src, "1\n")


if __name__ == "__main__":
    unittest.main()
