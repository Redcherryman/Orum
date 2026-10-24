"""End-to-end compiler tests for Orum v0.001.

Each test compiles an Orum program to a real ELF binary, runs it,
and asserts on exit codes / stdout. This is the proof that the full
pipeline (lex -> parse -> typecheck -> codegen -> ELF) works.
"""
import os
import subprocess
import tempfile
import unittest

from orum.driver import build_binary, compile_source
from orum.typecheck import TypeCheckError


class CompilerTestBase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp(prefix="orum_e2e_")
        cls.n = 0

    def run_orum(self, src, args=None):
        """Compile `src` and run it. Returns the CompletedProcess."""
        type(self).n += 1
        path = os.path.join(self.tmp, f"t{self.n}")
        build_binary(src, path)
        return subprocess.run([path] + (args or []), capture_output=True,
                              timeout=10)

    def assert_stdout(self, src, expected):
        p = self.run_orum(src)
        self.assertEqual(p.returncode, 0, f"stderr: {p.stderr!r}")
        self.assertEqual(p.stdout.decode(), expected)


class TestExitAndArithmetic(CompilerTestBase):
    def test_exit_code(self):
        p = self.run_orum("exit(42)\n")
        self.assertEqual(p.returncode, 42)

    def test_arithmetic_stdout(self):
        self.assert_stdout("say(2 + 3 * 4)\n", "14\n")

    def test_precedence(self):
        self.assert_stdout("say((2 + 3) * 4)\n", "20\n")

    def test_div_mod(self):
        self.assert_stdout("say(100 / 7)\nsay(100 % 7)\n", "14\n2\n")

    def test_negative(self):
        self.assert_stdout("say(0 - 42)\n", "-42\n")

    def test_unary(self):
        self.assert_stdout("x = 5\nsay(0 - x)\n", "-5\n")

    def test_locals(self):
        self.assert_stdout("x = 5\nsay(x * x + 1)\n", "26\n")

    def test_mut_reassign(self):
        self.assert_stdout("mut x = 1\nx = x + 41\nsay(x)\n", "42\n")

    def test_comparisons(self):
        self.assert_stdout(
            "say(1 < 2)\nsay(2 < 1)\nsay(3 == 3)\nsay(4 != 4)\n",
            "true\nfalse\ntrue\nfalse\n",
        )


class TestFunctions(CompilerTestBase):
    def test_call(self):
        self.assert_stdout(
            "fn add(a: int, b: int) -> int\nreturn a + b\nend\n"
            "say(add(20, 22))\n",
            "42\n",
        )

    def test_recursion_fib(self):
        self.assert_stdout(
            "fn fib(n: int) -> int\n"
            "if n < 2\nreturn n\n"
            "else\nreturn fib(n - 1) + fib(n - 2)\nend\nend\n"
            "say(fib(10))\n",
            "55\n",
        )

    def test_nested_calls(self):
        self.assert_stdout(
            "fn dbl(x: int) -> int\nreturn x * 2\nend\n"
            "say(dbl(dbl(dbl(1))))\n",
            "8\n",
        )

    def test_many_args(self):
        self.assert_stdout(
            "fn sum6(a: int, b: int, c: int, d: int, e: int, f: int) -> int\n"
            "return a + b + c + d + e + f\nend\n"
            "say(sum6(1, 2, 3, 4, 5, 6))\n",
            "21\n",
        )

    def test_exit_with_computed_code(self):
        p = self.run_orum(
            "fn fib(n: int) -> int\n"
            "if n < 2\nreturn n\n"
            "else\nreturn fib(n - 1) + fib(n - 2)\nend\nend\n"
            "exit(fib(10))\n"
        )
        self.assertEqual(p.returncode, 55)


class TestControlFlow(CompilerTestBase):
    def test_if_else(self):
        self.assert_stdout(
            "fn sign(x: int) -> int\n"
            "if x < 0\nreturn 0 - 1\nelse\nreturn 1\nend\nend\n"
            "say(sign(0 - 5))\nsay(sign(10))\n",
            "-1\n1\n",
        )

    def test_while_sum(self):
        self.assert_stdout(
            "mut i = 0\nmut s = 0\n"
            "while i < 10\ns = s + i\ni = i + 1\nend\n"
            "say(s)\n",
            "45\n",
        )

    def test_for(self):
        self.assert_stdout(
            "mut t = 0\nfor k in 5\nt = t + k\nend\nsay(t)\n",
            "10\n",
        )

    def test_nested_loops(self):
        self.assert_stdout(
            "mut t = 0\n"
            "for i in 3\nfor j in 4\nt = t + 1\nend\nend\n"
            "say(t)\n",
            "12\n",
        )


class TestStructsAndMatch(CompilerTestBase):
    def test_struct_field_access(self):
        self.assert_stdout(
            "struct Point\nx: int\ny: int\nend\n"
            "mut p = Point(x: 3, y: 4)\n"
            "say(p.x)\nsay(p.y)\n",
            "3\n4\n",
        )

    def test_struct_arg(self):
        self.assert_stdout(
            "struct Point\nx: int\ny: int\nend\n"
            "fn get_y(p: Point) -> int\nreturn p.y\nend\n"
            "say(get_y(Point(x: 1, y: 2)))\n",
            "2\n",
        )

    def test_struct_field_assign(self):
        self.assert_stdout(
            "struct Point\nx: int\ny: int\nend\n"
            "mut p = Point(x: 1, y: 1)\n"
            "p.x = 10\n"
            "say(p.x + p.y)\n",
            "11\n",
        )

    def test_match_ok(self):
        self.assert_stdout(
            "fn checked(v: int) -> int | int\n"
            "if v < 0\nreturn err(0 - 1)\nelse\nreturn ok(v * 2)\nend\nend\n"
            "match checked(21)\n"
            "ok(q) -> say(q)\n"
            "err(e) -> say(e)\n"
            "end\n",
            "42\n",
        )

    def test_match_err(self):
        self.assert_stdout(
            "fn checked(v: int) -> int | int\n"
            "if v < 0\nreturn err(0 - 1)\nelse\nreturn ok(v * 2)\nend\nend\n"
            "match checked(0 - 5)\n"
            "ok(q) -> say(q)\n"
            "err(e) -> say(e)\n"
            "end\n",
            "-1\n",
        )

    def test_struct_and_match_combined(self):
        self.assert_stdout(
            "struct Point\nx: int\ny: int\nend\n"
            "fn get_x(p: Point) -> int\nreturn p.x\nend\n"
            "fn safe_div(a: int, b: int) -> int | int\n"
            "if b == 0\nreturn err(0 - 1)\nelse\nreturn ok(a / b)\nend\nend\n"
            "mut total = 0\n"
            "mut pt = Point(x: 3, y: 4)\n"
            "total = total + get_x(pt)\n"
            "match safe_div(10, 2)\n"
            "ok(q) -> total = total + q\n"
            "err(e) -> total = total + e\n"
            "end\n"
            "match safe_div(10, 0)\n"
            "ok(q) -> total = total + q\n"
            "err(e) -> total = total + e\n"
            "end\n"
            "say(total)\n",
            "7\n",
        )


class TestStringsAndMisc(CompilerTestBase):
    def test_say_string(self):
        self.assert_stdout('say("hello")\n', "hello\n")

    def test_say_bool(self):
        self.assert_stdout("say(1 == 1)\nsay(1 == 2)\n", "true\nfalse\n")

    def test_str_variable(self):
        self.assert_stdout('s = "hi"\nsay(s)\n', "hi\n")

    def test_crash(self):
        p = self.run_orum('crash "kaboom"\n')
        self.assertEqual(p.returncode, 1)
        self.assertIn(b"kaboom", p.stderr)

    def test_unsigned_div(self):
        self.assert_stdout(
            "fn udiv(a: u64, b: u64) -> u64\nreturn a / b\nend\n"
            "say(udiv(100, 7))\n",
            "14\n",
        )

    def test_hex_literal(self):
        self.assert_stdout("say(0x1a)\n", "26\n")


class TestCompileErrors(CompilerTestBase):
    def test_type_error_surfaces(self):
        with self.assertRaises(TypeCheckError):
            compile_source("x: u8 = 300\n")

    def test_mut_error_surfaces(self):
        with self.assertRaises(TypeCheckError):
            compile_source("x = 1\nx = 2\n")

    def test_unsupported_codegen(self):
        from orum.codegen import CodegenError
        # list-typed struct fields parse and typecheck, but v0.001
        # codegen cannot lay them out.
        with self.assertRaises(CodegenError):
            compile_source("struct S\nxs: list[int]\nend\n")


if __name__ == "__main__":
    unittest.main()
