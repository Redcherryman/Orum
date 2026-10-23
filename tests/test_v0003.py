"""Tests for Orum v0.003: `mut` function parameters and `for x in list`.

Covers parsing, typechecking, and end-to-end compilation to real ELF
binaries (compiled, run, stdout asserted) — the same proof standard as
v0.001/v0.002.
"""
import unittest

from orum import ast_nodes as A
from orum.driver import compile_source
from orum.lexer import Lexer
from orum.parser import Parser, ParseError
from orum.typecheck import TypeCheckError

from tests.test_compiler import CompilerTestBase


def parse(src):
    return Parser(Lexer(src).tokenize()).parse_program()


class TestMutParamParsing(unittest.TestCase):
    def test_mut_param_ast(self):
        prog = parse("fn f(mut l: list[int], v: int)\nend\n")
        fn = prog.statements[0]
        self.assertIsInstance(fn, A.FnDef)
        self.assertEqual(len(fn.params), 2)
        self.assertTrue(fn.params[0].mutable)
        self.assertFalse(fn.params[1].mutable)
        self.assertEqual(fn.params[0].name, "l")

    def test_plain_param_defaults_immutable(self):
        prog = parse("fn f(x: int)\nend\n")
        self.assertFalse(prog.statements[0].params[0].mutable)

    def test_mut_param_ast_equality(self):
        # dataclass equality still works with the new field
        prog = parse("fn f(mut l: list[int])\nend\n")
        self.assertEqual(
            prog.statements[0].params[0],
            A.Param("l", A.ListType(A.NamedType("int")), True),
        )


class TestMutParamTypecheck(unittest.TestCase):
    def test_non_mut_list_to_mut_param_rejected(self):
        with self.assertRaises(TypeCheckError):
            compile_source(
                "fn fill(mut l: list[int], v: int)\nl.push(v)\nend\n"
                "l = list[int]()\nfill(l, 1)\n"
            )

    def test_mut_list_to_mut_param_ok(self):
        compile_source(
            "fn fill(mut l: list[int], v: int)\nl.push(v)\nend\n"
            "mut l = list[int]()\nfill(l, 1)\n"
        )

    def test_non_mut_list_to_plain_param_ok(self):
        compile_source(
            "fn n(l: list[int]) -> int\nreturn len(l)\nend\n"
            "l = list[int]()\nsay(n(l))\n"
        )

    def test_mut_list_to_plain_param_ok(self):
        # share-to-read: a mut list can be read through a plain param
        compile_source(
            "fn n(l: list[int]) -> int\nreturn len(l)\nend\n"
            "mut l = list[int]()\nsay(n(l))\n"
        )

    def test_push_on_mut_param_ok(self):
        compile_source(
            "fn fill(mut l: list[int], v: int)\nl.push(v)\nend\n"
        )

    def test_push_on_plain_param_rejected(self):
        with self.assertRaises(TypeCheckError):
            compile_source(
                "fn fill(l: list[int], v: int)\nl.push(v)\nend\n"
            )

    def test_index_assign_on_mut_param_ok(self):
        compile_source(
            "fn set0(mut l: list[int], v: int)\nl[0] = v\nend\n"
        )

    def test_reassign_non_mut_param_rejected(self):
        with self.assertRaises(TypeCheckError):
            compile_source("fn f(x: int)\nx = 5\nend\n")

    def test_reassign_mut_param_ok(self):
        compile_source("fn f(mut x: int)\nx = 5\nend\n")

    def test_mut_int_param_needs_no_mut_arg(self):
        # value-typed mut params are private copies: no caller rule
        compile_source(
            "fn inc(mut x: int) -> int\nx = x + 1\nreturn x\nend\n"
            "say(inc(41))\n"
        )

    def test_temporary_to_mut_param_rejected(self):
        with self.assertRaises(TypeCheckError):
            compile_source(
                "fn mk() -> list[int]\nreturn list[int]()\nend\n"
                "fn fill(mut l: list[int], v: int)\nl.push(v)\nend\n"
                "fill(mk(), 1)\n"
            )


class TestForInListTypecheck(unittest.TestCase):
    def test_for_list_bad_iter_rejected(self):
        with self.assertRaises(TypeCheckError):
            compile_source('for x in "hi"\nsay(x)\nend\n')

    def test_loop_var_is_immutable(self):
        with self.assertRaises(TypeCheckError):
            compile_source(
                "mut xs = list[int]()\nxs.push(1)\nfor x in xs\nx = 99\nend\n"
            )

    def test_push_during_iteration_rejected(self):
        with self.assertRaises(TypeCheckError):
            compile_source(
                'mut w = list[str]()\nw.push("a")\nfor s in w\nw.push("b")\nend\n'
            )

    def test_index_assign_during_iteration_rejected(self):
        with self.assertRaises(TypeCheckError):
            compile_source(
                "mut xs = list[int]()\nxs.push(1)\nfor x in xs\nxs[0] = 9\nend\n"
            )

    def test_mut_param_call_during_iteration_rejected(self):
        with self.assertRaises(TypeCheckError):
            compile_source(
                "fn fill(mut l: list[int], v: int)\nl.push(v)\nend\n"
                "mut xs = list[int]()\nxs.push(1)\nfor x in xs\nfill(xs, 2)\nend\n"
            )

    def test_push_nested_in_if_during_iteration_rejected(self):
        with self.assertRaises(TypeCheckError):
            compile_source(
                "mut xs = list[int]()\nxs.push(1)\nfor x in xs\n"
                "if x > 0\nxs.push(2)\nend\nend\n"
            )

    def test_reading_other_list_during_iteration_ok(self):
        compile_source(
            "mut a = list[int]()\nmut b = list[int]()\n"
            "a.push(1)\nb.push(2)\n"
            "for x in a\nsay(len(b))\nend\n"
        )


class TestV0003EndToEnd(CompilerTestBase):
    def test_mut_param_push_visible_to_caller(self):
        self.assert_stdout(
            "fn fill(mut l: list[int], v: int)\n"
            "mut i = 0\nwhile i < 3\nl.push(v + i)\ni = i + 1\nend\nend\n"
            "mut nums = list[int]()\nfill(nums, 10)\nsay(nums)\n",
            "[10, 11, 12]\n",
        )

    def test_mut_param_index_assign_visible(self):
        self.assert_stdout(
            "fn set0(mut l: list[int], v: int)\nl[0] = v\nend\n"
            "mut xs = list[int]()\nxs.push(1)\nxs.push(2)\nset0(xs, 99)\n"
            "say(xs)\n",
            "[99, 2]\n",
        )

    def test_for_in_int_list(self):
        self.assert_stdout(
            "mut xs = list[int]()\nmut i = 0\nwhile i < 5\nxs.push(i * i)\n"
            "i = i + 1\nend\n"
            "mut t = 0\nfor x in xs\nt = t + x\nend\nsay(t)\n",
            "30\n",
        )

    def test_for_in_str_list_with_interp(self):
        self.assert_stdout(
            'mut w = list[str]()\nw.push("alpha")\nw.push("beta")\n'
            'for s in w\nsay("got=$s")\nend\n',
            "got=alpha\ngot=beta\n",
        )

    def test_for_in_bool_list(self):
        self.assert_stdout(
            "mut f = list[bool]()\nf.push(1 == 1)\nf.push(1 == 2)\n"
            'for b in f\nsay("b=$b")\nend\n',
            "b=true\nb=false\n",
        )

    def test_nested_for_in_list(self):
        self.assert_stdout(
            "mut g = list[int]()\nmut i = 0\nwhile i < 4\ng.push(i)\n"
            "i = i + 1\nend\n"
            "mut n = 0\nfor a in g\nfor b in g\nn = n + 1\nend\nend\n"
            "say(n)\n",
            "16\n",
        )

    def test_for_in_empty_list(self):
        self.assert_stdout(
            "mut e = list[int]()\nfor x in e\nsay(x)\nend\nsay(42)\n",
            "42\n",
        )

    def test_for_in_list_param(self):
        self.assert_stdout(
            "fn total(l: list[int]) -> int\nmut t = 0\nfor x in l\n"
            "t = t + x\nend\nreturn t\nend\n"
            "mut xs = list[int]()\nxs.push(3)\nxs.push(4)\n"
            "say(total(xs))\n",
            "7\n",
        )

    def test_int_range_for_still_works(self):
        self.assert_stdout(
            "mut s = 0\nfor k in 5\ns = s + k\nend\nsay(s)\n",
            "10\n",
        )

    def test_mut_int_param(self):
        self.assert_stdout(
            "fn inc(mut x: int) -> int\nx = x + 1\nreturn x\nend\n"
            "say(inc(41))\n",
            "42\n",
        )


if __name__ == "__main__":
    unittest.main()
