"""Tests for Orum v0.002: runtime string interpolation and list[T] codegen.

Covers parsing, typechecking, and end-to-end compilation to real ELF
binaries (compiled, run, stdout asserted) — the same proof standard as
v0.001.
"""
import unittest

from orum import ast_nodes as A
from orum.codegen import CodegenError
from orum.driver import compile_source
from orum.lexer import Lexer
from orum.parser import Parser, ParseError
from orum.typecheck import TypeCheckError

from tests.test_compiler import CompilerTestBase


def parse(src):
    return Parser(Lexer(src).tokenize()).parse_program()


class TestListNewParsing(unittest.TestCase):
    def test_list_new_ast(self):
        prog = parse("mut l = list[int]()\n")
        binding = prog.statements[0]
        self.assertIsInstance(binding, A.Binding)
        self.assertIsInstance(binding.value, A.ListNew)
        self.assertEqual(binding.value.elem, A.NamedType("int"))

    def test_list_new_str(self):
        prog = parse('mut l = list[str]()\n')
        self.assertIsInstance(prog.statements[0].value, A.ListNew)

    def test_method_call_ast(self):
        prog = parse("mut l = list[int]()\nl.push(1)\n")
        stmt = prog.statements[1]
        self.assertIsInstance(stmt, A.ExprStmt)
        call = stmt.expr
        self.assertIsInstance(call, A.Call)
        self.assertIsInstance(call.func, A.FieldAccess)
        self.assertEqual(call.func.field, "push")

    def test_bare_list_index_still_works(self):
        # `list` as an ordinary name followed by `[0]`: the ListNew
        # parse fails (0 is not a type) and we fall back to indexing.
        # It then fails typechecking as undeclared — but must PARSE.
        prog = parse("x = list[0]\n")
        binding = prog.statements[0]
        self.assertIsInstance(binding.value, A.Index)

    def test_list_new_needs_parens(self):
        # `list[int]` without `()` is not construction: the ListNew
        # parse backtracks and it parses as an index expression, which
        # the typechecker then rejects (no variable named `list`).
        with self.assertRaises(TypeCheckError):
            compile_source("mut l = list[int]\n")


class TestListTypecheck(unittest.TestCase):
    def test_push_immutable(self):
        with self.assertRaises(TypeCheckError):
            compile_source("l = list[int]()\nl.push(1)\n")

    def test_push_wrong_type(self):
        with self.assertRaises(TypeCheckError):
            compile_source('mut l = list[int]()\nl.push("x")\n')

    def test_push_arity(self):
        with self.assertRaises(TypeCheckError):
            compile_source("mut l = list[int]()\nl.push()\n")

    def test_unknown_method(self):
        with self.assertRaises(TypeCheckError):
            compile_source("mut l = list[int]()\nl.pop()\n")

    def test_method_on_non_list(self):
        with self.assertRaises(TypeCheckError):
            compile_source("x = 5\nx.push(1)\n")

    def test_index_assign_immutable(self):
        with self.assertRaises(TypeCheckError):
            compile_source("l = list[int]()\nl[0] = 1\n")

    def test_index_assign_wrong_type(self):
        with self.assertRaises(TypeCheckError):
            compile_source('mut l = list[int]()\nl[0] = "x"\n')

    def test_len_needs_list(self):
        with self.assertRaises(TypeCheckError):
            compile_source("x = 5\nsay(len(x))\n")

    def test_index_non_integer(self):
        with self.assertRaises(TypeCheckError):
            compile_source('mut l = list[int]()\nsay(l["x"])\n')

    def test_say_list_bad_elem(self):
        with self.assertRaises(TypeCheckError):
            compile_source(
                "struct P\nx: int\nend\n"
                "mut l = list[P]()\nsay(l)\n"
            )


class TestInterpTypecheck(unittest.TestCase):
    def test_interp_rejects_list(self):
        with self.assertRaises(TypeCheckError):
            compile_source('mut l = list[int]()\nsay("x=$l")\n')

    def test_interp_int_str_bool_ok(self):
        # must compile (not run) without error
        compile_source(
            'n = 1\ns = "a"\nb = 1 == 1\nsay("$n $s $b")\n'
        )


class TestInterpEndToEnd(CompilerTestBase):
    def test_basic(self):
        self.assert_stdout(
            'name = "myriad"\nage = 42\nsay("hello $name, age $age")\n',
            "hello myriad, age 42\n",
        )

    def test_bool(self):
        self.assert_stdout(
            "t = 1 == 1\nf = 1 == 2\nsay(\"$t/$f\")\n",
            "true/false\n",
        )

    def test_negative(self):
        self.assert_stdout(
            "n = 0 - 17\nsay(\"n=$n\")\n",
            "n=-17\n",
        )

    def test_zero(self):
        self.assert_stdout(
            "z = 0\nsay(\"z=$z\")\n",
            "z=0\n",
        )

    def test_adjacent_vars(self):
        self.assert_stdout(
            'a = "x"\nb = "y"\nsay("$a$b")\n',
            "xy\n",
        )

    def test_dollar_escape(self):
        # `\$` is a literal dollar; `$5` (digit) is literal too
        self.assert_stdout(
            'say("cost: \\$5")\n',
            "cost: $5\n",
        )

    def test_in_function(self):
        self.assert_stdout(
            "fn greet(who: str, n: int) -> str\n"
            'return "hi $who #$n"\n'
            "end\n"
            'say(greet("aiden", 7))\n',
            "hi aiden #7\n",
        )

    def test_interp_result_is_str(self):
        # interpolated strings work with crash() (needs str)
        p = self.run_orum('x = 99\ncrash("boom $x")\n')
        self.assertEqual(p.returncode, 1)
        self.assertIn(b"boom 99", p.stderr)


class TestListEndToEnd(CompilerTestBase):
    def test_push_len_index(self):
        self.assert_stdout(
            "mut l = list[int]()\n"
            "l.push(10)\nl.push(20)\nl.push(30)\n"
            "say(len(l))\nsay(l[0])\nsay(l[2])\n",
            "3\n10\n30\n",
        )

    def test_growth(self):
        # initial cap is 4; 10 pushes force two reallocations
        self.assert_stdout(
            "mut l = list[int]()\n"
            "mut i = 0\n"
            "while i < 10\nl.push(i * i)\ni = i + 1\nend\n"
            "say(len(l))\nsay(l[9])\nsay(l[4])\n",
            "10\n81\n16\n",
        )

    def test_index_assign(self):
        self.assert_stdout(
            "mut l = list[int]()\n"
            "l.push(1)\nl.push(2)\nl[0] = 99\n"
            "say(l[0])\nsay(l[1])\n",
            "99\n2\n",
        )

    def test_say_int_list(self):
        self.assert_stdout(
            "mut l = list[int]()\n"
            "l.push(1)\nl.push(0 - 2)\nl.push(3)\n"
            "say(l)\n",
            "[1, -2, 3]\n",
        )

    def test_say_str_list(self):
        self.assert_stdout(
            'mut l = list[str]()\n'
            'l.push("a")\nl.push("b")\n'
            "say(l)\n",
            '["a", "b"]\n',
        )

    def test_say_bool_list(self):
        self.assert_stdout(
            "mut l = list[bool]()\n"
            "l.push(1 == 1)\nl.push(1 == 2)\n"
            "say(l)\n",
            "[true, false]\n",
        )

    def test_say_empty_list(self):
        self.assert_stdout(
            "mut l = list[int]()\nsay(l)\n",
            "[]\n",
        )

    def test_handle_aliasing(self):
        # copying a list copies the handle: both see the same elements
        self.assert_stdout(
            "mut a = list[int]()\n"
            "a.push(1)\n"
            "mut b = a\n"
            "b.push(2)\n"
            "say(len(a))\nsay(a[1])\n",
            "2\n2\n",
        )

    def test_list_in_function(self):
        self.assert_stdout(
            "fn total(l: list[int]) -> int\n"
            "mut s = 0\nmut i = 0\n"
            "while i < len(l)\ns = s + l[i]\ni = i + 1\nend\n"
            "return s\nend\n"
            "mut l = list[int]()\n"
            "l.push(5)\nl.push(7)\n"
            "say(total(l))\n",
            "12\n",
        )

    def test_oob_dies(self):
        p = self.run_orum(
            "mut l = list[int]()\nl.push(1)\nsay(l[5])\n"
        )
        self.assertEqual(p.returncode, 1)
        self.assertIn(b"index out of bounds", p.stderr)

    def test_oob_negative_dies(self):
        p = self.run_orum(
            "mut l = list[int]()\nl.push(1)\n"
            "mut i = 0 - 1\nsay(l[i])\n"
        )
        self.assertEqual(p.returncode, 1)
        self.assertIn(b"index out of bounds", p.stderr)

    def test_oob_store_dies(self):
        p = self.run_orum(
            "mut l = list[int]()\nl[0] = 1\n"
        )
        self.assertEqual(p.returncode, 1)
        self.assertIn(b"index out of bounds", p.stderr)


if __name__ == "__main__":
    unittest.main()
