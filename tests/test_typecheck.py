"""Typechecker tests for Orum v0.001."""
import unittest

from orum import ast_nodes as A
from orum.parser import parse
from orum.typecheck import TypeCheckError, typecheck


def check(src):
    return typecheck(parse(src))


class TestLiterals(unittest.TestCase):
    def test_u8_range_ok(self):
        check("x: u8 = 255\n")

    def test_u8_range_fail(self):
        with self.assertRaises(TypeCheckError) as ctx:
            check("x: u8 = 300\n")
        self.assertIn("300", str(ctx.exception))
        self.assertIn("u8", str(ctx.exception))

    def test_i8_range(self):
        check("x: i8 = 127\n")
        with self.assertRaises(TypeCheckError):
            check("x: i8 = 128\n")
        with self.assertRaises(TypeCheckError):
            check("x: i8 = 0 - 129\n")

    def test_untyped_literal_is_int(self):
        p = check("x = 42\n")
        self.assertEqual(p.statements[0].inferred, A.NamedType("int"))

    def test_annotated_literal_adapts(self):
        p = check("x: u16 = 42\n")
        self.assertEqual(p.statements[0].inferred, A.NamedType("u16"))


class TestMut(unittest.TestCase):
    def test_assign_to_immutable_fails(self):
        with self.assertRaises(TypeCheckError) as ctx:
            check("x = 1\nx = 2\n")
        self.assertIn("immutable", str(ctx.exception))

    def test_assign_to_mut_ok(self):
        check("mut x = 1\nx = 2\n")

    def test_field_of_immutable_fails(self):
        with self.assertRaises(TypeCheckError):
            check("struct P\nx: int\nend\np = P(x: 1)\np.x = 2\n")

    def test_field_of_mut_ok(self):
        check("struct P\nx: int\nend\nmut p = P(x: 1)\np.x = 2\n")

    def test_index_store_always_ok(self):
        # raw-pointer stores are exempt from `mut`
        check("fn f(fb: raw *u32)\nfb[0] = 1\nend\n")


class TestFunctions(unittest.TestCase):
    def test_arity_mismatch(self):
        with self.assertRaises(TypeCheckError):
            check("fn f(a: int) -> int\nreturn a\nend\nf(1, 2)\n")

    def test_arg_type_mismatch(self):
        with self.assertRaises(TypeCheckError):
            check('fn f(a: str)\nend\nf(1)\n')

    def test_return_type_mismatch(self):
        with self.assertRaises(TypeCheckError):
            check('fn f() -> int\nreturn "hi"\nend\n')

    def test_return_outside_fn(self):
        with self.assertRaises(TypeCheckError):
            check("return 1\n")

    def test_undeclared_call(self):
        with self.assertRaises(TypeCheckError):
            check("nosuchfn(1)\n")

    def test_recursion_ok(self):
        check("fn f(n: int) -> int\nreturn f(n)\nend\n")


class TestControlFlow(unittest.TestCase):
    def test_if_needs_bool(self):
        with self.assertRaises(TypeCheckError):
            check("if 1\nend\n")

    def test_while_needs_bool(self):
        with self.assertRaises(TypeCheckError):
            check("while 0\nend\n")

    def test_for_needs_int(self):
        with self.assertRaises(TypeCheckError):
            check('for i in "hi"\nend\n')

    def test_for_ok(self):
        check("for i in 10\nend\n")


class TestMatch(unittest.TestCase):
    def test_match_needs_union(self):
        with self.assertRaises(TypeCheckError):
            check("match 5\nok(x) -> say(x)\nerr(e) -> say(e)\nend\n")

    def test_match_ok(self):
        check(
            "fn f(v: int) -> int | int\n"
            "if v < 0\nreturn err(0 - 1)\nelse\nreturn ok(v)\nend\nend\n"
            "match f(1)\nok(x) -> say(x)\nerr(e) -> say(e)\nend\n"
        )

    def test_ok_without_union_context_fails(self):
        with self.assertRaises(TypeCheckError):
            check("fn f() -> int\nreturn ok(1)\nend\n")

    def test_ok_payload_must_be_int_like(self):
        with self.assertRaises(TypeCheckError):
            check('fn f() -> int | int\nreturn ok("hi")\nend\n')

    def test_match_needs_both_arms(self):
        with self.assertRaises(TypeCheckError):
            check(
                "fn f() -> int | int\nreturn ok(1)\nend\n"
                "match f()\nok(x) -> say(x)\nend\n"
            )


class TestOperators(unittest.TestCase):
    def test_str_plus_rejected(self):
        with self.assertRaises(TypeCheckError) as ctx:
            check('x = "a" + "b"\n')
        self.assertIn("interpolation", str(ctx.exception))

    def test_int_arith_ok(self):
        p = check("x = 1 + 2 * 3 - 4 / 2 % 3\n")
        self.assertEqual(p.statements[0].inferred, A.NamedType("int"))

    def test_comparison_is_bool(self):
        p = check("x = 1 < 2\n")
        self.assertEqual(p.statements[0].inferred, A.NamedType("bool"))

    def test_const_div_by_zero(self):
        with self.assertRaises(TypeCheckError):
            check("x = 1 / 0\n")

    def test_mixed_arith_rejected(self):
        with self.assertRaises(TypeCheckError):
            check('x = 1 + "a"\n')


class TestMisc(unittest.TestCase):
    def test_undeclared_var(self):
        with self.assertRaises(TypeCheckError) as ctx:
            check("say(y)\n")
        self.assertIn("undeclared", str(ctx.exception))

    def test_index_needs_raw(self):
        with self.assertRaises(TypeCheckError):
            check("x = 1\nsay(x[0])\n")

    def test_crash_needs_str(self):
        with self.assertRaises(TypeCheckError):
            check("crash 42\n")

    def test_say_types(self):
        check('say(1)\nsay("hi")\n')
        with self.assertRaises(TypeCheckError):
            check("struct P\nx: int\nend\nmut p = P(x: 1)\nsay(p)\n")

    def test_unknown_type(self):
        with self.assertRaises(TypeCheckError):
            check("x: nosuchtype = 1\n")

    def test_struct_field_mismatch(self):
        with self.assertRaises(TypeCheckError):
            check('struct P\nx: int\nend\np = P(x: "hi")\n')

    def test_full_program_ok(self):
        check(
            "struct Point\nx: int\ny: int\nend\n"
            "fn get_x(p: Point) -> int\nreturn p.x\nend\n"
            "fn safe_div(a: int, b: int) -> int | int\n"
            "if b == 0\nreturn err(0 - 1)\nelse\nreturn ok(a / b)\nend\nend\n"
            "mut pt = Point(x: 3, y: 4)\n"
            "say(get_x(pt))\n"
            "match safe_div(10, 2)\nok(q) -> say(q)\nerr(e) -> say(e)\nend\n"
        )


if __name__ == "__main__":
    unittest.main()
