"""Parser tests for Orum v0.001."""
import glob
import os
import unittest

from orum import ast_nodes as A
from orum.parser import ParseError, parse


def prog(src):
    return parse(src)


class TestBindings(unittest.TestCase):
    def test_plain_binding(self):
        p = prog("x = 42\n")
        self.assertEqual(
            p, A.Program([A.Binding("x", None, A.IntLit(42), False)])
        )

    def test_mut_binding(self):
        p = prog("mut x = 1\n")
        self.assertEqual(
            p, A.Program([A.Binding("x", None, A.IntLit(1), True)])
        )

    def test_annotated_binding(self):
        p = prog("x: u8 = 300\n")
        self.assertEqual(
            p,
            A.Program(
                [A.Binding("x", A.NamedType("u8"), A.IntLit(300), False)]
            ),
        )

    def test_mut_annotated_binding(self):
        p = prog("mut buf: bytes = b\n")
        self.assertEqual(
            p,
            A.Program(
                [A.Binding("buf", A.NamedType("bytes"), A.Var("b"), True)]
            ),
        )

    def test_reassignment_is_assign_not_binding(self):
        p = prog("mut x = 1\nx = 2\n")
        self.assertEqual(
            p,
            A.Program(
                [
                    A.Binding("x", None, A.IntLit(1), True),
                    A.Assign(A.Var("x"), A.IntLit(2)),
                ]
            ),
        )

    def test_first_use_declares(self):
        # `name = expr` on first use in a scope is a declaration.
        p = prog("x = 1\nx = x + 1\n")
        self.assertIsInstance(p.statements[0], A.Binding)
        self.assertIsInstance(p.statements[1], A.Assign)

    def test_field_assign(self):
        p = prog("p.x = 10\n")
        # `p` is undeclared so `p.x = 10`... `p` alone isn't a binding
        # (peek2 is DOT, not ASSIGN/COLON) -> expression -> assign path.
        self.assertEqual(
            p,
            A.Program(
                [A.Assign(A.FieldAccess(A.Var("p"), "x"), A.IntLit(10))]
            ),
        )

    def test_index_assign(self):
        p = prog("fb[i] = 0x1a\n")
        self.assertEqual(
            p,
            A.Program(
                [A.Assign(A.Index(A.Var("fb"), A.Var("i")), A.IntLit(26))]
            ),
        )

    def test_assign_to_literal_rejected(self):
        with self.assertRaises(ParseError):
            prog("1 = 2\n")


class TestFunctions(unittest.TestCase):
    def test_fn_def(self):
        p = prog("fn add(a: int, b: int) -> int\nreturn a + b\nend\n")
        self.assertEqual(
            p,
            A.Program(
                [
                    A.FnDef(
                        "add",
                        [A.Param("a", A.NamedType("int")),
                         A.Param("b", A.NamedType("int"))],
                        A.NamedType("int"),
                        [A.Return(A.BinOp("+", A.Var("a"), A.Var("b")))],
                    )
                ]
            ),
        )

    def test_fn_no_params_no_return(self):
        p = prog("fn greet()\nsay(hi)\nend\n")
        fn = p.statements[0]
        self.assertIsInstance(fn, A.FnDef)
        self.assertEqual(fn.params, [])
        self.assertIsNone(fn.ret)
        self.assertEqual(fn.body, [A.ExprStmt(A.Call(A.Var("say"), [A.Var("hi")]))])

    def test_bare_return(self):
        p = prog("fn f()\nreturn\nend\n")
        self.assertEqual(p.statements[0].body, [A.Return(None)])

    def test_fn_name_visible_for_recursion(self):
        p = prog("fn fib(n: int) -> int\nreturn fib(n - 1)\nend\n")
        body = p.statements[0].body[0]
        # inner `fib(n - 1)` must parse as a call, not a binding
        self.assertIsInstance(body.value, A.Call)

    def test_missing_end(self):
        with self.assertRaises(ParseError):
            prog("fn f()\nreturn 1\n")

    def test_anonymous_fn_value_rejected(self):
        with self.assertRaises(ParseError):
            prog("double = fn(n: int) -> int\nreturn n * 2\nend\nend\n")


class TestControlFlow(unittest.TestCase):
    def test_if_else(self):
        p = prog("if x < 2\nreturn x\nelse\nreturn 0\nend\n")
        self.assertEqual(
            p,
            A.Program(
                [
                    A.If(
                        A.BinOp("<", A.Var("x"), A.IntLit(2)),
                        [A.Return(A.Var("x"))],
                        [A.Return(A.IntLit(0))],
                    )
                ]
            ),
        )

    def test_if_no_else(self):
        p = prog("if done\nsay(hi)\nend\n")
        node = p.statements[0]
        self.assertIsInstance(node, A.If)
        self.assertEqual(node.else_body, [])

    def test_while(self):
        p = prog("while i < 10\ni = i + 1\nend\n")
        node = p.statements[0]
        self.assertIsInstance(node, A.While)
        self.assertEqual(node.cond, A.BinOp("<", A.Var("i"), A.IntLit(10)))
        # `i` undeclared at first `i = i + 1`? No: `i` never declared,
        # so `i = i + 1` is a Binding (first use declares).
        self.assertIsInstance(node.body[0], A.Binding)

    def test_while_reassign(self):
        p = prog("mut i = 0\nwhile i < 10\ni = i + 1\nend\n")
        node = p.statements[1]
        self.assertIsInstance(node.body[0], A.Assign)

    def test_for(self):
        # v0.001: `for x in <int>` iterates 0..n-1. Calls need parens.
        p = prog("for i in 3\nsay(i)\nend\n")
        node = p.statements[0]
        self.assertIsInstance(node, A.For)
        self.assertEqual(node.var, "i")
        self.assertEqual(node.iter, A.IntLit(3))
        self.assertEqual(
            node.body,
            [A.ExprStmt(A.Call(A.Var("say"), [A.Var("i")]))],
        )

    def test_nested_blocks(self):
        src = (
            "fn f()\n"
            "if a\n"
            "while b\n"
            "say c\n"
            "end\n"
            "end\n"
            "end\n"
        )
        p = prog(src)
        fn = p.statements[0]
        self.assertIsInstance(fn.body[0], A.If)
        self.assertIsInstance(fn.body[0].then_body[0], A.While)

    def test_stray_end_rejected(self):
        with self.assertRaises(ParseError):
            prog("end\n")

    def test_stray_else_rejected(self):
        with self.assertRaises(ParseError):
            prog("else\n")


class TestMatch(unittest.TestCase):
    def test_match_ok_err(self):
        src = (
            'match read_file("/boot/x")\n'
            "ok(data) -> load(data)\n"
            'err(e) -> crash "bad"\n'
            "end\n"
        )
        p = prog(src)
        m = p.statements[0]
        self.assertIsInstance(m, A.Match)
        self.assertEqual(len(m.arms), 2)
        ok, err = m.arms
        self.assertEqual(ok.kind, "ok")
        self.assertEqual(ok.binding, "data")
        self.assertEqual(
            ok.body, A.ExprStmt(A.Call(A.Var("load"), [A.Var("data")]))
        )
        self.assertEqual(err.kind, "err")
        self.assertEqual(err.binding, "e")
        self.assertEqual(err.body, A.Crash(A.StrLit("bad")))

    def test_match_bad_arm_rejected(self):
        with self.assertRaises(ParseError):
            prog("match x\nmaybe(v) -> say v\nend\n")

    def test_ok_usable_as_plain_name(self):
        # ok/err are contextual: fine as ordinary bindings elsewhere.
        p = prog("ok = 1\n")
        self.assertIsInstance(p.statements[0], A.Binding)


class TestStructs(unittest.TestCase):
    def test_struct_def(self):
        p = prog("struct Point\nx: int\ny: int\nend\n")
        self.assertEqual(
            p,
            A.Program(
                [
                    A.StructDef(
                        "Point",
                        [
                            A.StructField("x", A.NamedType("int")),
                            A.StructField("y", A.NamedType("int")),
                        ],
                    )
                ]
            ),
        )

    def test_struct_literal(self):
        p = prog("p = Point(x: 10, y: 20)\n")
        self.assertEqual(
            p.statements[0],
            A.Binding(
                "p", None,
                A.StructLit(
                    "Point",
                    [("x", A.IntLit(10)), ("y", A.IntLit(20))],
                ),
                False,
            ),
        )

    def test_empty_call_is_not_struct_literal(self):
        p = prog("p = Point()\n")
        self.assertIsInstance(p.statements[0].value, A.Call)


class TestModules(unittest.TestCase):
    def test_share(self):
        p = prog("share pi = 3\n")
        self.assertEqual(
            p, A.Program([A.Share("pi", A.IntLit(3))])
        )

    def test_take(self):
        p = prog("take pi, tau from math.circle\n")
        self.assertEqual(
            p,
            A.Program([A.Take(["pi", "tau"], ["math", "circle"])]),
        )

    def test_take_single(self):
        p = prog("take pi from math\n")
        self.assertEqual(p.statements[0], A.Take(["pi"], ["math"]))

    def test_take_missing_from_rejected(self):
        with self.assertRaises(ParseError):
            prog("take pi math.circle\n")


class TestMiscStatements(unittest.TestCase):
    def test_crash(self):
        p = prog('crash "boom $x"\n')
        self.assertIsInstance(p.statements[0], A.Crash)

    def test_expr_statement(self):
        p = prog("say(hi)\n")
        self.assertEqual(
            p.statements[0], A.ExprStmt(A.Call(A.Var("say"), [A.Var("hi")]))
        )


class TestExpressions(unittest.TestCase):
    def expr(self, src):
        return prog(f"x = {src}\n").statements[0].value

    def test_precedence_mul_over_add(self):
        self.assertEqual(
            self.expr("1 + 2 * 3"),
            A.BinOp("+", A.IntLit(1),
                    A.BinOp("*", A.IntLit(2), A.IntLit(3))),
        )

    def test_precedence_parens(self):
        self.assertEqual(
            self.expr("(1 + 2) * 3"),
            A.BinOp("*",
                    A.BinOp("+", A.IntLit(1), A.IntLit(2)),
                    A.IntLit(3)),
        )

    def test_precedence_unary(self):
        self.assertEqual(
            self.expr("-a * b"),
            A.BinOp("*", A.UnaryOp("-", A.Var("a")), A.Var("b")),
        )

    def test_comparison(self):
        self.assertEqual(
            self.expr("a <= b"),
            A.BinOp("<=", A.Var("a"), A.Var("b")),
        )

    def test_comparison_binds_looser_than_add(self):
        self.assertEqual(
            self.expr("a + b == c"),
            A.BinOp("==",
                    A.BinOp("+", A.Var("a"), A.Var("b")),
                    A.Var("c")),
        )

    def test_modulo(self):
        self.assertEqual(
            self.expr("n % 2"),
            A.BinOp("%", A.Var("n"), A.IntLit(2)),
        )

    def test_call(self):
        self.assertEqual(
            self.expr("f(1, x)"),
            A.Call(A.Var("f"), [A.IntLit(1), A.Var("x")]),
        )

    def test_method_call(self):
        self.assertEqual(
            self.expr("c.push(4)"),
            A.Call(A.FieldAccess(A.Var("c"), "push"), [A.IntLit(4)]),
        )

    def test_field_access(self):
        self.assertEqual(
            self.expr("p.x"),
            A.FieldAccess(A.Var("p"), "x"),
        )

    def test_index(self):
        self.assertEqual(
            self.expr("fb[i + 1]"),
            A.Index(A.Var("fb"),
                    A.BinOp("+", A.Var("i"), A.IntLit(1))),
        )

    def test_interpolation(self):
        self.assertEqual(
            self.expr('"hi $name"'),
            A.InterpString([A.StrLit("hi "), A.Var("name")]),
        )

    def test_chained_postfix(self):
        self.assertEqual(
            self.expr("a.b(1)[2]"),
            A.Index(
                A.Call(A.FieldAccess(A.Var("a"), "b"), [A.IntLit(1)]),
                A.IntLit(2),
            ),
        )


class TestTypes(unittest.TestCase):
    def decltype(self, src):
        return prog(f"x: {src} = y\n").statements[0].type

    def test_sized_ints(self):
        for name in ("i8", "i16", "i32", "i64",
                     "u8", "u16", "u32", "u64"):
            self.assertEqual(self.decltype(name), A.NamedType(name))

    def test_floats_bool(self):
        self.assertEqual(self.decltype("f32"), A.NamedType("f32"))
        self.assertEqual(self.decltype("f64"), A.NamedType("f64"))
        self.assertEqual(self.decltype("bool"), A.NamedType("bool"))

    def test_raw_pointer(self):
        self.assertEqual(
            self.decltype("raw *u32"),
            A.RawType(A.NamedType("u32")),
        )

    def test_nested_raw(self):
        self.assertEqual(
            self.decltype("raw *raw *u8"),
            A.RawType(A.RawType(A.NamedType("u8"))),
        )

    def test_list(self):
        self.assertEqual(
            self.decltype("list[int]"),
            A.ListType(A.NamedType("int")),
        )

    def test_array(self):
        self.assertEqual(
            self.decltype("array[u8, 16]"),
            A.ArrayType(A.NamedType("u8"), 16),
        )

    def test_union(self):
        self.assertEqual(
            self.decltype("string | error"),
            A.UnionType([A.NamedType("string"), A.NamedType("error")]),
        )

    def test_union_three(self):
        t = self.decltype("a | b | c")
        self.assertEqual(
            t,
            A.UnionType(
                [A.NamedType("a"), A.NamedType("b"), A.NamedType("c")]
            ),
        )

    def test_fn_type(self):
        self.assertEqual(
            self.decltype("fn(int, str) -> bool"),
            A.FnType(
                [A.NamedType("int"), A.NamedType("str")],
                A.NamedType("bool"),
            ),
        )

    def test_fn_type_no_params(self):
        self.assertEqual(
            self.decltype("fn() -> int"),
            A.FnType([], A.NamedType("int")),
        )

    def test_union_of_fn_types(self):
        t = self.decltype("fn(int) -> int | str")
        # `->` binds tighter than `|`: fn returning (int | str)
        self.assertEqual(
            t,
            A.FnType(
                [A.NamedType("int")],
                A.UnionType([A.NamedType("int"), A.NamedType("str")]),
            ),
        )


class TestIndentationIndependence(unittest.TestCase):
    def test_flat_vs_indented(self):
        flat = (
            "fn fib(n: int) -> int\n"
            "if n < 2\n"
            "return n\n"
            "else\n"
            "return fib(n - 1) + fib(n - 2)\n"
            "end\n"
            "end\n"
        )
        indented = (
            "    fn fib(n: int) -> int\n"
            "        if n < 2\n"
            "            return n\n"
            "        else\n"
            "            return fib(n - 1) + fib(n - 2)\n"
            "        end\n"
            "    end\n"
        )
        chaotic = (
            "fn fib(n: int) -> int\n"
            "              if n < 2\n"
            "return n\n"
            "\t\telse\n"
            "   return fib(n - 1) + fib(n - 2)\n"
            "end\n"
            "                    end\n"
        )
        self.assertEqual(prog(flat), prog(indented))
        self.assertEqual(prog(flat), prog(chaotic))


class TestExamples(unittest.TestCase):
    def test_all_examples_parse(self):
        here = os.path.dirname(os.path.abspath(__file__))
        files = sorted(glob.glob(os.path.join(here, "..", "examples", "*.orum")))
        self.assertGreaterEqual(len(files), 4, "expected 4+ example files")
        for path in files:
            with open(path) as f:
                src = f.read()
            try:
                prog(src)
            except Exception as e:
                self.fail(f"example {os.path.basename(path)} failed to parse: {e}")


if __name__ == "__main__":
    unittest.main()
