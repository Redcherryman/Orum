"""Lexer tests for Orum v0.001."""
import unittest

from orum.lexer import Lexer, LexError


def types(src):
    return [t.type for t in Lexer(src).tokenize()]


def toks(src):
    return [(t.type, t.value) for t in Lexer(src).tokenize()]


class TestLexerBasics(unittest.TestCase):
    def test_fn_header_tokens(self):
        ts = types("fn add(a: int, b: int) -> int\nreturn a + b\nend\n")
        self.assertEqual(
            ts,
            [
                "FN", "IDENT", "LPAREN",
                "IDENT", "COLON", "IDENT", "COMMA",
                "IDENT", "COLON", "IDENT",
                "RPAREN", "ARROW", "IDENT", "NEWLINE",
                "RETURN", "IDENT", "PLUS", "IDENT", "NEWLINE",
                "END", "NEWLINE",
                "EOF",
            ],
        )

    def test_type_names_lex_as_idents(self):
        # `int`, `str`, `raw`... `raw` is a keyword; the rest are idents
        # at lex time (the parser gives them meaning).
        ts = toks("int str u32 raw")
        self.assertEqual(
            [t for t, _ in ts],
            ["IDENT", "IDENT", "IDENT", "RAW", "EOF"],
        )

    def test_comments_skipped(self):
        ts = toks("# full line comment\nx = 1 # trailing\n")
        # the comment text is skipped, but the newline ending its line
        # is still a statement terminator
        self.assertEqual(
            ts,
            [
                ("NEWLINE", "\n"),
                ("IDENT", "x"), ("ASSIGN", "="), ("INT", 1),
                ("NEWLINE", "\n"), ("EOF", None),
            ],
        )

    def test_line_numbers(self):
        lexed = Lexer("x = 1\n\ny = 2\n").tokenize()
        lines = [t.line for t in lexed if t.type == "IDENT"]
        self.assertEqual(lines, [1, 3])

    def test_all_operators(self):
        ts = types("== != <= >= -> + - * / % < > = : , . ( ) [ ] |")
        self.assertEqual(
            ts,
            [
                "EQ", "NEQ", "LTE", "GTE", "ARROW",
                "PLUS", "MINUS", "STAR", "SLASH", "PERCENT",
                "LT", "GT", "ASSIGN", "COLON", "COMMA", "DOT",
                "LPAREN", "RPAREN", "LBRACKET", "RBRACKET", "PIPE",
                "EOF",
            ],
        )


class TestLexerLiterals(unittest.TestCase):
    def test_decimal_ints(self):
        ts = toks("0 42 1000")
        self.assertEqual([v for _, v in ts if _ != "EOF"], [0, 42, 1000])

    def test_hex_ints(self):
        ts = toks("0x1a 0XFF 0x0")
        vals = [v for t, v in ts if t == "INT"]
        self.assertEqual(vals, [26, 255, 0])

    def test_bad_hex(self):
        with self.assertRaises(LexError):
            Lexer("0x ").tokenize()

    def test_plain_string(self):
        ts = toks('"hello world"')
        self.assertEqual(ts[0], ("STRING", "hello world"))

    def test_string_escapes(self):
        ts = toks(r'"a\nb\tc\"d\\e"')
        self.assertEqual(ts[0], ("STRING", 'a\nb\tc"d\\e'))

    def test_interpolation(self):
        ts = toks('"hello $name, you are $age years old"')
        self.assertEqual(ts[0][0], "INTERP_STRING")
        self.assertEqual(
            ts[0][1],
            [
                ("str", "hello "),
                ("var", "name"),
                ("str", ", you are "),
                ("var", "age"),
                ("str", " years old"),
            ],
        )

    def test_lone_dollar_is_literal(self):
        # `$` not followed by a name character stays literal text,
        # and the string is NOT treated as interpolated.
        ts = toks('"$5 and a trailing $"')
        self.assertEqual(ts[0], ("STRING", "$5 and a trailing $"))

    def test_dollar_escape(self):
        ts = toks(r'"price: \$5"')
        self.assertEqual(ts[0], ("STRING", "price: $5"))

    def test_unterminated_string(self):
        with self.assertRaises(LexError):
            Lexer('"never ends').tokenize()

    def test_multiline_string_rejected(self):
        with self.assertRaises(LexError):
            Lexer('"line one\nline two"').tokenize()


class TestLexerStructure(unittest.TestCase):
    def test_braces_rejected(self):
        with self.assertRaises(LexError) as ctx:
            Lexer("{").tokenize()
        self.assertIn("end", str(ctx.exception))
        with self.assertRaises(LexError):
            Lexer("}").tokenize()

    def test_newlines_suppressed_inside_parens(self):
        ts = types("f(1,\n2)\n")
        self.assertEqual(
            ts,
            ["IDENT", "LPAREN", "INT", "COMMA", "INT", "RPAREN",
             "NEWLINE", "EOF"],
        )

    def test_newlines_suppressed_inside_brackets(self):
        ts = types("a[\n0\n]\n")
        self.assertEqual(
            ts,
            ["IDENT", "LBRACKET", "INT", "RBRACKET", "NEWLINE", "EOF"],
        )

    def test_unexpected_character(self):
        with self.assertRaises(LexError):
            Lexer("x = 1 @ 2").tokenize()


if __name__ == "__main__":
    unittest.main()
