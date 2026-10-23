"""Lexer (tokenizer) for Orum v0.001.

Produces a flat token stream with line/column info. Whitespace is never
significant (indentation is purely cosmetic in Orum); newlines terminate
statements, except inside parentheses/brackets where they are suppressed.
`#` starts a comment. `{` and `}` are hard errors — Orum has no braces.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any


KEYWORDS = {
    "fn", "return", "mut", "if", "else", "while", "for", "in",
    "match", "struct", "share", "take", "from", "crash", "end", "raw",
    "asm", "out", "clobber",
}
# NOTE: `ok` and `err` are deliberately NOT keywords — they are only special
# inside `match` arms and are parsed contextually, so they stay usable as
# ordinary names everywhere else.

MULTI_CHAR = {
    "==": "EQ",
    "!=": "NEQ",
    "<=": "LTE",
    ">=": "GTE",
    "->": "ARROW",
    "<<": "SHL",
    ">>": "SHR",
}

SINGLE_CHAR = {
    "+": "PLUS",
    "-": "MINUS",
    "*": "STAR",
    "/": "SLASH",
    "%": "PERCENT",
    "<": "LT",
    ">": "GT",
    "=": "ASSIGN",
    ":": "COLON",
    ",": "COMMA",
    ".": "DOT",
    "(": "LPAREN",
    ")": "RPAREN",
    "[": "LBRACKET",
    "]": "RBRACKET",
    "|": "PIPE",  # union in type position, bitwise-or in expression position
    "&": "BITAND",
    "^": "BITXOR",
    "~": "BITNOT",
}

ESCAPES = {
    "n": "\n",
    "t": "\t",
    '"': '"',
    "\\": "\\",
    "$": "$",
}


@dataclass
class Token:
    type: str  # e.g. 'IDENT', 'INT', 'FN', 'PLUS', 'NEWLINE', 'EOF'
    value: Any
    line: int
    col: int

    def __repr__(self) -> str:
        return f"Token({self.type}, {self.value!r}, {self.line}:{self.col})"


class LexError(Exception):
    """Raised for any tokenization failure, always with line:col."""


class Lexer:
    def __init__(self, source: str):
        self.src = source
        self.pos = 0
        self.line = 1
        self.col = 1
        self.depth = 0  # ( / [ nesting; newlines inside are not tokens
        self.tokens: list[Token] = []

    # -- character helpers -------------------------------------------------

    def at_end(self) -> bool:
        return self.pos >= len(self.src)

    def peek(self) -> str:
        if self.at_end():
            return "\0"
        return self.src[self.pos]

    def advance(self) -> str:
        c = self.src[self.pos]
        self.pos += 1
        if c == "\n":
            self.line += 1
            self.col = 1
        else:
            self.col += 1
        return c

    def emit(self, type_: str, value: Any, line: int, col: int) -> None:
        self.tokens.append(Token(type_, value, line, col))

    # -- main loop ----------------------------------------------------------

    def tokenize(self) -> list[Token]:
        while not self.at_end():
            self.scan()
        self.emit("EOF", None, self.line, self.col)
        return self.tokens

    def scan(self) -> None:
        line, col = self.line, self.col
        c = self.peek()

        if c in " \t\r":
            self.advance()
            return

        if c == "\n":
            self.advance()
            if self.depth == 0:
                self.emit("NEWLINE", "\n", line, col)
            return

        if c == "#":
            while not self.at_end() and self.peek() != "\n":
                self.advance()
            return

        if c == '"':
            self.scan_string(line, col)
            return

        if c.isdigit():
            self.scan_number(line, col)
            return

        if c.isalpha() or c == "_":
            self.scan_ident(line, col)
            return

        two = self.src[self.pos : self.pos + 2]
        if two in MULTI_CHAR:
            self.advance()
            self.advance()
            self.emit(MULTI_CHAR[two], two, line, col)
            return

        if c in SINGLE_CHAR:
            self.advance()
            if c in "([":
                self.depth += 1
            elif c in ")]":
                self.depth = max(0, self.depth - 1)
            self.emit(SINGLE_CHAR[c], c, line, col)
            return

        if c in "{}":
            raise LexError(
                f"{line}:{col}: braces are not used in Orum; "
                "blocks close with 'end'"
            )
        raise LexError(f"{line}:{col}: unexpected character {c!r}")

    # -- token kinds --------------------------------------------------------

    def scan_ident(self, line: int, col: int) -> None:
        chars = []
        while not self.at_end() and (self.peek().isalnum() or self.peek() == "_"):
            chars.append(self.advance())
        word = "".join(chars)
        if word in KEYWORDS:
            self.emit(word.upper(), word, line, col)
        else:
            self.emit("IDENT", word, line, col)

    def scan_number(self, line: int, col: int) -> None:
        if self.src.startswith(("0x", "0X"), self.pos):
            self.pos += 2
            self.col += 2
            digits = []
            while not self.at_end() and self.peek() in "0123456789abcdefABCDEF":
                digits.append(self.advance())
            if not digits:
                raise LexError(f"{line}:{col}: bad hex literal")
            self.emit("INT", int("".join(digits), 16), line, col)
            return
        digits = []
        while not self.at_end() and self.peek().isdigit():
            digits.append(self.advance())
        self.emit("INT", int("".join(digits)), line, col)

    def scan_string(self, line: int, col: int) -> None:
        self.advance()  # opening quote
        # parts: ('str', text) or ('var', name)
        parts: list[tuple[str, str]] = []
        buf: list[str] = []
        interpolated = False

        def flush() -> None:
            if buf:
                parts.append(("str", "".join(buf)))
                buf.clear()

        while not self.at_end():
            c = self.peek()
            if c == '"':
                self.advance()
                break
            if c == "\\":
                self.advance()
                e = self.peek()
                if e not in ESCAPES:
                    raise LexError(
                        f"{self.line}:{self.col}: unknown escape '\\{e}'"
                    )
                buf.append(ESCAPES[e])
                self.advance()
                continue
            if c == "\n":
                raise LexError(f"{line}:{col}: unterminated string literal")
            if c == "$":
                nxt = self.src[self.pos + 1 : self.pos + 2]
                if nxt.isalpha() or nxt == "_":
                    flush()
                    self.advance()  # $
                    name = []
                    while not self.at_end() and (
                        self.peek().isalnum() or self.peek() == "_"
                    ):
                        name.append(self.advance())
                    parts.append(("var", "".join(name)))
                    interpolated = True
                    continue
                # lone `$` (e.g. "$5", trailing "$") is literal
                buf.append(self.advance())
                continue
            buf.append(self.advance())
        else:
            raise LexError(f"{line}:{col}: unterminated string literal")

        flush()
        if interpolated:
            self.emit("INTERP_STRING", parts, line, col)
        else:
            text = "".join(text for kind, text in parts if kind == "str")
            self.emit("STRING", text, line, col)
