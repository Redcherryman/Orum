"""Recursive-descent parser for Orum v0.001.

Grammar sketch (statements are newline-terminated; blocks close with `end`):

    program     := statement* EOF
    statement   := fn_def | struct_def | share | take
                 | if | while | for | match
                 | return | crash
                 | binding | assign | expr
    binding     := ['mut'] IDENT [':' type] '=' expr
    assign      := target '=' expr          # target: name | field | index
    fn_def      := 'fn' IDENT '(' [IDENT ':' type (, ...)] ')' ['->' type]
                   NEWLINE block 'end'
    struct_def  := 'struct' IDENT NEWLINE (IDENT ':' type NEWLINE)* 'end'
    share       := 'share' IDENT '=' expr
    take        := 'take' IDENT (, IDENT)* 'from' IDENT (. IDENT)*
    if          := 'if' expr NEWLINE block ['else' NEWLINE block] 'end'
    while       := 'while' expr NEWLINE block 'end'
    for         := 'for' IDENT 'in' expr NEWLINE block 'end'
    match       := 'match' expr NEWLINE arm* 'end'
    arm         := ('ok' | 'err') '(' IDENT ')' '->' statement NEWLINE
    return      := 'return' [expr]
    crash       := 'crash' expr

    expr        := comparison
    comparison  := additive (('==' | '!=' | '<' | '<=' | '>' | '>=') additive)*
    additive    := multiplicative (('+' | '-') multiplicative)*
    multiplicative := unary (('*' | '/' | '%') unary)*
    unary       := '-' unary | postfix
    postfix     := primary (call | index | field)*
    call        := '(' [expr (, expr)*] ')'
                  | '(' IDENT ':' expr (, IDENT ':' expr)* ')'   # struct lit
    index       := '[' expr ']'
    field       := '.' IDENT
    primary     := INT | STRING | INTERP_STRING | IDENT
                 | '(' expr ')'

    type        := primary_type ('|' primary_type)*
    primary_type:= 'raw' '*' primary_type
                 | 'fn' '(' [type (, type)*] ')' '->' type
                 | IDENT ['[' (type | INT) (, ...)* ']']

Binding vs assignment: the parser tracks declared names per function scope.
`name = expr` declares on first use in scope, and assigns afterwards.
`mut` and annotated forms always declare.
"""
from __future__ import annotations

from . import ast_nodes as A
from .lexer import Lexer, Token


class ParseError(Exception):
    """Raised for any syntax error, always with line:col."""


ASSIGNABLE = (A.Var, A.FieldAccess, A.Index)
COMP_OPS = {"EQ", "NEQ", "LT", "LTE", "GT", "GTE"}


class Parser:
    def __init__(self, tokens: list[Token]):
        self.toks = tokens
        self.pos = 0
        self.scopes: list[set] = [set()]

    # ------------------------------------------------------------------
    # token helpers
    # ------------------------------------------------------------------

    def peek(self) -> Token:
        return self.toks[self.pos]

    def peek2(self) -> Token:
        if self.pos + 1 < len(self.toks):
            return self.toks[self.pos + 1]
        return self.toks[-1]  # EOF

    def advance(self) -> Token:
        t = self.toks[self.pos]
        if t.type != "EOF":
            self.pos += 1
        return t

    def expect(self, type_: str) -> Token:
        t = self.peek()
        if t.type != type_:
            raise ParseError(
                f"{t.line}:{t.col}: expected '{type_.lower()}', "
                f"got '{t.type.lower()}' ({t.value!r})"
            )
        return self.advance()

    def skip_newlines(self) -> None:
        while self.peek().type == "NEWLINE":
            self.advance()

    def expect_newline(self, where: str) -> None:
        t = self.peek()
        if t.type != "NEWLINE":
            raise ParseError(
                f"{t.line}:{t.col}: expected newline {where}, "
                f"got '{t.type.lower()}' ({t.value!r})"
            )
        self.skip_newlines()

    # ------------------------------------------------------------------
    # scopes (Binding vs Assign)
    # ------------------------------------------------------------------

    def declare(self, name: str) -> None:
        self.scopes[-1].add(name)

    def is_declared(self, name: str) -> bool:
        return any(name in scope for scope in self.scopes)

    def push_scope(self) -> None:
        self.scopes.append(set())

    def pop_scope(self) -> None:
        self.scopes.pop()

    # ------------------------------------------------------------------
    # entry points
    # ------------------------------------------------------------------

    def parse_program(self) -> A.Program:
        return A.Program(self.parse_block({"EOF"}, allow_eof=True))

    def parse_block(self, terminators: set, allow_eof: bool = False) -> list:
        """Parse statements until a terminator token type is peeked."""
        stmts = []
        while True:
            self.skip_newlines()
            t = self.peek()
            if t.type in terminators:
                return stmts
            if t.type == "EOF":
                if allow_eof:
                    return stmts
                raise ParseError(
                    f"{t.line}:{t.col}: expected 'end', got end of file"
                )
            if t.type in ("END", "ELSE"):
                want = " / ".join(sorted(x.lower() for x in terminators))
                raise ParseError(
                    f"{t.line}:{t.col}: unexpected '{t.value}', "
                    f"expected {want}"
                )
            stmts.append(self.parse_statement())

    # ------------------------------------------------------------------
    # statements
    # ------------------------------------------------------------------

    def parse_statement(self) -> object:
        start = self.peek()
        node = self._parse_statement()
        # Every statement carries its source line for error messages.
        # (Plain attributes — dataclass equality is unaffected.)
        node.line = start.line
        node.col = start.col
        return node

    def _parse_statement(self) -> object:
        t = self.peek()
        if t.type == "FN":
            return self.parse_fn_def()
        if t.type == "STRUCT":
            return self.parse_struct_def()
        if t.type == "SHARE":
            return self.parse_share()
        if t.type == "TAKE":
            return self.parse_take()
        if t.type == "IF":
            return self.parse_if()
        if t.type == "WHILE":
            return self.parse_while()
        if t.type == "FOR":
            return self.parse_for()
        if t.type == "MATCH":
            return self.parse_match()
        if t.type == "RETURN":
            return self.parse_return()
        if t.type == "CRASH":
            return self.parse_crash()
        if t.type == "ASM":
            return self.parse_asm_block()
        if t.type == "MUT":
            self.advance()
            return self.parse_binding(mutable=True)
        if t.type == "IDENT" and self.peek2().type == "COLON":
            # annotated form always declares: `name: type = expr`
            return self.parse_binding(mutable=False)
        if (
            t.type == "IDENT"
            and self.peek2().type == "ASSIGN"
            and not self.is_declared(t.value)
        ):
            # first `name = expr` in scope declares
            return self.parse_binding(mutable=False)
        # otherwise: expression statement, maybe an assignment
        expr = self.parse_expr()
        if self.peek().type == "ASSIGN":
            self.advance()
            value = self.parse_expr()
            if not isinstance(expr, ASSIGNABLE):
                at = self.peek()
                raise ParseError(
                    f"{at.line}:{at.col}: cannot assign to this expression"
                )
            return A.Assign(expr, value)
        return A.ExprStmt(expr)

    def parse_binding(self, mutable: bool) -> A.Binding:
        name = self.expect("IDENT").value
        type_ = None
        if self.peek().type == "COLON":
            self.advance()
            type_ = self.parse_type()
        self.expect("ASSIGN")
        value = self.parse_expr()
        self.declare(name)
        return A.Binding(name, type_, value, mutable)

    def parse_fn_def(self) -> A.FnDef:
        self.expect("FN")
        name = self.expect("IDENT").value
        self.expect("LPAREN")
        params = []
        if self.peek().type != "RPAREN":
            while True:
                pmut = False
                if self.peek().type == "MUT":
                    self.advance()
                    pmut = True
                pname = self.expect("IDENT").value
                self.expect("COLON")
                ptype = self.parse_type()
                params.append(A.Param(pname, ptype, pmut))
                if self.peek().type == "COMMA":
                    self.advance()
                else:
                    break
        self.expect("RPAREN")
        ret = None
        if self.peek().type == "ARROW":
            self.advance()
            ret = self.parse_type()
        self.expect_newline("after function header")
        self.declare(name)  # visible for recursion, in enclosing scope
        self.push_scope()
        for p in params:
            self.declare(p.name)
        body = self.parse_block({"END"})
        self.expect("END")
        self.pop_scope()
        return A.FnDef(name, params, ret, body)

    def parse_struct_def(self) -> A.StructDef:
        self.expect("STRUCT")
        name = self.expect("IDENT").value
        self.expect_newline("after struct header")
        fields = []
        while True:
            self.skip_newlines()
            t = self.peek()
            if t.type == "END":
                self.advance()
                break
            if t.type == "EOF":
                raise ParseError(
                    f"{t.line}:{t.col}: expected 'end', got end of file"
                )
            fname = self.expect("IDENT").value
            self.expect("COLON")
            ftype = self.parse_type()
            fields.append(A.StructField(fname, ftype))
            t = self.peek()
            if t.type not in ("NEWLINE", "END", "EOF"):
                raise ParseError(
                    f"{t.line}:{t.col}: expected newline after struct field, "
                    f"got '{t.type.lower()}' ({t.value!r})"
                )
        self.declare(name)
        return A.StructDef(name, fields)

    def parse_share(self) -> A.Share:
        self.expect("SHARE")
        name = self.expect("IDENT").value
        self.expect("ASSIGN")
        value = self.parse_expr()
        self.declare(name)
        return A.Share(name, value)

    def parse_take(self) -> A.Take:
        self.expect("TAKE")
        names = [self.expect("IDENT").value]
        while self.peek().type == "COMMA":
            self.advance()
            names.append(self.expect("IDENT").value)
        self.expect("FROM")
        path = [self.expect("IDENT").value]
        while self.peek().type == "DOT":
            self.advance()
            path.append(self.expect("IDENT").value)
        for n in names:
            self.declare(n)
        return A.Take(names, path)

    def parse_if(self) -> A.If:
        self.expect("IF")
        cond = self.parse_expr()
        self.expect_newline("after 'if' condition")
        then_body = self.parse_block({"ELSE", "END"})
        else_body: list = []
        if self.peek().type == "ELSE":
            self.advance()
            self.expect_newline("after 'else'")
            else_body = self.parse_block({"END"})
        self.expect("END")
        return A.If(cond, then_body, else_body)

    def parse_while(self) -> A.While:
        self.expect("WHILE")
        cond = self.parse_expr()
        self.expect_newline("after 'while' condition")
        body = self.parse_block({"END"})
        self.expect("END")
        return A.While(cond, body)

    def parse_for(self) -> A.For:
        self.expect("FOR")
        var = self.expect("IDENT").value
        self.expect("IN")
        it = self.parse_expr()
        self.expect_newline("after 'for' header")
        self.declare(var)
        body = self.parse_block({"END"})
        self.expect("END")
        return A.For(var, it, body)

    def parse_match(self) -> A.Match:
        self.expect("MATCH")
        subject = self.parse_expr()
        self.expect_newline("after 'match' subject")
        arms = []
        while True:
            self.skip_newlines()
            t = self.peek()
            if t.type == "END":
                self.advance()
                break
            if t.type == "EOF":
                raise ParseError(
                    f"{t.line}:{t.col}: expected 'end', got end of file"
                )
            if t.type != "IDENT" or t.value not in ("ok", "err"):
                raise ParseError(
                    f"{t.line}:{t.col}: expected 'ok' or 'err' arm, "
                    f"got ({t.value!r})"
                )
            kind = t.value
            self.advance()
            self.expect("LPAREN")
            binding = self.expect("IDENT").value
            self.expect("RPAREN")
            self.expect("ARROW")
            self.declare(binding)
            body = self.parse_statement()
            arms.append(A.MatchArm(kind, binding, body))
            t = self.peek()
            if t.type not in ("NEWLINE", "END", "EOF"):
                raise ParseError(
                    f"{t.line}:{t.col}: expected newline after match arm, "
                    f"got '{t.type.lower()}' ({t.value!r})"
                )
        return A.Match(subject, arms)

    def parse_return(self) -> A.Return:
        self.expect("RETURN")
        if self.peek().type in ("NEWLINE", "END", "ELSE", "EOF"):
            return A.Return(None)
        return A.Return(self.parse_expr())

    def parse_crash(self) -> A.Crash:
        self.expect("CRASH")
        return A.Crash(self.parse_expr())

    def parse_asm_block(self) -> A.AsmBlock:
        """`asm` NEWLINE (directive NEWLINE)* `end`.

        Directives (any order; emission is always in-moves, raw bytes,
        out-moves):
          in <var> = <reg>
          out <var> = <reg>
          clobber "<reg>"
          raw <byte>, <byte>, ...
        """
        self.expect("ASM")
        self.expect_newline("after 'asm'")
        ins, outs, clobbers, raw_bytes = [], [], [], []
        while True:
            self.skip_newlines()
            t = self.peek()
            if t.type == "END":
                self.advance()
                break
            if t.type == "EOF":
                raise ParseError(
                    f"{t.line}:{t.col}: expected 'end' to close 'asm' block, "
                    f"got end of file"
                )
            if t.type == "IN":
                self.advance()
                var = self.expect("IDENT").value
                self.expect("ASSIGN")
                reg = self.expect("IDENT").value
                ins.append(A.AsmIn(var, reg))
            elif t.type == "OUT":
                self.advance()
                var = self.expect("IDENT").value
                self.expect("ASSIGN")
                reg = self.expect("IDENT").value
                outs.append(A.AsmOut(var, reg))
            elif t.type == "CLOBBER":
                self.advance()
                st = self.expect("STRING")
                clobbers.append(A.AsmClobber(st.value))
            elif t.type == "RAW":
                self.advance()
                while True:
                    bt = self.expect("INT")
                    v = bt.value
                    if not 0 <= v <= 255:
                        raise ParseError(
                            f"{bt.line}:{bt.col}: asm raw byte out of "
                            f"range 0-255: {v}"
                        )
                    raw_bytes.append(v)
                    if self.peek().type == "COMMA":
                        self.advance()
                    else:
                        break
            else:
                raise ParseError(
                    f"{t.line}:{t.col}: expected 'in', 'out', 'clobber' or "
                    f"'raw' inside 'asm' block, got '{t.value}'"
                )
            self.expect_newline("after asm directive")
        return A.AsmBlock(ins, outs, clobbers, raw_bytes)

    # ------------------------------------------------------------------
    # expressions
    # ------------------------------------------------------------------

    def parse_expr(self) -> object:
        return self.parse_bitor()

    def parse_comparison(self) -> object:
        left = self.parse_shift()
        while self.peek().type in COMP_OPS:
            op = self.advance().value
            right = self.parse_shift()
            left = A.BinOp(op, left, right)
        return left

    # Bitwise operators, C-like precedence (loosest to tightest):
    #   |  (loosest)          — PIPE token; in type position this same
    #                            token means union (`A | B`), but types and
    #                            expressions never overlap grammatically,
    #                            so the meaning is fixed by context.
    #   ^
    #   &
    #   then comparison (`== != < <= > >=`) — binds TIGHTER than `&`,
    #                            exactly like C.
    #   << >>                — binds looser than `+`/`-`, exactly like C:
    #                            `1 << 2 + 3` is `1 << (2 + 3)` = 32.
    #   (then additive/multiplicative bind tighter, exactly like C)
    # Note the C gotcha is preserved: `a & mask == 0` parses as
    # `a & (mask == 0)` because `&` binds looser than `==`. Parenthesize.
    def parse_bitor(self) -> object:
        left = self.parse_bitxor()
        while self.peek().type == "PIPE":
            self.advance()
            right = self.parse_bitxor()
            left = A.BinOp("|", left, right)
        return left

    def parse_bitxor(self) -> object:
        left = self.parse_bitand()
        while self.peek().type == "BITXOR":
            self.advance()
            right = self.parse_bitand()
            left = A.BinOp("^", left, right)
        return left

    def parse_bitand(self) -> object:
        left = self.parse_comparison()
        while self.peek().type == "BITAND":
            self.advance()
            right = self.parse_comparison()
            left = A.BinOp("&", left, right)
        return left

    def parse_additive(self) -> object:
        left = self.parse_multiplicative()
        while self.peek().type in ("PLUS", "MINUS"):
            op = self.advance().value
            right = self.parse_multiplicative()
            left = A.BinOp(op, left, right)
        return left

    def parse_shift(self) -> object:
        left = self.parse_additive()
        while self.peek().type in ("SHL", "SHR"):
            op = self.advance().value  # "<<" or ">>"
            right = self.parse_additive()
            left = A.BinOp(op, left, right)
        return left

    def parse_multiplicative(self) -> object:
        left = self.parse_unary()
        while self.peek().type in ("STAR", "SLASH", "PERCENT"):
            op = self.advance().value
            right = self.parse_unary()
            left = A.BinOp(op, left, right)
        return left

    def parse_unary(self) -> object:
        if self.peek().type == "MINUS":
            self.advance()
            return A.UnaryOp("-", self.parse_unary())
        if self.peek().type == "BITNOT":
            self.advance()
            return A.UnaryOp("~", self.parse_unary())
        if self.peek().type == "STAR":
            # `*p` — dereference a raw pointer. Desugared to `p[0]`
            # (a zero-offset access); the typechecker stamps it volatile
            # like every other raw-pointer access. `*p = v` works because
            # the desugared Index is an assignable target.
            self.advance()
            return A.Index(self.parse_unary(), A.IntLit(0))
        return self.parse_postfix()

    def parse_postfix(self) -> object:
        expr = self.parse_primary()
        while True:
            t = self.peek()
            if t.type == "LPAREN":
                expr = self.parse_call_args(expr)
            elif t.type == "LBRACKET":
                # `list[T]()` — list construction. Only `list` gets this
                # treatment, and only when the brackets really hold a type
                # followed by `()`; otherwise fall back to indexing (so a
                # variable that happens to be named `list` still works).
                if isinstance(expr, A.Var) and expr.name == "list":
                    saved = self.pos
                    try:
                        expr = self.parse_list_new()
                        continue
                    except ParseError:
                        self.pos = saved
                self.advance()
                index = self.parse_expr()
                self.expect("RBRACKET")
                expr = A.Index(expr, index)
            elif t.type == "DOT":
                self.advance()
                fname = self.expect("IDENT").value
                expr = A.FieldAccess(expr, fname)
            else:
                return expr

    def parse_call_args(self, func: object) -> object:
        self.expect("LPAREN")
        # Struct literal: `Name(field: expr, ...)` — an all-named-argument
        # call form. `Name()` with zero args stays a plain call.
        if (
            isinstance(func, A.Var)
            and self.peek().type == "IDENT"
            and self.peek2().type == "COLON"
        ):
            fields = []
            while True:
                fname = self.expect("IDENT").value
                self.expect("COLON")
                fval = self.parse_expr()
                fields.append((fname, fval))
                if self.peek().type == "COMMA":
                    self.advance()
                else:
                    break
            self.expect("RPAREN")
            return A.StructLit(func.name, fields)
        args = []
        if self.peek().type != "RPAREN":
            while True:
                args.append(self.parse_expr())
                if self.peek().type == "COMMA":
                    self.advance()
                else:
                    break
        self.expect("RPAREN")
        return A.Call(func, args)

    def parse_list_new(self) -> object:
        """`list[T]()` — after seeing `list` `[`. Raises ParseError unless
        the brackets hold a type and are followed by an empty `()`."""
        self.expect("LBRACKET")
        elem = self.parse_type()
        self.expect("RBRACKET")
        self.expect("LPAREN")
        self.expect("RPAREN")
        return A.ListNew(elem)

    def parse_primary(self) -> object:
        t = self.peek()
        if t.type == "INT":
            self.advance()
            return A.IntLit(t.value)
        if t.type == "STRING":
            self.advance()
            return A.StrLit(t.value)
        if t.type == "INTERP_STRING":
            self.advance()
            parts = [
                A.StrLit(text) if kind == "str" else A.Var(text)
                for kind, text in t.value
            ]
            return A.InterpString(parts)
        if t.type == "IDENT":
            self.advance()
            return A.Var(t.value)
        if t.type == "LPAREN":
            self.advance()
            expr = self.parse_expr()
            self.expect("RPAREN")
            return expr
        if t.type == "LBRACKET":
            # Array literal: `[1, 2, 3]`
            self.advance()
            elems = []
            if self.peek().type != "RBRACKET":
                elems.append(self.parse_expr())
                while self.peek().type == "COMMA":
                    self.advance()
                    elems.append(self.parse_expr())
            self.expect("RBRACKET")
            return A.ArrayLit(elems)
        raise ParseError(
            f"{t.line}:{t.col}: unexpected '{t.type.lower()}' "
            f"({t.value!r}) in expression"
        )

    # ------------------------------------------------------------------
    # types (parsed, not checked)
    # ------------------------------------------------------------------

    def parse_type(self) -> object:
        first = self.parse_primary_type()
        if self.peek().type != "PIPE":
            return first
        options = [first]
        while self.peek().type == "PIPE":
            self.advance()
            options.append(self.parse_primary_type())
        return A.UnionType(options)

    def parse_primary_type(self) -> object:
        t = self.peek()
        if t.type == "RAW":
            self.advance()
            self.expect("STAR")
            return A.RawType(self.parse_primary_type())
        if t.type == "FN":
            self.advance()
            self.expect("LPAREN")
            params = []
            if self.peek().type != "RPAREN":
                while True:
                    params.append(self.parse_type())
                    if self.peek().type == "COMMA":
                        self.advance()
                    else:
                        break
            self.expect("RPAREN")
            self.expect("ARROW")
            return A.FnType(params, self.parse_type())
        if t.type == "IDENT":
            self.advance()
            name = t.value
            if self.peek().type == "LBRACKET":
                self.advance()
                args: list = []
                if self.peek().type != "RBRACKET":
                    while True:
                        if self.peek().type == "INT":
                            args.append(self.advance().value)
                        else:
                            args.append(self.parse_type())
                        if self.peek().type == "COMMA":
                            self.advance()
                        else:
                            break
                self.expect("RBRACKET")
                if (
                    name == "list"
                    and len(args) == 1
                    and not isinstance(args[0], int)
                ):
                    return A.ListType(args[0])
                if (
                    name == "array"
                    and len(args) == 2
                    and not isinstance(args[0], int)
                    and isinstance(args[1], int)
                ):
                    return A.ArrayType(args[0], args[1])
                return A.NamedType(name, args)
            return A.NamedType(name)
        raise ParseError(
            f"{t.line}:{t.col}: expected a type, "
            f"got '{t.type.lower()}' ({t.value!r})"
        )


def parse(source: str) -> A.Program:
    """Lex + parse `source`, returning the Program AST."""
    return Parser(Lexer(source).tokenize()).parse_program()
