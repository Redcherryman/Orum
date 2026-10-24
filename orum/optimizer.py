"""Constant folding optimizer for Orum.

Evaluates constant expressions at compile time:
- `2 + 3` -> `5`
- `10 * 4` -> `40`
- etc.

Run after parsing, before typechecking.

64-bit consistency: all folded results are masked to 64 bits (wrapping),
matching Orum's x86-64 runtime. Division and modulo use C/x86 truncating
semantics (quotient truncates toward zero, remainder sign follows the
dividend), NOT Python's floor semantics. Right-shift is NOT folded here:
it runs before typechecking, so signedness is unknown, and arithmetic
vs logical shift differ. Right-shift folding is deferred until the
typechecker knows whether the value is signed or unsigned.
"""
from . import ast_nodes as A

MASK64 = (1 << 64) - 1


def _wrap64(v):
    """Wrap an integer to the 64-bit unsigned range (0..2^64-1).

    The bit pattern is what matters for codegen: a wrapped value emitted
    as a 64-bit immediate has the same bits whether later interpreted as
    signed or unsigned.
    """
    return v & MASK64


def _trunc_div(a, b):
    """C/x86 truncating division: quotient truncates toward zero.

    Python's // is floor division (-7 // 2 == -4); x86 idiv truncates
    (-7 / 2 == -3). This matches the runtime.
    """
    q = abs(a) // abs(b)
    if (a < 0) != (b < 0):
        q = -q
    return q


def _c_mod(a, b):
    """C/x86 modulo: remainder sign follows the dividend.

    Python's % sign follows the divisor (-7 % 2 == 1); x86 gives -1.
    """
    return a - _trunc_div(a, b) * b


def fold_constants(node):
    """Recursively fold constants in the AST. Returns the (possibly new) node."""
    if isinstance(node, A.Program):
        node.statements = [fold_constants(s) for s in node.statements]
        return node
    if isinstance(node, A.FnDef):
        node.body = [fold_constants(s) for s in node.body]
        return node
    if isinstance(node, A.Binding):
        node.value = fold_constants(node.value)
        return node
    if isinstance(node, A.BinOp):
        node.left = fold_constants(node.left)
        node.right = fold_constants(node.right)
        # If both are IntLit, evaluate
        if isinstance(node.left, A.IntLit) and isinstance(node.right, A.IntLit):
            l, r = node.left.value, node.right.value
            op = node.op
            try:
                if op == "+":
                    return A.IntLit(_wrap64(l + r))
                elif op == "-":
                    return A.IntLit(_wrap64(l - r))
                elif op == "*":
                    return A.IntLit(_wrap64(l * r))
                elif op == "/" and r != 0:
                    return A.IntLit(_wrap64(_trunc_div(l, r)))
                elif op == "%" and r != 0:
                    return A.IntLit(_wrap64(_c_mod(l, r)))
                elif op == "&":
                    return A.IntLit(_wrap64(l & r))
                elif op == "|":
                    return A.IntLit(_wrap64(l | r))
                elif op == "^":
                    return A.IntLit(_wrap64(l ^ r))
                elif op == "<<":
                    return A.IntLit(_wrap64(l << r))
                # NOTE: ">>" is intentionally NOT folded here. The
                # optimizer runs before typechecking, so signedness is
                # unknown, and arithmetic shift (signed) vs logical
                # shift (unsigned) give different results for negative
                # values. Folding is deferred until the typechecker
                # knows the type.
            except (OverflowError, ValueError, ZeroDivisionError):
                pass
        return node
    if isinstance(node, A.UnaryOp):
        node.operand = fold_constants(node.operand)
        if isinstance(node.operand, A.IntLit):
            v = node.operand.value
            if node.op == "-":
                return A.IntLit(_wrap64(-v))
            elif node.op == "~":
                return A.IntLit(_wrap64(~v))
        return node
    # For other nodes, recursively process children if they have them
    # (simplified — just return as-is for now)
    return node
