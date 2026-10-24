"""Constant folding optimizer for Orum.

Evaluates constant expressions at compile time:
- `2 + 3` -> `5`
- `10 * 4` -> `40`
- etc.

Run after parsing, before typechecking.
"""
from . import ast_nodes as A


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
                    return A.IntLit(l + r)
                elif op == "-":
                    return A.IntLit(l - r)
                elif op == "*":
                    return A.IntLit(l * r)
                elif op == "/" and r != 0:
                    return A.IntLit(l // r)
                elif op == "%" and r != 0:
                    return A.IntLit(l % r)
                elif op == "&":
                    return A.IntLit(l & r)
                elif op == "|":
                    return A.IntLit(l | r)
                elif op == "^":
                    return A.IntLit(l ^ r)
                elif op == "<<":
                    return A.IntLit(l << r)
                elif op == ">>":
                    return A.IntLit(l >> r)
            except (OverflowError, ValueError):
                pass
        return node
    if isinstance(node, A.UnaryOp):
        node.operand = fold_constants(node.operand)
        if isinstance(node.operand, A.IntLit):
            v = node.operand.value
            if node.op == "-":
                return A.IntLit(-v)
            elif node.op == "~":
                return A.IntLit(~v)
        return node
    # For other nodes, recursively process children if they have them
    # (simplified — just return as-is for now)
    return node
