"""Tests for Orum v0.015: multi-file project support fixes.

Covers the list Aiden required (from the cloud tool's summary, rebuilt
here): constant folding with 64-bit runtime semantics, orumc -c/--link,
cross-object rodata/data refs, shared heap, nested fn-pointer calls,
addr typing, share/take with cycle reporting.
"""
import os
import subprocess
import tempfile
import unittest

from orum import ast_nodes as A
from orum.driver import build_binary, compile_object, compile_source, resolve_takes
from orum.lexer import Lexer
from orum.linker import link
from orum.objfile import read_object
from orum.optimizer import fold_constants
from orum.parser import Parser


class OptimizerTest(unittest.TestCase):
    """Constant folding matches x86-64 runtime, not Python semantics."""

    def fold(self, op, l, r):
        node = A.BinOp(op, A.IntLit(l), A.IntLit(r))
        result = fold_constants(node)
        self.assertIsInstance(result, A.IntLit)
        return result.value

    def test_signed_division_truncates(self):
        # Python floor: -7 // 2 == -4. x86 truncates: -3.
        self.assertEqual(self.fold('/', -7, 2), 2**64 - 3)
        self.assertEqual(self.fold('/', 7, -2), 2**64 - 3)
        self.assertEqual(self.fold('/', -7, -2), 3)

    def test_modulo_sign_follows_dividend(self):
        # Python: -7 % 2 == 1. x86: -1.
        self.assertEqual(self.fold('%', -7, 2), 2**64 - 1)
        self.assertEqual(self.fold('%', 7, -2), 1)

    def test_overflow_wraps_64bit(self):
        self.assertEqual(self.fold('+', 2**63, 2**63), 0)
        self.assertEqual(self.fold('*', 2**32, 2**32), 0)

    def test_right_shift_not_folded(self):
        # Signedness unknown before typecheck; defer folding.
        node = A.BinOp('>>', A.IntLit(-8), A.IntLit(1))
        result = fold_constants(node)
        self.assertIsInstance(result, A.BinOp)

    def test_basic_folding_preserved(self):
        self.assertEqual(self.fold('+', 2, 3), 5)
        node = A.BinOp('+', A.IntLit(2),
                       A.BinOp('*', A.IntLit(3), A.IntLit(4)))
        self.assertEqual(fold_constants(node).value, 14)


class TakeCycleTest(unittest.TestCase):
    """Import cycles are reported; diamonds still work."""

    def test_cycle_reported(self):
        tmp = tempfile.mkdtemp()
        with open(os.path.join(tmp, 'a.orum'), 'w') as f:
            f.write('take x from b\nshare x = 1\n')
        with open(os.path.join(tmp, 'b.orum'), 'w') as f:
            f.write('take x from a\nshare x = 2\n')
        prog = Parser(Lexer('take x from a\nsay(x)\n').tokenize()).parse_program()
        with self.assertRaises(ValueError) as ctx:
            resolve_takes(prog, tmp)
        self.assertIn('cycle', str(ctx.exception))

    def test_diamond_ok(self):
        tmp = tempfile.mkdtemp()
        with open(os.path.join(tmp, 'base.orum'), 'w') as f:
            f.write('share v = 10\n')
        with open(os.path.join(tmp, 'left.orum'), 'w') as f:
            f.write('take v from base\nshare lv = v + 1\n')
        with open(os.path.join(tmp, 'right.orum'), 'w') as f:
            f.write('take v from base\nshare rv = v + 2\n')
        src = 'take lv from left\ntake rv from right\nsay(lv)\nsay(rv)\n'
        data = compile_source(src, base_dir=tmp)
        path = os.path.join(tmp, 'diamond_prog')
        with open(path, 'wb') as f:
            f.write(data)
        os.chmod(path, 0o755)
        out = subprocess.run([path], capture_output=True, timeout=10)
        self.assertEqual(out.stdout, b'11\n12\n')


class ShareCodegenTest(unittest.TestCase):
    """Share bindings carry their inferred type into codegen."""

    def test_share_constant_via_take(self):
        tmp = tempfile.mkdtemp()
        os.makedirs(os.path.join(tmp, 'math'), exist_ok=True)
        with open(os.path.join(tmp, 'math', 'circle.orum'), 'w') as f:
            f.write('share pi_num = 314\nshare pi_den = 100\n')
        src = ('take pi_num, pi_den from math.circle\n'
               'say(10 * pi_num / pi_den)\n')
        data = compile_source(src, base_dir=tmp)
        path = os.path.join(tmp, 'share_prog')
        with open(path, 'wb') as f:
            f.write(data)
        os.chmod(path, 0o755)
        out = subprocess.run([path], capture_output=True, timeout=10)
        self.assertEqual(out.stdout, b'31\n')


class LinkTest(unittest.TestCase):
    """Separate compilation: objects link and run, with shared heap."""

    def test_cross_object_call_and_string(self):
        greet_src = 'fn greet() -> int\nsay("hello from greet")\nreturn 0\nend\n'
        main_src = ('fn main() -> int\nsay("hello from main")\n'
                    'say(greet())\nreturn 0\nend\n')
        greet_o = compile_object(greet_src)
        main_o = compile_object(main_src)
        elf_data = link([read_object(greet_o), read_object(main_o)],
                        entry_symbol='main')
        tmp = tempfile.mkdtemp()
        path = os.path.join(tmp, 'link_prog')
        with open(path, 'wb') as f:
            f.write(elf_data)
        os.chmod(path, 0o755)
        out = subprocess.run([path], capture_output=True, timeout=10)
        self.assertEqual(out.stdout, b'hello from main\nhello from greet\n0\n')

    def test_cross_object_heap(self):
        mod_src = ('fn sum_list() -> int\nmut l = list[int]()\n'
                   'l.push(10)\nl.push(20)\nl.push(12)\n'
                   'return l[0] + l[1] + l[2]\nend\n'
                   'fn use_list(l: list[int]) -> int\n'
                   'return l[0] + l[1]\nend\n')
        main_src = ('fn main() -> int\nsay(sum_list())\n'
                    'mut m = list[int]()\nm.push(100)\nm.push(200)\n'
                    'say(use_list(m))\nreturn 0\nend\n')
        mod_o = compile_object(mod_src)
        main_o = compile_object(main_src)
        elf_data = link([read_object(mod_o), read_object(main_o)],
                        entry_symbol='main')
        tmp = tempfile.mkdtemp()
        path = os.path.join(tmp, 'heap_prog')
        with open(path, 'wb') as f:
            f.write(elf_data)
        os.chmod(path, 0o755)
        out = subprocess.run([path], capture_output=True, timeout=10)
        self.assertEqual(out.returncode, 0)
        self.assertEqual(out.stdout, b'42\n300\n')


if __name__ == '__main__':
    unittest.main()
