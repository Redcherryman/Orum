"""Tests for Orum v0.016: sizeof/offsetof builtins.

sizeof(T)/sizeof(x) return the byte size as u64; offsetof(S, f) returns
the byte offset of field f in struct S's word-granular layout. Both are
compile-time constants computed by typecheck and emitted as immediates.
"""
import os
import subprocess
import tempfile
import unittest

from orum.driver import compile_source
from orum.typecheck import TypeCheckError


def run_prog(src, entry='main'):
    data = compile_source(src, entry=entry)
    tmp = tempfile.mkdtemp()
    path = os.path.join(tmp, 'prog')
    with open(path, 'wb') as f:
        f.write(data)
    os.chmod(path, 0o755)
    out = subprocess.run([path], capture_output=True, timeout=10)
    return out.returncode


class SizeofTest(unittest.TestCase):
    def test_sizeof_primitives(self):
        src = ('fn main() -> u64\n'
               'return sizeof(u8) + sizeof(u16) + sizeof(u32) + sizeof(u64)\nend\n')
        # 1 + 2 + 4 + 8 = 15
        self.assertEqual(run_prog(src), 15)

    def test_sizeof_struct_and_value(self):
        src = ('struct Point\nx: u64\ny: u64\nend\n'
               'struct Mixed\na: u8\nb: u64\nc: u32\nend\n'
               'fn main() -> u64\n'
               'mut p = Point(x: 1, y: 2)\n'
               'return sizeof(Point) + sizeof(Mixed) + sizeof(p)\nend\n')
        # Point 16 + Mixed 24 (3 word-granular fields) + p 16 = 56
        self.assertEqual(run_prog(src), 56)

    def test_offsetof(self):
        src = ('struct Point\nx: u64\ny: u64\nend\n'
               'struct Mixed\na: u8\nb: u64\nc: u32\nend\n'
               'fn main() -> u64\n'
               'return offsetof(Point, y) + offsetof(Mixed, b) + offsetof(Mixed, c)\nend\n')
        # y 8 + b 8 (after u8 word) + c 16 (after u8 + u64 words) = 32
        self.assertEqual(run_prog(src), 32)

    def test_offsetof_first_field_zero(self):
        src = ('struct Point\nx: u64\ny: u64\nend\n'
               'fn main() -> u64\nreturn offsetof(Point, x)\nend\n')
        self.assertEqual(run_prog(src), 0)

    def test_offsetof_unknown_field_errors(self):
        src = ('struct Point\nx: u64\ny: u64\nend\n'
               'fn main() -> u64\nreturn offsetof(Point, z)\nend\n')
        with self.assertRaises(TypeCheckError):
            compile_source(src, entry='main')

    def test_sizeof_undeclared_errors(self):
        src = ('fn main() -> u64\nreturn sizeof(nosuchthing)\nend\n')
        with self.assertRaises(TypeCheckError):
            compile_source(src, entry='main')


class TakeFilterTest(unittest.TestCase):
    """v0.016: take pulls only named decls + their deps, not the module."""

    def _mod_dir(self):
        tmp = tempfile.mkdtemp()
        with open(os.path.join(tmp, 'mod.orum'), 'w') as f:
            f.write('fn helper() -> u64\nreturn 7\nend\n'
                    'fn used() -> u64\nreturn helper() + 1\nend\n'
                    'fn unused() -> u64\nreturn 99\nend\n')
        return tmp

    def test_filter_drops_unused(self):
        from orum.driver import resolve_takes
        from orum.lexer import Lexer
        from orum.parser import Parser
        tmp = self._mod_dir()
        src = 'take used from mod\nfn main() -> u64\nreturn used()\nend\n'
        prog = Parser(Lexer(src).tokenize()).parse_program()
        prog = resolve_takes(prog, tmp)
        names = [getattr(s, 'name', None) for s in prog.statements]
        self.assertIn('used', names)
        self.assertIn('helper', names)
        self.assertNotIn('unused', names)

    def test_filter_missing_name_errors(self):
        from orum.driver import resolve_takes
        from orum.lexer import Lexer
        from orum.parser import Parser
        tmp = self._mod_dir()
        src = 'take nosuch from mod\nfn main() -> u64\nreturn 0\nend\n'
        prog = Parser(Lexer(src).tokenize()).parse_program()
        with self.assertRaises(ValueError):
            resolve_takes(prog, tmp)

    def test_filter_diamond_no_duplicates(self):
        from orum.driver import resolve_takes
        from orum.lexer import Lexer
        from orum.parser import Parser
        tmp = self._mod_dir()
        src = ('take used from mod\ntake unused from mod\n'
               'fn main() -> u64\nreturn used() + unused()\nend\n')
        prog = Parser(Lexer(src).tokenize()).parse_program()
        prog = resolve_takes(prog, tmp)
        names = [getattr(s, 'name', None) for s in prog.statements]
        self.assertEqual(names.count('used'), 1)
        self.assertEqual(names.count('helper'), 1)
        self.assertEqual(names.count('unused'), 1)

    def test_filter_compiles_and_runs(self):
        tmp = self._mod_dir()
        src = 'take used from mod\nfn main() -> u64\nreturn used()\nend\n'
        data = compile_source(src, entry='main', base_dir=tmp)
        tmp2 = tempfile.mkdtemp()
        path = os.path.join(tmp2, 'prog')
        with open(path, 'wb') as f:
            f.write(data)
        os.chmod(path, 0o755)
        out = subprocess.run([path], capture_output=True, timeout=10)
        self.assertEqual(out.returncode, 8)


if __name__ == '__main__':
    unittest.main()
