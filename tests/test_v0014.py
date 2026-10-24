"""End-to-end tests for Orum v0.014: hosted file I/O builtins.

fopen / fread / fwrite / fclose / fseek — the foundation for porting
Menagerie-Kernel's Python build tooling (PE writer, FAT writer, image
builder, proof scripts) to Orum, toward an Orum-only kernel repo.
"""
import os
import subprocess
import tempfile
import unittest

from orum.driver import build_binary, compile_source
from orum.typecheck import TypeCheckError


class FileIOTestBase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp(prefix="orum_fio_")
        cls.n = 0

    def run_orum(self, src):
        """Compile `src` and run it in the temp dir. Returns CompletedProcess."""
        type(self).n += 1
        path = os.path.join(self.tmp, f"f{self.n}")
        build_binary(src, path)
        return subprocess.run([path], capture_output=True, timeout=10,
                              cwd=self.tmp)


class TestFopenFwriteFclose(FileIOTestBase):
    def test_write_file(self):
        src = (
            'data: raw *u8 = [104, 101, 108, 108, 111]\n'  # "hello"
            'fd: int = fopen("t_write.bin", "w")\n'
            'if fd < 0\n'
            '  crash "open failed"\n'
            'end\n'
            'n: int = fwrite(fd, data, 5)\n'
            'if n != 5\n'
            '  crash "short write"\n'
            'end\n'
            'fclose(fd)\n'
        )
        p = self.run_orum(src)
        self.assertEqual(p.returncode, 0, f"stderr: {p.stderr!r}")
        with open(os.path.join(self.tmp, "t_write.bin"), "rb") as f:
            self.assertEqual(f.read(), b"hello")

    def test_rw_mode_creates_and_writes(self):
        src = (
            'data: raw *u8 = [65, 66]\n'  # "AB"
            'fd: int = fopen("t_rw.bin", "rw")\n'
            'if fd < 0\n'
            '  crash "open failed"\n'
            'end\n'
            'fwrite(fd, data, 2)\n'
            'fclose(fd)\n'
        )
        p = self.run_orum(src)
        self.assertEqual(p.returncode, 0, f"stderr: {p.stderr!r}")
        with open(os.path.join(self.tmp, "t_rw.bin"), "rb") as f:
            self.assertEqual(f.read(), b"AB")

    def test_open_missing_for_read_fails(self):
        src = (
            'fd: int = fopen("definitely-not-here-98765.bin", "r")\n'
            'if fd < 0\n'
            '  say(1)\n'
            'else\n'
            '  say(0)\n'
            'end\n'
        )
        p = self.run_orum(src)
        self.assertEqual(p.returncode, 0, f"stderr: {p.stderr!r}")
        self.assertEqual(p.stdout.decode(), "1\n")

    def test_bad_mode_returns_einval(self):
        src = 'fd: int = fopen("t_badmode.bin", "bogus")\nsay(fd)\n'
        p = self.run_orum(src)
        self.assertEqual(p.returncode, 0, f"stderr: {p.stderr!r}")
        self.assertEqual(p.stdout.decode(), "-22\n")


class TestFread(FileIOTestBase):
    def test_read_back_bytes(self):
        with open(os.path.join(self.tmp, "t_read.bin"), "wb") as f:
            f.write(bytes([10, 20, 30, 40, 50]))
        src = (
            'mut slot: u64 = 0\n'
            'buf: raw *u8 = rawptr(addr(slot))\n'
            'fd: int = fopen("t_read.bin", "r")\n'
            'if fd < 0\n'
            '  crash "open failed"\n'
            'end\n'
            'n: int = fread(fd, buf, 5)\n'
            'say(n)\n'
            'say(buf[0])\n'
            'say(buf[1])\n'
            'say(buf[2])\n'
            'say(buf[3])\n'
            'say(buf[4])\n'
            'fclose(fd)\n'
        )
        p = self.run_orum(src)
        self.assertEqual(p.returncode, 0, f"stderr: {p.stderr!r}")
        self.assertEqual(p.stdout.decode(), "5\n10\n20\n30\n40\n50\n")

    def test_read_eof_returns_zero(self):
        with open(os.path.join(self.tmp, "t_eof.bin"), "wb") as f:
            f.write(b"Z")
        src = (
            'mut slot: u64 = 0\n'
            'buf: raw *u8 = rawptr(addr(slot))\n'
            'fd: int = fopen("t_eof.bin", "r")\n'
            'mut n: int = fread(fd, buf, 1)\n'
            'say(n)\n'
            'n = fread(fd, buf, 1)\n'
            'say(n)\n'
            'fclose(fd)\n'
        )
        p = self.run_orum(src)
        self.assertEqual(p.returncode, 0, f"stderr: {p.stderr!r}")
        self.assertEqual(p.stdout.decode(), "1\n0\n")


class TestHeapAlloc(FileIOTestBase):
    def test_alloc_write_read(self):
        src = (
            'buf: raw *u8 = heap_alloc(16)\n'
            'buf[0] = 65\n'
            'buf[1] = 66\n'
            'buf[15] = 67\n'
            'say(buf[0])\n'
            'say(buf[1])\n'
            'say(buf[15])\n'
        )
        p = self.run_orum(src)
        self.assertEqual(p.returncode, 0, f"stderr: {p.stderr!r}")
        self.assertEqual(p.stdout.decode(), "65\n66\n67\n")

    def test_alloc_feeds_fwrite(self):
        src = (
            'buf: raw *u8 = heap_alloc(4)\n'
            'buf[0] = 68\n'
            'buf[1] = 69\n'
            'fd: int = fopen("t_heap.bin", "w")\n'
            'fwrite(fd, buf, 2)\n'
            'fclose(fd)\n'
        )
        p = self.run_orum(src)
        self.assertEqual(p.returncode, 0, f"stderr: {p.stderr!r}")
        with open(os.path.join(self.tmp, "t_heap.bin"), "rb") as f:
            self.assertEqual(f.read(), b"DE")


class TestFseek(FileIOTestBase):
    def test_seek_and_overwrite(self):
        src = (
            'data: raw *u8 = [97, 98, 99, 100, 101, 102, 103, 104]\n'  # abcdefgh
            'one: raw *u8 = [88]\n'  # "X"
            'fd: int = fopen("t_seek.bin", "w")\n'
            'fwrite(fd, data, 8)\n'
            'pos: u64 = fseek(fd, 4, 0)\n'
            'say(pos)\n'
            'fwrite(fd, one, 1)\n'
            'fclose(fd)\n'
        )
        p = self.run_orum(src)
        self.assertEqual(p.returncode, 0, f"stderr: {p.stderr!r}")
        self.assertEqual(p.stdout.decode(), "4\n")
        with open(os.path.join(self.tmp, "t_seek.bin"), "rb") as f:
            self.assertEqual(f.read(), b"abcdXfgh")


class TestFreestandingRejected(unittest.TestCase):
    def test_fopen_rejected(self):
        with self.assertRaises(TypeCheckError) as c:
            compile_source('fopen("x", "r")\n', freestanding=True)
        self.assertIn("hosted mode", str(c.exception))

    def test_fread_rejected(self):
        with self.assertRaises(TypeCheckError) as c:
            compile_source(
                'mut s: u64 = 0\n'
                'b: raw *u8 = rawptr(addr(s))\n'
                'fread(0, b, 1)\n',
                freestanding=True)
        self.assertIn("hosted mode", str(c.exception))

    def test_fwrite_rejected(self):
        with self.assertRaises(TypeCheckError) as c:
            compile_source(
                'b: raw *u8 = [65]\n'
                'fwrite(1, b, 1)\n',
                freestanding=True)
        self.assertIn("hosted mode", str(c.exception))

    def test_fclose_rejected(self):
        with self.assertRaises(TypeCheckError) as c:
            compile_source('fclose(1)\n', freestanding=True)
        self.assertIn("hosted mode", str(c.exception))

    def test_fseek_rejected(self):
        with self.assertRaises(TypeCheckError) as c:
            compile_source('fseek(0, 0, 0)\n', freestanding=True)
        self.assertIn("hosted mode", str(c.exception))

    def test_heap_alloc_rejected(self):
        with self.assertRaises(TypeCheckError) as c:
            compile_source('heap_alloc(16)\n', freestanding=True)
        self.assertIn("hosted mode", str(c.exception))

    def test_arg_type_errors(self):
        with self.assertRaises(TypeCheckError):
            compile_source('fopen(123, "r")\n')  # path must be str
        with self.assertRaises(TypeCheckError):
            compile_source('fclose("x")\n')  # fd must be int
        with self.assertRaises(TypeCheckError):
            compile_source('fwrite(1, 2, 3)\n')  # buf must be raw ptr


if __name__ == "__main__":
    unittest.main()
