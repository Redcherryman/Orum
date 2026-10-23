"""Orum v0.006 tests: freestanding target.

Verifies: zero syscall bytes in freestanding binaries, say/exit
rejected with clear errors, crash -> cli/hlt, configurable entry
symbol honored in the ELF header, heap externs as UND symbols with
real R_X86_64_64 relocations, and the kernel_min.orum example.
Hosted behavior must be unchanged.
"""
import struct
import subprocess
import unittest

from orum.driver import build_binary, compile_source
from orum.typecheck import TypeCheckError

SYSCALL = b"\x0f\x05"


def e_entry(data: bytes) -> int:
    return struct.unpack("<Q", data[24:32])[0]


def readelf(path: str, *args) -> str:
    return subprocess.run(["readelf", *args, path], capture_output=True,
                          text=True).stdout


class TestFreestandingNoSyscalls(unittest.TestCase):
    def test_zero_syscall_bytes(self):
        src = ("mut x = 1\n"
               "mut i = 0\n"
               "while i < 100\n"
               "x = x + i\n"
               "i = i + 1\n"
               "end\n"
               "mut l = list[int]()\n"
               "l.push(x)\n"
               "mut y = l[0]\n"
               "crash \"done\"\n")
        b = compile_source(src, freestanding=True)
        self.assertEqual(b.count(SYSCALL), 0)

    def test_hosted_still_has_syscalls(self):
        b = compile_source("say(1)\n")
        self.assertGreater(b.count(SYSCALL), 0)

    def test_hosted_bytes_unchanged_shape(self):
        # hosted output must not gain sections
        b = compile_source("mut x = 1\n")
        shnum = struct.unpack("<H", b[60:62])[0]
        self.assertEqual(shnum, 0)


class TestFreestandingErrors(unittest.TestCase):
    def test_say_rejected(self):
        with self.assertRaises(TypeCheckError) as c:
            compile_source("say(1)\n", freestanding=True)
        self.assertIn("hosted mode", str(c.exception))

    def test_exit_rejected(self):
        with self.assertRaises(TypeCheckError) as c:
            compile_source("exit(0)\n", freestanding=True)
        self.assertIn("hosted mode", str(c.exception))

    def test_say_exit_fine_hosted(self):
        b = compile_source("say(42)\nexit(0)\n")  # must not raise
        self.assertGreater(len(b), 0)


class TestFreestandingCrash(unittest.TestCase):
    def test_crash_is_cli_hlt(self):
        b = compile_source('crash "boom"\n', freestanding=True)
        self.assertEqual(b.count(SYSCALL), 0)
        # cli (FA) followed eventually by hlt (F4) in a tight loop
        self.assertIn(b"\xfa", b)
        self.assertIn(b"\xf4", b)


class TestEntrySymbol(unittest.TestCase):
    def test_default_entry(self):
        b = compile_source("mut x = 1\n", freestanding=True)
        with open("/tmp/fs_entry", "wb") as f:
            f.write(b)
        out = readelf("/tmp/fs_entry", "-s")
        # entry symbol recorded SHN_ABS with the e_entry value
        self.assertIn("_start", out)
        for line in out.splitlines():
            if "_start" in line and "ABS" in line:
                val = int(line.split()[1], 16)
                self.assertEqual(val, e_entry(b))
                break
        else:
            self.fail("ABS _start not in symtab")

    def test_custom_entry(self):
        b = compile_source("mut x = 1\n", freestanding=True,
                           entry="kmain")
        with open("/tmp/fs_kmain", "wb") as f:
            f.write(b)
        out = readelf("/tmp/fs_kmain", "-s")
        self.assertIn("kmain", out)
        for line in out.splitlines():
            if "kmain" in line and "ABS" in line:
                val = int(line.split()[1], 16)
                self.assertEqual(val, e_entry(b))
                break
        else:
            self.fail("ABS kmain not in symtab")
        # "_start" must not appear as a standalone symbol (it does
        # appear inside "__orum_heap_start", so match whole words)
        import re as _re
        self.assertIsNone(_re.search(r"(?<![\w])_start(?![\w])",
                                     out.replace("kmain", "")))


class TestHeapExterns(unittest.TestCase):
    def test_undefined_heap_symbols(self):
        b = compile_source("mut l = list[int]()\nl.push(1)\n",
                           freestanding=True)
        with open("/tmp/fs_heap", "wb") as f:
            f.write(b)
        out = readelf("/tmp/fs_heap", "-s")
        self.assertIn("__orum_heap_start", out)
        self.assertIn("__orum_heap_end", out)
        for line in out.splitlines():
            if "__orum_heap_start" in line or "__orum_heap_end" in line:
                self.assertIn("UND", line)

    def test_relocations_against_heap_symbols(self):
        b = compile_source("mut l = list[int]()\nl.push(1)\n",
                           freestanding=True)
        with open("/tmp/fs_reloc", "wb") as f:
            f.write(b)
        out = readelf("/tmp/fs_reloc", "-r")
        self.assertIn("R_X86_64_64", out)
        self.assertIn("__orum_heap_start", out)
        self.assertIn("__orum_heap_end", out)
        # exactly two relocations (one per heap bound)
        self.assertEqual(out.count("R_X86_64_64"), 2)

    def test_no_heap_symbols_hosted(self):
        b = compile_source("mut l = list[int]()\nl.push(1)\n")
        self.assertNotIn(b"__orum_heap_start", b)


class TestKernelMinExample(unittest.TestCase):
    def test_compiles_freestanding(self):
        with open("examples/kernel_min.orum") as f:
            src = f.read()
        b = compile_source(src, freestanding=True, entry="kmain")
        self.assertEqual(b.count(SYSCALL), 0)
        self.assertGreater(len(b), 0)

    def test_cli_present(self):
        with open("examples/kernel_min.orum") as f:
            src = f.read()
        b = compile_source(src, freestanding=True, entry="kmain")
        self.assertIn(b"\xfa", b)  # cli from the asm block


class TestOrumcCli(unittest.TestCase):
    def test_cli_freestanding(self):
        r = subprocess.run(
            ["python3", "-m", "orum.orumc", "examples/kernel_min.orum",
             "-o", "/tmp/km_cli", "--freestanding", "--entry", "kmain"],
            capture_output=True, text=True, cwd=".")
        self.assertEqual(r.returncode, 0, r.stderr)
        with open("/tmp/km_cli", "rb") as f:
            b = f.read()
        self.assertEqual(b.count(SYSCALL), 0)

    def test_cli_error_propagates(self):
        r = subprocess.run(
            ["python3", "-m", "orum.orumc", "examples/fibonacci.orum",
             "-o", "/tmp/fib_fs", "--freestanding"],
            capture_output=True, text=True, cwd=".")
        # fibonacci.orum uses say -> must fail cleanly, not traceback
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("hosted mode", r.stderr)


if __name__ == "__main__":
    unittest.main()
