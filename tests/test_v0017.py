"""Tests for Orum v0.017: x86-64 mnemonics in `asm` blocks.

Mnemonic lines assemble at parse time into the block's body bytes,
interleaved with `raw` bytes in source order. Labels are block-local
and sit on their own line; jumps/calls use rel32. Mnemonic operands
are registers, immediates, memory operands, and labels only — Orum
values cross via `in` / `out`. A line starting with `in` / `out` is
an Orum directive only as `in|out <ident> = <ident>`; otherwise it
is the x86 port instruction.
"""
import unittest

from orum.driver import compile_source
from orum.parser import ParseError
from orum.x86asm import AsmError, assemble_asm_body


def asm_bytes(items):
    return assemble_asm_body([(k, v, 1) for k, v in items])


def m(line):
    return ('mnem', line)


def L(name):
    return ('label', name)


class MnemonicEncodingTest(unittest.TestCase):
    def test_no_operand(self):
        self.assertEqual(asm_bytes([m('cli')]), bytes([0xFA]))
        self.assertEqual(asm_bytes([m('hlt')]), bytes([0xF4]))
        self.assertEqual(asm_bytes([m('nop')]), bytes([0x90]))
        self.assertEqual(asm_bytes([m('sti')]), bytes([0xFB]))
        self.assertEqual(asm_bytes([m('ret')]), bytes([0xC3]))

    def test_mov_reg_seg_and_cr(self):
        # mov rax, cs — the exact bytes the kernel's get_cs used to emit
        # via raw bytes.
        self.assertEqual(asm_bytes([m('mov rax, cs')]),
                         bytes([0x48, 0x8C, 0xC8]))
        self.assertEqual(asm_bytes([m('mov rax, cr0')]),
                         bytes([0x48, 0x0F, 0x20, 0xC0]))
        self.assertEqual(asm_bytes([m('mov cr3, rax')]),
                         bytes([0x48, 0x0F, 0x22, 0xD8]))

    def test_mov_reg_reg_and_mem(self):
        self.assertEqual(asm_bytes([m('mov rax, rbx')]),
                         bytes([0x48, 0x8B, 0xC3]))
        self.assertEqual(asm_bytes([m('mov rax, [rbx+8]')]),
                         bytes([0x48, 0x8B, 0x43, 0x08]))
        self.assertEqual(asm_bytes([m('mov [rax], rbx')]),
                         bytes([0x48, 0x89, 0x18]))
        self.assertEqual(asm_bytes([m('mov rax, [rbx+rcx*4+16]')]),
                         bytes([0x48, 0x8B, 0x44, 0x8B, 0x10]))

    def test_mov_imm(self):
        self.assertEqual(asm_bytes([m('mov rax, 0x2000000')]),
                         bytes([0x48, 0xB8, 0x00, 0x00, 0x00, 0x02,
                                0x00, 0x00, 0x00, 0x00]))
        self.assertEqual(asm_bytes([m('mov eax, 5')]),
                         bytes([0xB8, 0x05, 0x00, 0x00, 0x00]))
        self.assertEqual(asm_bytes([m('mov al, 0xFF')]),
                         bytes([0xB0, 0xFF]))

    def test_push_pop(self):
        self.assertEqual(asm_bytes([m('push r12')]), bytes([0x41, 0x54]))
        self.assertEqual(asm_bytes([m('pop rbp')]), bytes([0x5D]))
        self.assertEqual(asm_bytes([m('push 5')]), bytes([0x6A, 0x05]))

    def test_alu(self):
        self.assertEqual(asm_bytes([m('xor eax, eax')]),
                         bytes([0x31, 0xC0]))
        # add rax, 5 uses the accumulator short form.
        self.assertEqual(asm_bytes([m('add rax, 5')]),
                         bytes([0x48, 0x05, 0x05, 0x00, 0x00, 0x00]))
        self.assertEqual(asm_bytes([m('cmp rax, rbx')]),
                         bytes([0x48, 0x39, 0xD8]))
        self.assertEqual(asm_bytes([m('sub rsp, 8')]),
                         bytes([0x48, 0x83, 0xEC, 0x08]))

    def test_shifts(self):
        self.assertEqual(asm_bytes([m('shl rax, 1')]),
                         bytes([0x48, 0xD1, 0xE0]))
        self.assertEqual(asm_bytes([m('shr rax, cl')]),
                         bytes([0x48, 0xD3, 0xE8]))

    def test_labels_forward_and_backward(self):
        # Backward: hlt at 0, jmp back to 0 => rel32 -6 (end at 6).
        self.assertEqual(asm_bytes([L('spin'), m('hlt'), m('jmp spin')]),
                         bytes([0xF4, 0xE9, 0xFA, 0xFF, 0xFF, 0xFF]))
        # Forward: jmp over a 1-byte nop to the label at 6.
        self.assertEqual(asm_bytes([m('jmp done'), m('nop'), L('done'),
                                    m('nop')]),
                         bytes([0xE9, 0x01, 0x00, 0x00, 0x00,
                                0x90, 0x90]))
        # Conditional jump, forward.
        self.assertEqual(asm_bytes([m('je done'), m('nop'), L('done')]),
                         bytes([0x0F, 0x84, 0x01, 0x00, 0x00, 0x00,
                                0x90]))

    def test_call(self):
        self.assertEqual(asm_bytes([m('call target'), L('target'),
                                    m('ret')]),
                         bytes([0xE8, 0x00, 0x00, 0x00, 0x00, 0xC3]))

    def test_port_io(self):
        self.assertEqual(asm_bytes([m('in al, dx')]), bytes([0xEC]))
        self.assertEqual(asm_bytes([m('out dx, al')]), bytes([0xEE]))
        self.assertEqual(asm_bytes([m('in eax, 0x60')]),
                         bytes([0xE5, 0x60]))

    def test_system(self):
        self.assertEqual(asm_bytes([m('lgdt [rax]')]),
                         bytes([0x0F, 0x01, 0x10]))
        self.assertEqual(asm_bytes([m('lidt [rax]')]),
                         bytes([0x0F, 0x01, 0x18]))
        self.assertEqual(asm_bytes([m('invlpg [rax]')]),
                         bytes([0x0F, 0x01, 0x38]))

    def test_raw_and_mnemonic_interleave(self):
        # Source order is preserved: raw, mnemonic, raw.
        self.assertEqual(asm_bytes([('raw', 0x90), m('nop'),
                                    ('raw', 0x90)]),
                         bytes([0x90, 0x90, 0x90]))

    def test_errors(self):
        with self.assertRaises(AsmError):
            asm_bytes([m('frobnicate rax')])
        with self.assertRaises(AsmError):
            asm_bytes([m('jmp nowhere')])
        with self.assertRaises(AsmError):
            asm_bytes([L('dup'), L('dup')])
        with self.assertRaises(AsmError):
            asm_bytes([m('mov rax')])
        with self.assertRaises(AsmError):
            asm_bytes([m('mov [rax], [rbx]')])


class MnemonicIntegrationTest(unittest.TestCase):
    def test_mnemonic_block_compiles(self):
        src = ('fn main() -> int\n'
               'mut x: int = 0\n'
               'asm\n'
               'in x = rax\n'
               'cli\n'
               'mov rax, cr0\n'
               'spin:\n'
               'hlt\n'
               'jmp spin\n'
               'out x = rax\n'
               'end\n'
               'return 0\n'
               'end\n')
        data = compile_source(src, entry='main')
        blob = data[192:192 + 4096]
        # cli (FA), mov rax,cr0 (48 0F 20 C0), hlt (F4) all present.
        self.assertIn(b'\xfa', blob)
        self.assertIn(bytes([0x48, 0x0F, 0x20, 0xC0]), blob)
        self.assertIn(b'\xf4', blob)

    def test_port_instructions_not_directives(self):
        # `in al, dx` / `out dx, al` are x86 instructions, not Orum
        # in/out directives (which require `ident = ident`).
        src = ('fn main() -> int\n'
               'asm\n'
               'in al, dx\n'
               'out dx, al\n'
               'end\n'
               'return 0\n'
               'end\n')
        data = compile_source(src, entry='main')
        blob = data[192:192 + 4096]
        self.assertIn(b'\xec', blob)
        self.assertIn(b'\xee', blob)

    def test_bad_mnemonic_is_parse_error(self):
        src = ('fn main() -> int\n'
               'asm\n'
               'frobnicate rax\n'
               'end\n'
               'return 0\n'
               'end\n')
        with self.assertRaises(ParseError):
            compile_source(src, entry='main')

    def test_unknown_label_is_parse_error(self):
        src = ('fn main() -> int\n'
               'asm\n'
               'jmp nowhere\n'
               'end\n'
               'return 0\n'
               'end\n')
        with self.assertRaises(ParseError):
            compile_source(src, entry='main')


if __name__ == '__main__':
    unittest.main()
