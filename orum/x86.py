"""Minimal x86-64 machine-code emitter for Orum v0.001.

Emits raw bytes for a small, fixed instruction set: enough for integer
arithmetic, calls (System V), stack frames, branches, and syscalls.
Labels are resolved to relative offsets at the end (resolve()).
Absolute 64-bit addresses (for .rodata) are left as placeholders for
the ELF writer to patch.

Register numbers follow the standard encoding:
rax=0 rcx=1 rdx=2 rbx=3 rsp=4 rbp=5 rsi=6 rdi=7 r8=8 ... r15=15
"""
from __future__ import annotations


# registers
RAX, RCX, RDX, RBX, RSP, RBP, RSI, RDI = range(8)
R8, R9, R10, R11, R12, R13, R14, R15 = range(8, 16)

# condition codes for jcc/setcc
CC_O, CC_NO, CC_B, CC_AE, CC_E, CC_NE, CC_BE, CC_A = range(8)
CC_S, CC_NS, CC_P, CC_NP, CC_L, CC_GE, CC_LE, CC_G = range(8, 16)

# argument registers (System V ABI)
ARG_REGS = [RDI, RSI, RDX, RCX, R8, R9]


class AsmError(Exception):
    pass


class Emitter:
    def __init__(self):
        self.code = bytearray()
        self.labels: dict[str, int] = {}
        self.patches: list[tuple[int, str]] = []  # (pos_of_rel32, label)

    # -- low-level encoding ------------------------------------------------

    @staticmethod
    def _rex_w(r: int = 0, b: int = 0) -> bytes:
        return bytes([0x48 | ((r >> 3) & 1) << 2 | ((b >> 3) & 1)])

    @staticmethod
    def _rex(r: int = 0, b: int = 0) -> bytes:
        p = 0x40 | ((r >> 3) & 1) << 2 | ((b >> 3) & 1)
        return bytes([p]) if p != 0x40 else b""

    @staticmethod
    def _modrm(mod: int, reg: int, rm: int) -> bytes:
        return bytes([(mod << 6) | ((reg & 7) << 3) | (rm & 7)])

    def _mem(self, reg: int, base: int, disp: int) -> bytes:
        """ModRM [+ SIB] + displacement bytes for [base + disp],
        with `reg` in the ModRM reg field."""
        out = bytearray()
        bl = base & 7
        if disp == 0 and bl != 5:  # rbp/r13 need an explicit disp8 of 0
            mod, db = 0, b""
        elif -128 <= disp <= 127:
            mod, db = 1, int(disp).to_bytes(1, "little", signed=True)
        else:
            mod, db = 2, int(disp).to_bytes(4, "little", signed=True)
        if bl == 4:  # rsp/r12: SIB byte required
            out += self._modrm(mod, reg, 4)
            out.append(0x24)  # scale=0, index=none, base=rsp
        else:
            out += self._modrm(mod, reg, bl)
        out += db
        return bytes(out)

    def _rel32(self, label: str):
        # Caller has already emitted the opcode; record where the
        # 4-byte relative offset goes and leave a zero placeholder.
        self.patches.append((len(self.code), label))
        self.code += b"\x00\x00\x00\x00"

    # -- labels -------------------------------------------------------------

    def label(self, name: str):
        if name in self.labels:
            raise AsmError(f"duplicate label {name}")
        self.labels[name] = len(self.code)

    def resolve(self):
        for pos, label in self.patches:
            if label not in self.labels:
                raise AsmError(f"undefined label {label}")
            rel = self.labels[label] - (pos + 4)
            self.code[pos : pos + 4] = rel.to_bytes(4, "little", signed=True)

    # -- data movement -------------------------------------------------------

    def mov_rr(self, dst: int, src: int):
        self.code += self._rex_w(dst, src) + b"\x8b"
        self.code += self._modrm(3, dst, src)

    def mov_rr32(self, dst: int, src: int):
        """mov r32, r32 — no REX.W, so the destination's upper 32 bits
        are zeroed (zero-extension). Used for `out x = eax`-style asm."""
        self.code += self._rex(dst, src) + b"\x8b"
        self.code += self._modrm(3, dst, src)

    def mov_rm(self, dst: int, base: int, disp: int):
        self.code += self._rex_w(dst, base) + b"\x8b"
        self.code += self._mem(dst, base, disp)

    def mov_rm32(self, dst: int, base: int, disp: int):
        """mov r32, [base+disp] — 32-bit load, zero-extends to 64 bits.
        Used for `in x = eax`-style asm."""
        self.code += self._rex(dst, base) + b"\x8b"
        self.code += self._mem(dst, base, disp)

    def mov_mr(self, base: int, disp: int, src: int):
        self.code += self._rex_w(src, base) + b"\x89"
        self.code += self._mem(src, base, disp)

    def mov_ri64(self, dst: int, imm: int):
        self.code += self._rex_w(0, dst) + bytes([0xB8 + (dst & 7)])
        # v0.004: mask to 64 bits and encode unsigned — large u64
        # literals (e.g. 0x8000000000000000) don't fit signed=True.
        # The emitted bytes are identical for values that fit signed.
        self.code += (int(imm) & 0xFFFFFFFFFFFFFFFF).to_bytes(
            8, "little", signed=False)

    def movabs_placeholder(self, dst: int) -> int:
        """Emit mov r64, <8 zero bytes>; return offset of the immediate
        for later patching (used for .rodata addresses)."""
        self.code += self._rex_w(0, dst) + bytes([0xB8 + (dst & 7)])
        pos = len(self.code)
        self.code += b"\x00" * 8
        return pos

    def mov_ri32(self, dst: int, imm: int):
        # mov r/m64, imm32 (sign-extended) — 7 bytes, good for small consts
        self.code += self._rex_w(0, dst) + b"\xc7"
        self.code += self._modrm(3, 0, dst)
        self.code += int(imm).to_bytes(4, "little", signed=True)

    def lea(self, dst: int, base: int, disp: int):
        self.code += self._rex_w(dst, base) + b"\x8d"
        self.code += self._mem(dst, base, disp)

    def push_r(self, r: int):
        if r >= 8:
            self.code += b"\x41"
        self.code.append(0x50 + (r & 7))

    def pop_r(self, r: int):
        if r >= 8:
            self.code += b"\x41"
        self.code.append(0x58 + (r & 7))

    def load_sized(self, dst: int, base: int, disp: int, size: int,
                   signed: bool):
        """Load 1/2/4/8 bytes from [base+disp] into dst (zero/sign-extended)."""
        if size == 8:
            self.mov_rm(dst, base, disp)
            return
        if size == 4:
            # mov r32, r/m32 — the CPU zero-extends into the full 64-bit
            # dst. (There is no `movzx r64, r/m32`: 0x0F 0xB7 is r/m16,
            # so the old code loaded only 16 bits here.)
            if signed:
                # movsxd r64, r/m32
                self.code += self._rex_w(dst, base) + b"\x63"
                self.code += self._mem(dst, base, disp)
            else:
                self.mov_rm32(dst, base, disp)
            return
        if size == 2:
            op2 = 0xBF if signed else 0xB7
            # 0x0F 0xB7 = movzx r64, r/m16 ; 0x0F 0xBF = movsx r64, r/m16
        elif size == 1:
            op2 = 0xBE if signed else 0xB6
        else:
            raise AsmError(f"bad load size {size}")
        self.code += self._rex_w(dst, base) + b"\x0f" + bytes([op2])
        self.code += self._mem(dst, base, disp)

    def store_sized(self, base: int, disp: int, size: int, src: int):
        """Store low `size` bytes of src to [base+disp]."""
        if size == 8:
            self.mov_mr(base, disp, src)
        elif size == 4:
            self.code += self._rex(src, base) + b"\x89"
            self.code += self._mem(src, base, disp)
        elif size == 2:
            self.code += b"\x66" + self._rex(src, base) + b"\x89"
            self.code += self._mem(src, base, disp)
        elif size == 1:
            self.code += self._rex(src, base) + b"\x88"
            self.code += self._mem(src, base, disp)
        else:
            raise AsmError(f"bad store size {size}")

    # -- arithmetic ----------------------------------------------------------

    def _alu_ri(self, dst: int, imm: int, op_ext: int):
        if -128 <= imm <= 127:
            self.code += self._rex_w(0, dst) + b"\x83"
            self.code += self._modrm(3, op_ext, dst)
            self.code += int(imm).to_bytes(1, "little", signed=True)
        else:
            self.code += self._rex_w(0, dst) + b"\x81"
            self.code += self._modrm(3, op_ext, dst)
            self.code += int(imm).to_bytes(4, "little", signed=True)

    def add_rr(self, dst: int, src: int):
        # 0x01 is `add r/m64, r64`: ModRM.reg=src, ModRM.rm=dst, so
        # REX.R extends src and REX.B extends dst.
        self.code += self._rex_w(src, dst) + b"\x01"
        self.code += self._modrm(3, src, dst)

    def add_ri(self, dst: int, imm: int):
        self._alu_ri(dst, imm, 0)

    def sub_rr(self, dst: int, src: int):
        # 0x29 is `sub r/m64, r64`: ModRM.reg=src, ModRM.rm=dst.
        self.code += self._rex_w(src, dst) + b"\x29"
        self.code += self._modrm(3, src, dst)

    def sub_ri(self, dst: int, imm: int):
        self._alu_ri(dst, imm, 5)

    def imul_rr(self, dst: int, src: int):
        self.code += self._rex_w(dst, src) + b"\x0f\xaf"
        self.code += self._modrm(3, dst, src)

    def neg_r(self, r: int):
        self.code += self._rex_w(0, r) + b"\xf7"
        self.code += self._modrm(3, 3, r)

    def dec_r(self, r: int):
        self.code += self._rex_w(0, r) + b"\xff"
        self.code += self._modrm(3, 1, r)

    def mov_m8i(self, base: int, disp: int, imm: int):
        """mov byte [base+disp], imm8."""
        self.code += self._rex(0, base) + b"\xc6"
        self.code += self._mem(0, base, disp)
        self.code.append(imm & 0xFF)

    def mov_m8r(self, base: int, disp: int, src: int):
        """mov byte [base+disp], src_low8."""
        self.code += self._rex(0, base) + b"\x88"
        self.code += self._mem(src, base, disp)

    def cqo(self):
        self.code += b"\x48\x99"

    def xor_edx(self):
        self.code += b"\x31\xd2"  # xor edx, edx (no REX needed)

    def idiv_r(self, r: int):
        self.code += self._rex_w(0, r) + b"\xf7"
        self.code += self._modrm(3, 7, r)

    def div_r(self, r: int):
        self.code += self._rex_w(0, r) + b"\xf7"
        self.code += self._modrm(3, 6, r)

    def xor_rr(self, dst: int, src: int):
        # 0x31 is `xor r/m64, r64`: ModRM.reg=src, ModRM.rm=dst.
        self.code += self._rex_w(src, dst) + b"\x31"
        self.code += self._modrm(3, src, dst)

    # -- bitwise (v0.004) --------------------------------------------------------

    def and_rr(self, dst: int, src: int):
        # 0x21 is `and r/m64, r64`: ModRM.reg=src, ModRM.rm=dst, so
        # REX.R extends src and REX.B extends dst.
        self.code += self._rex_w(src, dst) + b"\x21"
        self.code += self._modrm(3, src, dst)

    def or_rr(self, dst: int, src: int):
        # 0x09 is `or r/m64, r64`: ModRM.reg=src, ModRM.rm=dst.
        self.code += self._rex_w(src, dst) + b"\x09"
        self.code += self._modrm(3, src, dst)

    def not_r(self, r: int):
        # 0xF7 /2 is `not r/m64`.
        self.code += self._rex_w(0, r) + b"\xf7"
        self.code += self._modrm(3, 2, r)

    def shl_rcl(self, dst: int):
        # 0xD3 /4 is `shl r/m64, cl`: count comes from CL.
        self.code += self._rex_w(0, dst) + b"\xd3"
        self.code += self._modrm(3, 4, dst)

    def shr_rcl(self, dst: int):
        # 0xD3 /5 is `shr r/m64, cl`: logical shift right.
        self.code += self._rex_w(0, dst) + b"\xd3"
        self.code += self._modrm(3, 5, dst)

    def sar_rcl(self, dst: int):
        # 0xD3 /7 is `sar r/m64, cl`: arithmetic shift right.
        self.code += self._rex_w(0, dst) + b"\xd3"
        self.code += self._modrm(3, 7, dst)

    def and_ri(self, dst: int, imm: int):
        self._alu_ri(dst, imm, 4)

    def rep_movsb(self):
        """rep movsb — copy RCX bytes from [RSI] to [RDI].
        RSI/RDI/RCX are advanced; RCX ends at 0."""
        self.code += b"\xf3\xa4"

    def shl_ri(self, dst: int, imm: int):
        self.code += self._rex_w(0, dst) + b"\xc1"
        self.code += self._modrm(3, 4, dst)
        self.code.append(imm & 0xFF)

    # -- comparison / branches -------------------------------------------------

    def cmp_rr(self, a: int, b: int):
        """cmp a, b  (sets flags for a - b)."""
        self.code += self._rex_w(b, a) + b"\x39"
        self.code += self._modrm(3, b, a)

    def cmp_rm(self, a: int, base: int, disp: int):
        """cmp a, [base+disp]."""
        self.code += self._rex_w(a, base) + b"\x3b"
        self.code += self._mem(a, base, disp)

    def test_rr(self, a: int, b: int):
        self.code += self._rex_w(b, a) + b"\x85"
        self.code += self._modrm(3, b, a)

    def jmp(self, label: str):
        self.code += b"\xe9"
        self._rel32(label)

    def jcc(self, cc: int, label: str):
        self.code += b"\x0f" + bytes([0x80 + cc])
        self._rel32(label)

    def setcc_bool(self, cc: int, dst: int):
        """dst = (condition) ? 1 : 0, zero-extended to 64 bits."""
        self.code += self._rex(0, dst) + b"\x0f" + bytes([0x90 + cc])
        self.code += self._modrm(3, 0, dst)
        # movzx dst, dst_low8
        self.code += self._rex_w(dst, dst) + b"\x0f\xb6"
        self.code += self._modrm(3, dst, dst)

    # -- calls -----------------------------------------------------------------

    def call(self, label: str):
        self.code += b"\xe8"
        self._rel32(label)

    def ret(self):
        self.code += b"\xc3"

    def syscall(self):
        self.code += b"\x0f\x05"

    def nop(self):
        self.code += b"\x90"
