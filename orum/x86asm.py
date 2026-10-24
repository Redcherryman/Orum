"""Small x86-64 assembler for Orum `asm` blocks.

Mnemonic lines live alongside the existing `in` / `out` / `clobber` /
`raw` directives. A mnemonic line operates on registers, immediates,
memory operands, and block-local labels only — it cannot name Orum
variables. Values cross the boundary via `in` / `out` (move a variable
into a register, use the register in mnemonics, move it back).

Labels are block-local and must sit on their own line (`name:`).
Jumps, calls, and rip-relative memory references always use a 32-bit
displacement, so instruction sizes are fixed in a single pass and
labels resolve without iteration.

Example block::

    asm
    in x = rax
    cli
    mov rax, cr0
    or rax, 0x80000000
    mov cr0, rax
    hlt_loop:
    hlt
    jmp hlt_loop
    out y = rax
    end

Operand syntax (case-insensitive):
  registers: rax/rcx/rdx/rbx/rsp/rbp/rsi/rdi/r8-r15 (+ d/w/b forms),
             al/cl/dl/bl/spl/bpl/sil/dil, ah/ch/dh/bh,
             cs/ds/es/fs/gs/ss, cr0/cr2/cr3/cr4/cr8
  immediates: decimal, 0x hex, 0b binary, negative with a leading '-'
  memory: [rax], [rax+8], [rax+rbx*4+16], [0x9000], [rip+label]
  labels: a bare identifier, only as a jmp/call/jcc target or [rip+label]

Supported instructions (v1): mov, push, pop, lea, movzx, movsx, xchg,
bswap, add, sub, adc, sbb, and, or, xor, cmp, test, inc, dec, neg, not,
shl, shr, sar, jmp, je/jz, jne/jnz, ja, jb, jae, jbe, jg, jl, jge, jle
(and aliases), call, ret, nop, cli, sti, hlt, cld, std, int3, int,
in, out, lgdt, lidt, sgdt, sidt, invlpg, ltr, rdtsc, cpuid, pause,
syscall, sysret, sysenter, sysexit.
"""


class AsmError(Exception):
    """Raised for an unparseable or unencodable mnemonic line."""


# ----------------------------------------------------------------------
# Registers: name -> (code, size_bits, kind)
# kind: 'gpr' (code >= 8 needs a REX bit), 'high8' (ah/ch/dh/bh, REX
# forbidden), 'seg', 'cr'.
# ----------------------------------------------------------------------

def _gpr(names64, names32, names16, names8):
    d = {}
    for i, n in enumerate(names64):
        d[n] = (i, 64, 'gpr')
    for i, n in enumerate(names32):
        d[n] = (i, 32, 'gpr')
    for i, n in enumerate(names16):
        d[n] = (i, 16, 'gpr')
    for i, n in enumerate(names8):
        d[n] = (i, 8, 'gpr')
    return d


REGISTERS = {}
REGISTERS.update(_gpr(
    ['rax', 'rcx', 'rdx', 'rbx', 'rsp', 'rbp', 'rsi', 'rdi',
     'r8', 'r9', 'r10', 'r11', 'r12', 'r13', 'r14', 'r15'],
    ['eax', 'ecx', 'edx', 'ebx', 'esp', 'ebp', 'esi', 'edi',
     'r8d', 'r9d', 'r10d', 'r11d', 'r12d', 'r13d', 'r14d', 'r15d'],
    ['ax', 'cx', 'dx', 'bx', 'sp', 'bp', 'si', 'di',
     'r8w', 'r9w', 'r10w', 'r11w', 'r12w', 'r13w', 'r14w', 'r15w'],
    ['al', 'cl', 'dl', 'bl', 'spl', 'bpl', 'sil', 'dil',
     'r8b', 'r9b', 'r10b', 'r11b', 'r12b', 'r13b', 'r14b', 'r15b'],
))
# High-byte registers: code is the low 3 bits; REX must never be emitted.
for _n, _c in [('ah', 4), ('ch', 5), ('dh', 6), ('bh', 7)]:
    REGISTERS[_n] = (_c, 8, 'high8')
# Segment registers: code is the ModRM.reg value (ES=0, CS=1, SS=2,
# DS=3, FS=4, GS=5).
for _n, _c in [('es', 0), ('cs', 1), ('ss', 2), ('ds', 3), ('fs', 4), ('gs', 5)]:
    REGISTERS[_n] = (_c, 16, 'seg')
# Control registers.
for _n, _c in [('cr0', 0), ('cr2', 2), ('cr3', 3), ('cr4', 4), ('cr8', 8)]:
    REGISTERS[_n] = (_c, 64, 'cr')


# ----------------------------------------------------------------------
# Operands
# ----------------------------------------------------------------------

class Operand:
    def __init__(self, kind, **kw):
        self.kind = kind  # 'reg' | 'imm' | 'mem' | 'label'
        self.__dict__.update(kw)

    def __repr__(self):
        return f"Operand({self.kind}, {self.__dict__})"


def _parse_int(text):
    """Parse a decimal / 0x / 0b integer literal (optional leading '-')."""
    t = text.strip()
    neg = False
    if t.startswith('-'):
        neg = True
        t = t[1:].strip()
    elif t.startswith('+'):
        t = t[1:].strip()
    try:
        if t.startswith(('0x', '0X')):
            v = int(t, 16)
        elif t.startswith(('0b', '0B')):
            v = int(t, 2)
        else:
            v = int(t, 10)
    except ValueError:
        raise AsmError(f"bad immediate: {text!r}")
    return -v if neg else v


def _split_top_commas(s):
    """Split an operand list on commas (no commas occur inside brackets)."""
    return [p.strip() for p in s.split(',') if p.strip() != '']


def _parse_mem(inner):
    """Parse the inside of [...] into a mem Operand."""
    # Tokenize: identifiers, numbers, +, -, *.
    toks = []
    i = 0
    while i < len(inner):
        ch = inner[i]
        if ch.isspace():
            i += 1
        elif ch in '+-*':
            toks.append(ch)
            i += 1
        elif ch.isalnum() or ch == '_':
            j = i
            while j < len(inner) and (inner[j].isalnum() or inner[j] == '_'):
                j += 1
            toks.append(inner[i:j])
            i = j
        else:
            raise AsmError(f"bad character in memory operand: {ch!r}")
    # Walk terms: [sign] term (('*' scale)?) ...
    base = None
    index = None
    scale = 1
    disp = 0
    rip = False
    rip_label = None
    sign = 1
    k = 0
    terms = 0
    while k < len(toks):
        t = toks[k]
        if t == '+':
            sign = 1
            k += 1
            continue
        if t == '-':
            sign = -1
            k += 1
            continue
        if t == '*':
            raise AsmError("bad memory operand: '*' without an index register")
        low = t.lower()
        if low == 'rip':
            if rip:
                raise AsmError("bad memory operand: duplicate 'rip'")
            rip = True
            k += 1
        elif low in REGISTERS and REGISTERS[low][2] == 'gpr':
            code, size, _ = REGISTERS[low]
            if size != 64:
                raise AsmError(
                    f"bad memory operand: address register must be 64-bit, "
                    f"got {t!r}")
            if k + 2 < len(toks) and toks[k + 1] == '*':
                # index*scale
                try:
                    sc = int(toks[k + 2], 10)
                except ValueError:
                    raise AsmError(
                        f"bad memory operand: bad scale {toks[k + 2]!r}")
                if sc not in (1, 2, 4, 8):
                    raise AsmError(
                        f"bad memory operand: scale must be 1, 2, 4 or 8, "
                        f"got {sc}")
                if index is not None:
                    raise AsmError("bad memory operand: two index registers")
                if code == 4:
                    raise AsmError(
                        "bad memory operand: rsp cannot be an index register")
                index = code
                scale = sc
                k += 3
            else:
                if base is not None:
                    raise AsmError("bad memory operand: two base registers")
                base = code
                k += 1
        else:
            # Numeric displacement, or a label (only valid with rip).
            try:
                v = _parse_int(t)
                disp += sign * v
                k += 1
            except AsmError:
                if rip and rip_label is None and sign == 1:
                    rip_label = t
                    k += 1
                else:
                    raise AsmError(
                        f"bad memory operand: unknown term {t!r}")
        terms += 1
        sign = 1
    if terms == 0:
        raise AsmError("empty memory operand")
    if rip:
        if base is not None or index is not None:
            raise AsmError("bad memory operand: rip with base/index register")
        if rip_label is not None and disp != 0:
            raise AsmError(
                "bad memory operand: rip + label + displacement unsupported")
        return Operand('mem', base=None, index=None, scale=1, disp=disp,
                       rip=True, rip_label=rip_label)
    if base is None and index is not None:
        raise AsmError("bad memory operand: index without a base register")
    return Operand('mem', base=base, index=index, scale=scale, disp=disp,
                   rip=False, rip_label=None)


def parse_operand(text):
    """Parse one operand string into an Operand."""
    t = text.strip()
    if not t:
        raise AsmError("empty operand")
    if t.startswith('['):
        if not t.endswith(']'):
            raise AsmError(f"unbalanced '[' in operand: {text!r}")
        return _parse_mem(t[1:-1])
    low = t.lower()
    if low in REGISTERS:
        code, size, kind = REGISTERS[low]
        return Operand('reg', code=code, size=size, rkind=kind, name=low)
    # Immediate? (a leading '-' or digit / 0x / 0b)
    if t[0].isdigit() or (t[0] == '-' and len(t) > 1 and t[1].isdigit()):
        return Operand('imm', value=_parse_int(t))
    # Otherwise it must be a label (jmp/call/jcc target).
    if all(c.isalnum() or c == '_' for c in t):
        return Operand('label', name=t)
    raise AsmError(f"cannot parse operand: {text!r}")


def parse_mnemonic_line(line):
    """Split a mnemonic line into (opcode, [operands])."""
    s = line.strip()
    if not s:
        raise AsmError("empty mnemonic line")
    parts = s.split(None, 1)
    op = parts[0].lower()
    ops = []
    if len(parts) > 1:
        for p in _split_top_commas(parts[1]):
            ops.append(parse_operand(p))
    return op, ops


# ----------------------------------------------------------------------
# Encoding helpers
# ----------------------------------------------------------------------

def _rex(w, r, x, b):
    """REX byte, or b'' when no REX bits are needed."""
    v = 0x40 | (w << 3) | (r << 2) | (x << 1) | b
    return bytes([v]) if v != 0x40 else b''


def _modrm(mod, reg, rm):
    return bytes([(mod << 6) | ((reg & 7) << 3) | (rm & 7)])


def _reg_rex_bit(op):
    """REX extension bit for a register operand (1 for r8-r15)."""
    if op.kind == 'reg' and op.rkind == 'gpr' and op.code >= 8:
        return 1
    return 0


def _check_no_high8_with_rex(ops):
    """ah/ch/dh/bh cannot appear in an instruction that needs REX."""
    for op in ops:
        if op.kind == 'reg' and op.rkind == 'high8':
            for other in ops:
                if other is not op and other.kind == 'reg' \
                        and other.rkind == 'gpr' and other.code >= 8:
                    raise AsmError(
                        "high-byte register cannot be used with r8-r15")


def _encode_rm(reg_field, rm_op, size, rex_w, rex_r_extra=0):
    """Encode ModRM (+SIB + displacement) for a reg/mem operand.

    reg_field: the 3-bit reg value. rm_op: a reg or mem Operand.
    size: operand size (for the 0x66 prefix decision, handled by caller).
    Returns (prefix_bytes, modrm_sib_disp_bytes, rex_r, rex_x, rex_b)
    where the caller assembles REX from w/r/x/b.
    """
    if rm_op.kind == 'reg':
        if rm_op.rkind == 'high8':
            # No REX allowed; caller must ensure no REX bits are needed.
            pass
        r = _reg_rex_bit(rm_op)
        return _modrm(3, reg_field, rm_op.code & 7), r, 0, 0
    if rm_op.kind != 'mem':
        raise AsmError("operand is neither register nor memory")
    base, index, scale, disp = rm_op.base, rm_op.index, rm_op.scale, rm_op.disp
    # RIP-relative.
    if rm_op.rip:
        # mod=00, rm=101, disp32. Displacement filled in by the caller
        # for labels; numeric disp is encoded directly.
        return _modrm(0, reg_field, 5), 0, 0, 0
    # Absolute [disp32]: SIB with no base.
    if base is None:
        sib_scale = {1: 0, 2: 1, 4: 2, 8: 3}[1]
        sib = bytes([(sib_scale << 6) | (4 << 3) | 5])
        return _modrm(0, reg_field, 4) + sib, 0, 0, 0
    # Decide mod from displacement size, with the rbp special case.
    if disp == 0 and (base & 7) != 5:
        mod = 0
        disp_bytes = b''
    elif -128 <= disp <= 127:
        mod = 1
        disp_bytes = bytes([disp & 0xFF])
    else:
        mod = 2
        disp_bytes = int(disp).to_bytes(4, 'little', signed=True)
    if (base & 7) == 5 and mod == 0:
        # [rbp] with no displacement must use mod=01, disp8=0.
        mod = 1
        disp_bytes = b'\x00'
    need_sib = ((base & 7) == 4) or (index is not None)
    if need_sib:
        scale_bits = {1: 0, 2: 1, 4: 2, 8: 3}[scale]
        idx_bits = (index & 7) if index is not None else 4
        sib = bytes([(scale_bits << 6) | (idx_bits << 3) | (base & 7)])
        out = _modrm(mod, reg_field, 4) + sib + disp_bytes
    else:
        out = _modrm(mod, reg_field, base & 7) + disp_bytes
    rex_b = 1 if base >= 8 else 0
    rex_x = 1 if (index is not None and index >= 8) else 0
    return out, rex_b, rex_x, 0


def _size_prefix(size):
    """Operand-size prefix for 16-bit ops (0x66); none otherwise."""
    return b'\x66' if size == 16 else b''


def _fits_i8(v):
    return -128 <= v <= 127


def _fits_i32(v):
    return -2**31 <= v < 2**31


# ----------------------------------------------------------------------
# Instruction encoders. Each returns bytes; `ctx` carries label offsets
# and the offset of the instruction being encoded (for rel32).
# ----------------------------------------------------------------------

class _Ctx:
    def __init__(self, labels, offset, sizing=False):
        self.labels = labels  # name -> block offset
        self.offset = offset  # block offset of this instruction
        # During pass 1 (sizing), forward labels are not defined yet;
        # their displacement is a placeholder because all jumps use
        # rel32, so size never depends on the displacement value.
        self.sizing = sizing

    def label_disp(self, name, instr_size):
        if name not in self.labels:
            if self.sizing:
                return 0
            raise AsmError(f"unknown label: {name!r}")
        return self.labels[name] - (self.offset + instr_size)


def _alu_opcode(op):
    """Return (r/m,reg opcode, reg,r/m opcode, /digit, acc opcode)."""
    table = {
        'add': (0x01, 0x03, 0, 0x05),
        'or': (0x09, 0x0B, 1, 0x0D),
        'adc': (0x11, 0x13, 2, 0x15),
        'sbb': (0x19, 0x1B, 3, 0x1D),
        'and': (0x21, 0x23, 4, 0x25),
        'sub': (0x29, 0x2B, 5, 0x2D),
        'xor': (0x31, 0x33, 6, 0x35),
        'cmp': (0x39, 0x3B, 7, 0x3D),
    }
    return table[op]


def _encode_mov(op, ops, ctx):
    if len(ops) != 2:
        raise AsmError("mov takes two operands")
    dst, src = ops
    # mov reg, seg / seg, reg
    if src.kind == 'reg' and src.rkind == 'seg':
        if dst.kind != 'reg' or dst.rkind == 'seg':
            raise AsmError("mov to segment register needs a GPR destination")
        _check_no_high8_with_rex(ops)
        rex = _rex(1 if dst.size == 64 else 0, 0, 0, _reg_rex_bit(dst))
        return rex + b'\x8c' + _modrm(3, src.code, dst.code & 7)
    if dst.kind == 'reg' and dst.rkind == 'seg':
        if src.kind != 'reg' or src.rkind == 'seg':
            raise AsmError("mov from segment register needs a GPR source")
        if dst.name == 'cs':
            raise AsmError("mov to cs is not a valid instruction")
        _check_no_high8_with_rex(ops)
        rex = _rex(0, 0, 0, _reg_rex_bit(src))
        return rex + b'\x8e' + _modrm(3, dst.code, src.code & 7)
    # mov reg, cr / cr, reg
    if src.kind == 'reg' and src.rkind == 'cr':
        if dst.kind != 'reg' or dst.size != 64:
            raise AsmError("mov from control register needs a 64-bit GPR")
        return b'\x48' + b'\x0f\x20' + _modrm(3, src.code, dst.code & 7) \
            if dst.code < 8 else _rex(1, 0, 0, 1) + b'\x0f\x20' \
            + _modrm(3, src.code, dst.code & 7)
    if dst.kind == 'reg' and dst.rkind == 'cr':
        if src.kind != 'reg' or src.size != 64:
            raise AsmError("mov to control register needs a 64-bit GPR")
        if src.code < 8:
            return b'\x48' + b'\x0f\x22' + _modrm(3, dst.code, src.code & 7)
        return _rex(1, 0, 0, 1) + b'\x0f\x22' \
            + _modrm(3, dst.code, src.code & 7)
    # mov reg, imm
    if src.kind == 'imm':
        if dst.kind != 'reg' or dst.rkind == 'seg' or dst.rkind == 'cr':
            raise AsmError("mov immediate needs a GPR destination")
        v = src.value
        if dst.size == 64:
            if not _fits_i32(v) and not (0 <= v < 2**64):
                raise AsmError(f"immediate out of range for mov: {v}")
            rex = _rex(1, 0, 0, _reg_rex_bit(dst))
            return rex + bytes([0xB8 + (dst.code & 7)]) \
                + int(v).to_bytes(8, 'little', signed=False)
        if dst.size == 32:
            if not (-2**31 <= v < 2**32):
                raise AsmError(f"immediate out of range for mov: {v}")
            rex = _rex(0, 0, 0, _reg_rex_bit(dst))
            return rex + bytes([0xB8 + (dst.code & 7)]) \
                + int(v & 0xFFFFFFFF).to_bytes(4, 'little')
        if dst.size == 16:
            if not (-2**15 <= v < 2**16):
                raise AsmError(f"immediate out of range for mov: {v}")
            rex = _rex(0, 0, 0, _reg_rex_bit(dst))
            return b'\x66' + rex + bytes([0xB8 + (dst.code & 7)]) \
                + int(v & 0xFFFF).to_bytes(2, 'little')
        # 8-bit
        if not (-2**7 <= v < 2**8):
            raise AsmError(f"immediate out of range for mov: {v}")
        if dst.rkind == 'high8':
            return bytes([0xB0 + (dst.code & 7)]) + bytes([v & 0xFF])
        rex = _rex(0, 0, 0, _reg_rex_bit(dst))
        return rex + bytes([0xB0 + (dst.code & 7)]) + bytes([v & 0xFF])
    # mov mem, imm
    if dst.kind == 'mem' and src.kind == 'imm':
        v = src.value
        # Size comes from... a mem,imm has no inherent size. Require the
        # user to disambiguate via a size on the immediate? v1: default
        # to 32-bit (the common case, e.g. mov [0x9000], 48). For other
        # sizes the user writes mov with an explicit register-sized op.
        if not _fits_i32(v):
            raise AsmError(f"immediate out of range for mov mem,imm: {v}")
        modrm, rex_b, rex_x, _ = _encode_rm(0, dst, 32, 0)
        return _rex(0, 0, rex_x, rex_b) + b'\xc7' + modrm \
            + int(v).to_bytes(4, 'little', signed=True)
    # mov reg/mem, reg and reg, reg/mem
    if dst.kind == 'reg' and src.kind in ('reg', 'mem'):
        if dst.rkind == 'seg' or dst.rkind == 'cr':
            raise AsmError("use the seg/cr forms for those moves")
        size = dst.size
        if src.kind == 'reg' and src.size != size:
            raise AsmError("mov operand sizes must match")
        _check_no_high8_with_rex(ops)
        if size == 8:
            opc = 0x8A
        elif size == 16:
            opc = 0x8B
        else:
            opc = 0x8B
        prefix = _size_prefix(size)
        if src.kind == 'reg':
            rex = _rex(1 if size == 64 else 0,
                       _reg_rex_bit(dst), 0, _reg_rex_bit(src))
            return prefix + rex + bytes([opc]) \
                + _modrm(3, dst.code & 7, src.code & 7)
        modrm, rex_b, rex_x, _ = _encode_rm(dst.code & 7, src, size, 0)
        rex = _rex(1 if size == 64 else 0,
                   _reg_rex_bit(dst), rex_x, rex_b)
        return prefix + rex + bytes([opc]) + modrm
    if dst.kind == 'mem' and src.kind == 'reg':
        if src.rkind == 'seg' or src.rkind == 'cr':
            raise AsmError("use the seg/cr forms for those moves")
        size = src.size
        _check_no_high8_with_rex(ops)
        opc = 0x88 if size == 8 else 0x89
        prefix = _size_prefix(size)
        modrm, rex_b, rex_x, _ = _encode_rm(src.code & 7, dst, size, 0)
        rex = _rex(1 if size == 64 else 0,
                   _reg_rex_bit(src), rex_x, rex_b)
        return prefix + rex + bytes([opc]) + modrm
    raise AsmError("unsupported mov operand combination")


def _encode_push_pop(op, ops, ctx):
    if len(ops) != 1:
        raise AsmError(f"{op} takes one operand")
    a = ops[0]
    if op == 'push':
        if a.kind == 'reg':
            if a.rkind == 'seg':
                raise AsmError("push segment register is not supported in v1")
            rex = _rex(0, 0, 0, _reg_rex_bit(a))
            return rex + bytes([0x50 + (a.code & 7)])
        if a.kind == 'imm':
            if _fits_i8(a.value):
                return b'\x6a' + bytes([a.value & 0xFF])
            if _fits_i32(a.value):
                return b'\x68' + int(a.value).to_bytes(4, 'little', signed=True)
            raise AsmError(f"push immediate out of range: {a.value}")
        if a.kind == 'mem':
            modrm, rex_b, rex_x, _ = _encode_rm(6, a, 64, 0)
            return _rex(0, 0, rex_x, rex_b) + b'\xff' + modrm
    else:  # pop
        if a.kind == 'reg':
            if a.rkind == 'seg':
                raise AsmError("pop segment register is not supported in v1")
            rex = _rex(0, 0, 0, _reg_rex_bit(a))
            return rex + bytes([0x58 + (a.code & 7)])
        if a.kind == 'mem':
            modrm, rex_b, rex_x, _ = _encode_rm(0, a, 64, 0)
            return _rex(0, 0, rex_x, rex_b) + b'\x8f' + modrm
    raise AsmError(f"unsupported {op} operand")


def _encode_lea(op, ops, ctx):
    if len(ops) != 2 or ops[0].kind != 'reg' or ops[1].kind != 'mem':
        raise AsmError("lea takes a register and a memory operand")
    dst, src = ops[0], ops[1]
    if dst.size not in (32, 64):
        raise AsmError("lea destination must be 32 or 64-bit")
    modrm, rex_b, rex_x, _ = _encode_rm(dst.code & 7, src, dst.size, 0)
    rex = _rex(1 if dst.size == 64 else 0,
               _reg_rex_bit(dst), rex_x, rex_b)
    # Rip-relative label displacement.
    if src.rip and src.rip_label is not None:
        size = len(rex) + 1 + len(modrm) + 4
        disp = ctx.label_disp(src.rip_label, size)
        modrm = modrm + int(disp).to_bytes(4, 'little', signed=True)
        return rex + b'\x8d' + modrm
    if src.rip:
        modrm = modrm + int(src.disp).to_bytes(4, 'little', signed=True)
        return rex + b'\x8d' + modrm
    if src.base is None:
        modrm = modrm + int(src.disp).to_bytes(4, 'little', signed=True)
        return rex + b'\x8d' + modrm
    return rex + b'\x8d' + modrm


def _encode_movzx_movsx(op, ops, ctx):
    if len(ops) != 2 or ops[0].kind != 'reg' or ops[1].kind not in ('reg', 'mem'):
        raise AsmError(f"{op} takes a register and a register/memory operand")
    dst, src = ops[0], ops[1]
    if src.kind == 'reg' and src.size not in (8, 16):
        raise AsmError(f"{op} source must be 8 or 16-bit")
    if dst.size not in (16, 32, 64):
        raise AsmError(f"{op} destination must be 16, 32 or 64-bit")
    if op == 'movzx':
        opc = 0xB6 if (src.kind == 'reg' and src.size == 8) or True else 0xB7
        # Decide by source size: 8-bit source -> B6, 16-bit -> B7. For a
        # mem source the size is ambiguous; v1 requires a register-sized
        # hint via the destination? No — mem sources default to 8-bit
        # (movzx rax, byte [mem] is the common form). The user writes
        # movzx with a mem operand meaning 8-bit; for 16-bit mem they
        # must use a 16-bit register source form. Documented limitation.
        if src.kind == 'mem':
            opc = 0xB6
        elif src.size == 8:
            opc = 0xB6
        else:
            opc = 0xB7
    else:
        if src.kind == 'mem':
            opc = 0xBE
        elif src.size == 8:
            opc = 0xBE
        else:
            opc = 0xBF
    prefix = _size_prefix(dst.size)
    if src.kind == 'reg':
        rex = _rex(1 if dst.size == 64 else 0,
                   _reg_rex_bit(dst), 0, _reg_rex_bit(src))
        return prefix + rex + b'\x0f' + bytes([opc]) \
            + _modrm(3, dst.code & 7, src.code & 7)
    modrm, rex_b, rex_x, _ = _encode_rm(dst.code & 7, src, dst.size, 0)
    rex = _rex(1 if dst.size == 64 else 0,
               _reg_rex_bit(dst), rex_x, rex_b)
    return prefix + rex + b'\x0f' + bytes([opc]) + modrm


def _encode_xchg(op, ops, ctx):
    if len(ops) != 2 or ops[0].kind not in ('reg', 'mem') \
            or ops[1].kind not in ('reg', 'mem'):
        raise AsmError("xchg takes two register/memory operands")
    a, b = ops
    if a.kind == 'reg' and b.kind == 'reg':
        if a.size != b.size:
            raise AsmError("xchg operand sizes must match")
        size = a.size
        prefix = _size_prefix(size)
        opc = 0x86 if size == 8 else 0x87
        rex = _rex(1 if size == 64 else 0,
                   _reg_rex_bit(a), 0, _reg_rex_bit(b))
        return prefix + rex + bytes([opc]) \
            + _modrm(3, a.code & 7, b.code & 7)
    if a.kind == 'reg' and b.kind == 'mem':
        size = a.size
        prefix = _size_prefix(size)
        opc = 0x86 if size == 8 else 0x87
        modrm, rex_b, rex_x, _ = _encode_rm(a.code & 7, b, size, 0)
        rex = _rex(1 if size == 64 else 0,
                   _reg_rex_bit(a), rex_x, rex_b)
        return prefix + rex + bytes([opc]) + modrm
    if a.kind == 'mem' and b.kind == 'reg':
        size = b.size
        prefix = _size_prefix(size)
        opc = 0x86 if size == 8 else 0x87
        modrm, rex_b, rex_x, _ = _encode_rm(b.code & 7, a, size, 0)
        rex = _rex(1 if size == 64 else 0,
                   _reg_rex_bit(b), rex_x, rex_b)
        return prefix + rex + bytes([opc]) + modrm
    raise AsmError("xchg mem,mem is not a valid instruction")


def _encode_bswap(op, ops, ctx):
    if len(ops) != 1 or ops[0].kind != 'reg' or ops[0].size not in (32, 64):
        raise AsmError("bswap takes a 32 or 64-bit register")
    a = ops[0]
    rex = _rex(1 if a.size == 64 else 0, 0, 0, _reg_rex_bit(a))
    return rex + b'\x0f' + bytes([0xC8 + (a.code & 7)])


def _encode_alu(op, ops, ctx):
    """add/sub/adc/sbb/and/or/xor/cmp/test, plus inc/dec/neg/not/shifts."""
    if op in ('inc', 'dec', 'neg', 'not'):
        if len(ops) != 1 or ops[0].kind not in ('reg', 'mem'):
            raise AsmError(f"{op} takes one register/memory operand")
        a = ops[0]
        size = a.size if a.kind == 'reg' else 64  # mem defaults to 64-bit
        digit = {'inc': 0, 'dec': 1, 'not': 2, 'neg': 3}[op]
        prefix = _size_prefix(size)
        if a.kind == 'reg':
            rex = _rex(1 if size == 64 else 0, 0, 0, _reg_rex_bit(a))
            if size == 8:
                return rex + b'\xfe' + _modrm(3, digit, a.code & 7)
            return prefix + rex + b'\xff' + _modrm(3, digit, a.code & 7)
        modrm, rex_b, rex_x, _ = _encode_rm(digit, a, size, 0)
        rex = _rex(1 if size == 64 else 0, 0, rex_x, rex_b)
        if size == 8:
            return rex + b'\xfe' + modrm
        return prefix + rex + b'\xff' + modrm
    if op in ('shl', 'shr', 'sar'):
        if len(ops) != 2 or ops[0].kind not in ('reg', 'mem'):
            raise AsmError(f"{op} takes a register/memory operand and a count")
        a, cnt = ops[0], ops[1]
        size = a.size if a.kind == 'reg' else 64
        digit = {'shl': 4, 'shr': 5, 'sar': 7}[op]
        prefix = _size_prefix(size)
        if cnt.kind == 'reg' and cnt.name == 'cl':
            if a.kind == 'reg':
                rex = _rex(1 if size == 64 else 0, 0, 0, _reg_rex_bit(a))
                opc = 0xD2 if size == 8 else 0xD3
                return prefix + rex + bytes([opc]) \
                    + _modrm(3, digit, a.code & 7)
            modrm, rex_b, rex_x, _ = _encode_rm(digit, a, size, 0)
            rex = _rex(1 if size == 64 else 0, 0, rex_x, rex_b)
            opc = 0xD2 if size == 8 else 0xD3
            return prefix + rex + bytes([opc]) + modrm
        if cnt.kind == 'imm':
            if cnt.value == 1:
                if a.kind == 'reg':
                    rex = _rex(1 if size == 64 else 0, 0, 0, _reg_rex_bit(a))
                    opc = 0xD0 if size == 8 else 0xD1
                    return prefix + rex + bytes([opc]) \
                        + _modrm(3, digit, a.code & 7)
                modrm, rex_b, rex_x, _ = _encode_rm(digit, a, size, 0)
                rex = _rex(1 if size == 64 else 0, 0, rex_x, rex_b)
                opc = 0xD0 if size == 8 else 0xD1
                return prefix + rex + bytes([opc]) + modrm
            if not _fits_i8(cnt.value):
                raise AsmError(f"shift count out of range: {cnt.value}")
            if a.kind == 'reg':
                rex = _rex(1 if size == 64 else 0, 0, 0, _reg_rex_bit(a))
                opc = 0xC0 if size == 8 else 0xC1
                return prefix + rex + bytes([opc]) \
                    + _modrm(3, digit, a.code & 7) \
                    + bytes([cnt.value & 0xFF])
            modrm, rex_b, rex_x, _ = _encode_rm(digit, a, size, 0)
            rex = _rex(1 if size == 64 else 0, 0, rex_x, rex_b)
            opc = 0xC0 if size == 8 else 0xC1
            return prefix + rex + bytes([opc]) + modrm \
                + bytes([cnt.value & 0xFF])
        raise AsmError(f"{op} count must be 1, an imm8, or cl")
    # Two-operand ALU: add/sub/adc/sbb/and/or/xor/cmp, test.
    if len(ops) != 2:
        raise AsmError(f"{op} takes two operands")
    dst, src = ops
    if op == 'test':
        if dst.kind == 'reg' and src.kind == 'reg':
            if dst.size != src.size:
                raise AsmError("test operand sizes must match")
            size = dst.size
            prefix = _size_prefix(size)
            opc = 0x84 if size == 8 else 0x85
            rex = _rex(1 if size == 64 else 0,
                       _reg_rex_bit(dst), 0, _reg_rex_bit(src))
            return prefix + rex + bytes([opc]) \
                + _modrm(3, dst.code & 7, src.code & 7)
        if dst.kind == 'reg' and src.kind == 'imm':
            size = dst.size
            prefix = _size_prefix(size)
            if size == 8:
                rex = _rex(0, 0, 0, _reg_rex_bit(dst))
                if dst.code == 0 and dst.rkind == 'gpr':
                    return bytes([0xA8]) + bytes([src.value & 0xFF])
                return rex + b'\xf6' + _modrm(3, 0, dst.code & 7) \
                    + bytes([src.value & 0xFF])
            if dst.code == 0 and dst.rkind == 'gpr':
                rex = _rex(1 if size == 64 else 0, 0, 0, 0)
                return prefix + rex + bytes([0xA9]) \
                    + int(src.value).to_bytes(4, 'little', signed=True)
            rex = _rex(1 if size == 64 else 0, 0, 0, _reg_rex_bit(dst))
            return prefix + rex + b'\xf7' + _modrm(3, 0, dst.code & 7) \
                + int(src.value).to_bytes(4, 'little', signed=True)
        if dst.kind == 'mem' and src.kind == 'reg':
            size = src.size
            prefix = _size_prefix(size)
            opc = 0x84 if size == 8 else 0x85
            modrm, rex_b, rex_x, _ = _encode_rm(src.code & 7, dst, size, 0)
            rex = _rex(1 if size == 64 else 0,
                       _reg_rex_bit(src), rex_x, rex_b)
            return prefix + rex + bytes([opc]) + modrm
        raise AsmError("unsupported test operand combination")
    rm_opc, r_opc, digit, acc_opc = _alu_opcode(op)
    # reg/mem, reg
    if dst.kind in ('reg', 'mem') and src.kind == 'reg':
        size = src.size
        if dst.kind == 'reg' and dst.size != size:
            raise AsmError(f"{op} operand sizes must match")
        _check_no_high8_with_rex(ops)
        prefix = _size_prefix(size)
        opc = (rm_opc - 1) if size == 8 else rm_opc
        if dst.kind == 'reg':
            rex = _rex(1 if size == 64 else 0,
                       _reg_rex_bit(src), 0, _reg_rex_bit(dst))
            return prefix + rex + bytes([opc]) \
                + _modrm(3, src.code & 7, dst.code & 7)
        modrm, rex_b, rex_x, _ = _encode_rm(src.code & 7, dst, size, 0)
        rex = _rex(1 if size == 64 else 0,
                   _reg_rex_bit(src), rex_x, rex_b)
        return prefix + rex + bytes([opc]) + modrm
    # reg, reg/mem
    if dst.kind == 'reg' and src.kind in ('reg', 'mem'):
        size = dst.size
        if src.kind == 'reg' and src.size != size:
            raise AsmError(f"{op} operand sizes must match")
        _check_no_high8_with_rex(ops)
        prefix = _size_prefix(size)
        opc = (r_opc - 1) if size == 8 else r_opc
        if src.kind == 'reg':
            rex = _rex(1 if size == 64 else 0,
                       _reg_rex_bit(dst), 0, _reg_rex_bit(src))
            return prefix + rex + bytes([opc]) \
                + _modrm(3, dst.code & 7, src.code & 7)
        modrm, rex_b, rex_x, _ = _encode_rm(dst.code & 7, src, size, 0)
        rex = _rex(1 if size == 64 else 0,
                   _reg_rex_bit(dst), rex_x, rex_b)
        return prefix + rex + bytes([opc]) + modrm
    # reg/mem, imm
    if dst.kind in ('reg', 'mem') and src.kind == 'imm':
        size = dst.size if dst.kind == 'reg' else 32
        if dst.kind == 'reg':
            _check_no_high8_with_rex(ops)
        prefix = _size_prefix(size)
        v = src.value
        # Accumulator short form for rax/eax/ax/al.
        if dst.kind == 'reg' and dst.code == 0 and dst.rkind == 'gpr':
            if size == 8:
                return bytes([acc_opc - 1]) + bytes([v & 0xFF])
            rex = _rex(1 if size == 64 else 0, 0, 0, 0)
            if size == 16:
                return b'\x66' + bytes([acc_opc]) \
                    + int(v & 0xFFFF).to_bytes(2, 'little')
            return rex + bytes([acc_opc]) \
                + int(v).to_bytes(4, 'little', signed=True)
        if _fits_i8(v):
            if dst.kind == 'reg':
                rex = _rex(1 if size == 64 else 0, 0, 0, _reg_rex_bit(dst))
                opc = 0x80 if size == 8 else 0x83
                return prefix + rex + bytes([opc]) \
                    + _modrm(3, digit, dst.code & 7) \
                    + bytes([v & 0xFF])
            modrm, rex_b, rex_x, _ = _encode_rm(digit, dst, size, 0)
            rex = _rex(1 if size == 64 else 0, 0, rex_x, rex_b)
            opc = 0x80 if size == 8 else 0x83
            return prefix + rex + bytes([opc]) + modrm \
                + bytes([v & 0xFF])
        if not _fits_i32(v):
            raise AsmError(f"{op} immediate out of range: {v}")
        if dst.kind == 'reg':
            rex = _rex(1 if size == 64 else 0, 0, 0, _reg_rex_bit(dst))
            opc = 0x80 if size == 8 else 0x81
            if size == 8:
                return rex + bytes([opc]) + _modrm(3, digit, dst.code & 7) \
                    + bytes([v & 0xFF])
            return prefix + rex + bytes([opc]) \
                + _modrm(3, digit, dst.code & 7) \
                + int(v).to_bytes(4, 'little', signed=True)
        modrm, rex_b, rex_x, _ = _encode_rm(digit, dst, size, 0)
        rex = _rex(1 if size == 64 else 0, 0, rex_x, rex_b)
        opc = 0x80 if size == 8 else 0x81
        return prefix + rex + bytes([opc]) + modrm \
            + int(v).to_bytes(4, 'little', signed=True)
    raise AsmError(f"unsupported {op} operand combination")


_JCC = {
    'jo': 0x80, 'jno': 0x81, 'jb': 0x82, 'jc': 0x82, 'jnae': 0x82,
    'jae': 0x83, 'jnb': 0x83, 'jnc': 0x83,
    'je': 0x84, 'jz': 0x84, 'jne': 0x85, 'jnz': 0x85,
    'jbe': 0x86, 'jna': 0x86, 'ja': 0x87, 'jnbe': 0x87,
    'js': 0x88, 'jns': 0x89, 'jp': 0x8A, 'jpe': 0x8A,
    'jnp': 0x8B, 'jpo': 0x8B, 'jl': 0x8C, 'jnge': 0x8C,
    'jge': 0x8D, 'jnl': 0x8D, 'jle': 0x8E, 'jng': 0x8E,
    'jg': 0x8F, 'jnle': 0x8F,
}


def _encode_control(op, ops, ctx):
    if op == 'jmp':
        if len(ops) != 1:
            raise AsmError("jmp takes one operand")
        a = ops[0]
        if a.kind == 'label':
            disp = ctx.label_disp(a.name, 5)
            return b'\xe9' + int(disp).to_bytes(4, 'little', signed=True)
        if a.kind == 'reg':
            rex = _rex(0, 0, 0, _reg_rex_bit(a))
            return rex + b'\xff' + _modrm(3, 4, a.code & 7)
        if a.kind == 'mem':
            modrm, rex_b, rex_x, _ = _encode_rm(4, a, 64, 0)
            return _rex(0, 0, rex_x, rex_b) + b'\xff' + modrm
        raise AsmError("unsupported jmp operand")
    if op in _JCC:
        if len(ops) != 1 or ops[0].kind != 'label':
            raise AsmError(f"{op} takes a label operand")
        disp = ctx.label_disp(ops[0].name, 6)
        return b'\x0f' + bytes([_JCC[op]]) \
            + int(disp).to_bytes(4, 'little', signed=True)
    if op == 'call':
        if len(ops) != 1:
            raise AsmError("call takes one operand")
        a = ops[0]
        if a.kind == 'label':
            disp = ctx.label_disp(a.name, 5)
            return b'\xe8' + int(disp).to_bytes(4, 'little', signed=True)
        if a.kind == 'reg':
            rex = _rex(0, 0, 0, _reg_rex_bit(a))
            return rex + b'\xff' + _modrm(3, 2, a.code & 7)
        if a.kind == 'mem':
            modrm, rex_b, rex_x, _ = _encode_rm(2, a, 64, 0)
            return _rex(0, 0, rex_x, rex_b) + b'\xff' + modrm
        raise AsmError("unsupported call operand")
    if op == 'ret':
        if len(ops) == 0:
            return b'\xc3'
        if len(ops) == 1 and ops[0].kind == 'imm':
            if not (0 <= ops[0].value < 2**16):
                raise AsmError("ret immediate out of range")
            return b'\xc2' + int(ops[0].value).to_bytes(2, 'little')
        raise AsmError("ret takes no operand or an imm16")
    raise AsmError(f"unsupported control instruction: {op}")


def _encode_simple(op, ops, ctx):
    table = {
        'nop': b'\x90', 'cli': b'\xfa', 'sti': b'\xfb', 'hlt': b'\xf4',
        'cld': b'\xfc', 'std': b'\xfd', 'int3': b'\xcc',
        'rdtsc': b'\x0f\x31', 'cpuid': b'\x0f\xa2',
        'pause': b'\xf3\x90',
        'syscall': b'\x0f\x05', 'sysret': b'\x0f\x07',
        'sysenter': b'\x0f\x34', 'sysexit': b'\x0f\x35',
    }
    if op not in table:
        raise AsmError(f"unsupported instruction: {op}")
    if ops:
        raise AsmError(f"{op} takes no operands")
    return table[op]


def _encode_int(op, ops, ctx):
    if len(ops) != 1 or ops[0].kind != 'imm':
        raise AsmError("int takes an imm8 operand")
    if not (0 <= ops[0].value < 256):
        raise AsmError("int immediate out of range")
    return b'\xcd' + bytes([ops[0].value])


def _encode_inout(op, ops, ctx):
    if len(ops) != 2:
        raise AsmError(f"{op} takes two operands")
    a, b = ops
    if op == 'in':
        # in al/eax/ax, dx | in al/eax/ax, imm8
        if a.kind == 'reg' and a.rkind == 'gpr' and a.size in (8, 16, 32) \
                and a.code == 0:
            if b.kind == 'reg' and b.name == 'dx':
                if a.size == 8:
                    return b'\xec'
                if a.size == 16:
                    return b'\x66\xed'
                return b'\xed'
            if b.kind == 'imm' and 0 <= b.value < 256:
                if a.size == 8:
                    return b'\xe4' + bytes([b.value])
                if a.size == 16:
                    return b'\x66\xe5' + bytes([b.value])
                return b'\xe5' + bytes([b.value])
        raise AsmError("unsupported in operand combination")
    # out dx, al/eax/ax | out imm8, al/eax/ax
    if b.kind == 'reg' and b.rkind == 'gpr' and b.size in (8, 16, 32) \
            and b.code == 0:
        if a.kind == 'reg' and a.name == 'dx':
            if b.size == 8:
                return b'\xee'
            if b.size == 16:
                return b'\x66\xef'
            return b'\xef'
        if a.kind == 'imm' and 0 <= a.value < 256:
            if b.size == 8:
                return b'\xe6' + bytes([a.value])
            if b.size == 16:
                return b'\x66\xe7' + bytes([a.value])
            return b'\xe7' + bytes([a.value])
    raise AsmError("unsupported out operand combination")


def _encode_table(op, ops, ctx):
    """lgdt/lidt/sgdt/sidt/invlpg/ltr (memory operand, /digit)."""
    table = {
        'lgdt': (0x01, 2), 'lidt': (0x01, 3),
        'sgdt': (0x01, 0), 'sidt': (0x01, 1),
        'invlpg': (0x01, 7),
    }
    if op == 'ltr':
        if len(ops) != 1 or ops[0].kind not in ('reg', 'mem'):
            raise AsmError("ltr takes a register/memory operand")
        a = ops[0]
        if a.kind == 'reg':
            return b'\x0f\x00' + _modrm(3, 3, a.code & 7)
        modrm, rex_b, rex_x, _ = _encode_rm(3, a, 16, 0)
        return _rex(0, 0, rex_x, rex_b) + b'\x0f\x00' + modrm
    if op not in table:
        raise AsmError(f"unsupported instruction: {op}")
    if len(ops) != 1 or ops[0].kind != 'mem':
        raise AsmError(f"{op} takes a memory operand")
    opc2, digit = table[op]
    a = ops[0]
    modrm, rex_b, rex_x, _ = _encode_rm(digit, a, 16, 0)
    out = _rex(0, 0, rex_x, rex_b) + b'\x0f' + bytes([opc2]) + modrm
    if a.rip and a.rip_label is not None:
        size = len(out) + 4
        disp = ctx.label_disp(a.rip_label, size)
        return out + int(disp).to_bytes(4, 'little', signed=True)
    if a.rip:
        return out + int(a.disp).to_bytes(4, 'little', signed=True)
    if a.base is None:
        return out + int(a.disp).to_bytes(4, 'little', signed=True)
    return out


def encode_instruction(op, ops, ctx):
    """Encode one mnemonic instruction to bytes."""
    if op == 'mov':
        return _encode_mov(op, ops, ctx)
    if op in ('push', 'pop'):
        return _encode_push_pop(op, ops, ctx)
    if op == 'lea':
        return _encode_lea(op, ops, ctx)
    if op in ('movzx', 'movsx'):
        return _encode_movzx_movsx(op, ops, ctx)
    if op == 'xchg':
        return _encode_xchg(op, ops, ctx)
    if op == 'bswap':
        return _encode_bswap(op, ops, ctx)
    if op in ('add', 'sub', 'adc', 'sbb', 'and', 'or', 'xor', 'cmp', 'test',
              'inc', 'dec', 'neg', 'not', 'shl', 'shr', 'sar'):
        return _encode_alu(op, ops, ctx)
    if op == 'jmp' or op in _JCC or op == 'call' or op == 'ret':
        return _encode_control(op, ops, ctx)
    if op == 'int':
        return _encode_int(op, ops, ctx)
    if op in ('in', 'out'):
        return _encode_inout(op, ops, ctx)
    if op in ('lgdt', 'lidt', 'sgdt', 'sidt', 'invlpg', 'ltr'):
        return _encode_table(op, ops, ctx)
    if op in ('nop', 'cli', 'sti', 'hlt', 'cld', 'std', 'int3',
              'rdtsc', 'cpuid', 'pause', 'syscall', 'sysret',
              'sysenter', 'sysexit'):
        return _encode_simple(op, ops, ctx)
    raise AsmError(f"unknown mnemonic: {op!r}")


# ----------------------------------------------------------------------
# Block assembly: ordered ('raw', byte) / ('label', name) / ('mnem', line)
# items, each carrying a source line number for error messages.
# ----------------------------------------------------------------------

def assemble_asm_body(items):
    """Assemble an asm block body into bytes.

    items: list of (kind, value, lineno) where kind is 'raw' (value is a
    byte int), 'label' (value is the label name), or 'mnem' (value is
    the mnemonic line text).
    """
    # Pass 1: sizes and label offsets. Jumps/calls/jcc always use rel32,
    # so sizes never depend on label values.
    labels = {}
    parsed = []  # (kind, value, lineno, offset, size)
    offset = 0
    for kind, value, lineno in items:
        if kind == 'raw':
            parsed.append((kind, value, lineno, offset, 1))
            offset += 1
        elif kind == 'label':
            if value in labels:
                raise AsmError(
                    f"line {lineno}: duplicate label: {value!r}")
            labels[value] = offset
            parsed.append((kind, value, lineno, offset, 0))
        elif kind == 'mnem':
            try:
                op, ops = parse_mnemonic_line(value)
                # Size with placeholder labels (rel32 => fixed size).
                dummy = _Ctx(labels, offset, sizing=True)
                size = len(encode_instruction(op, ops, dummy))
            except AsmError as e:
                raise AsmError(f"line {lineno}: {e}")
            parsed.append((kind, (op, ops), lineno, offset, size))
            offset += size
        else:
            raise AsmError(f"line {lineno}: bad asm body item: {kind!r}")
    # Pass 2: emit.
    out = bytearray()
    for kind, value, lineno, ioff, size in parsed:
        if kind == 'raw':
            out.append(value)
        elif kind == 'label':
            continue
        else:
            op, ops = value
            try:
                ctx = _Ctx(labels, ioff)
                out += encode_instruction(op, ops, ctx)
            except AsmError as e:
                raise AsmError(f"line {lineno}: {e}")
    return bytes(out)
