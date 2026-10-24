"""Code generator for Orum v0.002: typed AST -> x86-64 machine code.

Calling/value conventions (all Orum-generated code):
- Every expression result arrives in RAX as a 64-bit word.
- Union values (`T | E`) arrive as (RAX=tag, RDX=payload); callees
  preserve RDX across `ret` so unions work as return values.
- Struct values are ADDRESSES (RAX points at the object's words).
- Strings are ADDRESSES of len-prefixed data ([len:8][bytes]); literals
  live in .rodata, interpolated strings on the heap.
- List values are ADDRESSES of a 24-byte heap header
  {len:8, cap:8, data:8}; the header never moves, so copied handles
  stay valid across growth.
- Locals live in an rbp-anchored frame; all struct temporaries are
  allocated with push/sub and reclaimed per-statement via
  `lea rsp, [rbp - frame_size]`.
- Calls follow the System V ABI (rdi, rsi, rdx, rcx, r8, r9).
- Runtime helpers preserve rbx/r12-r15 (callee-saved).

v0.002 codegen supports everything v0.001 did, plus: runtime string
interpolation, `list[T]` (construction, push, index, len, say),
a 1 MiB brk bump heap (`__alloc`), and method-call syntax for
`l.push(x)`. Everything else raises CodegenError.
"""
from __future__ import annotations

from . import ast_nodes as A
from . import objfile
from .typecheck import (
    INT_TYPES, SIGNED_TYPES, UNSIGNED_TYPES, is_int_like, is_bool, is_str,
    type_name,
)
from .x86 import (
    Emitter, RAX, RBX, RCX, RDX, RDI, RSI, RBP, RSP, R8, R9, R10, R11,
    R12, R13, R14, R15,
    CC_E, CC_NE, CC_B, CC_AE, CC_BE, CC_A, CC_L, CC_GE, CC_LE, CC_G,
    ARG_REGS,
)


class CodegenError(Exception):
    """Code generation failed (unsupported construct), with line info."""


class VarInfo:
    __slots__ = ("kind", "off", "type", "words", "is_param", "param_idx")

    def __init__(self, kind, off, type_, words, is_param=False, param_idx=0):
        self.kind = kind  # 'word' | 'struct_local' | 'struct_param' | 'union'
        self.off = off  # byte offset: word i lives at [rbp - off - 8*i]
        self.type = type_
        self.words = words
        self.is_param = is_param
        self.param_idx = param_idx


class StructLayout:
    def __init__(self, name):
        self.name = name
        self.fields = []  # (name, type, offset_words)
        self.size_words = 0


def align16(n: int) -> int:
    return (n + 15) & ~15


ELEM_SIZE = {
    "i8": 1, "u8": 1,
    "i16": 2, "u16": 2,
    "i32": 4, "u32": 4, "f32": 4,
    "i64": 8, "u64": 8, "int": 8, "f64": 8,
}


class Codegen:
    def __init__(self, prog: A.Program, freestanding: bool = False,
                 entry_symbol: str = "_start"):
        self.prog = prog
        # v0.006 freestanding: no syscalls, no brk heap, configurable
        # entry symbol. Hosted behavior is unchanged.
        self.freestanding = freestanding
        self.entry_symbol = entry_symbol
        self.e = Emitter()
        self.rodata = bytearray()
        self.rodata_map: dict[bytes, int] = {}
        self.rodata_refs: list[tuple[int, int]] = []  # (text_pos, rodata_off)
        # Writable .data segment: heap_ptr at +0, heap_end at +8.
        self.data = bytearray(16)
        self.data_refs: list[tuple[int, int]] = []  # (text_pos, data_off)
        # v0.006: (text_pos, symbol_name) pairs for extern references,
        # resolved via .rela.text against UND symbols in .symtab.
        self.sym_refs: list[tuple[int, str]] = []
        self.obj_relocs: list[tuple[int, int, int, str]] = []
        self.obj_symbols: list[tuple[str, int, int]] = []
        self.struct_layouts: dict[str, StructLayout] = {}
        self.fn_labels: dict[str, str] = {}
        self.label_n = 0
        # per-function state
        self.cur_vars: dict[str, VarInfo] = {}
        self.cur_frame_size = 0
        self.next_off = 8
        self.stack_depth = 0
        self.call_stack_padding = 0

    # -- helpers --------------------------------------------------------

    def err(self, line, msg):
        return CodegenError(f"{line}: codegen: {msg}")

    def emit_syscall(self):
        """Safety net: any syscall emission in freestanding mode is a
        loud compile error instead of a silent violation."""
        if self.freestanding:
            raise self.err("?", "internal: syscall emitted in "
                                "freestanding mode")
        self.e.syscall()

    def gen_halt_loop(self):
        """cli; hlt; jmp hlt — the freestanding answer to unrecoverable
        errors: stop the CPU, forever."""
        e = self.e
        e.code += b"\xfa"  # cli
        loop = self.new_label("halt")
        e.label(loop)
        e.code += b"\xf4"  # hlt
        e.jmp(loop)

    def sym_addr(self, reg: int, name: str):
        """movabs reg, <address of extern symbol `name`>; the ELF writer
        records a R_X86_64_64 relocation against the UND symbol."""
        pos = self.e.movabs_placeholder(reg)
        self.sym_refs.append((pos, name))

    def new_label(self, prefix: str) -> str:
        self.label_n += 1
        return f"{prefix}_{self.label_n}"

    def intern_str(self, data: bytes) -> int:
        """Store a len-prefixed string in .rodata; return its offset."""
        if data in self.rodata_map:
            return self.rodata_map[data]
        while len(self.rodata) % 8:
            self.rodata.append(0)
        off = len(self.rodata)
        self.rodata += len(data).to_bytes(8, "little")
        self.rodata += data
        while len(self.rodata) % 8:
            self.rodata.append(0)
        self.rodata_map[data] = off
        return off

    def rodata_addr(self, reg: int, data: bytes):
        off = self.intern_str(data)
        pos = self.e.movabs_placeholder(reg)
        self.rodata_refs.append((pos, off))
        if getattr(self, "object_mode", False):
            self.obj_relocs.append((pos, objfile.RELOC_RODATA64, off, ""))

    def rodata_bytes_addr(self, reg: int, data: bytes):
        """Address raw byte data in .rodata, without a string length prefix."""
        while len(self.rodata) % 8:
            self.rodata.append(0)
        off = len(self.rodata)
        self.rodata += data
        pos = self.e.movabs_placeholder(reg)
        self.rodata_refs.append((pos, off))
        if getattr(self, "object_mode", False):
            self.obj_relocs.append((pos, objfile.RELOC_RODATA64, off, ""))

    def data_addr(self, reg: int, doff: int):
        """movabs reg, <absolute address of .data + doff>; the ELF writer
        patches the placeholder. Heap globals: heap_ptr at +0, heap_end
        at +8."""
        pos = self.e.movabs_placeholder(reg)
        self.data_refs.append((pos, doff))
        if getattr(self, "object_mode", False):
            self.obj_relocs.append((pos, objfile.RELOC_DATA64, doff, ""))

    HEAP_PTR_OFF = 0
    HEAP_END_OFF = 8
    HEAP_SIZE = 1 << 20  # 1 MiB bump heap, reserved via brk at startup

    # -- types -----------------------------------------------------------

    def check_gen_type(self, t, line, where):
        """Reject types v0.002 codegen cannot lower."""
        if isinstance(t, A.NamedType):
            if t.name in INT_TYPES or t.name in ("bool", "str", "unit"):
                return
            if t.name in ("f32", "f64", "bytes"):
                raise self.err(line, f"{where}: '{t.name}' not supported "
                                    "in v0.002 codegen")
            if t.name in self.struct_layouts:
                return
            if t.name in ("list", "array"):
                raise self.err(line, f"{where}: '{t.name}' not supported "
                                    "in v0.002 codegen")
            # unknown names were already rejected by the typechecker
            return
        if isinstance(t, A.RawType):
            return
        if isinstance(t, A.FnType):
            # v0.011: function pointers are word-sized addresses
            return
        if isinstance(t, A.ArrayType):
            if t.size < 0:
                raise self.err(line, f"{where}: array size cannot be negative")
            if not (is_int_like(t.elem) or is_bool(t.elem)):
                raise self.err(line, f"{where}: array element type "
                                    f"{type_name(t.elem)} is not supported")
            return
        if isinstance(t, A.ListType):
            el = t.elem
            if not (is_int_like(el) or is_bool(el) or is_str(el)):
                raise self.err(
                    line, f"{where}: list[{type_name(el)}] element type "
                    "not supported in v0.002 (element must be "
                    "int/str/bool)")
            return
        if isinstance(t, A.UnionType):
            for o in t.options:
                self.check_gen_type(o, line, where)
            return
        raise self.err(line, f"{where}: type not supported in v0.002 codegen")

    def type_words(self, t) -> int:
        if isinstance(t, A.NamedType) and t.name in self.struct_layouts:
            return self.struct_layouts[t.name].size_words
        if isinstance(t, A.UnionType):
            return 2
        return 1

    def layout_struct(self, sdef: A.StructDef) -> StructLayout:
        layout = StructLayout(sdef.name)
        off = 0
        for f in sdef.fields:
            if isinstance(f.type, (A.ListType, A.ArrayType)):
                # v0.001 test pins this: list/array-typed struct fields
                # are still unsupported in v0.002 (a future version can
                # lay out the word-sized handle).
                raise self.err(getattr(sdef, "line", "?"),
                               f"struct field '{f.name}': list/array-typed "
                               "struct fields are not supported in v0.002")
            self.check_gen_type(f.type, getattr(sdef, "line", "?"),
                                f"struct field '{f.name}'")
            if isinstance(f.type, A.UnionType):
                raise self.err(getattr(sdef, "line", "?"),
                               "union-typed struct fields are not "
                               "supported in v0.001")
            layout.fields.append((f.name, f.type, off))
            off += self.type_words(f.type)
        layout.size_words = off
        return layout

    # -- top-level driver --------------------------------------------------

    def compile(self):
        for stmt in self.prog.statements:
            if isinstance(stmt, A.StructDef):
                self.struct_layouts[stmt.name] = self.layout_struct(stmt)
        for stmt in self.prog.statements:
            if isinstance(stmt, A.FnDef):
                self.fn_labels[stmt.name] = f"fn_{stmt.name}"
        self.gen_helpers()
        entry_off = len(self.e.code)
        object_mode = getattr(self, "object_mode", False)
        if object_mode:
            entry_fndef = self.entry_symbol in self.fn_labels
            top = [s for s in self.prog.statements
                   if not isinstance(s, (A.FnDef, A.StructDef))]
            if entry_fndef or top:
                self.e.label(self.entry_symbol)
                self.gen_heap_init()
                if entry_fndef:
                    self.e.jmp(self.fn_labels[self.entry_symbol])
                elif top:
                    self.gen_function_body(top, params=[], is_entry=True)
                else:
                    self.gen_function_body([], params=[], is_entry=True)
                self.obj_symbols.append(("", entry_off,
                                         objfile.SYMBOL_ENTRY_STUB))
        else:
            # Entry point first: top-level statements run in the entry
            # symbol (default _start). In freestanding mode the entry still
            # runs the entry file's top-level — Orum's rule that the file
            # is the program holds on bare metal too.
            # v0.008: a user-defined entry function replaces top-level code.
            self.e.label(self.entry_symbol)
            self.gen_heap_init()
            if self.entry_symbol in self.fn_labels:
                self.e.jmp(self.fn_labels[self.entry_symbol])
            else:
                top = [s for s in self.prog.statements
                       if not isinstance(s, (A.FnDef, A.StructDef))]
                self.gen_function_body(top, params=[], is_entry=True)
        for stmt in self.prog.statements:
            if isinstance(stmt, A.FnDef):
                self.e.label(self.fn_labels[stmt.name])
                params = [(p.name, p.type) for p in stmt.params]
                self.gen_function_body(stmt.body, params=params,
                                       is_entry=False)
        self.e.resolve()
        return (bytes(self.e.code), bytes(self.rodata), self.rodata_refs,
                entry_off, bytes(self.data), self.data_refs,
                list(self.sym_refs))

    def gen_heap_init(self):
        """Reserve a 1 MiB bump heap via brk(2) before user code runs.

        heap_ptr = brk(0); heap_end = heap_ptr + HEAP_SIZE; brk(heap_end).
        Dies with 'out of memory' if the kernel refuses the reservation.

        Freestanding: no brk. The heap bounds come from the extern
        symbols __orum_heap_start / __orum_heap_end, which the kernel
        (or its linker script) must define as the addresses of the
        first usable heap byte and one past the last. Both must be
        8-aligned. heap_ptr_slot = __orum_heap_start,
        heap_end_slot = __orum_heap_end, via R_X86_64_64 relocations.
        """
        if self.freestanding:
            e = self.e
            if getattr(self, "object_mode", False):
                self.obj_symbols.extend((
                    ("__orum_heap_start", objfile.UNDEF, 0),
                    ("__orum_heap_end", objfile.UNDEF, 0),
                ))
            # rcx = &heap_ptr slot; rax = __orum_heap_start
            self.data_addr(RCX, self.HEAP_PTR_OFF)
            self.sym_addr(RAX, "__orum_heap_start")
            if getattr(self, "object_mode", False):
                pos = self.sym_refs[-1][0]
                self.obj_relocs.append((pos, objfile.RELOC_SYMBOL64, 0,
                                        "__orum_heap_start"))
            e.mov_mr(RCX, 0, RAX)  # heap_ptr = __orum_heap_start
            self.sym_addr(RAX, "__orum_heap_end")
            if getattr(self, "object_mode", False):
                pos = self.sym_refs[-1][0]
                self.obj_relocs.append((pos, objfile.RELOC_SYMBOL64, 0,
                                        "__orum_heap_end"))
            e.mov_mr(RCX, self.HEAP_END_OFF, RAX)  # heap_end = ...
            return
        e = self.e
        e.mov_ri64(RAX, 12)  # sys_brk
        e.xor_rr(RDI, RDI)  # brk(0) -> current break
        self.emit_syscall()
        self.data_addr(RCX, self.HEAP_PTR_OFF)
        e.mov_mr(RCX, 0, RAX)  # heap_ptr = rax
        e.mov_rr(RDI, RAX)
        e.add_ri(RDI, self.HEAP_SIZE)
        e.mov_mr(RCX, self.HEAP_END_OFF, RDI)  # heap_end = rdi
        e.mov_ri64(RAX, 12)  # sys_brk
        self.emit_syscall()
        e.cmp_rr(RAX, RDI)
        l_ok = self.new_label("heap_ok")
        e.jcc(CC_AE, l_ok)  # new break >= requested -> ok
        self.rodata_addr(RDI, b"out of memory\n")
        self.emit_call("__die")
        e.label(l_ok)

    # -- function bodies ----------------------------------------------------

    def alloc_var(self, name, type_, info, ctx_line="?"):
        if name in info:
            return
        self.check_gen_type(type_, ctx_line, f"variable '{name}'")
        words = self.type_words(type_)
        if isinstance(type_, A.NamedType) and \
                type_.name in self.struct_layouts:
            kind = "struct_local"
            # Struct objects grow UP from their base: word i lives at
            # [base + 8*i], so the base (lowest address) is at
            # rbp - (next_off + 8*(words-1)).
            off = self.next_off + (words - 1) * 8
        elif isinstance(type_, A.UnionType):
            kind = "union"
            # Unions keep the down-growing convention: word 0 (tag) at
            # [rbp - off], word 1 (payload) at [rbp - off - 8].
            off = self.next_off
        else:
            kind = "word"
            off = self.next_off
        self.next_off += words * 8
        info[name] = VarInfo(kind, off, type_, words)

    def collect_vars(self, stmts, info):
        for s in stmts:
            line = getattr(s, "line", "?")
            if isinstance(s, (A.Binding, A.Share)):
                self.alloc_var(s.name, s.inferred, info, line)
            elif isinstance(s, A.Take):
                for n in s.names:
                    self.alloc_var(n, A.NamedType("int"), info, line)
            elif isinstance(s, A.For):
                if getattr(s, "for_kind", "int") == "list":
                    # v0.003: `for x in list` — x takes the element type
                    # (immutable per iteration); hidden vars hold the list
                    # address, the index, and the snapshotted length.
                    it = getattr(s.iter, "inferred", None)
                    elem = it.elem if isinstance(it, A.ListType) \
                        else A.NamedType("int")
                    self.alloc_var(s.var, elem, info, line)
                    hid_i = f"$foridx{self.label_n}"
                    hid_l = f"$forlist{self.label_n}"
                    hid_n = f"$forlen{self.label_n}"
                    self.label_n += 1
                    s.hidden_idx, s.hidden_list, s.hidden_len = \
                        hid_i, hid_l, hid_n
                    self.alloc_var(hid_i, A.NamedType("int"), info, line)
                    self.alloc_var(hid_l, it if isinstance(it, A.ListType)
                                   else A.NamedType("int"), info, line)
                    self.alloc_var(hid_n, A.NamedType("int"), info, line)
                    self.collect_vars(s.body, info)
                else:
                    self.alloc_var(s.var, A.NamedType("int"), info, line)
                    hid = f"$forlimit{self.label_n}"
                    self.label_n += 1
                    s.hidden_limit = hid
                    self.alloc_var(hid, A.NamedType("int"), info, line)
                    self.collect_vars(s.body, info)
            elif isinstance(s, A.If):
                self.collect_vars(s.then_body, info)
                self.collect_vars(s.else_body, info)
            elif isinstance(s, A.While):
                self.collect_vars(s.body, info)
            elif isinstance(s, A.Match):
                st = s.subject.inferred
                for arm in s.arms:
                    side = st.options[0] if arm.kind == "ok" else st.options[1]
                    self.alloc_var(arm.binding, side, info, line)
                for arm in s.arms:
                    self.collect_vars([arm.body], info)
            # FnDef/StructDef: handled separately; other statements: nothing

    def gen_function_body(self, stmts, params, is_entry):
        e = self.e
        info: dict[str, VarInfo] = {}
        self.next_off = 8
        for i, (pname, ptype) in enumerate(params):
            line = "?"
            self.check_gen_type(ptype, line, f"parameter '{pname}'")
            if i >= len(ARG_REGS):
                raise self.err(line, "v0.001: at most 6 parameters")
            # Every parameter occupies one frame word. Struct-typed
            # parameters hold the address of the caller's object.
            kind = ("struct_param"
                    if isinstance(ptype, A.NamedType)
                    and ptype.name in self.struct_layouts
                    else "word")
            off = self.next_off
            self.next_off += 8
            info[pname] = VarInfo(kind, off, ptype, 1,
                                  is_param=True, param_idx=i)
        self.collect_vars(stmts, info)
        self.cur_vars = info
        frame_size = align16(self.next_off - 8)
        self.cur_frame_size = frame_size
        self.stack_depth = 0
        e.push_r(RBP)
        e.mov_rr(RBP, RSP)
        if frame_size:
            e.sub_ri(RSP, frame_size)
        # Spill register arguments (System V) into their frame slots.
        for pname, _ in params:
            pinfo = info[pname]
            e.mov_mr(RBP, -pinfo.off, ARG_REGS[pinfo.param_idx])
        self.gen_block(stmts)
        if is_entry:
            if self.freestanding:
                # Bare metal: the entry never returns. Halt the CPU.
                self.gen_halt_loop()
            else:
                # _start falls through to exit(0)
                e.mov_ri64(RAX, 60)
                e.xor_rr(RDI, RDI)
                self.emit_syscall()
        else:
            e.xor_rr(RAX, RAX)
            e.mov_rr(RSP, RBP)
            e.pop_r(RBP)
            e.ret()

    def push_temp(self, reg):
        self.e.push_r(reg)
        self.stack_depth += 8

    def pop_temp(self, reg):
        self.e.pop_r(reg)
        self.stack_depth -= 8

    def _align_call_stack(self, depth):
        """Align RSP to 16 bytes before a System V call."""
        padding = depth & 15
        if padding:
            self.e.sub_ri(RSP, padding)
        return padding

    def _restore_call_stack(self, padding):
        if padding:
            self.e.add_ri(RSP, padding)

    def emit_call(self, label):
        padding = self._align_call_stack(self.stack_depth)
        self.e.call(label)
        self._restore_call_stack(padding)

    def gen_block(self, stmts):
        for s in stmts:
            self.gen_stmt(s)
            if not isinstance(s, A.Return):
                # Reclaim any expression temporaries (struct temps).
                self.e.lea(RSP, RBP, -self.cur_frame_size)
            self.stack_depth = 0

    # -- statements ----------------------------------------------------------

    def gen_stmt(self, node):
        if isinstance(node, A.Binding):
            self.gen_binding(node)
        elif isinstance(node, A.Share):
            # Top-level shares initialize immutable module bindings.
            binding = A.Binding(node.name, None, node.value, False)
            binding.inferred = node.inferred
            self.gen_binding(binding)
        elif isinstance(node, A.Take):
            self.gen_take(node)
        elif isinstance(node, A.Assign):
            self.gen_assign(node)
        elif isinstance(node, A.If):
            self.gen_if(node)
        elif isinstance(node, A.While):
            self.gen_while(node)
        elif isinstance(node, A.For):
            self.gen_for(node)
        elif isinstance(node, A.Match):
            self.gen_match(node)
        elif isinstance(node, A.Return):
            self.gen_return(node)
        elif isinstance(node, A.Crash):
            self.gen_crash(node)
        elif isinstance(node, A.AsmBlock):
            self.gen_asm_block(node)
        elif isinstance(node, A.ExprStmt):
            self.gen_expr(node.expr)
        elif isinstance(node, (A.FnDef, A.StructDef)):
            pass  # emitted separately / no code
        else:
            raise self.err(getattr(node, "line", "?"),
                           f"unsupported statement {type(node).__name__}")

    def store_word_to_var(self, info: VarInfo, src_reg: int = RAX):
        # Every variable (locals and spilled params alike) lives at
        # [rbp - off]; struct params just happen to hold an address.
        self.e.mov_mr(RBP, -info.off, src_reg)

    def gen_binding(self, node: A.Binding):
        e = self.e
        info = self.cur_vars[node.name]
        self.gen_expr(node.value)
        t = node.inferred
        if info.kind == "struct_local":
            # rax = source address; deep-copy words into the local object
            n = info.words
            e.mov_rr(RDI, RAX)
            e.lea(RSI, RBP, -info.off)
            for i in range(n):
                e.mov_rm(RCX, RDI, i * 8)
                e.mov_mr(RSI, i * 8, RCX)
        elif info.kind == "union":
            # (rax=tag, rdx=payload)
            e.mov_mr(RBP, -info.off, RAX)
            e.mov_mr(RBP, -(info.off + 8), RDX)
        else:
            self.store_word_to_var(info)

    def gen_take(self, node: A.Take):
        # v0.001: no module loading; taken names are zero-initialized ints.
        e = self.e
        e.xor_rr(RAX, RAX)
        for n in node.names:
            self.store_word_to_var(self.cur_vars[n])

    def gen_assign(self, node: A.Assign):
        e = self.e
        tgt = node.target
        if isinstance(tgt, A.Var):
            info = self.cur_vars[tgt.name]
            self.gen_expr(node.value)
            if info.kind == "struct_local":
                n = info.words
                e.mov_rr(RDI, RAX)
                e.lea(RSI, RBP, -info.off)
                for i in range(n):
                    e.mov_rm(RCX, RDI, i * 8)
                    e.mov_mr(RSI, i * 8, RCX)
            elif info.kind == "struct_param":
                # copy into the object the param points to (address is
                # in the param's word slot)
                n = self.struct_layouts[info.type.name].size_words
                e.mov_rr(RDI, RAX)
                e.mov_rm(RSI, RBP, -info.off)
                for i in range(n):
                    e.mov_rm(RCX, RDI, i * 8)
                    e.mov_mr(RSI, i * 8, RCX)
            elif info.kind == "union":
                e.mov_mr(RBP, -info.off, RAX)
                e.mov_mr(RBP, -(info.off + 8), RDX)
            else:
                self.store_word_to_var(info)
            return
        # Field / index assignment: evaluate value first (calls may
        # clobber address registers), then compute the target address.
        self.gen_expr(node.value)
        e.push_r(RAX)
        if isinstance(tgt, A.FieldAccess):
            self.gen_expr(tgt.obj)  # rax = struct address
            e.mov_rr(RDI, RAX)
            layout = self.struct_layouts[tgt.obj.inferred.name]
            for fname, ftype, foff in layout.fields:
                if fname == tgt.field:
                    break
            else:
                raise self.err(getattr(node, "line", "?"), "internal error")
            if foff:
                e.add_ri(RDI, foff * 8)
            e.pop_r(RCX)  # value (or source address)
            if isinstance(ftype, A.NamedType) and \
                    ftype.name in self.struct_layouts:
                n = self.struct_layouts[ftype.name].size_words
                for i in range(n):
                    e.mov_rm(RDX, RCX, i * 8)
                    e.mov_mr(RDI, i * 8, RDX)
            else:
                e.mov_mr(RDI, 0, RCX)
        elif isinstance(tgt, A.Index):
            if isinstance(tgt.obj.inferred, A.ListType):
                self.gen_list_index_assign(node, tgt)
                return
            # VOLATILE (v0.004): this is a `raw *T` store. Exactly one
            # store is emitted per source access — never elided, merged,
            # or reordered. node.volatile was stamped by the typechecker.
            self.gen_expr(tgt.obj)
            e.push_r(RAX)  # base
            self.gen_expr(tgt.index)
            size, _ = self.elem_info(tgt.obj.inferred.inner)
            if size != 1:
                e.shl_ri(RAX, {2: 1, 4: 2, 8: 3}[size])
            e.mov_rr(RCX, RAX)
            e.pop_r(RDI)  # base
            e.add_rr(RDI, RCX)
            e.pop_r(RCX)  # value
            e.store_sized(RDI, 0, size, RCX)
        else:
            raise self.err(getattr(node, "line", "?"),
                           "cannot assign to this expression")

    def gen_if(self, node: A.If):
        e = self.e
        self.gen_expr(node.cond)
        e.test_rr(RAX, RAX)
        else_l = self.new_label("else")
        end_l = self.new_label("endif")
        e.jcc(CC_E, else_l)
        self.gen_block(node.then_body)
        e.jmp(end_l)
        e.label(else_l)
        self.gen_block(node.else_body)
        e.label(end_l)

    def gen_while(self, node: A.While):
        e = self.e
        loop_l = self.new_label("loop")
        end_l = self.new_label("wend")
        e.label(loop_l)
        self.gen_expr(node.cond)
        e.test_rr(RAX, RAX)
        e.jcc(CC_E, end_l)
        self.gen_block(node.body)
        e.jmp(loop_l)
        e.label(end_l)

    def gen_for(self, node: A.For):
        if getattr(node, "for_kind", "int") == "list":
            self.gen_for_list(node)
            return
        e = self.e
        var_info = self.cur_vars[node.var]
        lim_info = self.cur_vars[node.hidden_limit]
        self.gen_expr(node.iter)
        e.mov_mr(RBP, -lim_info.off, RAX)
        e.xor_rr(RAX, RAX)
        e.mov_mr(RBP, -var_info.off, RAX)
        loop_l = self.new_label("forloop")
        end_l = self.new_label("forend")
        e.label(loop_l)
        e.mov_rm(RAX, RBP, -var_info.off)
        e.cmp_rm(RAX, RBP, -lim_info.off)
        e.jcc(CC_GE, end_l)
        self.gen_block(node.body)
        e.mov_rm(RAX, RBP, -var_info.off)
        e.add_ri(RAX, 1)
        e.mov_mr(RBP, -var_info.off, RAX)
        e.jmp(loop_l)
        e.label(end_l)

    def gen_for_list(self, node: A.For):
        """`for x in list` — x is a fresh immutable word per iteration.

        The list address, index, and a snapshot of header.len live in
        hidden frame slots. Length is snapshotted so the loop always
        terminates; the header itself never moves, so the address stays
        valid even if the buffer is reallocated elsewhere. The data
        pointer is re-read each iteration (cheap, and robust).
        """
        e = self.e
        var_info = self.cur_vars[node.var]
        idx_info = self.cur_vars[node.hidden_idx]
        list_info = self.cur_vars[node.hidden_list]
        len_info = self.cur_vars[node.hidden_len]
        self.gen_expr(node.iter)  # rax = list address
        e.mov_mr(RBP, -list_info.off, RAX)
        e.mov_rm(RDX, RAX, 0)  # rdx = header.len (snapshot)
        e.mov_mr(RBP, -len_info.off, RDX)
        e.xor_rr(RAX, RAX)
        e.mov_mr(RBP, -idx_info.off, RAX)  # idx = 0
        loop_l = self.new_label("forlist")
        end_l = self.new_label("forlistend")
        e.label(loop_l)
        e.mov_rm(RAX, RBP, -idx_info.off)
        e.cmp_rm(RAX, RBP, -len_info.off)
        e.jcc(CC_GE, end_l)
        e.mov_rm(RSI, RBP, -list_info.off)  # rsi = list
        e.mov_rm(RSI, RSI, 16)  # rsi = data
        e.mov_rm(RCX, RBP, -idx_info.off)  # rcx = idx
        e.shl_ri(RCX, 3)
        e.add_rr(RCX, RSI)  # rcx = &data[idx]
        e.mov_rm(RAX, RCX, 0)  # rax = element
        e.mov_mr(RBP, -var_info.off, RAX)
        self.gen_block(node.body)
        e.mov_rm(RAX, RBP, -idx_info.off)
        e.add_ri(RAX, 1)
        e.mov_mr(RBP, -idx_info.off, RAX)
        e.jmp(loop_l)
        e.label(end_l)

    def gen_match(self, node: A.Match):
        e = self.e
        self.gen_expr(node.subject)  # rax=tag, rdx=payload
        ok_arm = next(a for a in node.arms if a.kind == "ok")
        err_arm = next(a for a in node.arms if a.kind == "err")
        err_l = self.new_label("matcherr")
        end_l = self.new_label("matchend")
        e.test_rr(RAX, RAX)
        e.jcc(CC_NE, err_l)
        e.mov_mr(RBP, -self.cur_vars[ok_arm.binding].off, RDX)
        self.gen_stmt(ok_arm.body)
        if not isinstance(ok_arm.body, A.Return):
            e.lea(RSP, RBP, -self.cur_frame_size)
        e.jmp(end_l)
        e.label(err_l)
        e.mov_mr(RBP, -self.cur_vars[err_arm.binding].off, RDX)
        self.gen_stmt(err_arm.body)
        if not isinstance(err_arm.body, A.Return):
            e.lea(RSP, RBP, -self.cur_frame_size)
        e.label(end_l)

    def gen_return(self, node: A.Return):
        e = self.e
        if node.value is not None:
            self.gen_expr(node.value)
            # union values: (rax, rdx) both survive the epilogue below
        else:
            e.xor_rr(RAX, RAX)
        e.mov_rr(RSP, RBP)
        e.pop_r(RBP)
        e.ret()

    def gen_crash(self, node: A.Crash):
        if self.freestanding:
            # No syscalls on bare metal: the message can't be printed.
            # (It is still typechecked.) Halt the CPU, forever.
            self.gen_halt_loop()
            return
        e = self.e
        self.gen_expr(node.message)  # rax = str address
        e.mov_rm(RDX, RAX, 0)  # len
        e.lea(RSI, RAX, 8)  # bytes
        e.mov_ri64(RAX, 1)  # sys_write
        e.mov_ri64(RDI, 2)  # stderr
        self.emit_syscall()
        e.mov_ri64(RAX, 60)  # sys_exit
        e.mov_ri64(RDI, 1)
        self.emit_syscall()

    def gen_asm_block(self, node: A.AsmBlock):
        """Inline `asm ... end`.

        Emission order is ALWAYS: in-moves, raw bytes verbatim,
        out-moves — regardless of source order. The block may trash any
        register except rsp/rbp; Orum values cross the boundary only via
        declared in/out (at statement level no Orum value is live in a
        register, so this is sound).
        """
        e = self.e
        for d in node.ins:
            info = self.cur_vars[d.var]
            if d.reg_bits == 64:
                e.mov_rm(d.reg_num, RBP, -info.off)
            else:
                # 32-bit load: low 32 bits, zero-extended to 64
                e.mov_rm32(d.reg_num, RBP, -info.off)
        if node.raw_bytes:
            e.code += bytes(node.raw_bytes)
        for d in node.outs:
            info = self.cur_vars[d.var]
            if d.reg_bits == 32:
                # mov r32,r32 zero-extends the full 64-bit register,
                # then store the whole word to the variable slot.
                e.mov_rr32(d.reg_num, d.reg_num)
            e.mov_mr(RBP, -info.off, d.reg_num)

    # -- expressions (result in rax; unions also in rdx) ---------------------

    def gen_expr(self, node):
        e = self.e
        if isinstance(node, A.IntLit):
            e.mov_ri64(RAX, node.value)
        elif isinstance(node, A.StrLit):
            self.rodata_addr(RAX, node.value.encode())
        elif isinstance(node, A.InterpString):
            self.gen_interp(node)
        elif isinstance(node, A.Var):
            self.gen_var_load(node)
        elif isinstance(node, A.BinOp):
            self.gen_binop(node)
        elif isinstance(node, A.UnaryOp):
            self.gen_expr(node.operand)
            if node.op == "~":
                e.not_r(RAX)  # v0.004: bitwise not
            else:
                e.neg_r(RAX)
        elif isinstance(node, A.Call):
            self.gen_call(node)
        elif isinstance(node, A.ListNew):
            self.gen_list_new(node)
        elif isinstance(node, A.ArrayLit):
            # Array literal: emit bytes to rodata, return address in RAX.
            # v0.010: only integer literals supported (for font data, etc.)
            # Each element must fit in a byte (u8).
            data = bytearray()
            for elem in node.elems:
                if not isinstance(elem, A.IntLit):
                    raise self.err(getattr(node, "line", "?"),
                                   "array literal elements must be integer literals")
                v = elem.value
                if not 0 <= v <= 255:
                    raise self.err(getattr(node, "line", "?"),
                                   f"array literal byte out of range: {v}")
                data.append(v)
            self.rodata_bytes_addr(RAX, bytes(data))
        elif isinstance(node, A.StructLit):
            self.gen_struct_lit(node)
        elif isinstance(node, A.FieldAccess):
            self.gen_field_access(node)
        elif isinstance(node, A.Index):
            self.gen_index(node)
        else:
            raise self.err(getattr(node, "line", "?"),
                           f"unsupported expression {type(node).__name__}")

    def gen_var_load(self, node: A.Var):
        e = self.e
        info = self.cur_vars[node.name]
        if info.kind == "struct_local":
            e.lea(RAX, RBP, -info.off)
        elif info.kind == "union":
            e.mov_rm(RAX, RBP, -info.off)  # tag
            e.mov_rm(RDX, RBP, -(info.off + 8))  # payload
        else:
            # word locals, word params, and struct params (whose slot
            # holds the object's address) all load one word here.
            e.mov_rm(RAX, RBP, -info.off)

    def gen_binop(self, node: A.BinOp):
        e = self.e
        self.gen_expr(node.left)
        self.push_temp(RAX)
        self.gen_expr(node.right)
        e.mov_rr(RCX, RAX)
        self.pop_temp(RAX)

        op = node.op
        signed = getattr(node, "signed", True)
        if getattr(node, "string_comparison", False):
            e.mov_rr(RDI, RAX)
            e.mov_rr(RSI, RCX)
            self.emit_call("__str_eq")
            if op == "!=":
                e.test_rr(RAX, RAX)
                e.setcc_bool(CC_E, RAX)
            return
        if op == "+":
            e.add_rr(RAX, RCX)
        elif op == "-":
            e.sub_rr(RAX, RCX)
        elif op == "*":
            e.imul_rr(RAX, RCX)
        elif op in ("/", "%"):
            if signed:
                e.cqo()
                e.idiv_r(RCX)
            else:
                e.xor_edx()
                e.div_r(RCX)
            if op == "%":
                e.mov_rr(RAX, RDX)
        elif op == "&":
            # v0.004: bitwise and — RAX = left & right
            e.and_rr(RAX, RCX)
        elif op == "|":
            # v0.004: bitwise or — RAX = left | right
            e.or_rr(RAX, RCX)
        elif op == "^":
            # v0.004: bitwise xor — RAX = left ^ right
            e.xor_rr(RAX, RCX)
        elif op == "<<":
            # v0.004: shift left — count is in RCX, x86 shifts r64 by CL.
            # Hardware masks the count to 6 bits; shl is the same for
            # signed and unsigned.
            e.shl_rcl(RAX)
        elif op == ">>":
            # v0.004: shift right — arithmetic (sar) if signed, logical
            # (shr) if unsigned, following node.signed from typecheck.
            if signed:
                e.sar_rcl(RAX)
            else:
                e.shr_rcl(RAX)
        else:
            e.cmp_rr(RAX, RCX)
            e.setcc_bool(self.cmp_cc(op, signed), RAX)

    @staticmethod
    def cmp_cc(op: str, signed: bool) -> int:
        if op == "==":
            return CC_E
        if op == "!=":
            return CC_NE
        if signed:
            return {"<": CC_L, "<=": CC_LE, ">": CC_G, ">=": CC_GE}[op]
        return {"<": CC_B, "<=": CC_BE, ">": CC_A, ">=": CC_AE}[op]

    def gen_call(self, node: A.Call):
        e = self.e
        builtin = getattr(node, "builtin", None)
        if builtin == "say":
            self.gen_say(node)
            return
        if builtin == "exit":
            self.gen_expr(node.args[0])
            e.mov_rr(RDI, RAX)
            e.mov_ri64(RAX, 60)
            self.emit_syscall()
            return
        if builtin == "ok":
            self.gen_expr(node.args[0])
            e.mov_rr(RDX, RAX)  # payload
            e.xor_rr(RAX, RAX)  # tag 0
            return
        if builtin == "err":
            self.gen_expr(node.args[0])
            e.mov_rr(RDX, RAX)  # payload
            e.mov_ri32(RAX, 1)  # tag 1
            return
        if builtin == "len":
            self.gen_expr(node.args[0])  # rax = list address
            e.mov_rm(RAX, RAX, 0)  # header.len
            return
        if builtin == "rawptr":
            # v0.004: `rawptr(addr)` — the address IS the pointer at
            # runtime (raw pointers are just addresses). Evaluate the
            # address expression; it stays in rax as the pointer value.
            self.gen_expr(node.args[0])
            return
        if builtin == "addr":
            # v0.009: `addr(var)` — LEA of a local's stack slot.
            info = self.cur_vars[node.addr_var]
            self.e.lea(RAX, RBP, -info.off)
            return
        if builtin == "fn_ptr_call":
            # Function pointers use the same System V ABI as direct Orum calls.
            if len(node.args) > len(ARG_REGS):
                raise self.err(getattr(node, "line", "?"),
                               "function pointer call supports max 6 args")
            for arg in node.args:
                self.gen_expr(arg)
                self.push_temp(RAX)
            for index in range(len(node.args) - 1, -1, -1):
                self.pop_temp(ARG_REGS[index])
            info = self.cur_vars[node.fn_ptr_name]
            e.mov_rm(R11, RBP, -info.off)
            padding = self._align_call_stack(self.stack_depth)
            e.call_reg(R11)
            self._restore_call_stack(padding)
            return
        if builtin == "ext_call":
            # v0.013 phase 2: call to external (undefined) function.
            # Emit relocation for the linker.
            # Uses System V for now (TODO: support UEFI convention)
            for arg in node.args:
                self.gen_expr(arg)
                self.push_temp(RAX)
            for i in range(len(node.args) - 1, -1, -1):
                self.pop_temp(ARG_REGS[i])
            pos = len(e.code)
            e.code += b"\xe8\x00\x00\x00\x00"  # call rel32 (placeholder)
            self.obj_relocs.append((pos + 1, objfile.RELOC_CALL, 0,
                                    node.ext_name))
            return
        if builtin == "list_push":
            # node.func is FieldAccess(receiver, 'push')
            self.gen_expr(node.func.obj)  # rax = list address
            e.push_r(RAX)
            self.gen_expr(node.args[0])  # rax = value
            e.mov_rr(RSI, RAX)
            e.pop_r(RDI)
            self.emit_call("__list_push")
            e.xor_rr(RAX, RAX)  # push returns unit
            return
        fname = node.func.name
        # v0.013 phase 2: allow undefined functions (external linkage).
        # If not defined in this module, emit a relocation.
        if fname not in self.fn_labels:
            if not getattr(self, 'object_mode', False):
                raise self.err(getattr(node, "line", "?"),
                               f"call to unknown function '{fname}'")
            # Object mode: record as external relocation
            # The call instruction's rel32 will be patched by the linker.
            for arg in node.args:
                self.gen_expr(arg)
                self.push_temp(RAX)
            for i in range(len(node.args) - 1, -1, -1):
                self.pop_temp(ARG_REGS[i])
            # Emit call with placeholder, record relocation
            pos = len(e.code)
            e.code += b"\xe8\x00\x00\x00\x00"  # call rel32 (placeholder)
            # The rel32 is at pos+1, needs to be patched to target - (pos+5)
            self.obj_relocs.append((pos + 1, objfile.RELOC_CALL, 0, fname))
            return
        if len(node.args) > len(ARG_REGS):
            raise self.err(getattr(node, "line", "?"),
                           "v0.001: at most 6 call arguments")
        original_depth = getattr(self, "stack_depth", 0)
        for arg in node.args:
            self.gen_expr(arg)
            e.push_r(RAX)
            self.stack_depth += 8
        for i in range(len(node.args) - 1, -1, -1):
            e.pop_r(ARG_REGS[i])
            self.stack_depth -= 8
        padding = self._align_call_stack(original_depth)
        e.call(self.fn_labels[fname])
        self._restore_call_stack(padding)
        # result in rax (and rdx for union returns)

    def gen_say(self, node: A.Call):
        e = self.e
        arg = node.args[0]
        at = arg.inferred
        self.gen_expr(arg)
        e.mov_rr(RDI, RAX)
        if isinstance(at, A.ListType):
            # element kind for __say_list: 0 = int, 1 = str, 2 = bool
            el = at.elem
            kind = 0 if is_int_like(el) else (1 if is_str(el) else 2)
            e.mov_ri64(RSI, kind)
            self.emit_call("__say_list")
        elif is_int_like(at):
            self.emit_call("__say_int")
        elif is_str(at):
            self.emit_call("__say_str")
        elif is_bool(at):
            self.emit_call("__say_bool")
        else:
            raise self.err(getattr(node, "line", "?"),
                           "say: unsupported type")

...[complete remainder of Orum/orum/codegen.py continues exactly from the local commit]