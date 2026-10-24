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
- Calls follow the System V ABI (rdi, rsi, rdx, rcx, n8, r9).
- Runtime helpers preserve rbx/r12-r15 (callee-saved).

v0.002 codegen supports everything v0.001 did, plus: runtime string
interpolation, `list[T]` (construction, push, index, len, say),
a 1 MiB brk bump heap (`__alloc`), and method-call syntax for
`l.push(x)`. Everything else raises CodegenError.
