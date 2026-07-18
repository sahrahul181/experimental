//! x86-64 managed-entry and cooperative slow-resolve machine-code shims.

const std = @import("std");
const builtin = @import("builtin");
const code_buffer = @import("code_buffer");
const frontend_interpreter = @import("interpreter");
const jit_memory = @import("jit_memory");
const optimizer = @import("optimizer");
const runtime_gc = @import("runtime_gc");
const runtime_heap = @import("runtime_heap");
const runtime_jit = @import("runtime_jit");
const runtime_code_manager = @import("runtime_code_manager");
const runtime_deopt = @import("runtime_deopt");
const runtime_stack_map = @import("runtime_stack_map");
const runtime_thread_registry = @import("runtime_thread_registry");
const runtime_value = @import("runtime_value");
const x64_encoder = @import("x64_register_encoder");
const Instruction = @import("instructions").Instruction;

pub const Error = code_buffer.Error || error{
    UnsupportedArchitecture,
};

const rax: u4 = 0;
const rcx: u4 = 1;
const rdx: u4 = 2;
const rsi: u4 = 6;
const rdi: u4 = 7;
const r8: u4 = 8;
const r9: u4 = 9;
const r10: u4 = 10;
const r11: u4 = 11;
const r12: u4 = 12;
const r13: u4 = 13;
const r14: u4 = 14;
const r15: u4 = 15;

pub const CallFrame = extern struct {
    image: runtime_jit.RegisterImage,
    target: usize,
    /// Stable interpreter/deoptimization entry used when `method_id` has no
    /// current compiled version. Zero selects the runtime's fail-closed stub.
    fallback_target: usize = 0,
    /// Owner-confined runtime_deopt.Request used by the managed fallback ABI.
    deopt_request: usize = 0,
    /// `unmanaged_method_id` preserves the direct-address fast path.
    method_id: u32 = runtime_jit.unmanaged_method_id,
    reserved: u32 = 0,
    gp_args: [6]usize = @splat(0),
};

pub const OsrCallFrame = extern struct {
    image: runtime_jit.RegisterImage,
    target: usize,
    gp: [16]u64 = @splat(0),
};

pub const EncodedShims = struct {
    allocator: std.mem.Allocator,
    entry: []u8,
    osr_entry: []u8,
    slow_helper: []u8,
    deopt_helper: []u8,
    bounds_exception_helper: []u8,
    satb_pre_write_helper: []u8,
    card_mark_helper: []u8,
    card_mark_repeat_helper: []u8,
    static_root_post_write_helper: []u8,

    pub fn deinit(self: *EncodedShims) void {
        self.allocator.free(self.static_root_post_write_helper);
        self.allocator.free(self.card_mark_repeat_helper);
        self.allocator.free(self.card_mark_helper);
        self.allocator.free(self.satb_pre_write_helper);
        self.allocator.free(self.bounds_exception_helper);
        self.allocator.free(self.slow_helper);
        self.allocator.free(self.deopt_helper);
        self.allocator.free(self.entry);
        self.allocator.free(self.osr_entry);
        self.* = undefined;
    }
};

fn emitRex(buffer: *code_buffer.Buffer, w: bool, reg: u4, rm: u4) Error!void {
    var rex: u8 = 0x40;
    if (w) rex |= 0x08;
    if ((reg & 8) != 0) rex |= 0x04;
    if ((rm & 8) != 0) rex |= 0x01;
    if (rex != 0x40) try buffer.emitU8(rex);
}

fn emitModRm(buffer: *code_buffer.Buffer, mode: u2, reg: u4, rm: u4) Error!void {
    try buffer.emitU8((@as(u8, mode) << 6) | ((@as(u8, reg) & 7) << 3) | (@as(u8, rm) & 7));
}

fn emitPush(buffer: *code_buffer.Buffer, reg: u4) Error!void {
    if ((reg & 8) != 0) try buffer.emitU8(0x41);
    try buffer.emitU8(0x50 + @as(u8, reg & 7));
}

fn emitPop(buffer: *code_buffer.Buffer, reg: u4) Error!void {
    if ((reg & 8) != 0) try buffer.emitU8(0x41);
    try buffer.emitU8(0x58 + @as(u8, reg & 7));
}

fn emitMovRegReg(buffer: *code_buffer.Buffer, dst: u4, src: u4) Error!void {
    if (dst == src) return;
    try emitRex(buffer, true, src, dst);
    try buffer.emitU8(0x89);
    try emitModRm(buffer, 3, src, dst);
}

fn emitMovRegImm64(buffer: *code_buffer.Buffer, dst: u4, value: u64) Error!void {
    try emitRex(buffer, true, 0, dst);
    try buffer.emitU8(0xb8 + @as(u8, dst & 7));
    try buffer.emitU64(value);
}

fn emitMovRegMem(buffer: *code_buffer.Buffer, dst: u4, base: u4, displacement: u32) Error!void {
    try emitRex(buffer, true, dst, base);
    try buffer.emitU8(0x8b);
    if (displacement <= std.math.maxInt(u8) / 2) {
        try emitModRm(buffer, 1, dst, base);
        try buffer.emitU8(@intCast(displacement));
    } else {
        try emitModRm(buffer, 2, dst, base);
        try buffer.emitU32(displacement);
    }
}

fn emitMovMemReg(buffer: *code_buffer.Buffer, base: u4, displacement: u32, src: u4) Error!void {
    try emitRex(buffer, true, src, base);
    try buffer.emitU8(0x89);
    if (displacement <= 127) {
        try emitModRm(buffer, 1, src, base);
        try buffer.emitU8(@intCast(displacement));
    } else {
        try emitModRm(buffer, 2, src, base);
        try buffer.emitU32(displacement);
    }
}

fn emitAdjustStack(buffer: *code_buffer.Buffer, subtract: bool, amount: u32) Error!void {
    if (amount <= 127) {
        try buffer.emitBytes(&.{ 0x48, 0x83, if (subtract) 0xec else 0xc4, @intCast(amount) });
    } else {
        try buffer.emitBytes(&.{ 0x48, 0x81, if (subtract) 0xec else 0xc4 });
        try buffer.emitU32(amount);
    }
}

fn emitCall(buffer: *code_buffer.Buffer, target: u4) Error!void {
    try emitRex(buffer, false, 2, target);
    try buffer.emitU8(0xff);
    try emitModRm(buffer, 3, 2, target);
}

fn emitSavedManagedRegisters(buffer: *code_buffer.Buffer) Error!void {
    for ([_]u4{ rax, rcx, rdx, rsi, rdi, r8, r9 }) |reg| try emitPush(buffer, reg);
}

fn emitRestoredManagedRegisters(buffer: *code_buffer.Buffer) Error!void {
    for ([_]u4{ r9, r8, rdi, rsi, rdx, rcx, rax }) |reg| try emitPop(buffer, reg);
}

fn emitCapturedGpRegisters(buffer: *code_buffer.Buffer) Error!void {
    const base: u32 = @offsetOf(runtime_jit.NativeThreadState, "captured_gp");
    for ([_]u4{ rax, rcx, rdx, rsi, rdi, r8, r9, r10 }) |reg| {
        try emitMovMemReg(buffer, r15, base + @as(u32, reg) * @sizeOf(u64), reg);
    }
}

fn emitCapturedXmmRegisters(buffer: *code_buffer.Buffer) Error!void {
    const base: u32 = @offsetOf(runtime_jit.NativeThreadState, "captured_xmm");
    for (0..8) |index| {
        const xmm: u4 = @intCast(index);
        // movdqu [r15 + disp32], xmmN. Keep the legacy prefix before REX.
        try buffer.emitU8(0xf3);
        try emitRex(buffer, false, xmm, r15);
        try buffer.emitBytes(&.{ 0x0f, 0x7f });
        try emitModRm(buffer, 2, xmm, r15);
        try buffer.emitU32(base + @as(u32, @intCast(index)) * 16);
    }
}

fn emitCapturedStackBase(buffer: *code_buffer.Buffer, frame_bytes: u32) Error!void {
    // Seven GP pushes and the helper CALL separate the adjusted helper rsp
    // from the managed frame's rsp by 64 bytes.
    try buffer.emitBytes(&.{ 0x48, 0x8d, 0x84, 0x24 }); // lea rax, [rsp + disp32]
    try buffer.emitU32(frame_bytes + 64);
    try emitMovMemReg(buffer, r15, @offsetOf(runtime_jit.NativeThreadState, "captured_stack_base"), rax);
}

const xmm_save_bytes: u32 = 8 * 16;

fn emitXmmStack(buffer: *code_buffer.Buffer, xmm: u3, displacement: u32, store: bool) Error!void {
    try buffer.emitBytes(&.{ 0xf3, 0x0f, if (store) 0x7f else 0x6f });
    try buffer.emitU8(0x84 | (@as(u8, xmm) << 3));
    try buffer.emitU8(0x24); // SIB: [rsp + disp32]
    try buffer.emitU32(displacement);
}

fn emitSavedXmmRegisters(buffer: *code_buffer.Buffer, base: u32) Error!void {
    for (0..8) |index| try emitXmmStack(buffer, @intCast(index), base + @as(u32, @intCast(index)) * 16, true);
}

fn emitRestoredXmmRegisters(buffer: *code_buffer.Buffer, base: u32) Error!void {
    for (0..8) |index| try emitXmmStack(buffer, @intCast(index), base + @as(u32, @intCast(index)) * 16, false);
}

fn emitSlowHelper(buffer: *code_buffer.Buffer, bridge_address: usize) Error!void {
    try emitCapturedGpRegisters(buffer);
    try emitSavedManagedRegisters(buffer);
    // Frameless generated code has rsp%16 == 8 at the private helper call;
    // after CALL and seven saves the adapter needs one alignment slot, plus
    // the Windows 32-byte home area, before entering a platform-ABI function.
    const home_bytes: u32 = if (builtin.os.tag == .windows) 32 else 0;
    const alignment_bytes: u32 = 8;
    const frame_bytes = home_bytes + xmm_save_bytes + alignment_bytes;
    try emitAdjustStack(buffer, true, frame_bytes);
    try emitSavedXmmRegisters(buffer, home_bytes);
    try emitCapturedStackBase(buffer, frame_bytes);

    if (builtin.os.tag == .windows) {
        try emitMovRegReg(buffer, rcx, r15);
        try emitMovRegReg(buffer, rdx, r10);
        try emitMovRegReg(buffer, r8, r11);
    } else {
        try emitMovRegReg(buffer, rdi, r15);
        try emitMovRegReg(buffer, rsi, r10);
        try emitMovRegReg(buffer, rdx, r11);
    }
    try emitMovRegImm64(buffer, rax, bridge_address);
    try emitCall(buffer, rax);
    try emitMovRegReg(buffer, r10, rax);
    try emitMovRegMem(buffer, r12, r15, @offsetOf(runtime_jit.NativeThreadState, "acknowledged_epoch"));

    try emitRestoredXmmRegisters(buffer, home_bytes);
    try emitAdjustStack(buffer, false, frame_bytes);
    try emitRestoredManagedRegisters(buffer);
    try buffer.emitBytes(&.{ 0x4d, 0x85, 0xd2 }); // test r10, r10
    const failed = try buffer.newLabel();
    try buffer.emitBytes(&.{ 0x0f, 0x84 });
    _ = try buffer.reloc(failed, .rel32, 0);
    try buffer.emitU8(0xc3);
    try buffer.bindLabel(failed);
    try buffer.emitBytes(&.{ 0x0f, 0x0b }); // fail closed: ud2
}

fn emitDeoptHelper(buffer: *code_buffer.Buffer, bridge_address: usize) Error!void {
    try emitCapturedGpRegisters(buffer);
    try emitCapturedXmmRegisters(buffer);
    try emitSavedManagedRegisters(buffer);
    const home_bytes: u32 = if (builtin.os.tag == .windows) 32 else 0;
    const alignment_bytes: u32 = 8;
    const frame_bytes = home_bytes + xmm_save_bytes + alignment_bytes;
    try emitAdjustStack(buffer, true, frame_bytes);
    try emitSavedXmmRegisters(buffer, home_bytes);
    try emitCapturedStackBase(buffer, frame_bytes);

    if (builtin.os.tag == .windows) {
        try emitMovRegReg(buffer, rcx, r15);
        try emitMovRegReg(buffer, rdx, r10);
    } else {
        try emitMovRegReg(buffer, rdi, r15);
        try emitMovRegReg(buffer, rsi, r10);
    }
    try emitMovRegImm64(buffer, rax, bridge_address);
    try emitCall(buffer, rax);
    try emitMovRegReg(buffer, r10, rax);
    try emitMovRegMem(buffer, r12, r15, @offsetOf(runtime_jit.NativeThreadState, "acknowledged_epoch"));

    try emitRestoredXmmRegisters(buffer, home_bytes);
    try emitAdjustStack(buffer, false, frame_bytes);
    try emitRestoredManagedRegisters(buffer);
    try buffer.emitU8(0xc3);
}

fn emitBarrierHelper(buffer: *code_buffer.Buffer, bridge_address: usize) Error!void {
    try emitSavedManagedRegisters(buffer);
    // The caller preserves rax before the private call, so this helper enters
    // with rsp%16 == 8. Seven GP saves restore platform-call alignment without
    // the extra slot required by the slow-resolve adapter.
    const home_bytes: u32 = if (builtin.os.tag == .windows) 32 else 0;
    const frame_bytes = home_bytes + xmm_save_bytes;
    try emitAdjustStack(buffer, true, frame_bytes);
    try emitSavedXmmRegisters(buffer, home_bytes);

    if (builtin.os.tag == .windows) {
        try emitMovRegReg(buffer, rcx, r15);
        try emitMovRegReg(buffer, rdx, r10);
        try emitMovRegReg(buffer, r8, r11);
    } else {
        try emitMovRegReg(buffer, rdi, r15);
        try emitMovRegReg(buffer, rsi, r10);
        try emitMovRegReg(buffer, rdx, r11);
    }
    try emitMovRegImm64(buffer, rax, bridge_address);
    try emitCall(buffer, rax);
    try emitMovRegReg(buffer, r10, rax);

    try emitRestoredXmmRegisters(buffer, home_bytes);
    try emitAdjustStack(buffer, false, frame_bytes);
    try emitRestoredManagedRegisters(buffer);
    try buffer.emitBytes(&.{ 0x4d, 0x85, 0xd2 });
    const failed = try buffer.newLabel();
    try buffer.emitBytes(&.{ 0x0f, 0x84 });
    _ = try buffer.reloc(failed, .rel32, 0);
    try buffer.emitU8(0xc3);
    try buffer.bindLabel(failed);
    try buffer.emitBytes(&.{ 0x0f, 0x0b });
}

fn emitReservedSave(buffer: *code_buffer.Buffer) Error!void {
    // The generated private ABI may allocate managed values in rsi/rdi. They
    // are caller-saved on SysV but nonvolatile on Windows, so the public entry
    // trampoline must preserve them before returning to platform-ABI code.
    if (builtin.os.tag == .windows) {
        try emitPush(buffer, rsi);
        try emitPush(buffer, rdi);
    }
    for ([_]u4{ r12, r13, r14, r15 }) |reg| try emitPush(buffer, reg);
}

fn emitReservedRestore(buffer: *code_buffer.Buffer) Error!void {
    for ([_]u4{ r15, r14, r13, r12 }) |reg| try emitPop(buffer, reg);
    if (builtin.os.tag == .windows) {
        try emitPop(buffer, rdi);
        try emitPop(buffer, rsi);
    }
}

fn imageOffset(comptime field: []const u8) u32 {
    return @offsetOf(CallFrame, "image") + @offsetOf(runtime_jit.RegisterImage, field);
}

fn argOffset(index: u32) u32 {
    return @offsetOf(CallFrame, "gp_args") + index * @sizeOf(usize);
}

fn emitRuntimeRegisters(buffer: *code_buffer.Buffer) Error!void {
    // r13 is the frame pointer until its region-base address is loaded last.
    // Its encoding avoids the mandatory SIB byte required by an r12 base.
    try emitMovRegMem(buffer, r14, r13, imageOffset("r14_descriptor_base"));
    try emitMovRegMem(buffer, r15, r13, imageOffset("r15_thread_state"));
}

fn emitOriginalArguments(buffer: *code_buffer.Buffer) Error!void {
    if (builtin.os.tag == .windows) {
        try emitMovRegMem(buffer, rdx, r13, argOffset(1));
        try emitMovRegMem(buffer, r8, r13, argOffset(2));
        try emitMovRegMem(buffer, r9, r13, argOffset(3));
        try emitMovRegMem(buffer, rcx, r13, argOffset(0));
    } else {
        try emitMovRegMem(buffer, rsi, r13, argOffset(1));
        try emitMovRegMem(buffer, rdx, r13, argOffset(2));
        try emitMovRegMem(buffer, rcx, r13, argOffset(3));
        try emitMovRegMem(buffer, r8, r13, argOffset(4));
        try emitMovRegMem(buffer, r9, r13, argOffset(5));
        try emitMovRegMem(buffer, rdi, r13, argOffset(0));
    }
}

fn emitFallbackArguments(buffer: *code_buffer.Buffer) Error!void {
    // Stable fallback ABI: NativeThreadState* followed by zeroed reserved
    // arguments. Original managed arguments are available in captured_gp.
    if (builtin.os.tag == .windows) {
        try emitMovRegReg(buffer, rcx, r15);
        try emitMovRegImm64(buffer, rdx, 0);
        try emitMovRegImm64(buffer, r8, 0);
        try emitMovRegImm64(buffer, r9, 0);
    } else {
        try emitMovRegReg(buffer, rdi, r15);
        try emitMovRegImm64(buffer, rsi, 0);
        try emitMovRegImm64(buffer, rdx, 0);
        try emitMovRegImm64(buffer, rcx, 0);
        try emitMovRegImm64(buffer, r8, 0);
        try emitMovRegImm64(buffer, r9, 0);
    }
}

fn emitFinalizedTargetCall(buffer: *code_buffer.Buffer) Error!void {
    try emitMovRegMem(buffer, r12, r13, imageOffset("r12_acknowledged_epoch"));
    try emitMovRegMem(buffer, r13, r13, imageOffset("r13_region_bases"));
    try emitCall(buffer, r10);
}

fn emitTargetCall(buffer: *code_buffer.Buffer) Error!void {
    try emitRuntimeRegisters(buffer);
    try emitOriginalArguments(buffer);
    try emitFinalizedTargetCall(buffer);
}

fn emitManagedTargetCall(buffer: *code_buffer.Buffer) Error!void {
    try emitRuntimeRegisters(buffer);
    try emitMovRegMem(buffer, r11, r15, @offsetOf(runtime_jit.NativeThreadState, "last_code_dispatch"));
    try buffer.emitBytes(&.{ 0x41, 0x83, 0xfb, 0x00 }); // cmp r11d, 0
    const fallback = try buffer.newLabel();
    try buffer.emitBytes(&.{ 0x0f, 0x85 }); // jne fallback
    _ = try buffer.reloc(fallback, .rel32, 0);
    try emitOriginalArguments(buffer);
    try emitFinalizedTargetCall(buffer);
    const done = try buffer.newLabel();
    try buffer.emitU8(0xe9);
    _ = try buffer.reloc(done, .rel32, 0);
    try buffer.bindLabel(fallback);
    try emitFallbackArguments(buffer);
    try emitFinalizedTargetCall(buffer);
    try buffer.bindLabel(done);
}

fn emitCapturedEntryArguments(buffer: *code_buffer.Buffer) Error!void {
    try emitMovRegMem(buffer, r11, r13, imageOffset("r15_thread_state"));
    const physical = if (builtin.os.tag == .windows)
        [_]u4{ rcx, rdx, r8, r9 }
    else
        [_]u4{ rdi, rsi, rdx, rcx, r8, r9 };
    const base: u32 = @offsetOf(runtime_jit.NativeThreadState, "captured_gp");
    for (physical, 0..) |register, argument| {
        try emitMovRegMem(buffer, rax, r13, argOffset(@intCast(argument)));
        try emitMovMemReg(buffer, r11, base + @as(u32, register) * @sizeOf(u64), rax);
    }
}

fn emitEntry(buffer: *code_buffer.Buffer) Error!void {
    const platform_frame = if (builtin.os.tag == .windows) rcx else rdi;
    try emitReservedSave(buffer);
    try emitAdjustStack(buffer, true, if (builtin.os.tag == .windows) 40 else 8);
    try emitMovRegReg(buffer, r13, platform_frame);

    // A sentinel method id keeps raw-address tests and non-managed stubs on a
    // zero-bridge fast path. All real method ids enter through the code lease.
    try emitMovRegMem(buffer, r11, r13, @offsetOf(CallFrame, "method_id"));
    try buffer.emitBytes(&.{ 0x41, 0x83, 0xfb, 0xff }); // cmp r11d, -1
    const managed = try buffer.newLabel();
    try buffer.emitBytes(&.{ 0x0f, 0x85 }); // jne managed
    _ = try buffer.reloc(managed, .rel32, 0);

    try emitMovRegMem(buffer, r10, r13, @offsetOf(CallFrame, "target"));
    try emitTargetCall(buffer);
    const done = try buffer.newLabel();
    try buffer.emitU8(0xe9);
    _ = try buffer.reloc(done, .rel32, 0);

    try buffer.bindLabel(managed);
    try emitCapturedEntryArguments(buffer);
    if (builtin.os.tag == .windows) {
        try emitMovRegMem(buffer, rcx, r13, imageOffset("r15_thread_state"));
        try emitMovRegMem(buffer, rdx, r13, @offsetOf(CallFrame, "fallback_target"));
        try emitMovRegMem(buffer, r8, r13, @offsetOf(CallFrame, "method_id"));
        try emitMovRegMem(buffer, r9, r13, @offsetOf(CallFrame, "deopt_request"));
    } else {
        try emitMovRegMem(buffer, rdi, r13, imageOffset("r15_thread_state"));
        try emitMovRegMem(buffer, rsi, r13, @offsetOf(CallFrame, "fallback_target"));
        try emitMovRegMem(buffer, rdx, r13, @offsetOf(CallFrame, "method_id"));
        try emitMovRegMem(buffer, rcx, r13, @offsetOf(CallFrame, "deopt_request"));
    }
    try emitMovRegImm64(buffer, rax, @intFromPtr(&runtime_jit.codeLeaseEnterBridge));
    try emitCall(buffer, rax);
    try emitMovRegReg(buffer, r10, rax);
    try emitManagedTargetCall(buffer);

    if (builtin.os.tag == .windows) {
        try emitMovRegReg(buffer, rcx, r15);
        try emitMovRegReg(buffer, rdx, rax);
    } else {
        try emitMovRegReg(buffer, rdi, r15);
        try emitMovRegReg(buffer, rsi, rax);
    }
    try emitMovRegImm64(buffer, rax, @intFromPtr(&runtime_jit.codeLeaseExitBridge));
    try emitCall(buffer, rax);

    try buffer.bindLabel(done);
    try emitAdjustStack(buffer, false, if (builtin.os.tag == .windows) 40 else 8);
    try emitReservedRestore(buffer);
    try buffer.emitU8(0xc3);
}

fn osrGpOffset(register: u4) u32 {
    return @offsetOf(OsrCallFrame, "gp") + @as(u32, register) * @sizeOf(u64);
}

fn emitOsrEntry(buffer: *code_buffer.Buffer) Error!void {
    const platform_frame = if (builtin.os.tag == .windows) rcx else rdi;
    try emitReservedSave(buffer);
    try emitAdjustStack(buffer, true, if (builtin.os.tag == .windows) 40 else 8);
    try emitMovRegReg(buffer, r13, platform_frame);
    try emitMovRegMem(buffer, r10, r13, @offsetOf(OsrCallFrame, "target"));
    try emitMovRegMem(buffer, r14, r13, @offsetOf(OsrCallFrame, "image") + @offsetOf(runtime_jit.RegisterImage, "r14_descriptor_base"));
    try emitMovRegMem(buffer, r15, r13, @offsetOf(OsrCallFrame, "image") + @offsetOf(runtime_jit.RegisterImage, "r15_thread_state"));
    for ([_]u4{ rax, rcx, rdx, rsi, rdi, r8, r9 }) |register| {
        try emitMovRegMem(buffer, register, r13, osrGpOffset(register));
    }
    try emitMovRegMem(buffer, r12, r13, @offsetOf(OsrCallFrame, "image") + @offsetOf(runtime_jit.RegisterImage, "r12_acknowledged_epoch"));
    try emitMovRegMem(buffer, r13, r13, @offsetOf(OsrCallFrame, "image") + @offsetOf(runtime_jit.RegisterImage, "r13_region_bases"));
    try emitCall(buffer, r10);
    try emitAdjustStack(buffer, false, if (builtin.os.tag == .windows) 40 else 8);
    try emitReservedRestore(buffer);
    try buffer.emitU8(0xc3);
}

pub fn encode(allocator: std.mem.Allocator, bridge_address: usize) Error!EncodedShims {
    if (builtin.cpu.arch != .x86_64) return error.UnsupportedArchitecture;
    var entry_buffer = code_buffer.Buffer.init(allocator);
    defer entry_buffer.deinit();
    try emitEntry(&entry_buffer);
    const entry = try entry_buffer.finalize();
    errdefer allocator.free(entry);

    var osr_entry_buffer = code_buffer.Buffer.init(allocator);
    defer osr_entry_buffer.deinit();
    try emitOsrEntry(&osr_entry_buffer);
    const osr_entry = try osr_entry_buffer.finalize();
    errdefer allocator.free(osr_entry);

    var helper_buffer = code_buffer.Buffer.init(allocator);
    defer helper_buffer.deinit();
    try emitSlowHelper(&helper_buffer, bridge_address);
    const slow_helper = try helper_buffer.finalize();
    errdefer allocator.free(slow_helper);

    var deopt_buffer = code_buffer.Buffer.init(allocator);
    defer deopt_buffer.deinit();
    try emitDeoptHelper(&deopt_buffer, @intFromPtr(&runtime_jit.midFunctionDeoptBridge));
    const deopt_helper = try deopt_buffer.finalize();
    errdefer allocator.free(deopt_helper);

    var bounds_buffer = code_buffer.Buffer.init(allocator);
    defer bounds_buffer.deinit();
    try emitSlowHelper(&bounds_buffer, @intFromPtr(&runtime_jit.boundsExceptionBridge));
    const bounds_exception_helper = try bounds_buffer.finalize();
    errdefer allocator.free(bounds_exception_helper);

    var satb_buffer = code_buffer.Buffer.init(allocator);
    defer satb_buffer.deinit();
    try emitBarrierHelper(&satb_buffer, @intFromPtr(&runtime_jit.referencePreWriteBridge));
    const satb_pre_write_helper = try satb_buffer.finalize();
    errdefer allocator.free(satb_pre_write_helper);

    var card_buffer = code_buffer.Buffer.init(allocator);
    defer card_buffer.deinit();
    try emitBarrierHelper(&card_buffer, @intFromPtr(&runtime_jit.referencePostWriteBridge));
    const card_mark_helper = try card_buffer.finalize();
    errdefer allocator.free(card_mark_helper);

    var card_repeat_buffer = code_buffer.Buffer.init(allocator);
    defer card_repeat_buffer.deinit();
    try emitBarrierHelper(&card_repeat_buffer, @intFromPtr(&runtime_jit.referencePostWriteRepeatBridge));
    const card_mark_repeat_helper = try card_repeat_buffer.finalize();
    errdefer allocator.free(card_mark_repeat_helper);

    var static_root_buffer = code_buffer.Buffer.init(allocator);
    defer static_root_buffer.deinit();
    try emitBarrierHelper(&static_root_buffer, @intFromPtr(&runtime_jit.referenceStaticPostWriteBridge));
    const static_root_post_write_helper = try static_root_buffer.finalize();
    return .{
        .allocator = allocator,
        .entry = entry,
        .osr_entry = osr_entry,
        .slow_helper = slow_helper,
        .deopt_helper = deopt_helper,
        .bounds_exception_helper = bounds_exception_helper,
        .satb_pre_write_helper = satb_pre_write_helper,
        .card_mark_helper = card_mark_helper,
        .card_mark_repeat_helper = card_mark_repeat_helper,
        .static_root_post_write_helper = static_root_post_write_helper,
    };
}

test "x64 shims execute generated field access across evacuation" {
    if (builtin.cpu.arch != .x86_64) return error.SkipZigTest;
    var from_space: [128]u8 align(runtime_value.object_alignment) = @splat(0);
    var to_space: [128]u8 align(runtime_value.object_alignment) = @splat(0);
    std.mem.writeInt(i32, from_space[16..20], 41, .little);
    std.mem.writeInt(i32, to_space[32..36], 99, .little);
    const regions = [_]runtime_value.Region{
        try runtime_value.Region.fromSlice(&from_space),
        try runtime_value.Region.fromSlice(&to_space),
    };
    var handles = try runtime_value.HandleTable.init(std.testing.allocator, 4, &regions);
    defer handles.deinit();
    const handle = try handles.reserve(0, 0);
    try handles.publish(handle, 0, @ptrCast(&from_space[8]));

    var registry = try runtime_thread_registry.Registry.init(std.testing.allocator, std.testing.io, 1);
    defer registry.deinit() catch unreachable;
    var context = try runtime_thread_registry.ThreadContext.init(std.testing.allocator, 4);
    defer context.deinit();
    try registry.register(&context);
    defer registry.unregister(&context) catch unreachable;
    var runtime = try runtime_jit.Runtime.init(std.testing.allocator, &handles, &registry);
    defer runtime.deinit() catch unreachable;
    var managed = try runtime.enter(&context);
    defer managed.deinit();

    var encoded_shims = try encode(std.testing.allocator, @intFromPtr(&runtime_jit.slowResolveBridge));
    defer encoded_shims.deinit();
    var cache = jit_memory.Cache.init(std.testing.allocator);
    defer cache.deinit();
    const entry_code = try cache.addBytes(encoded_shims.entry);
    const helper_code = try cache.addBytes(encoded_shims.slow_helper);

    const insts = [_]Instruction{
        .{ .iget = .{ .field_idx = 1, .dest_or_src = 1, .obj = 0 } },
        .{ .return_ = .{ .src = 1 } },
    };
    var optimized = try optimizer.optimize(std.testing.allocator, &insts, &.{}, .{});
    defer optimized.deinit();
    const layouts = [_]x64_encoder.FieldLayout{
        .{ .offset = 0, .storage = .i32 },
        .{ .offset = 8, .storage = .i32 },
    };
    var native = try x64_encoder.encodeWithOptions(std.testing.allocator, &optimized.machine, .{
        .runtime = .{
            .handle_capacity = handles.entryCapacity(),
            .region_count = handles.regionCount(),
            .slow_resolve_helper = helper_code.entryAddress(),
            .field_layouts = &layouts,
        },
    });
    defer native.deinit();
    const target_bytes = try native.finalize();
    defer std.testing.allocator.free(target_bytes);
    const target_code = try cache.addBytes(target_bytes);

    var frame = CallFrame{
        .image = undefined,
        .target = target_code.entryAddress(),
    };
    const root_maps = if (native.root_maps) |*maps| maps else return error.TestUnexpectedResult;
    try managed.installRootMaps(root_maps);
    frame.image = try managed.registerImage();
    frame.gp_args[0] = @bitCast(handle);
    const EntryFn = fn (*const CallFrame) callconv(.c) usize;
    const call = entry_code.typedEntry(EntryFn);
    try std.testing.expectEqual(@as(usize, 41), call(&frame));

    const ticket = try handles.beginRelocation(handle);
    try std.testing.expectEqual(runtime_value.EntryState.evacuating, (try handles.inspect(handle)).state);
    try std.testing.expectEqual(@as(usize, 41), call(&frame));
    try std.testing.expectEqual(@as(u64, 1), runtime.stats().slow_resolves);
    try std.testing.expectEqual(runtime_jit.SlowResolveStatus.ok, managed.native_state.last_error);

    try std.testing.expect(try handles.commitRelocation(ticket, 1, @ptrCast(&to_space[24])));
    try std.testing.expectEqual(@as(usize, 99), call(&frame));
}

test "x64 reference store executes SATB and card barriers" {
    if (builtin.cpu.arch != .x86_64) return error.SkipZigTest;

    var old_region: [128]u8 align(runtime_value.object_alignment) = @splat(0);
    var young_region: [128]u8 align(runtime_value.object_alignment) = @splat(0);
    const regions = [_]runtime_value.Region{
        try runtime_value.Region.fromSlice(&old_region),
        try runtime_value.Region.fromSlice(&young_region),
    };
    var handles = try runtime_value.HandleTable.init(std.testing.allocator, 5, &regions);
    defer handles.deinit();
    var heap = try runtime_heap.ManagedHeap.init(std.testing.allocator, &handles, 128);
    defer heap.deinit();
    const reference_offsets = [_]u32{0};
    const gc_layouts = [_]runtime_gc.LayoutSpec{.{
        .id = 1,
        .minimum_size = @sizeOf(runtime_value.Handle),
        .reference_offsets = &reference_offsets,
    }};
    var collector = try runtime_gc.ConcurrentCollector.init(std.testing.allocator, &heap, &handles, .{
        .satb_queue_capacity = 4,
        .max_satb_buffers = 1,
        .card_bytes = 64,
        .layouts = &gc_layouts,
    });
    defer collector.deinit() catch unreachable;

    var tlab = heap.threadAllocator();
    const destination_reservation = try tlab.allocate(112, runtime_value.object_alignment);
    const destination = try handles.reserve(0, 0);
    try heap.publishObjectWithLayout(destination_reservation, destination, 1);
    const old_reservation = try tlab.allocate(8, runtime_value.object_alignment);
    const old_value = try handles.reserve(0, 0);
    try heap.publishObject(old_reservation, old_value);
    const intermediate_reservation = try tlab.allocate(8, runtime_value.object_alignment);
    const intermediate_value = try handles.reserve(0, 0);
    try heap.publishObject(intermediate_reservation, intermediate_value);
    const new_reservation = try tlab.allocate(8, runtime_value.object_alignment);
    const new_value = try handles.reserve(0, 0);
    try heap.publishObject(new_reservation, new_value);
    try std.testing.expectEqual(@as(u8, 0), (try handles.inspect(destination)).region_id);
    try std.testing.expectEqual(@as(u8, 1), (try handles.inspect(old_value)).region_id);
    try std.testing.expectEqual(@as(u8, 1), (try handles.inspect(intermediate_value)).region_id);
    try std.testing.expectEqual(@as(u8, 1), (try handles.inspect(new_value)).region_id);
    const slot: *std.atomic.Value(u64) = @ptrCast(@alignCast(try handles.resolve(destination)));
    slot.store(@bitCast(old_value), .release);
    try collector.setRegionKind(0, .old);
    try collector.setRegionKind(1, .young);

    var registry = try runtime_thread_registry.Registry.init(std.testing.allocator, std.testing.io, 1);
    defer registry.deinit() catch unreachable;
    var context = try runtime_thread_registry.ThreadContext.init(std.testing.allocator, 4);
    defer context.deinit();
    var root = destination;
    try context.addRoot(&root);
    try registry.register(&context);
    defer registry.unregister(&context) catch unreachable;
    var satb = try runtime_gc.SatbBuffer.init(std.testing.allocator, 2);
    defer satb.deinit() catch unreachable;
    try collector.registerThreadSatbBuffer(&satb, &context);
    defer collector.unregisterSatbBuffer(&satb) catch unreachable;

    // A passive root handshake establishes the beginning-of-snapshot graph.
    try collector.enterBlockedForMark(&registry, &context, &satb);
    const mark_epoch = try collector.beginMark();
    var mark_handshake = try collector.beginThreadHandshake(&registry);
    for (0..1_000_000) |_| {
        if (try mark_handshake.advance()) break;
        std.atomic.spinLoopHint();
        std.Thread.yield() catch {};
    } else return error.Timeout;
    try registry.leaveBlocked(&context);

    var runtime = try runtime_jit.Runtime.init(std.testing.allocator, &handles, &registry);
    defer runtime.deinit() catch unreachable;
    try runtime.installCollector(&collector);
    var managed = try runtime.enter(&context);
    defer managed.deinit();

    var encoded_shims = try encode(std.testing.allocator, @intFromPtr(&runtime_jit.slowResolveBridge));
    defer encoded_shims.deinit();
    var cache = jit_memory.Cache.init(std.testing.allocator);
    defer cache.deinit();
    const entry_code = try cache.addBytes(encoded_shims.entry);
    const slow_code = try cache.addBytes(encoded_shims.slow_helper);
    const satb_code = try cache.addBytes(encoded_shims.satb_pre_write_helper);
    const card_code = try cache.addBytes(encoded_shims.card_mark_helper);
    const card_repeat_code = try cache.addBytes(encoded_shims.card_mark_repeat_helper);

    const insts = [_]Instruction{
        .{ .iput_object = .{ .field_idx = 0, .dest_or_src = 1, .obj = 0 } },
        .{ .iput_object = .{ .field_idx = 0, .dest_or_src = 2, .obj = 0 } },
        .return_void,
    };
    var optimized = try optimizer.optimize(std.testing.allocator, &insts, &.{}, .{});
    defer optimized.deinit();
    const field_layouts = [_]x64_encoder.FieldLayout{.{ .offset = 0, .storage = .reference }};
    var native = try x64_encoder.encodeWithOptions(std.testing.allocator, &optimized.machine, .{
        .runtime = .{
            .handle_capacity = handles.entryCapacity(),
            .region_count = handles.regionCount(),
            .slow_resolve_helper = slow_code.entryAddress(),
            .satb_pre_write_helper = satb_code.entryAddress(),
            .card_mark_helper = card_code.entryAddress(),
            .card_mark_repeat_helper = card_repeat_code.entryAddress(),
            .field_layouts = &field_layouts,
        },
    });
    defer native.deinit();
    const target_bytes = try native.finalize();
    defer std.testing.allocator.free(target_bytes);
    const target_code = try cache.addBytes(target_bytes);

    const root_maps = if (native.root_maps) |*maps| maps else return error.TestUnexpectedResult;
    try managed.installRootMaps(root_maps);
    var frame = CallFrame{
        .image = try managed.registerImage(),
        .target = target_code.entryAddress(),
    };
    frame.gp_args[0] = @bitCast(destination);
    frame.gp_args[1] = @bitCast(intermediate_value);
    frame.gp_args[2] = @bitCast(new_value);
    const EntryFn = fn (*const CallFrame) callconv(.c) usize;
    _ = entry_code.typedEntry(EntryFn)(&frame);

    try std.testing.expectEqual(@as(u64, @bitCast(new_value)), slot.load(.acquire));
    try std.testing.expectEqual(@as(usize, 1), satb.pendingCount());
    try std.testing.expect(try collector.isCardDirty(destination));
    try std.testing.expectEqual(@as(u64, 2), runtime.stats().pre_write_barriers);
    try std.testing.expectEqual(@as(u64, 2), runtime.stats().post_write_barriers);
    try std.testing.expectEqual(@as(u64, 0), runtime.stats().barrier_failures);
    try std.testing.expectEqual(@as(u64, 1), collector.stats().satb_repeat_elisions);
    try std.testing.expectEqual(@as(u64, 1), collector.stats().card_repeat_elisions);
    try std.testing.expectEqual(@as(u32, 1), native.stats.satb_repeat_barriers);
    try std.testing.expectEqual(@as(u32, 1), native.stats.card_repeat_barriers);

    try satb.flushForEpoch(mark_epoch);
    try std.testing.expectEqual(@as(usize, 1), try collector.drainSatb(4));
    try std.testing.expectEqual(@as(usize, 3), try collector.traceWork(8));
    try collector.tryFinishMark();
    try std.testing.expect(try collector.isMarked(destination));
    try std.testing.expect(try collector.isMarked(old_value));
    try std.testing.expect(!(try collector.isMarked(intermediate_value)));
    try std.testing.expect(try collector.isMarked(new_value));
}

test "x64 reference array store barriers use the exact element slot" {
    if (builtin.cpu.arch != .x86_64) return error.SkipZigTest;

    var old_region: [128]u8 align(runtime_value.object_alignment) = @splat(0);
    var young_region: [128]u8 align(runtime_value.object_alignment) = @splat(0);
    const regions = [_]runtime_value.Region{
        try runtime_value.Region.fromSlice(&old_region),
        try runtime_value.Region.fromSlice(&young_region),
    };
    var handles = try runtime_value.HandleTable.init(std.testing.allocator, 4, &regions);
    defer handles.deinit();
    var heap = try runtime_heap.ManagedHeap.init(std.testing.allocator, &handles, 64);
    defer heap.deinit();
    const gc_layouts = [_]runtime_gc.LayoutSpec{.{
        .id = 1,
        .minimum_size = 8,
        .trailing_references = .{ .offset = 8, .stride = 8 },
    }};
    var collector = try runtime_gc.ConcurrentCollector.init(std.testing.allocator, &heap, &handles, .{
        .satb_queue_capacity = 4,
        .max_satb_buffers = 1,
        .card_bytes = 64,
        .layouts = &gc_layouts,
    });
    defer collector.deinit() catch unreachable;

    var tlab = heap.threadAllocator();
    const array_reservation = try tlab.allocate(48, runtime_value.object_alignment);
    const array = try handles.reserve(0, 0);
    try heap.publishObjectWithLayout(array_reservation, array, 1);
    const old_reservation = try tlab.allocate(8, runtime_value.object_alignment);
    const old_value = try handles.reserve(0, 0);
    try heap.publishObject(old_reservation, old_value);
    const new_reservation = try tlab.allocate(8, runtime_value.object_alignment);
    const new_value = try handles.reserve(0, 0);
    try heap.publishObject(new_reservation, new_value);
    try std.testing.expectEqual(@as(u8, 0), (try handles.inspect(array)).region_id);
    try std.testing.expectEqual(@as(u8, 1), (try handles.inspect(old_value)).region_id);
    try std.testing.expectEqual(@as(u8, 1), (try handles.inspect(new_value)).region_id);
    const payload_address = @intFromPtr(try handles.resolve(array));
    const length: *std.atomic.Value(u32) = @ptrFromInt(payload_address);
    length.store(3, .release);
    var reference_offset: usize = 8;
    while (reference_offset < array_reservation.allocated_size) : (reference_offset += @sizeOf(runtime_value.Handle)) {
        const reference: *std.atomic.Value(u64) = @ptrFromInt(payload_address + reference_offset);
        reference.store(@bitCast(runtime_value.Handle.none), .release);
    }
    const element: *std.atomic.Value(u64) = @ptrFromInt(payload_address + 16);
    element.store(@bitCast(old_value), .release);
    try collector.setRegionKind(0, .old);
    try collector.setRegionKind(1, .young);

    var registry = try runtime_thread_registry.Registry.init(std.testing.allocator, std.testing.io, 1);
    defer registry.deinit() catch unreachable;
    var context = try runtime_thread_registry.ThreadContext.init(std.testing.allocator, 4);
    defer context.deinit();
    var root = array;
    try context.addRoot(&root);
    try registry.register(&context);
    defer registry.unregister(&context) catch unreachable;
    var satb = try runtime_gc.SatbBuffer.init(std.testing.allocator, 2);
    defer satb.deinit() catch unreachable;
    try collector.registerThreadSatbBuffer(&satb, &context);
    defer collector.unregisterSatbBuffer(&satb) catch unreachable;

    try collector.enterBlockedForMark(&registry, &context, &satb);
    const mark_epoch = try collector.beginMark();
    var mark_handshake = try collector.beginThreadHandshake(&registry);
    for (0..1_000_000) |_| {
        if (try mark_handshake.advance()) break;
        std.atomic.spinLoopHint();
        std.Thread.yield() catch {};
    } else return error.Timeout;
    try registry.leaveBlocked(&context);

    var runtime = try runtime_jit.Runtime.init(std.testing.allocator, &handles, &registry);
    defer runtime.deinit() catch unreachable;
    try runtime.installCollector(&collector);
    var managed = try runtime.enter(&context);
    defer managed.deinit();
    var encoded_shims = try encode(std.testing.allocator, @intFromPtr(&runtime_jit.slowResolveBridge));
    defer encoded_shims.deinit();
    var cache = jit_memory.Cache.init(std.testing.allocator);
    defer cache.deinit();
    const entry_code = try cache.addBytes(encoded_shims.entry);
    const slow_code = try cache.addBytes(encoded_shims.slow_helper);
    const bounds_code = try cache.addBytes(encoded_shims.bounds_exception_helper);
    const satb_code = try cache.addBytes(encoded_shims.satb_pre_write_helper);
    const card_code = try cache.addBytes(encoded_shims.card_mark_helper);
    const card_repeat_code = try cache.addBytes(encoded_shims.card_mark_repeat_helper);

    const insts = [_]Instruction{
        .{ .aput_object = .{ .dest_or_src = 1, .array = 0, .index = 2 } },
        .return_void,
    };
    var optimized = try optimizer.optimize(std.testing.allocator, &insts, &.{}, .{});
    defer optimized.deinit();
    var native = try x64_encoder.encodeWithOptions(std.testing.allocator, &optimized.machine, .{
        .runtime = .{
            .handle_capacity = handles.entryCapacity(),
            .region_count = handles.regionCount(),
            .slow_resolve_helper = slow_code.entryAddress(),
            .bounds_exception_helper = bounds_code.entryAddress(),
            .satb_pre_write_helper = satb_code.entryAddress(),
            .card_mark_helper = card_code.entryAddress(),
            .card_mark_repeat_helper = card_repeat_code.entryAddress(),
            .reference_array_layout = .{ .length_offset = 0, .data_offset = 8 },
            .field_layouts = &.{},
        },
    });
    defer native.deinit();
    const target_bytes = try native.finalize();
    defer std.testing.allocator.free(target_bytes);
    const target_code = try cache.addBytes(target_bytes);
    const root_maps = if (native.root_maps) |*maps| maps else return error.TestUnexpectedResult;
    try managed.installRootMaps(root_maps);
    var frame = CallFrame{
        .image = try managed.registerImage(),
        .target = target_code.entryAddress(),
    };
    frame.gp_args[0] = @bitCast(array);
    frame.gp_args[1] = @bitCast(new_value);
    frame.gp_args[2] = 1;
    const EntryFn = fn (*const CallFrame) callconv(.c) usize;
    const call = entry_code.typedEntry(EntryFn);
    _ = call(&frame);

    try std.testing.expectEqual(@as(u64, @bitCast(new_value)), element.load(.acquire));
    try std.testing.expectEqual(@as(usize, 1), satb.pendingCount());
    try std.testing.expect(try collector.isCardDirty(array));
    try std.testing.expectEqual(@as(u32, 1), native.stats.bounds_checks);
    try std.testing.expectEqual(@as(u32, 1), native.stats.array_stores);
    try std.testing.expectEqual(@as(u64, 1), runtime.stats().pre_write_barriers);
    try std.testing.expectEqual(@as(u64, 1), runtime.stats().post_write_barriers);

    try satb.flushForEpoch(mark_epoch);
    try std.testing.expectEqual(@as(usize, 1), try collector.drainSatb(4));
    try std.testing.expectEqual(@as(usize, 3), try collector.traceWork(8));
    try collector.tryFinishMark();
    try std.testing.expect(try collector.isMarked(array));
    try std.testing.expect(try collector.isMarked(old_value));
    try std.testing.expect(try collector.isMarked(new_value));

    frame.image = try managed.registerImage();
    frame.gp_args[2] = @as(u32, @bitCast(@as(i32, -1)));
    try std.testing.expectEqual(@as(usize, 0), call(&frame));
    try std.testing.expectError(error.PendingException, managed.registerImage());
    const negative = managed.takeException() orelse return error.TestUnexpectedResult;
    try std.testing.expectEqual(runtime_jit.ManagedExceptionKind.array_index_out_of_bounds, negative.kind);
    try std.testing.expectEqual(@as(u32, 0), negative.dex_pc);
    try std.testing.expectEqual(@as(i32, -1), negative.index);
    try std.testing.expectEqual(@as(u32, 3), negative.length);

    frame.image = try managed.registerImage();
    frame.gp_args[2] = 3;
    try std.testing.expectEqual(@as(usize, 0), call(&frame));
    const upper = managed.takeException() orelse return error.TestUnexpectedResult;
    try std.testing.expectEqual(@as(i32, 3), upper.index);
    try std.testing.expectEqual(@as(u32, 3), upper.length);
    try std.testing.expectEqual(@as(u64, 2), runtime.stats().exception_transfers);
    try std.testing.expectEqual(@as(u64, 0), runtime.stats().exception_failures);
    try std.testing.expectEqual(@as(u64, 1), runtime.stats().pre_write_barriers);
    try std.testing.expectEqual(@as(u64, 1), runtime.stats().post_write_barriers);
}

test "x64 static reference store uses SATB and root publication without a fake card" {
    if (builtin.cpu.arch != .x86_64) return error.SkipZigTest;

    var region_bytes: [256]u8 align(runtime_value.object_alignment) = @splat(0);
    const regions = [_]runtime_value.Region{try runtime_value.Region.fromSlice(&region_bytes)};
    var handles = try runtime_value.HandleTable.init(std.testing.allocator, 4, &regions);
    defer handles.deinit();
    var heap = try runtime_heap.ManagedHeap.init(std.testing.allocator, &handles, 64);
    defer heap.deinit();
    var allocator = heap.threadAllocator();
    const old_reservation = try allocator.allocate(8, runtime_value.object_alignment);
    const old_value = try handles.reserve(0, 0);
    try heap.publishObject(old_reservation, old_value);
    const new_reservation = try allocator.allocate(8, runtime_value.object_alignment);
    const new_value = try handles.reserve(0, 0);
    try heap.publishObject(new_reservation, new_value);
    var static_slot = std.atomic.Value(u64).init(@bitCast(old_value));
    const static_address = @intFromPtr(&static_slot);

    var collector = try runtime_gc.ConcurrentCollector.init(std.testing.allocator, &heap, &handles, .{
        .satb_queue_capacity = 4,
        .max_satb_buffers = 1,
        .card_bytes = 64,
        .static_root_slots = &.{static_address},
    });
    defer collector.deinit() catch unreachable;
    var registry = try runtime_thread_registry.Registry.init(std.testing.allocator, std.testing.io, 1);
    defer registry.deinit() catch unreachable;
    var context = try runtime_thread_registry.ThreadContext.init(std.testing.allocator, 2);
    defer context.deinit();
    try registry.register(&context);
    defer registry.unregister(&context) catch unreachable;
    var satb = try runtime_gc.SatbBuffer.init(std.testing.allocator, 2);
    defer satb.deinit() catch unreachable;
    try collector.registerThreadSatbBuffer(&satb, &context);
    defer collector.unregisterSatbBuffer(&satb) catch unreachable;

    try collector.enterBlockedForMark(&registry, &context, &satb);
    const mark_epoch = try collector.beginMark();
    var handshake = try collector.beginThreadHandshake(&registry);
    for (0..1_000_000) |_| {
        if (try handshake.advance()) break;
        std.atomic.spinLoopHint();
        std.Thread.yield() catch {};
    } else return error.Timeout;
    try registry.leaveBlocked(&context);

    var runtime = try runtime_jit.Runtime.init(std.testing.allocator, &handles, &registry);
    defer runtime.deinit() catch unreachable;
    try runtime.installCollector(&collector);
    var managed = try runtime.enter(&context);
    defer managed.deinit();
    var encoded_shims = try encode(std.testing.allocator, @intFromPtr(&runtime_jit.slowResolveBridge));
    defer encoded_shims.deinit();
    var cache = jit_memory.Cache.init(std.testing.allocator);
    defer cache.deinit();
    const entry_code = try cache.addBytes(encoded_shims.entry);
    const slow_code = try cache.addBytes(encoded_shims.slow_helper);
    const satb_code = try cache.addBytes(encoded_shims.satb_pre_write_helper);
    const static_post_code = try cache.addBytes(encoded_shims.static_root_post_write_helper);

    const insts = [_]Instruction{
        .{ .sput_object = .{ .field_idx = 0, .dest_or_src = 0 } },
        .return_void,
    };
    var optimized = try optimizer.optimize(std.testing.allocator, &insts, &.{}, .{});
    defer optimized.deinit();
    const static_plan = optimized.barriers.ops[optimized.function.graph.entry][0];
    try std.testing.expectEqualStrings("satb_guarded", @tagName(static_plan.pre_write));
    try std.testing.expectEqualStrings("root_guarded", @tagName(static_plan.post_write));
    const static_machine = optimized.machine.blocks[optimized.function.graph.entry].insts;
    try std.testing.expect(static_machine.len >= 3);
    try std.testing.expectEqualStrings("static_satb_pre_write", @tagName(static_machine[0].opcode));
    try std.testing.expectEqualStrings("static_store", @tagName(static_machine[1].opcode));
    try std.testing.expectEqualStrings("static_root_post_write", @tagName(static_machine[2].opcode));
    var native = try x64_encoder.encodeWithOptions(std.testing.allocator, &optimized.machine, .{
        .runtime = .{
            .handle_capacity = handles.entryCapacity(),
            .region_count = handles.regionCount(),
            .slow_resolve_helper = slow_code.entryAddress(),
            .satb_pre_write_helper = satb_code.entryAddress(),
            .static_root_post_write_helper = static_post_code.entryAddress(),
            .field_layouts = &.{},
            .static_field_layouts = &.{.{ .address = static_address, .storage = .reference }},
        },
    });
    defer native.deinit();
    const target_bytes = try native.finalize();
    defer std.testing.allocator.free(target_bytes);
    const target_code = try cache.addBytes(target_bytes);
    try std.testing.expect(native.root_maps == null);
    try std.testing.expectEqual(@as(u32, 1), native.stats.satb_barriers);
    try std.testing.expectEqual(@as(u32, 1), native.stats.static_root_barriers);
    try std.testing.expectEqual(@as(u32, 0), native.stats.card_barriers);

    var frame = CallFrame{
        .image = try managed.registerImage(),
        .target = target_code.entryAddress(),
    };
    frame.gp_args[0] = @bitCast(new_value);
    const EntryFn = fn (*const CallFrame) callconv(.c) usize;
    _ = entry_code.typedEntry(EntryFn)(&frame);

    try std.testing.expectEqual(@as(u64, @bitCast(new_value)), static_slot.load(.acquire));
    try std.testing.expectEqual(@as(usize, 1), satb.pendingCount());
    try std.testing.expectEqual(@as(u64, 1), runtime.stats().pre_write_barriers);
    try std.testing.expectEqual(@as(u64, 1), runtime.stats().post_write_barriers);
    try std.testing.expectEqual(@as(u64, 0), collector.stats().cards_dirtied);
    try std.testing.expectEqual(@as(u64, 1), collector.stats().static_root_writes);

    try satb.flushForEpoch(mark_epoch);
    try std.testing.expectEqual(@as(usize, 1), try collector.drainSatb(4));
    while (try collector.traceWork(4) != 0) {}
    try collector.tryFinishMark();
    try std.testing.expect(try collector.isMarked(old_value));
    try std.testing.expect(try collector.isMarked(new_value));
}

fn waitFlag(flag: *const std.atomic.Value(bool)) !void {
    for (0..1_000_000) |_| {
        if (flag.load(.acquire)) return;
        std.atomic.spinLoopHint();
        std.Thread.yield() catch {};
    }
    return error.Timeout;
}

fn waitCounter(counter: *const std.atomic.Value(u32), expected: u32) !void {
    for (0..2_000_000) |attempt| {
        if (counter.load(.acquire) >= expected) return;
        std.atomic.spinLoopHint();
        if ((attempt & 0xff) == 0) std.Thread.yield() catch {};
    }
    return error.Timeout;
}

fn waitHandshake(handshake: *const runtime_thread_registry.Handshake, context: *const runtime_thread_registry.ThreadContext) !void {
    for (0..1_000_000) |_| {
        if (handshake.isReady(context)) return;
        std.atomic.spinLoopHint();
        std.Thread.yield() catch {};
    }
    return error.Timeout;
}

fn containsHandle(handles: []const runtime_value.Handle, expected: runtime_value.Handle) bool {
    for (handles) |handle| if (@as(u64, @bitCast(handle)) == @as(u64, @bitCast(expected))) return true;
    return false;
}

test "x64 guarded loop refreshes one hoisted address after relocation epoch change" {
    if (builtin.cpu.arch != .x86_64) return error.SkipZigTest;
    var from_space: [128]u8 align(runtime_value.object_alignment) = @splat(0);
    var to_space: [128]u8 align(runtime_value.object_alignment) = @splat(0);
    std.mem.writeInt(i32, from_space[16..20], 41, .little);
    std.mem.writeInt(i32, to_space[32..36], 99, .little);
    const regions = [_]runtime_value.Region{
        try runtime_value.Region.fromSlice(&from_space),
        try runtime_value.Region.fromSlice(&to_space),
    };
    var handles = try runtime_value.HandleTable.init(std.testing.allocator, 4, &regions);
    defer handles.deinit();
    const handle = try handles.reserve(0, 0);
    try handles.publish(handle, 0, @ptrCast(&from_space[8]));

    var registry = try runtime_thread_registry.Registry.init(std.testing.allocator, std.testing.io, 1);
    defer registry.deinit() catch unreachable;
    var context = try runtime_thread_registry.ThreadContext.init(std.testing.allocator, 4);
    defer context.deinit();
    try registry.register(&context);
    defer registry.unregister(&context) catch unreachable;
    var runtime = try runtime_jit.Runtime.init(std.testing.allocator, &handles, &registry);
    defer runtime.deinit() catch unreachable;

    var encoded_shims = try encode(std.testing.allocator, @intFromPtr(&runtime_jit.slowResolveBridge));
    defer encoded_shims.deinit();
    var cache = jit_memory.Cache.init(std.testing.allocator);
    defer cache.deinit();
    const entry_code = try cache.addBytes(encoded_shims.entry);
    const helper_code = try cache.addBytes(encoded_shims.slow_helper);

    const insts = [_]Instruction{
        .{ .const_ = .{ .dest = 1, .value = 2 } },
        .{ .goto_ = .{ .offset = 1 } },
        .{ .iget = .{ .field_idx = 1, .dest_or_src = 2, .obj = 0 } },
        .{ .if_eqz = .{ .src = 1, .offset = 3 } },
        .{ .add_int_lit8 = .{ .dest = 1, .src = 1, .lit = -1 } },
        .{ .goto_ = .{ .offset = -3 } },
        .{ .return_ = .{ .src = 2 } },
    };
    var optimized = try optimizer.optimize(std.testing.allocator, &insts, &.{}, .{});
    defer optimized.deinit();
    try std.testing.expectEqual(@as(u32, 1), optimized.machine.stats.resolves);
    try std.testing.expectEqual(@as(u32, 1), optimized.machine.stats.loop_epoch_guards);

    const layouts = [_]x64_encoder.FieldLayout{
        .{ .offset = 0, .storage = .i32 },
        .{ .offset = 8, .storage = .i32 },
    };
    var native = try x64_encoder.encodeWithOptions(std.testing.allocator, &optimized.machine, .{
        .runtime = .{
            .handle_capacity = handles.entryCapacity(),
            .region_count = handles.regionCount(),
            .slow_resolve_helper = helper_code.entryAddress(),
            .field_layouts = &layouts,
        },
    });
    defer native.deinit();
    try native.verify();
    try std.testing.expectEqual(@as(u32, 1), native.stats.loop_epoch_guards);
    try std.testing.expectEqual(@as(u32, 2), native.stats.root_map_sites);
    const maps = if (native.root_maps) |*table| table else return error.TestUnexpectedResult;
    const target_bytes = try native.finalize();
    defer std.testing.allocator.free(target_bytes);
    const target_code = try cache.addBytes(target_bytes);

    const Worker = struct {
        runtime: *runtime_jit.Runtime,
        context: *runtime_thread_registry.ThreadContext,
        maps: *const runtime_stack_map.Table,
        entry_address: usize,
        target_address: usize,
        handle: runtime_value.Handle,
        ready: *std.atomic.Value(bool),
        done: *std.atomic.Value(bool),
        failed: *std.atomic.Value(bool),
        result: *std.atomic.Value(usize),

        fn fail(self: *@This()) void {
            self.failed.store(true, .release);
            self.ready.store(true, .release);
            self.done.store(true, .release);
        }

        fn run(self: *@This()) void {
            var managed = self.runtime.enter(self.context) catch {
                self.fail();
                return;
            };
            defer managed.deinit();
            managed.installRootMaps(self.maps) catch {
                self.fail();
                return;
            };
            var frame = CallFrame{
                .image = managed.registerImage() catch {
                    self.fail();
                    return;
                },
                .target = self.target_address,
            };
            frame.gp_args[0] = @bitCast(self.handle);
            const initial_epoch = managed.acknowledged_epoch;
            self.ready.store(true, .release);
            while (self.runtime.registry.requestEpoch() == initial_epoch) std.atomic.spinLoopHint();
            const EntryFn = fn (*const CallFrame) callconv(.c) usize;
            const call: *const EntryFn = @ptrFromInt(self.entry_address);
            self.result.store(call(&frame), .release);
            self.done.store(true, .release);
        }
    };

    var ready = std.atomic.Value(bool).init(false);
    var done = std.atomic.Value(bool).init(false);
    var failed = std.atomic.Value(bool).init(false);
    var result = std.atomic.Value(usize).init(0);
    var worker = Worker{
        .runtime = &runtime,
        .context = &context,
        .maps = maps,
        .entry_address = entry_code.entryAddress(),
        .target_address = target_code.entryAddress(),
        .handle = handle,
        .ready = &ready,
        .done = &done,
        .failed = &failed,
        .result = &result,
    };
    const thread = try std.Thread.spawn(.{}, Worker.run, .{&worker});
    try waitFlag(&ready);
    try std.testing.expect(!failed.load(.acquire));

    var member_storage: [1]*runtime_thread_registry.ThreadContext = undefined;
    var handshake = try registry.beginHandshake(&member_storage);
    try waitHandshake(&handshake, &context);
    const snapshot = try handshake.snapshot(&context);
    try std.testing.expect(containsHandle(snapshot, handle));
    const relocation = try handles.beginRelocation(handle);
    try std.testing.expect(try handles.commitRelocation(relocation, 1, @ptrCast(&to_space[24])));
    try handshake.release(&context);
    try handshake.finish();
    thread.join();

    try std.testing.expect(done.load(.acquire));
    try std.testing.expect(!failed.load(.acquire));
    try std.testing.expectEqual(@as(usize, 99), result.load(.acquire));
    try std.testing.expectEqual(@as(u64, 1), runtime.stats().handshake_polls);
    try std.testing.expectEqual(@as(u64, 1), runtime.stats().slow_resolves);
}

test "x64 bounds exception edge publishes roots during a concurrent handshake" {
    if (builtin.cpu.arch != .x86_64) return error.SkipZigTest;
    var storage: [128]u8 align(runtime_value.object_alignment) = @splat(0);
    const regions = [_]runtime_value.Region{try runtime_value.Region.fromSlice(&storage)};
    var handles = try runtime_value.HandleTable.init(std.testing.allocator, 2, &regions);
    defer handles.deinit();
    const array = try handles.reserve(0, 0);
    try handles.publish(array, 0, @ptrCast(&storage[16]));
    std.mem.writeInt(u32, storage[16..20], 1, .little);
    std.mem.writeInt(u64, storage[24..32], @bitCast(runtime_value.Handle.none), .little);

    var registry = try runtime_thread_registry.Registry.init(std.testing.allocator, std.testing.io, 1);
    defer registry.deinit() catch unreachable;
    var context = try runtime_thread_registry.ThreadContext.init(std.testing.allocator, 4);
    defer context.deinit();
    try registry.register(&context);
    defer registry.unregister(&context) catch unreachable;
    var runtime = try runtime_jit.Runtime.init(std.testing.allocator, &handles, &registry);
    defer runtime.deinit() catch unreachable;

    var encoded_shims = try encode(std.testing.allocator, @intFromPtr(&runtime_jit.slowResolveBridge));
    defer encoded_shims.deinit();
    var cache = jit_memory.Cache.init(std.testing.allocator);
    defer cache.deinit();
    const entry_code = try cache.addBytes(encoded_shims.entry);
    const slow_code = try cache.addBytes(encoded_shims.slow_helper);
    const bounds_code = try cache.addBytes(encoded_shims.bounds_exception_helper);

    const insts = [_]Instruction{
        .{ .aget_object = .{ .dest_or_src = 2, .array = 0, .index = 1 } },
        .{ .return_object = .{ .src = 2 } },
    };
    var optimized = try optimizer.optimize(std.testing.allocator, &insts, &.{}, .{});
    defer optimized.deinit();
    var native = try x64_encoder.encodeWithOptions(std.testing.allocator, &optimized.machine, .{
        .runtime = .{
            .handle_capacity = handles.entryCapacity(),
            .region_count = handles.regionCount(),
            .slow_resolve_helper = slow_code.entryAddress(),
            .bounds_exception_helper = bounds_code.entryAddress(),
            .reference_array_layout = .{ .length_offset = 0, .data_offset = 8 },
            .field_layouts = &.{},
        },
    });
    defer native.deinit();
    const target_bytes = try native.finalize();
    defer std.testing.allocator.free(target_bytes);
    const target_code = try cache.addBytes(target_bytes);
    const maps = if (native.root_maps) |*table| table else return error.TestUnexpectedResult;

    const Worker = struct {
        runtime: *runtime_jit.Runtime,
        context: *runtime_thread_registry.ThreadContext,
        maps: *const runtime_stack_map.Table,
        entry_address: usize,
        target_address: usize,
        array: runtime_value.Handle,
        ready: *std.atomic.Value(bool),
        done: *std.atomic.Value(bool),
        failed: *std.atomic.Value(bool),
        result: *std.atomic.Value(usize),
        exception_index: *std.atomic.Value(i32),
        exception_length: *std.atomic.Value(u32),

        fn fail(self: *@This()) void {
            self.failed.store(true, .release);
            self.ready.store(true, .release);
            self.done.store(true, .release);
        }

        fn run(self: *@This()) void {
            var managed = self.runtime.enter(self.context) catch {
                self.fail();
                return;
            };
            defer managed.deinit();
            managed.installRootMaps(self.maps) catch {
                self.fail();
                return;
            };
            var frame = CallFrame{
                .image = managed.registerImage() catch {
                    self.fail();
                    return;
                },
                .target = self.target_address,
            };
            frame.gp_args[0] = @bitCast(self.array);
            frame.gp_args[1] = 1;
            const initial_epoch = managed.acknowledged_epoch;
            self.ready.store(true, .release);
            while (self.runtime.registry.requestEpoch() == initial_epoch) std.atomic.spinLoopHint();
            const EntryFn = fn (*const CallFrame) callconv(.c) usize;
            const call: *const EntryFn = @ptrFromInt(self.entry_address);
            self.result.store(call(&frame), .release);
            const exception = managed.takeException() orelse {
                self.fail();
                return;
            };
            if (exception.kind != .array_index_out_of_bounds) {
                self.fail();
                return;
            }
            self.exception_index.store(exception.index, .release);
            self.exception_length.store(exception.length, .release);
            self.done.store(true, .release);
        }
    };

    var ready = std.atomic.Value(bool).init(false);
    var done = std.atomic.Value(bool).init(false);
    var failed = std.atomic.Value(bool).init(false);
    var result = std.atomic.Value(usize).init(std.math.maxInt(usize));
    var exception_index = std.atomic.Value(i32).init(0);
    var exception_length = std.atomic.Value(u32).init(0);
    var worker = Worker{
        .runtime = &runtime,
        .context = &context,
        .maps = maps,
        .entry_address = entry_code.entryAddress(),
        .target_address = target_code.entryAddress(),
        .array = array,
        .ready = &ready,
        .done = &done,
        .failed = &failed,
        .result = &result,
        .exception_index = &exception_index,
        .exception_length = &exception_length,
    };
    const thread = try std.Thread.spawn(.{}, Worker.run, .{&worker});
    try waitFlag(&ready);
    try std.testing.expect(!failed.load(.acquire));

    var member_storage: [1]*runtime_thread_registry.ThreadContext = undefined;
    var handshake = try registry.beginHandshake(&member_storage);
    try waitHandshake(&handshake, &context);
    try std.testing.expect(containsHandle(try handshake.snapshot(&context), array));
    try handshake.release(&context);
    try handshake.finish();
    thread.join();

    try std.testing.expect(done.load(.acquire));
    try std.testing.expect(!failed.load(.acquire));
    try std.testing.expectEqual(@as(usize, 0), result.load(.acquire));
    try std.testing.expectEqual(@as(i32, 1), exception_index.load(.acquire));
    try std.testing.expectEqual(@as(u32, 1), exception_length.load(.acquire));
    try std.testing.expectEqual(@as(u64, 1), runtime.stats().handshake_polls);
    try std.testing.expectEqual(@as(u64, 1), runtime.stats().exception_transfers);
}

test "x64 polling helper publishes every mapped register root" {
    if (builtin.cpu.arch != .x86_64) return error.SkipZigTest;
    var from_space: [192]u8 align(runtime_value.object_alignment) = @splat(0);
    var to_space: [128]u8 align(runtime_value.object_alignment) = @splat(0);
    std.mem.writeInt(i32, from_space[16..20], 41, .little);
    std.mem.writeInt(i32, from_space[56..60], 77, .little);
    const regions = [_]runtime_value.Region{
        try runtime_value.Region.fromSlice(&from_space),
        try runtime_value.Region.fromSlice(&to_space),
    };
    var handles = try runtime_value.HandleTable.init(std.testing.allocator, 4, &regions);
    defer handles.deinit();
    const first_handle = try handles.reserve(0, 0);
    const second_handle = try handles.reserve(0, 0);
    try handles.publish(first_handle, 0, @ptrCast(&from_space[8]));
    try handles.publish(second_handle, 0, @ptrCast(&from_space[48]));
    const relocation = try handles.beginRelocation(first_handle);

    var registry = try runtime_thread_registry.Registry.init(std.testing.allocator, std.testing.io, 1);
    defer registry.deinit() catch unreachable;
    var context = try runtime_thread_registry.ThreadContext.init(std.testing.allocator, 4);
    defer context.deinit();
    try registry.register(&context);
    defer registry.unregister(&context) catch unreachable;
    var runtime = try runtime_jit.Runtime.init(std.testing.allocator, &handles, &registry);
    defer runtime.deinit() catch unreachable;

    var encoded_shims = try encode(std.testing.allocator, @intFromPtr(&runtime_jit.slowResolveBridge));
    defer encoded_shims.deinit();
    var cache = jit_memory.Cache.init(std.testing.allocator);
    defer cache.deinit();
    const entry_code = try cache.addBytes(encoded_shims.entry);
    const helper_code = try cache.addBytes(encoded_shims.slow_helper);

    const insts = [_]Instruction{
        .{ .iget = .{ .field_idx = 1, .dest_or_src = 2, .obj = 0 } },
        .{ .iget = .{ .field_idx = 1, .dest_or_src = 3, .obj = 1 } },
        .{ .return_ = .{ .src = 3 } },
    };
    var optimized = try optimizer.optimize(std.testing.allocator, &insts, &.{}, .{});
    defer optimized.deinit();
    const layouts = [_]x64_encoder.FieldLayout{
        .{ .offset = 0, .storage = .i32 },
        .{ .offset = 8, .storage = .i32 },
    };
    var native = try x64_encoder.encodeWithOptions(std.testing.allocator, &optimized.machine, .{
        .runtime = .{
            .handle_capacity = handles.entryCapacity(),
            .region_count = handles.regionCount(),
            .slow_resolve_helper = helper_code.entryAddress(),
            .field_layouts = &layouts,
        },
    });
    defer native.deinit();
    const first_resolve = optimized.machine.blocks[0].insts[1];
    const maps = if (native.root_maps) |*table| table else return error.TestUnexpectedResult;
    const map_record = try maps.find(first_resolve.resolve_id orelse return error.TestUnexpectedResult);
    try std.testing.expectEqual(@as(usize, 2), maps.rootsFor(map_record).len);
    const target_bytes = try native.finalize();
    defer std.testing.allocator.free(target_bytes);
    const target_code = try cache.addBytes(target_bytes);

    const Worker = struct {
        runtime: *runtime_jit.Runtime,
        context: *runtime_thread_registry.ThreadContext,
        maps: *const runtime_stack_map.Table,
        entry_address: usize,
        target_address: usize,
        first: runtime_value.Handle,
        second: runtime_value.Handle,
        ready: *std.atomic.Value(bool),
        done: *std.atomic.Value(bool),
        result: *std.atomic.Value(usize),

        fn run(self: *@This()) void {
            var managed = self.runtime.enter(self.context) catch return;
            defer managed.deinit();
            managed.installRootMaps(self.maps) catch return;
            var frame = CallFrame{
                .image = managed.registerImage() catch return,
                .target = self.target_address,
            };
            frame.gp_args[0] = @bitCast(self.first);
            frame.gp_args[1] = @bitCast(self.second);
            self.ready.store(true, .release);
            while (self.runtime.registry.requestEpoch() == managed.acknowledged_epoch) std.atomic.spinLoopHint();
            const EntryFn = fn (*const CallFrame) callconv(.c) usize;
            const call: *const EntryFn = @ptrFromInt(self.entry_address);
            self.result.store(call(&frame), .release);
            self.done.store(true, .release);
        }
    };

    var ready = std.atomic.Value(bool).init(false);
    var done = std.atomic.Value(bool).init(false);
    var result = std.atomic.Value(usize).init(0);
    var worker = Worker{
        .runtime = &runtime,
        .context = &context,
        .maps = maps,
        .entry_address = entry_code.entryAddress(),
        .target_address = target_code.entryAddress(),
        .first = first_handle,
        .second = second_handle,
        .ready = &ready,
        .done = &done,
        .result = &result,
    };
    const thread = try std.Thread.spawn(.{}, Worker.run, .{&worker});
    try waitFlag(&ready);

    var member_storage: [1]*runtime_thread_registry.ThreadContext = undefined;
    var handshake = try registry.beginHandshake(&member_storage);
    try waitHandshake(&handshake, &context);
    const snapshot = try handshake.snapshot(&context);
    const root_count = snapshot.len;
    const contains_first = containsHandle(snapshot, first_handle);
    const contains_second = containsHandle(snapshot, second_handle);
    try handshake.release(&context);
    try handshake.finish();
    thread.join();
    try std.testing.expectEqual(@as(usize, 2), root_count);
    try std.testing.expect(contains_first);
    try std.testing.expect(contains_second);
    try std.testing.expect(done.load(.acquire));
    try std.testing.expectEqual(@as(usize, 77), result.load(.acquire));
    try std.testing.expectEqual(@as(u64, 1), runtime.stats().handshake_polls);
    try std.testing.expect(try handles.commitRelocation(relocation, 1, @ptrCast(&to_space[8])));
}

test "x64 generated worker crosses evacuation epoch before from-space reuse" {
    if (builtin.cpu.arch != .x86_64) return error.SkipZigTest;
    var from_space: [128]u8 align(runtime_value.object_alignment) = @splat(0);
    var to_space: [128]u8 align(runtime_value.object_alignment) = @splat(0);
    var poll_space: [128]u8 align(runtime_value.object_alignment) = @splat(0);
    var poll_to_space: [128]u8 align(runtime_value.object_alignment) = @splat(0);
    std.mem.writeInt(i32, from_space[16..20], 41, .little);
    std.mem.writeInt(i32, to_space[32..36], 99, .little);
    std.mem.writeInt(i32, poll_space[16..20], 7, .little);
    std.mem.writeInt(i32, poll_to_space[16..20], 17, .little);
    const regions = [_]runtime_value.Region{
        try runtime_value.Region.fromSlice(&from_space),
        try runtime_value.Region.fromSlice(&to_space),
        try runtime_value.Region.fromSlice(&poll_space),
        try runtime_value.Region.fromSlice(&poll_to_space),
    };
    var handles = try runtime_value.HandleTable.init(std.testing.allocator, 4, &regions);
    defer handles.deinit();
    const target_handle = try handles.reserve(0, 0);
    const poll_handle = try handles.reserve(0, 0);
    try handles.publish(target_handle, 0, @ptrCast(&from_space[8]));
    try handles.publish(poll_handle, 2, @ptrCast(&poll_space[8]));

    var registry = try runtime_thread_registry.Registry.init(std.testing.allocator, std.testing.io, 1);
    defer registry.deinit() catch unreachable;
    var context = try runtime_thread_registry.ThreadContext.init(std.testing.allocator, 4);
    defer context.deinit();
    try registry.register(&context);
    defer registry.unregister(&context) catch unreachable;
    var runtime = try runtime_jit.Runtime.init(std.testing.allocator, &handles, &registry);
    defer runtime.deinit() catch unreachable;
    var coordinator = try runtime_heap.EvacuationCoordinator.init(std.testing.allocator, &handles, &registry, 1);
    defer coordinator.deinit() catch unreachable;

    var encoded_shims = try encode(std.testing.allocator, @intFromPtr(&runtime_jit.slowResolveBridge));
    defer encoded_shims.deinit();
    var cache = jit_memory.Cache.init(std.testing.allocator);
    defer cache.deinit();
    const entry_code = try cache.addBytes(encoded_shims.entry);
    const helper_code = try cache.addBytes(encoded_shims.slow_helper);

    // The first access is deliberately evacuating and therefore reaches the
    // mapped polling helper. The relocated target remains live in the same map
    // and is accessed after the retirement handshake resumes this worker.
    const insts = [_]Instruction{
        .{ .iget = .{ .field_idx = 1, .dest_or_src = 2, .obj = 0 } },
        .{ .iget = .{ .field_idx = 1, .dest_or_src = 3, .obj = 1 } },
        .{ .return_ = .{ .src = 3 } },
    };
    var optimized = try optimizer.optimize(std.testing.allocator, &insts, &.{}, .{});
    defer optimized.deinit();
    const layouts = [_]x64_encoder.FieldLayout{
        .{ .offset = 0, .storage = .i32 },
        .{ .offset = 8, .storage = .i32 },
    };
    var native = try x64_encoder.encodeWithOptions(std.testing.allocator, &optimized.machine, .{
        .runtime = .{
            .handle_capacity = handles.entryCapacity(),
            .region_count = handles.regionCount(),
            .slow_resolve_helper = helper_code.entryAddress(),
            .field_layouts = &layouts,
        },
    });
    defer native.deinit();
    const maps = if (native.root_maps) |*table| table else return error.TestUnexpectedResult;
    const target_bytes = try native.finalize();
    defer std.testing.allocator.free(target_bytes);
    const target_code = try cache.addBytes(target_bytes);

    const Worker = struct {
        runtime: *runtime_jit.Runtime,
        context: *runtime_thread_registry.ThreadContext,
        maps: *const runtime_stack_map.Table,
        entry_address: usize,
        target_address: usize,
        poll_handle: runtime_value.Handle,
        target_handle: runtime_value.Handle,
        initial_epoch: u64,
        ready: *std.atomic.Value(bool),
        first_done: *std.atomic.Value(bool),
        allow_second: *std.atomic.Value(bool),
        done: *std.atomic.Value(bool),
        failed: *std.atomic.Value(bool),
        first_result: *std.atomic.Value(usize),
        second_result: *std.atomic.Value(usize),
        refreshed_epoch: *std.atomic.Value(u64),

        fn fail(self: *@This()) void {
            self.failed.store(true, .release);
        }

        fn run(self: *@This()) void {
            var managed = self.runtime.enter(self.context) catch {
                self.fail();
                return;
            };
            defer managed.deinit();
            managed.installRootMaps(self.maps) catch {
                self.fail();
                return;
            };
            var frame = CallFrame{
                .image = managed.registerImage() catch {
                    self.fail();
                    return;
                },
                .target = self.target_address,
            };
            frame.gp_args[0] = @bitCast(self.poll_handle);
            frame.gp_args[1] = @bitCast(self.target_handle);
            self.ready.store(true, .release);
            while (self.runtime.registry.requestEpoch() == self.initial_epoch) std.atomic.spinLoopHint();

            const EntryFn = fn (*const CallFrame) callconv(.c) usize;
            const call: *const EntryFn = @ptrFromInt(self.entry_address);
            self.first_result.store(call(&frame), .release);
            self.first_done.store(true, .release);
            while (!self.allow_second.load(.acquire)) std.atomic.spinLoopHint();

            frame.image = managed.registerImage() catch {
                self.fail();
                return;
            };
            self.refreshed_epoch.store(frame.image.r12_acknowledged_epoch, .release);
            self.second_result.store(call(&frame), .release);
            self.done.store(true, .release);
        }
    };

    var ready = std.atomic.Value(bool).init(false);
    var first_done = std.atomic.Value(bool).init(false);
    var allow_second = std.atomic.Value(bool).init(false);
    var done = std.atomic.Value(bool).init(false);
    var failed = std.atomic.Value(bool).init(false);
    var first_result = std.atomic.Value(usize).init(0);
    var second_result = std.atomic.Value(usize).init(0);
    var refreshed_epoch = std.atomic.Value(u64).init(0);
    var worker = Worker{
        .runtime = &runtime,
        .context = &context,
        .maps = maps,
        .entry_address = entry_code.entryAddress(),
        .target_address = target_code.entryAddress(),
        .poll_handle = poll_handle,
        .target_handle = target_handle,
        .initial_epoch = registry.requestEpoch(),
        .ready = &ready,
        .first_done = &first_done,
        .allow_second = &allow_second,
        .done = &done,
        .failed = &failed,
        .first_result = &first_result,
        .second_result = &second_result,
        .refreshed_epoch = &refreshed_epoch,
    };
    const thread = try std.Thread.spawn(.{}, Worker.run, .{&worker});
    try waitFlag(&ready);

    var cycle = try coordinator.begin(0);
    const target_ticket = try handles.beginRelocation(target_handle);
    try std.testing.expect(try cycle.commitRelocation(target_ticket, 1, @ptrCast(&to_space[24])));
    const poll_ticket = try handles.beginRelocation(poll_handle);
    const epoch = try cycle.requestRetirement();

    for (0..1_000_000) |_| {
        if (cycle.isReady(&context)) break;
        std.atomic.spinLoopHint();
        std.Thread.yield() catch {};
    } else return error.Timeout;
    const snapshot = try cycle.snapshot(&context);
    const root_count = snapshot.len;
    const contains_poll = containsHandle(snapshot, poll_handle);
    const contains_target = containsHandle(snapshot, target_handle);
    try std.testing.expect(try cycle.advance());
    try waitFlag(&first_done);

    const reclaimed = try cycle.reclaim();
    try std.testing.expectEqual(epoch, reclaimed.retirement_epoch);
    @memset(from_space[0..], 0xa5);
    try handles.activateRegionAfterReset(reclaimed.region_id);
    allow_second.store(true, .release);
    thread.join();

    try std.testing.expect(try handles.commitRelocation(poll_ticket, 3, @ptrCast(&poll_to_space[8])));
    try std.testing.expectEqual(@as(usize, 2), root_count);
    try std.testing.expect(contains_poll);
    try std.testing.expect(contains_target);
    try std.testing.expect(!failed.load(.acquire));
    try std.testing.expect(done.load(.acquire));
    try std.testing.expectEqual(@as(usize, 99), first_result.load(.acquire));
    try std.testing.expectEqual(@as(usize, 99), second_result.load(.acquire));
    try std.testing.expectEqual(epoch, refreshed_epoch.load(.acquire));
    try std.testing.expectEqual(runtime_value.RegionState.active, try handles.regionState(0));
    try std.testing.expectEqual(@as(u64, 1), runtime.stats().handshake_polls);
    try std.testing.expectEqual(@as(u64, 2), runtime.stats().slow_resolves);
}

test "x64 seeded relocation differential preserves progress and replayability" {
    if (builtin.cpu.arch != .x86_64) return error.SkipZigTest;
    const handle_count = 6;
    const object_stride = 64;
    const object_base = 16;
    const seeds = [_]u64{
        0x243f_6a88_85a3_08d3,
        0x1319_8a2e_0370_7344,
        0xa409_3822_299f_31d0,
        0x082e_fa98_ec4e_6c89,
    };
    const rounds_per_seed: u32 = if (builtin.mode == .Debug) 16 else 64;
    const total_rounds: u32 = rounds_per_seed * @as(u32, @intCast(seeds.len));

    var region_a: [512]u8 align(runtime_value.object_alignment) = @splat(0);
    var region_b: [512]u8 align(runtime_value.object_alignment) = @splat(0);
    const regions = [_]runtime_value.Region{
        try runtime_value.Region.fromSlice(&region_a),
        try runtime_value.Region.fromSlice(&region_b),
    };
    var table = try runtime_value.HandleTable.init(std.testing.allocator, 8, &regions);
    defer table.deinit();
    var object_handles: [handle_count]runtime_value.Handle = undefined;
    var current_regions: [handle_count]u8 = @splat(0);
    for (&object_handles, 0..) |*handle, index| {
        handle.* = try table.reserve(0, 0);
        const offset = object_base + index * object_stride;
        try table.publish(handle.*, 0, @ptrCast(&region_a[offset]));
    }

    var registry = try runtime_thread_registry.Registry.init(std.testing.allocator, std.testing.io, 1);
    defer registry.deinit() catch unreachable;
    var context = try runtime_thread_registry.ThreadContext.init(std.testing.allocator, 4);
    defer context.deinit();
    try registry.register(&context);
    defer registry.unregister(&context) catch unreachable;
    var runtime = try runtime_jit.Runtime.init(std.testing.allocator, &table, &registry);
    defer runtime.deinit() catch unreachable;

    var encoded_shims = try encode(std.testing.allocator, @intFromPtr(&runtime_jit.slowResolveBridge));
    defer encoded_shims.deinit();
    var cache = jit_memory.Cache.init(std.testing.allocator);
    defer cache.deinit();
    const entry_code = try cache.addBytes(encoded_shims.entry);
    const slow_code = try cache.addBytes(encoded_shims.slow_helper);
    const insts = [_]Instruction{
        .{ .iget = .{ .field_idx = 0, .dest_or_src = 1, .obj = 0 } },
        .{ .return_ = .{ .src = 1 } },
    };
    var optimized = try optimizer.optimize(std.testing.allocator, &insts, &.{}, .{});
    defer optimized.deinit();
    const layouts = [_]x64_encoder.FieldLayout{.{ .offset = 8, .storage = .i32 }};
    var native = try x64_encoder.encodeWithOptions(std.testing.allocator, &optimized.machine, .{
        .runtime = .{
            .handle_capacity = table.entryCapacity(),
            .region_count = table.regionCount(),
            .slow_resolve_helper = slow_code.entryAddress(),
            .field_layouts = &layouts,
        },
    });
    defer native.deinit();
    const maps = if (native.root_maps) |*root_maps| root_maps else return error.TestUnexpectedResult;
    const target_bytes = try native.finalize();
    defer std.testing.allocator.free(target_bytes);
    const target_code = try cache.addBytes(target_bytes);

    const Worker = struct {
        runtime: *runtime_jit.Runtime,
        context: *runtime_thread_registry.ThreadContext,
        maps: *const runtime_stack_map.Table,
        entry_address: usize,
        target_address: usize,
        total_rounds: u32,
        requested: *std.atomic.Value(u32),
        completed: *std.atomic.Value(u32),
        command_handle: *std.atomic.Value(u64),
        result: *std.atomic.Value(usize),
        ready: *std.atomic.Value(bool),
        failed: *std.atomic.Value(bool),

        fn fail(self: *@This()) void {
            self.failed.store(true, .release);
            self.ready.store(true, .release);
            self.completed.store(std.math.maxInt(u32), .release);
        }

        fn run(self: *@This()) void {
            var managed = self.runtime.enter(self.context) catch {
                self.fail();
                return;
            };
            defer managed.deinit();
            managed.installRootMaps(self.maps) catch {
                self.fail();
                return;
            };
            var frame = CallFrame{
                .image = managed.registerImage() catch {
                    self.fail();
                    return;
                },
                .target = self.target_address,
            };
            const EntryFn = fn (*const CallFrame) callconv(.c) usize;
            const call: *const EntryFn = @ptrFromInt(self.entry_address);
            self.ready.store(true, .release);
            var round: u32 = 1;
            while (round <= self.total_rounds) : (round += 1) {
                waitCounter(self.requested, round) catch {
                    self.fail();
                    return;
                };
                frame.image = managed.registerImage() catch {
                    self.fail();
                    return;
                };
                frame.gp_args[0] = self.command_handle.load(.acquire);
                self.result.store(call(&frame), .release);
                self.completed.store(round, .release);
            }
        }
    };

    var requested = std.atomic.Value(u32).init(0);
    var completed = std.atomic.Value(u32).init(0);
    var command_handle = std.atomic.Value(u64).init(0);
    var native_result = std.atomic.Value(usize).init(0);
    var ready = std.atomic.Value(bool).init(false);
    var worker_failed = std.atomic.Value(bool).init(false);
    var worker = Worker{
        .runtime = &runtime,
        .context = &context,
        .maps = maps,
        .entry_address = entry_code.entryAddress(),
        .target_address = target_code.entryAddress(),
        .total_rounds = total_rounds,
        .requested = &requested,
        .completed = &completed,
        .command_handle = &command_handle,
        .result = &native_result,
        .ready = &ready,
        .failed = &worker_failed,
    };
    const thread = try std.Thread.spawn(.{}, Worker.run, .{&worker});
    try waitFlag(&ready);
    try std.testing.expect(!worker_failed.load(.acquire));

    const ReplayEvent = struct {
        seed: u64,
        round: u32,
        handle_index: u8,
        from_region: u8,
        to_region: u8,
        epoch: u64,
        expected: u32,
        native: u32,
        baseline: u32,
        root_seen: bool,
    };
    var replay: [16]ReplayEvent = undefined;
    var replay_count: usize = 0;
    var mismatches: u32 = 0;
    var global_round: u32 = 0;
    for (seeds) |seed| {
        var prng = std.Random.DefaultPrng.init(seed);
        const random = prng.random();
        for (0..rounds_per_seed) |local_round| {
            global_round += 1;
            const handle_index = random.uintLessThanBiased(usize, handle_count);
            const handle = object_handles[handle_index];
            const from_region = current_regions[handle_index];
            const to_region: u8 = from_region ^ 1;
            const value = random.int(u32) & @as(u32, std.math.maxInt(i31));
            const offset = object_base + handle_index * object_stride;
            const destination: *anyopaque = if (to_region == 0)
                @ptrCast(&region_a[offset])
            else
                @ptrCast(&region_b[offset]);
            if (to_region == 0) {
                const field: *[4]u8 = @ptrCast(&region_a[offset + 8]);
                std.mem.writeInt(u32, field, value, .little);
            } else {
                const field: *[4]u8 = @ptrCast(&region_b[offset + 8]);
                std.mem.writeInt(u32, field, value, .little);
            }

            const ticket = try table.beginRelocation(handle);
            var member_storage: [1]*runtime_thread_registry.ThreadContext = undefined;
            var handshake = try registry.beginHandshake(&member_storage);
            command_handle.store(@bitCast(handle), .release);
            requested.store(global_round, .release);
            try waitHandshake(&handshake, &context);
            const root_seen = containsHandle(try handshake.snapshot(&context), handle);
            try std.testing.expect(try table.commitRelocation(ticket, to_region, destination));
            current_regions[handle_index] = to_region;
            try handshake.release(&context);
            try handshake.finish();
            try waitCounter(&completed, global_round);

            const native_bits: u32 = @truncate(native_result.load(.acquire));
            const baseline_address = @intFromPtr(try table.resolve(handle));
            const baseline_bytes: *const [12]u8 = @ptrFromInt(baseline_address);
            const baseline_bits = std.mem.readInt(u32, baseline_bytes[8..12], .little);
            replay[replay_count % replay.len] = .{
                .seed = seed,
                .round = @intCast(local_round),
                .handle_index = @intCast(handle_index),
                .from_region = from_region,
                .to_region = to_region,
                .epoch = handshake.epoch,
                .expected = value,
                .native = native_bits,
                .baseline = baseline_bits,
                .root_seen = root_seen,
            };
            replay_count += 1;
            if (!root_seen or native_bits != value or baseline_bits != value or native_bits != baseline_bits) mismatches += 1;
        }
    }
    thread.join();

    if (mismatches != 0 or worker_failed.load(.acquire)) {
        std.debug.print("relocation differential mismatches={d} worker_failed={} total={d}\n", .{ mismatches, worker_failed.load(.acquire), total_rounds });
        const retained = @min(replay_count, replay.len);
        const first = replay_count - retained;
        for (0..retained) |offset| {
            const event = replay[(first + offset) % replay.len];
            std.debug.print(
                "  seed=0x{x} round={d} handle={d} region={d}->{d} epoch={d} expected={d} native={d} baseline={d} root={}\n",
                .{ event.seed, event.round, event.handle_index, event.from_region, event.to_region, event.epoch, event.expected, event.native, event.baseline, event.root_seen },
            );
        }
    }
    try std.testing.expectEqual(@as(u32, 0), mismatches);
    try std.testing.expect(!worker_failed.load(.acquire));
    try std.testing.expectEqual(total_rounds, completed.load(.acquire));
    try std.testing.expectEqual(@as(u64, total_rounds), runtime.stats().slow_resolves);
    try std.testing.expectEqual(@as(u64, total_rounds), runtime.stats().handshake_polls);
    try std.testing.expectEqual(@as(u64, 0), runtime.stats().resolve_failures);
}

fn allocationFailureProbe(allocator: std.mem.Allocator) !void {
    var shims = try encode(allocator, 1);
    defer shims.deinit();
}

test "x64 entry trampoline leases code and falls back after concurrent invalidation" {
    if (builtin.cpu.arch != .x86_64) return error.SkipZigTest;

    var storage: [64]u8 align(runtime_value.object_alignment) = @splat(0);
    const regions = [_]runtime_value.Region{try runtime_value.Region.fromSlice(&storage)};
    var handles = try runtime_value.HandleTable.init(std.testing.allocator, 1, &regions);
    defer handles.deinit();
    var registry = try runtime_thread_registry.Registry.init(std.testing.allocator, std.testing.io, 1);
    defer registry.deinit() catch unreachable;
    var context = try runtime_thread_registry.ThreadContext.init(std.testing.allocator, 2);
    defer context.deinit();
    try registry.register(&context);
    defer registry.unregister(&context) catch unreachable;

    var manager = try runtime_code_manager.Manager.init(std.testing.allocator, 1, 1, 2);
    defer manager.deinit() catch unreachable;
    var runtime = try runtime_jit.Runtime.init(std.testing.allocator, &handles, &registry);
    defer runtime.deinit() catch unreachable;
    try runtime.installCodeManager(&manager);

    var encoded_shims = try encode(std.testing.allocator, @intFromPtr(&runtime_jit.slowResolveBridge));
    defer encoded_shims.deinit();
    var stable_cache = jit_memory.Cache.init(std.testing.allocator);
    defer stable_cache.deinit();
    const entry_code = try stable_cache.addBytes(encoded_shims.entry);
    const fallback_bytes = [_]u8{ 0xb8, 77, 0, 0, 0, 0xc3 };
    const fallback_code = try stable_cache.addBytes(&fallback_bytes);

    const Gate = struct {
        entered: std.atomic.Value(bool) = std.atomic.Value(bool).init(false),
        release: std.atomic.Value(bool) = std.atomic.Value(bool).init(false),

        fn target(raw: usize, _: usize, _: usize, _: usize, _: usize, _: usize) callconv(.c) usize {
            const self: *@This() = @ptrFromInt(raw);
            self.entered.store(true, .release);
            while (!self.release.load(.acquire)) std.atomic.spinLoopHint();
            return 42;
        }
    };
    var gate = Gate{};
    const target_address = @intFromPtr(&Gate.target);
    const target_bytes = [_]u8{
        0x48,                            0xb8,
        @truncate(target_address),       @truncate(target_address >> 8),
        @truncate(target_address >> 16), @truncate(target_address >> 24),
        @truncate(target_address >> 32), @truncate(target_address >> 40),
        @truncate(target_address >> 48), @truncate(target_address >> 56),
        0xff,                            0xe0,
    };
    var candidate = try manager.prepare(&target_bytes);
    defer candidate.deinit();
    try manager.publish(0, &candidate);

    const Worker = struct {
        runtime: *runtime_jit.Runtime,
        context: *runtime_thread_registry.ThreadContext,
        entry_address: usize,
        fallback_address: usize,
        gate: *Gate,
        result: *std.atomic.Value(usize),
        failed: *std.atomic.Value(bool),

        fn run(self: *@This()) void {
            var managed_entry = self.runtime.enter(self.context) catch {
                self.failed.store(true, .release);
                return;
            };
            defer managed_entry.deinit();
            var frame = CallFrame{
                .image = managed_entry.registerImage() catch {
                    self.failed.store(true, .release);
                    return;
                },
                .target = 0,
                .fallback_target = self.fallback_address,
                .method_id = 0,
            };
            frame.gp_args[0] = @intFromPtr(self.gate);
            const EntryFn = fn (*const CallFrame) callconv(.c) usize;
            const call: *const EntryFn = @ptrFromInt(self.entry_address);
            self.result.store(call(&frame), .release);
        }
    };

    var result = std.atomic.Value(usize).init(0);
    var failed = std.atomic.Value(bool).init(false);
    var worker = Worker{
        .runtime = &runtime,
        .context = &context,
        .entry_address = entry_code.entryAddress(),
        .fallback_address = fallback_code.entryAddress(),
        .gate = &gate,
        .result = &result,
        .failed = &failed,
    };
    const thread = try std.Thread.spawn(.{}, Worker.run, .{&worker});
    try waitFlag(&gate.entered);
    try std.testing.expectEqual(@as(u64, 1), manager.stats().active_leases);
    try std.testing.expect(try manager.invalidate(0));
    try std.testing.expectEqual(@as(u32, 0), try manager.reclaim());
    gate.release.store(true, .release);
    thread.join();
    try std.testing.expect(!failed.load(.acquire));
    try std.testing.expectEqual(@as(usize, 42), result.load(.acquire));
    try std.testing.expectEqual(@as(u64, 0), manager.stats().active_leases);
    try std.testing.expectEqual(@as(u32, 1), try manager.reclaim());

    {
        var managed_entry = try runtime.enter(&context);
        defer managed_entry.deinit();
        const frame = CallFrame{
            .image = try managed_entry.registerImage(),
            .target = 0,
            .fallback_target = fallback_code.entryAddress(),
            .method_id = 0,
        };
        const EntryFn = fn (*const CallFrame) callconv(.c) usize;
        const call = entry_code.typedEntry(EntryFn);
        try std.testing.expectEqual(@as(usize, 77), call(&frame));
        try std.testing.expectEqual(runtime_jit.CodeDispatchStatus.fallback_no_code, managed_entry.native_state.last_code_dispatch);
    }
    const stats = runtime.stats();
    try std.testing.expectEqual(@as(u64, 1), stats.code_entries);
    try std.testing.expectEqual(@as(u64, 1), stats.code_fallbacks);
    try std.testing.expectEqual(@as(u64, 0), stats.code_failures);
    try manager.verify();
}

test "x64 invalidated entry reconstructs and resumes an interpreter frame" {
    if (builtin.cpu.arch != .x86_64) return error.SkipZigTest;

    var storage: [64]u8 align(runtime_value.object_alignment) = @splat(0);
    const regions = [_]runtime_value.Region{try runtime_value.Region.fromSlice(&storage)};
    var handles = try runtime_value.HandleTable.init(std.testing.allocator, 1, &regions);
    defer handles.deinit();
    var registry = try runtime_thread_registry.Registry.init(std.testing.allocator, std.testing.io, 1);
    defer registry.deinit() catch unreachable;
    var context = try runtime_thread_registry.ThreadContext.init(std.testing.allocator, 2);
    defer context.deinit();
    try registry.register(&context);
    defer registry.unregister(&context) catch unreachable;

    var manager = try runtime_code_manager.Manager.init(std.testing.allocator, 1, 1, 1);
    defer manager.deinit() catch unreachable;
    var runtime = try runtime_jit.Runtime.init(std.testing.allocator, &handles, &registry);
    defer runtime.deinit() catch unreachable;
    try runtime.installCodeManager(&manager);

    var encoded_shims = try encode(std.testing.allocator, @intFromPtr(&runtime_jit.slowResolveBridge));
    defer encoded_shims.deinit();
    var stable_cache = jit_memory.Cache.init(std.testing.allocator);
    defer stable_cache.deinit();
    const entry_code = try stable_cache.addBytes(encoded_shims.entry);

    const argument_registers = if (builtin.os.tag == .windows)
        [3]u8{ rcx, rdx, r8 }
    else
        [3]u8{ rdi, rsi, rdx };
    const values = [_]runtime_deopt.ValueSpec{
        .{ .vreg = 0, .kind = .scalar32, .source = .{ .native_register = argument_registers[0] } },
        .{ .vreg = 1, .kind = .reference, .source = .{ .native_register = argument_registers[1] } },
        .{ .vreg = 2, .kind = .scalar64, .source = .{ .native_register = argument_registers[2] } },
    };
    var deopt_table = try runtime_deopt.Table.init(std.testing.allocator, &.{.{
        .id = 5,
        .method_id = 0,
        .dex_pc = 0,
        .values = &values,
    }}, .{
        .register_count = 4,
        .native_register_count = 16,
        .max_dex_pc = 0,
    });
    defer deopt_table.deinit();

    const instructions = [_]Instruction{.{ .return_ = .{ .src = 0 } }};
    var registers: [4]u32 = @splat(0);
    var references: [4]u64 = @splat(@as(u64, @bitCast(runtime_value.Handle.none)));
    var reference_kinds: [4]bool = @splat(false);
    var reconstructed = runtime_deopt.Frame{ .execution = .{
        .pc = 0,
        .registers = &registers,
        .instructions = &instructions,
        .register_is_ref = &reference_kinds,
        .reference_registers = &references,
    } };
    var scratch: [3]u64 = undefined;
    var stack_anchor: u64 = 0;
    const ResumeContext = struct {
        called: bool = false,
        failed: bool = false,

        fn run(raw: *anyopaque, frame: *runtime_deopt.Frame) usize {
            const self: *@This() = @ptrCast(@alignCast(raw));
            self.called = true;
            const result = frontend_interpreter.execute(&frame.execution) catch {
                self.failed = true;
                return 0;
            };
            if (result.kind != .single) {
                self.failed = true;
                return 0;
            }
            return result.value32;
        }
    };
    var resume_context = ResumeContext{};
    var request = runtime_deopt.Request{
        .table = &deopt_table,
        .point_id = 5,
        .destination = .{ .frame = &reconstructed, .scratch = &scratch },
        .stack_base = @ptrCast(&stack_anchor),
        .stack_max_offset = @sizeOf(@TypeOf(stack_anchor)),
        .resume_context = &resume_context,
        .resume_fn = ResumeContext.run,
    };

    var managed_entry = try runtime.enter(&context);
    defer managed_entry.deinit();
    var frame = CallFrame{
        .image = try managed_entry.registerImage(),
        .target = 0,
        .deopt_request = @intFromPtr(&request),
        .method_id = 0,
    };
    const handle = runtime_value.Handle{ .index = 9, .generation = 2 };
    frame.gp_args[0] = 0x11223344;
    frame.gp_args[1] = @bitCast(handle);
    frame.gp_args[2] = 0x8877665544332211;
    const EntryFn = fn (*const CallFrame) callconv(.c) usize;
    const result = entry_code.typedEntry(EntryFn)(&frame);

    try std.testing.expectEqual(@as(usize, 0x11223344), result);
    try std.testing.expect(resume_context.called);
    try std.testing.expect(!resume_context.failed);
    try std.testing.expect(reconstructed.active);
    try std.testing.expectEqual(@as(u32, 0), reconstructed.method_id);
    try std.testing.expectEqual(@as(u64, @bitCast(handle)), reconstructed.execution.reference_registers[1]);
    try std.testing.expect(reconstructed.execution.register_is_ref[1]);
    try std.testing.expectEqual(@as(u64, 0x8877665544332211), reconstructed.execution.getWide(2));
    try std.testing.expectEqual(runtime_jit.CodeDispatchStatus.deoptimized, managed_entry.native_state.last_code_dispatch);
    try std.testing.expectEqual(@as(u64, 1), runtime.stats().deoptimizations);
    try std.testing.expectEqual(@as(u64, 0), runtime.stats().deopt_failures);
    try std.testing.expectEqual(@as(u64, 0), manager.stats().active_leases);

    // A non-null generation-zero value is not a Handle. Reconstruction must
    // fail transactionally and must never invoke the interpreter callback.
    resume_context.called = false;
    reconstructed.active = false;
    frame.image = try managed_entry.registerImage();
    frame.gp_args[1] = 1;
    try std.testing.expectEqual(@as(usize, 0), entry_code.typedEntry(EntryFn)(&frame));
    try std.testing.expect(!resume_context.called);
    try std.testing.expect(!reconstructed.active);
    try std.testing.expectEqual(runtime_jit.CodeDispatchStatus.deopt_failed, managed_entry.native_state.last_code_dispatch);
    try std.testing.expectEqual(@as(u64, 1), runtime.stats().deopt_failures);
    try std.testing.expectEqual(@as(u64, 1), runtime.stats().code_failures);
}

test "x64 deoptimization entry publishes captured references during handshake" {
    if (builtin.cpu.arch != .x86_64) return error.SkipZigTest;

    var storage: [64]u8 align(runtime_value.object_alignment) = @splat(0);
    const regions = [_]runtime_value.Region{try runtime_value.Region.fromSlice(&storage)};
    var handles = try runtime_value.HandleTable.init(std.testing.allocator, 1, &regions);
    defer handles.deinit();
    var registry = try runtime_thread_registry.Registry.init(std.testing.allocator, std.testing.io, 1);
    defer registry.deinit() catch unreachable;
    var context = try runtime_thread_registry.ThreadContext.init(std.testing.allocator, 2);
    defer context.deinit();
    try registry.register(&context);
    defer registry.unregister(&context) catch unreachable;
    var manager = try runtime_code_manager.Manager.init(std.testing.allocator, 1, 1, 1);
    defer manager.deinit() catch unreachable;
    var runtime = try runtime_jit.Runtime.init(std.testing.allocator, &handles, &registry);
    defer runtime.deinit() catch unreachable;
    try runtime.installCodeManager(&manager);

    var encoded_shims = try encode(std.testing.allocator, @intFromPtr(&runtime_jit.slowResolveBridge));
    defer encoded_shims.deinit();
    var stable_cache = jit_memory.Cache.init(std.testing.allocator);
    defer stable_cache.deinit();
    const entry_code = try stable_cache.addBytes(encoded_shims.entry);

    const argument_register: u8 = if (builtin.os.tag == .windows) rcx else rdi;
    const values = [_]runtime_deopt.ValueSpec{.{
        .vreg = 0,
        .kind = .reference,
        .source = .{ .native_register = argument_register },
    }};
    var deopt_table = try runtime_deopt.Table.init(std.testing.allocator, &.{.{
        .id = 1,
        .method_id = 0,
        .dex_pc = 0,
        .values = &values,
    }}, .{ .register_count = 1, .native_register_count = 16, .max_dex_pc = 0 });
    defer deopt_table.deinit();
    var registers: [1]u32 = .{0};
    var references: [1]u64 = .{@bitCast(runtime_value.Handle.none)};
    var reference_kinds: [1]bool = .{false};
    var reconstructed = runtime_deopt.Frame{ .execution = .{
        .pc = 0,
        .registers = &registers,
        .instructions = &.{},
        .register_is_ref = &reference_kinds,
        .reference_registers = &references,
    } };
    var scratch: [1]u64 = undefined;
    var stack_anchor: u64 = 0;
    const ResumeContext = struct {
        called: bool = false,
        expected: runtime_value.Handle,

        fn run(raw: *anyopaque, frame: *runtime_deopt.Frame) usize {
            const self: *@This() = @ptrCast(@alignCast(raw));
            self.called = frame.execution.reference_registers[0] == @as(u64, @bitCast(self.expected));
            return if (self.called) 55 else 0;
        }
    };
    const handle = runtime_value.Handle{ .index = 13, .generation = 6 };
    var resume_context = ResumeContext{ .expected = handle };
    var request = runtime_deopt.Request{
        .table = &deopt_table,
        .point_id = 1,
        .destination = .{ .frame = &reconstructed, .scratch = &scratch },
        .stack_base = @ptrCast(&stack_anchor),
        .stack_max_offset = @sizeOf(@TypeOf(stack_anchor)),
        .resume_context = &resume_context,
        .resume_fn = ResumeContext.run,
    };

    const Worker = struct {
        runtime: *runtime_jit.Runtime,
        context: *runtime_thread_registry.ThreadContext,
        entry_address: usize,
        request: *runtime_deopt.Request,
        handle: runtime_value.Handle,
        ready: *std.atomic.Value(bool),
        go: *std.atomic.Value(bool),
        failed: *std.atomic.Value(bool),
        result: *std.atomic.Value(usize),

        fn run(self: *@This()) void {
            var managed = self.runtime.enter(self.context) catch {
                self.failed.store(true, .release);
                self.ready.store(true, .release);
                return;
            };
            defer managed.deinit();
            var frame = CallFrame{
                .image = managed.registerImage() catch {
                    self.failed.store(true, .release);
                    self.ready.store(true, .release);
                    return;
                },
                .target = 0,
                .deopt_request = @intFromPtr(self.request),
                .method_id = 0,
            };
            frame.gp_args[0] = @bitCast(self.handle);
            self.ready.store(true, .release);
            while (!self.go.load(.acquire)) std.atomic.spinLoopHint();
            const EntryFn = fn (*const CallFrame) callconv(.c) usize;
            const call: *const EntryFn = @ptrFromInt(self.entry_address);
            self.result.store(call(&frame), .release);
            if (managed.native_state.last_code_dispatch != .deoptimized) self.failed.store(true, .release);
        }
    };
    var ready = std.atomic.Value(bool).init(false);
    var go = std.atomic.Value(bool).init(false);
    var failed = std.atomic.Value(bool).init(false);
    var result = std.atomic.Value(usize).init(0);
    var worker = Worker{
        .runtime = &runtime,
        .context = &context,
        .entry_address = entry_code.entryAddress(),
        .request = &request,
        .handle = handle,
        .ready = &ready,
        .go = &go,
        .failed = &failed,
        .result = &result,
    };
    const thread = try std.Thread.spawn(.{}, Worker.run, .{&worker});
    try waitFlag(&ready);
    try std.testing.expect(!failed.load(.acquire));
    var member_storage: [1]*runtime_thread_registry.ThreadContext = undefined;
    var handshake = try registry.beginHandshake(&member_storage);
    go.store(true, .release);
    try waitHandshake(&handshake, &context);
    try std.testing.expect(containsHandle(try handshake.snapshot(&context), handle));
    try handshake.release(&context);
    try handshake.finish();
    thread.join();

    try std.testing.expect(!failed.load(.acquire));
    try std.testing.expectEqual(@as(usize, 55), result.load(.acquire));
    try std.testing.expect(resume_context.called);
    try std.testing.expectEqual(@as(u64, 1), runtime.stats().handshake_polls);
    try std.testing.expectEqual(@as(u64, 1), runtime.stats().deoptimizations);
    try std.testing.expectEqual(@as(u64, 0), manager.stats().active_leases);
}

test "x64 dependency epoch trap deoptimizes through version-owned metadata" {
    if (builtin.cpu.arch != .x86_64) return error.SkipZigTest;

    var storage: [64]u8 align(runtime_value.object_alignment) = @splat(0);
    const regions = [_]runtime_value.Region{try runtime_value.Region.fromSlice(&storage)};
    var handles = try runtime_value.HandleTable.init(std.testing.allocator, 2, &regions);
    defer handles.deinit();
    var registry = try runtime_thread_registry.Registry.init(std.testing.allocator, std.testing.io, 1);
    defer registry.deinit() catch unreachable;
    var context = try runtime_thread_registry.ThreadContext.init(std.testing.allocator, 4);
    defer context.deinit();
    try registry.register(&context);
    defer registry.unregister(&context) catch unreachable;

    var shims = try encode(std.testing.allocator, @intFromPtr(&runtime_jit.slowResolveBridge));
    defer shims.deinit();
    var stable_cache = jit_memory.Cache.init(std.testing.allocator);
    defer stable_cache.deinit();
    const entry_code = try stable_cache.addBytes(shims.entry);
    const slow_code = try stable_cache.addBytes(shims.slow_helper);
    const deopt_code = try stable_cache.addBytes(shims.deopt_helper);

    const insts = [_]Instruction{
        .{ .iget = .{ .field_idx = 0, .dest_or_src = 1, .obj = 0 } },
        .{ .return_ = .{ .src = 1 } },
    };
    var optimized = try optimizer.optimize(std.testing.allocator, &insts, &.{}, .{});
    defer optimized.deinit();
    var handle_reg: ?u32 = null;
    var site_id: ?u32 = null;
    for (optimized.machine.blocks) |block| for (block.insts) |inst| {
        if (inst.opcode == .resolve_handle and inst.pc == 0) {
            handle_reg = inst.state_handle;
            site_id = inst.resolve_id;
        }
    };
    const point_values = [_]x64_encoder.DeoptValueSpec{
        .{ .vreg = 0, .kind = .reference, .source = .{ .machine_register = handle_reg orelse return error.TestUnexpectedResult } },
        .{ .vreg = 1, .kind = .scalar32, .source = .{ .constant = 0 } },
    };
    const caller_values = [_]x64_encoder.DeoptValueSpec{
        .{ .vreg = 0, .kind = .reference, .source = .{ .machine_register = handle_reg orelse return error.TestUnexpectedResult } },
        .{ .vreg = 1, .kind = .scalar32, .source = .{ .constant = 91 } },
    };
    const inline_frames = [_]x64_encoder.DeoptInlineFrameSpec{.{
        .method_id = 700,
        .dex_pc = 1,
        .register_count = 2,
        .values = &caller_values,
    }};
    const points = [_]x64_encoder.DeoptPointSpec{.{
        .id = 5,
        .safepoint_id = site_id orelse return error.TestUnexpectedResult,
        .method_id = 0,
        .dex_pc = 0,
        .values = &point_values,
        .inline_frames = &inline_frames,
    }};
    var dependency_epoch = std.atomic.Value(u64).init(1);
    const layouts = [_]x64_encoder.FieldLayout{.{ .offset = 8, .storage = .i32 }};
    var native = try x64_encoder.encodeWithOptions(std.testing.allocator, &optimized.machine, .{
        .runtime = .{
            .handle_capacity = handles.entryCapacity(),
            .region_count = handles.regionCount(),
            .slow_resolve_helper = slow_code.entryAddress(),
            .deopt_epoch_address = @intFromPtr(&dependency_epoch),
            .compiled_deopt_epoch = 1,
            .deopt_helper = deopt_code.entryAddress(),
            .field_layouts = &layouts,
        },
        .deopt = .{ .points = &points, .register_count = 2, .max_dex_pc = 1 },
    });
    defer native.deinit();
    try std.testing.expectEqual(@as(u32, 1), native.stats.deopt_guards);
    try std.testing.expectEqual(@as(u32, 1), native.stats.deopt_traps);
    try std.testing.expectEqual(@as(u32, 2), native.stats.deopt_frames);
    try std.testing.expectEqual(@as(u32, 4), native.stats.deopt_values);
    const native_bytes = try native.finalize();
    defer std.testing.allocator.free(native_bytes);
    const maps = if (native.root_maps) |*value| value else return error.TestUnexpectedResult;
    const deopt_table = if (native.deopt_table) |*value| value else return error.TestUnexpectedResult;

    const MetadataOwner = struct {
        references: std.atomic.Value(u32) = std.atomic.Value(u32).init(1),

        fn retain(raw: *anyopaque) void {
            const self: *@This() = @ptrCast(@alignCast(raw));
            _ = self.references.fetchAdd(1, .acq_rel);
        }

        fn release(raw: *anyopaque) void {
            const self: *@This() = @ptrCast(@alignCast(raw));
            const previous = self.references.fetchSub(1, .acq_rel);
            std.debug.assert(previous > 1);
        }
    };
    var metadata_owner = MetadataOwner{};
    var manager = try runtime_code_manager.Manager.init(std.testing.allocator, 1, 1, 1);
    defer manager.deinit() catch unreachable;
    var candidate = try manager.prepareWithMetadata(native_bytes, .{
        .context = &metadata_owner,
        .stack_maps = @intFromPtr(maps),
        .deopt_table = @intFromPtr(deopt_table),
        .retain = MetadataOwner.retain,
        .release = MetadataOwner.release,
    });
    defer candidate.deinit();
    try manager.publish(0, &candidate);
    try std.testing.expectEqual(@as(u32, 2), metadata_owner.references.load(.acquire));

    var runtime = try runtime_jit.Runtime.init(std.testing.allocator, &handles, &registry);
    defer runtime.deinit() catch unreachable;
    try runtime.installCodeManager(&manager);
    var registers: [2]u32 = @splat(0);
    var references: [2]u64 = @splat(@as(u64, @bitCast(runtime_value.Handle.none)));
    var reference_kinds: [2]bool = @splat(false);
    var reconstructed = runtime_deopt.Frame{ .execution = .{
        .pc = 0,
        .registers = &registers,
        .instructions = &.{},
        .register_is_ref = &reference_kinds,
        .reference_registers = &references,
    } };
    var caller_registers: [2]u32 = @splat(0);
    var caller_references: [2]u64 = @splat(@as(u64, @bitCast(runtime_value.Handle.none)));
    var caller_reference_kinds: [2]bool = @splat(false);
    var reconstructed_callers = [_]runtime_deopt.Frame{.{ .execution = .{
        .pc = 0,
        .registers = &caller_registers,
        .instructions = &.{},
        .register_is_ref = &caller_reference_kinds,
        .reference_registers = &caller_references,
    } }};
    var scratch: [4]u64 = undefined;
    var stack_anchor: u64 = 0;
    const Resume = struct {
        called: bool = false,

        fn run(raw: *anyopaque, frame: *runtime_deopt.Frame) usize {
            const self: *@This() = @ptrCast(@alignCast(raw));
            const caller = frame.previous orelse return 0;
            self.called = frame.active and frame.execution.register_is_ref[0] and
                caller.active and caller.method_id == 700 and caller.execution.pc == 1 and
                caller.execution.register_is_ref[0] and caller.execution.registers[1] == 91 and
                caller.previous == null;
            return if (self.called) 77 else 0;
        }
    };
    var resume_state = Resume{};
    var request = runtime_deopt.Request{
        .table = deopt_table,
        .point_id = 5,
        .destination = .{ .frame = &reconstructed, .inline_frames = &reconstructed_callers, .scratch = &scratch },
        .stack_base = @ptrCast(&stack_anchor),
        .stack_max_offset = @sizeOf(@TypeOf(stack_anchor)),
        .resume_context = &resume_state,
        .resume_fn = Resume.run,
    };
    var managed = try runtime.enter(&context);
    defer managed.deinit();
    var frame = CallFrame{
        .image = try managed.registerImage(),
        .target = 0,
        .deopt_request = @intFromPtr(&request),
        .method_id = 0,
    };
    const handle = runtime_value.Handle{ .index = 0, .generation = 1 };
    frame.gp_args[0] = @bitCast(handle);
    dependency_epoch.store(2, .release);
    const EntryFn = fn (*const CallFrame) callconv(.c) usize;
    try std.testing.expectEqual(@as(usize, 77), entry_code.typedEntry(EntryFn)(&frame));
    try std.testing.expect(resume_state.called);
    try std.testing.expectEqual(runtime_jit.CodeDispatchStatus.deoptimized, managed.native_state.last_code_dispatch);
    try std.testing.expectEqual(@as(u64, 0), manager.stats().active_leases);
    try std.testing.expect(try manager.invalidate(0));
    try std.testing.expectEqual(@as(u32, 1), try manager.reclaim());
    try std.testing.expectEqual(@as(u32, 1), metadata_owner.references.load(.acquire));
}

test "x64 deoptimization captures a managed spill slot and low XMM lane" {
    if (builtin.cpu.arch != .x86_64) return error.SkipZigTest;

    var storage: [64]u8 align(runtime_value.object_alignment) = @splat(0);
    const regions = [_]runtime_value.Region{try runtime_value.Region.fromSlice(&storage)};
    var handles = try runtime_value.HandleTable.init(std.testing.allocator, 1, &regions);
    defer handles.deinit();
    var registry = try runtime_thread_registry.Registry.init(std.testing.allocator, std.testing.io, 1);
    defer registry.deinit() catch unreachable;
    var context = try runtime_thread_registry.ThreadContext.init(std.testing.allocator, 2);
    defer context.deinit();
    try registry.register(&context);
    defer registry.unregister(&context) catch unreachable;

    var shims = try encode(std.testing.allocator, @intFromPtr(&runtime_jit.slowResolveBridge));
    defer shims.deinit();
    var stable_cache = jit_memory.Cache.init(std.testing.allocator);
    defer stable_cache.deinit();
    const entry_code = try stable_cache.addBytes(shims.entry);
    const deopt_code = try stable_cache.addBytes(shims.deopt_helper);

    const spill_bits: u64 = 0x1122334455667788;
    const xmm_bits: u64 = 0x8877665544332211;
    var target_buffer = code_buffer.Buffer.init(std.testing.allocator);
    defer target_buffer.deinit();
    try emitAdjustStack(&target_buffer, true, 16);
    try emitMovRegImm64(&target_buffer, rax, spill_bits);
    try target_buffer.emitBytes(&.{ 0x48, 0x89, 0x04, 0x24 }); // mov [rsp], rax
    try emitMovRegImm64(&target_buffer, rax, xmm_bits);
    try target_buffer.emitBytes(&.{ 0x66, 0x48, 0x0f, 0x6e, 0xd8 }); // movq xmm3, rax
    try emitMovRegImm64(&target_buffer, r10, 7);
    try emitMovRegImm64(&target_buffer, rax, deopt_code.entryAddress());
    try emitCall(&target_buffer, rax);
    try emitAdjustStack(&target_buffer, false, 16);
    try emitMovRegReg(&target_buffer, rax, r10);
    try target_buffer.emitU8(0xc3);
    const target_bytes = try target_buffer.finalize();
    defer std.testing.allocator.free(target_bytes);

    const values = [_]runtime_deopt.ValueSpec{
        .{ .vreg = 0, .kind = .scalar64, .source = .{ .stack_slot = 0 } },
        .{ .vreg = 2, .kind = .scalar64, .source = .{ .xmm_register = 3 } },
    };
    var deopt_table = try runtime_deopt.Table.init(std.testing.allocator, &.{.{
        .id = 44,
        .method_id = 9,
        .dex_pc = 3,
        .values = &values,
    }}, .{
        .register_count = 4,
        .native_register_count = 16,
        .xmm_register_count = 8,
        .max_dex_pc = 3,
    });
    defer deopt_table.deinit();
    var stack_maps = try runtime_stack_map.Table.init(std.testing.allocator, &.{.{
        .pc_offset = 7,
        .roots = &.{},
        .deopt_id = 44,
    }}, .{
        .native_register_count = 16,
        .interpreter_register_count = 0,
        .max_frame_depth = 0,
        .max_shadow_roots = 0,
    });
    defer stack_maps.deinit();
    try deopt_table.validateStackMaps(&stack_maps, true);
    try deopt_table.validateAllLinked(&stack_maps);

    const MetadataOwner = struct {
        references: std.atomic.Value(u32) = std.atomic.Value(u32).init(1),

        fn retain(raw: *anyopaque) void {
            const self: *@This() = @ptrCast(@alignCast(raw));
            _ = self.references.fetchAdd(1, .acq_rel);
        }

        fn release(raw: *anyopaque) void {
            const self: *@This() = @ptrCast(@alignCast(raw));
            const previous = self.references.fetchSub(1, .acq_rel);
            std.debug.assert(previous > 1);
        }
    };
    var owner = MetadataOwner{};
    var manager = try runtime_code_manager.Manager.init(std.testing.allocator, 1, 1, 1);
    defer manager.deinit() catch unreachable;
    var candidate = try manager.prepareWithMetadata(target_bytes, .{
        .context = &owner,
        .stack_maps = @intFromPtr(&stack_maps),
        .deopt_table = @intFromPtr(&deopt_table),
        .retain = MetadataOwner.retain,
        .release = MetadataOwner.release,
    });
    defer candidate.deinit();
    try manager.publish(0, &candidate);

    var runtime = try runtime_jit.Runtime.init(std.testing.allocator, &handles, &registry);
    defer runtime.deinit() catch unreachable;
    try runtime.installCodeManager(&manager);
    var registers: [4]u32 = @splat(0);
    var references: [4]u64 = @splat(@as(u64, @bitCast(runtime_value.Handle.none)));
    var kinds: [4]bool = @splat(false);
    var reconstructed = runtime_deopt.Frame{ .execution = .{
        .pc = 0,
        .registers = &registers,
        .instructions = &.{},
        .register_is_ref = &kinds,
        .reference_registers = &references,
    } };
    var scratch: [2]u64 = undefined;
    var unused_stack: u64 = 0;
    const Resume = struct {
        called: bool = false,
        expected_spill: u64,
        expected_xmm: u64,

        fn run(raw: *anyopaque, frame: *runtime_deopt.Frame) usize {
            const self: *@This() = @ptrCast(@alignCast(raw));
            self.called = frame.execution.getWide(0) == self.expected_spill and frame.execution.getWide(2) == self.expected_xmm;
            return if (self.called) 66 else 0;
        }
    };
    var resume_state = Resume{ .expected_spill = spill_bits, .expected_xmm = xmm_bits };
    var request = runtime_deopt.Request{
        .table = &deopt_table,
        .point_id = 44,
        .destination = .{ .frame = &reconstructed, .scratch = &scratch },
        .stack_base = @ptrCast(&unused_stack),
        .stack_max_offset = @sizeOf(@TypeOf(unused_stack)),
        .resume_context = &resume_state,
        .resume_fn = Resume.run,
    };
    var managed = try runtime.enter(&context);
    defer managed.deinit();
    const frame = CallFrame{
        .image = try managed.registerImage(),
        .target = 0,
        .deopt_request = @intFromPtr(&request),
        .method_id = 0,
    };
    const EntryFn = fn (*const CallFrame) callconv(.c) usize;
    try std.testing.expectEqual(@as(usize, 66), entry_code.typedEntry(EntryFn)(&frame));
    try std.testing.expect(resume_state.called);
    try std.testing.expectEqual(runtime_jit.CodeDispatchStatus.deoptimized, managed.native_state.last_code_dispatch);
    try std.testing.expectEqual(@as(usize, 0), manager.stats().active_leases);
    try std.testing.expect(try manager.invalidate(0));
    try std.testing.expectEqual(@as(u32, 1), try manager.reclaim());
    try std.testing.expectEqual(@as(u32, 1), owner.references.load(.acquire));
}

test "x64 OSR entry installs the exported physical GP image" {
    if (builtin.cpu.arch != .x86_64) return error.SkipZigTest;
    var shims = try encode(std.testing.allocator, @intFromPtr(&runtime_jit.slowResolveBridge));
    defer shims.deinit();
    var cache = jit_memory.Cache.init(std.testing.allocator);
    defer cache.deinit();
    const osr_entry = try cache.addBytes(shims.osr_entry);
    // mov rax, r9; ret -- observes an allocator-visible register directly.
    const target = try cache.addBytes(&.{ 0x4c, 0x89, 0xc8, 0xc3 });
    var frame = OsrCallFrame{
        .image = .{
            .r12_acknowledged_epoch = 0,
            .r13_region_bases = 0,
            .r14_descriptor_base = 0,
            .r15_thread_state = 0,
            .handle_capacity = 0,
            .region_count = 0,
            .descriptor_stride = 0,
        },
        .target = target.entryAddress(),
    };
    frame.gp[r9] = 0x8877665544332211;
    const EntryFn = fn (*const OsrCallFrame) callconv(.c) usize;
    try std.testing.expectEqual(@as(usize, 0x8877665544332211), osr_entry.typedEntry(EntryFn)(&frame));
}

test "x64 compiler-owned OSR label executes under its code-version lease" {
    if (builtin.cpu.arch != .x86_64) return error.SkipZigTest;
    var object_space: [64]u8 align(runtime_value.object_alignment) = @splat(0);
    std.mem.writeInt(i32, object_space[16..20], 0, .little);
    const regions = [_]runtime_value.Region{try runtime_value.Region.fromSlice(&object_space)};
    var handles = try runtime_value.HandleTable.init(std.testing.allocator, 2, &regions);
    defer handles.deinit();
    const handle = try handles.reserve(0, 0);
    try handles.publish(handle, 0, @ptrCast(&object_space[8]));

    var registry = try runtime_thread_registry.Registry.init(std.testing.allocator, std.testing.io, 1);
    defer registry.deinit() catch unreachable;
    var context = try runtime_thread_registry.ThreadContext.init(std.testing.allocator, 2);
    defer context.deinit();
    try registry.register(&context);
    defer registry.unregister(&context) catch unreachable;

    var shims = try encode(std.testing.allocator, @intFromPtr(&runtime_jit.slowResolveBridge));
    defer shims.deinit();
    var shim_cache = jit_memory.Cache.init(std.testing.allocator);
    defer shim_cache.deinit();
    const slow_helper = try shim_cache.addBytes(shims.slow_helper);
    const deopt_helper = try shim_cache.addBytes(shims.deopt_helper);
    const osr_adapter = try shim_cache.addBytes(shims.osr_entry);

    const insts = [_]Instruction{
        .{ .const_ = .{ .dest = 9, .value = 123 } },
        .{ .const_ = .{ .dest = 1, .value = 2 } },
        .{ .goto_ = .{ .offset = 1 } },
        .{ .iget = .{ .field_idx = 1, .dest_or_src = 2, .obj = 0 } },
        .{ .if_eqz = .{ .src = 1, .offset = 3 } },
        .{ .add_int_lit8 = .{ .dest = 1, .src = 1, .lit = -1 } },
        .{ .goto_ = .{ .offset = -3 } },
        .{ .return_ = .{ .src = 2 } },
    };
    var optimized = try optimizer.optimize(std.testing.allocator, &insts, &.{}, .{});
    defer optimized.deinit();
    try std.testing.expectEqual(@as(u32, 1), optimized.stats.loop_resolves_hoisted);
    try std.testing.expectEqual(@as(usize, 1), optimized.barriers.loop_reuses.len);
    const loop_header = optimized.barriers.loop_reuses[0].header;
    const point_pc = optimized.machine.blocks[loop_header].insts[0].pc orelse return error.TestUnexpectedResult;
    const required = try x64_encoder.osrRequiredRegistersAtBlockEntry(std.testing.allocator, &optimized.machine, loop_header);
    defer std.testing.allocator.free(required);
    var values: [10]x64_encoder.DeoptValueSpec = undefined;
    var assigned = [_]bool{false} ** 10;
    for (required) |reg| {
        const runtime_class = optimized.machine.runtime_values[reg];
        const value_id = switch (runtime_class) {
            .dalvik => |value| value.value,
            .derived_ptr => continue,
        };
        const vreg = optimized.function.values[value_id].reg;
        if (vreg >= values.len or assigned[vreg]) return error.TestUnexpectedResult;
        values[vreg] = .{
            .vreg = vreg,
            .kind = if (optimized.machine.isGcRoot(reg)) .reference else .scalar32,
            .source = .{ .machine_register = reg },
        };
        assigned[vreg] = true;
    }
    if (!assigned[0]) return error.TestUnexpectedResult;
    for (assigned, 0..) |is_assigned, vreg| {
        if (is_assigned) continue;
        if (vreg == 0) return error.TestUnexpectedResult;
        values[vreg] = .{
            .vreg = @intCast(vreg),
            .kind = .scalar32,
            .source = .{ .constant = if (vreg == 9) 123 else 0 },
        };
    }
    const points = [_]x64_encoder.DeoptPointSpec{.{
        .id = 51,
        .block_entry = loop_header,
        .method_id = 0,
        .dex_pc = point_pc,
        .values = &values,
    }};
    const osr_specs = [_]x64_encoder.OsrEntrySpec{.{ .point_id = 51, .block = loop_header }};
    var dependency_epoch = std.atomic.Value(u64).init(0);
    const layouts = [_]x64_encoder.FieldLayout{
        .{ .offset = 0, .storage = .i32 },
        .{ .offset = 8, .storage = .i32 },
    };
    var native = try x64_encoder.encodeWithOptions(std.testing.allocator, &optimized.machine, .{
        .runtime = .{
            .handle_capacity = handles.entryCapacity(),
            .region_count = handles.regionCount(),
            .slow_resolve_helper = slow_helper.entryAddress(),
            .deopt_epoch_address = @intFromPtr(&dependency_epoch),
            .compiled_deopt_epoch = 0,
            .deopt_helper = deopt_helper.entryAddress(),
            .field_layouts = &layouts,
        },
        .deopt = .{ .points = &points, .register_count = 10, .max_dex_pc = 7 },
        .osr_entries = &osr_specs,
    });
    defer native.deinit();
    try std.testing.expect(native.stats.frame_bytes > 0);
    try std.testing.expectEqual(@as(u32, 1), native.stats.osr_frame_landings);
    try std.testing.expectEqual(@as(u32, 1), native.stats.deopt_block_entries);
    try std.testing.expectEqual(@as(u32, 1), native.stats.osr_landing_safepoints);
    try std.testing.expectEqual(@as(u32, 1), native.stats.osr_derived_rematerializations);
    const stack_maps = if (native.root_maps) |*table| table else return error.TestUnexpectedResult;
    const deopt_table = if (native.deopt_table) |*table| table else return error.TestUnexpectedResult;
    var block_entry_map: ?*const runtime_stack_map.Record = null;
    for (stack_maps.records) |*record| {
        if (record.deopt_id == 51) block_entry_map = record;
    }
    const entry_map = block_entry_map orelse return error.TestUnexpectedResult;
    for (stack_maps.rootsFor(entry_map)) |root| {
        try std.testing.expectEqual(runtime_stack_map.LocationKind.native_register, root.kind);
    }
    const native_bytes = try native.finalize();
    defer std.testing.allocator.free(native_bytes);

    const Owner = struct {
        references: std.atomic.Value(u32) = std.atomic.Value(u32).init(1),
        fn retain(raw: *anyopaque) void {
            const self: *@This() = @ptrCast(@alignCast(raw));
            _ = self.references.fetchAdd(1, .acq_rel);
        }
        fn release(raw: *anyopaque) void {
            const self: *@This() = @ptrCast(@alignCast(raw));
            _ = self.references.fetchSub(1, .acq_rel);
        }
    };
    var owner: Owner = .{};
    var manager = try runtime_code_manager.Manager.init(std.testing.allocator, 1, 1, 1);
    defer manager.deinit() catch unreachable;
    var candidate = try manager.prepareWithMetadata(native_bytes, .{
        .context = @ptrCast(&owner),
        .stack_maps = @intFromPtr(stack_maps),
        .deopt_table = @intFromPtr(deopt_table),
        .osr_entries = @intFromPtr(native.osr_entries.ptr),
        .osr_entry_count = @intCast(native.osr_entries.len),
        .retain = Owner.retain,
        .release = Owner.release,
    });
    try manager.publish(0, &candidate);

    var runtime = try runtime_jit.Runtime.init(std.testing.allocator, &handles, &registry);
    defer runtime.deinit() catch unreachable;
    try runtime.installCodeManager(&manager);
    {
        var managed = try runtime.enter(&context);
        defer managed.deinit();
        try managed.installRootMaps(stack_maps);
        var frame_registers: [10]u32 = @splat(0);
        frame_registers[1] = 2;
        frame_registers[9] = 123;
        var frame_references: [10]u64 = @splat(@as(u64, @bitCast(runtime_value.Handle.none)));
        frame_references[0] = @bitCast(handle);
        var frame_reference_kinds: [10]bool = @splat(false);
        frame_reference_kinds[0] = true;
        var interpreter_frame = runtime_deopt.Frame{
            .method_id = 0,
            .active = true,
            .execution = .{
                .pc = point_pc,
                .registers = &frame_registers,
                .instructions = &.{},
                .register_is_ref = &frame_reference_kinds,
                .reference_registers = &frame_references,
            },
        };
        var gp: [16]u64 = @splat(0);
        var scratch: [26]u64 = undefined;
        try deopt_table.exportOsr(51, &interpreter_frame, .{ .native_registers = &gp, .scratch = &scratch });
        const target = runtime_jit.codeOsrEnterBridge(&managed.native_state, 0, 0, 51);
        var call_frame = OsrCallFrame{ .image = try managed.registerImage(), .target = target, .gp = gp };
        const EntryFn = fn (*const OsrCallFrame) callconv(.c) usize;
        const value = osr_adapter.typedEntry(EntryFn)(&call_frame);
        try std.testing.expectEqual(@as(usize, 0), value);
        try std.testing.expectEqual(@as(usize, 0), runtime_jit.codeLeaseExitBridge(&managed.native_state, value));
        try std.testing.expectEqual(@as(u64, 0), manager.stats().active_leases);
    }
    try std.testing.expect(try manager.invalidate(0));
    try std.testing.expectEqual(@as(u32, 1), try manager.reclaim());
    try std.testing.expectEqual(@as(u32, 1), owner.references.load(.acquire));
}

test "x64 shim encoding is leak-free at every allocation failure" {
    if (builtin.cpu.arch != .x86_64) return error.SkipZigTest;
    try std.testing.checkAllAllocationFailures(std.testing.allocator, allocationFailureProbe, .{});
}
