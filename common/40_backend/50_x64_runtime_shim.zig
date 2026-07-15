//! x86-64 managed-entry and cooperative slow-resolve machine-code shims.

const std = @import("std");
const builtin = @import("builtin");
const code_buffer = @import("code_buffer");
const jit_memory = @import("jit_memory");
const optimizer = @import("optimizer");
const runtime_gc = @import("runtime_gc");
const runtime_heap = @import("runtime_heap");
const runtime_jit = @import("runtime_jit");
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
    gp_args: [6]usize = @splat(0),
};

pub const EncodedShims = struct {
    allocator: std.mem.Allocator,
    entry: []u8,
    slow_helper: []u8,
    satb_pre_write_helper: []u8,
    card_mark_helper: []u8,
    card_mark_repeat_helper: []u8,
    static_root_post_write_helper: []u8,

    pub fn deinit(self: *EncodedShims) void {
        self.allocator.free(self.static_root_post_write_helper);
        self.allocator.free(self.card_mark_repeat_helper);
        self.allocator.free(self.card_mark_helper);
        self.allocator.free(self.satb_pre_write_helper);
        self.allocator.free(self.slow_helper);
        self.allocator.free(self.entry);
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

fn emitEntry(buffer: *code_buffer.Buffer) Error!void {
    const frame = if (builtin.os.tag == .windows) rcx else rdi;
    try emitReservedSave(buffer);
    try emitAdjustStack(buffer, true, if (builtin.os.tag == .windows) 40 else 8);

    try emitMovRegMem(buffer, rax, frame, @offsetOf(CallFrame, "target"));
    try emitMovRegMem(buffer, r12, frame, imageOffset("r12_acknowledged_epoch"));
    try emitMovRegMem(buffer, r13, frame, imageOffset("r13_region_bases"));
    try emitMovRegMem(buffer, r14, frame, imageOffset("r14_descriptor_base"));
    try emitMovRegMem(buffer, r15, frame, imageOffset("r15_thread_state"));

    if (builtin.os.tag == .windows) {
        try emitMovRegMem(buffer, rdx, frame, argOffset(1));
        try emitMovRegMem(buffer, r8, frame, argOffset(2));
        try emitMovRegMem(buffer, r9, frame, argOffset(3));
        try emitMovRegMem(buffer, rcx, frame, argOffset(0));
    } else {
        try emitMovRegMem(buffer, rsi, frame, argOffset(1));
        try emitMovRegMem(buffer, rdx, frame, argOffset(2));
        try emitMovRegMem(buffer, rcx, frame, argOffset(3));
        try emitMovRegMem(buffer, r8, frame, argOffset(4));
        try emitMovRegMem(buffer, r9, frame, argOffset(5));
        try emitMovRegMem(buffer, rdi, frame, argOffset(0));
    }
    try emitCall(buffer, rax);
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

    var helper_buffer = code_buffer.Buffer.init(allocator);
    defer helper_buffer.deinit();
    try emitSlowHelper(&helper_buffer, bridge_address);
    const slow_helper = try helper_buffer.finalize();
    errdefer allocator.free(slow_helper);

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
        .slow_helper = slow_helper,
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
    _ = entry_code.typedEntry(EntryFn)(&frame);

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

fn allocationFailureProbe(allocator: std.mem.Allocator) !void {
    var shims = try encode(allocator, 1);
    defer shims.deinit();
}

test "x64 shim encoding is leak-free at every allocation failure" {
    if (builtin.cpu.arch != .x86_64) return error.SkipZigTest;
    try std.testing.checkAllAllocationFailures(std.testing.allocator, allocationFailureProbe, .{});
}
