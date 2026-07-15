//! Register-allocated x86-64 encoder.
//!
//! This backend emits a no-frame native subset from the register-machine IR
//! after physical register allocation. It has no stack-slot traffic on the
//! supported path; if allocation spills or an operation is outside the native
//! register subset, it fails explicitly.

const std = @import("std");
const builtin = @import("builtin");
const cfg = @import("cfg");
const code_buffer = @import("code_buffer");
const jit_memory = @import("jit_memory");
const machine = @import("machine_bridge");
const optimizer = @import("optimizer");
const regalloc = @import("regalloc");
const runtime_stack_map = @import("runtime_stack_map");
const runtime_value = @import("runtime_value");
const Instruction = @import("instructions").Instruction;

pub const Error = code_buffer.Error || regalloc.Error || runtime_stack_map.Error || error{
    InvalidMachine,
    InvalidRuntimeAbi,
    MissingBarrierHelper,
    MissingArrayLayout,
    MissingFieldLayout,
    MissingStaticFieldLayout,
    MissingRuntimeAbi,
    RootMapsRequireSingleBlock,
    SpillsUnsupported,
    UnsupportedInstruction,
};

/// Register contract installed by the managed-to-native entry trampoline.
/// r12 = acknowledged read epoch, r13 = region base table, r14 = descriptor
/// table, r15 = pinned NativeThreadState. From-space cannot be reclaimed while r12's
/// epoch is active. The slow helper uses the private preserve-all ABI: r10 is
/// handle/result, r11 is `(pc << 32) | resolve_id`, and every managed-value
/// register plus r12-r15 is preserved. The initial adapter is no-safepoint and
/// relies on r12's epoch lease; a polling adapter requires complete stack-map
/// root publication first.
pub const RuntimeAbi = struct {
    handle_capacity: u32,
    region_count: u16,
    slow_resolve_helper: usize,
    satb_pre_write_helper: usize = 0,
    card_mark_helper: usize = 0,
    card_mark_repeat_helper: usize = 0,
    static_root_post_write_helper: usize = 0,
    reference_array_layout: ?ReferenceArrayLayout = null,
    /// Immutable layouts indexed by resolved field id.
    field_layouts: []const FieldLayout,
    /// Immutable layouts indexed by resolved static field id. Static storage
    /// is pinned and never represented by a managed destination Handle.
    static_field_layouts: []const StaticFieldLayout = &.{},

    fn verify(self: RuntimeAbi) Error!void {
        if (self.handle_capacity == 0 or self.region_count == 0 or self.region_count > 256) return error.InvalidRuntimeAbi;
        if (self.slow_resolve_helper == 0) return error.InvalidRuntimeAbi;
        for (self.field_layouts) |layout| if (layout.offset > std.math.maxInt(i32)) return error.InvalidRuntimeAbi;
        for (self.static_field_layouts) |layout| {
            if (layout.address == 0 or !std.mem.isAligned(layout.address, storageAlignment(layout.storage))) {
                return error.InvalidRuntimeAbi;
            }
        }
        if (self.reference_array_layout) |layout| {
            if (layout.length_offset > std.math.maxInt(i32) or layout.data_offset > std.math.maxInt(i32) or
                !std.mem.isAligned(layout.length_offset, @sizeOf(u32)) or
                !std.mem.isAligned(layout.data_offset, @sizeOf(runtime_value.Handle)) or
                layout.element_stride != @sizeOf(runtime_value.Handle) or
                layout.length_offset > std.math.maxInt(u32) - @sizeOf(u32) or
                layout.data_offset < layout.length_offset + @sizeOf(u32)) return error.InvalidRuntimeAbi;
        }
    }
};

pub const FieldStorage = enum(u8) {
    i8,
    u8,
    i16,
    u16,
    i32,
    i64,
    reference,
};

pub const FieldLayout = struct {
    offset: u32,
    storage: FieldStorage,
};

pub const StaticFieldLayout = struct {
    address: usize,
    storage: FieldStorage,
};

pub const ReferenceArrayLayout = struct {
    length_offset: u32,
    data_offset: u32,
    element_stride: u8 = @sizeOf(runtime_value.Handle),
};

pub const Options = struct {
    runtime: ?RuntimeAbi = null,
};

pub const Stats = struct {
    bytes: u32 = 0,
    blocks: u32 = 0,
    native_insts: u32 = 0,
    register_moves: u32 = 0,
    constants: u32 = 0,
    returns: u32 = 0,
    jumps: u32 = 0,
    branches: u32 = 0,
    descriptor_loads: u32 = 0,
    fast_resolves: u32 = 0,
    cold_resolve_sites: u32 = 0,
    pointer_loads: u32 = 0,
    pointer_stores: u32 = 0,
    bounds_checks: u32 = 0,
    array_loads: u32 = 0,
    array_stores: u32 = 0,
    satb_barriers: u32 = 0,
    satb_repeat_barriers: u32 = 0,
    card_barriers: u32 = 0,
    card_repeat_barriers: u32 = 0,
    static_root_barriers: u32 = 0,
    root_map_sites: u32 = 0,
    root_map_locations: u32 = 0,
};

pub const Function = struct {
    allocator: std.mem.Allocator,
    source: *const machine.Function,
    allocation: regalloc.Allocation,
    buffer: code_buffer.Buffer,
    block_labels: []code_buffer.LabelId,
    root_maps: ?runtime_stack_map.Table,
    stats: Stats,

    pub fn deinit(self: *Function) void {
        if (self.root_maps) |*maps| maps.deinit();
        self.allocator.free(self.block_labels);
        self.buffer.deinit();
        self.allocation.deinit();
        self.* = undefined;
    }

    pub fn verify(self: *Function) Error!void {
        self.source.verify() catch return error.InvalidMachine;
        try self.allocation.verify();
        if (self.block_labels.len != self.source.blocks.len) return error.InvalidMachine;
        if (self.stats.descriptor_loads != self.source.stats.resolves or self.stats.fast_resolves != self.source.stats.resolves or self.stats.cold_resolve_sites != self.source.stats.resolves) return error.InvalidMachine;
        if ((self.source.stats.resolves == 0) != (self.root_maps == null)) return error.InvalidMachine;
        if (self.root_maps) |maps| {
            if (maps.records.len != self.source.stats.resolves or self.stats.root_map_sites != self.source.stats.resolves) return error.InvalidMachine;
            if (maps.locations.len != self.stats.root_map_locations) return error.InvalidMachine;
        }
        try self.buffer.verify();
    }

    pub fn finalize(self: *Function) Error![]u8 {
        try self.verify();
        const bytes = try self.buffer.finalize();
        self.stats.bytes = @intCast(bytes.len);
        return bytes;
    }

    pub fn print(self: *const Function, writer: anytype) !void {
        try writer.print(
            "x64_register_encoder bytes={d} blocks={d} native_insts={d} moves={d} constants={d} returns={d} jumps={d} branches={d} descriptor_loads={d} fast_resolves={d} cold_resolves={d} bounds={d} array_loads={d} array_stores={d} ptr_loads={d} ptr_stores={d} satb={d} satb_repeats={d} cards={d} card_repeats={d} root_sites={d} root_locations={d}\n",
            .{
                self.buffer.len(),
                self.stats.blocks,
                self.stats.native_insts,
                self.stats.register_moves,
                self.stats.constants,
                self.stats.returns,
                self.stats.jumps,
                self.stats.branches,
                self.stats.descriptor_loads,
                self.stats.fast_resolves,
                self.stats.cold_resolve_sites,
                self.stats.bounds_checks,
                self.stats.array_loads,
                self.stats.array_stores,
                self.stats.pointer_loads,
                self.stats.pointer_stores,
                self.stats.satb_barriers,
                self.stats.satb_repeat_barriers,
                self.stats.card_barriers,
                self.stats.card_repeat_barriers,
                self.stats.root_map_sites,
                self.stats.root_map_locations,
            },
        );
        try self.allocation.print(writer);
        try self.buffer.print(writer);
    }
};

const scratch_index: u4 = 10;
const scratch_descriptor: u4 = 11;
const region_table: u4 = 13;
const descriptor_table: u4 = 14;

const descriptor_offset_clear_shift: u6 = 28;
const descriptor_region_shift: u6 = 36;
const descriptor_generation_shift: u6 = 44;
const descriptor_state_shift: u6 = 60;
const object_alignment_shift: u6 = 3;

const RESOLVER_GP_REGS = [_]regalloc.PhysReg{ .rax, .rcx, .rdx, .rsi, .rdi, .r8, .r9 };

const ColdResolve = struct {
    entry: code_buffer.LabelId,
    continuation: code_buffer.LabelId,
    handle: regalloc.PhysReg,
    destination: regalloc.PhysReg,
    site_key: u64,
};

fn x64Reg(reg: regalloc.PhysReg) Error!u4 {
    return switch (reg) {
        .rax => 0,
        .rcx => 1,
        .rdx => 2,
        .rsi => 6,
        .rdi => 7,
        .r8 => 8,
        .r9 => 9,
        .r10 => 10,
        .r11 => 11,
        else => error.UnsupportedInstruction,
    };
}

fn emitRexX(buffer: *code_buffer.Buffer, w: bool, reg: u4, index: u4, base: u4) Error!void {
    var rex: u8 = 0x40;
    if (w) rex |= 0x08;
    if ((reg & 8) != 0) rex |= 0x04;
    if ((index & 8) != 0) rex |= 0x02;
    if ((base & 8) != 0) rex |= 0x01;
    if (rex != 0x40) try buffer.emitU8(rex);
}

fn emitModRmRaw(buffer: *code_buffer.Buffer, mode: u2, reg: u4, rm: u4) Error!void {
    try buffer.emitU8((@as(u8, mode) << 6) | ((@as(u8, reg) & 7) << 3) | (@as(u8, rm) & 7));
}

fn emitMovRawRaw(buffer: *code_buffer.Buffer, dst: u4, src: u4, wide: bool) Error!void {
    if (dst == src) return;
    try emitRex(buffer, wide, src, dst);
    try buffer.emitU8(0x89);
    try emitModRm(buffer, src, dst);
}

fn emitMovRawImm64(buffer: *code_buffer.Buffer, dst: u4, value: u64) Error!void {
    try emitRex(buffer, true, 0, dst);
    try buffer.emitU8(0xb8 + @as(u8, dst & 7));
    try buffer.emitU64(value);
}

fn emitMovRawIndexed(buffer: *code_buffer.Buffer, dst: u4, base: u4, index: u4, scale: u2) Error!void {
    try emitRexX(buffer, true, dst, index, base);
    try buffer.emitU8(0x8b);
    const mode: u2 = if ((base & 7) == 5) 1 else 0;
    try emitModRmRaw(buffer, mode, dst, 4);
    try buffer.emitU8((@as(u8, scale) << 6) | ((@as(u8, index) & 7) << 3) | (@as(u8, base) & 7));
    if (mode == 1) try buffer.emitU8(0);
}

fn emitMemoryOperand(buffer: *code_buffer.Buffer, reg: u4, base: u4, displacement: i32) Error!void {
    const base_low = base & 7;
    const mode: u2 = if (displacement == 0 and base_low != 5)
        0
    else if (displacement >= std.math.minInt(i8) and displacement <= std.math.maxInt(i8))
        1
    else
        2;
    const uses_sib = base_low == 4;
    try emitModRmRaw(buffer, mode, reg, if (uses_sib) 4 else base_low);
    if (uses_sib) try buffer.emitU8((4 << 3) | @as(u8, base_low));
    switch (mode) {
        0 => {},
        1 => try buffer.emitU8(@bitCast(@as(i8, @intCast(displacement)))),
        2 => try buffer.emitU32(@bitCast(displacement)),
        else => unreachable,
    }
}

fn emitIndexedMemoryOperand(
    buffer: *code_buffer.Buffer,
    reg: u4,
    base: u4,
    index: u4,
    scale: u2,
    displacement: i32,
) Error!void {
    const base_low = base & 7;
    const mode: u2 = if (displacement == 0 and base_low != 5)
        0
    else if (displacement >= std.math.minInt(i8) and displacement <= std.math.maxInt(i8))
        1
    else
        2;
    try emitModRmRaw(buffer, mode, reg, 4);
    try buffer.emitU8((@as(u8, scale) << 6) | ((@as(u8, index) & 7) << 3) | @as(u8, base_low));
    switch (mode) {
        0 => {},
        1 => try buffer.emitU8(@bitCast(@as(i8, @intCast(displacement)))),
        2 => try buffer.emitU32(@bitCast(displacement)),
        else => unreachable,
    }
}

fn emitArrayElementAddress(
    buffer: *code_buffer.Buffer,
    destination: u4,
    base: u4,
    index: u4,
    layout: ReferenceArrayLayout,
) Error!void {
    try emitMovRawRaw(buffer, destination, index, false);
    try emitRexX(buffer, true, destination, destination, base);
    try buffer.emitU8(0x8d);
    try emitIndexedMemoryOperand(buffer, destination, base, destination, 3, @intCast(layout.data_offset));
}

fn emitArrayBoundsCheck(
    buffer: *code_buffer.Buffer,
    base: u4,
    index: u4,
    layout: ReferenceArrayLayout,
) Error!void {
    // Unsigned comparison rejects negative indices as well as index >= length.
    try emitRex(buffer, false, index, base);
    try buffer.emitU8(0x3b);
    try emitMemoryOperand(buffer, index, base, @intCast(layout.length_offset));
    const valid = try buffer.newLabel();
    try emitJccOpcode(buffer, 0x82, valid); // jb
    try buffer.emitBytes(&.{ 0x0f, 0x0b }); // fail closed until exception stubs land
    try buffer.bindLabel(valid);
}

fn emitMovRawFromMemory(buffer: *code_buffer.Buffer, dst: u4, base: u4, displacement: i32, wide: bool) Error!void {
    try emitRex(buffer, wide, dst, base);
    try buffer.emitU8(0x8b);
    try emitMemoryOperand(buffer, dst, base, displacement);
}

fn emitLoadField(buffer: *code_buffer.Buffer, dst: u4, base: u4, displacement: i32, storage: FieldStorage) Error!void {
    switch (storage) {
        .i8, .u8, .i16, .u16 => {
            try emitRex(buffer, false, dst, base);
            try buffer.emitU8(0x0f);
            try buffer.emitU8(switch (storage) {
                .i8 => 0xbe,
                .u8 => 0xb6,
                .i16 => 0xbf,
                .u16 => 0xb7,
                else => unreachable,
            });
            try emitMemoryOperand(buffer, dst, base, displacement);
        },
        .i32 => try emitMovRawFromMemory(buffer, dst, base, displacement, false),
        .i64, .reference => try emitMovRawFromMemory(buffer, dst, base, displacement, true),
    }
}

fn emitByteRex(buffer: *code_buffer.Buffer, reg: u4, base: u4) Error!void {
    var rex: u8 = 0x40;
    if ((reg & 8) != 0) rex |= 0x04;
    if ((base & 8) != 0) rex |= 0x01;
    // A bare REX prefix selects SIL/DIL instead of AH/BH for low ids 4..7.
    if (rex != 0x40 or (reg & 7) >= 4) try buffer.emitU8(rex);
}

fn emitStoreField(buffer: *code_buffer.Buffer, base: u4, displacement: i32, src: u4, storage: FieldStorage) Error!void {
    switch (storage) {
        .i8, .u8 => {
            try emitByteRex(buffer, src, base);
            try buffer.emitU8(0x88);
            try emitMemoryOperand(buffer, src, base, displacement);
        },
        .i16, .u16 => {
            try buffer.emitU8(0x66);
            try emitRex(buffer, false, src, base);
            try buffer.emitU8(0x89);
            try emitMemoryOperand(buffer, src, base, displacement);
        },
        .i32 => try emitMovMemoryFromRaw(buffer, base, displacement, src, false),
        .i64, .reference => try emitMovMemoryFromRaw(buffer, base, displacement, src, true),
    }
}

fn emitMovMemoryFromRaw(buffer: *code_buffer.Buffer, base: u4, displacement: i32, src: u4, wide: bool) Error!void {
    try emitRex(buffer, wide, src, base);
    try buffer.emitU8(0x89);
    try emitMemoryOperand(buffer, src, base, displacement);
}

fn emitShiftRawImm(buffer: *code_buffer.Buffer, reg: u4, extension: u4, amount: u6) Error!void {
    try emitRex(buffer, true, extension, reg);
    try buffer.emitU8(0xc1);
    try emitModRm(buffer, extension, reg);
    try buffer.emitU8(amount);
}

fn emitAndRawImm32(buffer: *code_buffer.Buffer, reg: u4, value: u32) Error!void {
    try emitRex(buffer, true, 4, reg);
    try buffer.emitU8(0x81);
    try emitModRm(buffer, 4, reg);
    try buffer.emitU32(value);
}

fn emitCmpRawImm32(buffer: *code_buffer.Buffer, reg: u4, value: u32) Error!void {
    try emitRex(buffer, false, 7, reg);
    try buffer.emitU8(0x81);
    try emitModRm(buffer, 7, reg);
    try buffer.emitU32(value);
}

fn emitCmpRawRaw(buffer: *code_buffer.Buffer, lhs: u4, rhs: u4, wide: bool) Error!void {
    try emitRex(buffer, wide, lhs, rhs);
    try buffer.emitU8(0x3b);
    try emitModRm(buffer, lhs, rhs);
}

fn emitAddRawRaw(buffer: *code_buffer.Buffer, dst: u4, src: u4) Error!void {
    try emitRex(buffer, true, dst, src);
    try buffer.emitU8(0x03);
    try emitModRm(buffer, dst, src);
}

fn emitCallRaw(buffer: *code_buffer.Buffer, target: u4) Error!void {
    try emitRex(buffer, false, 2, target);
    try buffer.emitU8(0xff);
    try emitModRm(buffer, 2, target);
}

fn emitPushRaw(buffer: *code_buffer.Buffer, reg: u4) Error!void {
    if ((reg & 8) != 0) try buffer.emitU8(0x41);
    try buffer.emitU8(0x50 + @as(u8, reg & 7));
}

fn emitPopRaw(buffer: *code_buffer.Buffer, reg: u4) Error!void {
    if ((reg & 8) != 0) try buffer.emitU8(0x41);
    try buffer.emitU8(0x58 + @as(u8, reg & 7));
}

fn emitLeaRawMemory(buffer: *code_buffer.Buffer, dst: u4, base: u4, displacement: i32) Error!void {
    try emitRex(buffer, true, dst, base);
    try buffer.emitU8(0x8d);
    try emitMemoryOperand(buffer, dst, base, displacement);
}

/// rax may contain a live managed value. Preserve it around the private
/// no-safepoint adapter; r10/r11 are reserved barrier arguments.
fn emitBarrierCall(buffer: *code_buffer.Buffer, helper: usize) Error!void {
    if (helper == 0) return error.MissingBarrierHelper;
    try emitPushRaw(buffer, 0);
    try emitMovRawImm64(buffer, 0, helper);
    try emitCallRaw(buffer, 0);
    try emitPopRaw(buffer, 0);
}

fn physOf(allocation: *const regalloc.Allocation, reg: machine.RegId) Error!regalloc.PhysReg {
    return switch (allocation.locationOf(reg) orelse return error.InvalidMachine) {
        .phys => |phys| phys,
        .spill => error.SpillsUnsupported,
        .none => error.InvalidMachine,
    };
}

fn emitRex(buffer: *code_buffer.Buffer, w: bool, reg: u4, rm: u4) Error!void {
    var rex: u8 = 0x40;
    if (w) rex |= 0x08;
    if ((reg & 8) != 0) rex |= 0x04;
    if ((rm & 8) != 0) rex |= 0x01;
    if (rex != 0x40) try buffer.emitU8(rex);
}

fn emitModRm(buffer: *code_buffer.Buffer, reg: u4, rm: u4) Error!void {
    try buffer.emitU8(0xc0 | ((@as(u8, reg) & 7) << 3) | (@as(u8, rm) & 7));
}

fn isWideType(ty: anytype) bool {
    return switch (ty) {
        .long, .double, .object => true,
        else => false,
    };
}

fn emitMovRegReg(buffer: *code_buffer.Buffer, dst: regalloc.PhysReg, src: regalloc.PhysReg, wide: bool) Error!void {
    if (dst == src) return;
    const d = try x64Reg(dst);
    const s = try x64Reg(src);
    try emitRex(buffer, wide, s, d);
    try buffer.emitU8(0x89);
    try emitModRm(buffer, s, d);
}

fn emitMovRegImm32(buffer: *code_buffer.Buffer, dst: regalloc.PhysReg, value: i32) Error!void {
    const d = try x64Reg(dst);
    try emitRex(buffer, false, 0, d);
    try buffer.emitU8(0xb8 + @as(u8, d & 7));
    try buffer.emitU32(@bitCast(value));
}

fn emitBinaryRegReg(buffer: *code_buffer.Buffer, opcode: machine.Opcode, dst: regalloc.PhysReg, rhs: regalloc.PhysReg) Error!void {
    const d = try x64Reg(dst);
    const r = try x64Reg(rhs);
    try emitRex(buffer, false, d, r);
    switch (opcode) {
        .add_i32 => try buffer.emitU8(0x03),
        .sub_i32 => try buffer.emitU8(0x2b),
        .and_i32 => try buffer.emitU8(0x23),
        .or_i32 => try buffer.emitU8(0x0b),
        .xor_i32 => try buffer.emitU8(0x33),
        .mul_i32 => {
            try buffer.emitU8(0x0f);
            try buffer.emitU8(0xaf);
        },
        else => return error.UnsupportedInstruction,
    }
    try emitModRm(buffer, d, r);
}

fn emitCmpRegZero(buffer: *code_buffer.Buffer, reg: regalloc.PhysReg) Error!void {
    const r = try x64Reg(reg);
    try emitRex(buffer, false, 7, r);
    try buffer.emitU8(0x83);
    try emitModRm(buffer, 7, r);
    try buffer.emitU8(0);
}

fn emitCmpRegs(buffer: *code_buffer.Buffer, lhs: regalloc.PhysReg, rhs: regalloc.PhysReg) Error!void {
    const l = try x64Reg(lhs);
    const r = try x64Reg(rhs);
    try emitRex(buffer, false, l, r);
    try buffer.emitU8(0x3b);
    try emitModRm(buffer, l, r);
}

fn emitRet(buffer: *code_buffer.Buffer) Error!void {
    try buffer.emitU8(0xc3);
}

fn emitJump(buffer: *code_buffer.Buffer, label: code_buffer.LabelId) Error!void {
    try buffer.emitU8(0xe9);
    _ = try buffer.reloc(label, .rel32, 0);
}

fn emitJcc(buffer: *code_buffer.Buffer, condition: machine.Condition, label: code_buffer.LabelId) Error!void {
    try buffer.emitU8(0x0f);
    try buffer.emitU8(switch (condition) {
        .eq => 0x84,
        .ne => 0x85,
        .lt => 0x8c,
        .ge => 0x8d,
        .gt => 0x8f,
        .le => 0x8e,
    });
    _ = try buffer.reloc(label, .rel32, 0);
}

fn emitJccOpcode(buffer: *code_buffer.Buffer, opcode: u8, label: code_buffer.LabelId) Error!void {
    try buffer.emitU8(0x0f);
    try buffer.emitU8(opcode);
    _ = try buffer.reloc(label, .rel32, 0);
}

fn fieldLayout(runtime: RuntimeAbi, field_idx: ?u32) Error!FieldLayout {
    const index = field_idx orelse return error.InvalidMachine;
    if (index >= runtime.field_layouts.len) return error.MissingFieldLayout;
    return runtime.field_layouts[index];
}

fn staticFieldLayout(runtime: RuntimeAbi, field_idx: ?u32) Error!StaticFieldLayout {
    const index = field_idx orelse return error.InvalidMachine;
    if (index >= runtime.static_field_layouts.len) return error.MissingStaticFieldLayout;
    return runtime.static_field_layouts[index];
}

fn storageAlignment(storage: FieldStorage) usize {
    return switch (storage) {
        .i8, .u8 => 1,
        .i16, .u16 => 2,
        .i32 => 4,
        .i64, .reference => 8,
    };
}

fn referenceArrayLayout(runtime: RuntimeAbi) Error!ReferenceArrayLayout {
    return runtime.reference_array_layout orelse error.MissingArrayLayout;
}

fn verifyFieldType(source: *const machine.Function, reg: machine.RegId, storage: FieldStorage) Error!void {
    if (reg >= source.reg_types.len) return error.InvalidMachine;
    const valid = switch (source.reg_types[reg]) {
        .int => switch (storage) {
            .i8, .u8, .i16, .u16, .i32 => true,
            else => false,
        },
        .long => storage == .i64,
        .object => storage == .reference,
        // A write-only parameter may remain directionally unknown. Immutable
        // field metadata supplies its concrete machine width, but reference
        // stores are accepted only when lowering independently classified the
        // value as a GC root.
        .unknown => storage != .reference or source.isGcRoot(reg),
        else => false,
    };
    if (!valid) return error.InvalidRuntimeAbi;
}

fn emitResolve(
    allocator: std.mem.Allocator,
    buffer: *code_buffer.Buffer,
    allocation: *const regalloc.Allocation,
    inst: machine.Inst,
    runtime: RuntimeAbi,
    cold: *std.ArrayList(ColdResolve),
    stats: *Stats,
) Error!void {
    if (inst.defs.len != 1 or inst.uses.len != 1 or inst.state_handle != inst.uses[0]) return error.InvalidMachine;
    const handle = try physOf(allocation, inst.uses[0]);
    const destination = try physOf(allocation, inst.defs[0]);
    if (handle == destination) return error.InvalidMachine;
    const handle_raw = try x64Reg(handle);
    const destination_raw = try x64Reg(destination);

    const cold_label = try buffer.newLabel();
    const resume_label = try buffer.newLabel();
    try cold.append(allocator, .{
        .entry = cold_label,
        .continuation = resume_label,
        .handle = handle,
        .destination = destination,
        .site_key = (@as(u64, inst.pc orelse std.math.maxInt(u32)) << 32) |
            @as(u64, inst.resolve_id orelse return error.InvalidMachine),
    });

    // The 32-bit move both extracts and zero-extends the handle-table index.
    try emitMovRawRaw(buffer, scratch_index, handle_raw, false);
    try emitCmpRawImm32(buffer, scratch_index, runtime_value.null_index);
    try emitJccOpcode(buffer, 0x84, cold_label); // je: null
    try emitCmpRawImm32(buffer, scratch_index, runtime.handle_capacity);
    try emitJccOpcode(buffer, 0x83, cold_label); // jae: malformed index

    // Aligned ordinary MOV is an acquire load on x86-64's TSO common path.
    try emitMovRawIndexed(buffer, scratch_descriptor, descriptor_table, scratch_index, 3);
    stats.descriptor_loads += 1;

    // Only stable live entries remain on the hot path. Evacuation is assisted
    // by the cold helper before a derived address is exposed.
    try emitMovRawRaw(buffer, destination_raw, scratch_descriptor, true);
    try emitShiftRawImm(buffer, destination_raw, 5, descriptor_state_shift);
    try emitCmpRawImm32(buffer, destination_raw, @intFromEnum(runtime_value.EntryState.live));
    try emitJccOpcode(buffer, 0x85, cold_label); // jne

    // Compare the handle and descriptor generations before address formation.
    try emitMovRawRaw(buffer, destination_raw, handle_raw, true);
    try emitShiftRawImm(buffer, destination_raw, 5, 32);
    try emitAndRawImm32(buffer, destination_raw, std.math.maxInt(u16));
    try emitMovRawRaw(buffer, scratch_index, scratch_descriptor, true);
    try emitShiftRawImm(buffer, scratch_index, 5, descriptor_generation_shift);
    try emitAndRawImm32(buffer, scratch_index, std.math.maxInt(u16));
    try emitCmpRawRaw(buffer, destination_raw, scratch_index, false);
    try emitJccOpcode(buffer, 0x85, cold_label); // jne

    // Validate the region id before indexing its immutable base table.
    try emitMovRawRaw(buffer, scratch_index, scratch_descriptor, true);
    try emitShiftRawImm(buffer, scratch_index, 5, descriptor_region_shift);
    try emitAndRawImm32(buffer, scratch_index, std.math.maxInt(u8));
    try emitCmpRawImm32(buffer, scratch_index, runtime.region_count);
    try emitJccOpcode(buffer, 0x83, cold_label); // jae

    // Isolate the 36-bit offset without a non-encodable imm64 mask.
    try emitMovRawRaw(buffer, destination_raw, scratch_descriptor, true);
    try emitShiftRawImm(buffer, destination_raw, 4, descriptor_offset_clear_shift);
    try emitShiftRawImm(buffer, destination_raw, 5, descriptor_offset_clear_shift);
    try emitShiftRawImm(buffer, destination_raw, 4, object_alignment_shift);
    try emitMovRawIndexed(buffer, scratch_descriptor, region_table, scratch_index, 3);
    try emitAddRawRaw(buffer, destination_raw, scratch_descriptor);
    try buffer.bindLabel(resume_label);

    stats.fast_resolves += 1;
    stats.cold_resolve_sites += 1;
    stats.native_insts += 27;
}

fn emitColdResolves(buffer: *code_buffer.Buffer, cold: []const ColdResolve, runtime: RuntimeAbi, stats: *Stats) Error!void {
    if (cold.len == 0) return;
    try buffer.alignTo(16, 0x90);
    for (cold) |site| {
        try buffer.bindLabel(site.entry);
        try emitMovRawRaw(buffer, scratch_index, try x64Reg(site.handle), true);
        try emitMovRawImm64(buffer, scratch_descriptor, site.site_key);
        const call_target = try x64Reg(site.destination);
        try emitMovRawImm64(buffer, call_target, @intCast(runtime.slow_resolve_helper));
        try emitCallRaw(buffer, call_target);
        try emitMovRawRaw(buffer, try x64Reg(site.destination), scratch_index, true);
        try emitJump(buffer, site.continuation);
        stats.native_insts += 6;
        stats.jumps += 1;
    }
}

fn checkedLabel(labels: []const code_buffer.LabelId, target: ?cfg.BlockId) Error!code_buffer.LabelId {
    const block = target orelse return error.InvalidMachine;
    if (block >= labels.len) return error.InvalidMachine;
    return labels[block];
}

fn abiParamReg(index: u32) Error!regalloc.PhysReg {
    return switch (builtin.os.tag) {
        .windows => switch (index) {
            0 => .rcx,
            1 => .rdx,
            2 => .r8,
            3 => .r9,
            else => error.UnsupportedInstruction,
        },
        else => switch (index) {
            0 => .rdi,
            1 => .rsi,
            2 => .rdx,
            3 => .rcx,
            4 => .r8,
            5 => .r9,
            else => error.UnsupportedInstruction,
        },
    };
}

const ParamMove = struct {
    dst: regalloc.PhysReg,
    src: regalloc.PhysReg,
    wide: bool,
};

const PARAM_MOVE_SCRATCHES = [_]regalloc.PhysReg{
    .r11, .r10, .rax, .rcx, .rdx, .rsi, .rdi, .r8, .r9,
};

fn moveUsesRegister(moves: []const ParamMove, reg: regalloc.PhysReg) bool {
    for (moves) |move| if (move.src == reg or move.dst == reg) return true;
    return false;
}

fn removeParamMove(moves: []ParamMove, count: *usize, index: usize) void {
    var cursor = index;
    while (cursor + 1 < count.*) : (cursor += 1) moves[cursor] = moves[cursor + 1];
    count.* -= 1;
}

fn recordParamMove(buffer: *code_buffer.Buffer, move: ParamMove, stats: *Stats) Error!void {
    try emitMovRegReg(buffer, move.dst, move.src, move.wide);
    stats.register_moves += 1;
    stats.native_insts += 1;
}

/// Materialize ABI parameters as a true parallel copy. A linear sequence is
/// incorrect when an allocated destination is also a later ABI source. Cycles
/// are broken with a register that is absent from the complete move graph, so
/// the generated function remains frameless and performs no memory traffic.
fn emitParamMoves(buffer: *code_buffer.Buffer, allocation: *const regalloc.Allocation, source: *const machine.Function, stats: *Stats) Error!void {
    var moves: [6]ParamMove = undefined;
    var move_count: usize = 0;
    var param_index: u32 = 0;
    for (source.value_kinds, 0..) |kind, value_id| {
        if (kind != .parameter) continue;
        const dst = try physOf(allocation, @intCast(value_id));
        const src = try abiParamReg(param_index);
        // Runtime root metadata is authoritative even while bytecode type
        // inference remains directional/unknown. Handles always carry a
        // 32-bit generation above their 32-bit table index.
        const wide = isWideType(source.reg_types[value_id]) or source.isGcRoot(@intCast(value_id));
        if (dst != src) {
            if (move_count == moves.len) return error.UnsupportedInstruction;
            moves[move_count] = .{ .dst = dst, .src = src, .wide = wide };
            move_count += 1;
        }
        param_index += 1;
    }

    while (move_count != 0) {
        var ready: ?usize = null;
        for (moves[0..move_count], 0..) |candidate, index| {
            var destination_is_source = false;
            for (moves[0..move_count], 0..) |other, other_index| {
                if (other_index != index and other.src == candidate.dst) {
                    destination_is_source = true;
                    break;
                }
            }
            if (!destination_is_source) {
                ready = index;
                break;
            }
        }

        if (ready) |index| {
            try recordParamMove(buffer, moves[index], stats);
            removeParamMove(&moves, &move_count, index);
            continue;
        }

        const scratch = for (PARAM_MOVE_SCRATCHES) |candidate| {
            if (!moveUsesRegister(moves[0..move_count], candidate)) break candidate;
        } else return error.UnsupportedInstruction;
        const saved_source = moves[0].src;
        try recordParamMove(buffer, .{ .dst = scratch, .src = saved_source, .wide = true }, stats);
        for (moves[0..move_count]) |*move| {
            if (move.src == saved_source) move.src = scratch;
        }
    }
}

fn encodeInst(
    allocator: std.mem.Allocator,
    buffer: *code_buffer.Buffer,
    labels: []const code_buffer.LabelId,
    allocation: *const regalloc.Allocation,
    source: *const machine.Function,
    inst: machine.Inst,
    runtime: ?RuntimeAbi,
    cold: *std.ArrayList(ColdResolve),
    stats: *Stats,
) Error!void {
    switch (inst.opcode) {
        .const_i32 => {
            if (inst.defs.len != 1) return error.InvalidMachine;
            try emitMovRegImm32(buffer, try physOf(allocation, inst.defs[0]), @intCast(inst.imm));
            stats.constants += 1;
            stats.native_insts += 1;
        },
        .mov => {
            if (inst.defs.len != 1 or inst.uses.len != 1) return error.UnsupportedInstruction;
            const dst = try physOf(allocation, inst.defs[0]);
            const src = try physOf(allocation, inst.uses[0]);
            const wide = isWideType(source.reg_types[inst.defs[0]]);
            try emitMovRegReg(buffer, dst, src, wide);
            stats.register_moves += 1;
            stats.native_insts += 1;
        },
        .add_i32, .sub_i32, .mul_i32, .and_i32, .or_i32, .xor_i32 => {
            if (inst.defs.len != 1 or inst.uses.len != 2) return error.InvalidMachine;
            const dst = try physOf(allocation, inst.defs[0]);
            try emitMovRegReg(buffer, dst, try physOf(allocation, inst.uses[0]), false);
            try emitBinaryRegReg(buffer, inst.opcode, dst, try physOf(allocation, inst.uses[1]));
            stats.native_insts += 2;
        },
        // The following resolve performs the same null discrimination and
        // transfers failure to the runtime helper, so no duplicate branch is
        // emitted for this lowering marker.
        .check_null => {},
        .check_bounds => {
            if (inst.defs.len != 0 or inst.uses.len != 1 or inst.address == null) return error.InvalidMachine;
            const abi = runtime orelse return error.MissingRuntimeAbi;
            const layout = try referenceArrayLayout(abi);
            const index_reg = inst.uses[0];
            if (source.isGcRoot(index_reg)) return error.InvalidMachine;
            const address = try x64Reg(try physOf(allocation, inst.address.?));
            const index = try x64Reg(try physOf(allocation, index_reg));
            try emitArrayBoundsCheck(buffer, address, index, layout);
            stats.bounds_checks += 1;
            stats.native_insts += 3;
        },
        .resolve_handle => try emitResolve(allocator, buffer, allocation, inst, runtime orelse return error.MissingRuntimeAbi, cold, stats),
        .array_load_ptr => {
            if (inst.defs.len != 1 or inst.uses.len != 1) return error.InvalidMachine;
            const abi = runtime orelse return error.MissingRuntimeAbi;
            const layout = try referenceArrayLayout(abi);
            if (!source.isGcRoot(inst.defs[0]) or source.isGcRoot(inst.uses[0])) return error.InvalidRuntimeAbi;
            const destination = try x64Reg(try physOf(allocation, inst.defs[0]));
            const address = try x64Reg(try physOf(allocation, inst.address orelse return error.InvalidMachine));
            const index = try x64Reg(try physOf(allocation, inst.uses[0]));
            try emitMovRawRaw(buffer, scratch_index, index, false);
            try emitRexX(buffer, true, destination, scratch_index, address);
            try buffer.emitU8(0x8b);
            try emitIndexedMemoryOperand(buffer, destination, address, scratch_index, 3, @intCast(layout.data_offset));
            stats.array_loads += 1;
            stats.pointer_loads += 1;
            stats.native_insts += 2;
        },
        .field_load_ptr => {
            if (inst.defs.len != 1 or inst.uses.len != 0) return error.InvalidMachine;
            const abi = runtime orelse return error.MissingRuntimeAbi;
            const layout = try fieldLayout(abi, inst.field_idx);
            try verifyFieldType(source, inst.defs[0], layout.storage);
            const dst = try x64Reg(try physOf(allocation, inst.defs[0]));
            const address = try x64Reg(try physOf(allocation, inst.address orelse return error.InvalidMachine));
            try emitLoadField(buffer, dst, address, @intCast(layout.offset), layout.storage);
            stats.pointer_loads += 1;
            stats.native_insts += 1;
        },
        .satb_pre_write => {
            if (inst.defs.len != 0) return error.InvalidMachine;
            const abi = runtime orelse return error.MissingRuntimeAbi;
            const address = try x64Reg(try physOf(allocation, inst.address orelse return error.InvalidMachine));
            if (inst.field_idx) |field_idx| {
                if (inst.uses.len != 0) return error.InvalidMachine;
                const layout = try fieldLayout(abi, field_idx);
                if (layout.storage != .reference) return error.InvalidRuntimeAbi;
                try emitLeaRawMemory(buffer, scratch_index, address, @intCast(layout.offset));
            } else {
                if (inst.uses.len != 1 or source.isGcRoot(inst.uses[0])) return error.InvalidMachine;
                const layout = try referenceArrayLayout(abi);
                const index = try x64Reg(try physOf(allocation, inst.uses[0]));
                try emitArrayElementAddress(buffer, scratch_index, address, index, layout);
            }
            const repeat_proven = inst.pre_write == .satb_repeat_guarded;
            if (inst.pre_write == .none) return error.InvalidMachine;
            try emitMovRawImm64(buffer, scratch_descriptor, @intFromBool(repeat_proven));
            try emitBarrierCall(buffer, abi.satb_pre_write_helper);
            stats.satb_barriers += 1;
            stats.satb_repeat_barriers += @intFromBool(repeat_proven);
            stats.native_insts += 6;
        },
        .array_store_ptr => {
            if (inst.defs.len != 0 or inst.uses.len != 2) return error.InvalidMachine;
            const abi = runtime orelse return error.MissingRuntimeAbi;
            const layout = try referenceArrayLayout(abi);
            if (!source.isGcRoot(inst.uses[0]) or source.isGcRoot(inst.uses[1])) return error.InvalidRuntimeAbi;
            const value = try x64Reg(try physOf(allocation, inst.uses[0]));
            const index = try x64Reg(try physOf(allocation, inst.uses[1]));
            const address = try x64Reg(try physOf(allocation, inst.address orelse return error.InvalidMachine));
            try emitArrayElementAddress(buffer, scratch_index, address, index, layout);
            try emitMovMemoryFromRaw(buffer, scratch_index, 0, value, true);
            stats.array_stores += 1;
            stats.pointer_stores += 1;
            stats.native_insts += 3;
        },
        .field_store_ptr => {
            if (inst.defs.len != 0 or inst.uses.len != 1) return error.InvalidMachine;
            const abi = runtime orelse return error.MissingRuntimeAbi;
            const layout = try fieldLayout(abi, inst.field_idx);
            try verifyFieldType(source, inst.uses[0], layout.storage);
            const src = try x64Reg(try physOf(allocation, inst.uses[0]));
            const address = try x64Reg(try physOf(allocation, inst.address orelse return error.InvalidMachine));
            try emitStoreField(buffer, address, @intCast(layout.offset), src, layout.storage);
            stats.pointer_stores += 1;
            stats.native_insts += 1;
        },
        .static_satb_pre_write => {
            if (inst.defs.len != 0 or inst.uses.len != 0 or inst.pre_write != .satb_guarded) return error.InvalidMachine;
            const abi = runtime orelse return error.MissingRuntimeAbi;
            const layout = try staticFieldLayout(abi, inst.field_idx);
            if (layout.storage != .reference) return error.InvalidRuntimeAbi;
            try emitMovRawImm64(buffer, scratch_index, layout.address);
            try emitMovRawImm64(buffer, scratch_descriptor, 0);
            try emitBarrierCall(buffer, abi.satb_pre_write_helper);
            stats.satb_barriers += 1;
            stats.native_insts += 5;
        },
        .static_store => {
            if (inst.defs.len != 0 or inst.uses.len != 1) return error.InvalidMachine;
            const abi = runtime orelse return error.MissingRuntimeAbi;
            const layout = try staticFieldLayout(abi, inst.field_idx);
            try verifyFieldType(source, inst.uses[0], layout.storage);
            const value = try x64Reg(try physOf(allocation, inst.uses[0]));
            try emitMovRawImm64(buffer, scratch_index, layout.address);
            try emitStoreField(buffer, scratch_index, 0, value, layout.storage);
            stats.pointer_stores += @intFromBool(layout.storage == .reference);
            stats.native_insts += 2;
        },
        .static_root_post_write => {
            if (inst.defs.len != 0 or inst.uses.len != 1 or inst.post_write != .root_guarded) return error.InvalidMachine;
            const abi = runtime orelse return error.MissingRuntimeAbi;
            const layout = try staticFieldLayout(abi, inst.field_idx);
            if (layout.storage != .reference or !source.isGcRoot(inst.uses[0])) return error.InvalidRuntimeAbi;
            const stored = try x64Reg(try physOf(allocation, inst.uses[0]));
            try emitMovRawImm64(buffer, scratch_index, layout.address);
            try emitMovRawRaw(buffer, scratch_descriptor, stored, true);
            try emitBarrierCall(buffer, abi.static_root_post_write_helper);
            stats.static_root_barriers += 1;
            stats.native_insts += 5;
        },
        .card_mark => {
            if (inst.defs.len != 0 or inst.uses.len != 1) return error.InvalidMachine;
            const abi = runtime orelse return error.MissingRuntimeAbi;
            const destination = try x64Reg(try physOf(allocation, inst.state_handle orelse return error.InvalidMachine));
            const stored = try x64Reg(try physOf(allocation, inst.uses[0]));
            const repeat_proven = inst.post_write == .card_repeat_guarded;
            if (inst.post_write == .none) return error.InvalidMachine;
            try emitMovRawRaw(buffer, scratch_index, destination, true);
            try emitMovRawRaw(buffer, scratch_descriptor, stored, true);
            try emitBarrierCall(buffer, if (repeat_proven) abi.card_mark_repeat_helper else abi.card_mark_helper);
            stats.card_barriers += 1;
            stats.card_repeat_barriers += @intFromBool(repeat_proven);
            stats.native_insts += 6;
        },
        .jump => {
            try emitJump(buffer, try checkedLabel(labels, inst.target));
            stats.jumps += 1;
            stats.native_insts += 1;
        },
        .branch => {
            const condition = inst.condition orelse return error.InvalidMachine;
            if (inst.uses.len == 1) {
                try emitCmpRegZero(buffer, try physOf(allocation, inst.uses[0]));
            } else if (inst.uses.len == 2) {
                try emitCmpRegs(buffer, try physOf(allocation, inst.uses[0]), try physOf(allocation, inst.uses[1]));
            } else {
                return error.InvalidMachine;
            }
            try emitJcc(buffer, condition, try checkedLabel(labels, inst.target));
            try emitJump(buffer, try checkedLabel(labels, inst.false_target));
            stats.branches += 1;
            stats.jumps += 1;
            stats.native_insts += 3;
        },
        .ret => {
            if (inst.uses.len == 1) {
                const src = try physOf(allocation, inst.uses[0]);
                const wide = isWideType(source.reg_types[inst.uses[0]]);
                try emitMovRegReg(buffer, .rax, src, wide);
            }
            if (inst.uses.len > 1) return error.UnsupportedInstruction;
            try emitRet(buffer);
            stats.returns += 1;
            stats.native_insts += 1;
        },
        else => return error.UnsupportedInstruction,
    }
}

fn ensureNoSpills(allocation: *const regalloc.Allocation) Error!void {
    for (allocation.intervals) |interval| {
        switch (allocation.locations[interval.reg]) {
            .spill => return error.SpillsUnsupported,
            .phys => {},
            .none => return error.InvalidMachine,
        }
    }
}

fn verifyFoldedNullChecks(source: *const machine.Function) Error!void {
    for (source.blocks) |block| {
        for (block.insts) |inst| {
            if (inst.opcode != .check_null) continue;
            if (inst.uses.len != 1) return error.InvalidMachine;
            var protected = false;
            for (source.blocks) |definition_block| {
                if (!source.source.source.tree.dominates(definition_block.id, block.id)) continue;
                for (definition_block.insts) |candidate| {
                    if (candidate.opcode == .resolve_handle and candidate.uses.len == 1 and candidate.uses[0] == inst.uses[0]) {
                        protected = true;
                        break;
                    }
                }
                if (protected) break;
            }
            if (!protected) return error.InvalidMachine;
        }
    }
}

const PendingRootMap = struct {
    resolve_id: u32,
    roots: []runtime_stack_map.RootLocation,
};

fn rootLocationLess(_: void, a: runtime_stack_map.RootLocation, b: runtime_stack_map.RootLocation) bool {
    return a.payload < b.payload;
}

fn pendingRootMapLess(_: void, a: PendingRootMap, b: PendingRootMap) bool {
    return a.resolve_id < b.resolve_id;
}

fn buildRootMaps(
    allocator: std.mem.Allocator,
    source: *const machine.Function,
    allocation: *const regalloc.Allocation,
    stats: *Stats,
) Error!?runtime_stack_map.Table {
    if (source.stats.resolves == 0) return null;
    if (source.blocks.len != 1) return error.RootMapsRequireSingleBlock;

    var pending: std.ArrayList(PendingRootMap) = .empty;
    defer {
        for (pending.items) |site| allocator.free(site.roots);
        pending.deinit(allocator);
    }

    var position: u32 = 2;
    for (source.blocks[0].insts) |inst| {
        if (inst.opcode == .resolve_handle) {
            const resolve_id = inst.resolve_id orelse return error.InvalidMachine;
            const canonical = inst.state_handle orelse return error.InvalidMachine;
            const canonical_phys = try x64Reg(try physOf(allocation, canonical));
            var roots: std.ArrayList(runtime_stack_map.RootLocation) = .empty;
            errdefer roots.deinit(allocator);
            var found_canonical = false;
            for (allocation.intervals) |interval| {
                if (!interval.covers(position) or !source.isGcRoot(interval.reg)) continue;
                const physical = try x64Reg(try physOf(allocation, interval.reg));
                try roots.append(allocator, runtime_stack_map.RootLocation.nativeRegister(physical));
                if (physical == canonical_phys) found_canonical = true;
            }
            if (!found_canonical) return error.InvalidMachine;
            std.mem.sort(runtime_stack_map.RootLocation, roots.items, {}, rootLocationLess);
            const owned = try roots.toOwnedSlice(allocator);
            pending.append(allocator, .{ .resolve_id = resolve_id, .roots = owned }) catch |err| {
                allocator.free(owned);
                return err;
            };
            stats.root_map_sites += 1;
            stats.root_map_locations += @intCast(owned.len);
        }
        position += 2;
    }
    if (pending.items.len != source.stats.resolves) return error.InvalidMachine;
    std.mem.sort(PendingRootMap, pending.items, {}, pendingRootMapLess);

    var specs: std.ArrayList(runtime_stack_map.MapSpec) = .empty;
    defer specs.deinit(allocator);
    for (pending.items) |site| {
        try specs.append(allocator, .{ .pc_offset = site.resolve_id, .roots = site.roots });
    }
    return try runtime_stack_map.Table.init(allocator, specs.items, .{
        .native_register_count = 16,
        .interpreter_register_count = 0,
        .max_frame_depth = 0,
        .max_shadow_roots = 0,
    });
}

pub fn encodeWithOptions(allocator: std.mem.Allocator, source: *const machine.Function, options: Options) Error!Function {
    source.verify() catch return error.InvalidMachine;
    if (source.edges.len != 0) return error.UnsupportedInstruction;
    if (options.runtime) |runtime| try runtime.verify();
    if (source.stats.resolves != 0 and options.runtime == null) return error.MissingRuntimeAbi;
    try verifyFoldedNullChecks(source);

    // r10/r11 are private ABI scratch for barriers as well as resolution.
    // Static-only methods have no resolve op, but must obey the same contract.
    var allocation = try regalloc.allocate(allocator, source, .{ .gp_registers = &RESOLVER_GP_REGS });
    errdefer allocation.deinit();
    try ensureNoSpills(&allocation);

    var buffer = code_buffer.Buffer.init(allocator);
    errdefer buffer.deinit();

    const labels = try allocator.alloc(code_buffer.LabelId, source.blocks.len);
    errdefer allocator.free(labels);
    for (labels) |*label| label.* = try buffer.newLabel();

    var stats: Stats = .{ .blocks = @intCast(source.blocks.len) };
    var root_maps = try buildRootMaps(allocator, source, &allocation, &stats);
    errdefer if (root_maps) |*maps| maps.deinit();
    try emitParamMoves(&buffer, &allocation, source, &stats);

    var cold: std.ArrayList(ColdResolve) = .empty;
    defer cold.deinit(allocator);

    for (source.blocks) |block| {
        try buffer.alignTo(16, 0x90);
        try buffer.bindLabel(labels[block.id]);
        for (block.insts) |inst| try encodeInst(allocator, &buffer, labels, &allocation, source, inst, options.runtime, &cold, &stats);
    }
    if (options.runtime) |runtime| try emitColdResolves(&buffer, cold.items, runtime, &stats);
    try buffer.verify();
    stats.bytes = buffer.len();

    return .{
        .allocator = allocator,
        .source = source,
        .allocation = allocation,
        .buffer = buffer,
        .block_labels = labels,
        .root_maps = root_maps,
        .stats = stats,
    };
}

pub fn encode(allocator: std.mem.Allocator, source: *const machine.Function) Error!Function {
    return encodeWithOptions(allocator, source, .{});
}

fn optimizedMachine(allocator: std.mem.Allocator, insts: []const Instruction) !*optimizer.OptimizedFunction {
    return try optimizer.optimize(allocator, insts, &.{}, .{});
}

fn expectBinaryI32(insts: []const Instruction, a: i32, b: i32, expected: i32) !void {
    if (builtin.cpu.arch != .x86_64) return error.SkipZigTest;

    var optimized = try optimizedMachine(std.testing.allocator, insts);
    defer optimized.deinit();
    var native = try encode(std.testing.allocator, &optimized.machine);
    defer native.deinit();
    const bytes = try native.finalize();
    defer std.testing.allocator.free(bytes);
    try std.testing.expect(std.mem.indexOfScalar(u8, bytes, 0x55) == null);

    var cache = jit_memory.Cache.init(std.testing.allocator);
    defer cache.deinit();
    const allocation = try cache.addBytes(bytes);
    const Fn = fn (i32, i32) callconv(.c) i32;
    try std.testing.expectEqual(expected, allocation.typedEntry(Fn)(a, b));
}

test "x64_register_encoder executes parameter arithmetic without frame prologue" {
    try expectBinaryI32(&[_]Instruction{
        .{ .add_int = .{ .dest = 2, .src1 = 0, .src2 = 1 } },
        .{ .return_ = .{ .src = 2 } },
    }, 30, 12, 42);
    try expectBinaryI32(&[_]Instruction{
        .{ .mul_int = .{ .dest = 2, .src1 = 0, .src2 = 1 } },
        .{ .return_ = .{ .src = 2 } },
    }, -7, 6, -42);
}

test "x64_register_encoder rejects spills explicitly" {
    const insts = [_]Instruction{
        .{ .add_int = .{ .dest = 2, .src1 = 0, .src2 = 1 } },
        .{ .return_ = .{ .src = 2 } },
    };
    var optimized = try optimizedMachine(std.testing.allocator, &insts);
    defer optimized.deinit();
    var allocation = try regalloc.allocate(std.testing.allocator, &optimized.machine, .{ .gp_registers = &[_]regalloc.PhysReg{} });
    defer allocation.deinit();
    try std.testing.expect(allocation.stats.spills > 0);
}

test "x64_register_encoder print helper emits stable summary" {
    const insts = [_]Instruction{
        .{ .const_ = .{ .dest = 0, .value = 11 } },
        .{ .return_ = .{ .src = 0 } },
    };
    var optimized = try optimizedMachine(std.testing.allocator, &insts);
    defer optimized.deinit();
    var native = try encode(std.testing.allocator, &optimized.machine);
    defer native.deinit();

    var storage: [1024]u8 = undefined;
    var stream = std.Io.Writer.fixed(&storage);
    try native.print(&stream);
    const output = stream.buffered();
    try std.testing.expect(std.mem.indexOf(u8, output, "x64_register_encoder bytes=") != null);
    try std.testing.expect(std.mem.indexOf(u8, output, "linear_scan intervals=") != null);
}

const test_field_layouts = [_]FieldLayout{
    .{ .offset = 8, .storage = .i32 },
    .{ .offset = 16, .storage = .i32 },
    .{ .offset = 24, .storage = .i32 },
    .{ .offset = 32, .storage = .i32 },
};

fn testRuntimeAbi() RuntimeAbi {
    return .{
        .handle_capacity = 4096,
        .region_count = 8,
        .slow_resolve_helper = 0x1234_5678,
        .satb_pre_write_helper = 0x2345_6789,
        .card_mark_helper = 0x3456_789a,
        .card_mark_repeat_helper = 0x4567_89ab,
        .field_layouts = &test_field_layouts,
    };
}

fn resolverEncodingFailureProbe(allocator: std.mem.Allocator, source: *const machine.Function) !void {
    var native = try encodeWithOptions(allocator, source, .{ .runtime = testRuntimeAbi() });
    defer native.deinit();
    const bytes = try native.finalize();
    defer allocator.free(bytes);
}

test "x64_register_encoder emits one descriptor load for reused hot-loop resolve" {
    if (builtin.cpu.arch != .x86_64) return error.SkipZigTest;
    const insts = [_]Instruction{
        .{ .iget = .{ .field_idx = 1, .dest_or_src = 1, .obj = 0 } },
        .{ .iget = .{ .field_idx = 2, .dest_or_src = 2, .obj = 0 } },
        .return_void,
    };
    var optimized = try optimizedMachine(std.testing.allocator, &insts);
    defer optimized.deinit();
    try std.testing.expectEqual(@as(u32, 1), optimized.machine.stats.resolves);
    try std.testing.expectEqual(@as(u32, 2), optimized.machine.stats.pointer_accesses);

    try std.testing.expectError(error.MissingRuntimeAbi, encode(std.testing.allocator, &optimized.machine));
    var native = try encodeWithOptions(std.testing.allocator, &optimized.machine, .{ .runtime = testRuntimeAbi() });
    defer native.deinit();
    try native.verify();
    try std.testing.expectEqual(@as(u32, 1), native.stats.descriptor_loads);
    try std.testing.expectEqual(@as(u32, 1), native.stats.fast_resolves);
    try std.testing.expectEqual(@as(u32, 1), native.stats.cold_resolve_sites);
    try std.testing.expectEqual(@as(u32, 2), native.stats.pointer_loads);

    const bytes = try native.finalize();
    defer std.testing.allocator.free(bytes);
    // mov r11, [r14 + r10*8]: the sole atomic descriptor snapshot.
    try std.testing.expectEqual(@as(usize, 1), std.mem.count(u8, bytes, &.{ 0x4f, 0x8b, 0x1c, 0xd6 }));
    // mov r11, [r13 + r10*8]: immutable region-base lookup.
    try std.testing.expectEqual(@as(usize, 1), std.mem.count(u8, bytes, &.{ 0x4f, 0x8b, 0x5c, 0xd5, 0x00 }));
    try std.testing.checkAllAllocationFailures(std.testing.allocator, resolverEncodingFailureProbe, .{&optimized.machine});
}

test "x64_register_encoder validates descriptor layout and field metadata" {
    const descriptor = runtime_value.LocationDescriptor{
        .offset_units = 0x12345,
        .region_id = 0x67,
        .generation = 0x89ab,
        .state = .live,
    };
    const expected = @as(u64, 0x12345) |
        (@as(u64, 0x67) << descriptor_region_shift) |
        (@as(u64, 0x89ab) << descriptor_generation_shift) |
        (@as(u64, @intFromEnum(runtime_value.EntryState.live)) << descriptor_state_shift);
    try std.testing.expectEqual(expected, descriptor.bits());

    const insts = [_]Instruction{
        .{ .iget = .{ .field_idx = 3, .dest_or_src = 1, .obj = 0 } },
        .return_void,
    };
    var optimized = try optimizedMachine(std.testing.allocator, &insts);
    defer optimized.deinit();
    var bad = testRuntimeAbi();
    bad.field_layouts = &.{
        .{ .offset = 8, .storage = .i32 },
        .{ .offset = 16, .storage = .i32 },
    };
    try std.testing.expectError(error.MissingFieldLayout, encodeWithOptions(std.testing.allocator, &optimized.machine, .{ .runtime = bad }));

    var malformed_array = testRuntimeAbi();
    malformed_array.reference_array_layout = .{
        .length_offset = 0,
        .data_offset = 8,
        .element_stride = 4,
    };
    try std.testing.expectError(
        error.InvalidRuntimeAbi,
        encodeWithOptions(std.testing.allocator, &optimized.machine, .{ .runtime = malformed_array }),
    );
}

test "x64_register_encoder emits scalar and fully barriered reference stores" {
    if (builtin.cpu.arch != .x86_64) return error.SkipZigTest;
    const scalar = [_]Instruction{
        .{ .iput = .{ .field_idx = 1, .dest_or_src = 1, .obj = 0 } },
        .return_void,
    };
    var scalar_optimized = try optimizedMachine(std.testing.allocator, &scalar);
    defer scalar_optimized.deinit();
    var native = try encodeWithOptions(std.testing.allocator, &scalar_optimized.machine, .{ .runtime = testRuntimeAbi() });
    defer native.deinit();
    try std.testing.expectEqual(@as(u32, 1), native.stats.descriptor_loads);
    try std.testing.expectEqual(@as(u32, 1), native.stats.pointer_stores);
    try std.testing.expectEqual(@as(u32, 0), native.stats.pointer_loads);
    const bytes = try native.finalize();
    defer std.testing.allocator.free(bytes);

    const reference = [_]Instruction{
        .{ .iput_object = .{ .field_idx = 1, .dest_or_src = 1, .obj = 0 } },
        .return_void,
    };
    var reference_optimized = try optimizedMachine(std.testing.allocator, &reference);
    defer reference_optimized.deinit();
    var reference_layouts = test_field_layouts;
    reference_layouts[1].storage = .reference;
    var reference_abi = testRuntimeAbi();
    reference_abi.field_layouts = &reference_layouts;
    var reference_native = try encodeWithOptions(std.testing.allocator, &reference_optimized.machine, .{ .runtime = reference_abi });
    defer reference_native.deinit();
    try std.testing.expectEqual(@as(u32, 1), reference_native.stats.pointer_stores);
    try std.testing.expectEqual(@as(u32, 1), reference_native.stats.satb_barriers);
    try std.testing.expectEqual(@as(u32, 1), reference_native.stats.card_barriers);
    const reference_bytes = try reference_native.finalize();
    defer std.testing.allocator.free(reference_bytes);

    var missing = reference_abi;
    missing.card_mark_helper = 0;
    try std.testing.expectError(
        error.MissingBarrierHelper,
        encodeWithOptions(std.testing.allocator, &reference_optimized.machine, .{ .runtime = missing }),
    );
}

test "x64_register_encoder honors signed and narrow field storage" {
    if (builtin.cpu.arch != .x86_64) return error.SkipZigTest;

    var byte_layouts = test_field_layouts;
    byte_layouts[1].storage = .i8;
    var byte_abi = testRuntimeAbi();
    byte_abi.field_layouts = &byte_layouts;
    const byte_load = [_]Instruction{
        .{ .iget_byte = .{ .field_idx = 1, .dest_or_src = 1, .obj = 0 } },
        .{ .return_ = .{ .src = 1 } },
    };
    var byte_optimized = try optimizedMachine(std.testing.allocator, &byte_load);
    defer byte_optimized.deinit();
    var byte_native = try encodeWithOptions(std.testing.allocator, &byte_optimized.machine, .{ .runtime = byte_abi });
    defer byte_native.deinit();
    const byte_code = try byte_native.finalize();
    defer std.testing.allocator.free(byte_code);
    try std.testing.expect(std.mem.indexOf(u8, byte_code, &.{ 0x0f, 0xbe }) != null); // movsx r32, byte ptr

    var short_layouts = test_field_layouts;
    short_layouts[1].storage = .i16;
    var short_abi = testRuntimeAbi();
    short_abi.field_layouts = &short_layouts;
    const short_store = [_]Instruction{
        .{ .iput_short = .{ .field_idx = 1, .dest_or_src = 1, .obj = 0 } },
        .return_void,
    };
    var short_optimized = try optimizedMachine(std.testing.allocator, &short_store);
    defer short_optimized.deinit();
    var short_native = try encodeWithOptions(std.testing.allocator, &short_optimized.machine, .{ .runtime = short_abi });
    defer short_native.deinit();
    const short_code = try short_native.finalize();
    defer std.testing.allocator.free(short_code);
    try std.testing.expect(std.mem.indexOf(u8, short_code, &.{ 0x66, 0x89 }) != null); // mov word ptr, r16
}
