//! Register-allocated x86-64 encoder.
//!
//! This backend emits a register-allocated native subset from the
//! register-machine IR. Spill slots live in one deterministic, 16-byte-aligned
//! managed frame addressed from rsp. Operations whose private runtime ABI does
//! not yet have a proven spill-scratch policy continue to fail explicitly.

const std = @import("std");
const builtin = @import("builtin");
const cfg = @import("cfg");
const code_buffer = @import("code_buffer");
const jit_memory = @import("jit_memory");
const machine = @import("machine_bridge");
const optimizer = @import("optimizer");
const regalloc = @import("regalloc");
const runtime_stack_map = @import("runtime_stack_map");
const runtime_jit = @import("runtime_jit");
const runtime_deopt = @import("runtime_deopt");
const runtime_value = @import("runtime_value");
const Instruction = @import("instructions").Instruction;

pub const Error = code_buffer.Error || regalloc.Error || runtime_stack_map.Error || runtime_deopt.Error || error{
    InvalidMachine,
    AliasedDeoptValue,
    DeadDeoptValue,
    InvalidDeoptMetadata,
    InvalidRuntimeAbi,
    MissingBarrierHelper,
    MissingExceptionHelper,
    MissingArrayLayout,
    MissingFieldLayout,
    MissingStaticFieldLayout,
    MissingRuntimeAbi,
    MissingDeoptSafepoint,
    SpillsUnsupported,
    UnsupportedInstruction,
};

pub const DeoptSource = union(enum) {
    machine_register: machine.RegId,
    constant: u64,
};

pub const DeoptValueSpec = struct {
    vreg: u16,
    kind: runtime_deopt.ValueKind,
    source: DeoptSource,
};

pub const DeoptInlineFrameSpec = struct {
    method_id: u32,
    dex_pc: u32,
    register_count: u16,
    values: []const DeoptValueSpec,
};

pub const DeoptPointSpec = struct {
    id: u32,
    /// Existing machine safepoint carrying this state. Omit when the compiler
    /// must create a position at `block_entry`.
    safepoint_id: ?u32 = null,
    /// Compiler-owned position immediately before the first instruction in a
    /// block. Exactly one of this field and `safepoint_id` must be present.
    block_entry: ?cfg.BlockId = null,
    method_id: u32,
    dex_pc: u32,
    values: []const DeoptValueSpec,
    /// Outer-to-inner caller activations materialized before the leaf frame.
    inline_frames: []const DeoptInlineFrameSpec = &.{},
};

pub const DeoptOptions = struct {
    points: []const DeoptPointSpec,
    register_count: u16,
    max_dex_pc: u32,
};

pub const OsrEntrySpec = struct {
    point_id: u32,
    block: cfg.BlockId,
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
    bounds_exception_helper: usize = 0,
    satb_pre_write_helper: usize = 0,
    card_mark_helper: usize = 0,
    card_mark_repeat_helper: usize = 0,
    static_root_post_write_helper: usize = 0,
    deopt_epoch_address: usize = 0,
    compiled_deopt_epoch: u64 = 0,
    deopt_helper: usize = 0,
    reference_array_layout: ?ReferenceArrayLayout = null,
    /// Immutable layouts indexed by resolved field id.
    field_layouts: []const FieldLayout,
    /// Immutable layouts indexed by resolved static field id. Static storage
    /// is pinned and never represented by a managed destination Handle.
    static_field_layouts: []const StaticFieldLayout = &.{},

    fn verify(self: RuntimeAbi) Error!void {
        if (self.handle_capacity == 0 or self.region_count == 0 or self.region_count > 256) return error.InvalidRuntimeAbi;
        if (self.slow_resolve_helper == 0) return error.InvalidRuntimeAbi;
        const has_deopt = self.deopt_epoch_address != 0 or self.deopt_helper != 0;
        if (has_deopt and (self.deopt_epoch_address == 0 or self.deopt_helper == 0 or
            !std.mem.isAligned(self.deopt_epoch_address, @alignOf(u64)))) return error.InvalidRuntimeAbi;
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
    deopt: ?DeoptOptions = null,
    osr_entries: []const OsrEntrySpec = &.{},
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
    loop_epoch_guards: u32 = 0,
    loop_epoch_slow_sites: u32 = 0,
    pointer_loads: u32 = 0,
    pointer_stores: u32 = 0,
    bounds_checks: u32 = 0,
    bounds_exception_sites: u32 = 0,
    array_loads: u32 = 0,
    array_stores: u32 = 0,
    satb_barriers: u32 = 0,
    satb_repeat_barriers: u32 = 0,
    card_barriers: u32 = 0,
    card_repeat_barriers: u32 = 0,
    static_root_barriers: u32 = 0,
    root_map_sites: u32 = 0,
    root_map_locations: u32 = 0,
    edge_copy_sites: u32 = 0,
    edge_copy_moves: u32 = 0,
    edge_copy_cycles: u32 = 0,
    deopt_points: u32 = 0,
    deopt_frames: u32 = 0,
    deopt_values: u32 = 0,
    deopt_xmm_values: u32 = 0,
    deopt_stack_values: u32 = 0,
    deopt_block_entries: u32 = 0,
    deopt_guards: u32 = 0,
    deopt_traps: u32 = 0,
    osr_entries: u32 = 0,
    osr_frame_landings: u32 = 0,
    osr_landing_safepoints: u32 = 0,
    osr_derived_rematerializations: u32 = 0,
    osr_remat_restart_edges: u32 = 0,
    frame_bytes: u32 = 0,
    nonvolatile_frame_bytes: u32 = 0,
    spill_loads: u32 = 0,
    spill_stores: u32 = 0,
    xmm_insts: u32 = 0,
    xmm_spill_loads: u32 = 0,
    xmm_spill_stores: u32 = 0,
    xmm_negations: u32 = 0,
    xmm_comparisons: u32 = 0,
    xmm_conversions: u32 = 0,
    xmm_saturating_conversions: u32 = 0,
};

pub const Function = struct {
    allocator: std.mem.Allocator,
    source: *const machine.Function,
    allocation: regalloc.Allocation,
    buffer: code_buffer.Buffer,
    block_labels: []code_buffer.LabelId,
    root_maps: ?runtime_stack_map.Table,
    deopt_table: ?runtime_deopt.Table,
    osr_entries: []runtime_jit.OsrEntry,
    stats: Stats,

    pub fn deinit(self: *Function) void {
        if (self.root_maps) |*maps| maps.deinit();
        if (self.deopt_table) |*table| table.deinit();
        self.allocator.free(self.osr_entries);
        self.allocator.free(self.block_labels);
        self.buffer.deinit();
        self.allocation.deinit();
        self.* = undefined;
    }

    pub fn verify(self: *Function) Error!void {
        self.source.verify() catch return error.InvalidMachine;
        try self.allocation.verify();
        if (self.block_labels.len != self.source.blocks.len) return error.InvalidMachine;
        if (!std.mem.isAligned(self.stats.frame_bytes, 16) or
            (self.stats.nonvolatile_frame_bytes != 0 and self.stats.nonvolatile_frame_bytes != 16) or
            ((self.stats.frame_bytes == 0) != (self.allocation.stats.spills == 0))) return error.InvalidMachine;
        const xmm_spill_insts = std.math.add(u32, self.stats.xmm_spill_loads, self.stats.xmm_spill_stores) catch return error.InvalidMachine;
        const xmm_unary_and_compare = std.math.add(u32, self.stats.xmm_negations, self.stats.xmm_comparisons) catch return error.InvalidMachine;
        const xmm_semantic_sites = std.math.add(u32, xmm_unary_and_compare, self.stats.xmm_conversions) catch return error.InvalidMachine;
        if (self.stats.xmm_spill_loads > self.stats.spill_loads or
            self.stats.xmm_spill_stores > self.stats.spill_stores or
            self.stats.xmm_saturating_conversions > self.stats.xmm_conversions or
            xmm_spill_insts > self.stats.xmm_insts or xmm_semantic_sites > self.stats.xmm_insts) return error.InvalidMachine;
        if (self.stats.edge_copy_sites != self.source.edges.len or self.stats.edge_copy_moves != self.source.stats.edge_moves) return error.InvalidMachine;
        const resolve_sites = std.math.add(u32, self.source.stats.resolves, self.source.stats.loop_epoch_guards) catch return error.InvalidMachine;
        const emitted_resolves = std.math.add(u32, self.source.stats.resolves, self.stats.osr_derived_rematerializations) catch return error.InvalidMachine;
        const emitted_cold_resolves = std.math.add(u32, resolve_sites, self.stats.osr_derived_rematerializations) catch return error.InvalidMachine;
        const machine_safepoint_sites = std.math.add(u32, resolve_sites, self.source.stats.bounds_exception_sites) catch return error.InvalidMachine;
        const deopt_sites = std.math.add(u32, machine_safepoint_sites, self.stats.deopt_block_entries) catch return error.InvalidMachine;
        const safepoint_sites = std.math.add(u32, deopt_sites, self.stats.osr_landing_safepoints) catch return error.InvalidMachine;
        if (self.stats.descriptor_loads != emitted_resolves or self.stats.fast_resolves != emitted_resolves or
            self.stats.cold_resolve_sites != emitted_cold_resolves or self.stats.loop_epoch_guards != self.source.stats.loop_epoch_guards or
            self.stats.loop_epoch_slow_sites != self.source.stats.loop_epoch_guards or
            self.stats.bounds_exception_sites != self.source.stats.bounds_exception_sites) return error.InvalidMachine;
        if ((safepoint_sites == 0) != (self.root_maps == null)) return error.InvalidMachine;
        if (self.root_maps) |maps| {
            if (maps.records.len != safepoint_sites or self.stats.root_map_sites != safepoint_sites) return error.InvalidMachine;
            if (maps.locations.len != self.stats.root_map_locations) return error.InvalidMachine;
        }
        if ((self.stats.deopt_points == 0) != (self.deopt_table == null)) return error.InvalidMachine;
        if (self.deopt_table) |table| {
            if (table.records.len != self.stats.deopt_points or table.frames.len != self.stats.deopt_frames or
                table.values.len != self.stats.deopt_values) return error.InvalidMachine;
            if (self.stats.deopt_block_entries > self.stats.deopt_points or
                self.stats.deopt_guards != self.stats.deopt_points or self.stats.deopt_traps != self.stats.deopt_points) return error.InvalidMachine;
            const maps = &(self.root_maps orelse return error.InvalidMachine);
            table.validateStackMaps(maps, false) catch return error.InvalidMachine;
            table.validateAllLinked(maps) catch return error.InvalidMachine;
        }
        if (self.osr_entries.len != self.stats.osr_entries) return error.InvalidMachine;
        if (self.stats.osr_landing_safepoints > self.stats.osr_entries or
            self.stats.osr_derived_rematerializations < self.stats.osr_landing_safepoints or
            self.stats.osr_remat_restart_edges != self.stats.osr_derived_rematerializations) return error.InvalidMachine;
        const has_native_frame = self.stats.frame_bytes != 0 or self.stats.nonvolatile_frame_bytes != 0;
        if ((!has_native_frame and self.stats.osr_frame_landings != 0) or
            (has_native_frame and self.stats.osr_frame_landings != self.stats.osr_entries)) return error.InvalidMachine;
        for (self.osr_entries, 0..) |entry, index| {
            if (entry.code_offset == 0 or entry.code_offset >= self.buffer.len() or !std.mem.isAligned(entry.code_offset, 16)) {
                return error.InvalidMachine;
            }
            if (index != 0 and self.osr_entries[index - 1].point_id >= entry.point_id) return error.InvalidMachine;
            const table = &(self.deopt_table orelse return error.InvalidMachine);
            _ = table.find(entry.point_id) catch return error.InvalidMachine;
        }
        try self.buffer.verify();
    }

    pub fn finalize(self: *Function) Error![]u8 {
        try self.verify();
        const bytes = try self.buffer.finalize();
        self.stats.bytes = @intCast(bytes.len);
        return bytes;
    }

    pub fn osrEntry(self: *const Function, point_id: u32) Error!runtime_jit.OsrEntry {
        for (self.osr_entries) |entry| {
            if (entry.point_id == point_id) return entry;
            if (entry.point_id > point_id) break;
        }
        return error.InvalidDeoptMetadata;
    }

    pub fn print(self: *const Function, writer: anytype) !void {
        try writer.print(
            "x64_register_encoder bytes={d} blocks={d} native_insts={d} moves={d} constants={d} returns={d} jumps={d} branches={d} descriptor_loads={d} fast_resolves={d} cold_resolves={d} loop_guards={d} loop_slow={d} bounds={d} bounds_exceptions={d} array_loads={d} array_stores={d} ptr_loads={d} ptr_stores={d} satb={d} satb_repeats={d} cards={d} card_repeats={d} root_sites={d} root_locations={d} edge_sites={d} edge_moves={d} edge_cycles={d} frame_bytes={d} spill_loads={d} spill_stores={d}",
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
                self.stats.loop_epoch_guards,
                self.stats.loop_epoch_slow_sites,
                self.stats.bounds_checks,
                self.stats.bounds_exception_sites,
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
                self.stats.edge_copy_sites,
                self.stats.edge_copy_moves,
                self.stats.edge_copy_cycles,
                self.stats.frame_bytes,
                self.stats.spill_loads,
                self.stats.spill_stores,
            },
        );
        try writer.print(
            " nonvolatile_frame_bytes={d} xmm_insts={d} xmm_spill_loads={d} xmm_spill_stores={d} xmm_negations={d} xmm_comparisons={d} xmm_conversions={d} xmm_saturating={d} deopt_block_entries={d} osr_entries={d} osr_frame_landings={d} osr_landing_safepoints={d} osr_remats={d} osr_restart_edges={d}\n",
            .{
                self.stats.nonvolatile_frame_bytes,
                self.stats.xmm_insts,
                self.stats.xmm_spill_loads,
                self.stats.xmm_spill_stores,
                self.stats.xmm_negations,
                self.stats.xmm_comparisons,
                self.stats.xmm_conversions,
                self.stats.xmm_saturating_conversions,
                self.stats.deopt_block_entries,
                self.stats.osr_entries,
                self.stats.osr_frame_landings,
                self.stats.osr_landing_safepoints,
                self.stats.osr_derived_rematerializations,
                self.stats.osr_remat_restart_edges,
            },
        );
        try self.allocation.print(writer);
        try self.buffer.print(writer);
    }
};

const scratch_index: u4 = 10;
const scratch_descriptor: u4 = 11;
const acknowledged_epoch: u4 = 12;
const region_table: u4 = 13;
const descriptor_table: u4 = 14;
const thread_state: u4 = 15;

const descriptor_offset_clear_shift: u6 = 28;
const descriptor_region_shift: u6 = 36;
const descriptor_generation_shift: u6 = 44;
const descriptor_state_shift: u6 = 60;
const object_alignment_shift: u6 = 3;

const WINDOWS_ALLOCATABLE_GP_REGS = [_]regalloc.PhysReg{ .rax, .rcx, .rdx, .r8, .r9 };
const SYSV_ALLOCATABLE_GP_REGS = [_]regalloc.PhysReg{ .rax, .rcx, .rdx, .rsi, .rdi, .r8, .r9 };
const WINDOWS_ALLOCATABLE_XMM_REGS = [_]regalloc.PhysReg{ .xmm0, .xmm1, .xmm2, .xmm3 };
const SYSV_ALLOCATABLE_XMM_REGS = [_]regalloc.PhysReg{ .xmm0, .xmm1, .xmm2, .xmm3, .xmm4, .xmm5 };
const xmm_scratch_primary: regalloc.PhysReg = if (builtin.os.tag == .windows) .xmm4 else .xmm6;
const xmm_scratch_secondary: regalloc.PhysReg = if (builtin.os.tag == .windows) .xmm5 else .xmm7;

fn allocatableXmmRegisters() []const regalloc.PhysReg {
    return if (builtin.os.tag == .windows) &WINDOWS_ALLOCATABLE_XMM_REGS else &SYSV_ALLOCATABLE_XMM_REGS;
}

fn allocatableGpRegisters() []const regalloc.PhysReg {
    return if (builtin.os.tag == .windows) &WINDOWS_ALLOCATABLE_GP_REGS else &SYSV_ALLOCATABLE_GP_REGS;
}

fn allGpRegisters() []const regalloc.PhysReg {
    return &SYSV_ALLOCATABLE_GP_REGS;
}

const ColdResolve = struct {
    entry: code_buffer.LabelId,
    continuation: code_buffer.LabelId,
    handle: regalloc.PhysReg,
    destination: regalloc.PhysReg,
    site_key: u64,
    epoch_restart: ?code_buffer.LabelId = null,
    epoch_slot_offset: i32 = 0,
};

const ColdBoundsException = struct {
    entry: code_buffer.LabelId,
    handle: regalloc.PhysReg,
    index: regalloc.PhysReg,
    address: regalloc.PhysReg,
    site_key: u64,
};

const ColdDeopt = struct {
    entry: code_buffer.LabelId,
    site_id: u32,
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

fn xmmReg(reg: regalloc.PhysReg) Error!u3 {
    return switch (reg) {
        .xmm0 => 0,
        .xmm1 => 1,
        .xmm2 => 2,
        .xmm3 => 3,
        .xmm4 => 4,
        .xmm5 => 5,
        .xmm6 => 6,
        .xmm7 => 7,
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
    invalid: code_buffer.LabelId,
) Error!void {
    // Unsigned comparison rejects negative indices as well as index >= length.
    try emitRex(buffer, false, index, base);
    try buffer.emitU8(0x3b);
    try emitMemoryOperand(buffer, index, base, @intCast(layout.length_offset));
    try emitJccOpcode(buffer, 0x83, invalid); // jae
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

fn spillSlot(plan: *const regalloc.SpillPlan, reg: machine.RegId) Error!regalloc.SpillSlot {
    return spillSlotFor(plan, reg) orelse error.InvalidMachine;
}

fn emitAdjustStack(buffer: *code_buffer.Buffer, subtract: bool, amount: u32) Error!void {
    if (amount == 0) return;
    if (!std.mem.isAligned(amount, 16)) return error.InvalidMachine;
    if (amount <= std.math.maxInt(i8)) {
        try buffer.emitBytes(&.{ 0x48, 0x83, if (subtract) 0xec else 0xc4, @intCast(amount) });
    } else {
        try buffer.emitBytes(&.{ 0x48, 0x81, if (subtract) 0xec else 0xc4 });
        try buffer.emitU32(amount);
    }
}

fn frameSavedBytes(preserve_nonvolatile: bool) u32 {
    return if (builtin.os.tag == .windows and preserve_nonvolatile) 2 * @sizeOf(u64) else 0;
}

fn emitFrameEnter(buffer: *code_buffer.Buffer, frame_bytes: u32, preserve_nonvolatile: bool) Error!void {
    if (frameSavedBytes(preserve_nonvolatile) != 0) {
        try emitPushRaw(buffer, 6); // rsi
        try emitPushRaw(buffer, 7); // rdi
    }
    try emitAdjustStack(buffer, true, frame_bytes);
}

fn emitFrameLeave(buffer: *code_buffer.Buffer, frame_bytes: u32, preserve_nonvolatile: bool) Error!void {
    try emitAdjustStack(buffer, false, frame_bytes);
    if (frameSavedBytes(preserve_nonvolatile) != 0) {
        try emitPopRaw(buffer, 7); // rdi
        try emitPopRaw(buffer, 6); // rsi
    }
}

fn emitSpillLoad(
    buffer: *code_buffer.Buffer,
    slot: regalloc.SpillSlot,
    dst: regalloc.PhysReg,
    stats: *Stats,
) Error!void {
    if (slot.byte_offset > std.math.maxInt(i32) or (slot.size != 4 and slot.size != 8)) return error.UnsupportedInstruction;
    switch (dst.class()) {
        .gp => try emitMovRawFromMemory(buffer, try x64Reg(dst), 4, @intCast(slot.byte_offset), slot.size == 8),
        .xmm => {
            try emitXmmMemory(buffer, dst, 4, @intCast(slot.byte_offset), slot.size == 8, true);
            stats.xmm_insts += 1;
            stats.xmm_spill_loads += 1;
        },
    }
    stats.spill_loads += 1;
    stats.native_insts += 1;
}

fn emitSpillStore(
    buffer: *code_buffer.Buffer,
    slot: regalloc.SpillSlot,
    src: regalloc.PhysReg,
    stats: *Stats,
) Error!void {
    if (slot.byte_offset > std.math.maxInt(i32) or (slot.size != 4 and slot.size != 8)) return error.UnsupportedInstruction;
    switch (src.class()) {
        .gp => try emitMovMemoryFromRaw(buffer, 4, @intCast(slot.byte_offset), try x64Reg(src), slot.size == 8),
        .xmm => {
            try emitXmmMemory(buffer, src, 4, @intCast(slot.byte_offset), slot.size == 8, false);
            stats.xmm_insts += 1;
            stats.xmm_spill_stores += 1;
        },
    }
    stats.spill_stores += 1;
    stats.native_insts += 1;
}

fn readInto(
    buffer: *code_buffer.Buffer,
    allocation: *const regalloc.Allocation,
    plan: *const regalloc.SpillPlan,
    reg: machine.RegId,
    scratch: regalloc.PhysReg,
    stats: *Stats,
) Error!regalloc.PhysReg {
    return switch (allocation.locationOf(reg) orelse return error.InvalidMachine) {
        .phys => |physical| physical,
        .spill => {
            try emitSpillLoad(buffer, try spillSlot(plan, reg), scratch, stats);
            return scratch;
        },
        .none => error.InvalidMachine,
    };
}

fn writeFrom(
    buffer: *code_buffer.Buffer,
    allocation: *const regalloc.Allocation,
    plan: *const regalloc.SpillPlan,
    reg: machine.RegId,
    src: regalloc.PhysReg,
    wide: bool,
    stats: *Stats,
) Error!void {
    switch (allocation.locationOf(reg) orelse return error.InvalidMachine) {
        .phys => |physical| {
            try emitPhysicalMove(buffer, physical, src, wide);
            if (physical != src) {
                stats.register_moves += 1;
                stats.native_insts += 1;
                stats.xmm_insts += @intFromBool(physical.class() == .xmm or src.class() == .xmm);
            }
        },
        .spill => try emitSpillStore(buffer, try spillSlot(plan, reg), src, stats),
        .none => return error.InvalidMachine,
    }
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

fn emitXmmMemory(
    buffer: *code_buffer.Buffer,
    xmm: regalloc.PhysReg,
    base: u4,
    displacement: i32,
    wide: bool,
    load: bool,
) Error!void {
    const x: u4 = @intCast(try xmmReg(xmm));
    try buffer.emitU8(if (wide) 0xf2 else 0xf3);
    try emitRex(buffer, false, x, base);
    try buffer.emitU8(0x0f);
    try buffer.emitU8(if (load) 0x10 else 0x11);
    try emitMemoryOperand(buffer, x, base, displacement);
}

fn emitXmmRegReg(buffer: *code_buffer.Buffer, dst: regalloc.PhysReg, src: regalloc.PhysReg) Error!void {
    if (dst == src) return;
    const d: u4 = @intCast(try xmmReg(dst));
    const s: u4 = @intCast(try xmmReg(src));
    try emitRex(buffer, false, d, s);
    try buffer.emitBytes(&.{ 0x0f, 0x28 }); // movaps xmm, xmm
    try emitModRm(buffer, d, s);
}

fn emitGpToXmm(buffer: *code_buffer.Buffer, dst: regalloc.PhysReg, src: regalloc.PhysReg, wide: bool) Error!void {
    const d: u4 = @intCast(try xmmReg(dst));
    const s = try x64Reg(src);
    try buffer.emitU8(0x66);
    try emitRex(buffer, wide, d, s);
    try buffer.emitBytes(&.{ 0x0f, 0x6e });
    try emitModRm(buffer, d, s);
}

fn emitXmmToGp(buffer: *code_buffer.Buffer, dst: regalloc.PhysReg, src: regalloc.PhysReg, wide: bool) Error!void {
    const d = try x64Reg(dst);
    const s: u4 = @intCast(try xmmReg(src));
    try buffer.emitU8(0x66);
    try emitRex(buffer, wide, s, d);
    try buffer.emitBytes(&.{ 0x0f, 0x7e });
    try emitModRm(buffer, s, d);
}

fn emitPhysicalMove(buffer: *code_buffer.Buffer, dst: regalloc.PhysReg, src: regalloc.PhysReg, wide: bool) Error!void {
    if (dst == src) return;
    switch (dst.class()) {
        .gp => switch (src.class()) {
            .gp => try emitMovRegReg(buffer, dst, src, wide),
            .xmm => try emitXmmToGp(buffer, dst, src, wide),
        },
        .xmm => switch (src.class()) {
            .gp => try emitGpToXmm(buffer, dst, src, wide),
            .xmm => try emitXmmRegReg(buffer, dst, src),
        },
    }
}

fn emitXmmBinary(
    buffer: *code_buffer.Buffer,
    operation: machine.FloatOperation,
    dst: regalloc.PhysReg,
    rhs: regalloc.PhysReg,
    wide: bool,
) Error!void {
    const d: u4 = @intCast(try xmmReg(dst));
    const r: u4 = @intCast(try xmmReg(rhs));
    const opcode: u8 = switch (operation) {
        .add => 0x58,
        .sub => 0x5c,
        .mul => 0x59,
        .div => 0x5e,
        else => return error.UnsupportedInstruction,
    };
    try buffer.emitU8(if (wide) 0xf2 else 0xf3);
    try emitRex(buffer, false, d, r);
    try buffer.emitBytes(&.{ 0x0f, opcode });
    try emitModRm(buffer, d, r);
}

fn emitXmmXor(buffer: *code_buffer.Buffer, dst: regalloc.PhysReg, rhs: regalloc.PhysReg, wide: bool) Error!void {
    const d: u4 = @intCast(try xmmReg(dst));
    const r: u4 = @intCast(try xmmReg(rhs));
    if (wide) try buffer.emitU8(0x66); // xorpd; xorps has no mandatory prefix
    try emitRex(buffer, false, d, r);
    try buffer.emitBytes(&.{ 0x0f, 0x57 });
    try emitModRm(buffer, d, r);
}

fn emitXmmCompare(buffer: *code_buffer.Buffer, lhs: regalloc.PhysReg, rhs: regalloc.PhysReg, wide: bool) Error!void {
    const l: u4 = @intCast(try xmmReg(lhs));
    const r: u4 = @intCast(try xmmReg(rhs));
    if (wide) try buffer.emitU8(0x66); // ucomisd; ucomiss has no mandatory prefix
    try emitRex(buffer, false, l, r);
    try buffer.emitBytes(&.{ 0x0f, 0x2e });
    try emitModRm(buffer, l, r);
}

fn emitXmmCompareResult(
    buffer: *code_buffer.Buffer,
    operation: machine.FloatOperation,
    dst: regalloc.PhysReg,
    lhs: regalloc.PhysReg,
    rhs: regalloc.PhysReg,
    wide: bool,
    stats: *Stats,
) Error!void {
    if (operation != .compare_l and operation != .compare_g) return error.InvalidMachine;
    if (dst.class() != .gp or lhs.class() != .xmm or rhs.class() != .xmm) return error.InvalidMachine;
    const unordered = try buffer.newLabel();
    const greater = try buffer.newLabel();
    const less = try buffer.newLabel();
    const done = try buffer.newLabel();

    try emitXmmCompare(buffer, lhs, rhs, wide);
    try emitJccOpcode(buffer, 0x8a, unordered); // jp: unordered/NaN
    try emitJccOpcode(buffer, 0x87, greater); // ja: lhs > rhs
    try emitJccOpcode(buffer, 0x82, less); // jb: lhs < rhs
    try emitMovRegImm32(buffer, dst, 0);
    try emitJump(buffer, done);
    try buffer.bindLabel(greater);
    try emitMovRegImm32(buffer, dst, 1);
    try emitJump(buffer, done);
    try buffer.bindLabel(less);
    try emitMovRegImm32(buffer, dst, -1);
    try emitJump(buffer, done);
    try buffer.bindLabel(unordered);
    try emitMovRegImm32(buffer, dst, if (operation == .compare_l) -1 else 1);
    try buffer.bindLabel(done);

    stats.xmm_insts += 1;
    stats.native_insts += 11;
    stats.branches += 3;
    stats.jumps += 3;
}

fn emitGpToFloatConvert(
    buffer: *code_buffer.Buffer,
    dst: regalloc.PhysReg,
    src: regalloc.PhysReg,
    input_wide: bool,
    output_wide: bool,
) Error!void {
    const d: u4 = @intCast(try xmmReg(dst));
    const s = try x64Reg(src);
    try buffer.emitU8(if (output_wide) 0xf2 else 0xf3);
    try emitRex(buffer, input_wide, d, s);
    try buffer.emitBytes(&.{ 0x0f, 0x2a });
    try emitModRm(buffer, d, s);
}

fn emitFloatToGpTruncate(
    buffer: *code_buffer.Buffer,
    dst: regalloc.PhysReg,
    src: regalloc.PhysReg,
    input_wide: bool,
    output_wide: bool,
) Error!void {
    const d = try x64Reg(dst);
    const s: u4 = @intCast(try xmmReg(src));
    try buffer.emitU8(if (input_wide) 0xf2 else 0xf3);
    try emitRex(buffer, output_wide, d, s);
    try buffer.emitBytes(&.{ 0x0f, 0x2c });
    try emitModRm(buffer, d, s);
}

fn emitFloatWidthConvert(
    buffer: *code_buffer.Buffer,
    dst: regalloc.PhysReg,
    src: regalloc.PhysReg,
    input_wide: bool,
) Error!void {
    const d: u4 = @intCast(try xmmReg(dst));
    const s: u4 = @intCast(try xmmReg(src));
    try buffer.emitU8(if (input_wide) 0xf2 else 0xf3);
    try emitRex(buffer, false, d, s);
    try buffer.emitBytes(&.{ 0x0f, 0x5a });
    try emitModRm(buffer, d, s);
}

fn emitIntegerLimitAsFloat(
    buffer: *code_buffer.Buffer,
    dst: regalloc.PhysReg,
    input_wide: bool,
    output_wide: bool,
    upper: bool,
) Error!void {
    const integer_limit: i64 = if (output_wide)
        (if (upper) std.math.maxInt(i64) else std.math.minInt(i64))
    else if (upper)
        std.math.maxInt(i32)
    else
        std.math.minInt(i32);
    if (input_wide) {
        const value: f64 = @floatFromInt(integer_limit);
        try emitMovRegImm64(buffer, .r11, @bitCast(value));
    } else {
        const value: f32 = @floatFromInt(integer_limit);
        try emitMovRegImm32(buffer, .r11, @bitCast(value));
    }
    try emitGpToXmm(buffer, dst, .r11, input_wide);
}

fn emitSaturatingFloatToInteger(
    buffer: *code_buffer.Buffer,
    dst: regalloc.PhysReg,
    src: regalloc.PhysReg,
    input_wide: bool,
    output_wide: bool,
    stats: *Stats,
) Error!void {
    if (dst.class() != .gp or src.class() != .xmm) return error.InvalidMachine;
    const nan = try buffer.newLabel();
    const upper = try buffer.newLabel();
    const done = try buffer.newLabel();

    // CVTT's overwhelmingly common in-range result is accepted immediately.
    // x86 reports NaN and every overflow as the signed minimum sentinel; only
    // that uncommon result needs semantic disambiguation.
    try emitFloatToGpTruncate(buffer, dst, src, input_wide, output_wide);
    if (output_wide) {
        try emitMovRegImm64(buffer, .r11, std.math.minInt(i64));
        try emitCmpRawRaw(buffer, try x64Reg(dst), try x64Reg(.r11), true);
    } else {
        try emitCmpRawImm32(buffer, try x64Reg(dst), @bitCast(@as(i32, std.math.minInt(i32))));
    }
    try emitJccOpcode(buffer, 0x85, done); // jne: ordinary in-range result

    try emitXmmCompare(buffer, src, src, input_wide);
    try emitJccOpcode(buffer, 0x8a, nan); // jp: NaN -> zero
    try emitIntegerLimitAsFloat(buffer, xmm_scratch_secondary, input_wide, output_wide, true);
    try emitXmmCompare(buffer, src, xmm_scratch_secondary, input_wide);
    try emitJccOpcode(buffer, 0x83, upper); // jae: saturate to maximum
    // Ordered values that still produced the minimum sentinel are either the
    // exact/truncated minimum or lower overflow; Dalvik requires minimum for
    // both, so the CVTT result is already final.
    try emitJump(buffer, done);

    try buffer.bindLabel(nan);
    if (output_wide) try emitMovRegImm64(buffer, dst, 0) else try emitMovRegImm32(buffer, dst, 0);
    try emitJump(buffer, done);
    try buffer.bindLabel(upper);
    if (output_wide) try emitMovRegImm64(buffer, dst, std.math.maxInt(i64)) else try emitMovRegImm32(buffer, dst, std.math.maxInt(i32));
    try buffer.bindLabel(done);

    stats.xmm_insts += 4;
    stats.native_insts += if (output_wide) 14 else 13;
    stats.branches += 3;
    stats.jumps += 2;
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

fn emitMovRegImm64(buffer: *code_buffer.Buffer, dst: regalloc.PhysReg, value: i64) Error!void {
    try emitMovRawImm64(buffer, try x64Reg(dst), @bitCast(value));
}

fn valueClass(ty: anytype) regalloc.RegClass {
    return if (ty == .float or ty == .double) .xmm else .gp;
}

fn recordPhysicalMove(
    buffer: *code_buffer.Buffer,
    dst: regalloc.PhysReg,
    src: regalloc.PhysReg,
    wide: bool,
    stats: *Stats,
) Error!void {
    if (dst == src) return;
    try emitPhysicalMove(buffer, dst, src, wide);
    stats.register_moves += 1;
    stats.native_insts += 1;
    stats.xmm_insts += @intFromBool(dst.class() == .xmm or src.class() == .xmm);
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

fn emitBinaryRegImm32(buffer: *code_buffer.Buffer, opcode: machine.Opcode, dst: regalloc.PhysReg, value: i32) Error!void {
    const d = try x64Reg(dst);
    const extension: u4 = switch (opcode) {
        .add_i32 => 0,
        .or_i32 => 1,
        .and_i32 => 4,
        .sub_i32 => 5,
        .xor_i32 => 6,
        else => return error.UnsupportedInstruction,
    };
    try emitRex(buffer, false, extension, d);
    try buffer.emitU8(0x81);
    try emitModRm(buffer, extension, d);
    try buffer.emitU32(@bitCast(value));
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

fn emitLoopEpochGuard(
    allocator: std.mem.Allocator,
    buffer: *code_buffer.Buffer,
    allocation: *const regalloc.Allocation,
    inst: machine.Inst,
    cold: *std.ArrayList(ColdResolve),
    stats: *Stats,
) Error!void {
    if (inst.defs.len != 0 or inst.uses.len != 0) return error.InvalidMachine;
    const address_reg = inst.address orelse return error.InvalidMachine;
    const handle_reg = inst.state_handle orelse return error.InvalidMachine;
    const site_id = inst.guard_site_id orelse return error.InvalidMachine;
    const address = try physOf(allocation, address_reg);
    const handle = try physOf(allocation, handle_reg);
    if (address == handle or address == .r10 or address == .r11 or handle == .r10 or handle == .r11) return error.InvalidMachine;

    const cold_label = try buffer.newLabel();
    const resume_label = try buffer.newLabel();
    try cold.append(allocator, .{
        .entry = cold_label,
        .continuation = resume_label,
        .handle = handle,
        .destination = address,
        .site_key = (@as(u64, inst.pc orelse std.math.maxInt(u32)) << 32) | site_id,
    });

    const request_pointer_offset = @offsetOf(runtime_jit.NativeThreadState, "request_epoch_address");
    if (request_pointer_offset > std.math.maxInt(i32) or !std.mem.isAligned(request_pointer_offset, @alignOf(usize))) {
        return error.InvalidRuntimeAbi;
    }
    // r15 owns a stable pointer to the registry's aligned atomic epoch. On
    // x86-64 these ordinary aligned loads are acquire observations. A request
    // racing after the comparison cannot reclaim from-space because r12 has
    // not acknowledged it; the next backedge takes the slow edge.
    try emitMovRawFromMemory(buffer, scratch_index, thread_state, @intCast(request_pointer_offset), true);
    try emitMovRawFromMemory(buffer, scratch_index, scratch_index, 0, true);
    try emitCmpRawRaw(buffer, acknowledged_epoch, scratch_index, true);
    try emitJccOpcode(buffer, 0x85, cold_label); // jne
    try buffer.bindLabel(resume_label);

    stats.loop_epoch_guards += 1;
    stats.loop_epoch_slow_sites += 1;
    stats.cold_resolve_sites += 1;
    stats.native_insts += 4;
}

fn emitBoundsExceptionCheck(
    allocator: std.mem.Allocator,
    buffer: *code_buffer.Buffer,
    allocation: *const regalloc.Allocation,
    spill_plan: *const regalloc.SpillPlan,
    inst: machine.Inst,
    runtime: RuntimeAbi,
    cold: *std.ArrayList(ColdBoundsException),
    stats: *Stats,
) Error!void {
    if (runtime.bounds_exception_helper == 0) return error.MissingExceptionHelper;
    if (inst.defs.len != 0 or inst.uses.len != 1) return error.InvalidMachine;
    const address = try physOf(allocation, inst.address orelse return error.InvalidMachine);
    const handle = try physOf(allocation, inst.state_handle orelse return error.InvalidMachine);
    const index = try readInto(buffer, allocation, spill_plan, inst.uses[0], .r10, stats);
    if (address == handle or address == index or handle == index) return error.InvalidMachine;
    const site_id = inst.exception_site_id orelse return error.InvalidMachine;
    const layout = try referenceArrayLayout(runtime);
    const cold_label = try buffer.newLabel();
    try cold.append(allocator, .{
        .entry = cold_label,
        .handle = handle,
        .index = index,
        .address = address,
        .site_key = (@as(u64, inst.pc orelse std.math.maxInt(u32)) << 32) | site_id,
    });
    try emitArrayBoundsCheck(buffer, try x64Reg(address), try x64Reg(index), layout, cold_label);
    stats.bounds_checks += 1;
    stats.bounds_exception_sites += 1;
    stats.native_insts += 2;
}

fn emitColdBoundsExceptions(
    buffer: *code_buffer.Buffer,
    cold: []const ColdBoundsException,
    runtime: RuntimeAbi,
    frame_bytes: u32,
    preserve_nonvolatile: bool,
    stats: *Stats,
) Error!void {
    if (cold.len == 0) return;
    const pending_offset = @offsetOf(runtime_jit.NativeThreadState, "pending_exception");
    const index_offset = pending_offset + @offsetOf(runtime_jit.ManagedException, "index");
    const length_offset = pending_offset + @offsetOf(runtime_jit.ManagedException, "length");
    if (index_offset > std.math.maxInt(i32) or length_offset > std.math.maxInt(i32) or
        !std.mem.isAligned(index_offset, @alignOf(i32)) or !std.mem.isAligned(length_offset, @alignOf(u32)))
        return error.InvalidRuntimeAbi;
    const layout = try referenceArrayLayout(runtime);
    for (cold) |site| {
        try buffer.bindLabel(site.entry);
        const address = try x64Reg(site.address);
        try emitMovMemoryFromRaw(buffer, thread_state, @intCast(index_offset), try x64Reg(site.index), false);
        try emitMovRawFromMemory(buffer, scratch_descriptor, address, @intCast(layout.length_offset), false);
        try emitMovMemoryFromRaw(buffer, thread_state, @intCast(length_offset), scratch_descriptor, false);
        try emitMovRawRaw(buffer, scratch_index, try x64Reg(site.handle), true);
        try emitMovRawImm64(buffer, scratch_descriptor, site.site_key);
        try emitMovRawImm64(buffer, address, @intCast(runtime.bounds_exception_helper));
        try emitCallRaw(buffer, address);
        try emitMovRegImm32(buffer, .rax, 0);
        try emitFrameLeave(buffer, frame_bytes, preserve_nonvolatile);
        try emitRet(buffer);
        stats.native_insts += 9;
        stats.returns += 1;
    }
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
        if (site.epoch_restart) |restart| {
            try emitMovRawFromMemory(buffer, scratch_descriptor, 4, site.epoch_slot_offset, true);
            try emitCmpRawRaw(buffer, acknowledged_epoch, scratch_descriptor, true);
            try emitJccOpcode(buffer, 0x85, restart); // jne: helper acknowledged relocation
            stats.native_insts += 3;
            stats.branches += 1;
        }
        try emitJump(buffer, site.continuation);
        stats.native_insts += 6;
        stats.jumps += 1;
    }
}

fn emitDeoptGuard(
    allocator: std.mem.Allocator,
    buffer: *code_buffer.Buffer,
    runtime: RuntimeAbi,
    site_id: u32,
    cold: *std.ArrayList(ColdDeopt),
    stats: *Stats,
) Error!void {
    if (runtime.deopt_epoch_address == 0 or runtime.deopt_helper == 0) return error.InvalidRuntimeAbi;
    const entry = try buffer.newLabel();
    try cold.append(allocator, .{ .entry = entry, .site_id = site_id });
    try emitMovRawImm64(buffer, scratch_index, runtime.deopt_epoch_address);
    try emitMovRawFromMemory(buffer, scratch_descriptor, scratch_index, 0, true);
    try emitMovRawImm64(buffer, scratch_index, runtime.compiled_deopt_epoch);
    try emitCmpRawRaw(buffer, scratch_descriptor, scratch_index, true);
    try emitJccOpcode(buffer, 0x85, entry); // jne
    stats.deopt_guards += 1;
    stats.native_insts += 5;
}

fn emitColdDeopts(
    buffer: *code_buffer.Buffer,
    cold: []const ColdDeopt,
    runtime: RuntimeAbi,
    frame_bytes: u32,
    preserve_nonvolatile: bool,
    stats: *Stats,
) Error!void {
    if (cold.len == 0) return;
    try buffer.alignTo(16, 0x90);
    for (cold) |site| {
        try buffer.bindLabel(site.entry);
        try emitMovRawImm64(buffer, scratch_index, site.site_id);
        try emitMovRawImm64(buffer, scratch_descriptor, runtime.deopt_helper);
        try emitCallRaw(buffer, scratch_descriptor);
        try emitMovRawRaw(buffer, 0, scratch_index, true);
        try emitFrameLeave(buffer, frame_bytes, preserve_nonvolatile);
        try emitRet(buffer);
        stats.deopt_traps += 1;
        stats.native_insts += 5;
        stats.returns += 1;
    }
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

fn abiRegisterParamCount() u32 {
    return if (builtin.os.tag == .windows) 4 else 6;
}

fn abiStackParamOffset(index: u32, frame_bytes: u32, preserve_nonvolatile: bool) Error!i32 {
    const register_count = abiRegisterParamCount();
    if (index < register_count) return error.InvalidMachine;
    // Windows reserves four eight-byte home slots between the return address
    // and its first stack argument. SysV places its first stack argument
    // immediately above the return address.
    const first: u32 = if (builtin.os.tag == .windows) 40 else 8;
    const argument_delta = std.math.mul(u32, index - register_count, 8) catch return error.UnsupportedInstruction;
    const entry_offset = std.math.add(u32, first, argument_delta) catch return error.UnsupportedInstruction;
    const frame_and_saves = std.math.add(u32, frame_bytes, frameSavedBytes(preserve_nonvolatile)) catch return error.UnsupportedInstruction;
    const framed_offset = std.math.add(u32, frame_and_saves, entry_offset) catch return error.UnsupportedInstruction;
    if (framed_offset > std.math.maxInt(i32)) return error.UnsupportedInstruction;
    return @intCast(framed_offset);
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
    try emitPhysicalMove(buffer, move.dst, move.src, move.wide);
    stats.register_moves += 1;
    stats.native_insts += 1;
    stats.xmm_insts += @intFromBool(move.dst.class() == .xmm or move.src.class() == .xmm);
}

fn parameterIsUsed(source: *const machine.Function, reg: machine.RegId) bool {
    for (source.blocks) |block| {
        for (block.insts) |inst| {
            for (inst.uses) |used| if (used == reg) return true;
            if (inst.address == reg or inst.state_handle == reg) return true;
        }
    }
    for (source.edges) |edge| {
        for (edge.moves) |move| if (move.src == reg) return true;
    }
    return false;
}

/// Materialize ABI parameters as a true parallel copy. A linear sequence is
/// incorrect when an allocated destination is also a later ABI source. Cycles
/// are broken with a register that is absent from the complete move graph.
fn emitParamMoves(
    buffer: *code_buffer.Buffer,
    allocation: *const regalloc.Allocation,
    spill_plan: *const regalloc.SpillPlan,
    source: *const machine.Function,
    frame_bytes: u32,
    preserve_nonvolatile: bool,
    stats: *Stats,
) Error!void {
    var moves: [6]ParamMove = undefined;
    var move_count: usize = 0;
    var param_index: u32 = 0;
    // Save stack-assigned parameters before any register parallel copy can
    // overwrite their incoming ABI register.
    for (source.value_kinds, 0..) |kind, value_id| {
        if (kind != .parameter) continue;
        const parameter_reg: machine.RegId = @intCast(value_id);
        const used = parameterIsUsed(source, parameter_reg);
        if (param_index >= abiRegisterParamCount()) {
            param_index += 1;
            continue;
        }
        if (!used) {
            param_index += 1;
            continue;
        }
        const src = try abiParamReg(param_index);
        // Runtime root metadata is authoritative even while bytecode type
        // inference remains directional/unknown. Handles always carry a
        // 32-bit generation above their 32-bit table index.
        const wide = isWideType(source.reg_types[value_id]) or source.isGcRoot(@intCast(value_id));
        switch (allocation.locationOf(@intCast(value_id)) orelse return error.InvalidMachine) {
            .phys => |dst| if (dst != src) {
                if (move_count == moves.len) return error.UnsupportedInstruction;
                moves[move_count] = .{ .dst = dst, .src = src, .wide = wide };
                move_count += 1;
            },
            .spill => try emitSpillStore(buffer, try spillSlot(spill_plan, @intCast(value_id)), src, stats),
            .none => return error.InvalidMachine,
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

    // Register sources are now dead. Stack arguments can be loaded directly
    // into their final physical destination, or through r10 for stack slots.
    param_index = 0;
    for (source.value_kinds, 0..) |kind, value_id| {
        if (kind != .parameter) continue;
        const reg: machine.RegId = @intCast(value_id);
        const used = parameterIsUsed(source, reg);
        if (param_index < abiRegisterParamCount()) {
            param_index += 1;
            continue;
        }
        if (!used) {
            param_index += 1;
            continue;
        }
        const wide = isWideType(source.reg_types[value_id]) or source.isGcRoot(reg);
        const offset = try abiStackParamOffset(param_index, frame_bytes, preserve_nonvolatile);
        switch (allocation.locationOf(reg) orelse return error.InvalidMachine) {
            .phys => |dst| {
                switch (dst.class()) {
                    .gp => try emitMovRawFromMemory(buffer, try x64Reg(dst), 4, offset, wide),
                    .xmm => {
                        try emitXmmMemory(buffer, dst, 4, offset, wide, true);
                        stats.xmm_insts += 1;
                    },
                }
                stats.native_insts += 1;
            },
            .spill => {
                try emitMovRawFromMemory(buffer, scratch_index, 4, offset, wide);
                stats.native_insts += 1;
                try emitSpillStore(buffer, try spillSlot(spill_plan, reg), .r10, stats);
            },
            .none => return error.InvalidMachine,
        }
        param_index += 1;
    }
}

const EdgeCopy = struct {
    dst: regalloc.Location,
    src: regalloc.Location,
    dst_reg: machine.RegId,
    src_reg: machine.RegId,
    wide: bool,
    class: regalloc.RegClass,
};

fn sameLocation(a: regalloc.Location, b: regalloc.Location) bool {
    return switch (a) {
        .none => b == .none,
        .phys => |reg| switch (b) {
            .phys => |other| reg == other,
            else => false,
        },
        .spill => |slot| switch (b) {
            .spill => |other| slot == other,
            else => false,
        },
    };
}

fn removeEdgeCopy(copies: []EdgeCopy, count: *usize, index: usize) void {
    var cursor = index;
    while (cursor + 1 < count.*) : (cursor += 1) copies[cursor] = copies[cursor + 1];
    count.* -= 1;
}

fn emitRecordedEdgeCopy(
    buffer: *code_buffer.Buffer,
    spill_plan: ?*const regalloc.SpillPlan,
    copy: EdgeCopy,
    stats: *Stats,
) Error!void {
    switch (copy.src) {
        .none => return error.InvalidMachine,
        .phys => |src| switch (copy.dst) {
            .none => return error.InvalidMachine,
            .phys => |dst| {
                if (dst.class() != copy.class or src.class() != copy.class) return error.InvalidMachine;
                try emitPhysicalMove(buffer, dst, src, copy.wide);
                stats.register_moves += 1;
                stats.native_insts += 1;
                stats.xmm_insts += @intFromBool(copy.class == .xmm);
            },
            .spill => {
                if (src.class() != copy.class) return error.InvalidMachine;
                const plan = spill_plan orelse return error.InvalidMachine;
                try emitSpillStore(buffer, try spillSlot(plan, copy.dst_reg), src, stats);
            },
        },
        .spill => switch (copy.dst) {
            .none => return error.InvalidMachine,
            .phys => |dst| {
                if (dst.class() != copy.class) return error.InvalidMachine;
                const plan = spill_plan orelse return error.InvalidMachine;
                try emitSpillLoad(buffer, try spillSlot(plan, copy.src_reg), dst, stats);
            },
            .spill => {
                const plan = spill_plan orelse return error.InvalidMachine;
                const scratch: regalloc.PhysReg = if (copy.class == .xmm) xmm_scratch_secondary else .r11;
                try emitSpillLoad(buffer, try spillSlot(plan, copy.src_reg), scratch, stats);
                try emitSpillStore(buffer, try spillSlot(plan, copy.dst_reg), scratch, stats);
            },
        },
    }
}

fn emitParallelEdgeCopies(buffer: *code_buffer.Buffer, spill_plan: ?*const regalloc.SpillPlan, copies: []EdgeCopy, stats: *Stats) Error!void {
    var count = copies.len;
    while (count != 0) {
        var ready: ?usize = null;
        for (copies[0..count], 0..) |candidate, index| {
            var destination_is_source = false;
            for (copies[0..count], 0..) |other, other_index| {
                if (other_index != index and sameLocation(other.src, candidate.dst)) {
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
            try emitRecordedEdgeCopy(buffer, spill_plan, copies[index], stats);
            removeEdgeCopy(copies, &count, index);
            continue;
        }

        const saved_source = copies[0].src;
        const scratch: regalloc.PhysReg = if (copies[0].class == .xmm) xmm_scratch_secondary else .r10;
        if (sameLocation(saved_source, .{ .phys = scratch })) return error.InvalidMachine;
        try emitRecordedEdgeCopy(buffer, spill_plan, .{
            .dst = .{ .phys = scratch },
            .src = saved_source,
            .dst_reg = std.math.maxInt(machine.RegId),
            .src_reg = copies[0].src_reg,
            .wide = true,
            .class = copies[0].class,
        }, stats);
        for (copies[0..count]) |*copy| {
            if (sameLocation(copy.src, saved_source)) {
                if (copy.class != copies[0].class) return error.InvalidMachine;
                copy.src = .{ .phys = scratch };
                copy.src_reg = std.math.maxInt(machine.RegId);
            }
        }
        stats.edge_copy_cycles += 1;
    }
}

fn emitEdgeCopies(
    allocator: std.mem.Allocator,
    buffer: *code_buffer.Buffer,
    allocation: *const regalloc.Allocation,
    spill_plan: *const regalloc.SpillPlan,
    source: *const machine.Function,
    edge: machine.EdgeMoves,
    stats: *Stats,
) Error!void {
    const copies = try allocator.alloc(EdgeCopy, edge.moves.len);
    defer allocator.free(copies);
    var count: usize = 0;
    var physical_destinations = [_]bool{false} ** 16;
    var xmm_destinations = [_]bool{false} ** 8;
    var spill_destinations = try allocator.alloc(bool, spill_plan.slots.len);
    defer allocator.free(spill_destinations);
    @memset(spill_destinations, false);
    for (edge.moves) |move| {
        if (move.dst >= source.reg_types.len or move.src >= source.reg_types.len) return error.InvalidMachine;
        const class = valueClass(move.ty);
        const dst = allocation.locationOf(move.dst) orelse return error.InvalidMachine;
        const src = allocation.locationOf(move.src) orelse return error.InvalidMachine;
        switch (dst) {
            .phys => |physical_reg| {
                if (physical_reg.class() != class) return error.InvalidMachine;
                switch (physical_reg.class()) {
                    .gp => {
                        if (physical_reg == .r10 or physical_reg == .r11) return error.UnsupportedInstruction;
                        const physical = try x64Reg(physical_reg);
                        if (physical_destinations[physical]) return error.InvalidMachine;
                        physical_destinations[physical] = true;
                    },
                    .xmm => {
                        if (physical_reg == xmm_scratch_primary or physical_reg == xmm_scratch_secondary) return error.UnsupportedInstruction;
                        const physical = try xmmReg(physical_reg);
                        if (xmm_destinations[physical]) return error.InvalidMachine;
                        xmm_destinations[physical] = true;
                    },
                }
            },
            .spill => |slot| {
                if (slot >= spill_destinations.len or spill_destinations[slot]) return error.InvalidMachine;
                spill_destinations[slot] = true;
            },
            .none => return error.InvalidMachine,
        }
        switch (src) {
            .phys => |physical_reg| {
                if (physical_reg.class() != class) return error.InvalidMachine;
                switch (physical_reg.class()) {
                    .gp => if (physical_reg == .r10 or physical_reg == .r11) return error.UnsupportedInstruction,
                    .xmm => if (physical_reg == xmm_scratch_primary or physical_reg == xmm_scratch_secondary) return error.UnsupportedInstruction,
                }
            },
            .spill => {},
            .none => return error.InvalidMachine,
        }
        if (sameLocation(dst, src)) continue;
        copies[count] = .{
            .dst = dst,
            .src = src,
            .dst_reg = move.dst,
            .src_reg = move.src,
            .wide = isWideType(move.ty) or source.isGcRoot(move.dst) or source.isGcRoot(move.src),
            .class = class,
        };
        count += 1;
    }
    try emitParallelEdgeCopies(buffer, spill_plan, copies[0..count], stats);
}

fn edgeTargetLabel(
    source: *const machine.Function,
    block_labels: []const code_buffer.LabelId,
    edge_labels: []const code_buffer.LabelId,
    from: cfg.BlockId,
    target: ?cfg.BlockId,
) Error!code_buffer.LabelId {
    const to = target orelse return error.InvalidMachine;
    if (from >= source.successors.len or to >= block_labels.len or edge_labels.len != source.edges.len) return error.InvalidMachine;
    var is_successor = false;
    for (source.successors[from]) |successor| {
        if (successor == to) {
            is_successor = true;
            break;
        }
    }
    if (!is_successor) return error.InvalidMachine;

    var selected: ?code_buffer.LabelId = null;
    for (source.edges, 0..) |edge, index| {
        if (edge.from != from or edge.to != to) continue;
        if (selected != null) return error.InvalidMachine;
        selected = edge_labels[index];
    }
    return selected orelse block_labels[to];
}

fn hasExplicitTransfer(block: machine.Block) bool {
    if (block.insts.len == 0) return false;
    return switch (block.insts[block.insts.len - 1].opcode) {
        .jump, .branch, .switch_, .ret, .throw_ => true,
        else => false,
    };
}

fn encodeInst(
    allocator: std.mem.Allocator,
    buffer: *code_buffer.Buffer,
    labels: []const code_buffer.LabelId,
    edge_labels: []const code_buffer.LabelId,
    block_id: cfg.BlockId,
    allocation: *const regalloc.Allocation,
    spill_plan: *const regalloc.SpillPlan,
    frame_bytes: u32,
    preserve_nonvolatile: bool,
    source: *const machine.Function,
    inst: machine.Inst,
    runtime: ?RuntimeAbi,
    cold: *std.ArrayList(ColdResolve),
    cold_bounds: *std.ArrayList(ColdBoundsException),
    stats: *Stats,
) Error!void {
    switch (inst.opcode) {
        .const_i32 => {
            if (inst.defs.len != 1) return error.InvalidMachine;
            const ty = source.reg_types[inst.defs[0]];
            const class = valueClass(ty);
            switch (allocation.locationOf(inst.defs[0]) orelse return error.InvalidMachine) {
                .phys => |dst| {
                    if (dst.class() != class) return error.InvalidMachine;
                    if (class == .xmm) {
                        try emitMovRegImm32(buffer, .r10, @intCast(inst.imm));
                        try emitGpToXmm(buffer, dst, .r10, false);
                        stats.native_insts += 2;
                        stats.xmm_insts += 1;
                    } else {
                        try emitMovRegImm32(buffer, dst, @intCast(inst.imm));
                        stats.native_insts += 1;
                    }
                },
                .spill => {
                    try emitMovRegImm32(buffer, .r10, @intCast(inst.imm));
                    stats.native_insts += 1;
                    try emitSpillStore(buffer, try spillSlot(spill_plan, inst.defs[0]), .r10, stats);
                },
                .none => return error.InvalidMachine,
            }
            stats.constants += 1;
        },
        .const_i64 => {
            if (inst.defs.len == 0 or inst.defs.len > 2) return error.InvalidMachine;
            const ty = source.reg_types[inst.defs[0]];
            const class = valueClass(ty);
            switch (allocation.locationOf(inst.defs[0]) orelse return error.InvalidMachine) {
                .phys => |dst| {
                    if (dst.class() != class) return error.InvalidMachine;
                    if (class == .xmm) {
                        try emitMovRegImm64(buffer, .r10, inst.imm);
                        try emitGpToXmm(buffer, dst, .r10, true);
                        stats.native_insts += 2;
                        stats.xmm_insts += 1;
                    } else {
                        try emitMovRegImm64(buffer, dst, inst.imm);
                        stats.native_insts += 1;
                    }
                },
                .spill => {
                    try emitMovRegImm64(buffer, .r10, inst.imm);
                    stats.native_insts += 1;
                    try emitSpillStore(buffer, try spillSlot(spill_plan, inst.defs[0]), .r10, stats);
                },
                .none => return error.InvalidMachine,
            }
            stats.constants += 1;
        },
        .mov => {
            if (inst.defs.len == 0 or inst.defs.len > 2 or inst.uses.len != inst.defs.len) return error.UnsupportedInstruction;
            const ty = source.reg_types[inst.defs[0]];
            const class = valueClass(ty);
            if (valueClass(source.reg_types[inst.uses[0]]) != class) return error.InvalidMachine;
            const wide = isWideType(ty) or source.isGcRoot(inst.defs[0]) or source.isGcRoot(inst.uses[0]);
            const scratch: regalloc.PhysReg = if (class == .xmm) xmm_scratch_primary else .r10;
            const src = try readInto(buffer, allocation, spill_plan, inst.uses[0], scratch, stats);
            if (src.class() != class) return error.InvalidMachine;
            try writeFrom(buffer, allocation, spill_plan, inst.defs[0], src, wide, stats);
        },
        .add_i32, .sub_i32, .mul_i32, .and_i32, .or_i32, .xor_i32 => {
            if (inst.defs.len != 1 or (inst.uses.len != 1 and inst.uses.len != 2)) return error.InvalidMachine;
            const dst: regalloc.PhysReg = switch (allocation.locationOf(inst.defs[0]) orelse return error.InvalidMachine) {
                .phys => |physical| physical,
                .spill => .r10,
                .none => return error.InvalidMachine,
            };
            const lhs = try readInto(buffer, allocation, spill_plan, inst.uses[0], dst, stats);
            try emitMovRegReg(buffer, dst, lhs, false);
            if (inst.uses.len == 2) {
                const rhs = try readInto(buffer, allocation, spill_plan, inst.uses[1], if (dst == .r10) .r11 else .r10, stats);
                try emitBinaryRegReg(buffer, inst.opcode, dst, rhs);
            } else {
                if (inst.imm < std.math.minInt(i32) or inst.imm > std.math.maxInt(i32)) return error.UnsupportedInstruction;
                try emitBinaryRegImm32(buffer, inst.opcode, dst, @intCast(inst.imm));
            }
            stats.native_insts += 2;
            switch (allocation.locationOf(inst.defs[0]).?) {
                .spill => try emitSpillStore(buffer, try spillSlot(spill_plan, inst.defs[0]), dst, stats),
                else => {},
            }
        },
        .f32_op, .f64_op => {
            const operation = inst.float_op orelse return error.InvalidMachine;
            const wide = inst.opcode == .f64_op;
            switch (operation) {
                .add, .sub, .mul, .div => {
                    if (wide) {
                        if (inst.defs.len != 2 or inst.uses.len != 4) return error.InvalidMachine;
                        if (source.reg_types[inst.defs[0]] != .double or
                            source.reg_types[inst.uses[0]] != .double or
                            source.reg_types[inst.uses[2]] != .double) return error.InvalidMachine;
                    } else {
                        if (inst.defs.len != 1 or inst.uses.len != 2) return error.InvalidMachine;
                        if (source.reg_types[inst.defs[0]] != .float or
                            source.reg_types[inst.uses[0]] != .float or
                            source.reg_types[inst.uses[1]] != .float) return error.InvalidMachine;
                    }
                    const rhs_use: machine.RegId = inst.uses[if (wide) 2 else 1];
                    const dst_location = allocation.locationOf(inst.defs[0]) orelse return error.InvalidMachine;
                    const lhs_location = allocation.locationOf(inst.uses[0]) orelse return error.InvalidMachine;
                    const rhs_location = allocation.locationOf(rhs_use) orelse return error.InvalidMachine;
                    const dst: regalloc.PhysReg = switch (dst_location) {
                        .phys => |physical| physical,
                        .spill => xmm_scratch_primary,
                        .none => return error.InvalidMachine,
                    };
                    if (dst.class() != .xmm) return error.InvalidMachine;

                    var saved_rhs: ?regalloc.PhysReg = null;
                    switch (rhs_location) {
                        .phys => |rhs| {
                            if (rhs.class() != .xmm) return error.InvalidMachine;
                            if (rhs == dst and !sameLocation(lhs_location, rhs_location)) {
                                try recordPhysicalMove(buffer, xmm_scratch_secondary, rhs, wide, stats);
                                saved_rhs = xmm_scratch_secondary;
                            }
                        },
                        .spill => {},
                        .none => return error.InvalidMachine,
                    }

                    const lhs = try readInto(buffer, allocation, spill_plan, inst.uses[0], dst, stats);
                    if (lhs.class() != .xmm) return error.InvalidMachine;
                    try recordPhysicalMove(buffer, dst, lhs, wide, stats);
                    const rhs = saved_rhs orelse try readInto(
                        buffer,
                        allocation,
                        spill_plan,
                        rhs_use,
                        if (dst == xmm_scratch_primary) xmm_scratch_secondary else xmm_scratch_primary,
                        stats,
                    );
                    if (rhs.class() != .xmm) return error.InvalidMachine;
                    try emitXmmBinary(buffer, operation, dst, rhs, wide);
                    stats.native_insts += 1;
                    stats.xmm_insts += 1;
                    switch (dst_location) {
                        .spill => try emitSpillStore(buffer, try spillSlot(spill_plan, inst.defs[0]), dst, stats),
                        else => {},
                    }
                },
                .neg => {
                    if (wide) {
                        if (inst.defs.len != 2 or inst.uses.len != 2 or
                            source.reg_types[inst.defs[0]] != .double or source.reg_types[inst.uses[0]] != .double) return error.InvalidMachine;
                    } else {
                        if (inst.defs.len != 1 or inst.uses.len != 1 or
                            source.reg_types[inst.defs[0]] != .float or source.reg_types[inst.uses[0]] != .float) return error.InvalidMachine;
                    }
                    const dst_location = allocation.locationOf(inst.defs[0]) orelse return error.InvalidMachine;
                    const dst: regalloc.PhysReg = switch (dst_location) {
                        .phys => |physical| physical,
                        .spill => xmm_scratch_primary,
                        .none => return error.InvalidMachine,
                    };
                    if (dst.class() != .xmm) return error.InvalidMachine;
                    const src = try readInto(buffer, allocation, spill_plan, inst.uses[0], dst, stats);
                    if (src.class() != .xmm) return error.InvalidMachine;
                    try recordPhysicalMove(buffer, dst, src, wide, stats);
                    if (wide) {
                        try emitMovRegImm64(buffer, .r10, std.math.minInt(i64));
                    } else {
                        try emitMovRegImm32(buffer, .r10, std.math.minInt(i32));
                    }
                    try emitGpToXmm(buffer, xmm_scratch_secondary, .r10, wide);
                    try emitXmmXor(buffer, dst, xmm_scratch_secondary, wide);
                    stats.native_insts += 3;
                    stats.xmm_insts += 2;
                    stats.xmm_negations += 1;
                    switch (dst_location) {
                        .spill => try emitSpillStore(buffer, try spillSlot(spill_plan, inst.defs[0]), dst, stats),
                        else => {},
                    }
                },
                .compare_l, .compare_g => {
                    if (inst.defs.len != 1 or source.reg_types[inst.defs[0]] != .int) return error.InvalidMachine;
                    if (wide) {
                        if (inst.uses.len != 4 or source.reg_types[inst.uses[0]] != .double or
                            source.reg_types[inst.uses[2]] != .double) return error.InvalidMachine;
                    } else {
                        if (inst.uses.len != 2 or source.reg_types[inst.uses[0]] != .float or
                            source.reg_types[inst.uses[1]] != .float) return error.InvalidMachine;
                    }
                    const rhs_use: machine.RegId = inst.uses[if (wide) 2 else 1];
                    const dst_location = allocation.locationOf(inst.defs[0]) orelse return error.InvalidMachine;
                    const dst: regalloc.PhysReg = switch (dst_location) {
                        .phys => |physical| physical,
                        .spill => .r10,
                        .none => return error.InvalidMachine,
                    };
                    if (dst.class() != .gp) return error.InvalidMachine;
                    const lhs = try readInto(buffer, allocation, spill_plan, inst.uses[0], xmm_scratch_primary, stats);
                    const rhs = try readInto(buffer, allocation, spill_plan, rhs_use, xmm_scratch_secondary, stats);
                    try emitXmmCompareResult(buffer, operation, dst, lhs, rhs, wide, stats);
                    stats.xmm_comparisons += 1;
                    switch (dst_location) {
                        .spill => try emitSpillStore(buffer, try spillSlot(spill_plan, inst.defs[0]), dst, stats),
                        else => {},
                    }
                },
                .int_to_float, .int_to_double, .long_to_float, .long_to_double => {
                    const input_wide = operation == .long_to_float or operation == .long_to_double;
                    const output_wide = operation == .int_to_double or operation == .long_to_double;
                    const expected_defs: usize = if (output_wide) 2 else 1;
                    const expected_uses: usize = if (input_wide) 2 else 1;
                    if (wide != output_wide or
                        inst.defs.len != expected_defs or
                        inst.uses.len != expected_uses) return error.InvalidMachine;
                    const expected_src: machine.ValueType = if (input_wide) .long else .int;
                    const expected_dst: machine.ValueType = if (output_wide) .double else .float;
                    if (source.reg_types[inst.defs[0]] != expected_dst or source.reg_types[inst.uses[0]] != expected_src) return error.InvalidMachine;
                    const dst_location = allocation.locationOf(inst.defs[0]) orelse return error.InvalidMachine;
                    const dst: regalloc.PhysReg = switch (dst_location) {
                        .phys => |physical| physical,
                        .spill => xmm_scratch_primary,
                        .none => return error.InvalidMachine,
                    };
                    if (dst.class() != .xmm) return error.InvalidMachine;
                    const src = try readInto(buffer, allocation, spill_plan, inst.uses[0], .r10, stats);
                    if (src.class() != .gp) return error.InvalidMachine;
                    try emitGpToFloatConvert(buffer, dst, src, input_wide, output_wide);
                    stats.xmm_insts += 1;
                    stats.xmm_conversions += 1;
                    stats.native_insts += 1;
                    switch (dst_location) {
                        .spill => try emitSpillStore(buffer, try spillSlot(spill_plan, inst.defs[0]), dst, stats),
                        else => {},
                    }
                },
                .float_to_int, .float_to_long, .double_to_int, .double_to_long => {
                    const input_wide = operation == .double_to_int or operation == .double_to_long;
                    const output_wide = operation == .float_to_long or operation == .double_to_long;
                    const expected_defs: usize = if (output_wide) 2 else 1;
                    const expected_uses: usize = if (input_wide) 2 else 1;
                    if (wide != input_wide or
                        inst.defs.len != expected_defs or
                        inst.uses.len != expected_uses) return error.InvalidMachine;
                    const expected_src: machine.ValueType = if (input_wide) .double else .float;
                    const expected_dst: machine.ValueType = if (output_wide) .long else .int;
                    if (source.reg_types[inst.defs[0]] != expected_dst or source.reg_types[inst.uses[0]] != expected_src) return error.InvalidMachine;
                    const dst_location = allocation.locationOf(inst.defs[0]) orelse return error.InvalidMachine;
                    const dst: regalloc.PhysReg = switch (dst_location) {
                        .phys => |physical| physical,
                        .spill => .r10,
                        .none => return error.InvalidMachine,
                    };
                    if (dst.class() != .gp) return error.InvalidMachine;
                    const src = try readInto(buffer, allocation, spill_plan, inst.uses[0], xmm_scratch_primary, stats);
                    if (src.class() != .xmm) return error.InvalidMachine;
                    try emitSaturatingFloatToInteger(buffer, dst, src, input_wide, output_wide, stats);
                    stats.xmm_conversions += 1;
                    stats.xmm_saturating_conversions += 1;
                    switch (dst_location) {
                        .spill => try emitSpillStore(buffer, try spillSlot(spill_plan, inst.defs[0]), dst, stats),
                        else => {},
                    }
                },
                .float_to_double, .double_to_float => {
                    const input_wide = operation == .double_to_float;
                    const output_wide = !input_wide;
                    const expected_defs: usize = if (output_wide) 2 else 1;
                    const expected_uses: usize = if (input_wide) 2 else 1;
                    if (wide != input_wide or
                        inst.defs.len != expected_defs or
                        inst.uses.len != expected_uses) return error.InvalidMachine;
                    const expected_src: machine.ValueType = if (input_wide) .double else .float;
                    const expected_dst: machine.ValueType = if (output_wide) .double else .float;
                    if (source.reg_types[inst.defs[0]] != expected_dst or source.reg_types[inst.uses[0]] != expected_src) return error.InvalidMachine;
                    const dst_location = allocation.locationOf(inst.defs[0]) orelse return error.InvalidMachine;
                    const dst: regalloc.PhysReg = switch (dst_location) {
                        .phys => |physical| physical,
                        .spill => xmm_scratch_primary,
                        .none => return error.InvalidMachine,
                    };
                    if (dst.class() != .xmm) return error.InvalidMachine;
                    const src = try readInto(buffer, allocation, spill_plan, inst.uses[0], dst, stats);
                    if (src.class() != .xmm) return error.InvalidMachine;
                    try emitFloatWidthConvert(buffer, dst, src, input_wide);
                    stats.xmm_insts += 1;
                    stats.xmm_conversions += 1;
                    stats.native_insts += 1;
                    switch (dst_location) {
                        .spill => try emitSpillStore(buffer, try spillSlot(spill_plan, inst.defs[0]), dst, stats),
                        else => {},
                    }
                },
                else => return error.UnsupportedInstruction,
            }
        },
        // The following resolve performs the same null discrimination and
        // transfers failure to the runtime helper, so no duplicate branch is
        // emitted for this lowering marker.
        .check_null => {},
        .check_bounds => {
            if (inst.uses.len != 1 or source.isGcRoot(inst.uses[0])) return error.InvalidMachine;
            try emitBoundsExceptionCheck(allocator, buffer, allocation, spill_plan, inst, runtime orelse return error.MissingRuntimeAbi, cold_bounds, stats);
        },
        .resolve_handle => try emitResolve(allocator, buffer, allocation, inst, runtime orelse return error.MissingRuntimeAbi, cold, stats),
        .loop_epoch_guard => try emitLoopEpochGuard(allocator, buffer, allocation, inst, cold, stats),
        .array_load_ptr => {
            if (inst.defs.len != 1 or inst.uses.len != 1) return error.InvalidMachine;
            const abi = runtime orelse return error.MissingRuntimeAbi;
            const layout = try referenceArrayLayout(abi);
            if (!source.isGcRoot(inst.defs[0]) or source.isGcRoot(inst.uses[0])) return error.InvalidRuntimeAbi;
            const destination_reg: regalloc.PhysReg = switch (allocation.locationOf(inst.defs[0]) orelse return error.InvalidMachine) {
                .phys => |physical| physical,
                .spill => .r11,
                .none => return error.InvalidMachine,
            };
            const destination = try x64Reg(destination_reg);
            const address = try x64Reg(try physOf(allocation, inst.address orelse return error.InvalidMachine));
            const index = try x64Reg(try readInto(buffer, allocation, spill_plan, inst.uses[0], .r10, stats));
            try emitMovRawRaw(buffer, scratch_index, index, false);
            try emitRexX(buffer, true, destination, scratch_index, address);
            try buffer.emitU8(0x8b);
            try emitIndexedMemoryOperand(buffer, destination, address, scratch_index, 3, @intCast(layout.data_offset));
            try writeFrom(buffer, allocation, spill_plan, inst.defs[0], destination_reg, true, stats);
            stats.array_loads += 1;
            stats.pointer_loads += 1;
            stats.native_insts += 2;
        },
        .field_load_ptr => {
            if (inst.defs.len != 1 or inst.uses.len != 0) return error.InvalidMachine;
            const abi = runtime orelse return error.MissingRuntimeAbi;
            const layout = try fieldLayout(abi, inst.field_idx);
            try verifyFieldType(source, inst.defs[0], layout.storage);
            const destination_reg: regalloc.PhysReg = switch (allocation.locationOf(inst.defs[0]) orelse return error.InvalidMachine) {
                .phys => |physical| physical,
                .spill => .r10,
                .none => return error.InvalidMachine,
            };
            const dst = try x64Reg(destination_reg);
            const address = try x64Reg(try physOf(allocation, inst.address orelse return error.InvalidMachine));
            try emitLoadField(buffer, dst, address, @intCast(layout.offset), layout.storage);
            try writeFrom(buffer, allocation, spill_plan, inst.defs[0], destination_reg, layout.storage == .i64 or layout.storage == .reference, stats);
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
                const index = try x64Reg(try readInto(buffer, allocation, spill_plan, inst.uses[0], .r10, stats));
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
            const value = try x64Reg(try readInto(buffer, allocation, spill_plan, inst.uses[0], .r11, stats));
            const index = try x64Reg(try readInto(buffer, allocation, spill_plan, inst.uses[1], .r10, stats));
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
            const src = try x64Reg(try readInto(buffer, allocation, spill_plan, inst.uses[0], .r10, stats));
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
            const value = try x64Reg(try readInto(buffer, allocation, spill_plan, inst.uses[0], .r11, stats));
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
            const stored = try x64Reg(try readInto(buffer, allocation, spill_plan, inst.uses[0], .r11, stats));
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
            const stored = try x64Reg(try readInto(buffer, allocation, spill_plan, inst.uses[0], .r11, stats));
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
            try emitJump(buffer, try edgeTargetLabel(source, labels, edge_labels, block_id, inst.target));
            stats.jumps += 1;
            stats.native_insts += 1;
        },
        .branch => {
            const condition = inst.condition orelse return error.InvalidMachine;
            if (inst.uses.len == 1) {
                try emitCmpRegZero(buffer, try readInto(buffer, allocation, spill_plan, inst.uses[0], .r10, stats));
            } else if (inst.uses.len == 2) {
                const lhs = try readInto(buffer, allocation, spill_plan, inst.uses[0], .r10, stats);
                const rhs = try readInto(buffer, allocation, spill_plan, inst.uses[1], .r11, stats);
                try emitCmpRegs(buffer, lhs, rhs);
            } else {
                return error.InvalidMachine;
            }
            try emitJcc(buffer, condition, try edgeTargetLabel(source, labels, edge_labels, block_id, inst.target));
            try emitJump(buffer, try edgeTargetLabel(source, labels, edge_labels, block_id, inst.false_target));
            stats.branches += 1;
            stats.jumps += 1;
            stats.native_insts += 3;
        },
        .ret => {
            if (inst.uses.len == 1 or inst.uses.len == 2) {
                const ty = source.reg_types[inst.uses[0]];
                if (inst.uses.len == 2 and ty != .long and ty != .double) return error.InvalidMachine;
                const wide = isWideType(ty) or source.isGcRoot(inst.uses[0]);
                if (valueClass(ty) == .xmm) {
                    const src = try readInto(buffer, allocation, spill_plan, inst.uses[0], xmm_scratch_primary, stats);
                    if (src.class() != .xmm) return error.InvalidMachine;
                    try emitXmmToGp(buffer, .rax, src, wide);
                    stats.native_insts += 1;
                    stats.xmm_insts += 1;
                } else {
                    const src = try readInto(buffer, allocation, spill_plan, inst.uses[0], .rax, stats);
                    try recordPhysicalMove(buffer, .rax, src, wide, stats);
                }
            }
            if (inst.uses.len > 2) return error.UnsupportedInstruction;
            try emitFrameLeave(buffer, frame_bytes, preserve_nonvolatile);
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
    site_id: u32,
    deopt_id: ?u32,
    roots: []runtime_stack_map.RootLocation,
};

fn instructionSafepointId(inst: machine.Inst) ?u32 {
    return if (inst.opcode == .resolve_handle)
        inst.resolve_id
    else if (inst.opcode == .loop_epoch_guard)
        inst.guard_site_id
    else if (inst.opcode == .check_bounds)
        inst.exception_site_id
    else
        null;
}

fn blockEntryDeoptCount(options: ?DeoptOptions) Error!u32 {
    const deopt = options orelse return 0;
    var count: u32 = 0;
    for (deopt.points) |point| {
        if ((point.safepoint_id == null) == (point.block_entry == null)) return error.InvalidDeoptMetadata;
        if (point.block_entry != null) count = std.math.add(u32, count, 1) catch return error.InvalidDeoptMetadata;
    }
    return count;
}

fn deoptPointSiteId(source: *const machine.Function, options: DeoptOptions, point_index: usize) Error!u32 {
    if (point_index >= options.points.len) return error.InvalidDeoptMetadata;
    const point = options.points[point_index];
    if ((point.safepoint_id == null) == (point.block_entry == null)) return error.InvalidDeoptMetadata;
    if (point.safepoint_id) |site_id| return site_id;

    const block = point.block_entry orelse return error.InvalidDeoptMetadata;
    if (block >= source.blocks.len or source.blocks[block].insts.len == 0 or
        source.blocks[block].insts[0].pc != point.dex_pc) return error.InvalidDeoptMetadata;
    var ordinal: u32 = 0;
    for (options.points[0..point_index]) |previous| {
        if (previous.block_entry) |previous_block| {
            if (previous_block == block) return error.InvalidDeoptMetadata;
            ordinal = std.math.add(u32, ordinal, 1) catch return error.InvalidDeoptMetadata;
        }
    }
    return std.math.add(u32, try machineSafepointCount(source), ordinal) catch return error.InvalidDeoptMetadata;
}

fn deoptIdForSite(source: *const machine.Function, options: ?DeoptOptions, site_id: u32) Error!?u32 {
    const deopt = options orelse return null;
    var result: ?u32 = null;
    for (deopt.points, 0..) |point, point_index| {
        if (try deoptPointSiteId(source, deopt, point_index) != site_id) continue;
        if (result != null) return error.InvalidDeoptMetadata;
        result = point.id;
    }
    return result;
}

const RootLiveness = struct {
    allocator: std.mem.Allocator,
    words_per_block: usize,
    live_in: []usize,
    live_out: []usize,

    fn deinit(self: *RootLiveness) void {
        self.allocator.free(self.live_out);
        self.allocator.free(self.live_in);
        self.* = undefined;
    }

    fn blockOut(self: *const RootLiveness, block: cfg.BlockId) []const usize {
        const first = @as(usize, block) * self.words_per_block;
        return self.live_out[first .. first + self.words_per_block];
    }

    fn blockIn(self: *const RootLiveness, block: cfg.BlockId) []const usize {
        const first = @as(usize, block) * self.words_per_block;
        return self.live_in[first .. first + self.words_per_block];
    }
};

const root_word_bits = @bitSizeOf(usize);

fn rootIsSet(words: []const usize, reg: machine.RegId) bool {
    const index: usize = reg;
    return (words[index / root_word_bits] & (@as(usize, 1) << @intCast(index % root_word_bits))) != 0;
}

fn setRoot(words: []usize, reg: machine.RegId) void {
    const index: usize = reg;
    words[index / root_word_bits] |= @as(usize, 1) << @intCast(index % root_word_bits);
}

fn unsetRoot(words: []usize, reg: machine.RegId) void {
    const index: usize = reg;
    words[index / root_word_bits] &= ~(@as(usize, 1) << @intCast(index % root_word_bits));
}

fn trackedRegister(source: *const machine.Function, reg: machine.RegId, roots_only: bool) bool {
    return !roots_only or source.isGcRoot(reg);
}

fn addTrackedUse(source: *const machine.Function, words: []usize, reg: machine.RegId, roots_only: bool) Error!void {
    if (reg >= source.reg_types.len) return error.InvalidMachine;
    if (trackedRegister(source, reg, roots_only)) setRoot(words, reg);
}

fn addInstructionTrackedUses(source: *const machine.Function, words: []usize, inst: machine.Inst, roots_only: bool) Error!void {
    for (inst.uses) |reg| try addTrackedUse(source, words, reg, roots_only);
    if (inst.address) |reg| try addTrackedUse(source, words, reg, roots_only);
    if (inst.state_handle) |reg| try addTrackedUse(source, words, reg, roots_only);
}

fn isPhiOwnedBy(source: *const machine.Function, reg: machine.RegId, block: cfg.BlockId) bool {
    if (reg >= source.value_kinds.len or source.value_kinds[reg] != .phi) return false;
    return reg < source.source.source.values.len and source.source.source.values[reg].block == block;
}

fn buildLiveness(allocator: std.mem.Allocator, source: *const machine.Function, roots_only: bool) Error!RootLiveness {
    const word_count = (source.reg_types.len + root_word_bits - 1) / root_word_bits;
    if (word_count == 0 and source.stats.resolves != 0) return error.InvalidMachine;
    if (word_count != 0 and source.blocks.len > std.math.maxInt(usize) / word_count) return error.InvalidMachine;
    const slots = source.blocks.len * word_count;

    const live_in = try allocator.alloc(usize, slots);
    errdefer allocator.free(live_in);
    const live_out = try allocator.alloc(usize, slots);
    errdefer allocator.free(live_out);
    const block_uses = try allocator.alloc(usize, slots);
    defer allocator.free(block_uses);
    const block_defs = try allocator.alloc(usize, slots);
    defer allocator.free(block_defs);
    const next_out = try allocator.alloc(usize, word_count);
    defer allocator.free(next_out);
    const edge_live = try allocator.alloc(usize, word_count);
    defer allocator.free(edge_live);

    @memset(live_in, 0);
    @memset(live_out, 0);
    @memset(block_uses, 0);
    @memset(block_defs, 0);

    for (source.blocks) |block| {
        const first = @as(usize, block.id) * word_count;
        const uses = block_uses[first .. first + word_count];
        const defs = block_defs[first .. first + word_count];
        for (block.insts) |inst| {
            for (inst.uses) |reg| {
                if (reg >= source.reg_types.len) return error.InvalidMachine;
                if (trackedRegister(source, reg, roots_only) and !rootIsSet(defs, reg)) setRoot(uses, reg);
            }
            if (inst.address) |reg| {
                if (reg >= source.reg_types.len) return error.InvalidMachine;
                if (trackedRegister(source, reg, roots_only) and !rootIsSet(defs, reg)) setRoot(uses, reg);
            }
            if (inst.state_handle) |reg| {
                if (reg >= source.reg_types.len) return error.InvalidMachine;
                if (trackedRegister(source, reg, roots_only) and !rootIsSet(defs, reg)) setRoot(uses, reg);
            }
            for (inst.defs) |reg| {
                if (reg >= source.reg_types.len) return error.InvalidMachine;
                if (trackedRegister(source, reg, roots_only)) setRoot(defs, reg);
            }
        }
    }

    var changed = true;
    while (changed) {
        changed = false;
        var block_index = source.blocks.len;
        while (block_index != 0) {
            block_index -= 1;
            const block = source.blocks[block_index];
            @memset(next_out, 0);

            for (source.successors[block.id]) |successor| {
                if (successor >= source.blocks.len) return error.InvalidMachine;
                const successor_first = @as(usize, successor) * word_count;
                const successor_in = live_in[successor_first .. successor_first + word_count];
                @memcpy(edge_live, successor_in);

                var matching_edges: u32 = 0;
                for (source.edges) |edge| {
                    if (edge.from != block.id or edge.to != successor) continue;
                    matching_edges += 1;
                    if (matching_edges != 1) return error.InvalidMachine;
                    // Phi copies are parallel: first kill every destination,
                    // then add only sources whose destinations are live.
                    for (edge.moves) |move| {
                        if (move.dst >= source.reg_types.len or move.src >= source.reg_types.len) return error.InvalidMachine;
                        if (trackedRegister(source, move.dst, roots_only)) unsetRoot(edge_live, move.dst);
                    }
                    for (edge.moves) |move| {
                        if (!trackedRegister(source, move.dst, roots_only) or !rootIsSet(successor_in, move.dst)) continue;
                        if (!trackedRegister(source, move.src, roots_only)) return error.InvalidMachine;
                        setRoot(edge_live, move.src);
                    }
                }
                for (0..source.reg_types.len) |reg_index| {
                    const reg: machine.RegId = @intCast(reg_index);
                    if (!rootIsSet(successor_in, reg) or !isPhiOwnedBy(source, reg, successor)) continue;
                    var has_incoming = false;
                    for (source.edges) |edge| {
                        if (edge.from != block.id or edge.to != successor) continue;
                        for (edge.moves) |move| {
                            if (move.dst == reg) {
                                has_incoming = true;
                                break;
                            }
                        }
                        if (has_incoming) break;
                    }
                    if (!has_incoming) return error.InvalidMachine;
                }
                for (next_out, edge_live) |*out_word, edge_word| out_word.* |= edge_word;
            }

            const first = block_index * word_count;
            const old_in = live_in[first .. first + word_count];
            const old_out = live_out[first .. first + word_count];
            const uses = block_uses[first .. first + word_count];
            const defs = block_defs[first .. first + word_count];
            for (0..word_count) |word| {
                const next_in = uses[word] | (next_out[word] & ~defs[word]);
                if (old_out[word] != next_out[word] or old_in[word] != next_in) changed = true;
                old_out[word] = next_out[word];
                old_in[word] = next_in;
            }
        }
    }

    return .{
        .allocator = allocator,
        .words_per_block = word_count,
        .live_in = live_in,
        .live_out = live_out,
    };
}

fn rootLocationLess(_: void, a: runtime_stack_map.RootLocation, b: runtime_stack_map.RootLocation) bool {
    return a.payload < b.payload;
}

fn pendingRootMapLess(_: void, a: PendingRootMap, b: PendingRootMap) bool {
    return a.site_id < b.site_id;
}

fn spillSlotFor(plan: *const regalloc.SpillPlan, reg: machine.RegId) ?regalloc.SpillSlot {
    for (plan.slots) |slot| if (slot.reg == reg) return slot;
    return null;
}

fn validateSpillPlan(
    source: *const machine.Function,
    allocation: *const regalloc.Allocation,
    plan: *const regalloc.SpillPlan,
) Error!void {
    try plan.verify();
    if (plan.source != source or plan.location_count != allocation.locations.len) return error.InvalidMachine;
    for (allocation.intervals) |interval| {
        const slot = spillSlotFor(plan, interval.reg);
        switch (allocation.locations[interval.reg]) {
            .spill => |index| if (slot == null or slot.?.slot != index) return error.InvalidMachine,
            .phys => if (slot != null) return error.InvalidMachine,
            .none => return error.InvalidMachine,
        }
    }
}

fn collectLiveRoots(
    allocator: std.mem.Allocator,
    source: *const machine.Function,
    allocation: *const regalloc.Allocation,
    spill_plan: *const regalloc.SpillPlan,
    live: []const usize,
) Error![]runtime_stack_map.RootLocation {
    var roots: std.ArrayList(runtime_stack_map.RootLocation) = .empty;
    errdefer roots.deinit(allocator);
    var physical_roots = [_]bool{false} ** 16;
    for (0..source.reg_types.len) |reg_index| {
        const reg: machine.RegId = @intCast(reg_index);
        if (!rootIsSet(live, reg)) continue;
        switch (allocation.locationOf(reg) orelse return error.InvalidMachine) {
            .phys => |assigned| {
                const physical = try x64Reg(assigned);
                if (physical_roots[physical]) return error.InvalidMachine;
                physical_roots[physical] = true;
                try roots.append(allocator, runtime_stack_map.RootLocation.nativeRegister(physical));
            },
            .spill => {
                const slot = spillSlotFor(spill_plan, reg) orelse return error.InvalidMachine;
                try roots.append(allocator, runtime_stack_map.RootLocation.stackSlot(@intCast(slot.byte_offset)));
            },
            .none => return error.InvalidMachine,
        }
    }
    std.mem.sort(runtime_stack_map.RootLocation, roots.items, {}, rootLocationLess);
    return roots.toOwnedSlice(allocator);
}

fn buildRootMaps(
    allocator: std.mem.Allocator,
    source: *const machine.Function,
    allocation: *const regalloc.Allocation,
    spill_plan: *const regalloc.SpillPlan,
    stats: *Stats,
    deopt: ?DeoptOptions,
) Error!?runtime_stack_map.Table {
    try validateSpillPlan(source, allocation, spill_plan);
    const resolve_sites = std.math.add(u32, source.stats.resolves, source.stats.loop_epoch_guards) catch return error.InvalidMachine;
    const machine_sites = std.math.add(u32, resolve_sites, source.stats.bounds_exception_sites) catch return error.InvalidMachine;
    const expected_sites = std.math.add(u32, machine_sites, try blockEntryDeoptCount(deopt)) catch return error.InvalidMachine;
    if (expected_sites == 0) return null;

    var liveness = try buildLiveness(allocator, source, true);
    defer liveness.deinit();
    const live = try allocator.alloc(usize, liveness.words_per_block);
    defer allocator.free(live);

    var pending: std.ArrayList(PendingRootMap) = .empty;
    defer {
        for (pending.items) |site| allocator.free(site.roots);
        pending.deinit(allocator);
    }

    for (source.blocks) |block| {
        @memcpy(live, liveness.blockOut(block.id));
        var instruction_index = block.insts.len;
        while (instruction_index != 0) {
            instruction_index -= 1;
            const inst = block.insts[instruction_index];
            for (inst.defs) |reg| {
                if (reg >= source.reg_types.len) return error.InvalidMachine;
                if (source.isGcRoot(reg)) unsetRoot(live, reg);
            }
            try addInstructionTrackedUses(source, live, inst, true);
            if (inst.opcode == .resolve_handle or inst.opcode == .loop_epoch_guard or
                (inst.opcode == .check_bounds and inst.exception_site_id != null))
            {
                const site_id = if (inst.opcode == .resolve_handle)
                    inst.resolve_id orelse return error.InvalidMachine
                else if (inst.opcode == .loop_epoch_guard)
                    inst.guard_site_id orelse return error.InvalidMachine
                else
                    inst.exception_site_id orelse return error.InvalidMachine;
                const canonical = inst.state_handle orelse return error.InvalidMachine;
                if (canonical >= source.reg_types.len or !source.isGcRoot(canonical) or !rootIsSet(live, canonical)) return error.InvalidMachine;
                const deopt_id = try deoptIdForSite(source, deopt, site_id);
                const owned = try collectLiveRoots(allocator, source, allocation, spill_plan, live);
                pending.append(allocator, .{
                    .site_id = site_id,
                    .deopt_id = deopt_id,
                    .roots = owned,
                }) catch |err| {
                    allocator.free(owned);
                    return err;
                };
                stats.root_map_sites += 1;
                stats.root_map_locations += @intCast(owned.len);
            }
        }
    }

    if (deopt) |deopt_options| {
        for (deopt_options.points, 0..) |point, point_index| {
            const block = point.block_entry orelse continue;
            if (block >= source.blocks.len) return error.InvalidDeoptMetadata;
            const owned = try collectLiveRoots(
                allocator,
                source,
                allocation,
                spill_plan,
                liveness.blockIn(block),
            );
            pending.append(allocator, .{
                .site_id = try deoptPointSiteId(source, deopt_options, point_index),
                .deopt_id = point.id,
                .roots = owned,
            }) catch |err| {
                allocator.free(owned);
                return err;
            };
            stats.deopt_block_entries += 1;
            stats.root_map_sites += 1;
            stats.root_map_locations = std.math.add(u32, stats.root_map_locations, @intCast(owned.len)) catch
                return error.InvalidDeoptMetadata;
        }
    }
    if (pending.items.len != expected_sites) return error.InvalidMachine;
    std.mem.sort(PendingRootMap, pending.items, {}, pendingRootMapLess);

    var specs: std.ArrayList(runtime_stack_map.MapSpec) = .empty;
    defer specs.deinit(allocator);
    for (pending.items) |site| {
        try specs.append(allocator, .{
            .pc_offset = site.site_id,
            .roots = site.roots,
            .deopt_id = site.deopt_id,
        });
    }
    return try runtime_stack_map.Table.init(allocator, specs.items, .{
        .native_register_count = 16,
        .interpreter_register_count = 0,
        .max_frame_depth = 0,
        .max_shadow_roots = 0,
    });
}

const SafepointPosition = struct {
    position: u32,
    site_id: u32,
    pc: u32,
    block: cfg.BlockId,
    instruction: u32,
};

fn findSafepointPosition(source: *const machine.Function, site_id: u32) Error!SafepointPosition {
    var position: u32 = 2;
    var result: ?SafepointPosition = null;
    for (source.blocks) |block| {
        for (block.insts, 0..) |inst, instruction| {
            if (instructionSafepointId(inst)) |candidate| {
                if (candidate == site_id) {
                    if (result != null) return error.InvalidDeoptMetadata;
                    result = .{
                        .position = position,
                        .site_id = site_id,
                        .pc = inst.pc orelse return error.InvalidDeoptMetadata,
                        .block = block.id,
                        .instruction = @intCast(instruction),
                    };
                }
            }
            position = std.math.add(u32, position, 2) catch return error.InvalidMachine;
        }
    }
    return result orelse error.MissingDeoptSafepoint;
}

fn resolveDeoptPosition(
    source: *const machine.Function,
    options: DeoptOptions,
    point_index: usize,
) Error!SafepointPosition {
    if (point_index >= options.points.len) return error.InvalidDeoptMetadata;
    const point = options.points[point_index];
    if (point.safepoint_id) |site_id| {
        const position = try findSafepointPosition(source, site_id);
        if (position.pc != point.dex_pc) return error.InvalidDeoptMetadata;
        return position;
    }
    const block = point.block_entry orelse return error.InvalidDeoptMetadata;
    const site_id = try deoptPointSiteId(source, options, point_index);
    if (block >= source.blocks.len or source.blocks[block].insts.len == 0) return error.InvalidDeoptMetadata;
    return .{
        .position = 0,
        .site_id = site_id,
        .pc = point.dex_pc,
        .block = block,
        .instruction = 0,
    };
}

fn computeInstructionLiveIn(
    source: *const machine.Function,
    liveness: *const RootLiveness,
    safepoint: SafepointPosition,
    live: []usize,
) Error!void {
    if (safepoint.block >= source.blocks.len or live.len != liveness.words_per_block) return error.InvalidMachine;
    @memcpy(live, liveness.blockOut(safepoint.block));
    const instructions = source.blocks[safepoint.block].insts;
    if (safepoint.instruction >= instructions.len) return error.InvalidMachine;
    var index = instructions.len;
    while (index > safepoint.instruction) {
        index -= 1;
        const inst = instructions[index];
        for (inst.defs) |reg| unsetRoot(live, reg);
        try addInstructionTrackedUses(source, live, inst, false);
    }
}

fn validateDeoptValueType(source: *const machine.Function, value: DeoptValueSpec, reg: machine.RegId) Error!void {
    if (reg >= source.reg_types.len) return error.InvalidDeoptMetadata;
    const ty = source.reg_types[reg];
    switch (value.kind) {
        .reference => if (!source.isGcRoot(reg) or (ty != .object and ty != .unknown)) return error.InvalidDeoptMetadata,
        .scalar64 => if (source.isGcRoot(reg) or (ty != .long and ty != .double and ty != .unknown)) return error.InvalidDeoptMetadata,
        .scalar32 => if (source.isGcRoot(reg) or isWideType(ty) or ty == .conflict) return error.InvalidDeoptMetadata,
    }
}

fn rootMapContainsSource(maps: *const runtime_stack_map.Table, site_id: u32, source: runtime_deopt.Source) Error!bool {
    const record = try maps.find(site_id);
    for (maps.rootsFor(record)) |root| {
        switch (source) {
            .native_register => |physical| if (root.kind == .native_register and root.payload == physical) return true,
            .stack_slot => |offset| if (root.kind == .stack_slot and root.stackOffset() == offset) return true,
            else => return false,
        }
    }
    return false;
}

fn translateDeoptValues(
    allocator: std.mem.Allocator,
    source: *const machine.Function,
    allocation: *const regalloc.Allocation,
    spill_plan: *const regalloc.SpillPlan,
    maps: *const runtime_stack_map.Table,
    safepoint_id: u32,
    live_values: []const usize,
    physical_owners: *[16]?machine.RegId,
    xmm_owners: *[8]?machine.RegId,
    specs: []const DeoptValueSpec,
    stats: *Stats,
) Error![]runtime_deopt.ValueSpec {
    const values = try allocator.alloc(runtime_deopt.ValueSpec, specs.len);
    errdefer allocator.free(values);
    for (specs, 0..) |value, value_index| {
        const translated_source: runtime_deopt.Source = switch (value.source) {
            .constant => |bits| .{ .constant = bits },
            .machine_register => |reg| blk: {
                if (reg >= source.reg_types.len or !rootIsSet(live_values, reg)) return error.DeadDeoptValue;
                try validateDeoptValueType(source, value, reg);
                const translated: runtime_deopt.Source = switch (allocation.locationOf(reg) orelse return error.InvalidMachine) {
                    .phys => |assigned| switch (assigned.class()) {
                        .gp => gp: {
                            const physical = try x64Reg(assigned);
                            if (physical_owners[physical]) |owner| {
                                if (owner != reg) return error.AliasedDeoptValue;
                            } else {
                                physical_owners[physical] = reg;
                            }
                            break :gp .{ .native_register = physical };
                        },
                        .xmm => xmm: {
                            if (value.kind == .reference) return error.InvalidDeoptMetadata;
                            const physical = try xmmReg(assigned);
                            if (xmm_owners[physical]) |owner| {
                                if (owner != reg) return error.AliasedDeoptValue;
                            } else {
                                xmm_owners[physical] = reg;
                            }
                            stats.deopt_xmm_values += 1;
                            break :xmm .{ .xmm_register = physical };
                        },
                    },
                    .spill => spill: {
                        const slot = spillSlotFor(spill_plan, reg) orelse return error.InvalidDeoptMetadata;
                        if (slot.byte_offset > std.math.maxInt(i32)) return error.InvalidDeoptMetadata;
                        stats.deopt_stack_values += 1;
                        break :spill .{ .stack_slot = @intCast(slot.byte_offset) };
                    },
                    .none => return error.InvalidMachine,
                };
                if (value.kind == .reference and !try rootMapContainsSource(maps, safepoint_id, translated)) {
                    return error.InvalidDeoptMetadata;
                }
                break :blk translated;
            },
        };
        values[value_index] = .{
            .vreg = value.vreg,
            .kind = value.kind,
            .source = translated_source,
        };
    }
    return values;
}

fn buildDeoptTable(
    allocator: std.mem.Allocator,
    source: *const machine.Function,
    allocation: *const regalloc.Allocation,
    spill_plan: *const regalloc.SpillPlan,
    root_maps: ?*const runtime_stack_map.Table,
    options: ?DeoptOptions,
    stats: *Stats,
) Error!?runtime_deopt.Table {
    try validateSpillPlan(source, allocation, spill_plan);
    const deopt = options orelse return null;
    if (deopt.points.len == 0 or deopt.register_count == 0) return error.InvalidDeoptMetadata;
    const maps = root_maps orelse return error.MissingDeoptSafepoint;
    var value_liveness = try buildLiveness(allocator, source, false);
    defer value_liveness.deinit();
    const live_values = try allocator.alloc(usize, value_liveness.words_per_block);
    defer allocator.free(live_values);

    const translated_points = try allocator.alloc(runtime_deopt.PointSpec, deopt.points.len);
    defer allocator.free(translated_points);
    const translated_inline_frames = try allocator.alloc([]runtime_deopt.InlineFrameSpec, deopt.points.len);
    var initialized_inline_frames: usize = 0;
    defer {
        for (translated_inline_frames[0..initialized_inline_frames]) |frames| allocator.free(frames);
        allocator.free(translated_inline_frames);
    }
    var owned_values: std.ArrayList([]runtime_deopt.ValueSpec) = .empty;
    defer {
        for (owned_values.items) |values| allocator.free(values);
        owned_values.deinit(allocator);
    }

    for (deopt.points, 0..) |point, point_index| {
        const safepoint = try resolveDeoptPosition(source, deopt, point_index);
        try computeInstructionLiveIn(source, &value_liveness, safepoint, live_values);
        var physical_owners: [16]?machine.RegId = @splat(null);
        var xmm_owners: [8]?machine.RegId = @splat(null);
        const inline_frames = try allocator.alloc(runtime_deopt.InlineFrameSpec, point.inline_frames.len);
        translated_inline_frames[point_index] = inline_frames;
        initialized_inline_frames += 1;
        for (point.inline_frames, 0..) |frame, frame_index| {
            const values = try translateDeoptValues(
                allocator,
                source,
                allocation,
                spill_plan,
                maps,
                safepoint.site_id,
                live_values,
                &physical_owners,
                &xmm_owners,
                frame.values,
                stats,
            );
            owned_values.append(allocator, values) catch |err| {
                allocator.free(values);
                return err;
            };
            inline_frames[frame_index] = .{
                .method_id = frame.method_id,
                .dex_pc = frame.dex_pc,
                .register_count = frame.register_count,
                .values = values,
            };
        }
        const values = try translateDeoptValues(
            allocator,
            source,
            allocation,
            spill_plan,
            maps,
            safepoint.site_id,
            live_values,
            &physical_owners,
            &xmm_owners,
            point.values,
            stats,
        );
        owned_values.append(allocator, values) catch |err| {
            allocator.free(values);
            return err;
        };
        translated_points[point_index] = .{
            .id = point.id,
            .method_id = point.method_id,
            .dex_pc = point.dex_pc,
            .values = values,
            .inline_frames = inline_frames,
        };
        stats.deopt_points += 1;
        stats.deopt_frames = std.math.add(u32, stats.deopt_frames, @intCast(point.inline_frames.len + 1)) catch
            return error.InvalidDeoptMetadata;
        var point_value_count = point.values.len;
        for (point.inline_frames) |frame| point_value_count += frame.values.len;
        stats.deopt_values = std.math.add(u32, stats.deopt_values, @intCast(point_value_count)) catch
            return error.InvalidDeoptMetadata;
    }

    var table = try runtime_deopt.Table.init(allocator, translated_points, .{
        .register_count = deopt.register_count,
        .native_register_count = 16,
        .xmm_register_count = 8,
        .max_dex_pc = deopt.max_dex_pc,
    });
    errdefer table.deinit();
    try table.validateStackMaps(maps, false);
    try table.validateAllLinked(maps);
    return table;
}

const OsrRematOperands = struct {
    handle: machine.RegId,
    address: machine.RegId,
};

fn osrRematOperands(source: *const machine.Function, resolve_id: u32) Error!OsrRematOperands {
    const barriers = source.source.barriers orelse return error.InvalidDeoptMetadata;
    if (resolve_id >= barriers.resolves.len or resolve_id >= source.source.resolve_values.len) {
        return error.InvalidDeoptMetadata;
    }
    const resolve = barriers.resolves[resolve_id];
    const address = source.source.resolve_values[resolve_id];
    if (resolve.handle >= source.runtime_values.len or address >= source.runtime_values.len) {
        return error.InvalidDeoptMetadata;
    }
    switch (source.runtime_values[resolve.handle]) {
        .dalvik => |value| if (value.value != resolve.handle or !value.gc_root) return error.InvalidDeoptMetadata,
        .derived_ptr => return error.InvalidDeoptMetadata,
    }
    switch (source.runtime_values[address]) {
        .derived_ptr => |ptr| if (ptr.handle != resolve.handle or ptr.resolve != resolve_id or ptr.token != resolve.token) {
            return error.InvalidDeoptMetadata;
        },
        .dalvik => return error.InvalidDeoptMetadata,
    }
    return .{ .handle = resolve.handle, .address = address };
}

fn osrBlockNeedsRematerialization(source: *const machine.Function, block: cfg.BlockId) bool {
    const barriers = source.source.barriers orelse return false;
    for (barriers.loop_reuses) |reuse| if (reuse.header == block) return true;
    return false;
}

fn osrRematerializesRegister(source: *const machine.Function, block: cfg.BlockId, reg: machine.RegId) Error!bool {
    const barriers = source.source.barriers orelse return error.InvalidDeoptMetadata;
    for (barriers.loop_reuses) |reuse| {
        if (reuse.header != block) continue;
        if ((try osrRematOperands(source, reuse.resolve)).address == reg) return true;
    }
    return false;
}

fn machineSafepointCount(source: *const machine.Function) Error!u32 {
    const resolves = std.math.add(u32, source.stats.resolves, source.stats.loop_epoch_guards) catch return error.InvalidDeoptMetadata;
    return std.math.add(u32, resolves, source.stats.bounds_exception_sites) catch return error.InvalidDeoptMetadata;
}

fn osrLandingSiteId(source: *const machine.Function, deopt: ?DeoptOptions, entry_index: usize) Error!u32 {
    if (entry_index > std.math.maxInt(u32)) return error.InvalidDeoptMetadata;
    const first_landing = std.math.add(u32, try machineSafepointCount(source), try blockEntryDeoptCount(deopt)) catch
        return error.InvalidDeoptMetadata;
    return std.math.add(u32, first_landing, @intCast(entry_index)) catch return error.InvalidDeoptMetadata;
}

fn validateOsrLiveRegister(
    source: *const machine.Function,
    allocation: *const regalloc.Allocation,
    mapped_values: []const runtime_deopt.ValueSpec,
    osr_block: cfg.BlockId,
    reg: machine.RegId,
) Error!void {
    if (reg >= source.runtime_values.len) return error.InvalidDeoptMetadata;
    const value_id = switch (source.runtime_values[reg]) {
        .derived_ptr => {
            if (!try osrRematerializesRegister(source, osr_block, reg)) return error.InvalidDeoptMetadata;
            _ = try physOf(allocation, reg);
            return;
        },
        .dalvik => |value| value.value,
    };
    if (value_id >= source.source.source.values.len) return error.InvalidDeoptMetadata;
    const vreg = source.source.source.values[value_id].reg;
    const physical = try x64Reg(try physOf(allocation, reg));
    for (mapped_values) |value| {
        const width = value.kind.registerWidth();
        if (vreg < value.vreg or vreg >= value.vreg + width) continue;
        switch (value.source) {
            .native_register => |mapped| if (mapped == physical) return,
            else => {},
        }
    }
    return error.InvalidDeoptMetadata;
}

fn validateOsrMappedSources(values: []const runtime_deopt.ValueSpec) Error!void {
    for (values) |value| switch (value.source) {
        .native_register, .constant => {},
        .stack_slot, .xmm_register => return error.InvalidDeoptMetadata,
    };
}

fn appendUniqueRoot(
    roots: *std.ArrayList(runtime_stack_map.RootLocation),
    allocator: std.mem.Allocator,
    root: runtime_stack_map.RootLocation,
) Error!void {
    for (roots.items) |existing| if (existing.bits() == root.bits()) return;
    try roots.append(allocator, root);
}

/// Extends immutable machine safepoint metadata with one compiler-owned GC
/// site per OSR landing that may call the relocation helper. The landing image
/// contains only interpreter-exported GP values, so its roots come directly
/// from the already translated deoptimization record rather than from body
/// liveness (whose spill slots have not been initialized yet).
fn buildOsrAugmentedRootMaps(
    allocator: std.mem.Allocator,
    source: *const machine.Function,
    allocation: *const regalloc.Allocation,
    existing: *const runtime_stack_map.Table,
    deopt_table: *const runtime_deopt.Table,
    deopt: ?DeoptOptions,
    specs: []const OsrEntrySpec,
    stats: *Stats,
) Error!?runtime_stack_map.Table {
    var remat_count: usize = 0;
    for (specs) |spec| remat_count += @intFromBool(osrBlockNeedsRematerialization(source, spec.block));
    if (remat_count == 0) return null;

    var map_specs: std.ArrayList(runtime_stack_map.MapSpec) = .empty;
    defer map_specs.deinit(allocator);
    try map_specs.ensureTotalCapacity(allocator, existing.records.len + remat_count);
    for (existing.records) |record| {
        try map_specs.append(allocator, .{
            .pc_offset = record.pc_offset,
            .roots = existing.rootsFor(&record),
            .deopt_id = if (record.deopt_id == runtime_stack_map.no_deopt) null else record.deopt_id,
        });
    }

    var owned_roots: std.ArrayList([]runtime_stack_map.RootLocation) = .empty;
    defer {
        for (owned_roots.items) |roots| allocator.free(roots);
        owned_roots.deinit(allocator);
    }
    const barriers = source.source.barriers orelse return error.InvalidDeoptMetadata;
    for (specs, 0..) |spec, entry_index| {
        if (!osrBlockNeedsRematerialization(source, spec.block)) continue;
        const deopt_record = deopt_table.find(spec.point_id) catch return error.InvalidDeoptMetadata;
        const mapped_values = deopt_table.valuesFor(deopt_record);
        try validateOsrMappedSources(mapped_values);

        var roots: std.ArrayList(runtime_stack_map.RootLocation) = .empty;
        errdefer roots.deinit(allocator);
        for (mapped_values) |value| {
            if (value.kind != .reference) continue;
            switch (value.source) {
                .native_register => |physical| try appendUniqueRoot(
                    &roots,
                    allocator,
                    runtime_stack_map.RootLocation.nativeRegister(physical),
                ),
                .constant => {},
                .stack_slot, .xmm_register => return error.InvalidDeoptMetadata,
            }
        }
        for (barriers.loop_reuses) |reuse| {
            if (reuse.header != spec.block) continue;
            const operands = try osrRematOperands(source, reuse.resolve);
            const handle = try x64Reg(try physOf(allocation, operands.handle));
            var found = false;
            for (roots.items) |root| {
                if (root.kind == .native_register and root.payload == handle) {
                    found = true;
                    break;
                }
            }
            if (!found) return error.InvalidDeoptMetadata;
        }
        std.mem.sort(runtime_stack_map.RootLocation, roots.items, {}, rootLocationLess);
        const owned = try roots.toOwnedSlice(allocator);
        owned_roots.append(allocator, owned) catch |err| {
            allocator.free(owned);
            return err;
        };
        try map_specs.append(allocator, .{
            .pc_offset = try osrLandingSiteId(source, deopt, entry_index),
            .roots = owned,
        });
        stats.osr_landing_safepoints += 1;
        stats.root_map_sites += 1;
        stats.root_map_locations = std.math.add(u32, stats.root_map_locations, @intCast(owned.len)) catch
            return error.InvalidDeoptMetadata;
    }

    return try runtime_stack_map.Table.init(allocator, map_specs.items, .{
        .native_register_count = 16,
        .interpreter_register_count = 0,
        .max_frame_depth = 0,
        .max_shadow_roots = 0,
    });
}

const OsrCursor = struct {
    block: cfg.BlockId,
    instruction: u32,
};

fn osrRegisterReadBeforeDefinition(
    allocator: std.mem.Allocator,
    source: *const machine.Function,
    safepoint: SafepointPosition,
    reg: machine.RegId,
) Error!bool {
    const visited = try allocator.alloc(bool, source.blocks.len);
    defer allocator.free(visited);
    @memset(visited, false);
    var stack: std.ArrayList(OsrCursor) = .empty;
    defer stack.deinit(allocator);
    try stack.append(allocator, .{ .block = safepoint.block, .instruction = safepoint.instruction });

    while (stack.pop()) |cursor| {
        if (cursor.block >= source.blocks.len) return error.InvalidMachine;
        const block = source.blocks[cursor.block];
        if (cursor.instruction == 0) {
            if (visited[cursor.block]) continue;
            visited[cursor.block] = true;
        }
        if (cursor.instruction > block.insts.len) return error.InvalidMachine;
        var killed = false;
        for (block.insts[cursor.instruction..]) |inst| {
            for (inst.uses) |use| if (use == reg) return true;
            if (inst.address == reg or inst.state_handle == reg) return true;
            for (inst.defs) |definition| {
                if (definition == reg) {
                    killed = true;
                    break;
                }
            }
            if (killed) break;
        }
        if (killed) continue;

        for (source.successors[cursor.block]) |successor| {
            var edge_kill = false;
            for (source.edges) |edge| {
                if (edge.from != cursor.block or edge.to != successor) continue;
                for (edge.moves) |move| {
                    if (move.src == reg) return true;
                }
                for (edge.moves) |move| {
                    if (move.dst == reg) edge_kill = true;
                }
            }
            if (!edge_kill) try stack.append(allocator, .{ .block = successor, .instruction = 0 });
        }
    }
    return false;
}

pub fn osrRequiredRegistersAtSite(
    allocator: std.mem.Allocator,
    source: *const machine.Function,
    site_id: u32,
) Error![]machine.RegId {
    const safepoint = try findSafepointPosition(source, site_id);
    return osrRequiredRegistersAtPosition(allocator, source, safepoint);
}

pub fn osrRequiredRegistersAtBlockEntry(
    allocator: std.mem.Allocator,
    source: *const machine.Function,
    block: cfg.BlockId,
) Error![]machine.RegId {
    if (block >= source.blocks.len or source.blocks[block].insts.len == 0) return error.InvalidDeoptMetadata;
    return osrRequiredRegistersAtPosition(allocator, source, .{
        .position = 0,
        .site_id = 0,
        .pc = source.blocks[block].insts[0].pc orelse return error.InvalidDeoptMetadata,
        .block = block,
        .instruction = 0,
    });
}

fn osrRequiredRegistersAtPosition(
    allocator: std.mem.Allocator,
    source: *const machine.Function,
    safepoint: SafepointPosition,
) Error![]machine.RegId {
    var required: std.ArrayList(machine.RegId) = .empty;
    errdefer required.deinit(allocator);
    for (0..source.reg_types.len) |reg_index| {
        const reg: machine.RegId = @intCast(reg_index);
        if (try osrRegisterReadBeforeDefinition(allocator, source, safepoint, reg)) {
            try required.append(allocator, reg);
        }
    }
    return required.toOwnedSlice(allocator);
}

fn buildOsrEntries(
    allocator: std.mem.Allocator,
    source: *const machine.Function,
    allocation: *const regalloc.Allocation,
    buffer: *const code_buffer.Buffer,
    osr_labels: []const code_buffer.LabelId,
    deopt: ?DeoptOptions,
    table: ?*const runtime_deopt.Table,
    specs: []const OsrEntrySpec,
    stats: *Stats,
) Error![]runtime_jit.OsrEntry {
    if (specs.len == 0) return allocator.alloc(runtime_jit.OsrEntry, 0);
    const options = deopt orelse return error.InvalidDeoptMetadata;
    const deopt_table = table orelse return error.InvalidDeoptMetadata;
    const barriers = source.source.barriers orelse return error.InvalidDeoptMetadata;
    const loops = barriers.loops orelse return error.InvalidDeoptMetadata;
    const entries = try allocator.alloc(runtime_jit.OsrEntry, specs.len);
    errdefer allocator.free(entries);

    for (specs, 0..) |spec, entry_index| {
        if (entry_index != 0 and specs[entry_index - 1].point_id >= spec.point_id) return error.InvalidDeoptMetadata;
        if (spec.block >= source.blocks.len or entry_index >= osr_labels.len) return error.InvalidDeoptMetadata;
        var point_index: ?usize = null;
        for (options.points, 0..) |candidate, candidate_index| {
            if (candidate.id != spec.point_id) continue;
            if (point_index != null) return error.InvalidDeoptMetadata;
            point_index = candidate_index;
        }
        const selected_index = point_index orelse return error.InvalidDeoptMetadata;
        const deopt_point = options.points[selected_index];
        if (deopt_point.inline_frames.len != 0) return error.InvalidDeoptMetadata;
        const safepoint = try resolveDeoptPosition(source, options, selected_index);
        if (safepoint.block != spec.block or safepoint.pc != deopt_point.dex_pc) {
            return error.InvalidDeoptMetadata;
        }

        var natural_loop = false;
        for (loops.loops) |loop| {
            if (loop.header == spec.block) {
                natural_loop = true;
                break;
            }
        }
        if (!natural_loop) return error.InvalidDeoptMetadata;
        const record = deopt_table.find(spec.point_id) catch return error.InvalidDeoptMetadata;
        const mapped_values = deopt_table.valuesFor(record);
        try validateOsrMappedSources(mapped_values);
        for (0..source.reg_types.len) |reg_index| {
            const reg: machine.RegId = @intCast(reg_index);
            if (try osrRegisterReadBeforeDefinition(allocator, source, safepoint, reg)) {
                try validateOsrLiveRegister(source, allocation, mapped_values, spec.block, reg);
            }
        }
        const offset = try buffer.labelOffset(osr_labels[entry_index]);
        if (offset == 0 or !std.mem.isAligned(offset, 16)) return error.InvalidDeoptMetadata;
        entries[entry_index] = .{ .point_id = spec.point_id, .code_offset = offset };
    }
    stats.osr_entries = @intCast(entries.len);
    return entries;
}

fn collectOsrRequiredRegisters(
    allocator: std.mem.Allocator,
    source: *const machine.Function,
    deopt: ?DeoptOptions,
    specs: []const OsrEntrySpec,
) Error![]machine.RegId {
    if (specs.len == 0) return allocator.alloc(machine.RegId, 0);
    const options = deopt orelse return error.InvalidDeoptMetadata;
    const barriers = source.source.barriers orelse return error.InvalidDeoptMetadata;
    const loops = barriers.loops orelse return error.InvalidDeoptMetadata;
    var required: std.ArrayList(machine.RegId) = .empty;
    errdefer required.deinit(allocator);
    for (specs, 0..) |spec, entry_index| {
        if (entry_index != 0 and specs[entry_index - 1].point_id >= spec.point_id) {
            return error.InvalidDeoptMetadata;
        }
        if (spec.block >= source.blocks.len) return error.InvalidDeoptMetadata;
        var point_index: ?usize = null;
        for (options.points, 0..) |candidate, candidate_index| {
            if (candidate.id == spec.point_id) {
                if (point_index != null) return error.InvalidDeoptMetadata;
                point_index = candidate_index;
            }
        }
        const selected_index = point_index orelse return error.InvalidDeoptMetadata;
        const deopt_point = options.points[selected_index];
        if (deopt_point.inline_frames.len != 0) return error.InvalidDeoptMetadata;
        for (deopt_point.values) |value| switch (value.source) {
            .machine_register => |reg| {
                if (reg >= source.reg_types.len) return error.InvalidDeoptMetadata;
                try appendUniqueRegister(&required, allocator, reg);
            },
            .constant => {},
        };
        const safepoint = try resolveDeoptPosition(source, options, selected_index);
        if (safepoint.block != spec.block or safepoint.pc != deopt_point.dex_pc) {
            return error.InvalidDeoptMetadata;
        }
        var natural_loop = false;
        for (loops.loops) |loop| {
            if (loop.header == spec.block) {
                natural_loop = true;
                break;
            }
        }
        if (!natural_loop) return error.InvalidDeoptMetadata;
        for (0..source.reg_types.len) |reg_index| {
            const reg: machine.RegId = @intCast(reg_index);
            if (!try osrRegisterReadBeforeDefinition(allocator, source, safepoint, reg)) continue;
            var duplicate = false;
            for (required.items) |existing| {
                if (existing == reg) {
                    duplicate = true;
                    break;
                }
            }
            if (!duplicate) try required.append(allocator, reg);
        }
    }
    return required.toOwnedSlice(allocator);
}

fn appendUniqueRegister(list: *std.ArrayList(machine.RegId), allocator: std.mem.Allocator, reg: machine.RegId) Error!void {
    for (list.items) |existing| if (existing == reg) return;
    try list.append(allocator, reg);
}

fn collectMustRegisterOperands(
    allocator: std.mem.Allocator,
    source: *const machine.Function,
    additional: []const machine.RegId,
) Error![]machine.RegId {
    var required: std.ArrayList(machine.RegId) = .empty;
    errdefer required.deinit(allocator);
    for (additional) |reg| try appendUniqueRegister(&required, allocator, reg);
    for (source.blocks) |block| {
        for (block.insts) |inst| {
            const spill_capable = switch (inst.opcode) {
                .const_i32,
                .const_i64,
                .mov,
                .add_i32,
                .sub_i32,
                .mul_i32,
                .and_i32,
                .or_i32,
                .xor_i32,
                .f32_op,
                .f64_op,
                .check_null,
                .check_bounds,
                .array_load_ptr,
                .field_load_ptr,
                .satb_pre_write,
                .array_store_ptr,
                .field_store_ptr,
                .static_satb_pre_write,
                .static_store,
                .static_root_post_write,
                .card_mark,
                .jump,
                .branch,
                .ret,
                => true,
                else => false,
            };
            if (!spill_capable) {
                for (inst.defs) |reg| try appendUniqueRegister(&required, allocator, reg);
                for (inst.uses) |reg| try appendUniqueRegister(&required, allocator, reg);
            }
            if (inst.address) |reg| try appendUniqueRegister(&required, allocator, reg);
            if (inst.state_handle) |reg| try appendUniqueRegister(&required, allocator, reg);
        }
    }
    return required.toOwnedSlice(allocator);
}

fn emitOsrRematerializations(
    allocator: std.mem.Allocator,
    buffer: *code_buffer.Buffer,
    source: *const machine.Function,
    allocation: *const regalloc.Allocation,
    runtime: RuntimeAbi,
    block: cfg.BlockId,
    dex_pc: u32,
    landing_site_id: u32,
    restart: code_buffer.LabelId,
    cold: *std.ArrayList(ColdResolve),
    stats: *Stats,
) Error!void {
    const barriers = source.source.barriers orelse return error.InvalidDeoptMetadata;
    var emitted: usize = 0;
    for (barriers.loop_reuses, 0..) |reuse, reuse_index| {
        if (reuse.header != block) continue;
        for (barriers.loop_reuses[0..reuse_index]) |previous| {
            if (previous.header == block and previous.resolve == reuse.resolve) return error.InvalidDeoptMetadata;
        }
        const operands = try osrRematOperands(source, reuse.resolve);
        const before = cold.items.len;
        const resolve = barriers.resolves[reuse.resolve];
        var defs = [_]machine.RegId{operands.address};
        var uses = [_]machine.RegId{operands.handle};
        try emitResolve(allocator, buffer, allocation, .{
            .opcode = .resolve_handle,
            .pc = dex_pc,
            .defs = &defs,
            .uses = &uses,
            .state_handle = operands.handle,
            .reloc_token = resolve.token,
            // Every landing resolve uses the same immutable root map. A slow
            // edge restarts the complete sequence because its poll may have
            // invalidated addresses rematerialized by earlier fast paths.
            .resolve_id = landing_site_id,
        }, runtime, cold, stats);
        if (cold.items.len != before + 1) return error.InvalidMachine;
        cold.items[cold.items.len - 1].epoch_restart = restart;
        cold.items[cold.items.len - 1].epoch_slot_offset = 0;
        stats.osr_derived_rematerializations += 1;
        stats.osr_remat_restart_edges += 1;
        emitted += 1;
    }
    if (emitted == 0) return error.InvalidDeoptMetadata;
}

pub fn encodeWithOptions(allocator: std.mem.Allocator, source: *const machine.Function, options: Options) Error!Function {
    source.verify() catch return error.InvalidMachine;
    if (options.runtime) |runtime| try runtime.verify();
    if ((source.stats.resolves != 0 or source.stats.bounds_exception_sites != 0) and options.runtime == null) return error.MissingRuntimeAbi;
    if (source.stats.bounds_exception_sites != 0 and options.runtime.?.bounds_exception_helper == 0) return error.MissingExceptionHelper;
    if (options.deopt != null and (options.runtime == null or options.runtime.?.deopt_epoch_address == 0 or options.runtime.?.deopt_helper == 0)) {
        return error.InvalidRuntimeAbi;
    }
    try verifyFoldedNullChecks(source);

    // r10/r11 are private ABI scratch for barriers as well as resolution.
    // Static-only methods have no resolve op, but must obey the same contract.
    const osr_required = try collectOsrRequiredRegisters(allocator, source, options.deopt, options.osr_entries);
    defer allocator.free(osr_required);
    const must_register = try collectMustRegisterOperands(allocator, source, osr_required);
    defer allocator.free(must_register);
    var preserve_nonvolatile = false;
    var allocation = regalloc.allocate(allocator, source, .{
        .gp_registers = allocatableGpRegisters(),
        .xmm_registers = allocatableXmmRegisters(),
        .distinct_registers = osr_required,
        .must_registers = must_register,
    }) catch |err| switch (err) {
        error.BadAllocation => blk: {
            if (builtin.os.tag != .windows) return err;
            preserve_nonvolatile = true;
            break :blk try regalloc.allocate(allocator, source, .{
                .gp_registers = allGpRegisters(),
                .xmm_registers = allocatableXmmRegisters(),
                .distinct_registers = osr_required,
                .must_registers = must_register,
            });
        },
        else => return err,
    };
    errdefer allocation.deinit();
    var spill_plan = try regalloc.planSpills(allocator, &allocation);
    defer spill_plan.deinit();
    const frame_bytes = spill_plan.stats.stack_bytes;

    var buffer = code_buffer.Buffer.init(allocator);
    errdefer buffer.deinit();

    const labels = try allocator.alloc(code_buffer.LabelId, source.blocks.len);
    errdefer allocator.free(labels);
    for (labels) |*label| label.* = try buffer.newLabel();
    const edge_labels = try allocator.alloc(code_buffer.LabelId, source.edges.len);
    defer allocator.free(edge_labels);
    for (edge_labels) |*label| label.* = try buffer.newLabel();
    const osr_labels = try allocator.alloc(code_buffer.LabelId, options.osr_entries.len);
    defer allocator.free(osr_labels);
    for (osr_labels) |*label| label.* = try buffer.newLabel();
    const osr_body_labels = try allocator.alloc(code_buffer.LabelId, options.osr_entries.len);
    defer allocator.free(osr_body_labels);
    for (osr_body_labels) |*label| label.* = try buffer.newLabel();

    var stats: Stats = .{
        .blocks = @intCast(source.blocks.len),
        .frame_bytes = frame_bytes,
        .nonvolatile_frame_bytes = frameSavedBytes(preserve_nonvolatile),
    };
    var root_maps = try buildRootMaps(allocator, source, &allocation, &spill_plan, &stats, options.deopt);
    errdefer if (root_maps) |*maps| maps.deinit();
    var deopt_table = try buildDeoptTable(
        allocator,
        source,
        &allocation,
        &spill_plan,
        if (root_maps) |*maps| maps else null,
        options.deopt,
        &stats,
    );
    errdefer if (deopt_table) |*table| table.deinit();
    if (options.osr_entries.len != 0) {
        const existing_maps = if (root_maps) |*maps| maps else return error.MissingDeoptSafepoint;
        const existing_deopt = if (deopt_table) |*table| table else return error.InvalidDeoptMetadata;
        if (try buildOsrAugmentedRootMaps(
            allocator,
            source,
            &allocation,
            existing_maps,
            existing_deopt,
            options.deopt,
            options.osr_entries,
            &stats,
        )) |augmented| {
            existing_maps.deinit();
            root_maps = augmented;
            try existing_deopt.validateStackMaps(&root_maps.?, false);
            try existing_deopt.validateAllLinked(&root_maps.?);
        }
    }
    try emitFrameEnter(&buffer, frame_bytes, preserve_nonvolatile);
    if (frame_bytes != 0 or preserve_nonvolatile) stats.native_insts += 1;
    try emitParamMoves(&buffer, &allocation, &spill_plan, source, frame_bytes, preserve_nonvolatile, &stats);

    var cold: std.ArrayList(ColdResolve) = .empty;
    defer cold.deinit(allocator);
    var cold_bounds: std.ArrayList(ColdBoundsException) = .empty;
    defer cold_bounds.deinit(allocator);
    var cold_deopts: std.ArrayList(ColdDeopt) = .empty;
    defer cold_deopts.deinit(allocator);

    for (source.blocks) |block| {
        try buffer.alignTo(16, 0x90);
        try buffer.bindLabel(labels[block.id]);
        if (options.deopt) |deopt| {
            for (deopt.points, 0..) |point, point_index| {
                if (point.block_entry != block.id) continue;
                const site_id = try deoptPointSiteId(source, deopt, point_index);
                for (options.osr_entries, 0..) |osr, osr_index| {
                    if (osr.block != block.id or osr.point_id != point.id) continue;
                    try buffer.alignTo(16, 0x90);
                    const out_of_line = frame_bytes != 0 or preserve_nonvolatile or osrBlockNeedsRematerialization(source, osr.block);
                    try buffer.bindLabel(if (out_of_line) osr_body_labels[osr_index] else osr_labels[osr_index]);
                }
                try emitDeoptGuard(allocator, &buffer, options.runtime.?, site_id, &cold_deopts, &stats);
            }
        }
        for (block.insts) |inst| {
            if (instructionSafepointId(inst)) |site_id| {
                if (options.deopt) |deopt| {
                    for (options.osr_entries, 0..) |osr, osr_index| {
                        if (osr.block != block.id) continue;
                        for (deopt.points) |point| {
                            if (point.id != osr.point_id or (point.safepoint_id orelse continue) != site_id) continue;
                            try buffer.alignTo(16, 0x90);
                            const out_of_line = frame_bytes != 0 or preserve_nonvolatile or osrBlockNeedsRematerialization(source, osr.block);
                            try buffer.bindLabel(if (out_of_line) osr_body_labels[osr_index] else osr_labels[osr_index]);
                        }
                    }
                }
                if (try deoptIdForSite(source, options.deopt, site_id) != null) {
                    try emitDeoptGuard(allocator, &buffer, options.runtime.?, site_id, &cold_deopts, &stats);
                }
            }
            try encodeInst(allocator, &buffer, labels, edge_labels, block.id, &allocation, &spill_plan, frame_bytes, preserve_nonvolatile, source, inst, options.runtime, &cold, &cold_bounds, &stats);
        }
        if (!hasExplicitTransfer(block)) {
            const successors = source.successors[block.id];
            if (successors.len != 1) return error.InvalidMachine;
            const target = try edgeTargetLabel(source, labels, edge_labels, block.id, successors[0]);
            const natural_fallthrough = successors[0] == block.id + 1 and target == labels[successors[0]];
            if (!natural_fallthrough) {
                try emitJump(&buffer, target);
                stats.jumps += 1;
                stats.native_insts += 1;
            }
        }
    }
    for (source.edges, 0..) |edge, index| {
        if (edge.from >= source.blocks.len or edge.to >= labels.len) return error.InvalidMachine;
        try buffer.bindLabel(edge_labels[index]);
        try emitEdgeCopies(allocator, &buffer, &allocation, &spill_plan, source, edge, &stats);
        try emitJump(&buffer, labels[edge.to]);
        stats.edge_copy_sites += 1;
        stats.edge_copy_moves += @intCast(edge.moves.len);
        stats.jumps += 1;
        stats.native_insts += 1;
    }
    for (osr_labels, 0..) |landing, index| {
        const spec = options.osr_entries[index];
        const rematerializes = osrBlockNeedsRematerialization(source, spec.block);
        if (frame_bytes == 0 and !preserve_nonvolatile and !rematerializes) continue;
        try buffer.alignTo(16, 0x90);
        try buffer.bindLabel(landing);
        try emitFrameEnter(&buffer, frame_bytes, preserve_nonvolatile);
        if (frame_bytes != 0 or preserve_nonvolatile) {
            stats.osr_frame_landings += 1;
            stats.native_insts += 1;
        }
        if (rematerializes) {
            // A private, non-root landing scratch word remembers the epoch
            // across preserve-all helper calls. It is removed before entering
            // the body, so normal spill offsets and the common epilogue remain
            // unchanged.
            try emitAdjustStack(&buffer, true, 16);
            const restart = try buffer.newLabel();
            try buffer.bindLabel(restart);
            try emitMovMemoryFromRaw(&buffer, 4, 0, acknowledged_epoch, true);
            stats.native_insts += 2;
            const deopt = options.deopt orelse return error.InvalidDeoptMetadata;
            var dex_pc: ?u32 = null;
            for (deopt.points) |point| {
                if (point.id != spec.point_id) continue;
                if (dex_pc != null) return error.InvalidDeoptMetadata;
                dex_pc = point.dex_pc;
            }
            try emitOsrRematerializations(
                allocator,
                &buffer,
                source,
                &allocation,
                options.runtime orelse return error.MissingRuntimeAbi,
                spec.block,
                dex_pc orelse return error.InvalidDeoptMetadata,
                try osrLandingSiteId(source, options.deopt, index),
                restart,
                &cold,
                &stats,
            );
            try emitAdjustStack(&buffer, false, 16);
            stats.native_insts += 1;
        }
        try emitJump(&buffer, osr_body_labels[index]);
        stats.jumps += 1;
        stats.native_insts += 1;
    }
    if (options.runtime) |runtime| try emitColdResolves(&buffer, cold.items, runtime, &stats);
    if (options.runtime) |runtime| try emitColdBoundsExceptions(&buffer, cold_bounds.items, runtime, frame_bytes, preserve_nonvolatile, &stats);
    if (options.runtime) |runtime| try emitColdDeopts(&buffer, cold_deopts.items, runtime, frame_bytes, preserve_nonvolatile, &stats);
    try buffer.verify();
    const osr_entries = try buildOsrEntries(
        allocator,
        source,
        &allocation,
        &buffer,
        osr_labels,
        options.deopt,
        if (deopt_table) |*table| table else null,
        options.osr_entries,
        &stats,
    );
    errdefer allocator.free(osr_entries);
    stats.bytes = buffer.len();

    return .{
        .allocator = allocator,
        .source = source,
        .allocation = allocation,
        .buffer = buffer,
        .block_labels = labels,
        .root_maps = root_maps,
        .deopt_table = deopt_table,
        .osr_entries = osr_entries,
        .stats = stats,
    };
}

pub fn encode(allocator: std.mem.Allocator, source: *const machine.Function) Error!Function {
    return encodeWithOptions(allocator, source, .{});
}

fn optimizedMachine(allocator: std.mem.Allocator, insts: []const Instruction) !*optimizer.OptimizedFunction {
    return try optimizer.optimize(allocator, insts, &.{}, .{});
}

fn encodedTestInstructions(allocator: std.mem.Allocator, insts: []const Instruction) ![]u8 {
    var optimized = try optimizedMachine(allocator, insts);
    defer optimized.deinit();
    var native = try encode(allocator, &optimized.machine);
    defer native.deinit();
    return try native.finalize();
}

fn addOwnedTestBytes(cache: *jit_memory.Cache, bytes: []u8) !*jit_memory.Allocation {
    defer std.testing.allocator.free(bytes);
    return try cache.addBytes(bytes);
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
    try expectBinaryI32(&[_]Instruction{
        .{ .add_int_lit8 = .{ .dest = 1, .src = 0, .lit = -7 } },
        .{ .return_ = .{ .src = 1 } },
    }, 49, 0, 42);
}

fn edgeEncodingFailureProbe(allocator: std.mem.Allocator, source: *const machine.Function) !void {
    var native = try encode(allocator, source);
    defer native.deinit();
    const bytes = try native.finalize();
    defer allocator.free(bytes);
}

test "x64_register_encoder executes branch-specific phi edge copies" {
    if (builtin.cpu.arch != .x86_64) return error.SkipZigTest;
    const insts = [_]Instruction{
        .{ .if_eqz = .{ .src = 0, .offset = 3 } },
        .{ .const_ = .{ .dest = 1, .value = 10 } },
        .{ .goto_ = .{ .offset = 2 } },
        .{ .const_ = .{ .dest = 1, .value = 20 } },
        .{ .return_ = .{ .src = 1 } },
    };
    var optimized = try optimizedMachine(std.testing.allocator, &insts);
    defer optimized.deinit();
    try std.testing.expectEqual(@as(usize, 2), optimized.machine.edges.len);
    try std.testing.expectEqual(@as(u32, 2), optimized.machine.stats.edge_moves);

    var native = try encode(std.testing.allocator, &optimized.machine);
    defer native.deinit();
    try std.testing.expectEqual(@as(u32, 2), native.stats.edge_copy_sites);
    try std.testing.expectEqual(@as(u32, 2), native.stats.edge_copy_moves);
    const bytes = try native.finalize();
    defer std.testing.allocator.free(bytes);

    var cache = jit_memory.Cache.init(std.testing.allocator);
    defer cache.deinit();
    const allocation = try cache.addBytes(bytes);
    const Fn = fn (i32, i32) callconv(.c) i32;
    try std.testing.expectEqual(@as(i32, 20), allocation.typedEntry(Fn)(0, -1));
    try std.testing.expectEqual(@as(i32, 10), allocation.typedEntry(Fn)(1, -1));
    try std.testing.checkAllAllocationFailures(std.testing.allocator, edgeEncodingFailureProbe, .{&optimized.machine});
}

test "x64 edge parallel copy breaks physical register cycles" {
    if (builtin.cpu.arch != .x86_64) return error.SkipZigTest;
    var buffer = code_buffer.Buffer.init(std.testing.allocator);
    defer buffer.deinit();
    try emitMovRawImm64(&buffer, 0, 11);
    try emitMovRawImm64(&buffer, 1, 22);
    var copies = [_]EdgeCopy{
        .{ .dst = .{ .phys = .rax }, .src = .{ .phys = .rcx }, .dst_reg = 0, .src_reg = 1, .wide = true, .class = .gp },
        .{ .dst = .{ .phys = .rcx }, .src = .{ .phys = .rax }, .dst_reg = 1, .src_reg = 0, .wide = true, .class = .gp },
    };
    var stats: Stats = .{};
    try emitParallelEdgeCopies(&buffer, null, &copies, &stats);
    try emitRet(&buffer);
    try std.testing.expectEqual(@as(u32, 1), stats.edge_copy_cycles);
    try std.testing.expectEqual(@as(u32, 3), stats.register_moves);

    const bytes = try buffer.finalize();
    defer std.testing.allocator.free(bytes);
    var cache = jit_memory.Cache.init(std.testing.allocator);
    defer cache.deinit();
    const allocation = try cache.addBytes(bytes);
    const Fn = fn () callconv(.c) u64;
    try std.testing.expectEqual(@as(u64, 22), allocation.typedEntry(Fn)());
}

test "x64 edge parallel copy breaks XMM register cycles" {
    if (builtin.cpu.arch != .x86_64) return error.SkipZigTest;
    const first_bits: u32 = @bitCast(@as(f32, 11.25));
    const second_bits: u32 = @bitCast(@as(f32, -7.5));
    var buffer = code_buffer.Buffer.init(std.testing.allocator);
    defer buffer.deinit();
    try emitMovRegImm32(&buffer, .rax, @bitCast(first_bits));
    try emitGpToXmm(&buffer, .xmm0, .rax, false);
    try emitMovRegImm32(&buffer, .rcx, @bitCast(second_bits));
    try emitGpToXmm(&buffer, .xmm1, .rcx, false);
    var copies = [_]EdgeCopy{
        .{ .dst = .{ .phys = .xmm0 }, .src = .{ .phys = .xmm1 }, .dst_reg = 0, .src_reg = 1, .wide = false, .class = .xmm },
        .{ .dst = .{ .phys = .xmm1 }, .src = .{ .phys = .xmm0 }, .dst_reg = 1, .src_reg = 0, .wide = false, .class = .xmm },
    };
    var stats: Stats = .{};
    try emitParallelEdgeCopies(&buffer, null, &copies, &stats);
    try emitXmmToGp(&buffer, .rax, .xmm0, false);
    try emitRet(&buffer);
    try std.testing.expectEqual(@as(u32, 1), stats.edge_copy_cycles);
    try std.testing.expectEqual(@as(u32, 3), stats.register_moves);
    try std.testing.expectEqual(@as(u32, 3), stats.xmm_insts);

    const bytes = try buffer.finalize();
    defer std.testing.allocator.free(bytes);
    var cache = jit_memory.Cache.init(std.testing.allocator);
    defer cache.deinit();
    const allocation = try cache.addBytes(bytes);
    const Fn = fn () callconv(.c) u32;
    try std.testing.expectEqual(second_bits, allocation.typedEntry(Fn)());
}

test "x64_register_encoder executes spill-heavy scalar XMM arithmetic" {
    if (builtin.cpu.arch != .x86_64) return error.SkipZigTest;
    const insts = [_]Instruction{
        .{ .add_float = .{ .dest = 7, .src1 = 0, .src2 = 1 } },
        .{ .add_float = .{ .dest = 8, .src1 = 2, .src2 = 3 } },
        .{ .add_float = .{ .dest = 9, .src1 = 4, .src2 = 5 } },
        .{ .add_float = .{ .dest = 10, .src1 = 7, .src2 = 6 } },
        .{ .add_float = .{ .dest = 11, .src1 = 10, .src2 = 8 } },
        .{ .add_float = .{ .dest = 12, .src1 = 11, .src2 = 9 } },
        .{ .return_ = .{ .src = 12 } },
    };
    var optimized = try optimizedMachine(std.testing.allocator, &insts);
    defer optimized.deinit();
    for (optimized.machine.blocks) |block| for (block.insts) |inst| {
        if (inst.opcode == .f32_op) try std.testing.expectEqual(machine.FloatOperation.add, inst.float_op.?);
    };

    var native = try encode(std.testing.allocator, &optimized.machine);
    defer native.deinit();
    try std.testing.expect(native.allocation.stats.spills > 0);
    try std.testing.expect(native.stats.frame_bytes > 0);
    try std.testing.expect(std.mem.isAligned(native.stats.frame_bytes, 16));
    try std.testing.expect(native.stats.xmm_insts > 0);
    try std.testing.expect(native.stats.xmm_spill_loads > 0);
    try std.testing.expect(native.stats.xmm_spill_stores > 0);
    const bytes = try native.finalize();
    defer std.testing.allocator.free(bytes);

    var cache = jit_memory.Cache.init(std.testing.allocator);
    defer cache.deinit();
    const allocation = try cache.addBytes(bytes);
    const Fn = fn (u32, u32, u32, u32, u32, u32, u32) callconv(.c) u32;
    const result_bits = allocation.typedEntry(Fn)(
        @bitCast(@as(f32, 1)),
        @bitCast(@as(f32, 2)),
        @bitCast(@as(f32, 3)),
        @bitCast(@as(f32, 4)),
        @bitCast(@as(f32, 5)),
        @bitCast(@as(f32, 6)),
        @bitCast(@as(f32, 7)),
    );
    try std.testing.expectEqual(@as(f32, 28), @as(f32, @bitCast(result_bits)));
    try std.testing.checkAllAllocationFailures(std.testing.allocator, edgeEncodingFailureProbe, .{&optimized.machine});
}

test "x64_register_encoder preserves scalar XMM rhs across two-address aliases" {
    if (builtin.cpu.arch != .x86_64) return error.SkipZigTest;
    const insts = [_]Instruction{
        .{ .sub_float = .{ .dest = 2, .src1 = 0, .src2 = 1 } },
        .{ .mul_float = .{ .dest = 3, .src1 = 2, .src2 = 1 } },
        .{ .div_float = .{ .dest = 4, .src1 = 3, .src2 = 0 } },
        .{ .return_ = .{ .src = 4 } },
    };
    var optimized = try optimizedMachine(std.testing.allocator, &insts);
    defer optimized.deinit();
    const expected_operations = [_]machine.FloatOperation{ .sub, .mul, .div };
    var operation_index: usize = 0;
    for (optimized.machine.blocks) |block| for (block.insts) |inst| {
        if (inst.opcode != .f32_op) continue;
        try std.testing.expect(operation_index < expected_operations.len);
        try std.testing.expectEqual(expected_operations[operation_index], inst.float_op.?);
        operation_index += 1;
    };
    try std.testing.expectEqual(expected_operations.len, operation_index);

    var native = try encode(std.testing.allocator, &optimized.machine);
    defer native.deinit();
    const bytes = try native.finalize();
    defer std.testing.allocator.free(bytes);
    var cache = jit_memory.Cache.init(std.testing.allocator);
    defer cache.deinit();
    const allocation = try cache.addBytes(bytes);
    const Fn = fn (u32, u32) callconv(.c) u32;
    const result_bits = allocation.typedEntry(Fn)(@bitCast(@as(f32, 10)), @bitCast(@as(f32, 3)));
    const result: f32 = @bitCast(result_bits);
    try std.testing.expectApproxEqAbs(@as(f32, 2.1), result, 0.000001);
}

test "x64_register_encoder executes bit-exact scalar XMM negation" {
    if (builtin.cpu.arch != .x86_64) return error.SkipZigTest;
    const float_insts = [_]Instruction{
        .{ .neg_float = .{ .dest = 1, .src = 0 } },
        .{ .return_ = .{ .src = 1 } },
    };
    var float_optimized = try optimizedMachine(std.testing.allocator, &float_insts);
    defer float_optimized.deinit();
    try std.testing.expectEqual(machine.FloatOperation.neg, float_optimized.machine.blocks[0].insts[0].float_op.?);
    var float_native = try encode(std.testing.allocator, &float_optimized.machine);
    defer float_native.deinit();
    try std.testing.expectEqual(@as(u32, 1), float_native.stats.xmm_negations);
    const float_bytes = try float_native.finalize();
    defer std.testing.allocator.free(float_bytes);
    var float_cache = jit_memory.Cache.init(std.testing.allocator);
    defer float_cache.deinit();
    const float_allocation = try float_cache.addBytes(float_bytes);
    const FloatFn = fn (u32) callconv(.c) u32;
    const negate_float = float_allocation.typedEntry(FloatFn);
    try std.testing.expectEqual(@as(u32, 0x80000000), negate_float(0x00000000));
    try std.testing.expectEqual(@as(u32, 0x00000000), negate_float(0x80000000));
    try std.testing.expectEqual(@as(u32, @bitCast(@as(f32, -1.5))), negate_float(@bitCast(@as(f32, 1.5))));
    try std.testing.expectEqual(@as(u32, 0xffc12345), negate_float(0x7fc12345));

    const double_insts = [_]Instruction{
        .{ .neg_double = .{ .dest = 2, .src = 0 } },
        .{ .return_wide = .{ .src = 2 } },
    };
    var double_optimized = try optimizedMachine(std.testing.allocator, &double_insts);
    defer double_optimized.deinit();
    var double_native = try encode(std.testing.allocator, &double_optimized.machine);
    defer double_native.deinit();
    try std.testing.expectEqual(@as(u32, 1), double_native.stats.xmm_negations);
    const double_bytes = try double_native.finalize();
    defer std.testing.allocator.free(double_bytes);
    var double_cache = jit_memory.Cache.init(std.testing.allocator);
    defer double_cache.deinit();
    const double_allocation = try double_cache.addBytes(double_bytes);
    const DoubleFn = fn (u64, u64) callconv(.c) u64;
    const negate_double = double_allocation.typedEntry(DoubleFn);
    try std.testing.expectEqual(@as(u64, 0x8000000000000000), negate_double(0, 0));
    try std.testing.expectEqual(@as(u64, 0), negate_double(0x8000000000000000, 0));
    try std.testing.expectEqual(@as(u64, 0xfff8123456789abc), negate_double(0x7ff8123456789abc, 0));
}

const F32CompareCase = struct {
    lhs: f32,
    rhs: f32,
    expected: i32,
};

fn expectF32Compare(insts: []const Instruction, operation: machine.FloatOperation, cases: []const F32CompareCase) !void {
    var optimized = try optimizedMachine(std.testing.allocator, insts);
    defer optimized.deinit();
    try std.testing.expectEqual(operation, optimized.machine.blocks[0].insts[0].float_op.?);
    var native = try encode(std.testing.allocator, &optimized.machine);
    defer native.deinit();
    try std.testing.expectEqual(@as(u32, 1), native.stats.xmm_comparisons);
    const bytes = try native.finalize();
    defer std.testing.allocator.free(bytes);
    var cache = jit_memory.Cache.init(std.testing.allocator);
    defer cache.deinit();
    const allocation = try cache.addBytes(bytes);
    const Fn = fn (u32, u32) callconv(.c) i32;
    const compare = allocation.typedEntry(Fn);
    for (cases) |case| try std.testing.expectEqual(case.expected, compare(@bitCast(case.lhs), @bitCast(case.rhs)));
}

test "x64_register_encoder executes Dalvik float compare NaN policy" {
    if (builtin.cpu.arch != .x86_64) return error.SkipZigTest;
    const nan = std.math.nan(f32);
    const common_cases = [_]F32CompareCase{
        .{ .lhs = -3, .rhs = 2, .expected = -1 },
        .{ .lhs = 2, .rhs = 2, .expected = 0 },
        .{ .lhs = 9, .rhs = 2, .expected = 1 },
        .{ .lhs = -0.0, .rhs = 0.0, .expected = 0 },
        .{ .lhs = -std.math.inf(f32), .rhs = std.math.inf(f32), .expected = -1 },
    };
    try expectF32Compare(&.{
        .{ .cmpl_float = .{ .dest = 2, .src1 = 0, .src2 = 1 } },
        .{ .return_ = .{ .src = 2 } },
    }, .compare_l, &common_cases);
    try expectF32Compare(&.{
        .{ .cmpg_float = .{ .dest = 2, .src1 = 0, .src2 = 1 } },
        .{ .return_ = .{ .src = 2 } },
    }, .compare_g, &common_cases);
    try expectF32Compare(&.{
        .{ .cmpl_float = .{ .dest = 2, .src1 = 0, .src2 = 1 } },
        .{ .return_ = .{ .src = 2 } },
    }, .compare_l, &.{
        .{ .lhs = nan, .rhs = 1, .expected = -1 },
        .{ .lhs = 1, .rhs = nan, .expected = -1 },
    });
    try expectF32Compare(&.{
        .{ .cmpg_float = .{ .dest = 2, .src1 = 0, .src2 = 1 } },
        .{ .return_ = .{ .src = 2 } },
    }, .compare_g, &.{
        .{ .lhs = nan, .rhs = 1, .expected = 1 },
        .{ .lhs = 1, .rhs = nan, .expected = 1 },
    });
}

test "x64_register_encoder executes Dalvik double compare NaN policy" {
    if (builtin.cpu.arch != .x86_64) return error.SkipZigTest;
    const insts = [_]Instruction{
        .{ .cmpg_double = .{ .dest = 4, .src1 = 0, .src2 = 2 } },
        .{ .return_ = .{ .src = 4 } },
    };
    var optimized = try optimizedMachine(std.testing.allocator, &insts);
    defer optimized.deinit();
    try std.testing.expectEqual(machine.FloatOperation.compare_g, optimized.machine.blocks[0].insts[0].float_op.?);
    var native = try encode(std.testing.allocator, &optimized.machine);
    defer native.deinit();
    const bytes = try native.finalize();
    defer std.testing.allocator.free(bytes);
    var cache = jit_memory.Cache.init(std.testing.allocator);
    defer cache.deinit();
    const allocation = try cache.addBytes(bytes);
    const Fn = fn (u64, u64, u64, u64) callconv(.c) i32;
    const compare = allocation.typedEntry(Fn);
    try std.testing.expectEqual(@as(i32, -1), compare(@bitCast(@as(f64, -4.0)), 0, @bitCast(@as(f64, 7.0)), 0));
    try std.testing.expectEqual(@as(i32, 0), compare(@bitCast(@as(f64, -0.0)), 0, @bitCast(@as(f64, 0.0)), 0));
    try std.testing.expectEqual(@as(i32, 1), compare(@bitCast(std.math.nan(f64)), 0, @bitCast(@as(f64, 1.0)), 0));
}

test "x64_register_encoder compares through scalar XMM pressure and spills" {
    if (builtin.cpu.arch != .x86_64) return error.SkipZigTest;
    const insts = [_]Instruction{
        .{ .add_float = .{ .dest = 7, .src1 = 0, .src2 = 1 } },
        .{ .add_float = .{ .dest = 8, .src1 = 2, .src2 = 3 } },
        .{ .add_float = .{ .dest = 9, .src1 = 4, .src2 = 5 } },
        .{ .add_float = .{ .dest = 10, .src1 = 7, .src2 = 6 } },
        .{ .cmpl_float = .{ .dest = 11, .src1 = 10, .src2 = 8 } },
        .{ .return_ = .{ .src = 11 } },
    };
    var optimized = try optimizedMachine(std.testing.allocator, &insts);
    defer optimized.deinit();
    var native = try encode(std.testing.allocator, &optimized.machine);
    defer native.deinit();
    try std.testing.expect(native.allocation.stats.spills > 0);
    try std.testing.expect(native.stats.xmm_spill_loads > 0);
    try std.testing.expect(native.stats.xmm_spill_stores > 0);
    const bytes = try native.finalize();
    defer std.testing.allocator.free(bytes);
    var cache = jit_memory.Cache.init(std.testing.allocator);
    defer cache.deinit();
    const allocation = try cache.addBytes(bytes);
    const Fn = fn (u32, u32, u32, u32, u32, u32, u32) callconv(.c) i32;
    const result = allocation.typedEntry(Fn)(
        @bitCast(@as(f32, 1)),
        @bitCast(@as(f32, 2)),
        @bitCast(@as(f32, 3)),
        @bitCast(@as(f32, 4)),
        @bitCast(@as(f32, 5)),
        @bitCast(@as(f32, 6)),
        @bitCast(@as(f32, 7)),
    );
    try std.testing.expectEqual(@as(i32, 1), result);
    try std.testing.checkAllAllocationFailures(std.testing.allocator, edgeEncodingFailureProbe, .{&optimized.machine});
}

test "x64_register_encoder converts signed integers to float and double" {
    if (builtin.cpu.arch != .x86_64) return error.SkipZigTest;
    var cache = jit_memory.Cache.init(std.testing.allocator);
    defer cache.deinit();

    const int_float_bytes = try encodedTestInstructions(std.testing.allocator, &.{
        .{ .int_to_float = .{ .dest = 1, .src = 0 } },
        .{ .return_ = .{ .src = 1 } },
    });
    const IntFloatFn = fn (i32) callconv(.c) u32;
    const int_to_float = (try addOwnedTestBytes(&cache, int_float_bytes)).typedEntry(IntFloatFn);
    try std.testing.expectEqual(@as(u32, @bitCast(@as(f32, @floatFromInt(std.math.minInt(i32))))), int_to_float(std.math.minInt(i32)));
    try std.testing.expectEqual(@as(u32, @bitCast(@as(f32, @floatFromInt(std.math.maxInt(i32))))), int_to_float(std.math.maxInt(i32)));
    try std.testing.expectEqual(@as(u32, @bitCast(@as(f32, @floatFromInt(@as(i32, 123456789))))), int_to_float(123456789));

    const int_double_bytes = try encodedTestInstructions(std.testing.allocator, &.{
        .{ .int_to_double = .{ .dest = 1, .src = 0 } },
        .{ .return_wide = .{ .src = 1 } },
    });
    const IntDoubleFn = fn (i32) callconv(.c) u64;
    const int_to_double = (try addOwnedTestBytes(&cache, int_double_bytes)).typedEntry(IntDoubleFn);
    try std.testing.expectEqual(@as(u64, @bitCast(@as(f64, @floatFromInt(std.math.minInt(i32))))), int_to_double(std.math.minInt(i32)));
    try std.testing.expectEqual(@as(u64, @bitCast(@as(f64, @floatFromInt(std.math.maxInt(i32))))), int_to_double(std.math.maxInt(i32)));

    const long_float_bytes = try encodedTestInstructions(std.testing.allocator, &.{
        .{ .long_to_float = .{ .dest = 2, .src = 0 } },
        .{ .return_ = .{ .src = 2 } },
    });
    const LongFloatFn = fn (i64, i64) callconv(.c) u32;
    const long_to_float = (try addOwnedTestBytes(&cache, long_float_bytes)).typedEntry(LongFloatFn);
    try std.testing.expectEqual(@as(u32, @bitCast(@as(f32, @floatFromInt(std.math.minInt(i64))))), long_to_float(std.math.minInt(i64), 0));
    try std.testing.expectEqual(@as(u32, @bitCast(@as(f32, @floatFromInt(std.math.maxInt(i64))))), long_to_float(std.math.maxInt(i64), 0));

    const long_double_bytes = try encodedTestInstructions(std.testing.allocator, &.{
        .{ .long_to_double = .{ .dest = 2, .src = 0 } },
        .{ .return_wide = .{ .src = 2 } },
    });
    const LongDoubleFn = fn (i64, i64) callconv(.c) u64;
    const long_to_double = (try addOwnedTestBytes(&cache, long_double_bytes)).typedEntry(LongDoubleFn);
    try std.testing.expectEqual(@as(u64, @bitCast(@as(f64, @floatFromInt(std.math.maxInt(i64))))), long_to_double(std.math.maxInt(i64), 0));
    try std.testing.expectEqual(@as(u64, @bitCast(@as(f64, @floatFromInt(std.math.minInt(i64))))), long_to_double(std.math.minInt(i64), 0));
}

test "x64_register_encoder changes scalar XMM precision" {
    if (builtin.cpu.arch != .x86_64) return error.SkipZigTest;
    var cache = jit_memory.Cache.init(std.testing.allocator);
    defer cache.deinit();

    const widen_bytes = try encodedTestInstructions(std.testing.allocator, &.{
        .{ .float_to_double = .{ .dest = 1, .src = 0 } },
        .{ .return_wide = .{ .src = 1 } },
    });
    const WidenFn = fn (u32) callconv(.c) u64;
    const widen = (try addOwnedTestBytes(&cache, widen_bytes)).typedEntry(WidenFn);
    try std.testing.expectEqual(@as(u64, @bitCast(@as(f64, @floatCast(@as(f32, 1.25))))), widen(@bitCast(@as(f32, 1.25))));
    try std.testing.expectEqual(@as(u64, @bitCast(@as(f64, -0.0))), widen(@bitCast(@as(f32, -0.0))));
    try std.testing.expectEqual(@as(u64, @bitCast(std.math.inf(f64))), widen(@bitCast(std.math.inf(f32))));

    const narrow_bytes = try encodedTestInstructions(std.testing.allocator, &.{
        .{ .double_to_float = .{ .dest = 2, .src = 0 } },
        .{ .return_ = .{ .src = 2 } },
    });
    const NarrowFn = fn (u64, u64) callconv(.c) u32;
    const narrow = (try addOwnedTestBytes(&cache, narrow_bytes)).typedEntry(NarrowFn);
    try std.testing.expectEqual(@as(u32, @bitCast(@as(f32, @floatCast(@as(f64, 1.25))))), narrow(@bitCast(@as(f64, 1.25)), 0));
    try std.testing.expectEqual(@as(u32, @bitCast(@as(f32, -0.0))), narrow(@bitCast(@as(f64, -0.0)), 0));
    try std.testing.expectEqual(@as(u32, @bitCast(std.math.inf(f32))), narrow(@bitCast(std.math.inf(f64)), 0));
}

test "x64_register_encoder saturates float and double integer conversions" {
    if (builtin.cpu.arch != .x86_64) return error.SkipZigTest;
    var cache = jit_memory.Cache.init(std.testing.allocator);
    defer cache.deinit();

    const float_int_bytes = try encodedTestInstructions(std.testing.allocator, &.{
        .{ .float_to_int = .{ .dest = 1, .src = 0 } },
        .{ .return_ = .{ .src = 1 } },
    });
    const FloatIntFn = fn (u32) callconv(.c) i32;
    const float_to_int = (try addOwnedTestBytes(&cache, float_int_bytes)).typedEntry(FloatIntFn);
    try std.testing.expectEqual(@as(i32, 0), float_to_int(@bitCast(std.math.nan(f32))));
    try std.testing.expectEqual(std.math.maxInt(i32), float_to_int(@bitCast(std.math.inf(f32))));
    try std.testing.expectEqual(std.math.minInt(i32), float_to_int(@bitCast(-std.math.inf(f32))));
    try std.testing.expectEqual(std.math.maxInt(i32), float_to_int(@bitCast(@as(f32, @floatFromInt(std.math.maxInt(i32))))));
    try std.testing.expectEqual(std.math.minInt(i32), float_to_int(@bitCast(@as(f32, @floatFromInt(std.math.minInt(i32))))));
    try std.testing.expectEqual(@as(i32, 42), float_to_int(@bitCast(@as(f32, 42.875))));
    try std.testing.expectEqual(@as(i32, -42), float_to_int(@bitCast(@as(f32, -42.875))));

    const float_long_bytes = try encodedTestInstructions(std.testing.allocator, &.{
        .{ .float_to_long = .{ .dest = 1, .src = 0 } },
        .{ .return_wide = .{ .src = 1 } },
    });
    const FloatLongFn = fn (u32) callconv(.c) i64;
    const float_to_long = (try addOwnedTestBytes(&cache, float_long_bytes)).typedEntry(FloatLongFn);
    try std.testing.expectEqual(@as(i64, 0), float_to_long(@bitCast(std.math.nan(f32))));
    try std.testing.expectEqual(std.math.maxInt(i64), float_to_long(@bitCast(std.math.inf(f32))));
    try std.testing.expectEqual(std.math.minInt(i64), float_to_long(@bitCast(-std.math.inf(f32))));
    try std.testing.expectEqual(std.math.maxInt(i64), float_to_long(@bitCast(@as(f32, @floatFromInt(std.math.maxInt(i64))))));
    try std.testing.expectEqual(std.math.minInt(i64), float_to_long(@bitCast(@as(f32, @floatFromInt(std.math.minInt(i64))))));
    try std.testing.expectEqual(@as(i64, 12345), float_to_long(@bitCast(@as(f32, 12345.75))));

    const double_int_bytes = try encodedTestInstructions(std.testing.allocator, &.{
        .{ .double_to_int = .{ .dest = 2, .src = 0 } },
        .{ .return_ = .{ .src = 2 } },
    });
    const DoubleIntFn = fn (u64, u64) callconv(.c) i32;
    const double_to_int = (try addOwnedTestBytes(&cache, double_int_bytes)).typedEntry(DoubleIntFn);
    try std.testing.expectEqual(@as(i32, 0), double_to_int(@bitCast(std.math.nan(f64)), 0));
    try std.testing.expectEqual(std.math.maxInt(i32), double_to_int(@bitCast(@as(f64, @floatFromInt(std.math.maxInt(i32))) + 1024.0), 0));
    try std.testing.expectEqual(std.math.minInt(i32), double_to_int(@bitCast(@as(f64, @floatFromInt(std.math.minInt(i32))) - 1024.0), 0));
    try std.testing.expectEqual(std.math.maxInt(i32), double_to_int(@bitCast(@as(f64, @floatFromInt(std.math.maxInt(i32)))), 0));
    try std.testing.expectEqual(std.math.minInt(i32), double_to_int(@bitCast(@as(f64, @floatFromInt(std.math.minInt(i32)))), 0));
    try std.testing.expectEqual(@as(i32, -42), double_to_int(@bitCast(@as(f64, -42.875)), 0));

    const double_long_bytes = try encodedTestInstructions(std.testing.allocator, &.{
        .{ .double_to_long = .{ .dest = 2, .src = 0 } },
        .{ .return_wide = .{ .src = 2 } },
    });
    const DoubleLongFn = fn (u64, u64) callconv(.c) i64;
    const double_to_long = (try addOwnedTestBytes(&cache, double_long_bytes)).typedEntry(DoubleLongFn);
    try std.testing.expectEqual(@as(i64, 0), double_to_long(@bitCast(std.math.nan(f64)), 0));
    try std.testing.expectEqual(std.math.maxInt(i64), double_to_long(@bitCast(std.math.inf(f64)), 0));
    try std.testing.expectEqual(std.math.minInt(i64), double_to_long(@bitCast(-std.math.inf(f64)), 0));
    try std.testing.expectEqual(std.math.maxInt(i64), double_to_long(@bitCast(@as(f64, @floatFromInt(std.math.maxInt(i64)))), 0));
    try std.testing.expectEqual(std.math.minInt(i64), double_to_long(@bitCast(@as(f64, @floatFromInt(std.math.minInt(i64)))), 0));
    try std.testing.expectEqual(@as(i64, 1234567890123), double_to_long(@bitCast(@as(f64, 1234567890123.75)), 0));
}

test "x64_register_encoder converts through scalar XMM pressure and spills" {
    if (builtin.cpu.arch != .x86_64) return error.SkipZigTest;
    const insts = [_]Instruction{
        .{ .add_float = .{ .dest = 7, .src1 = 0, .src2 = 1 } },
        .{ .add_float = .{ .dest = 8, .src1 = 2, .src2 = 3 } },
        .{ .add_float = .{ .dest = 9, .src1 = 4, .src2 = 5 } },
        .{ .add_float = .{ .dest = 10, .src1 = 7, .src2 = 6 } },
        .{ .float_to_int = .{ .dest = 11, .src = 10 } },
        .{ .return_ = .{ .src = 11 } },
    };
    var optimized = try optimizedMachine(std.testing.allocator, &insts);
    defer optimized.deinit();
    var native = try encode(std.testing.allocator, &optimized.machine);
    defer native.deinit();
    try std.testing.expect(native.allocation.stats.spills > 0);
    try std.testing.expect(native.stats.xmm_spill_loads > 0);
    try std.testing.expect(native.stats.xmm_spill_stores > 0);
    try std.testing.expectEqual(@as(u32, 1), native.stats.xmm_saturating_conversions);
    const bytes = try native.finalize();
    defer std.testing.allocator.free(bytes);
    var cache = jit_memory.Cache.init(std.testing.allocator);
    defer cache.deinit();
    const allocation = try cache.addBytes(bytes);
    const Fn = fn (u32, u32, u32, u32, u32, u32, u32) callconv(.c) i32;
    const result = allocation.typedEntry(Fn)(
        @bitCast(@as(f32, 1)),
        @bitCast(@as(f32, 2)),
        @bitCast(@as(f32, 3)),
        @bitCast(@as(f32, 4)),
        @bitCast(@as(f32, 5)),
        @bitCast(@as(f32, 6)),
        @bitCast(@as(f32, 7)),
    );
    try std.testing.expectEqual(@as(i32, 10), result);
    try std.testing.checkAllAllocationFailures(std.testing.allocator, edgeEncodingFailureProbe, .{&optimized.machine});
}

test "x64_register_encoder executes scalar double constants and arithmetic" {
    if (builtin.cpu.arch != .x86_64) return error.SkipZigTest;
    const insts = [_]Instruction{
        .{ .const_wide = .{ .dest = 0, .value = @bitCast(@as(f64, 1.5)) } },
        .{ .const_wide = .{ .dest = 2, .value = @bitCast(@as(f64, 2.25)) } },
        .{ .add_double = .{ .dest = 4, .src1 = 0, .src2 = 2 } },
        .{ .return_wide = .{ .src = 4 } },
    };
    var optimized = try optimizedMachine(std.testing.allocator, &insts);
    defer optimized.deinit();
    var native = try encode(std.testing.allocator, &optimized.machine);
    defer native.deinit();
    const bytes = try native.finalize();
    defer std.testing.allocator.free(bytes);

    var cache = jit_memory.Cache.init(std.testing.allocator);
    defer cache.deinit();
    const allocation = try cache.addBytes(bytes);
    const Fn = fn () callconv(.c) u64;
    const result: f64 = @bitCast(allocation.typedEntry(Fn)());
    try std.testing.expectEqual(@as(f64, 3.75), result);
}

test "x64_register_encoder rejects unsupported scalar XMM remainder" {
    const insts = [_]Instruction{
        .{ .rem_float = .{ .dest = 2, .src1 = 0, .src2 = 1 } },
        .{ .return_ = .{ .src = 2 } },
    };
    var optimized = try optimizedMachine(std.testing.allocator, &insts);
    defer optimized.deinit();
    try std.testing.expectError(error.UnsupportedInstruction, encode(std.testing.allocator, &optimized.machine));
}

test "x64_register_encoder executes a spill-heavy aligned frame" {
    if (builtin.cpu.arch != .x86_64) return error.SkipZigTest;
    const insts = [_]Instruction{
        .{ .add_int = .{ .dest = 8, .src1 = 0, .src2 = 1 } },
        .{ .add_int = .{ .dest = 9, .src1 = 2, .src2 = 3 } },
        .{ .add_int = .{ .dest = 10, .src1 = 4, .src2 = 5 } },
        .{ .add_int = .{ .dest = 11, .src1 = 6, .src2 = 7 } },
        .{ .add_int = .{ .dest = 12, .src1 = 8, .src2 = 9 } },
        .{ .add_int = .{ .dest = 13, .src1 = 10, .src2 = 11 } },
        .{ .add_int = .{ .dest = 14, .src1 = 12, .src2 = 13 } },
        .{ .return_ = .{ .src = 14 } },
    };
    var optimized = try optimizedMachine(std.testing.allocator, &insts);
    defer optimized.deinit();
    var native = try encode(std.testing.allocator, &optimized.machine);
    defer native.deinit();
    try std.testing.expect(native.allocation.stats.spills > 0);
    try std.testing.expect(native.stats.frame_bytes > 0);
    try std.testing.expect(std.mem.isAligned(native.stats.frame_bytes, 16));
    try std.testing.expect(native.stats.spill_loads > 0);
    try std.testing.expect(native.stats.spill_stores > 0);

    const bytes = try native.finalize();
    defer std.testing.allocator.free(bytes);
    var cache = jit_memory.Cache.init(std.testing.allocator);
    defer cache.deinit();
    const allocation = try cache.addBytes(bytes);
    const Fn = fn (i32, i32, i32, i32, i32, i32, i32, i32) callconv(.c) i32;
    try std.testing.expectEqual(@as(i32, 36), allocation.typedEntry(Fn)(1, 2, 3, 4, 5, 6, 7, 8));
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
        .bounds_exception_helper = 0x1abc_def0,
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

fn boundsEncodingFailureProbe(allocator: std.mem.Allocator, source: *const machine.Function) !void {
    var abi = testRuntimeAbi();
    abi.reference_array_layout = .{ .length_offset = 0, .data_offset = 8 };
    var native = try encodeWithOptions(allocator, source, .{ .runtime = abi });
    defer native.deinit();
    const bytes = try native.finalize();
    defer allocator.free(bytes);
}

test "x64_register_encoder maps bounds exceptions and rejects forged sites" {
    if (builtin.cpu.arch != .x86_64) return error.SkipZigTest;
    const insts = [_]Instruction{
        .{ .aget_object = .{ .dest_or_src = 2, .array = 0, .index = 1 } },
        .{ .return_object = .{ .src = 2 } },
    };
    var optimized = try optimizedMachine(std.testing.allocator, &insts);
    defer optimized.deinit();
    try std.testing.expectEqual(@as(u32, 1), optimized.machine.stats.bounds_exception_sites);

    var abi = testRuntimeAbi();
    abi.reference_array_layout = .{ .length_offset = 0, .data_offset = 8 };
    var native = try encodeWithOptions(std.testing.allocator, &optimized.machine, .{ .runtime = abi });
    defer native.deinit();
    try native.verify();
    try std.testing.expectEqual(@as(u32, 1), native.stats.bounds_exception_sites);
    try std.testing.expectEqual(@as(u32, 2), native.stats.root_map_sites);
    const bytes = try native.finalize();
    defer std.testing.allocator.free(bytes);
    try std.testing.expect(bytes.len != 0);

    var missing = abi;
    missing.bounds_exception_helper = 0;
    try std.testing.expectError(error.MissingExceptionHelper, encodeWithOptions(std.testing.allocator, &optimized.machine, .{ .runtime = missing }));

    var found = false;
    for (optimized.machine.blocks) |block| {
        for (block.insts) |*inst| {
            if (inst.opcode != .check_bounds) continue;
            const saved = inst.exception_site_id;
            inst.exception_site_id = std.math.maxInt(u32);
            try std.testing.expectError(error.BadInstruction, optimized.machine.verify());
            inst.exception_site_id = saved;
            found = true;
        }
    }
    try std.testing.expect(found);
    try std.testing.checkAllAllocationFailures(std.testing.allocator, boundsEncodingFailureProbe, .{&optimized.machine});
}

test "x64_register_encoder emits allocation-safe guarded loop refresh" {
    if (builtin.cpu.arch != .x86_64) return error.SkipZigTest;
    const insts = [_]Instruction{
        .{ .const_ = .{ .dest = 1, .value = 2 } },
        .{ .goto_ = .{ .offset = 1 } },
        .{ .iget = .{ .field_idx = 1, .dest_or_src = 2, .obj = 0 } },
        .{ .if_eqz = .{ .src = 1, .offset = 3 } },
        .{ .add_int_lit8 = .{ .dest = 1, .src = 1, .lit = -1 } },
        .{ .goto_ = .{ .offset = -3 } },
        .{ .return_ = .{ .src = 2 } },
    };
    var optimized = try optimizedMachine(std.testing.allocator, &insts);
    defer optimized.deinit();
    try std.testing.expectEqual(@as(u32, 1), optimized.machine.stats.resolves);
    try std.testing.expectEqual(@as(u32, 1), optimized.machine.stats.loop_epoch_guards);

    var native = try encodeWithOptions(std.testing.allocator, &optimized.machine, .{ .runtime = testRuntimeAbi() });
    defer native.deinit();
    try native.verify();
    try std.testing.expectEqual(@as(u32, 1), native.stats.loop_epoch_guards);
    try std.testing.expectEqual(@as(u32, 1), native.stats.loop_epoch_slow_sites);
    try std.testing.expectEqual(@as(u32, 2), native.stats.root_map_sites);
    const bytes = try native.finalize();
    defer std.testing.allocator.free(bytes);
    try std.testing.expect(bytes.len != 0);

    try std.testing.checkAllAllocationFailures(std.testing.allocator, resolverEncodingFailureProbe, .{&optimized.machine});
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

test "x64_register_encoder publishes precise roots across sibling branches" {
    if (builtin.cpu.arch != .x86_64) return error.SkipZigTest;
    const insts = [_]Instruction{
        .{ .if_eqz = .{ .src = 2, .offset = 3 } },
        .{ .iget = .{ .field_idx = 1, .dest_or_src = 2, .obj = 0 } },
        .return_void,
        .{ .iget = .{ .field_idx = 2, .dest_or_src = 2, .obj = 1 } },
        .return_void,
    };
    var optimized = try optimizedMachine(std.testing.allocator, &insts);
    defer optimized.deinit();
    try std.testing.expect(optimized.machine.blocks.len > 1);
    try std.testing.expectEqual(@as(usize, 0), optimized.machine.edges.len);
    try std.testing.expectEqual(@as(u32, 2), optimized.machine.stats.resolves);

    var native = try encodeWithOptions(std.testing.allocator, &optimized.machine, .{ .runtime = testRuntimeAbi() });
    defer native.deinit();
    try native.verify();
    const maps = &(native.root_maps orelse return error.TestUnexpectedResult);
    try std.testing.expectEqual(@as(usize, 2), maps.records.len);
    for (maps.records) |*record| {
        // The other branch's receiver is globally allocated but path-dead.
        try std.testing.expectEqual(@as(usize, 1), maps.rootsFor(record).len);
    }
    try std.testing.expectEqual(@as(u32, 2), native.stats.root_map_locations);
    const bytes = try native.finalize();
    defer std.testing.allocator.free(bytes);
    try std.testing.expect(bytes.len != 0);
    try std.testing.checkAllAllocationFailures(std.testing.allocator, resolverEncodingFailureProbe, .{&optimized.machine});
}

test "x64_register_encoder maps a live spilled Handle in its real frame" {
    if (builtin.cpu.arch != .x86_64) return error.SkipZigTest;
    const insts = [_]Instruction{
        .{ .iget = .{ .field_idx = 1, .dest_or_src = 8, .obj = 0 } },
        .{ .add_int = .{ .dest = 9, .src1 = 2, .src2 = 3 } },
        .{ .add_int = .{ .dest = 10, .src1 = 4, .src2 = 5 } },
        .{ .add_int = .{ .dest = 11, .src1 = 6, .src2 = 7 } },
        .{ .add_int = .{ .dest = 12, .src1 = 9, .src2 = 10 } },
        .{ .add_int = .{ .dest = 13, .src1 = 12, .src2 = 11 } },
        .{ .return_ = .{ .src = 1 } },
    };
    var optimized = try optimizedMachine(std.testing.allocator, &insts);
    defer optimized.deinit();
    // This backend-level test makes the second incoming value a Handle after
    // front-end construction so it can isolate frame/root-map behavior from
    // object-bytecode lowering. The machine verifier remains authoritative.
    optimized.machine.reg_types[1] = .object;
    switch (optimized.machine.runtime_values[1]) {
        .dalvik => |*value| {
            value.ty = .object;
            value.gc_root = true;
        },
        .derived_ptr => return error.TestUnexpectedResult,
    }
    var native = try encodeWithOptions(std.testing.allocator, &optimized.machine, .{ .runtime = testRuntimeAbi() });
    defer native.deinit();
    try native.verify();
    try std.testing.expect(native.stats.frame_bytes > 0);

    const maps = &(native.root_maps orelse return error.TestUnexpectedResult);
    try std.testing.expectEqual(@as(usize, 1), maps.records.len);
    var found_stack_handle = false;
    for (maps.rootsFor(&maps.records[0])) |root| {
        if (root.kind != .stack_slot) continue;
        found_stack_handle = true;
        try std.testing.expect(root.stackOffset() >= 0);
        try std.testing.expect(@as(u32, @intCast(root.stackOffset())) < native.stats.frame_bytes);
    }
    try std.testing.expect(found_stack_handle);
    const bytes = try native.finalize();
    defer std.testing.allocator.free(bytes);
    try std.testing.expect(bytes.len != 0);
}

test "x64_register_encoder reloads pointer-access operands under pressure" {
    if (builtin.cpu.arch != .x86_64) return error.SkipZigTest;
    const insts = [_]Instruction{
        .{ .iget = .{ .field_idx = 1, .dest_or_src = 8, .obj = 0 } },
        .{ .iget = .{ .field_idx = 1, .dest_or_src = 9, .obj = 1 } },
        .{ .iget = .{ .field_idx = 1, .dest_or_src = 10, .obj = 2 } },
        .{ .iget = .{ .field_idx = 1, .dest_or_src = 11, .obj = 3 } },
        .{ .iget = .{ .field_idx = 1, .dest_or_src = 12, .obj = 4 } },
        .{ .iget = .{ .field_idx = 1, .dest_or_src = 13, .obj = 5 } },
        .{ .return_ = .{ .src = 8 } },
    };
    var optimized = try optimizedMachine(std.testing.allocator, &insts);
    defer optimized.deinit();
    var native = try encodeWithOptions(std.testing.allocator, &optimized.machine, .{ .runtime = testRuntimeAbi() });
    defer native.deinit();
    try native.verify();
    try std.testing.expect(native.stats.frame_bytes > 0);
    try std.testing.expect(native.stats.spill_loads > 0);
    try std.testing.expect(native.stats.spill_stores > 0);

    var pointer_result_spilled = false;
    for (optimized.machine.blocks) |block| {
        for (block.insts) |inst| {
            if (inst.opcode != .field_load_ptr or inst.defs.len != 1) continue;
            switch (native.allocation.locationOf(inst.defs[0]).?) {
                .spill => pointer_result_spilled = true,
                else => {},
            }
        }
    }
    try std.testing.expect(pointer_result_spilled);
    const bytes = try native.finalize();
    defer std.testing.allocator.free(bytes);
    try std.testing.expect(bytes.len != 0);
}

test "x64_register_encoder reloads spilled array barrier operands" {
    if (builtin.cpu.arch != .x86_64) return error.SkipZigTest;
    const insts = [_]Instruction{
        .{ .aput_object = .{ .dest_or_src = 1, .array = 0, .index = 2 } },
        .{ .iget = .{ .field_idx = 1, .dest_or_src = 8, .obj = 3 } },
        .{ .iget = .{ .field_idx = 1, .dest_or_src = 9, .obj = 4 } },
        .{ .iget = .{ .field_idx = 1, .dest_or_src = 10, .obj = 5 } },
        .{ .iget = .{ .field_idx = 1, .dest_or_src = 11, .obj = 6 } },
        .{ .iget = .{ .field_idx = 1, .dest_or_src = 12, .obj = 7 } },
        .return_void,
    };
    var optimized = try optimizedMachine(std.testing.allocator, &insts);
    defer optimized.deinit();
    var abi = testRuntimeAbi();
    abi.reference_array_layout = .{ .length_offset = 0, .data_offset = 8 };
    var native = try encodeWithOptions(std.testing.allocator, &optimized.machine, .{ .runtime = abi });
    defer native.deinit();
    try native.verify();
    try std.testing.expect(native.stats.frame_bytes > 0);
    try std.testing.expect(native.stats.spill_loads > 0);

    var found_spilled_store_operand = false;
    var found_spilled_handle = false;
    var found_spilled_index = false;
    for (optimized.machine.blocks) |block| {
        for (block.insts) |inst| {
            if (inst.opcode != .array_store_ptr or inst.uses.len != 2) continue;
            for (inst.uses) |use| switch (native.allocation.locationOf(use).?) {
                .spill => {
                    found_spilled_store_operand = true;
                    if (optimized.machine.isGcRoot(use)) found_spilled_handle = true;
                    if (!optimized.machine.isGcRoot(use)) found_spilled_index = true;
                },
                else => {},
            };
        }
    }
    try std.testing.expect(found_spilled_store_operand);
    try std.testing.expect(found_spilled_handle);
    try std.testing.expect(found_spilled_index);
    const maps = &(native.root_maps orelse return error.TestUnexpectedResult);
    var found_stack_root = false;
    for (maps.locations) |root| {
        if (root.kind == .stack_slot) found_stack_root = true;
    }
    try std.testing.expect(found_stack_root);
    const bytes = try native.finalize();
    defer std.testing.allocator.free(bytes);
    try std.testing.expect(bytes.len != 0);
}

fn deoptEncodingFailureProbe(
    allocator: std.mem.Allocator,
    source: *const machine.Function,
    deopt: DeoptOptions,
) !void {
    var epoch = std.atomic.Value(u64).init(0);
    var runtime = testRuntimeAbi();
    runtime.deopt_epoch_address = @intFromPtr(&epoch);
    runtime.deopt_helper = 1;
    var native = try encodeWithOptions(allocator, source, .{
        .runtime = runtime,
        .deopt = deopt,
    });
    defer native.deinit();
}

fn osrEncodingFailureProbe(
    allocator: std.mem.Allocator,
    source: *const machine.Function,
    deopt: DeoptOptions,
    osr: []const OsrEntrySpec,
    runtime: RuntimeAbi,
) !void {
    var native = try encodeWithOptions(allocator, source, .{
        .runtime = runtime,
        .deopt = deopt,
        .osr_entries = osr,
    });
    defer native.deinit();
}

test "x64 compiler translates deoptimization values after register allocation" {
    if (builtin.cpu.arch != .x86_64) return error.SkipZigTest;
    const insts = [_]Instruction{
        .{ .iget = .{ .field_idx = 1, .dest_or_src = 1, .obj = 0 } },
        .{ .add_int = .{ .dest = 3, .src1 = 2, .src2 = 1 } },
        .{ .return_ = .{ .src = 3 } },
    };
    var optimized = try optimizedMachine(std.testing.allocator, &insts);
    defer optimized.deinit();

    var scalar_before: ?machine.RegId = null;
    var scalar_after: ?machine.RegId = null;
    var receiver: ?machine.RegId = null;
    var site_id: ?u32 = null;
    for (optimized.machine.blocks) |block| {
        for (block.insts) |inst| {
            if (inst.opcode == .add_i32 and inst.pc == 1) {
                scalar_after = inst.defs[0];
                for (inst.uses) |use| {
                    if (use < optimized.machine.value_kinds.len and
                        optimized.machine.value_kinds[use] == .parameter and
                        !optimized.machine.isGcRoot(use)) scalar_before = use;
                }
            }
            if (inst.opcode == .resolve_handle and inst.pc == 0) {
                receiver = inst.state_handle;
                site_id = inst.resolve_id;
            }
        }
    }
    const live_scalar = scalar_before orelse return error.TestUnexpectedResult;
    const dead_scalar = scalar_after orelse return error.TestUnexpectedResult;
    const handle_reg = receiver orelse return error.TestUnexpectedResult;
    const safepoint = site_id orelse return error.TestUnexpectedResult;
    const values = [_]DeoptValueSpec{
        .{ .vreg = 0, .kind = .reference, .source = .{ .machine_register = handle_reg } },
        .{ .vreg = 1, .kind = .scalar32, .source = .{ .machine_register = live_scalar } },
        .{ .vreg = 2, .kind = .scalar32, .source = .{ .constant = 99 } },
    };
    const points = [_]DeoptPointSpec{.{
        .id = 17,
        .safepoint_id = safepoint,
        .method_id = 23,
        .dex_pc = 0,
        .values = &values,
    }};
    var deopt_epoch = std.atomic.Value(u64).init(0);
    var deopt_runtime = testRuntimeAbi();
    deopt_runtime.deopt_epoch_address = @intFromPtr(&deopt_epoch);
    deopt_runtime.deopt_helper = 1;
    var native = try encodeWithOptions(std.testing.allocator, &optimized.machine, .{
        .runtime = deopt_runtime,
        .deopt = .{ .points = &points, .register_count = 3, .max_dex_pc = 2 },
    });
    defer native.deinit();
    try native.verify();
    try std.testing.expectEqual(@as(u32, 1), native.stats.deopt_points);
    try std.testing.expectEqual(@as(u32, 3), native.stats.deopt_values);
    const table = if (native.deopt_table) |*value| value else return error.TestUnexpectedResult;
    const record = try table.find(17);
    const translated = table.valuesFor(record);
    try std.testing.expectEqual(runtime_deopt.Source.native_register, std.meta.activeTag(translated[0].source));
    try std.testing.expectEqual(runtime_deopt.Source.native_register, std.meta.activeTag(translated[1].source));
    const maps = if (native.root_maps) |*value| value else return error.TestUnexpectedResult;
    try std.testing.expectEqual(@as(u32, 17), (try maps.find(safepoint)).deopt_id);

    const handle = runtime_value.Handle{ .index = 8, .generation = 3 };
    var captured: [16]u64 = @splat(0);
    captured[translated[0].source.native_register] = @bitCast(handle);
    captured[translated[1].source.native_register] = 1234;
    var registers: [3]u32 = @splat(0);
    var references: [3]u64 = @splat(@as(u64, @bitCast(runtime_value.Handle.none)));
    var reference_kinds: [3]bool = @splat(false);
    var frame = runtime_deopt.Frame{ .execution = .{
        .pc = 0,
        .registers = &registers,
        .instructions = &.{},
        .register_is_ref = &reference_kinds,
        .reference_registers = &references,
    } };
    var scratch: [3]u64 = undefined;
    var anchor: u64 = 0;
    _ = try table.reconstruct(17, .{
        .native_registers = &captured,
        .stack_base = @ptrCast(&anchor),
        .stack_min_offset = 0,
        .stack_max_offset = @sizeOf(@TypeOf(anchor)),
    }, .{ .frame = &frame, .scratch = &scratch }, .invalidation, .{});
    try std.testing.expectEqual(@as(u64, @bitCast(handle)), frame.execution.reference_registers[0]);
    try std.testing.expectEqual(@as(u32, 1234), frame.execution.registers[1]);
    try std.testing.expectEqual(@as(u32, 99), frame.execution.registers[2]);
    var osr_image: [16]u64 = @splat(0);
    var osr_scratch: [19]u64 = undefined;
    try table.exportOsr(17, &frame, .{ .native_registers = &osr_image, .scratch = &osr_scratch });
    try std.testing.expectEqual(@as(u64, @bitCast(handle)), osr_image[translated[0].source.native_register]);
    try std.testing.expectEqual(@as(u64, 1234), osr_image[translated[1].source.native_register]);

    const dead_values = [_]DeoptValueSpec{
        .{ .vreg = 0, .kind = .reference, .source = .{ .machine_register = handle_reg } },
        .{ .vreg = 1, .kind = .scalar32, .source = .{ .machine_register = dead_scalar } },
        .{ .vreg = 2, .kind = .scalar32, .source = .{ .constant = 0 } },
    };
    const dead_points = [_]DeoptPointSpec{.{
        .id = 18,
        .safepoint_id = safepoint,
        .method_id = 23,
        .dex_pc = 0,
        .values = &dead_values,
    }};
    try std.testing.expectError(error.DeadDeoptValue, encodeWithOptions(std.testing.allocator, &optimized.machine, .{
        .runtime = deopt_runtime,
        .deopt = .{ .points = &dead_points, .register_count = 3, .max_dex_pc = 2 },
    }));

    const missing_points = [_]DeoptPointSpec{.{
        .id = 19,
        .safepoint_id = 999,
        .method_id = 23,
        .dex_pc = 0,
        .values = &values,
    }};
    try std.testing.expectError(error.MissingDeoptSafepoint, encodeWithOptions(std.testing.allocator, &optimized.machine, .{
        .runtime = deopt_runtime,
        .deopt = .{ .points = &missing_points, .register_count = 3, .max_dex_pc = 2 },
    }));

    const duplicate_points = [_]DeoptPointSpec{
        points[0],
        .{ .id = 20, .safepoint_id = safepoint, .method_id = 23, .dex_pc = 0, .values = &values },
    };
    try std.testing.expectError(error.InvalidDeoptMetadata, encodeWithOptions(std.testing.allocator, &optimized.machine, .{
        .runtime = deopt_runtime,
        .deopt = .{ .points = &duplicate_points, .register_count = 3, .max_dex_pc = 2 },
    }));

    var allocation = try regalloc.allocate(std.testing.allocator, &optimized.machine, .{ .gp_registers = allGpRegisters() });
    defer allocation.deinit();
    try ensureNoSpills(&allocation);
    var spill_plan = try regalloc.planSpills(std.testing.allocator, &allocation);
    defer spill_plan.deinit();
    var linked_stats: Stats = .{};
    const deopt_options = DeoptOptions{ .points = &points, .register_count = 3, .max_dex_pc = 2 };
    var linked_maps = (try buildRootMaps(
        std.testing.allocator,
        &optimized.machine,
        &allocation,
        &spill_plan,
        &linked_stats,
        deopt_options,
    )) orelse return error.TestUnexpectedResult;
    defer linked_maps.deinit();
    const saved_scalar_location = allocation.locations[live_scalar];
    allocation.locations[live_scalar] = allocation.locations[handle_reg];
    defer allocation.locations[live_scalar] = saved_scalar_location;
    var alias_stats: Stats = .{};
    try std.testing.expectError(error.AliasedDeoptValue, buildDeoptTable(
        std.testing.allocator,
        &optimized.machine,
        &allocation,
        &spill_plan,
        &linked_maps,
        deopt_options,
        &alias_stats,
    ));
    allocation.locations[live_scalar] = saved_scalar_location;

    {
        allocation.locations[live_scalar] = .{ .spill = 0 };
        defer allocation.locations[live_scalar] = saved_scalar_location;
        var slots = [_]regalloc.SpillSlot{.{
            .reg = live_scalar,
            .slot = 0,
            .ty = optimized.machine.reg_types[live_scalar],
            .size = 4,
            .byte_offset = 0,
        }};
        var translated_spill_plan = regalloc.SpillPlan{
            .allocator = std.testing.allocator,
            .source = &optimized.machine,
            .location_count = allocation.locations.len,
            .slots = &slots,
            .stats = .{ .slots = 1, .stack_bytes = 16 },
        };
        try translated_spill_plan.verify();
        var spill_stats: Stats = .{};
        var spill_table = (try buildDeoptTable(
            std.testing.allocator,
            &optimized.machine,
            &allocation,
            &translated_spill_plan,
            &linked_maps,
            deopt_options,
            &spill_stats,
        )).?;
        defer spill_table.deinit();
        const spill_values = spill_table.valuesFor(try spill_table.find(17));
        try std.testing.expectEqual(runtime_deopt.Source.stack_slot, std.meta.activeTag(spill_values[1].source));
        try std.testing.expectEqual(@as(i32, 0), spill_values[1].source.stack_slot);
        try std.testing.expectEqual(@as(u32, 1), spill_stats.deopt_stack_values);
    }

    {
        allocation.locations[live_scalar] = .{ .phys = .xmm3 };
        defer allocation.locations[live_scalar] = saved_scalar_location;
        var xmm_stats: Stats = .{};
        var xmm_table = (try buildDeoptTable(
            std.testing.allocator,
            &optimized.machine,
            &allocation,
            &spill_plan,
            &linked_maps,
            deopt_options,
            &xmm_stats,
        )).?;
        defer xmm_table.deinit();
        const xmm_values = xmm_table.valuesFor(try xmm_table.find(17));
        try std.testing.expectEqual(runtime_deopt.Source.xmm_register, std.meta.activeTag(xmm_values[1].source));
        try std.testing.expectEqual(@as(u8, 3), xmm_values[1].source.xmm_register);
        try std.testing.expectEqual(@as(u32, 1), xmm_stats.deopt_xmm_values);
    }

    try std.testing.checkAllAllocationFailures(
        std.testing.allocator,
        deoptEncodingFailureProbe,
        .{ &optimized.machine, DeoptOptions{ .points = &points, .register_count = 3, .max_dex_pc = 2 } },
    );

    const allocation_failure_callers = [_]DeoptInlineFrameSpec{.{
        .method_id = 22,
        .dex_pc = 0,
        .register_count = 3,
        .values = &values,
    }};
    const allocation_failure_points = [_]DeoptPointSpec{.{
        .id = 17,
        .safepoint_id = safepoint,
        .method_id = 23,
        .dex_pc = 0,
        .values = &values,
        .inline_frames = &allocation_failure_callers,
    }};
    try std.testing.checkAllAllocationFailures(
        std.testing.allocator,
        deoptEncodingFailureProbe,
        .{ &optimized.machine, DeoptOptions{ .points = &allocation_failure_points, .register_count = 3, .max_dex_pc = 2 } },
    );
}

test "x64 compiler publishes a verified no-prologue OSR safepoint label" {
    if (builtin.cpu.arch != .x86_64) return error.SkipZigTest;
    const insts = [_]Instruction{
        .{ .const_ = .{ .dest = 1, .value = 2 } },
        .{ .goto_ = .{ .offset = 1 } },
        .{ .iget = .{ .field_idx = 1, .dest_or_src = 2, .obj = 0 } },
        .{ .if_eqz = .{ .src = 1, .offset = 3 } },
        .{ .add_int_lit8 = .{ .dest = 1, .src = 1, .lit = -1 } },
        .{ .goto_ = .{ .offset = -3 } },
        .{ .return_ = .{ .src = 2 } },
    };
    var optimized = try optimizer.optimize(std.testing.allocator, &insts, &.{}, .{
        .enable_loop_resolve_hoisting = false,
    });
    defer optimized.deinit();
    try std.testing.expectEqual(@as(u32, 1), optimized.loops.stats.loops);
    try std.testing.expectEqual(@as(u32, 0), optimized.stats.loop_resolves_hoisted);

    var handle_reg: ?machine.RegId = null;
    var safepoint_id: ?u32 = null;
    var safepoint_pc: ?u32 = null;
    var header: ?cfg.BlockId = null;
    for (optimized.machine.blocks) |block| {
        for (block.insts) |inst| {
            if (inst.opcode == .resolve_handle) {
                handle_reg = inst.state_handle;
                safepoint_id = inst.resolve_id;
                safepoint_pc = inst.pc;
                header = block.id;
            }
        }
    }
    const analyzed_loop = optimized.loops.loops[0];
    const object = handle_reg orelse return error.TestUnexpectedResult;
    const site = safepoint_id orelse return error.TestUnexpectedResult;
    const dex_pc = safepoint_pc orelse return error.TestUnexpectedResult;
    const loop_header = header orelse return error.TestUnexpectedResult;
    try std.testing.expectEqual(analyzed_loop.header, loop_header);
    const osr_probe = try findSafepointPosition(&optimized.machine, site);
    var result_reg: ?machine.RegId = null;
    for (0..optimized.machine.reg_types.len) |reg_index| {
        const reg: machine.RegId = @intCast(reg_index);
        if (!try osrRegisterReadBeforeDefinition(std.testing.allocator, &optimized.machine, osr_probe, reg)) continue;
        switch (optimized.machine.runtime_values[reg]) {
            .dalvik => |value| if (optimized.function.values[value.value].reg == 1) {
                result_reg = reg;
                break;
            },
            .derived_ptr => {},
        }
    }
    const loop_result = result_reg orelse return error.TestUnexpectedResult;

    const values = [_]DeoptValueSpec{
        .{ .vreg = 0, .kind = .reference, .source = .{ .machine_register = object } },
        .{ .vreg = 1, .kind = .scalar32, .source = .{ .machine_register = loop_result } },
    };
    const points = [_]DeoptPointSpec{.{
        .id = 41,
        .safepoint_id = site,
        .method_id = 7,
        .dex_pc = dex_pc,
        .values = &values,
    }};
    const osr = [_]OsrEntrySpec{.{ .point_id = 41, .block = loop_header }};
    var epoch = std.atomic.Value(u64).init(0);
    var abi = testRuntimeAbi();
    abi.deopt_epoch_address = @intFromPtr(&epoch);
    abi.deopt_helper = 1;
    var native = try encodeWithOptions(std.testing.allocator, &optimized.machine, .{
        .runtime = abi,
        .deopt = .{ .points = &points, .register_count = 2, .max_dex_pc = 4 },
        .osr_entries = &osr,
    });
    defer native.deinit();
    try native.verify();
    try std.testing.expectEqual(@as(u32, 1), native.stats.osr_entries);
    const entry = try native.osrEntry(41);
    try std.testing.expectEqual(@as(u32, 41), entry.point_id);
    try std.testing.expect(entry.code_offset != 0);
    try std.testing.expect(std.mem.isAligned(entry.code_offset, 16));
    try std.testing.expect(entry.code_offset < native.buffer.len());

    const invalid = [_]OsrEntrySpec{.{ .point_id = 41, .block = analyzed_loop.preheader orelse return error.TestUnexpectedResult }};
    try std.testing.expectError(error.InvalidDeoptMetadata, encodeWithOptions(std.testing.allocator, &optimized.machine, .{
        .runtime = abi,
        .deopt = .{ .points = &points, .register_count = 2, .max_dex_pc = 4 },
        .osr_entries = &invalid,
    }));
}

test "x64 compiler publishes a frameful OSR landing stub" {
    if (builtin.cpu.arch != .x86_64) return error.SkipZigTest;
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
    var optimized = try optimizer.optimize(std.testing.allocator, &insts, &.{}, .{
        .enable_loop_resolve_hoisting = false,
    });
    defer optimized.deinit();

    var object: ?machine.RegId = null;
    var loop_value: ?machine.RegId = null;
    var site: ?u32 = null;
    var dex_pc: ?u32 = null;
    var header: ?cfg.BlockId = null;
    for (optimized.machine.blocks) |block| {
        for (block.insts) |inst| {
            if (inst.opcode != .resolve_handle) continue;
            object = inst.state_handle;
            site = inst.resolve_id;
            dex_pc = inst.pc;
            header = block.id;
        }
    }
    const safepoint = site orelse return error.TestUnexpectedResult;
    const probe = try findSafepointPosition(&optimized.machine, safepoint);
    for (0..optimized.machine.reg_types.len) |reg_index| {
        const reg: machine.RegId = @intCast(reg_index);
        if (!try osrRegisterReadBeforeDefinition(std.testing.allocator, &optimized.machine, probe, reg)) continue;
        switch (optimized.machine.runtime_values[reg]) {
            .dalvik => |value| {
                if (optimized.function.values[value.value].reg == 1) loop_value = reg;
            },
            .derived_ptr => {},
        }
    }

    var values: [10]DeoptValueSpec = undefined;
    for (&values, 0..) |*value, vreg| value.* = .{
        .vreg = @intCast(vreg),
        .kind = .scalar32,
        .source = .{ .constant = if (vreg == 9) 123 else 0 },
    };
    values[0] = .{ .vreg = 0, .kind = .reference, .source = .{ .machine_register = object orelse return error.TestUnexpectedResult } };
    values[1] = .{ .vreg = 1, .kind = .scalar32, .source = .{ .machine_register = loop_value orelse return error.TestUnexpectedResult } };
    const points = [_]DeoptPointSpec{.{
        .id = 61,
        .safepoint_id = safepoint,
        .method_id = 9,
        .dex_pc = dex_pc orelse return error.TestUnexpectedResult,
        .values = &values,
    }};
    const osr = [_]OsrEntrySpec{.{ .point_id = 61, .block = header orelse return error.TestUnexpectedResult }};
    var epoch = std.atomic.Value(u64).init(0);
    var abi = testRuntimeAbi();
    abi.deopt_epoch_address = @intFromPtr(&epoch);
    abi.deopt_helper = 1;
    var native = try encodeWithOptions(std.testing.allocator, &optimized.machine, .{
        .runtime = abi,
        .deopt = .{ .points = &points, .register_count = 10, .max_dex_pc = 7 },
        .osr_entries = &osr,
    });
    defer native.deinit();
    try native.verify();
    try std.testing.expect(native.stats.frame_bytes > 0);
    try std.testing.expectEqual(@as(u32, 1), native.stats.osr_frame_landings);
    const maps = &(native.root_maps orelse return error.TestUnexpectedResult);
    for (maps.rootsFor(try maps.find(safepoint))) |root| {
        try std.testing.expectEqual(runtime_stack_map.LocationKind.native_register, root.kind);
    }
    const entry = try native.osrEntry(61);
    try std.testing.expect(std.mem.isAligned(entry.code_offset, 16));
    const bytes = try native.finalize();
    defer std.testing.allocator.free(bytes);
    try std.testing.expect(entry.code_offset < bytes.len);
    if (builtin.os.tag == .windows and native.stats.nonvolatile_frame_bytes != 0) {
        try std.testing.expectEqual(@as(u8, 0x56), bytes[entry.code_offset]); // push rsi
        try std.testing.expectEqual(@as(u8, 0x57), bytes[entry.code_offset + 1]); // push rdi
    } else {
        try std.testing.expectEqual(@as(u8, 0x48), bytes[entry.code_offset]);
        try std.testing.expect(bytes[entry.code_offset + 1] == 0x83 or bytes[entry.code_offset + 1] == 0x81);
    }
}

test "x64 OSR landing rematerializes a hoisted derived address from mapped Handles" {
    if (builtin.cpu.arch != .x86_64) return error.SkipZigTest;
    const insts = [_]Instruction{
        .{ .const_ = .{ .dest = 9, .value = 123 } },
        .{ .const_ = .{ .dest = 1, .value = 2 } },
        .{ .const_ = .{ .dest = 2, .value = 0 } },
        .{ .goto_ = .{ .offset = 1 } },
        .{ .aget_object = .{ .dest_or_src = 3, .array = 0, .index = 2 } },
        .{ .if_eqz = .{ .src = 1, .offset = 3 } },
        .{ .add_int_lit8 = .{ .dest = 1, .src = 1, .lit = -1 } },
        .{ .goto_ = .{ .offset = -3 } },
        .{ .return_object = .{ .src = 3 } },
    };
    var optimized = try optimizer.optimize(std.testing.allocator, &insts, &.{}, .{});
    defer optimized.deinit();
    try std.testing.expectEqual(@as(u32, 1), optimized.stats.loop_resolves_hoisted);
    try std.testing.expectEqual(@as(usize, 1), optimized.barriers.loop_reuses.len);
    const reuse = optimized.barriers.loop_reuses[0];
    const remat = try osrRematOperands(&optimized.machine, reuse.resolve);

    var safepoint: ?u32 = null;
    var dex_pc: ?u32 = null;
    for (optimized.machine.blocks[reuse.header].insts) |inst| {
        if (inst.opcode != .check_bounds or inst.exception_site_id == null) continue;
        safepoint = inst.exception_site_id;
        dex_pc = inst.pc;
        break;
    }
    const site = safepoint orelse return error.TestUnexpectedResult;
    const probe = try findSafepointPosition(&optimized.machine, site);
    var loop_value: ?machine.RegId = null;
    var index_value: ?machine.RegId = null;
    for (0..optimized.machine.reg_types.len) |reg_index| {
        const reg: machine.RegId = @intCast(reg_index);
        if (!try osrRegisterReadBeforeDefinition(std.testing.allocator, &optimized.machine, probe, reg)) continue;
        switch (optimized.machine.runtime_values[reg]) {
            .dalvik => |value| switch (optimized.function.values[value.value].reg) {
                1 => loop_value = reg,
                2 => index_value = reg,
                else => {},
            },
            .derived_ptr => try std.testing.expectEqual(remat.address, reg),
        }
    }

    var values: [10]DeoptValueSpec = undefined;
    for (&values, 0..) |*value, vreg| value.* = .{
        .vreg = @intCast(vreg),
        .kind = .scalar32,
        .source = .{ .constant = if (vreg == 9) 123 else 0 },
    };
    values[0] = .{ .vreg = 0, .kind = .reference, .source = .{ .machine_register = remat.handle } };
    values[1] = .{ .vreg = 1, .kind = .scalar32, .source = .{ .machine_register = loop_value orelse return error.TestUnexpectedResult } };
    values[2] = .{ .vreg = 2, .kind = .scalar32, .source = .{ .machine_register = index_value orelse return error.TestUnexpectedResult } };
    values[3] = .{ .vreg = 3, .kind = .reference, .source = .{ .constant = @bitCast(runtime_value.Handle.none) } };
    const points = [_]DeoptPointSpec{.{
        .id = 71,
        .safepoint_id = site,
        .method_id = 10,
        .dex_pc = dex_pc orelse return error.TestUnexpectedResult,
        .values = &values,
    }};
    const osr = [_]OsrEntrySpec{.{ .point_id = 71, .block = reuse.header }};
    var epoch = std.atomic.Value(u64).init(0);
    var abi = testRuntimeAbi();
    abi.reference_array_layout = .{ .length_offset = 0, .data_offset = 8 };
    abi.deopt_epoch_address = @intFromPtr(&epoch);
    abi.deopt_helper = 1;
    var native = try encodeWithOptions(std.testing.allocator, &optimized.machine, .{
        .runtime = abi,
        .deopt = .{ .points = &points, .register_count = 10, .max_dex_pc = 8 },
        .osr_entries = &osr,
    });
    defer native.deinit();
    try native.verify();
    try std.testing.expect(native.stats.frame_bytes > 0);
    try std.testing.expectEqual(@as(u32, 1), native.stats.osr_frame_landings);
    try std.testing.expectEqual(@as(u32, 1), native.stats.osr_landing_safepoints);
    try std.testing.expectEqual(@as(u32, 1), native.stats.osr_derived_rematerializations);
    try std.testing.expectEqual(@as(u32, 1), native.stats.osr_remat_restart_edges);
    const maps = &(native.root_maps orelse return error.TestUnexpectedResult);
    const landing_map = try maps.find(try osrLandingSiteId(
        &optimized.machine,
        DeoptOptions{ .points = &points, .register_count = 10, .max_dex_pc = 8 },
        0,
    ));
    try std.testing.expectEqual(runtime_stack_map.no_deopt, landing_map.deopt_id);
    const handle_physical = try x64Reg(try physOf(&native.allocation, remat.handle));
    try std.testing.expectEqualSlices(
        runtime_stack_map.RootLocation,
        &.{runtime_stack_map.RootLocation.nativeRegister(handle_physical)},
        maps.rootsFor(landing_map),
    );
    const entry = try native.osrEntry(71);
    try std.testing.expect(std.mem.isAligned(entry.code_offset, 16));
    const bytes = try native.finalize();
    defer std.testing.allocator.free(bytes);
    // Landing epoch snapshot, cold-edge reload, and r12 comparison. The slow
    // edge restarts only when the helper actually acknowledged a new epoch.
    try std.testing.expectEqual(@as(usize, 1), std.mem.count(u8, bytes, &.{ 0x4c, 0x89, 0x24, 0x24 }));
    try std.testing.expectEqual(@as(usize, 1), std.mem.count(u8, bytes, &.{ 0x4c, 0x8b, 0x1c, 0x24 }));
    try std.testing.expectEqual(@as(usize, 1), std.mem.count(u8, bytes, &.{ 0x4d, 0x3b, 0xe3 }));

    const saved = values[0];
    values[0] = .{ .vreg = 0, .kind = .reference, .source = .{ .constant = @bitCast(runtime_value.Handle.none) } };
    defer values[0] = saved;
    try std.testing.expectError(error.InvalidDeoptMetadata, encodeWithOptions(std.testing.allocator, &optimized.machine, .{
        .runtime = abi,
        .deopt = .{ .points = &points, .register_count = 10, .max_dex_pc = 8 },
        .osr_entries = &osr,
    }));

    values[0] = saved;
    try std.testing.checkAllAllocationFailures(
        std.testing.allocator,
        osrEncodingFailureProbe,
        .{
            &optimized.machine,
            DeoptOptions{ .points = &points, .register_count = 10, .max_dex_pc = 8 },
            osr[0..],
            abi,
        },
    );
}

test "x64 compiler creates a collision-free block-entry deoptimization OSR position" {
    if (builtin.cpu.arch != .x86_64) return error.SkipZigTest;
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
    const reuse = optimized.barriers.loop_reuses[0];
    const remat = try osrRematOperands(&optimized.machine, reuse.resolve);
    for (optimized.machine.blocks[reuse.header].insts) |inst| {
        try std.testing.expect(instructionSafepointId(inst) == null);
    }

    const required = try osrRequiredRegistersAtBlockEntry(std.testing.allocator, &optimized.machine, reuse.header);
    defer std.testing.allocator.free(required);
    var loop_value: ?machine.RegId = null;
    var found_address = false;
    for (required) |reg| switch (optimized.machine.runtime_values[reg]) {
        .dalvik => |value| if (optimized.function.values[value.value].reg == 1) {
            loop_value = reg;
        },
        .derived_ptr => found_address = reg == remat.address,
    };
    try std.testing.expect(found_address);

    var values: [10]DeoptValueSpec = undefined;
    for (&values, 0..) |*value, vreg| value.* = .{
        .vreg = @intCast(vreg),
        .kind = .scalar32,
        .source = .{ .constant = if (vreg == 9) 123 else 0 },
    };
    values[0] = .{ .vreg = 0, .kind = .reference, .source = .{ .machine_register = remat.handle } };
    values[1] = .{ .vreg = 1, .kind = .scalar32, .source = .{ .machine_register = loop_value orelse return error.TestUnexpectedResult } };
    var points = [_]DeoptPointSpec{.{
        .id = 81,
        .block_entry = reuse.header,
        .method_id = 11,
        .dex_pc = optimized.machine.blocks[reuse.header].insts[0].pc orelse return error.TestUnexpectedResult,
        .values = &values,
    }};
    const deopt = DeoptOptions{ .points = &points, .register_count = 10, .max_dex_pc = 7 };
    const osr = [_]OsrEntrySpec{.{ .point_id = 81, .block = reuse.header }};
    var epoch = std.atomic.Value(u64).init(0);
    var abi = testRuntimeAbi();
    abi.deopt_epoch_address = @intFromPtr(&epoch);
    abi.deopt_helper = 1;
    var native = try encodeWithOptions(std.testing.allocator, &optimized.machine, .{
        .runtime = abi,
        .deopt = deopt,
        .osr_entries = &osr,
    });
    defer native.deinit();
    try native.verify();
    try std.testing.expectEqual(@as(u32, 1), native.stats.deopt_block_entries);
    try std.testing.expectEqual(@as(u32, 1), native.stats.deopt_guards);
    try std.testing.expectEqual(@as(u32, 1), native.stats.osr_landing_safepoints);
    try std.testing.expectEqual(@as(u32, 1), native.stats.osr_derived_rematerializations);

    const maps = &(native.root_maps orelse return error.TestUnexpectedResult);
    const block_site = try deoptPointSiteId(&optimized.machine, deopt, 0);
    const block_map = try maps.find(block_site);
    try std.testing.expectEqual(@as(u32, 81), block_map.deopt_id);
    const landing_site = try osrLandingSiteId(&optimized.machine, deopt, 0);
    try std.testing.expect(landing_site > block_site);
    const landing_map = try maps.find(landing_site);
    try std.testing.expectEqual(runtime_stack_map.no_deopt, landing_map.deopt_id);
    const handle_physical = try x64Reg(try physOf(&native.allocation, remat.handle));
    var block_has_handle = false;
    for (maps.rootsFor(block_map)) |root| {
        try std.testing.expectEqual(runtime_stack_map.LocationKind.native_register, root.kind);
        if (root.payload == handle_physical) block_has_handle = true;
    }
    try std.testing.expect(block_has_handle);
    try std.testing.expectEqualSlices(
        runtime_stack_map.RootLocation,
        &.{runtime_stack_map.RootLocation.nativeRegister(handle_physical)},
        maps.rootsFor(landing_map),
    );
    const bytes = try native.finalize();
    defer std.testing.allocator.free(bytes);
    try std.testing.expect(bytes.len != 0);

    points[0].safepoint_id = 0;
    try std.testing.expectError(error.InvalidDeoptMetadata, encodeWithOptions(std.testing.allocator, &optimized.machine, .{
        .runtime = abi,
        .deopt = deopt,
        .osr_entries = &osr,
    }));
    points[0].safepoint_id = null;
    points[0].dex_pc += 1;
    try std.testing.expectError(error.InvalidDeoptMetadata, encodeWithOptions(std.testing.allocator, &optimized.machine, .{
        .runtime = abi,
        .deopt = deopt,
        .osr_entries = &osr,
    }));
    points[0].dex_pc -= 1;
    const duplicate_points = [_]DeoptPointSpec{
        points[0],
        .{
            .id = 82,
            .block_entry = reuse.header,
            .method_id = 11,
            .dex_pc = points[0].dex_pc,
            .values = &values,
        },
    };
    try std.testing.expectError(
        error.InvalidDeoptMetadata,
        deoptPointSiteId(
            &optimized.machine,
            .{ .points = &duplicate_points, .register_count = 10, .max_dex_pc = 7 },
            1,
        ),
    );
    try std.testing.checkAllAllocationFailures(
        std.testing.allocator,
        osrEncodingFailureProbe,
        .{ &optimized.machine, deopt, osr[0..], abi },
    );
}

test "x64 frameful OSR rejects stack and XMM image sources" {
    const valid = [_]runtime_deopt.ValueSpec{
        .{ .vreg = 0, .kind = .scalar32, .source = .{ .native_register = 1 } },
        .{ .vreg = 1, .kind = .scalar32, .source = .{ .constant = 7 } },
    };
    try validateOsrMappedSources(&valid);
    const stack = [_]runtime_deopt.ValueSpec{
        .{ .vreg = 0, .kind = .scalar32, .source = .{ .stack_slot = 0 } },
    };
    try std.testing.expectError(error.InvalidDeoptMetadata, validateOsrMappedSources(&stack));
    const xmm = [_]runtime_deopt.ValueSpec{
        .{ .vreg = 0, .kind = .scalar64, .source = .{ .xmm_register = 3 } },
    };
    try std.testing.expectError(error.InvalidDeoptMetadata, validateOsrMappedSources(&xmm));
}

test "x64 root liveness translates live phi destinations on predecessor edges" {
    if (builtin.cpu.arch != .x86_64) return error.SkipZigTest;
    const insts = [_]Instruction{
        .{ .iget = .{ .field_idx = 1, .dest_or_src = 2, .obj = 2 } },
        .{ .if_eqz = .{ .src = 3, .offset = 3 } },
        .{ .iget_object = .{ .field_idx = 2, .dest_or_src = 2, .obj = 0 } },
        .{ .goto_ = .{ .offset = 2 } },
        .{ .iget_object = .{ .field_idx = 3, .dest_or_src = 2, .obj = 1 } },
        .{ .iget = .{ .field_idx = 1, .dest_or_src = 3, .obj = 2 } },
        .return_void,
    };
    var optimized = try optimizedMachine(std.testing.allocator, &insts);
    defer optimized.deinit();
    try std.testing.expect(optimized.machine.blocks.len > 1);
    try std.testing.expectEqual(@as(usize, 2), optimized.machine.edges.len);
    try std.testing.expectEqual(@as(u32, 4), optimized.machine.stats.resolves);

    var allocation = try regalloc.allocate(std.testing.allocator, &optimized.machine, .{ .gp_registers = allGpRegisters() });
    defer allocation.deinit();
    try ensureNoSpills(&allocation);
    var spill_plan = try regalloc.planSpills(std.testing.allocator, &allocation);
    defer spill_plan.deinit();
    var stats: Stats = .{};
    var maps = (try buildRootMaps(std.testing.allocator, &optimized.machine, &allocation, &spill_plan, &stats, null)) orelse return error.TestUnexpectedResult;
    defer maps.deinit();

    var before_branch_id: ?u32 = null;
    var after_merge_id: ?u32 = null;
    for (optimized.machine.blocks) |block| {
        for (block.insts) |inst| {
            if (inst.opcode != .resolve_handle) continue;
            if (inst.pc == 0) before_branch_id = inst.resolve_id;
            if (inst.pc == 5) after_merge_id = inst.resolve_id;
        }
    }
    const before_branch = try maps.find(before_branch_id orelse return error.TestUnexpectedResult);
    const after_merge = try maps.find(after_merge_id orelse return error.TestUnexpectedResult);
    // Before the branch: receiver v4 plus each path's incoming object.
    try std.testing.expectEqual(@as(usize, 3), maps.rootsFor(before_branch).len);
    // After the merge: only the selected phi result remains live.
    try std.testing.expectEqual(@as(usize, 1), maps.rootsFor(after_merge).len);
    try std.testing.expectEqual(@as(u32, 6), stats.root_map_locations);

    const phi_field_layouts = [_]FieldLayout{
        .{ .offset = 8, .storage = .i32 },
        .{ .offset = 16, .storage = .i32 },
        .{ .offset = 24, .storage = .reference },
        .{ .offset = 32, .storage = .reference },
    };
    var abi = testRuntimeAbi();
    abi.field_layouts = &phi_field_layouts;
    var native = try encodeWithOptions(std.testing.allocator, &optimized.machine, .{ .runtime = abi });
    defer native.deinit();
    try std.testing.expectEqual(@as(u32, 2), native.stats.edge_copy_sites);
    const bytes = try native.finalize();
    defer std.testing.allocator.free(bytes);
    try std.testing.expect(bytes.len != 0);

    try std.testing.expect(optimized.machine.isGcRoot(0));
    try std.testing.expect(optimized.machine.isGcRoot(1));
    {
        const saved_location = allocation.locations[0];
        allocation.locations[0] = .{ .spill = 0 };
        defer allocation.locations[0] = saved_location;
        var root_slots = [_]regalloc.SpillSlot{.{
            .reg = 0,
            .slot = 0,
            .ty = optimized.machine.reg_types[0],
            .size = 8,
            .byte_offset = 0,
        }};
        const root_spill_plan = regalloc.SpillPlan{
            .allocator = std.testing.allocator,
            .source = &optimized.machine,
            .location_count = allocation.locations.len,
            .slots = &root_slots,
            .stats = .{ .slots = 1, .stack_bytes = 16 },
        };
        try root_spill_plan.verify();
        var spill_root_stats: Stats = .{};
        var spill_maps = (try buildRootMaps(
            std.testing.allocator,
            &optimized.machine,
            &allocation,
            &root_spill_plan,
            &spill_root_stats,
            null,
        )) orelse return error.TestUnexpectedResult;
        defer spill_maps.deinit();
        var found_stack_root = false;
        for (spill_maps.locations) |location| {
            if (location.kind == .stack_slot and location.stackOffset() == 0) found_stack_root = true;
        }
        try std.testing.expect(found_stack_root);
    }
    {
        const saved_location = allocation.locations[1];
        allocation.locations[1] = allocation.locations[0];
        defer allocation.locations[1] = saved_location;
        var bad_stats: Stats = .{};
        try std.testing.expectError(error.InvalidMachine, buildRootMaps(std.testing.allocator, &optimized.machine, &allocation, &spill_plan, &bad_stats, null));
    }
    {
        const saved_edges = optimized.machine.edges;
        optimized.machine.edges = saved_edges[0..1];
        defer optimized.machine.edges = saved_edges;
        var bad_stats: Stats = .{};
        try std.testing.expectError(error.InvalidMachine, buildRootMaps(std.testing.allocator, &optimized.machine, &allocation, &spill_plan, &bad_stats, null));
    }
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
