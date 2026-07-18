//! Backend-facing lowered IR.
//!
//! This is the bridge between the analysis-heavy SSA pipeline and a machine
//! code backend. It keeps SSA value ids as virtual registers, but lowers broad
//! Dalvik operations into a compact set of typed, backend-friendly opcodes.

const std = @import("std");
const cfg = @import("cfg");
const dom = @import("dominator");
const ssa = @import("ssa");
const ssa_phase = @import("ssa_phase");
const typedir = @import("typedir");
const typed_ir = @import("typed_ir");
const optimizer = @import("optimizer");
const memory_phase = @import("memory_phase");
const barrier_phase = @import("barrier_phase");
const instmod = @import("instructions");
const Instruction = instmod.Instruction;

pub const Error = error{
    InvalidInput,
    OutOfMemory,
};

pub const VerifyError = error{
    BadBlock,
    BadInstruction,
    BadValue,
    InvalidInput,
    OutOfMemory,
};

pub const RuntimeValueId = u32;
pub const ValueId = RuntimeValueId;
pub const INVALID_RUNTIME_VALUE: RuntimeValueId = std.math.maxInt(RuntimeValueId);

pub const RuntimeValueClass = union(enum) {
    dalvik: struct {
        value: ssa.ValueId,
        ty: typedir.Type,
        gc_root: bool,
    },
    derived_ptr: struct {
        handle: ssa.ValueId,
        token: barrier_phase.RelocTokenId,
        resolve: barrier_phase.ResolveId,
    },

    pub fn isGcRoot(self: RuntimeValueClass) bool {
        return switch (self) {
            .dalvik => |value| value.gc_root,
            .derived_ptr => false,
        };
    }
};

pub const Kind = enum(u8) {
    phi,
    const_i32,
    const_i64,
    copy,
    add_i32,
    sub_i32,
    mul_i32,
    div_i32,
    rem_i32,
    and_i32,
    or_i32,
    xor_i32,
    add_i64,
    sub_i64,
    mul_i64,
    div_i64,
    rem_i64,
    f32_op,
    f64_op,
    check_null,
    check_bounds,
    resolve_handle,
    loop_epoch_guard,
    array_load,
    array_store,
    field_load,
    field_store,
    array_load_ptr,
    array_store_ptr,
    field_load_ptr,
    field_store_ptr,
    satb_pre_write,
    card_mark,
    static_satb_pre_write,
    static_root_post_write,
    static_load,
    static_store,
    call_direct,
    call_static,
    call_virtual,
    call_quick,
    branch,
    cond_branch,
    switch_,
    ret,
    throw_,
    memory_barrier,
    unsupported,
};

/// Preserves the bytecode operation after type-directed lowering selects the
/// scalar SSE lane width. Unsupported operations remain explicit so a backend
/// can fail closed instead of silently choosing the wrong floating semantics.
pub const FloatOperation = enum(u8) {
    add,
    sub,
    mul,
    div,
    rem,
    neg,
    compare_l,
    compare_g,
    int_to_float,
    int_to_double,
    long_to_float,
    long_to_double,
    float_to_int,
    float_to_long,
    float_to_double,
    double_to_int,
    double_to_long,
    double_to_float,
};

pub const Flags = packed struct {
    dead: bool = false,
    null_check_elided: bool = false,
    bounds_check_elided: bool = false,
    forwarded: bool = false,
    cse: bool = false,
};

pub const Inst = struct {
    kind: Kind,
    pc: ?u32 = null,
    defs: []ValueId = &.{},
    uses: []ValueId = &.{},
    target: ?cfg.BlockId = null,
    false_target: ?cfg.BlockId = null,
    imm: i64 = 0,
    float_op: ?FloatOperation = null,
    field_idx: ?u32 = null,
    address: ?RuntimeValueId = null,
    state_handle: ?RuntimeValueId = null,
    reloc_token: ?barrier_phase.RelocTokenId = null,
    resolve_id: ?barrier_phase.ResolveId = null,
    guard_site_id: ?u32 = null,
    exception_site_id: ?u32 = null,
    pre_write: barrier_phase.PreWriteBarrier = .none,
    post_write: barrier_phase.PostWriteBarrier = .none,
    flags: Flags = .{},
};

pub const Block = struct {
    id: cfg.BlockId,
    insts: []Inst,
};

pub const Inputs = struct {
    function: *const ssa.Function,
    types: *const typedir.Function,
    typed: *const typed_ir.Function,
    ssa_facts: ?*const ssa_phase.Result = null,
    memory: ?*const memory_phase.Result = null,
    barriers: ?*const barrier_phase.Result = null,
};

pub const Stats = struct {
    lowered: u32 = 0,
    skipped_dead: u32 = 0,
    constants_materialized: u32 = 0,
    null_checks_elided: u32 = 0,
    bounds_checks_elided: u32 = 0,
    forwarded_loads: u32 = 0,
    direct_calls: u32 = 0,
    runtime_values: u32 = 0,
    handle_resolves: u32 = 0,
    loop_epoch_guards: u32 = 0,
    bounds_exception_sites: u32 = 0,
    pointer_accesses: u32 = 0,
    satb_barriers: u32 = 0,
    card_barriers: u32 = 0,
    static_root_barriers: u32 = 0,
};

pub const Function = struct {
    allocator: std.mem.Allocator,
    source: *const ssa.Function,
    blocks: []Block,
    value_types: []typedir.Type,
    runtime_values: []RuntimeValueClass,
    resolve_values: []RuntimeValueId,
    barriers: ?*const barrier_phase.Result,
    stats: Stats,

    pub fn deinit(self: *Function) void {
        for (self.blocks) |block| {
            for (block.insts) |inst| {
                self.allocator.free(inst.defs);
                self.allocator.free(inst.uses);
            }
            self.allocator.free(block.insts);
        }
        self.allocator.free(self.blocks);
        self.allocator.free(self.value_types);
        self.allocator.free(self.runtime_values);
        self.allocator.free(self.resolve_values);
        self.* = undefined;
    }

    pub fn verify(self: *const Function) VerifyError!void {
        if (self.blocks.len != self.source.blocks.len or self.value_types.len != self.runtime_values.len) return error.InvalidInput;
        if (self.runtime_values.len < self.source.values.len) return error.InvalidInput;
        for (self.source.values, 0..) |value, i| {
            switch (self.runtime_values[i]) {
                .dalvik => |runtime| {
                    if (runtime.value != value.id or runtime.ty != self.value_types[i]) return error.BadValue;
                },
                .derived_ptr => return error.BadValue,
            }
        }
        if (self.barriers) |barriers| {
            if (barriers.function != self.source or self.resolve_values.len != barriers.resolves.len or self.runtime_values.len != self.source.values.len + barriers.resolves.len) return error.InvalidInput;
            for (barriers.resolves, 0..) |resolve, resolve_index| {
                const value = self.resolve_values[resolve_index];
                const expected_value: RuntimeValueId = @intCast(self.source.values.len + resolve_index);
                const expected_resolve: barrier_phase.ResolveId = @intCast(resolve_index);
                if (value != expected_value) return error.BadValue;
                switch (self.runtime_values[value]) {
                    .derived_ptr => |ptr| if (ptr.handle != resolve.handle or ptr.token != resolve.token or ptr.resolve != expected_resolve) return error.BadValue,
                    .dalvik => return error.BadValue,
                }
                if (!self.isGcRoot(resolve.handle)) return error.BadValue;
            }
        } else if (self.resolve_values.len != 0 or self.runtime_values.len != self.source.values.len) {
            return error.InvalidInput;
        }
        const exception_site_base: u32 = if (self.barriers) |barriers|
            std.math.add(u32, @intCast(barriers.resolves.len), barriers.stats.loop_epoch_guards) catch return error.BadInstruction
        else
            0;
        const loop_guard_count: u32 = if (self.barriers) |barriers| barriers.stats.loop_epoch_guards else 0;
        const loop_guard_seen = try self.allocator.alloc(bool, loop_guard_count);
        defer self.allocator.free(loop_guard_seen);
        @memset(loop_guard_seen, false);
        var verified_exception_sites: u32 = 0;
        for (self.blocks, 0..) |block, i| {
            if (block.id != i) return error.BadBlock;
            for (block.insts) |inst| {
                for (inst.defs) |def| {
                    if (def >= self.value_types.len) return error.BadValue;
                    switch (self.runtime_values[def]) {
                        .dalvik => {},
                        .derived_ptr => if (inst.kind != .resolve_handle) return error.BadInstruction,
                    }
                }
                for (inst.uses) |use| if (use >= self.value_types.len) return error.BadValue;
                if (inst.address) |address| if (address >= self.runtime_values.len) return error.BadValue;
                if (inst.state_handle) |handle| if (handle >= self.runtime_values.len) return error.BadValue;
                switch (inst.kind) {
                    .branch => if (inst.target == null) return error.BadInstruction,
                    .cond_branch => if (inst.target == null or inst.false_target == null or inst.uses.len == 0) return error.BadInstruction,
                    .field_load, .field_store, .static_load, .static_store, .static_satb_pre_write, .static_root_post_write => if (inst.field_idx == null) return error.BadInstruction,
                    .resolve_handle => try self.verifyResolve(inst),
                    .loop_epoch_guard => {
                        const guard_id = try self.verifyLoopEpochGuard(block.id, inst);
                        if (guard_id >= loop_guard_seen.len or loop_guard_seen[guard_id]) return error.BadInstruction;
                        loop_guard_seen[guard_id] = true;
                    },
                    .check_bounds, .field_load_ptr, .field_store_ptr, .array_load_ptr, .array_store_ptr, .satb_pre_write, .card_mark => if (inst.address != null) try self.verifyAddress(inst),
                    .call_direct, .call_static, .call_virtual, .call_quick => if (inst.address != null) try self.verifyAddress(inst),
                    else => {},
                }
                if ((inst.kind == .f32_op or inst.kind == .f64_op) != (inst.float_op != null)) return error.BadInstruction;
                for (inst.uses) |use| switch (self.runtime_values[use]) {
                    .dalvik => {},
                    .derived_ptr => return error.BadInstruction,
                };
                if (inst.address == null and inst.state_handle != null and inst.kind != .resolve_handle) return error.BadInstruction;
                const permits_address = switch (inst.kind) {
                    .field_load_ptr,
                    .field_store_ptr,
                    .array_load_ptr,
                    .array_store_ptr,
                    .check_bounds,
                    .satb_pre_write,
                    .card_mark,
                    .call_direct,
                    .call_static,
                    .call_virtual,
                    .call_quick,
                    .loop_epoch_guard,
                    => true,
                    else => false,
                };
                if (inst.address != null and !permits_address) return error.BadInstruction;
                if (inst.address == null and inst.kind != .resolve_handle and (inst.reloc_token != null or inst.resolve_id != null)) return error.BadInstruction;
                if ((inst.kind == .loop_epoch_guard) != (inst.guard_site_id != null)) return error.BadInstruction;
                const is_mapped_bounds = inst.kind == .check_bounds and inst.address != null;
                if (is_mapped_bounds != (inst.exception_site_id != null)) return error.BadInstruction;
                if (is_mapped_bounds) {
                    const expected = std.math.add(u32, exception_site_base, verified_exception_sites) catch return error.BadInstruction;
                    if (inst.exception_site_id.? != expected or inst.uses.len != 1) return error.BadInstruction;
                    verified_exception_sites += 1;
                }
                const is_pre_write = inst.kind == .satb_pre_write or inst.kind == .static_satb_pre_write;
                const is_post_write = inst.kind == .card_mark or inst.kind == .static_root_post_write;
                if (is_pre_write != (inst.pre_write != .none)) return error.BadInstruction;
                if (is_post_write != (inst.post_write != .none)) return error.BadInstruction;
                if (inst.kind == .satb_pre_write) {
                    if (inst.field_idx == null and inst.uses.len != 1) return error.BadInstruction;
                    if (inst.field_idx != null and inst.uses.len != 0) return error.BadInstruction;
                }
                if (inst.kind == .static_satb_pre_write and (inst.uses.len != 0 or inst.pre_write != .satb_guarded)) return error.BadInstruction;
                if (inst.kind == .static_root_post_write and (inst.uses.len != 1 or inst.post_write != .root_guarded)) return error.BadInstruction;
            }
        }
        for (loop_guard_seen) |seen| if (!seen) return error.BadInstruction;
        if (loop_guard_count != self.stats.loop_epoch_guards) return error.BadInstruction;
        if (verified_exception_sites != self.stats.bounds_exception_sites) return error.BadInstruction;
    }

    fn verifyResolve(self: *const Function, inst: Inst) VerifyError!void {
        if (inst.defs.len != 1 or inst.uses.len != 1 or inst.address != null or inst.state_handle == null or inst.reloc_token == null or inst.resolve_id == null) return error.BadInstruction;
        if (inst.state_handle.? != inst.uses[0]) return error.BadInstruction;
        if (!self.isGcRoot(inst.state_handle.?)) return error.BadInstruction;
        const barriers = self.barriers orelse return error.BadInstruction;
        const resolve_id = inst.resolve_id.?;
        if (resolve_id >= barriers.resolves.len or self.resolve_values[resolve_id] != inst.defs[0]) return error.BadInstruction;
        const resolve = barriers.resolves[resolve_id];
        if (resolve.token != inst.reloc_token.?) return error.BadInstruction;
        switch (self.runtime_values[inst.defs[0]]) {
            .derived_ptr => |ptr| if (ptr.handle != resolve.handle or ptr.token != resolve.token or ptr.resolve != resolve_id) return error.BadInstruction,
            .dalvik => return error.BadInstruction,
        }
        switch (self.runtime_values[inst.uses[0]]) {
            .dalvik => |value| if (value.value != resolve.handle or (value.ty != .object and value.ty != .unknown)) return error.BadInstruction,
            .derived_ptr => return error.BadInstruction,
        }
    }

    fn verifyAddress(self: *const Function, inst: Inst) VerifyError!void {
        if ((inst.kind == .field_load_ptr or inst.kind == .field_store_ptr) and inst.field_idx == null) return error.BadInstruction;
        const address = inst.address orelse return error.BadInstruction;
        const state_handle = inst.state_handle orelse return error.BadInstruction;
        const token = inst.reloc_token orelse return error.BadInstruction;
        const resolve_id = inst.resolve_id orelse return error.BadInstruction;
        if (!self.isGcRoot(state_handle)) return error.BadInstruction;
        switch (self.runtime_values[address]) {
            .derived_ptr => |ptr| {
                if (ptr.token != token or ptr.resolve != resolve_id) return error.BadInstruction;
                switch (self.runtime_values[state_handle]) {
                    .dalvik => |handle| if (handle.value != ptr.handle or (handle.ty != .object and handle.ty != .unknown)) return error.BadInstruction,
                    .derived_ptr => return error.BadInstruction,
                }
            },
            .dalvik => return error.BadInstruction,
        }
    }

    fn verifyLoopEpochGuard(self: *const Function, block_id: cfg.BlockId, inst: Inst) VerifyError!u32 {
        if (inst.defs.len != 0 or inst.uses.len != 0 or inst.field_idx != null or
            inst.pre_write != .none or inst.post_write != .none) return error.BadInstruction;
        try self.verifyAddress(inst);
        const barriers = self.barriers orelse return error.BadInstruction;
        const resolve_id = inst.resolve_id orelse return error.BadInstruction;
        if (resolve_id >= barriers.resolves.len) return error.BadInstruction;
        const resolve = barriers.resolves[resolve_id];
        if (resolve.handle != inst.state_handle.? or resolve.token != inst.reloc_token.?) return error.BadInstruction;
        const site_id = inst.guard_site_id.?;
        if (site_id < barriers.resolves.len) return error.BadInstruction;
        const guard_id: u32 = @intCast(site_id - barriers.resolves.len);
        if (guard_id >= barriers.stats.loop_epoch_guards) return error.BadInstruction;
        for (barriers.loop_reuses) |reuse| {
            if (reuse.resolve != resolve_id or guard_id < reuse.guard_start) continue;
            const offset = guard_id - reuse.guard_start;
            if (offset >= reuse.latches.len) continue;
            if (reuse.latches[offset] != block_id) return error.BadInstruction;
            return guard_id;
        }
        return error.BadInstruction;
    }

    pub fn runtimeValue(self: *const Function, value: RuntimeValueId) ?RuntimeValueClass {
        if (value >= self.runtime_values.len) return null;
        return self.runtime_values[value];
    }

    pub fn isGcRoot(self: *const Function, value: RuntimeValueId) bool {
        const class = self.runtimeValue(value) orelse return false;
        return class.isGcRoot();
    }

    pub fn print(self: *const Function, writer: anytype) !void {
        try writer.print(
            "lowering blocks={d} values={d} runtime_values={d} lowered={d} skipped_dead={d} consts={d} null_elided={d} bounds_elided={d} forwarded={d} direct_calls={d} resolves={d} loop_guards={d} bounds_exceptions={d} ptr_accesses={d} satb={d} cards={d}\n",
            .{
                self.blocks.len,
                self.value_types.len,
                self.stats.runtime_values,
                self.stats.lowered,
                self.stats.skipped_dead,
                self.stats.constants_materialized,
                self.stats.null_checks_elided,
                self.stats.bounds_checks_elided,
                self.stats.forwarded_loads,
                self.stats.direct_calls,
                self.stats.handle_resolves,
                self.stats.loop_epoch_guards,
                self.stats.bounds_exception_sites,
                self.stats.pointer_accesses,
                self.stats.satb_barriers,
                self.stats.card_barriers,
            },
        );
        for (self.blocks) |block| {
            try writer.print("b{d}\n", .{block.id});
            for (block.insts) |inst| {
                try writer.print("  {s}", .{@tagName(inst.kind)});
                if (inst.pc) |pc| try writer.print(" pc{d}", .{pc});
                if (inst.defs.len != 0) {
                    try writer.print(" defs:", .{});
                    for (inst.defs) |def| try writer.print(" v{d}:{s}", .{ def, @tagName(self.value_types[def]) });
                }
                if (inst.uses.len != 0) {
                    try writer.print(" uses:", .{});
                    for (inst.uses) |use| try writer.print(" v{d}", .{use});
                }
                if (inst.target) |target| try writer.print(" target=b{d}", .{target});
                if (inst.false_target) |target| try writer.print(" false=b{d}", .{target});
                if (inst.field_idx) |field| try writer.print(" field={d}", .{field});
                if (inst.address) |address| try writer.print(" address=rv{d}", .{address});
                if (inst.state_handle) |handle| try writer.print(" state=rv{d}", .{handle});
                if (inst.reloc_token) |token| try writer.print(" token={d}", .{token});
                if (inst.resolve_id) |resolve| try writer.print(" resolve={d}", .{resolve});
                if (inst.exception_site_id) |site| try writer.print(" exception_site={d}", .{site});
                if (inst.imm != 0 or inst.kind == .const_i32 or inst.kind == .const_i64) try writer.print(" imm={d}", .{inst.imm});
                if (inst.flags.null_check_elided) try writer.print(" null_elided", .{});
                if (inst.flags.bounds_check_elided) try writer.print(" bounds_elided", .{});
                if (inst.flags.forwarded) try writer.print(" forwarded", .{});
                if (inst.flags.cse) try writer.print(" cse", .{});
                try writer.print("\n", .{});
            }
        }
    }
};

fn dupeValues(allocator: std.mem.Allocator, values: []const ValueId) ![]ValueId {
    return try allocator.dupe(ValueId, values);
}

fn appendInst(list: *std.ArrayList(Inst), allocator: std.mem.Allocator, inst: Inst, stats: *Stats) !void {
    list.append(allocator, inst) catch |err| {
        allocator.free(inst.defs);
        allocator.free(inst.uses);
        return err;
    };
    stats.lowered += 1;
}

fn ownOperands(
    allocator: std.mem.Allocator,
    template: Inst,
    defs: []const ValueId,
    uses: []const ValueId,
) !Inst {
    var inst = template;
    inst.defs = try dupeValues(allocator, defs);
    errdefer allocator.free(inst.defs);
    inst.uses = try dupeValues(allocator, uses);
    return inst;
}

fn foldedConstant(inputs: Inputs, op: ssa.Operation) ?ssa_phase.Constant {
    const facts = inputs.ssa_facts orelse return null;
    if (op.pc == std.math.maxInt(u32)) return null;
    if (op.defs.len == 0) return null;
    return facts.values[op.defs[0]].constant;
}

fn opDead(inputs: Inputs, block_id: cfg.BlockId, index: usize) bool {
    const facts = inputs.ssa_facts orelse return false;
    if (block_id >= facts.ops.len or index >= facts.ops[block_id].len) return false;
    return !facts.ops[block_id][index].live and !facts.ops[block_id][index].side_effect;
}

fn memInfo(inputs: Inputs, block_id: cfg.BlockId, index: usize) ?memory_phase.OpInfo {
    const memory = inputs.memory orelse return null;
    if (block_id >= memory.ops.len or index >= memory.ops[block_id].len) return null;
    return memory.ops[block_id][index];
}

fn typedInfo(inputs: Inputs, block_id: cfg.BlockId, index: usize) typed_ir.OpInfo {
    return inputs.typed.opInfo(block_id, index) orelse .{};
}

const AccessState = struct {
    address: RuntimeValueId,
    handle: RuntimeValueId,
    token: barrier_phase.RelocTokenId,
    resolve: barrier_phase.ResolveId,
    defines: bool,
    pre_write: barrier_phase.PreWriteBarrier,
    post_write: barrier_phase.PostWriteBarrier,
};

fn accessState(inputs: Inputs, block_id: cfg.BlockId, op_index: usize) ?AccessState {
    const barriers = inputs.barriers orelse return null;
    if (block_id >= barriers.ops.len or op_index >= barriers.ops[block_id].len) return null;
    const plan = barriers.ops[block_id][op_index];
    const handle = plan.base_handle orelse return null;
    const resolved = switch (plan.resolve) {
        .none => return null,
        .define => |id| .{ id, true },
        .reuse => |id| .{ id, false },
    };
    const base: u64 = inputs.function.values.len;
    const address: u64 = base + resolved[0];
    if (address >= std.math.maxInt(RuntimeValueId)) return null;
    return .{
        .address = @intCast(address),
        .handle = handle,
        .token = plan.token_in,
        .resolve = resolved[0],
        .defines = resolved[1],
        .pre_write = plan.pre_write,
        .post_write = plan.post_write,
    };
}

fn emitResolve(
    allocator: std.mem.Allocator,
    access: AccessState,
    pc: ?u32,
    list: *std.ArrayList(Inst),
    stats: *Stats,
) !void {
    if (!access.defines) return;
    try appendInst(list, allocator, try ownOperands(allocator, .{
        .kind = .resolve_handle,
        .pc = pc,
        .state_handle = access.handle,
        .reloc_token = access.token,
        .resolve_id = access.resolve,
    }, &.{access.address}, &.{access.handle}), stats);
    stats.handle_resolves += 1;
}

fn emitHoistedResolves(
    allocator: std.mem.Allocator,
    inputs: Inputs,
    block_id: cfg.BlockId,
    list: *std.ArrayList(Inst),
    stats: *Stats,
) !void {
    const barriers = inputs.barriers orelse return;
    for (barriers.resolves, 0..) |resolve, resolve_index| {
        if (!resolve.hoisted or resolve.placement_block != block_id) continue;
        if (resolve.defining_op.block >= inputs.function.blocks.len or
            resolve.defining_op.index >= inputs.function.blocks[resolve.defining_op.block].ops.len) return error.InvalidInput;
        const pc = inputs.function.blocks[resolve.defining_op.block].ops[resolve.defining_op.index].pc;
        const address_value: u64 = inputs.function.values.len + resolve_index;
        if (address_value >= std.math.maxInt(RuntimeValueId)) return error.InvalidInput;
        try appendInst(list, allocator, try ownOperands(allocator, .{
            .kind = .resolve_handle,
            .pc = pc,
            .state_handle = resolve.handle,
            .reloc_token = resolve.token,
            .resolve_id = @intCast(resolve_index),
        }, &.{@as(RuntimeValueId, @intCast(address_value))}, &.{resolve.handle}), stats);
        stats.handle_resolves += 1;
    }
}

fn emitLoopEpochGuards(
    allocator: std.mem.Allocator,
    inputs: Inputs,
    block_id: cfg.BlockId,
    list: *std.ArrayList(Inst),
    stats: *Stats,
) !void {
    const barriers = inputs.barriers orelse return;
    for (barriers.loop_reuses) |reuse| {
        var latch_offset: ?u32 = null;
        for (reuse.latches, 0..) |latch, index| {
            if (latch == block_id) {
                latch_offset = @intCast(index);
                break;
            }
        }
        const offset = latch_offset orelse continue;
        if (reuse.resolve >= barriers.resolves.len) return error.InvalidInput;
        const resolve = barriers.resolves[reuse.resolve];
        const guard_id = std.math.add(u32, reuse.guard_start, offset) catch return error.InvalidInput;
        const site_value: u64 = @as(u64, @intCast(barriers.resolves.len)) + guard_id;
        const resolve_index = reuse.resolve;
        const address_value: u64 = inputs.function.values.len + resolve_index;
        if (site_value > std.math.maxInt(u32) or address_value >= std.math.maxInt(RuntimeValueId)) return error.InvalidInput;
        if (resolve.defining_op.block >= inputs.function.blocks.len or
            resolve.defining_op.index >= inputs.function.blocks[resolve.defining_op.block].ops.len) return error.InvalidInput;
        const pc = inputs.function.blocks[resolve.defining_op.block].ops[resolve.defining_op.index].pc;
        try appendInst(list, allocator, try ownOperands(allocator, .{
            .kind = .loop_epoch_guard,
            .pc = pc,
            .address = @intCast(address_value),
            .state_handle = resolve.handle,
            .reloc_token = resolve.token,
            .resolve_id = @intCast(resolve_index),
            .guard_site_id = @intCast(site_value),
        }, &.{}, &.{}), stats);
        stats.loop_epoch_guards += 1;
    }
}

fn isControlTransfer(inst: Instruction) bool {
    return switch (inst) {
        .goto_,
        .if_eq,
        .if_ne,
        .if_lt,
        .if_ge,
        .if_gt,
        .if_le,
        .if_eqz,
        .if_nez,
        .if_ltz,
        .if_gez,
        .if_gtz,
        .if_lez,
        .packed_switch,
        .sparse_switch,
        .return_void,
        .return_,
        .return_wide,
        .return_object,
        .throw_,
        => true,
        else => false,
    };
}

fn emitPreWrite(
    allocator: std.mem.Allocator,
    access: AccessState,
    field_idx: ?u32,
    array_index: ?RuntimeValueId,
    pc: ?u32,
    list: *std.ArrayList(Inst),
    stats: *Stats,
) !void {
    if (access.pre_write == .none) return;
    const uses = if (array_index) |index| &[_]RuntimeValueId{index} else &.{};
    try appendInst(list, allocator, .{
        .kind = .satb_pre_write,
        .pc = pc,
        .uses = try dupeValues(allocator, uses),
        .field_idx = field_idx,
        .address = access.address,
        .state_handle = access.handle,
        .reloc_token = access.token,
        .resolve_id = access.resolve,
        .pre_write = access.pre_write,
    }, stats);
    stats.satb_barriers += 1;
}

fn emitPostWrite(
    allocator: std.mem.Allocator,
    access: AccessState,
    value_uses: []const RuntimeValueId,
    pc: ?u32,
    list: *std.ArrayList(Inst),
    stats: *Stats,
) !void {
    if (access.post_write == .none) return;
    try appendInst(list, allocator, .{
        .kind = .card_mark,
        .pc = pc,
        .uses = try dupeValues(allocator, value_uses),
        .address = access.address,
        .state_handle = access.handle,
        .reloc_token = access.token,
        .resolve_id = access.resolve,
        .post_write = access.post_write,
    }, stats);
    stats.card_barriers += 1;
}

fn emitStaticReferenceStore(
    allocator: std.mem.Allocator,
    field_idx: u32,
    value: RuntimeValueId,
    pre_write: barrier_phase.PreWriteBarrier,
    post_write: barrier_phase.PostWriteBarrier,
    pc: ?u32,
    flags: Flags,
    list: *std.ArrayList(Inst),
    stats: *Stats,
) !void {
    if (pre_write != .satb_guarded or post_write != .root_guarded) return error.InvalidInput;
    try appendInst(list, allocator, .{
        .kind = .static_satb_pre_write,
        .pc = pc,
        .field_idx = field_idx,
        .pre_write = pre_write,
    }, stats);
    stats.satb_barriers += 1;
    try appendInst(list, allocator, .{
        .kind = .static_store,
        .pc = pc,
        .uses = try dupeValues(allocator, &[_]RuntimeValueId{value}),
        .field_idx = field_idx,
        .flags = flags,
    }, stats);
    try appendInst(list, allocator, .{
        .kind = .static_root_post_write,
        .pc = pc,
        .uses = try dupeValues(allocator, &[_]RuntimeValueId{value}),
        .field_idx = field_idx,
        .post_write = post_write,
    }, stats);
    stats.static_root_barriers += 1;
}

fn fieldIndex(inst: Instruction) ?u32 {
    return switch (inst) {
        .iget,
        .iget_wide,
        .iget_object,
        .iget_boolean,
        .iget_byte,
        .iget_char,
        .iget_short,
        .iput,
        .iput_wide,
        .iput_object,
        .iput_boolean,
        .iput_byte,
        .iput_char,
        .iput_short,
        .iget_quick,
        .iget_wide_quick,
        .iget_object_quick,
        .iput_quick,
        .iput_wide_quick,
        .iput_object_quick,
        => |op| op.field_idx,
        .sget,
        .sget_wide,
        .sget_object,
        .sget_boolean,
        .sget_byte,
        .sget_char,
        .sget_short,
        .sput,
        .sput_wide,
        .sput_object,
        .sput_boolean,
        .sput_byte,
        .sput_char,
        .sput_short,
        => |op| op.field_idx,
        else => null,
    };
}

fn successorByKind(function: *const ssa.Function, block_id: cfg.BlockId, kind: cfg.EdgeKind) ?cfg.BlockId {
    for (function.graph.edges) |edge| {
        if (edge.from == block_id and edge.kind == kind) return edge.to;
    }
    return null;
}

fn lowerArithmetic(inst: Instruction, choice: typed_ir.LoweringChoice) Kind {
    return switch (choice) {
        .int32 => switch (inst) {
            .add_int, .add_int_lit8, .add_int_lit16 => .add_i32,
            .sub_int, .rsub_int_lit8, .rsub_int_lit16 => .sub_i32,
            .mul_int, .mul_int_lit8, .mul_int_lit16 => .mul_i32,
            .div_int, .div_int_lit8, .div_int_lit16 => .div_i32,
            .rem_int, .rem_int_lit8, .rem_int_lit16 => .rem_i32,
            .and_int, .and_int_lit8, .and_int_lit16 => .and_i32,
            .or_int, .or_int_lit8, .or_int_lit16 => .or_i32,
            .xor_int, .xor_int_lit8, .xor_int_lit16 => .xor_i32,
            else => .copy,
        },
        .int64 => switch (inst) {
            .add_long => .add_i64,
            .sub_long => .sub_i64,
            .mul_long => .mul_i64,
            .div_long => .div_i64,
            .rem_long => .rem_i64,
            else => .copy,
        },
        .float32 => .f32_op,
        .float64 => .f64_op,
        else => .unsupported,
    };
}

fn floatOperation(inst: Instruction) ?FloatOperation {
    return switch (inst) {
        .add_float, .add_double => .add,
        .sub_float, .sub_double => .sub,
        .mul_float, .mul_double => .mul,
        .div_float, .div_double => .div,
        .rem_float, .rem_double => .rem,
        .neg_float, .neg_double => .neg,
        .cmpl_float, .cmpl_double => .compare_l,
        .cmpg_float, .cmpg_double => .compare_g,
        .int_to_float => .int_to_float,
        .int_to_double => .int_to_double,
        .long_to_float => .long_to_float,
        .long_to_double => .long_to_double,
        .float_to_int => .float_to_int,
        .float_to_long => .float_to_long,
        .float_to_double => .float_to_double,
        .double_to_int => .double_to_int,
        .double_to_long => .double_to_long,
        .double_to_float => .double_to_float,
        else => null,
    };
}

fn arithmeticImmediate(inst: Instruction) i64 {
    return switch (inst) {
        .add_int_lit8,
        .rsub_int_lit8,
        .mul_int_lit8,
        .div_int_lit8,
        .rem_int_lit8,
        .and_int_lit8,
        .or_int_lit8,
        .xor_int_lit8,
        .shl_int_lit8,
        .shr_int_lit8,
        .ushr_int_lit8,
        => |literal| literal.lit,
        .add_int_lit16,
        .rsub_int_lit16,
        .mul_int_lit16,
        .div_int_lit16,
        .rem_int_lit16,
        .and_int_lit16,
        .or_int_lit16,
        .xor_int_lit16,
        => |literal| literal.lit,
        else => 0,
    };
}

fn lowerCallKind(inst: Instruction, info: typed_ir.OpInfo) Kind {
    return switch (info.devirt) {
        .direct_exact => .call_direct,
        .static_exact => .call_static,
        .quickened_virtual, .quickened_super => .call_quick,
        else => switch (inst) {
            .invoke => |invoke| switch (invoke.kind) {
                .direct => .call_direct,
                .static => .call_static,
                else => .call_virtual,
            },
            .invoke_virtual_quick, .invoke_super_quick => .call_quick,
            else => .call_virtual,
        },
    };
}

fn nextBoundsExceptionSite(inputs: Inputs, cursor: *u32) !u32 {
    const barriers = inputs.barriers orelse return error.InvalidInput;
    const base = std.math.add(u32, @intCast(barriers.resolves.len), barriers.stats.loop_epoch_guards) catch return error.InvalidInput;
    const site = std.math.add(u32, base, cursor.*) catch return error.InvalidInput;
    cursor.* = std.math.add(u32, cursor.*, 1) catch return error.InvalidInput;
    return site;
}

fn lowerOperation(
    allocator: std.mem.Allocator,
    inputs: Inputs,
    block_id: cfg.BlockId,
    op_index: usize,
    op: ssa.Operation,
    list: *std.ArrayList(Inst),
    stats: *Stats,
    bounds_exception_cursor: *u32,
) !void {
    if (opDead(inputs, block_id, op_index)) {
        stats.skipped_dead += 1;
        return;
    }

    const tinfo = typedInfo(inputs, block_id, op_index);
    const minfo = memInfo(inputs, block_id, op_index);
    const access = accessState(inputs, block_id, op_index);
    const flags = Flags{
        .null_check_elided = tinfo.null_check_elided,
        .bounds_check_elided = tinfo.bounds_check_elided,
        .forwarded = if (minfo) |m| m.forwarded_value != null else false,
        .cse = if (inputs.ssa_facts) |facts| op.defs.len > 0 and facts.values[op.defs[0]].cse_of != null else false,
    };
    if (flags.null_check_elided) stats.null_checks_elided += 1;
    if (flags.bounds_check_elided) stats.bounds_checks_elided += 1;
    if (flags.forwarded) stats.forwarded_loads += 1;

    if (foldedConstant(inputs, op)) |constant| {
        const kind: Kind = switch (constant) {
            .int => .const_i32,
            .wide => .const_i64,
        };
        const imm: i64 = switch (constant) {
            .int => |v| v,
            .wide => |v| v,
        };
        try appendInst(list, allocator, .{
            .kind = kind,
            .pc = op.pc,
            .defs = try dupeValues(allocator, op.defs),
            .uses = &.{},
            .imm = imm,
            .flags = flags,
        }, stats);
        stats.constants_materialized += 1;
        return;
    }

    switch (op.inst) {
        .nop => {},
        .const_ => |inst| {
            try appendInst(list, allocator, .{ .kind = .const_i32, .pc = op.pc, .defs = try dupeValues(allocator, op.defs), .imm = inst.value, .flags = flags }, stats);
            stats.constants_materialized += 1;
        },
        .const_wide => |inst| {
            try appendInst(list, allocator, .{ .kind = .const_i64, .pc = op.pc, .defs = try dupeValues(allocator, op.defs), .imm = inst.value, .flags = flags }, stats);
            stats.constants_materialized += 1;
        },
        .move, .move_wide, .move_object => {
            try appendInst(list, allocator, try ownOperands(allocator, .{ .kind = .copy, .pc = op.pc, .flags = flags }, op.defs, op.uses), stats);
        },
        .return_void, .return_, .return_wide, .return_object => {
            try appendInst(list, allocator, .{ .kind = .ret, .pc = op.pc, .uses = try dupeValues(allocator, op.uses), .flags = flags }, stats);
        },
        .throw_ => {
            try appendInst(list, allocator, .{ .kind = .throw_, .pc = op.pc, .uses = try dupeValues(allocator, op.uses), .flags = flags }, stats);
        },
        .goto_ => {
            try appendInst(list, allocator, .{ .kind = .branch, .pc = op.pc, .target = successorByKind(inputs.function, block_id, .branch), .flags = flags }, stats);
        },
        .if_eq, .if_ne, .if_lt, .if_ge, .if_gt, .if_le, .if_eqz, .if_nez, .if_ltz, .if_gez, .if_gtz, .if_lez => {
            try appendInst(list, allocator, .{
                .kind = .cond_branch,
                .pc = op.pc,
                .uses = try dupeValues(allocator, op.uses),
                .target = successorByKind(inputs.function, block_id, .branch),
                .false_target = successorByKind(inputs.function, block_id, .fallthrough),
                .flags = flags,
            }, stats);
        },
        .packed_switch, .sparse_switch => {
            try appendInst(list, allocator, .{ .kind = .switch_, .pc = op.pc, .uses = try dupeValues(allocator, op.uses), .flags = flags }, stats);
        },
        .aget, .aget_wide, .aget_object, .aget_boolean, .aget_byte, .aget_char, .aget_short => {
            if (!flags.null_check_elided) try appendInst(list, allocator, .{ .kind = .check_null, .pc = op.pc, .uses = try dupeValues(allocator, op.uses[0..1]) }, stats);
            if (access) |state| {
                try emitResolve(allocator, state, op.pc, list, stats);
                if (!flags.bounds_check_elided) {
                    const exception_site = try nextBoundsExceptionSite(inputs, bounds_exception_cursor);
                    try appendInst(list, allocator, .{
                        .kind = .check_bounds,
                        .pc = op.pc,
                        .uses = try dupeValues(allocator, op.uses[1..2]),
                        .address = state.address,
                        .state_handle = state.handle,
                        .reloc_token = state.token,
                        .resolve_id = state.resolve,
                        .exception_site_id = exception_site,
                    }, stats);
                    stats.bounds_exception_sites += 1;
                }
                try appendInst(list, allocator, try ownOperands(allocator, .{
                    .kind = .array_load_ptr,
                    .pc = op.pc,
                    .address = state.address,
                    .state_handle = state.handle,
                    .reloc_token = state.token,
                    .resolve_id = state.resolve,
                    .flags = flags,
                }, op.defs, op.uses[1..]), stats);
                stats.pointer_accesses += 1;
            } else {
                if (!flags.bounds_check_elided) try appendInst(list, allocator, .{ .kind = .check_bounds, .pc = op.pc, .uses = try dupeValues(allocator, op.uses[0..2]) }, stats);
                try appendInst(list, allocator, try ownOperands(allocator, .{ .kind = .array_load, .pc = op.pc, .flags = flags }, op.defs, op.uses), stats);
            }
        },
        .aput, .aput_wide, .aput_object, .aput_boolean, .aput_byte, .aput_char, .aput_short => {
            const array_and_index = op.uses[op.uses.len - 2 ..];
            const index = op.uses[op.uses.len - 1];
            if (!flags.null_check_elided) try appendInst(list, allocator, .{ .kind = .check_null, .pc = op.pc, .uses = try dupeValues(allocator, array_and_index[0..1]) }, stats);
            if (access) |state| {
                try emitResolve(allocator, state, op.pc, list, stats);
                if (!flags.bounds_check_elided) {
                    const exception_site = try nextBoundsExceptionSite(inputs, bounds_exception_cursor);
                    try appendInst(list, allocator, .{
                        .kind = .check_bounds,
                        .pc = op.pc,
                        .uses = try dupeValues(allocator, &[_]RuntimeValueId{index}),
                        .address = state.address,
                        .state_handle = state.handle,
                        .reloc_token = state.token,
                        .resolve_id = state.resolve,
                        .exception_site_id = exception_site,
                    }, stats);
                    stats.bounds_exception_sites += 1;
                }
                try emitPreWrite(allocator, state, null, index, op.pc, list, stats);
                const uses = [_]RuntimeValueId{ op.uses[0], index };
                try appendInst(list, allocator, .{
                    .kind = .array_store_ptr,
                    .pc = op.pc,
                    .uses = try dupeValues(allocator, &uses),
                    .address = state.address,
                    .state_handle = state.handle,
                    .reloc_token = state.token,
                    .resolve_id = state.resolve,
                    .flags = flags,
                }, stats);
                stats.pointer_accesses += 1;
                try emitPostWrite(allocator, state, op.uses[0..1], op.pc, list, stats);
            } else {
                if (!flags.bounds_check_elided) try appendInst(list, allocator, .{ .kind = .check_bounds, .pc = op.pc, .uses = try dupeValues(allocator, array_and_index) }, stats);
                try appendInst(list, allocator, .{ .kind = .array_store, .pc = op.pc, .uses = try dupeValues(allocator, op.uses), .flags = flags }, stats);
            }
        },
        .iget, .iget_wide, .iget_object, .iget_boolean, .iget_byte, .iget_char, .iget_short, .iget_quick, .iget_wide_quick, .iget_object_quick => {
            if (!flags.null_check_elided) try appendInst(list, allocator, .{ .kind = .check_null, .pc = op.pc, .uses = try dupeValues(allocator, op.uses[0..1]) }, stats);
            if (access) |state| {
                try emitResolve(allocator, state, op.pc, list, stats);
                try appendInst(list, allocator, try ownOperands(allocator, .{
                    .kind = .field_load_ptr,
                    .pc = op.pc,
                    .field_idx = fieldIndex(op.inst),
                    .address = state.address,
                    .state_handle = state.handle,
                    .reloc_token = state.token,
                    .resolve_id = state.resolve,
                    .flags = flags,
                }, op.defs, &.{}), stats);
                stats.pointer_accesses += 1;
            } else {
                try appendInst(list, allocator, try ownOperands(allocator, .{ .kind = .field_load, .pc = op.pc, .field_idx = fieldIndex(op.inst), .flags = flags }, op.defs, op.uses), stats);
            }
        },
        .iput, .iput_wide, .iput_object, .iput_boolean, .iput_byte, .iput_char, .iput_short, .iput_quick, .iput_wide_quick, .iput_object_quick => {
            if (access) |state| {
                try emitResolve(allocator, state, op.pc, list, stats);
                try emitPreWrite(allocator, state, fieldIndex(op.inst), null, op.pc, list, stats);
                try appendInst(list, allocator, .{
                    .kind = .field_store_ptr,
                    .pc = op.pc,
                    .uses = try dupeValues(allocator, op.uses[0 .. op.uses.len - 1]),
                    .field_idx = fieldIndex(op.inst),
                    .address = state.address,
                    .state_handle = state.handle,
                    .reloc_token = state.token,
                    .resolve_id = state.resolve,
                    .flags = flags,
                }, stats);
                stats.pointer_accesses += 1;
                try emitPostWrite(allocator, state, op.uses[0..1], op.pc, list, stats);
            } else {
                try appendInst(list, allocator, .{ .kind = .field_store, .pc = op.pc, .uses = try dupeValues(allocator, op.uses), .field_idx = fieldIndex(op.inst), .flags = flags }, stats);
            }
        },
        .sget, .sget_wide, .sget_object, .sget_boolean, .sget_byte, .sget_char, .sget_short => {
            try appendInst(list, allocator, .{ .kind = .static_load, .pc = op.pc, .defs = try dupeValues(allocator, op.defs), .field_idx = fieldIndex(op.inst), .flags = flags }, stats);
        },
        .sput_object => {
            const plan = inputs.barriers orelse return error.InvalidInput;
            const barrier = plan.ops[block_id][op_index];
            const index = fieldIndex(op.inst) orelse return error.InvalidInput;
            if (op.uses.len != 1) return error.InvalidInput;
            try emitStaticReferenceStore(allocator, index, op.uses[0], barrier.pre_write, barrier.post_write, op.pc, flags, list, stats);
        },
        .sput, .sput_wide, .sput_boolean, .sput_byte, .sput_char, .sput_short => {
            try appendInst(list, allocator, .{ .kind = .static_store, .pc = op.pc, .uses = try dupeValues(allocator, op.uses), .field_idx = fieldIndex(op.inst), .flags = flags }, stats);
        },
        .invoke, .invoke_virtual_quick, .invoke_super_quick => {
            const kind = lowerCallKind(op.inst, tinfo);
            if (kind == .call_direct or kind == .call_static or kind == .call_quick) stats.direct_calls += 1;
            if (access) |state| {
                try emitResolve(allocator, state, op.pc, list, stats);
                const call = try ownOperands(allocator, .{
                    .kind = kind,
                    .pc = op.pc,
                    .address = state.address,
                    .state_handle = state.handle,
                    .reloc_token = state.token,
                    .resolve_id = state.resolve,
                    .flags = flags,
                }, op.defs, op.uses);
                try appendInst(list, allocator, call, stats);
            } else {
                try appendInst(list, allocator, try ownOperands(allocator, .{ .kind = kind, .pc = op.pc, .flags = flags }, op.defs, op.uses), stats);
            }
        },
        else => {
            const kind = lowerArithmetic(op.inst, tinfo.lowering);
            try appendInst(list, allocator, try ownOperands(allocator, .{
                .kind = kind,
                .pc = op.pc,
                .imm = arithmeticImmediate(op.inst),
                .float_op = if (kind == .f32_op or kind == .f64_op) floatOperation(op.inst) else null,
                .field_idx = fieldIndex(op.inst),
                .flags = flags,
            }, op.defs, op.uses), stats);
        },
    }
}

pub fn build(allocator: std.mem.Allocator, inputs: Inputs) Error!Function {
    inputs.function.verify() catch |err| switch (err) {
        error.OutOfMemory => return error.OutOfMemory,
        else => return error.InvalidInput,
    };
    inputs.types.verify() catch |err| switch (err) {
        error.OutOfMemory => return error.OutOfMemory,
        else => return error.InvalidInput,
    };
    inputs.typed.verify() catch |err| switch (err) {
        error.OutOfMemory => return error.OutOfMemory,
        else => return error.InvalidInput,
    };
    if (inputs.types.source != inputs.function or inputs.typed.source != inputs.function) return error.InvalidInput;
    if (inputs.ssa_facts) |facts| if (facts.function != inputs.function) return error.InvalidInput;
    if (inputs.memory) |memory| if (memory.function != inputs.function) return error.InvalidInput;
    if (inputs.barriers) |barriers| {
        barriers.verify() catch |err| switch (err) {
            error.OutOfMemory => return error.OutOfMemory,
            else => return error.InvalidInput,
        };
        if (barriers.function != inputs.function or barriers.types != inputs.types) return error.InvalidInput;
    }

    const resolve_count = if (inputs.barriers) |barriers| barriers.resolves.len else 0;
    if (resolve_count > std.math.maxInt(RuntimeValueId) - inputs.function.values.len) return error.InvalidInput;
    const runtime_value_count = inputs.function.values.len + resolve_count;

    const value_types = try allocator.alloc(typedir.Type, runtime_value_count);
    errdefer allocator.free(value_types);
    const runtime_values = try allocator.alloc(RuntimeValueClass, runtime_value_count);
    errdefer allocator.free(runtime_values);
    const resolve_values = try allocator.alloc(RuntimeValueId, resolve_count);
    errdefer allocator.free(resolve_values);
    for (inputs.function.values, 0..) |value, i| {
        const ty = inputs.types.typeOf(value.id) orelse .unknown;
        value_types[i] = ty;
        runtime_values[i] = .{ .dalvik = .{ .value = value.id, .ty = ty, .gc_root = ty == .object } };
    }
    // Directional type inference intentionally leaves write-only parameters
    // unknown. Object-store opcodes nevertheless prove that their first use is
    // a Handle, so keep it visible to root maps and native barrier lowering.
    for (inputs.function.blocks) |block| {
        for (block.ops) |op| switch (op.inst) {
            .iput_object,
            .iput_object_quick,
            .aput_object,
            .sput_object,
            => {
                if (op.uses.len == 0) return error.InvalidInput;
                const value = op.uses[0];
                if (value >= inputs.function.values.len) return error.InvalidInput;
                switch (runtime_values[value]) {
                    .dalvik => |*stored| stored.gc_root = true,
                    .derived_ptr => return error.InvalidInput,
                }
            },
            else => {},
        };
    }
    if (inputs.barriers) |barriers| {
        for (barriers.resolves, 0..) |resolve, i| {
            const value: RuntimeValueId = @intCast(inputs.function.values.len + i);
            resolve_values[i] = value;
            value_types[value] = .long;
            runtime_values[value] = .{ .derived_ptr = .{
                .handle = resolve.handle,
                .token = resolve.token,
                .resolve = @intCast(i),
            } };
            const handle_class = &runtime_values[resolve.handle];
            switch (handle_class.*) {
                .dalvik => |*handle| handle.gc_root = true,
                .derived_ptr => return error.InvalidInput,
            }
        }
    }

    const blocks = try allocator.alloc(Block, inputs.function.blocks.len);
    errdefer allocator.free(blocks);
    var built_blocks: usize = 0;
    errdefer {
        for (blocks[0..built_blocks]) |block| {
            for (block.insts) |inst| {
                allocator.free(inst.defs);
                allocator.free(inst.uses);
            }
            allocator.free(block.insts);
        }
    }

    var stats: Stats = .{ .runtime_values = @intCast(runtime_value_count) };
    var bounds_exception_cursor: u32 = 0;
    for (inputs.function.blocks, 0..) |block, block_i| {
        var list: std.ArrayList(Inst) = .empty;
        errdefer {
            for (list.items) |inst| {
                allocator.free(inst.defs);
                allocator.free(inst.uses);
            }
            list.deinit(allocator);
        }

        for (block.phis) |phi| {
            try appendInst(&list, allocator, .{
                .kind = .phi,
                .defs = try dupeValues(allocator, &[_]ValueId{phi.dest}),
            }, &stats);
        }

        var emitted_hoists = false;
        for (block.ops, 0..) |op, op_i| {
            if (!emitted_hoists and isControlTransfer(op.inst)) {
                try emitHoistedResolves(allocator, inputs, @intCast(block_i), &list, &stats);
                try emitLoopEpochGuards(allocator, inputs, @intCast(block_i), &list, &stats);
                emitted_hoists = true;
            }
            try lowerOperation(allocator, inputs, @intCast(block_i), op_i, op, &list, &stats, &bounds_exception_cursor);
        }
        if (!emitted_hoists) {
            try emitHoistedResolves(allocator, inputs, @intCast(block_i), &list, &stats);
            try emitLoopEpochGuards(allocator, inputs, @intCast(block_i), &list, &stats);
        }

        blocks[block_i] = .{ .id = @intCast(block_i), .insts = try list.toOwnedSlice(allocator) };
        built_blocks += 1;
    }
    if (bounds_exception_cursor != stats.bounds_exception_sites) return error.InvalidInput;

    return .{
        .allocator = allocator,
        .source = inputs.function,
        .blocks = blocks,
        .value_types = value_types,
        .runtime_values = runtime_values,
        .resolve_values = resolve_values,
        .barriers = inputs.barriers,
        .stats = stats,
    };
}

fn initPipeline(
    allocator: std.mem.Allocator,
    insts: []const Instruction,
    graph: *cfg.Graph,
    tree: *dom.Tree,
    function: *ssa.Function,
    types: *typedir.Function,
    facts: *ssa_phase.Result,
    opt: *optimizer.Result,
    typed: *typed_ir.Function,
    memory: *memory_phase.Result,
) !void {
    graph.* = try cfg.build(allocator, insts);
    errdefer graph.deinit();
    tree.* = try dom.build(allocator, graph);
    errdefer tree.deinit();
    function.* = try ssa.build(allocator, graph, tree);
    errdefer function.deinit();
    types.* = try typedir.build(allocator, function);
    errdefer types.deinit();
    facts.* = try ssa_phase.run(allocator, function);
    errdefer facts.deinit();
    opt.* = try optimizer.run(allocator, function, .{});
    errdefer opt.deinit();
    typed.* = try typed_ir.build(allocator, function, types, opt);
    errdefer typed.deinit();
    memory.* = try memory_phase.run(allocator, function, types);
}

test "lowering emits typed arithmetic and materialized constants" {
    const insts = [_]Instruction{
        .{ .const_ = .{ .dest = 0, .value = 1 } },
        .{ .const_ = .{ .dest = 1, .value = 2 } },
        .{ .add_int = .{ .dest = 2, .src1 = 0, .src2 = 1 } },
        .{ .return_ = .{ .src = 2 } },
    };
    var graph: cfg.Graph = undefined;
    var tree: dom.Tree = undefined;
    var function: ssa.Function = undefined;
    var types: typedir.Function = undefined;
    var facts: ssa_phase.Result = undefined;
    var opt: optimizer.Result = undefined;
    var typed: typed_ir.Function = undefined;
    var memory: memory_phase.Result = undefined;
    try initPipeline(std.testing.allocator, &insts, &graph, &tree, &function, &types, &facts, &opt, &typed, &memory);
    defer memory.deinit();
    defer typed.deinit();
    defer opt.deinit();
    defer facts.deinit();
    defer types.deinit();
    defer function.deinit();
    defer tree.deinit();
    defer graph.deinit();

    var lowered = try build(std.testing.allocator, .{ .function = &function, .types = &types, .typed = &typed, .ssa_facts = &facts, .memory = &memory });
    defer lowered.deinit();
    try lowered.verify();
    try std.testing.expect(lowered.stats.constants_materialized >= 3);
}

test "lowering preserves signed arithmetic literal immediates" {
    try std.testing.expectEqual(@as(i64, -1), arithmeticImmediate(.{ .add_int_lit8 = .{ .dest = 1, .src = 0, .lit = -1 } }));
    try std.testing.expectEqual(@as(i64, -30_000), arithmeticImmediate(.{ .xor_int_lit16 = .{ .dest = 1, .src = 0, .lit = -30_000 } }));
    try std.testing.expectEqual(@as(i64, 0), arithmeticImmediate(.{ .add_int = .{ .dest = 2, .src1 = 0, .src2 = 1 } }));
}

test "lowering skips dead pure instructions" {
    const insts = [_]Instruction{
        .{ .const_ = .{ .dest = 0, .value = 1 } },
        .{ .const_ = .{ .dest = 1, .value = 2 } },
        .{ .add_int = .{ .dest = 2, .src1 = 0, .src2 = 1 } },
        .return_void,
    };
    var graph: cfg.Graph = undefined;
    var tree: dom.Tree = undefined;
    var function: ssa.Function = undefined;
    var types: typedir.Function = undefined;
    var facts: ssa_phase.Result = undefined;
    var opt: optimizer.Result = undefined;
    var typed: typed_ir.Function = undefined;
    var memory: memory_phase.Result = undefined;
    try initPipeline(std.testing.allocator, &insts, &graph, &tree, &function, &types, &facts, &opt, &typed, &memory);
    defer memory.deinit();
    defer typed.deinit();
    defer opt.deinit();
    defer facts.deinit();
    defer types.deinit();
    defer function.deinit();
    defer tree.deinit();
    defer graph.deinit();

    var lowered = try build(std.testing.allocator, .{ .function = &function, .types = &types, .typed = &typed, .ssa_facts = &facts, .memory = &memory });
    defer lowered.deinit();
    try std.testing.expect(lowered.stats.skipped_dead >= 1);
}

test "lowering emits elided array checks from typed ir" {
    const insts = [_]Instruction{
        .{ .const_ = .{ .dest = 0, .value = 4 } },
        .{ .new_array = .{ .dest = 1, .size = 0, .type_idx = 1 } },
        .{ .const_ = .{ .dest = 2, .value = 2 } },
        .{ .aget = .{ .dest_or_src = 3, .array = 1, .index = 2 } },
        .{ .return_ = .{ .src = 3 } },
    };
    var graph: cfg.Graph = undefined;
    var tree: dom.Tree = undefined;
    var function: ssa.Function = undefined;
    var types: typedir.Function = undefined;
    var facts: ssa_phase.Result = undefined;
    var opt: optimizer.Result = undefined;
    var typed: typed_ir.Function = undefined;
    var memory: memory_phase.Result = undefined;
    try initPipeline(std.testing.allocator, &insts, &graph, &tree, &function, &types, &facts, &opt, &typed, &memory);
    defer memory.deinit();
    defer typed.deinit();
    defer opt.deinit();
    defer facts.deinit();
    defer types.deinit();
    defer function.deinit();
    defer tree.deinit();
    defer graph.deinit();

    var lowered = try build(std.testing.allocator, .{ .function = &function, .types = &types, .typed = &typed, .ssa_facts = &facts, .memory = &memory });
    defer lowered.deinit();
    try lowered.verify();
    try std.testing.expect(lowered.stats.null_checks_elided >= 1);
    try std.testing.expect(lowered.stats.bounds_checks_elided >= 1);
}

test "lowering preserves devirtualized direct call hints" {
    var invoke = instmod.Invoke{
        .class_name = "LExample;",
        .method_name = "f",
        .signature = "()V",
        .dest = null,
        .kind = .direct,
    };
    const insts = [_]Instruction{
        .{ .invoke = &invoke },
        .return_void,
    };
    var graph: cfg.Graph = undefined;
    var tree: dom.Tree = undefined;
    var function: ssa.Function = undefined;
    var types: typedir.Function = undefined;
    var facts: ssa_phase.Result = undefined;
    var opt: optimizer.Result = undefined;
    var typed: typed_ir.Function = undefined;
    var memory: memory_phase.Result = undefined;
    try initPipeline(std.testing.allocator, &insts, &graph, &tree, &function, &types, &facts, &opt, &typed, &memory);
    defer memory.deinit();
    defer typed.deinit();
    defer opt.deinit();
    defer facts.deinit();
    defer types.deinit();
    defer function.deinit();
    defer tree.deinit();
    defer graph.deinit();

    var lowered = try build(std.testing.allocator, .{ .function = &function, .types = &types, .typed = &typed, .ssa_facts = &facts, .memory = &memory });
    defer lowered.deinit();
    try lowered.verify();
    try std.testing.expectEqual(Kind.call_direct, lowered.blocks[graph.entry].insts[0].kind);
    try std.testing.expectEqual(@as(u32, 1), lowered.stats.direct_calls);
}

test "lowering carries memory forwarding flags" {
    const insts = [_]Instruction{
        .{ .new_instance = .{ .dest = 0, .type_idx = 1 } },
        .{ .const_ = .{ .dest = 1, .value = 7 } },
        .{ .iput = .{ .field_idx = 10, .dest_or_src = 1, .obj = 0 } },
        .{ .iget = .{ .field_idx = 10, .dest_or_src = 2, .obj = 0 } },
        .return_void,
    };
    var graph: cfg.Graph = undefined;
    var tree: dom.Tree = undefined;
    var function: ssa.Function = undefined;
    var types: typedir.Function = undefined;
    var facts: ssa_phase.Result = undefined;
    var opt: optimizer.Result = undefined;
    var typed: typed_ir.Function = undefined;
    var memory: memory_phase.Result = undefined;
    try initPipeline(std.testing.allocator, &insts, &graph, &tree, &function, &types, &facts, &opt, &typed, &memory);
    defer memory.deinit();
    defer typed.deinit();
    defer opt.deinit();
    defer facts.deinit();
    defer types.deinit();
    defer function.deinit();
    defer tree.deinit();
    defer graph.deinit();

    var lowered = try build(std.testing.allocator, .{ .function = &function, .types = &types, .typed = &typed, .ssa_facts = &facts, .memory = &memory });
    defer lowered.deinit();
    try lowered.verify();
    try std.testing.expect(lowered.stats.forwarded_loads >= 1);
}

test "lowering print helper emits stable summary" {
    const insts = [_]Instruction{
        .{ .const_ = .{ .dest = 0, .value = 1 } },
        .return_void,
    };
    var graph: cfg.Graph = undefined;
    var tree: dom.Tree = undefined;
    var function: ssa.Function = undefined;
    var types: typedir.Function = undefined;
    var facts: ssa_phase.Result = undefined;
    var opt: optimizer.Result = undefined;
    var typed: typed_ir.Function = undefined;
    var memory: memory_phase.Result = undefined;
    try initPipeline(std.testing.allocator, &insts, &graph, &tree, &function, &types, &facts, &opt, &typed, &memory);
    defer memory.deinit();
    defer typed.deinit();
    defer opt.deinit();
    defer facts.deinit();
    defer types.deinit();
    defer function.deinit();
    defer tree.deinit();
    defer graph.deinit();

    var lowered = try build(std.testing.allocator, .{ .function = &function, .types = &types, .typed = &typed, .ssa_facts = &facts, .memory = &memory });
    defer lowered.deinit();
    var buf: [512]u8 = undefined;
    var stream = std.Io.Writer.fixed(&buf);
    try lowered.print(&stream);
    const output = stream.buffered();
    try std.testing.expect(std.mem.indexOf(u8, output, "lowering blocks=") != null);
    try std.testing.expect(std.mem.indexOf(u8, output, "ret") != null);
}
