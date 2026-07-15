//! Early SSA optimization pipeline.
//!
//! This module is intentionally conservative: it records optimization facts
//! over SSA without rewriting the function yet. Later lowering can consume
//! these facts for constant materialization, copy forwarding, and dead-code
//! skipping while the original SSA remains verifier-friendly.

const std = @import("std");
const cfg = @import("cfg");
const cfg_rewrite = @import("cfg_rewrite");
const dom = @import("dominator");
const ssa = @import("ssa");
const ssa_phase = @import("ssa_phase");
const typedir = @import("typedir");
const typed_ir = @import("typed_ir");
const loop_phase = @import("loop_phase");
const memory_phase = @import("memory_phase");
const barrier_phase = @import("barrier_phase");
const lowering = @import("lowering");
const machine_bridge = @import("machine_bridge");
const derived_verify = @import("derived_verify");
const instmod = @import("instructions");
const Instruction = instmod.Instruction;
const TryBlock = instmod.TryBlock;

pub const Error = error{
    InvalidSsa,
    OutOfMemory,
};

pub const ValueType = enum(u8) {
    unknown,
    int,
    long,
    float,
    double,
    object,
};

pub const Constant = union(enum) {
    int: i32,
    wide: i64,

    pub fn eql(a: Constant, b: Constant) bool {
        return switch (a) {
            .int => |av| switch (b) {
                .int => |bv| av == bv,
                else => false,
            },
            .wide => |av| switch (b) {
                .wide => |bv| av == bv,
                else => false,
            },
        };
    }

    pub fn print(self: Constant, writer: anytype) !void {
        switch (self) {
            .int => |value| try writer.print("i32({d})", .{value}),
            .wide => |value| try writer.print("i64({d})", .{value}),
        }
    }
};

pub const ValueFact = struct {
    ty: ValueType = .unknown,
    constant: ?Constant = null,
    copy_of: ?ssa.ValueId = null,
    live: bool = false,
};

pub const OpState = struct {
    live: bool = false,
    side_effect: bool = false,
    folded: ?Constant = null,
};

pub const Stats = struct {
    constants: u32 = 0,
    copies: u32 = 0,
    live_ops: u32 = 0,
    dead_ops: u32 = 0,
};

pub const Options = struct {
    constant_folding: bool = true,
    copy_propagation: bool = true,
    dead_code: bool = true,
};

pub const OptimizeOptions = struct {
    cfg_options: cfg.Options = .{},
    rewrite_options: cfg_rewrite.Options = .{},
    facts: Options = .{},
};

pub const PipelineStats = struct {
    original_blocks: u32 = 0,
    rewritten_blocks: u32 = 0,
    lowered_insts: u32 = 0,
    cfg_removed_blocks: u32 = 0,
    cfg_merged_blocks: u32 = 0,
    ssa_constants: u32 = 0,
    ssa_dead_ops: u32 = 0,
    loop_count: u32 = 0,
    memory_forwarded: u32 = 0,
    handle_resolves: u32 = 0,
    handle_resolve_reuses: u32 = 0,
    loop_resolves_hoisted: u32 = 0,
    derived_address_uses: u32 = 0,
    machine_insts: u32 = 0,
    machine_edge_moves: u32 = 0,
};

pub const OptimizedFunction = struct {
    allocator: std.mem.Allocator,
    original_cfg: cfg.Graph,
    rewritten_cfg: cfg_rewrite.Rewrite,
    tree: dom.Tree,
    function: ssa.Function,
    ssa_facts: ssa_phase.Result,
    facts: Result,
    types: typedir.Function,
    typed: typed_ir.Function,
    loops: loop_phase.Result,
    memory: memory_phase.Result,
    barriers: barrier_phase.Result,
    lowered: lowering.Function,
    machine: machine_bridge.Function,
    derived: derived_verify.Result,
    stats: PipelineStats,

    pub fn deinit(self: *OptimizedFunction) void {
        self.derived.deinit();
        self.machine.deinit();
        self.lowered.deinit();
        self.barriers.deinit();
        self.memory.deinit();
        self.loops.deinit();
        self.typed.deinit();
        self.types.deinit();
        self.facts.deinit();
        self.ssa_facts.deinit();
        self.function.deinit();
        self.tree.deinit();
        self.rewritten_cfg.deinit();
        self.original_cfg.deinit();
        const allocator = self.allocator;
        allocator.destroy(self);
    }

    pub fn print(self: *const OptimizedFunction, writer: anytype) !void {
        try writer.print(
            "optimizer_pipeline original_blocks={d} rewritten_blocks={d} lowered_insts={d} machine_insts={d} machine_edge_moves={d} cfg_removed={d} cfg_merged={d} ssa_constants={d} ssa_dead_ops={d} loops={d} memory_forwarded={d} handle_resolves={d} handle_resolve_reuses={d} loop_resolve_hoists={d} derived_address_uses={d}\n",
            .{
                self.stats.original_blocks,
                self.stats.rewritten_blocks,
                self.stats.lowered_insts,
                self.stats.machine_insts,
                self.stats.machine_edge_moves,
                self.stats.cfg_removed_blocks,
                self.stats.cfg_merged_blocks,
                self.stats.ssa_constants,
                self.stats.ssa_dead_ops,
                self.stats.loop_count,
                self.stats.memory_forwarded,
                self.stats.handle_resolves,
                self.stats.handle_resolve_reuses,
                self.stats.loop_resolves_hoisted,
                self.stats.derived_address_uses,
            },
        );
        try writer.print("\n-- cfg rewrite --\n", .{});
        try self.rewritten_cfg.print(writer);
        try writer.print("\n-- ssa facts --\n", .{});
        try self.ssa_facts.print(writer);
        try writer.print("\n-- types --\n", .{});
        try self.types.print(writer);
        try writer.print("\n-- typed ir --\n", .{});
        try self.typed.print(writer);
        try writer.print("\n-- loops --\n", .{});
        try self.loops.print(writer);
        try writer.print("\n-- memory --\n", .{});
        try self.memory.print(writer);
        try writer.print("\n-- barriers --\n", .{});
        try self.barriers.print(writer);
        try writer.print("\n-- lowering --\n", .{});
        try self.lowered.print(writer);
        try writer.print("\n-- machine bridge --\n", .{});
        try self.machine.print(writer);
        try writer.print("\n-- derived pointer verification --\n", .{});
        try self.derived.print(writer);
    }
};

pub const Result = struct {
    allocator: std.mem.Allocator,
    function: *const ssa.Function,
    facts: []ValueFact,
    op_states: [][]OpState,
    stats: Stats,

    pub fn deinit(self: *Result) void {
        for (self.op_states) |states| self.allocator.free(states);
        self.allocator.free(self.op_states);
        self.allocator.free(self.facts);
        self.* = undefined;
    }

    pub inline fn fact(self: *const Result, value: ssa.ValueId) ?ValueFact {
        if (value >= self.facts.len) return null;
        return self.facts[value];
    }

    pub fn print(self: *const Result, writer: anytype) !void {
        try writer.print(
            "optimizer values={d} live_ops={d} dead_ops={d} constants={d} copies={d}\n",
            .{ self.facts.len, self.stats.live_ops, self.stats.dead_ops, self.stats.constants, self.stats.copies },
        );

        for (self.function.graph.rpo) |block_id| {
            const block = self.function.blocks[block_id];
            try writer.print("b{d}\n", .{block_id});
            for (block.phis) |phi| {
                const fact_value = self.facts[phi.dest];
                try writer.print("  v{d} phi r{d} live={}\n", .{ phi.dest, phi.reg, fact_value.live });
            }
            for (block.ops, 0..) |op, i| {
                const state = self.op_states[block_id][i];
                try writer.print("  pc{d} {s} live={} side_effect={}", .{ op.pc, @tagName(op.inst), state.live, state.side_effect });
                if (state.folded) |constant| {
                    try writer.print(" folded=", .{});
                    try constant.print(writer);
                }
                try writer.print("\n", .{});
            }
        }

        try writer.print("values:\n", .{});
        for (self.facts, 0..) |fact_value, i| {
            try writer.print("  v{d} {s} live={}", .{ i, @tagName(fact_value.ty), fact_value.live });
            if (fact_value.copy_of) |copy| try writer.print(" copy=v{d}", .{copy});
            if (fact_value.constant) |constant| {
                try writer.print(" const=", .{});
                try constant.print(writer);
            }
            try writer.print("\n", .{});
        }
    }
};

fn sameFact(a: ValueFact, b: ValueFact) bool {
    if (a.ty != b.ty or a.copy_of != b.copy_of or a.live != b.live) return false;
    if (a.constant == null and b.constant == null) return true;
    if (a.constant == null or b.constant == null) return false;
    return a.constant.?.eql(b.constant.?);
}

fn constForValue(facts: []const ValueFact, value: ssa.ValueId) ?Constant {
    if (value >= facts.len) return null;
    var current = value;
    var guard: usize = 0;
    while (guard < facts.len) : (guard += 1) {
        const fact_value = facts[current];
        if (fact_value.constant) |constant| return constant;
        if (fact_value.copy_of) |copy| {
            if (copy >= facts.len or copy == current) return null;
            current = copy;
            continue;
        }
        return null;
    }
    return null;
}

fn canonicalCopy(facts: []const ValueFact, value: ssa.ValueId) ssa.ValueId {
    var current = value;
    var guard: usize = 0;
    while (current < facts.len and guard < facts.len) : (guard += 1) {
        const copy = facts[current].copy_of orelse return current;
        if (copy == current) return current;
        current = copy;
    }
    return value;
}

fn mergePhiConstant(facts: []const ValueFact, phi: ssa.Phi) ?Constant {
    if (phi.incoming.len == 0) return null;
    const first = constForValue(facts, phi.incoming[0].value) orelse return null;
    for (phi.incoming[1..]) |incoming| {
        const other = constForValue(facts, incoming.value) orelse return null;
        if (!first.eql(other)) return null;
    }
    return first;
}

fn intConst(facts: []const ValueFact, value: ssa.ValueId) ?i32 {
    const constant = constForValue(facts, value) orelse return null;
    return switch (constant) {
        .int => |v| v,
        else => null,
    };
}

fn wideConst(facts: []const ValueFact, value: ssa.ValueId) ?i64 {
    const constant = constForValue(facts, value) orelse return null;
    return switch (constant) {
        .wide => |v| v,
        else => null,
    };
}

fn foldIntBin(inst: Instruction, a: i32, b: i32) ?i32 {
    return switch (inst) {
        .add_int => a +% b,
        .sub_int => a -% b,
        .mul_int => a *% b,
        .and_int => @bitCast(@as(u32, @bitCast(a)) & @as(u32, @bitCast(b))),
        .or_int => @bitCast(@as(u32, @bitCast(a)) | @as(u32, @bitCast(b))),
        .xor_int => @bitCast(@as(u32, @bitCast(a)) ^ @as(u32, @bitCast(b))),
        .shl_int => a << @as(u5, @truncate(@as(u32, @bitCast(b)))),
        .shr_int => a >> @as(u5, @truncate(@as(u32, @bitCast(b)))),
        .ushr_int => @bitCast(@as(u32, @bitCast(a)) >> @as(u5, @truncate(@as(u32, @bitCast(b))))),
        .div_int => if (b == 0) null else if (a == std.math.minInt(i32) and b == -1) a else @divTrunc(a, b),
        .rem_int => if (b == 0) null else if (a == std.math.minInt(i32) and b == -1) 0 else @rem(a, b),
        else => null,
    };
}

fn foldWideBin(inst: Instruction, a: i64, b: i64) ?i64 {
    return switch (inst) {
        .add_long => a +% b,
        .sub_long => a -% b,
        .mul_long => a *% b,
        .and_long => @bitCast(@as(u64, @bitCast(a)) & @as(u64, @bitCast(b))),
        .or_long => @bitCast(@as(u64, @bitCast(a)) | @as(u64, @bitCast(b))),
        .xor_long => @bitCast(@as(u64, @bitCast(a)) ^ @as(u64, @bitCast(b))),
        .div_long => if (b == 0) null else if (a == std.math.minInt(i64) and b == -1) a else @divTrunc(a, b),
        .rem_long => if (b == 0) null else if (a == std.math.minInt(i64) and b == -1) 0 else @rem(a, b),
        else => null,
    };
}

fn foldLit16(inst: Instruction, a: i32, lit: i16) ?i32 {
    const b: i32 = lit;
    return switch (inst) {
        .add_int_lit16 => a +% b,
        .rsub_int_lit16 => b -% a,
        .mul_int_lit16 => a *% b,
        .and_int_lit16 => @bitCast(@as(u32, @bitCast(a)) & @as(u32, @bitCast(b))),
        .or_int_lit16 => @bitCast(@as(u32, @bitCast(a)) | @as(u32, @bitCast(b))),
        .xor_int_lit16 => @bitCast(@as(u32, @bitCast(a)) ^ @as(u32, @bitCast(b))),
        .div_int_lit16 => if (b == 0) null else if (a == std.math.minInt(i32) and b == -1) a else @divTrunc(a, b),
        .rem_int_lit16 => if (b == 0) null else if (a == std.math.minInt(i32) and b == -1) 0 else @rem(a, b),
        else => null,
    };
}

fn foldLit8(inst: Instruction, a: i32, lit: i8) ?i32 {
    const b: i32 = lit;
    return switch (inst) {
        .add_int_lit8 => a +% b,
        .rsub_int_lit8 => b -% a,
        .mul_int_lit8 => a *% b,
        .and_int_lit8 => @bitCast(@as(u32, @bitCast(a)) & @as(u32, @bitCast(b))),
        .or_int_lit8 => @bitCast(@as(u32, @bitCast(a)) | @as(u32, @bitCast(b))),
        .xor_int_lit8 => @bitCast(@as(u32, @bitCast(a)) ^ @as(u32, @bitCast(b))),
        .shl_int_lit8 => a << @as(u5, @truncate(@as(u8, @bitCast(lit)))),
        .shr_int_lit8 => a >> @as(u5, @truncate(@as(u8, @bitCast(lit)))),
        .ushr_int_lit8 => @bitCast(@as(u32, @bitCast(a)) >> @as(u5, @truncate(@as(u8, @bitCast(lit))))),
        .div_int_lit8 => if (b == 0) null else if (a == std.math.minInt(i32) and b == -1) a else @divTrunc(a, b),
        .rem_int_lit8 => if (b == 0) null else if (a == std.math.minInt(i32) and b == -1) 0 else @rem(a, b),
        else => null,
    };
}

fn sideEffecting(inst: Instruction) bool {
    return switch (inst) {
        .nop,
        .move,
        .move_wide,
        .move_object,
        .const_,
        .const_wide,
        .const_string,
        .const_class,
        .const_method_handle,
        .const_method_type,
        .neg_int,
        .not_int,
        .neg_long,
        .not_long,
        .neg_float,
        .neg_double,
        .int_to_long,
        .int_to_float,
        .int_to_double,
        .long_to_int,
        .long_to_float,
        .long_to_double,
        .float_to_int,
        .float_to_long,
        .float_to_double,
        .double_to_int,
        .double_to_long,
        .double_to_float,
        .int_to_byte,
        .int_to_char,
        .int_to_short,
        .add_int,
        .sub_int,
        .mul_int,
        .and_int,
        .or_int,
        .xor_int,
        .shl_int,
        .shr_int,
        .ushr_int,
        .add_long,
        .sub_long,
        .mul_long,
        .and_long,
        .or_long,
        .xor_long,
        .shl_long,
        .shr_long,
        .ushr_long,
        .add_float,
        .sub_float,
        .mul_float,
        .div_float,
        .rem_float,
        .add_double,
        .sub_double,
        .mul_double,
        .div_double,
        .rem_double,
        .add_int_lit16,
        .rsub_int_lit16,
        .mul_int_lit16,
        .and_int_lit16,
        .or_int_lit16,
        .xor_int_lit16,
        .add_int_lit8,
        .rsub_int_lit8,
        .mul_int_lit8,
        .and_int_lit8,
        .or_int_lit8,
        .xor_int_lit8,
        .shl_int_lit8,
        .shr_int_lit8,
        .ushr_int_lit8,
        .cmpl_float,
        .cmpg_float,
        .cmpl_double,
        .cmpg_double,
        .cmp_long,
        => false,
        else => true,
    };
}

fn setDefFact(facts: []ValueFact, op: ssa.Operation, fact: ValueFact) void {
    if (op.defs.len == 0) return;
    facts[op.defs[0]] = fact;
}

fn deriveOperationFacts(facts: []ValueFact, op: ssa.Operation) ?Constant {
    switch (op.inst) {
        .const_ => |inst| {
            setDefFact(facts, op, .{ .ty = .int, .constant = .{ .int = inst.value } });
            return .{ .int = inst.value };
        },
        .const_wide => |inst| {
            if (op.defs.len >= 1) facts[op.defs[0]] = .{ .ty = .long, .constant = .{ .wide = inst.value } };
            if (op.defs.len >= 2) facts[op.defs[1]] = .{ .ty = .long, .constant = .{ .wide = inst.value } };
            return .{ .wide = inst.value };
        },
        .move, .move_object => {
            if (op.uses.len >= 1 and op.defs.len >= 1) {
                const source = canonicalCopy(facts, op.uses[0]);
                facts[op.defs[0]].copy_of = source;
                facts[op.defs[0]].ty = facts[source].ty;
                facts[op.defs[0]].constant = constForValue(facts, source);
            }
            return if (op.defs.len >= 1) facts[op.defs[0]].constant else null;
        },
        .move_wide => {
            var folded: ?Constant = null;
            for (op.defs, 0..) |def, i| {
                if (i >= op.uses.len) break;
                const source = canonicalCopy(facts, op.uses[i]);
                facts[def].copy_of = source;
                facts[def].ty = facts[source].ty;
                facts[def].constant = constForValue(facts, source);
                if (i == 0) folded = facts[def].constant;
            }
            return folded;
        },
        .add_int,
        .sub_int,
        .mul_int,
        .div_int,
        .rem_int,
        .and_int,
        .or_int,
        .xor_int,
        .shl_int,
        .shr_int,
        .ushr_int,
        => {
            if (op.uses.len < 2 or op.defs.len < 1) return null;
            const a = intConst(facts, op.uses[0]) orelse return null;
            const b = intConst(facts, op.uses[1]) orelse return null;
            const folded = foldIntBin(op.inst, a, b) orelse return null;
            facts[op.defs[0]] = .{ .ty = .int, .constant = .{ .int = folded } };
            return .{ .int = folded };
        },
        .add_long,
        .sub_long,
        .mul_long,
        .div_long,
        .rem_long,
        .and_long,
        .or_long,
        .xor_long,
        => {
            if (op.uses.len < 3 or op.defs.len < 1) return null;
            const a = wideConst(facts, op.uses[0]) orelse return null;
            const b = wideConst(facts, op.uses[2]) orelse return null;
            const folded = foldWideBin(op.inst, a, b) orelse return null;
            for (op.defs) |def| facts[def] = .{ .ty = .long, .constant = .{ .wide = folded } };
            return .{ .wide = folded };
        },
        .add_int_lit16,
        .rsub_int_lit16,
        .mul_int_lit16,
        .div_int_lit16,
        .rem_int_lit16,
        .and_int_lit16,
        .or_int_lit16,
        .xor_int_lit16,
        => |inst| {
            if (op.uses.len < 1 or op.defs.len < 1) return null;
            const a = intConst(facts, op.uses[0]) orelse return null;
            const folded = foldLit16(op.inst, a, inst.lit) orelse return null;
            facts[op.defs[0]] = .{ .ty = .int, .constant = .{ .int = folded } };
            return .{ .int = folded };
        },
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
        => |inst| {
            if (op.uses.len < 1 or op.defs.len < 1) return null;
            const a = intConst(facts, op.uses[0]) orelse return null;
            const folded = foldLit8(op.inst, a, inst.lit) orelse return null;
            facts[op.defs[0]] = .{ .ty = .int, .constant = .{ .int = folded } };
            return .{ .int = folded };
        },
        else => {
            return null;
        },
    }
}

fn markValueLive(
    facts: []ValueFact,
    work: *std.ArrayList(ssa.ValueId),
    allocator: std.mem.Allocator,
    value: ssa.ValueId,
) !void {
    if (value >= facts.len or facts[value].live) return;
    facts[value].live = true;
    try work.append(allocator, value);
}

const PhiRef = struct { block: cfg.BlockId, index: usize };

fn computeLiveness(
    allocator: std.mem.Allocator,
    function: *const ssa.Function,
    facts: []ValueFact,
    op_states: [][]OpState,
) !void {
    var def_to_op = try allocator.alloc(?ssa.OpRef, facts.len);
    defer allocator.free(def_to_op);
    @memset(def_to_op, null);
    var def_to_phi = try allocator.alloc(?PhiRef, facts.len);
    defer allocator.free(def_to_phi);
    @memset(def_to_phi, null);

    for (function.blocks) |block| {
        for (block.phis, 0..) |phi, i| {
            if (phi.dest < def_to_phi.len) def_to_phi[phi.dest] = .{ .block = block.id, .index = i };
        }
        for (block.ops, 0..) |op, i| {
            for (op.defs) |def| {
                if (def < def_to_op.len) def_to_op[def] = .{ .block = block.id, .index = @intCast(i) };
            }
        }
    }

    var work: std.ArrayList(ssa.ValueId) = .empty;
    defer work.deinit(allocator);

    for (function.blocks) |block| {
        for (block.ops, 0..) |op, i| {
            const side_effect = sideEffecting(op.inst);
            op_states[block.id][i].side_effect = side_effect;
            if (!side_effect) continue;
            op_states[block.id][i].live = true;
            for (op.uses) |use| try markValueLive(facts, &work, allocator, use);
        }
    }

    var cursor: usize = 0;
    while (cursor < work.items.len) : (cursor += 1) {
        const value = work.items[cursor];
        if (def_to_op[value]) |op_ref| {
            const op = function.blocks[op_ref.block].ops[op_ref.index];
            op_states[op_ref.block][op_ref.index].live = true;
            for (op.uses) |use| try markValueLive(facts, &work, allocator, use);
        } else if (def_to_phi[value]) |phi_ref| {
            const phi = function.blocks[phi_ref.block].phis[phi_ref.index];
            for (phi.incoming) |incoming| try markValueLive(facts, &work, allocator, incoming.value);
        }

        if (facts[value].copy_of) |copy| try markValueLive(facts, &work, allocator, copy);
    }
}

pub fn run(allocator: std.mem.Allocator, function: *const ssa.Function, options: Options) Error!Result {
    function.verify() catch |err| switch (err) {
        error.OutOfMemory => return error.OutOfMemory,
        else => return error.InvalidSsa,
    };

    const facts = try allocator.alloc(ValueFact, function.values.len);
    errdefer allocator.free(facts);
    for (facts) |*fact_value| fact_value.* = .{};

    const op_states = try allocator.alloc([]OpState, function.blocks.len);
    errdefer allocator.free(op_states);
    var states_built: usize = 0;
    errdefer {
        for (op_states[0..states_built]) |states| allocator.free(states);
    }
    for (function.blocks, 0..) |block, i| {
        op_states[i] = try allocator.alloc(OpState, block.ops.len);
        @memset(op_states[i], .{});
        states_built += 1;
    }

    if (options.constant_folding or options.copy_propagation) {
        var changed = true;
        var before_defs: std.ArrayList(ValueFact) = .empty;
        defer before_defs.deinit(allocator);
        while (changed) {
            changed = false;
            for (function.graph.rpo) |block_id| {
                for (function.blocks[block_id].phis) |phi| {
                    const before = facts[phi.dest];
                    if (options.constant_folding) {
                        if (mergePhiConstant(facts, phi)) |constant| {
                            facts[phi.dest].constant = constant;
                            facts[phi.dest].ty = switch (constant) {
                                .int => .int,
                                .wide => .long,
                            };
                        }
                    }
                    if (!sameFact(before, facts[phi.dest])) changed = true;
                }

                for (function.blocks[block_id].ops, 0..) |op, i| {
                    before_defs.clearRetainingCapacity();
                    for (op.defs) |def| try before_defs.append(allocator, facts[def]);
                    const folded = if (options.constant_folding or options.copy_propagation) deriveOperationFacts(facts, op) else null;
                    op_states[block_id][i].folded = folded;
                    for (op.defs, 0..) |def, def_index| {
                        if (!sameFact(before_defs.items[def_index], facts[def])) changed = true;
                    }
                }
            }
        }
    }

    if (options.dead_code) try computeLiveness(allocator, function, facts, op_states);

    var stats: Stats = .{};
    for (facts) |fact_value| {
        if (fact_value.constant != null) stats.constants += 1;
        if (fact_value.copy_of != null) stats.copies += 1;
    }
    for (function.blocks) |block| {
        for (op_states[block.id]) |state| {
            if (state.live) {
                stats.live_ops += 1;
            } else {
                stats.dead_ops += 1;
            }
        }
    }

    return .{
        .allocator = allocator,
        .function = function,
        .facts = facts,
        .op_states = op_states,
        .stats = stats,
    };
}

pub fn optimize(
    allocator: std.mem.Allocator,
    insts: []const Instruction,
    tries: []const TryBlock,
    options: OptimizeOptions,
) !*OptimizedFunction {
    const self = try allocator.create(OptimizedFunction);
    errdefer allocator.destroy(self);

    self.original_cfg = try cfg.buildWithOptions(allocator, insts, tries, options.cfg_options);
    errdefer self.original_cfg.deinit();

    self.rewritten_cfg = try cfg_rewrite.rewrite(allocator, &self.original_cfg, options.rewrite_options);
    errdefer self.rewritten_cfg.deinit();

    self.tree = try dom.build(allocator, &self.rewritten_cfg.graph);
    errdefer self.tree.deinit();

    self.function = try ssa.build(allocator, &self.rewritten_cfg.graph, &self.tree);
    errdefer self.function.deinit();

    self.ssa_facts = try ssa_phase.run(allocator, &self.function);
    errdefer self.ssa_facts.deinit();

    self.facts = try run(allocator, &self.function, options.facts);
    errdefer self.facts.deinit();

    self.types = try typedir.build(allocator, &self.function);
    errdefer self.types.deinit();

    self.typed = try typed_ir.build(allocator, &self.function, &self.types, &self.facts);
    errdefer self.typed.deinit();

    self.loops = try loop_phase.run(allocator, &self.function, &self.tree, &self.ssa_facts);
    errdefer self.loops.deinit();

    self.memory = try memory_phase.run(allocator, &self.function, &self.types);
    errdefer self.memory.deinit();

    self.barriers = try barrier_phase.runWithOptions(allocator, &self.function, &self.types, .{
        .loops = &self.loops,
    });
    errdefer self.barriers.deinit();

    self.lowered = try lowering.build(allocator, .{
        .function = &self.function,
        .types = &self.types,
        .typed = &self.typed,
        .ssa_facts = &self.ssa_facts,
        .memory = &self.memory,
        .barriers = &self.barriers,
    });
    errdefer self.lowered.deinit();

    self.machine = try machine_bridge.build(allocator, &self.lowered);
    errdefer self.machine.deinit();

    self.derived = try derived_verify.run(allocator, &self.machine);
    errdefer self.derived.deinit();

    self.allocator = allocator;
    self.stats = .{
        .original_blocks = @intCast(self.original_cfg.blocks.len),
        .rewritten_blocks = @intCast(self.rewritten_cfg.graph.blocks.len),
        .lowered_insts = self.lowered.stats.lowered,
        .cfg_removed_blocks = self.rewritten_cfg.stats.removed_blocks,
        .cfg_merged_blocks = self.rewritten_cfg.stats.merged_blocks,
        .ssa_constants = self.ssa_facts.stats.constants,
        .ssa_dead_ops = self.ssa_facts.stats.dead_ops,
        .loop_count = self.loops.stats.loops,
        .memory_forwarded = self.memory.stats.forwarded_fields,
        .handle_resolves = self.barriers.stats.resolves_inserted,
        .handle_resolve_reuses = self.barriers.stats.resolves_reused,
        .loop_resolves_hoisted = self.barriers.stats.loop_resolves_hoisted,
        .derived_address_uses = self.derived.stats.address_uses,
        .machine_insts = self.machine.stats.instructions,
        .machine_edge_moves = self.machine.stats.edge_moves,
    };

    return self;
}

test "optimizer folds integer constants through SSA values" {
    const insts = [_]Instruction{
        .{ .const_ = .{ .dest = 0, .value = 7 } },
        .{ .const_ = .{ .dest = 1, .value = 5 } },
        .{ .add_int = .{ .dest = 2, .src1 = 0, .src2 = 1 } },
        .{ .return_ = .{ .src = 2 } },
    };
    var graph = try cfg.build(std.testing.allocator, &insts);
    defer graph.deinit();
    var tree = try dom.build(std.testing.allocator, &graph);
    defer tree.deinit();
    var function = try ssa.build(std.testing.allocator, &graph, &tree);
    defer function.deinit();
    var result = try run(std.testing.allocator, &function, .{});
    defer result.deinit();

    const op = function.blocks[graph.entry].ops[2];
    try std.testing.expectEqual(Constant{ .int = 12 }, result.facts[op.defs[0]].constant.?);
    try std.testing.expectEqual(Constant{ .int = 12 }, result.op_states[graph.entry][2].folded.?);
}

test "optimizer tracks copy propagation" {
    const insts = [_]Instruction{
        .{ .const_ = .{ .dest = 0, .value = 42 } },
        .{ .move = .{ .dest = 1, .src = 0 } },
        .{ .return_ = .{ .src = 1 } },
    };
    var graph = try cfg.build(std.testing.allocator, &insts);
    defer graph.deinit();
    var tree = try dom.build(std.testing.allocator, &graph);
    defer tree.deinit();
    var function = try ssa.build(std.testing.allocator, &graph, &tree);
    defer function.deinit();
    var result = try run(std.testing.allocator, &function, .{});
    defer result.deinit();

    const move_op = function.blocks[graph.entry].ops[1];
    try std.testing.expect(result.facts[move_op.defs[0]].copy_of != null);
    try std.testing.expectEqual(Constant{ .int = 42 }, result.facts[move_op.defs[0]].constant.?);
}

test "optimizer marks dead pure instructions and keeps side effects" {
    const insts = [_]Instruction{
        .{ .const_ = .{ .dest = 0, .value = 1 } },
        .{ .const_ = .{ .dest = 1, .value = 2 } },
        .{ .add_int = .{ .dest = 2, .src1 = 0, .src2 = 1 } },
        .return_void,
    };
    var graph = try cfg.build(std.testing.allocator, &insts);
    defer graph.deinit();
    var tree = try dom.build(std.testing.allocator, &graph);
    defer tree.deinit();
    var function = try ssa.build(std.testing.allocator, &graph, &tree);
    defer function.deinit();
    var result = try run(std.testing.allocator, &function, .{});
    defer result.deinit();

    const states = result.op_states[graph.entry];
    try std.testing.expect(!states[2].live);
    try std.testing.expect(states[3].live);
    try std.testing.expect(states[3].side_effect);
}

test "optimizer folds phi only when all incoming constants match" {
    const insts = [_]Instruction{
        .{ .if_eqz = .{ .src = 0, .offset = 3 } },
        .{ .const_ = .{ .dest = 1, .value = 9 } },
        .{ .goto_ = .{ .offset = 2 } },
        .{ .const_ = .{ .dest = 1, .value = 9 } },
        .{ .return_ = .{ .src = 1 } },
    };
    var graph = try cfg.build(std.testing.allocator, &insts);
    defer graph.deinit();
    var tree = try dom.build(std.testing.allocator, &graph);
    defer tree.deinit();
    var function = try ssa.build(std.testing.allocator, &graph, &tree);
    defer function.deinit();
    var result = try run(std.testing.allocator, &function, .{});
    defer result.deinit();

    const join = graph.blockForPc(4).?.id;
    const phi = function.blocks[join].phis[0];
    try std.testing.expectEqual(Constant{ .int = 9 }, result.facts[phi.dest].constant.?);
}

test "optimizer print helper emits stable summary" {
    const insts = [_]Instruction{
        .{ .const_ = .{ .dest = 0, .value = 1 } },
        .return_void,
    };
    var graph = try cfg.build(std.testing.allocator, &insts);
    defer graph.deinit();
    var tree = try dom.build(std.testing.allocator, &graph);
    defer tree.deinit();
    var function = try ssa.build(std.testing.allocator, &graph, &tree);
    defer function.deinit();
    var result = try run(std.testing.allocator, &function, .{});
    defer result.deinit();

    var buf: [512]u8 = undefined;
    var stream = std.Io.Writer.fixed(&buf);
    try result.print(&stream);
    const output = stream.buffered();

    try std.testing.expect(std.mem.indexOf(u8, output, "optimizer values=") != null);
    try std.testing.expect(std.mem.indexOf(u8, output, "live_ops=") != null);
    try std.testing.expect(std.mem.indexOf(u8, output, "values:") != null);
}

test "optimizer orchestrates full pipeline into lowered IR" {
    const insts = [_]Instruction{
        .{ .const_ = .{ .dest = 0, .value = 4 } },
        .{ .new_array = .{ .dest = 1, .size = 0, .type_idx = 1 } },
        .{ .const_ = .{ .dest = 2, .value = 2 } },
        .{ .aget = .{ .dest_or_src = 3, .array = 1, .index = 2 } },
        .{ .return_ = .{ .src = 3 } },
    };
    var optimized = try optimize(std.testing.allocator, &insts, &.{}, .{});
    defer optimized.deinit();

    try optimized.lowered.verify();
    try optimized.machine.verify();
    try optimized.derived.verify();
    try std.testing.expect(optimized.stats.lowered_insts > 0);
    try std.testing.expect(optimized.stats.machine_insts > 0);
    try std.testing.expect(optimized.lowered.stats.null_checks_elided >= 1);
    try std.testing.expect(optimized.lowered.stats.bounds_checks_elided >= 1);
}

test "optimizer pipeline owns and verifies relocation barrier plan" {
    const insts = [_]Instruction{
        .{ .const_string = .{ .dest = 0, .index = 1 } },
        .{ .iget = .{ .field_idx = 1, .dest_or_src = 1, .obj = 0 } },
        .{ .iget = .{ .field_idx = 2, .dest_or_src = 2, .obj = 0 } },
        .return_void,
    };
    var optimized = try optimize(std.testing.allocator, &insts, &.{}, .{});
    defer optimized.deinit();

    try optimized.barriers.verify();
    try optimized.lowered.verify();
    try optimized.machine.verify();
    try std.testing.expectEqual(@as(u32, 1), optimized.stats.handle_resolves);
    try std.testing.expectEqual(@as(u32, 1), optimized.stats.handle_resolve_reuses);
    try std.testing.expectEqual(@as(u32, 1), optimized.lowered.stats.handle_resolves);
    try std.testing.expectEqual(@as(u32, 2), optimized.lowered.stats.pointer_accesses);
    try std.testing.expectEqual(@as(u32, 1), optimized.machine.stats.resolves);
    try std.testing.expectEqual(@as(u32, 2), optimized.machine.stats.pointer_accesses);
    try std.testing.expectEqual(@as(u32, 2), optimized.derived.stats.address_uses);
    try std.testing.expectEqual(@as(usize, 1), optimized.lowered.resolve_values.len);

    const address = optimized.lowered.resolve_values[0];
    const handle = optimized.barriers.resolves[0].handle;
    try std.testing.expect(optimized.lowered.isGcRoot(handle));
    try std.testing.expect(!optimized.lowered.isGcRoot(address));
    try std.testing.expect(optimized.machine.isGcRoot(handle));
    try std.testing.expect(!optimized.machine.isGcRoot(address));

    var resolves: u32 = 0;
    var pointer_accesses: u32 = 0;
    for (optimized.lowered.blocks) |block| {
        for (block.insts) |inst| switch (inst.kind) {
            .resolve_handle => {
                resolves += 1;
                try std.testing.expectEqual(address, inst.defs[0]);
                try std.testing.expectEqual(@as(?lowering.RuntimeValueId, handle), inst.state_handle);
            },
            .field_load_ptr => {
                pointer_accesses += 1;
                try std.testing.expectEqual(@as(?lowering.RuntimeValueId, address), inst.address);
                try std.testing.expectEqual(@as(?lowering.RuntimeValueId, handle), inst.state_handle);
                try std.testing.expectEqual(@as(usize, 0), inst.uses.len);
            },
            else => {},
        };
    }
    try std.testing.expectEqual(@as(u32, 1), resolves);
    try std.testing.expectEqual(@as(u32, 2), pointer_accesses);
}

test "optimizer materializes guarded reference write barriers around pointer stores" {
    const insts = [_]Instruction{
        .{ .const_string = .{ .dest = 0, .index = 1 } },
        .{ .const_string = .{ .dest = 1, .index = 2 } },
        .{ .iput_object = .{ .field_idx = 3, .dest_or_src = 1, .obj = 0 } },
        .return_void,
    };
    var optimized = try optimize(std.testing.allocator, &insts, &.{}, .{});
    defer optimized.deinit();

    try optimized.lowered.verify();
    try optimized.machine.verify();
    try std.testing.expectEqual(@as(u32, 1), optimized.lowered.stats.handle_resolves);
    try std.testing.expectEqual(@as(u32, 1), optimized.lowered.stats.pointer_accesses);
    try std.testing.expectEqual(@as(u32, 1), optimized.lowered.stats.satb_barriers);
    try std.testing.expectEqual(@as(u32, 1), optimized.lowered.stats.card_barriers);

    var sequence: [4]lowering.Kind = undefined;
    var count: usize = 0;
    for (optimized.lowered.blocks) |block| {
        for (block.insts) |inst| switch (inst.kind) {
            .resolve_handle, .satb_pre_write, .field_store_ptr, .card_mark => {
                if (count >= sequence.len) return error.TestUnexpectedResult;
                sequence[count] = inst.kind;
                count += 1;
            },
            else => {},
        };
    }
    try std.testing.expectEqual(@as(usize, 4), count);
    try std.testing.expectEqualSlices(lowering.Kind, &.{ .resolve_handle, .satb_pre_write, .field_store_ptr, .card_mark }, &sequence);
}

test "optimizer hoists invariant handle resolution out of a no-safepoint loop" {
    const insts = [_]Instruction{
        .{ .const_ = .{ .dest = 1, .value = 2 } },
        .{ .goto_ = .{ .offset = 1 } },
        .{ .iget = .{ .field_idx = 1, .dest_or_src = 2, .obj = 0 } },
        .{ .if_eqz = .{ .src = 1, .offset = 3 } },
        .{ .add_int_lit8 = .{ .dest = 1, .src = 1, .lit = -1 } },
        .{ .goto_ = .{ .offset = -3 } },
        .{ .return_ = .{ .src = 2 } },
    };
    var optimized = try optimize(std.testing.allocator, &insts, &.{}, .{});
    defer optimized.deinit();

    try std.testing.expectEqual(@as(u32, 1), optimized.loops.stats.loops);
    try std.testing.expectEqual(@as(u32, 1), optimized.stats.handle_resolves);
    try std.testing.expectEqual(@as(u32, 1), optimized.stats.loop_resolves_hoisted);
    const loop = optimized.loops.loops[0];
    const preheader = loop.preheader orelse return error.TestUnexpectedResult;
    const resolve = optimized.barriers.resolves[0];
    try std.testing.expect(resolve.hoisted);
    try std.testing.expectEqual(preheader, resolve.placement_block);
    try std.testing.expectEqual(loop.header, resolve.loop_header.?);
    try std.testing.expectEqual(loop.header, resolve.defining_op.block);
    try std.testing.expectEqual(@as(u32, 0), resolve.defining_op.index);
    switch (optimized.barriers.ops[loop.header][0].resolve) {
        .reuse => |id| try std.testing.expectEqual(@as(u32, 0), id),
        else => return error.TestUnexpectedResult,
    }

    var preheader_resolves: u32 = 0;
    var header_resolves: u32 = 0;
    var resolve_before_jump = false;
    for (optimized.machine.blocks[preheader].insts, 0..) |inst, index| {
        if (inst.opcode != .resolve_handle) continue;
        preheader_resolves += 1;
        for (optimized.machine.blocks[preheader].insts[index + 1 ..]) |later| {
            if (later.opcode == .jump) resolve_before_jump = true;
        }
    }
    for (optimized.machine.blocks[loop.header].insts) |inst| {
        if (inst.opcode == .resolve_handle) header_resolves += 1;
    }
    try std.testing.expectEqual(@as(u32, 1), preheader_resolves);
    try std.testing.expectEqual(@as(u32, 0), header_resolves);
    try std.testing.expect(resolve_before_jump);
    try optimized.barriers.verify();
    try optimized.machine.verify();
    try optimized.derived.verify();

    const proof_op = &optimized.barriers.ops[loop.header][1];
    const old_kill = proof_op.relocation_kill;
    proof_op.relocation_kill = true;
    try std.testing.expectError(error.InvalidPlan, optimized.barriers.verify());
    proof_op.relocation_kill = old_kill;
}

test "optimizer keeps loop resolution local when a safepoint can relocate" {
    var invoke = instmod.Invoke{
        .class_name = "LRuntime;",
        .method_name = "poll",
        .signature = "()V",
        .dest = null,
        .kind = .static,
    };
    const insts = [_]Instruction{
        .{ .const_ = .{ .dest = 1, .value = 2 } },
        .{ .goto_ = .{ .offset = 1 } },
        .{ .iget = .{ .field_idx = 1, .dest_or_src = 2, .obj = 0 } },
        .{ .if_eqz = .{ .src = 1, .offset = 4 } },
        .{ .invoke = &invoke },
        .{ .add_int_lit8 = .{ .dest = 1, .src = 1, .lit = -1 } },
        .{ .goto_ = .{ .offset = -4 } },
        .{ .return_ = .{ .src = 2 } },
    };
    var optimized = try optimize(std.testing.allocator, &insts, &.{}, .{});
    defer optimized.deinit();

    try std.testing.expectEqual(@as(u32, 1), optimized.loops.stats.loops);
    try std.testing.expectEqual(@as(u32, 0), optimized.stats.loop_resolves_hoisted);
    const loop = optimized.loops.loops[0];
    const resolve = optimized.barriers.resolves[0];
    try std.testing.expect(!resolve.hoisted);
    try std.testing.expectEqual(loop.header, resolve.placement_block);
    switch (optimized.barriers.ops[loop.header][0].resolve) {
        .define => |id| try std.testing.expectEqual(@as(u32, 0), id),
        else => return error.TestUnexpectedResult,
    }
    try optimized.barriers.verify();
    try optimized.derived.verify();
}

test "derived verifier rejects an address use moved before its definition" {
    const insts = [_]Instruction{
        .{ .const_string = .{ .dest = 0, .index = 1 } },
        .{ .iget = .{ .field_idx = 1, .dest_or_src = 1, .obj = 0 } },
        .return_void,
    };
    var optimized = try optimize(std.testing.allocator, &insts, &.{}, .{});
    defer optimized.deinit();

    const block = &optimized.machine.blocks[optimized.machine.source.source.graph.entry];
    var resolve_index: ?usize = null;
    var access_index: ?usize = null;
    for (block.insts, 0..) |inst, index| switch (inst.opcode) {
        .resolve_handle => resolve_index = index,
        .field_load_ptr => access_index = index,
        else => {},
    };
    const resolve = resolve_index orelse return error.TestUnexpectedResult;
    const access = access_index orelse return error.TestUnexpectedResult;
    std.mem.swap(machine_bridge.Inst, &block.insts[resolve], &block.insts[access]);
    defer std.mem.swap(machine_bridge.Inst, &block.insts[resolve], &block.insts[access]);

    try optimized.machine.verify();
    try std.testing.expectError(error.AddressDefinitionNotDominating, optimized.derived.verify());
}

test "derived verifier rejects reuse forged across a relocation kill" {
    var invoke = instmod.Invoke{
        .class_name = "LRuntime;",
        .method_name = "pollingCall",
        .signature = "()V",
        .dest = null,
        .kind = .static,
    };
    const insts = [_]Instruction{
        .{ .const_string = .{ .dest = 0, .index = 1 } },
        .{ .iget = .{ .field_idx = 1, .dest_or_src = 1, .obj = 0 } },
        .{ .invoke = &invoke },
        .{ .iget = .{ .field_idx = 2, .dest_or_src = 2, .obj = 0 } },
        .return_void,
    };
    var optimized = try optimize(std.testing.allocator, &insts, &.{}, .{});
    defer optimized.deinit();

    var first: ?*machine_bridge.Inst = null;
    var second: ?*machine_bridge.Inst = null;
    for (optimized.machine.blocks) |*block| {
        for (block.insts) |*inst| {
            if (inst.opcode != .field_load_ptr) continue;
            if (first == null) {
                first = inst;
            } else {
                second = inst;
            }
        }
    }
    const before = first orelse return error.TestUnexpectedResult;
    const after = second orelse return error.TestUnexpectedResult;
    const saved_address = after.address;
    const saved_token = after.reloc_token;
    const saved_resolve = after.resolve_id;
    defer {
        after.address = saved_address;
        after.reloc_token = saved_token;
        after.resolve_id = saved_resolve;
    }
    after.address = before.address;
    after.reloc_token = before.reloc_token;
    after.resolve_id = before.resolve_id;

    try optimized.machine.verify();
    try std.testing.expectError(error.InvalidPlan, optimized.derived.verify());
}

fn optimizerAllocationFailureProbe(allocator: std.mem.Allocator, insts: []const Instruction) !void {
    var optimized = try optimize(allocator, insts, &.{}, .{});
    defer optimized.deinit();
    try optimized.barriers.verify();
}

test "optimizer stable ownership is leak-free at every allocation failure" {
    const insts = [_]Instruction{
        .{ .const_ = .{ .dest = 1, .value = 2 } },
        .{ .goto_ = .{ .offset = 1 } },
        .{ .iget = .{ .field_idx = 1, .dest_or_src = 2, .obj = 0 } },
        .{ .if_eqz = .{ .src = 1, .offset = 3 } },
        .{ .add_int_lit8 = .{ .dest = 1, .src = 1, .lit = -1 } },
        .{ .goto_ = .{ .offset = -3 } },
        .{ .return_ = .{ .src = 2 } },
    };
    try std.testing.checkAllAllocationFailures(
        std.testing.allocator,
        optimizerAllocationFailureProbe,
        .{&insts},
    );
}

test "optimizer orchestrator applies cfg rewrite before SSA" {
    const insts = [_]Instruction{
        .{ .goto_ = .{ .offset = 2 } },
        .{ .const_ = .{ .dest = 0, .value = 99 } },
        .return_void,
    };
    var optimized = try optimize(std.testing.allocator, &insts, &.{}, .{
        .cfg_options = .{ .prune_unreachable = false, .order = .linear },
    });
    defer optimized.deinit();

    try std.testing.expectEqual(@as(u32, 1), optimized.stats.cfg_removed_blocks);
    try std.testing.expectEqual(@as(cfg.BlockId, cfg.INVALID_BLOCK), optimized.rewritten_cfg.graph.inst_to_block[1]);
}

test "optimizer pipeline print helper emits stable summary" {
    const insts = [_]Instruction{
        .{ .const_ = .{ .dest = 0, .value = 1 } },
        .return_void,
    };
    var optimized = try optimize(std.testing.allocator, &insts, &.{}, .{});
    defer optimized.deinit();

    var buf: [8192]u8 = undefined;
    var stream = std.Io.Writer.fixed(&buf);
    try optimized.print(&stream);
    const output = stream.buffered();

    try std.testing.expect(std.mem.indexOf(u8, output, "optimizer_pipeline") != null);
    try std.testing.expect(std.mem.indexOf(u8, output, "-- barriers --") != null);
    try std.testing.expect(std.mem.indexOf(u8, output, "-- lowering --") != null);
    try std.testing.expect(std.mem.indexOf(u8, output, "-- machine bridge --") != null);
}
