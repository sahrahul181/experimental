//! SSA optimization phase metadata.
//!
//! This pass keeps SSA immutable and records the decisions needed by later
//! rewriting/lowering: constant propagation, copy propagation, dead code/phi
//! detection, local value numbering/CSE representatives, and sparse conditional
//! reachability from constant branches.

const std = @import("std");
const cfg = @import("cfg");
const dom = @import("dominator");
const ssa = @import("ssa");
const instmod = @import("instructions");
const Instruction = instmod.Instruction;

pub const Error = error{
    InvalidSsa,
    OutOfMemory,
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
            .int => |v| try writer.print("i32({d})", .{v}),
            .wide => |v| try writer.print("i64({d})", .{v}),
        }
    }
};

pub const Lattice = enum(u8) {
    unknown,
    constant,
    overdefined,
};

pub const ValueFact = struct {
    lattice: Lattice = .unknown,
    constant: ?Constant = null,
    copy_of: ?ssa.ValueId = null,
    live: bool = false,
    value_number: u32 = 0,
    cse_of: ?ssa.ValueId = null,
};

pub const OpFact = struct {
    live: bool = false,
    side_effect: bool = false,
    folded: ?Constant = null,
};

pub const Stats = struct {
    constants: u32 = 0,
    copies: u32 = 0,
    dead_ops: u32 = 0,
    dead_phis: u32 = 0,
    cse_hits: u32 = 0,
    executable_blocks: u32 = 0,
};

pub const Result = struct {
    allocator: std.mem.Allocator,
    function: *const ssa.Function,
    values: []ValueFact,
    ops: [][]OpFact,
    executable_blocks: []bool,
    stats: Stats,

    pub fn deinit(self: *Result) void {
        self.allocator.free(self.executable_blocks);
        for (self.ops) |ops| self.allocator.free(ops);
        self.allocator.free(self.ops);
        self.allocator.free(self.values);
        self.* = undefined;
    }

    pub fn print(self: *const Result, writer: anytype) !void {
        try writer.print(
            "ssa_phase values={d} constants={d} copies={d} dead_ops={d} dead_phis={d} cse_hits={d} executable_blocks={d}\n",
            .{ self.values.len, self.stats.constants, self.stats.copies, self.stats.dead_ops, self.stats.dead_phis, self.stats.cse_hits, self.stats.executable_blocks },
        );
        for (self.function.graph.rpo) |block_id| {
            try writer.print("b{d} executable={}\n", .{ block_id, self.executable_blocks[block_id] });
            const block = self.function.blocks[block_id];
            for (block.phis) |phi| {
                const fact = self.values[phi.dest];
                try writer.print("  v{d} phi live={} vn={d}", .{ phi.dest, fact.live, fact.value_number });
                if (fact.constant) |constant| {
                    try writer.print(" const=", .{});
                    try constant.print(writer);
                }
                try writer.print("\n", .{});
            }
            for (block.ops, 0..) |op, i| {
                const fact = self.ops[block_id][i];
                try writer.print("  pc{d} {s} live={} side_effect={}", .{ op.pc, @tagName(op.inst), fact.live, fact.side_effect });
                if (fact.folded) |constant| {
                    try writer.print(" folded=", .{});
                    try constant.print(writer);
                }
                try writer.print("\n", .{});
            }
        }
    }
};

fn sameConstant(a: ?Constant, b: ?Constant) bool {
    if (a == null and b == null) return true;
    if (a == null or b == null) return false;
    return a.?.eql(b.?);
}

fn sameFact(a: ValueFact, b: ValueFact) bool {
    return a.lattice == b.lattice and sameConstant(a.constant, b.constant) and a.copy_of == b.copy_of;
}

fn constantOf(values: []const ValueFact, value: ssa.ValueId) ?Constant {
    var current = value;
    var guard: usize = 0;
    while (current < values.len and guard < values.len) : (guard += 1) {
        const fact = values[current];
        if (fact.constant) |constant| return constant;
        if (fact.copy_of) |copy| {
            current = copy;
            continue;
        }
        return null;
    }
    return null;
}

fn canonicalCopy(values: []const ValueFact, value: ssa.ValueId) ssa.ValueId {
    var current = value;
    var guard: usize = 0;
    while (current < values.len and guard < values.len) : (guard += 1) {
        const copy = values[current].copy_of orelse return current;
        if (copy == current) return current;
        current = copy;
    }
    return value;
}

fn mergePhi(values: []const ValueFact, phi: ssa.Phi) ValueFact {
    if (phi.incoming.len == 0) return .{};
    var first_const: ?Constant = null;
    var first_copy: ?ssa.ValueId = null;
    var all_const = true;
    var all_copy = true;
    var saw_defined = false;

    for (phi.incoming) |incoming| {
        const incoming_const = constantOf(values, incoming.value);
        if (incoming_const) |constant| {
            if (!saw_defined) first_const = constant;
            if (first_const == null or !first_const.?.eql(constant)) all_const = false;
        } else {
            all_const = false;
        }

        const copy = canonicalCopy(values, incoming.value);
        if (!saw_defined) first_copy = copy;
        if (first_copy == null or first_copy.? != copy) all_copy = false;
        saw_defined = true;
    }

    if (all_const and first_const != null) return .{ .lattice = .constant, .constant = first_const };
    if (all_copy and first_copy != null) return .{ .copy_of = first_copy };
    return .{ .lattice = .overdefined };
}

fn intConst(values: []const ValueFact, value: ssa.ValueId) ?i32 {
    const constant = constantOf(values, value) orelse return null;
    return switch (constant) {
        .int => |v| v,
        else => null,
    };
}

fn wideConst(values: []const ValueFact, value: ssa.ValueId) ?i64 {
    const constant = constantOf(values, value) orelse return null;
    return switch (constant) {
        .wide => |v| v,
        else => null,
    };
}

fn foldInt(inst: Instruction, a: i32, b: i32) ?i32 {
    return switch (inst) {
        .add_int => a +% b,
        .sub_int => a -% b,
        .mul_int => a *% b,
        .and_int => @bitCast(@as(u32, @bitCast(a)) & @as(u32, @bitCast(b))),
        .or_int => @bitCast(@as(u32, @bitCast(a)) | @as(u32, @bitCast(b))),
        .xor_int => @bitCast(@as(u32, @bitCast(a)) ^ @as(u32, @bitCast(b))),
        .div_int => if (b == 0) null else if (a == std.math.minInt(i32) and b == -1) a else @divTrunc(a, b),
        .rem_int => if (b == 0) null else if (a == std.math.minInt(i32) and b == -1) 0 else @rem(a, b),
        else => null,
    };
}

fn foldWide(inst: Instruction, a: i64, b: i64) ?i64 {
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

fn derive(values: []ValueFact, op: ssa.Operation) ?Constant {
    switch (op.inst) {
        .const_ => |inst| {
            if (op.defs.len > 0) values[op.defs[0]] = .{ .lattice = .constant, .constant = .{ .int = inst.value } };
            return .{ .int = inst.value };
        },
        .const_wide => |inst| {
            for (op.defs) |def| values[def] = .{ .lattice = .constant, .constant = .{ .wide = inst.value } };
            return .{ .wide = inst.value };
        },
        .move, .move_object, .move_wide => {
            for (op.defs, 0..) |def, i| {
                if (i >= op.uses.len) break;
                const source = canonicalCopy(values, op.uses[i]);
                values[def].copy_of = source;
                values[def].constant = constantOf(values, source);
                values[def].lattice = if (values[def].constant != null) .constant else .unknown;
            }
            return if (op.defs.len > 0) values[op.defs[0]].constant else null;
        },
        .add_int, .sub_int, .mul_int, .div_int, .rem_int, .and_int, .or_int, .xor_int => {
            if (op.uses.len < 2 or op.defs.len == 0) return null;
            const a = intConst(values, op.uses[0]) orelse return null;
            const b = intConst(values, op.uses[1]) orelse return null;
            const folded = foldInt(op.inst, a, b) orelse return null;
            values[op.defs[0]] = .{ .lattice = .constant, .constant = .{ .int = folded } };
            return .{ .int = folded };
        },
        .add_long, .sub_long, .mul_long, .div_long, .rem_long, .and_long, .or_long, .xor_long => {
            if (op.uses.len < 3 or op.defs.len == 0) return null;
            const a = wideConst(values, op.uses[0]) orelse return null;
            const b = wideConst(values, op.uses[2]) orelse return null;
            const folded = foldWide(op.inst, a, b) orelse return null;
            for (op.defs) |def| values[def] = .{ .lattice = .constant, .constant = .{ .wide = folded } };
            return .{ .wide = folded };
        },
        else => return null,
    }
}

fn sideEffecting(inst: Instruction) bool {
    return switch (inst) {
        .nop,
        .move,
        .move_wide,
        .move_object,
        .const_,
        .const_wide,
        .add_int,
        .sub_int,
        .mul_int,
        .and_int,
        .or_int,
        .xor_int,
        .add_long,
        .sub_long,
        .mul_long,
        .and_long,
        .or_long,
        .xor_long,
        => false,
        else => true,
    };
}

fn markLive(values: []ValueFact, work: *std.ArrayList(ssa.ValueId), allocator: std.mem.Allocator, value: ssa.ValueId) !void {
    if (value >= values.len or values[value].live) return;
    values[value].live = true;
    try work.append(allocator, value);
}

fn computeLiveness(allocator: std.mem.Allocator, function: *const ssa.Function, values: []ValueFact, ops: [][]OpFact) !void {
    var def_to_op = try allocator.alloc(?ssa.OpRef, values.len);
    defer allocator.free(def_to_op);
    @memset(def_to_op, null);
    var def_to_phi = try allocator.alloc(?struct { block: cfg.BlockId, index: usize }, values.len);
    defer allocator.free(def_to_phi);
    @memset(def_to_phi, null);

    for (function.blocks) |block| {
        for (block.phis, 0..) |phi, i| def_to_phi[phi.dest] = .{ .block = block.id, .index = i };
        for (block.ops, 0..) |op, i| {
            for (op.defs) |def| def_to_op[def] = .{ .block = block.id, .index = @intCast(i) };
        }
    }

    var work: std.ArrayList(ssa.ValueId) = .empty;
    defer work.deinit(allocator);
    for (function.blocks) |block| {
        for (block.ops, 0..) |op, i| {
            ops[block.id][i].side_effect = sideEffecting(op.inst);
            if (!ops[block.id][i].side_effect) continue;
            ops[block.id][i].live = true;
            for (op.uses) |use| try markLive(values, &work, allocator, use);
        }
    }

    var cursor: usize = 0;
    while (cursor < work.items.len) : (cursor += 1) {
        const value = work.items[cursor];
        if (def_to_op[value]) |ref| {
            ops[ref.block][ref.index].live = true;
            for (function.blocks[ref.block].ops[ref.index].uses) |use| try markLive(values, &work, allocator, use);
        } else if (def_to_phi[value]) |ref| {
            for (function.blocks[ref.block].phis[ref.index].incoming) |incoming| try markLive(values, &work, allocator, incoming.value);
        }
        if (values[value].copy_of) |copy| try markLive(values, &work, allocator, copy);
    }
}

fn branchKnownTarget(function: *const ssa.Function, values: []const ValueFact, op: ssa.Operation, block: ssa.Block) ?cfg.BlockId {
    if (op.uses.len == 0 or block.id == cfg.INVALID_BLOCK) return null;
    const c = intConst(values, op.uses[0]) orelse return null;
    const wanted: cfg.EdgeKind = switch (op.inst) {
        .if_eqz => if (c == 0) .branch else .fallthrough,
        .if_nez => if (c != 0) .branch else .fallthrough,
        else => return null,
    };
    for (function.graph.edges) |edge| {
        if (edge.from == block.id and edge.kind == wanted) return edge.to;
    }
    return null;
}

fn markExecutable(function: *const ssa.Function, values: []const ValueFact, executable: []bool, id: cfg.BlockId) void {
    if (id >= executable.len or executable[id]) return;
    executable[id] = true;
    const block = function.blocks[id];
    if (block.ops.len != 0) {
        const term = block.ops[block.ops.len - 1];
        if (branchKnownTarget(function, values, term, block)) |target| {
            markExecutable(function, values, executable, target);
            return;
        }
    }
    for (function.graph.blocks[id].successors) |succ| markExecutable(function, values, executable, succ);
}

fn computeExecutable(function: *const ssa.Function, values: []const ValueFact, executable: []bool) void {
    @memset(executable, false);
    markExecutable(function, values, executable, function.graph.entry);
}

const VNKey = struct {
    tag: u16,
    a: u32,
    b: u32,
};

fn activeTagId(inst: Instruction) u16 {
    return @intFromEnum(std.meta.activeTag(inst));
}

fn valueNumber(values: []ValueFact, value: ssa.ValueId) u32 {
    const canonical = canonicalCopy(values, value);
    if (canonical < values.len and values[canonical].value_number != 0) return values[canonical].value_number;
    return canonical + 1;
}

fn computeValueNumbers(allocator: std.mem.Allocator, function: *const ssa.Function, values: []ValueFact) !u32 {
    var map = std.AutoHashMap(VNKey, ssa.ValueId).init(allocator);
    defer map.deinit();
    var next: u32 = 1;
    for (values) |*fact| {
        fact.value_number = next;
        next += 1;
    }
    var cse_hits: u32 = 0;
    for (function.graph.rpo) |block_id| {
        for (function.blocks[block_id].ops) |op| {
            if (op.defs.len == 0 or sideEffecting(op.inst)) continue;
            var a: u32 = 0;
            var b: u32 = 0;
            if (op.uses.len > 0) a = valueNumber(values, op.uses[0]);
            if (op.uses.len > 1) b = valueNumber(values, op.uses[1]);
            switch (op.inst) {
                .add_int, .mul_int, .and_int, .or_int, .xor_int => if (b < a) std.mem.swap(u32, &a, &b),
                else => {},
            }
            const key = VNKey{ .tag = activeTagId(op.inst), .a = a, .b = b };
            const entry = try map.getOrPut(key);
            if (entry.found_existing) {
                values[op.defs[0]].cse_of = entry.value_ptr.*;
                values[op.defs[0]].value_number = values[entry.value_ptr.*].value_number;
                cse_hits += 1;
            } else {
                entry.value_ptr.* = op.defs[0];
            }
        }
    }
    return cse_hits;
}

pub fn run(allocator: std.mem.Allocator, function: *const ssa.Function) Error!Result {
    function.verify() catch return error.InvalidSsa;
    const values = try allocator.alloc(ValueFact, function.values.len);
    errdefer allocator.free(values);
    @memset(values, .{});

    const ops = try allocator.alloc([]OpFact, function.blocks.len);
    errdefer allocator.free(ops);
    var built_ops: usize = 0;
    errdefer for (ops[0..built_ops]) |slice| allocator.free(slice);
    for (function.blocks, 0..) |block, i| {
        ops[i] = try allocator.alloc(OpFact, block.ops.len);
        @memset(ops[i], .{});
        built_ops += 1;
    }

    const executable = try allocator.alloc(bool, function.blocks.len);
    errdefer allocator.free(executable);

    var changed = true;
    while (changed) {
        changed = false;
        for (function.graph.rpo) |block_id| {
            for (function.blocks[block_id].phis) |phi| {
                const before = values[phi.dest];
                const merged = mergePhi(values, phi);
                if (merged.lattice != .unknown or merged.copy_of != null) values[phi.dest] = merged;
                if (!sameFact(before, values[phi.dest])) changed = true;
            }
            for (function.blocks[block_id].ops, 0..) |op, i| {
                var before: [2]ValueFact = undefined;
                for (op.defs, 0..) |def, def_i| {
                    if (def_i < before.len) before[def_i] = values[def];
                }
                ops[block_id][i].folded = derive(values, op);
                for (op.defs, 0..) |def, def_i| {
                    if (def_i < before.len and !sameFact(before[def_i], values[def])) changed = true;
                }
            }
        }
    }

    try computeLiveness(allocator, function, values, ops);
    computeExecutable(function, values, executable);
    const cse_hits = try computeValueNumbers(allocator, function, values);

    var stats: Stats = .{ .cse_hits = cse_hits };
    for (values) |fact| {
        if (fact.constant != null) stats.constants += 1;
        if (fact.copy_of != null) stats.copies += 1;
    }
    for (function.blocks) |block| {
        for (block.phis) |phi| {
            if (!values[phi.dest].live) stats.dead_phis += 1;
        }
    }
    for (function.blocks) |block| {
        if (executable[block.id]) stats.executable_blocks += 1;
        for (ops[block.id]) |op| {
            if (!op.live) stats.dead_ops += 1;
        }
    }

    return .{
        .allocator = allocator,
        .function = function,
        .values = values,
        .ops = ops,
        .executable_blocks = executable,
        .stats = stats,
    };
}

fn buildPipeline(insts: []const Instruction, graph: *cfg.Graph, tree: *dom.Tree, function: *ssa.Function) !void {
    graph.* = try cfg.build(std.testing.allocator, insts);
    errdefer graph.deinit();
    tree.* = try dom.build(std.testing.allocator, graph);
    errdefer tree.deinit();
    function.* = try ssa.build(std.testing.allocator, graph, tree);
}

test "ssa_phase propagates constants and folds arithmetic" {
    const insts = [_]Instruction{
        .{ .const_ = .{ .dest = 0, .value = 10 } },
        .{ .const_ = .{ .dest = 1, .value = 5 } },
        .{ .sub_int = .{ .dest = 2, .src1 = 0, .src2 = 1 } },
        .{ .return_ = .{ .src = 2 } },
    };
    var graph: cfg.Graph = undefined;
    var tree: dom.Tree = undefined;
    var function: ssa.Function = undefined;
    try buildPipeline(&insts, &graph, &tree, &function);
    defer function.deinit();
    defer tree.deinit();
    defer graph.deinit();
    var result = try run(std.testing.allocator, &function);
    defer result.deinit();

    const op = function.blocks[graph.entry].ops[2];
    try std.testing.expectEqual(Constant{ .int = 5 }, result.values[op.defs[0]].constant.?);
}

test "ssa_phase tracks copy propagation" {
    const insts = [_]Instruction{
        .{ .const_ = .{ .dest = 0, .value = 42 } },
        .{ .move = .{ .dest = 1, .src = 0 } },
        .{ .return_ = .{ .src = 1 } },
    };
    var graph: cfg.Graph = undefined;
    var tree: dom.Tree = undefined;
    var function: ssa.Function = undefined;
    try buildPipeline(&insts, &graph, &tree, &function);
    defer function.deinit();
    defer tree.deinit();
    defer graph.deinit();
    var result = try run(std.testing.allocator, &function);
    defer result.deinit();

    const move_op = function.blocks[graph.entry].ops[1];
    try std.testing.expect(result.values[move_op.defs[0]].copy_of != null);
    try std.testing.expectEqual(Constant{ .int = 42 }, result.values[move_op.defs[0]].constant.?);
}

test "ssa_phase marks dead pure code and keeps side effects" {
    const insts = [_]Instruction{
        .{ .const_ = .{ .dest = 0, .value = 1 } },
        .{ .const_ = .{ .dest = 1, .value = 2 } },
        .{ .add_int = .{ .dest = 2, .src1 = 0, .src2 = 1 } },
        .return_void,
    };
    var graph: cfg.Graph = undefined;
    var tree: dom.Tree = undefined;
    var function: ssa.Function = undefined;
    try buildPipeline(&insts, &graph, &tree, &function);
    defer function.deinit();
    defer tree.deinit();
    defer graph.deinit();
    var result = try run(std.testing.allocator, &function);
    defer result.deinit();

    try std.testing.expect(!result.ops[graph.entry][2].live);
    try std.testing.expect(result.ops[graph.entry][3].live);
}

test "ssa_phase finds common subexpressions with value numbers" {
    const insts = [_]Instruction{
        .{ .const_ = .{ .dest = 0, .value = 1 } },
        .{ .const_ = .{ .dest = 1, .value = 2 } },
        .{ .add_int = .{ .dest = 2, .src1 = 0, .src2 = 1 } },
        .{ .add_int = .{ .dest = 3, .src1 = 1, .src2 = 0 } },
        .{ .return_ = .{ .src = 3 } },
    };
    var graph: cfg.Graph = undefined;
    var tree: dom.Tree = undefined;
    var function: ssa.Function = undefined;
    try buildPipeline(&insts, &graph, &tree, &function);
    defer function.deinit();
    defer tree.deinit();
    defer graph.deinit();
    var result = try run(std.testing.allocator, &function);
    defer result.deinit();

    const second = function.blocks[graph.entry].ops[3];
    try std.testing.expect(result.values[second.defs[0]].cse_of != null);
    try std.testing.expect(result.stats.cse_hits >= 1);
}

test "ssa_phase detects dead phi values" {
    const insts = [_]Instruction{
        .{ .if_eqz = .{ .src = 0, .offset = 3 } },
        .{ .const_ = .{ .dest = 1, .value = 1 } },
        .{ .goto_ = .{ .offset = 2 } },
        .{ .const_ = .{ .dest = 1, .value = 1 } },
        .return_void,
    };
    var graph: cfg.Graph = undefined;
    var tree: dom.Tree = undefined;
    var function: ssa.Function = undefined;
    try buildPipeline(&insts, &graph, &tree, &function);
    defer function.deinit();
    defer tree.deinit();
    defer graph.deinit();
    var result = try run(std.testing.allocator, &function);
    defer result.deinit();

    try std.testing.expect(result.stats.dead_phis >= 1);
}

test "ssa_phase sparse conditional propagation marks only feasible constant branch" {
    const insts = [_]Instruction{
        .{ .const_ = .{ .dest = 0, .value = 0 } },
        .{ .if_eqz = .{ .src = 0, .offset = 3 } },
        .{ .const_ = .{ .dest = 1, .value = 1 } },
        .{ .goto_ = .{ .offset = 2 } },
        .{ .const_ = .{ .dest = 1, .value = 2 } },
        .{ .return_ = .{ .src = 1 } },
    };
    var graph: cfg.Graph = undefined;
    var tree: dom.Tree = undefined;
    var function: ssa.Function = undefined;
    try buildPipeline(&insts, &graph, &tree, &function);
    defer function.deinit();
    defer tree.deinit();
    defer graph.deinit();
    var result = try run(std.testing.allocator, &function);
    defer result.deinit();

    try std.testing.expect(!result.executable_blocks[graph.blockForPc(2).?.id]);
    try std.testing.expect(result.executable_blocks[graph.blockForPc(4).?.id]);
}

test "ssa_phase print helper emits stable summary" {
    const insts = [_]Instruction{
        .{ .const_ = .{ .dest = 0, .value = 1 } },
        .return_void,
    };
    var graph: cfg.Graph = undefined;
    var tree: dom.Tree = undefined;
    var function: ssa.Function = undefined;
    try buildPipeline(&insts, &graph, &tree, &function);
    defer function.deinit();
    defer tree.deinit();
    defer graph.deinit();
    var result = try run(std.testing.allocator, &function);
    defer result.deinit();

    var buf: [512]u8 = undefined;
    var stream = std.Io.Writer.fixed(&buf);
    try result.print(&stream);
    const output = stream.buffered();
    try std.testing.expect(std.mem.indexOf(u8, output, "ssa_phase values=") != null);
    try std.testing.expect(std.mem.indexOf(u8, output, "dead_ops=") != null);
}
