//! Loop optimization phase metadata.
//!
//! This phase analyzes SSA plus CFG/dominator facts and records conservative
//! loop optimization opportunities: LICM candidates, strength reduction,
//! induction variables, unroll/delete candidates, and simple integer ranges.

const std = @import("std");
const cfg = @import("cfg");
const dom = @import("dominator");
const ssa = @import("ssa");
const ssa_phase = @import("ssa_phase");
const instmod = @import("instructions");
const Instruction = instmod.Instruction;

pub const Error = error{
    InvalidInput,
    OutOfMemory,
};

pub const Range = struct {
    min: ?i64 = null,
    max: ?i64 = null,

    pub fn constant(value: i64) Range {
        return .{ .min = value, .max = value };
    }
};

pub const Loop = struct {
    header: cfg.BlockId,
    latch: cfg.BlockId,
    blocks: []cfg.BlockId,
    preheader: ?cfg.BlockId,
    invariant_ops: []ssa.OpRef,
    induction_values: []ssa.ValueId,
    strength_reduction_ops: []ssa.OpRef,
    unroll_factor: u8,
    delete_candidate: bool,
};

pub const Stats = struct {
    loops: u32 = 0,
    invariants: u32 = 0,
    inductions: u32 = 0,
    strength_reductions: u32 = 0,
    unroll_candidates: u32 = 0,
    delete_candidates: u32 = 0,
};

pub const Result = struct {
    allocator: std.mem.Allocator,
    function: *const ssa.Function,
    tree: *const dom.Tree,
    ssa_facts: ?*const ssa_phase.Result,
    loops: []Loop,
    ranges: []Range,
    stats: Stats,

    pub fn deinit(self: *Result) void {
        for (self.loops) |loop| {
            self.allocator.free(loop.strength_reduction_ops);
            self.allocator.free(loop.induction_values);
            self.allocator.free(loop.invariant_ops);
            self.allocator.free(loop.blocks);
        }
        self.allocator.free(self.loops);
        self.allocator.free(self.ranges);
        self.* = undefined;
    }

    pub fn print(self: *const Result, writer: anytype) !void {
        try writer.print(
            "loop_phase loops={d} invariants={d} inductions={d} strength={d} unroll={d} delete={d}\n",
            .{
                self.stats.loops,
                self.stats.invariants,
                self.stats.inductions,
                self.stats.strength_reductions,
                self.stats.unroll_candidates,
                self.stats.delete_candidates,
            },
        );
        for (self.loops, 0..) |loop, i| {
            try writer.print("loop{d} header=b{d} latch=b{d} preheader=", .{ i, loop.header, loop.latch });
            if (loop.preheader) |preheader| {
                try writer.print("b{d}", .{preheader});
            } else {
                try writer.print("<none>", .{});
            }
            try writer.print(" blocks:", .{});
            for (loop.blocks) |block| try writer.print(" b{d}", .{block});
            try writer.print(" unroll={d} delete={}\n", .{ loop.unroll_factor, loop.delete_candidate });

            for (loop.invariant_ops) |op| try writer.print("  invariant b{d}:op{d}\n", .{ op.block, op.index });
            for (loop.induction_values) |value| try writer.print("  induction v{d}\n", .{value});
            for (loop.strength_reduction_ops) |op| try writer.print("  strength b{d}:op{d}\n", .{ op.block, op.index });
        }
        try writer.print("ranges:\n", .{});
        for (self.ranges, 0..) |range, i| {
            if (range.min == null and range.max == null) continue;
            try writer.print("  v{d} [", .{i});
            if (range.min) |min| try writer.print("{d}", .{min}) else try writer.print("?", .{});
            try writer.print(",", .{});
            if (range.max) |max| try writer.print("{d}", .{max}) else try writer.print("?", .{});
            try writer.print("]\n", .{});
        }
    }
};

fn containsBlock(blocks: []const cfg.BlockId, id: cfg.BlockId) bool {
    for (blocks) |block| if (block == id) return true;
    return false;
}

fn appendUniqueBlock(list: *std.ArrayList(cfg.BlockId), allocator: std.mem.Allocator, id: cfg.BlockId) !void {
    if (containsBlock(list.items, id)) return;
    try list.append(allocator, id);
}

fn blockLess(_: void, a: cfg.BlockId, b: cfg.BlockId) bool {
    return a < b;
}

fn collectLoopBlocks(allocator: std.mem.Allocator, graph: *const cfg.Graph, header: cfg.BlockId, latch: cfg.BlockId) ![]cfg.BlockId {
    var blocks: std.ArrayList(cfg.BlockId) = .empty;
    errdefer blocks.deinit(allocator);
    var stack: std.ArrayList(cfg.BlockId) = .empty;
    defer stack.deinit(allocator);

    try appendUniqueBlock(&blocks, allocator, header);
    try appendUniqueBlock(&blocks, allocator, latch);
    try stack.append(allocator, latch);

    while (stack.pop()) |block_id| {
        for (graph.blocks[block_id].predecessors) |pred| {
            if (containsBlock(blocks.items, pred)) continue;
            try appendUniqueBlock(&blocks, allocator, pred);
            if (pred != header) try stack.append(allocator, pred);
        }
    }
    std.mem.sort(cfg.BlockId, blocks.items, {}, blockLess);
    return try blocks.toOwnedSlice(allocator);
}

fn findPreheader(graph: *const cfg.Graph, loop_blocks: []const cfg.BlockId, header: cfg.BlockId) ?cfg.BlockId {
    var found: ?cfg.BlockId = null;
    for (graph.blocks[header].predecessors) |pred| {
        if (containsBlock(loop_blocks, pred)) continue;
        if (found != null) return null;
        found = pred;
    }
    return found;
}

fn opPure(inst: Instruction) bool {
    return switch (inst) {
        .const_,
        .const_wide,
        .move,
        .move_wide,
        .move_object,
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
        .add_int_lit8,
        .mul_int_lit8,
        .add_int_lit16,
        .mul_int_lit16,
        => true,
        else => false,
    };
}

fn defInLoop(function: *const ssa.Function, loop_blocks: []const cfg.BlockId, value: ssa.ValueId) bool {
    if (value >= function.values.len) return false;
    const owner = function.values[value].block;
    return containsBlock(loop_blocks, owner);
}

fn opLoopInvariant(function: *const ssa.Function, loop_blocks: []const cfg.BlockId, op: ssa.Operation) bool {
    if (!opPure(op.inst) or op.defs.len == 0) return false;
    for (op.uses) |use| {
        if (defInLoop(function, loop_blocks, use)) return false;
    }
    return true;
}

fn isAddOne(op: ssa.Operation) bool {
    return switch (op.inst) {
        .add_int_lit8 => |inst| inst.lit == 1,
        .add_int_lit16 => |inst| inst.lit == 1,
        else => false,
    };
}

fn findInductions(function: *const ssa.Function, header: cfg.BlockId, loop_blocks: []const cfg.BlockId, out: *std.ArrayList(ssa.ValueId), allocator: std.mem.Allocator) !void {
    for (function.blocks[header].phis) |phi| {
        for (loop_blocks) |block_id| {
            for (function.blocks[block_id].ops) |op| {
                if (!isAddOne(op) or op.uses.len == 0 or op.defs.len == 0) continue;
                if (op.uses[0] == phi.dest) try out.append(allocator, phi.dest);
            }
        }
    }
}

fn strengthReductionCandidate(function: *const ssa.Function, inductions: []const ssa.ValueId, op: ssa.Operation) bool {
    if (op.defs.len == 0 or op.uses.len == 0) return false;
    return switch (op.inst) {
        .mul_int_lit8 => |inst| inst.lit > 1 and containsValue(inductions, op.uses[0]),
        .mul_int_lit16 => |inst| inst.lit > 1 and containsValue(inductions, op.uses[0]),
        .mul_int => blk: {
            _ = function;
            break :blk containsValue(inductions, op.uses[0]) or (op.uses.len > 1 and containsValue(inductions, op.uses[1]));
        },
        else => false,
    };
}

fn containsValue(values: []const ssa.ValueId, value: ssa.ValueId) bool {
    for (values) |candidate| if (candidate == value) return true;
    return false;
}

fn loopHasSideEffects(function: *const ssa.Function, blocks: []const cfg.BlockId) bool {
    for (blocks) |block_id| {
        for (function.blocks[block_id].ops) |op| {
            if (!opPure(op.inst)) {
                switch (op.inst) {
                    .goto_, .if_eqz, .if_nez, .if_ltz, .if_gez, .if_gtz, .if_lez, .if_eq, .if_ne, .if_lt, .if_ge, .if_gt, .if_le => {},
                    else => return true,
                }
            }
        }
    }
    return false;
}

fn constantFor(facts: ?*const ssa_phase.Result, value: ssa.ValueId) ?i64 {
    const f = facts orelse return null;
    if (value >= f.values.len) return null;
    const c = f.values[value].constant orelse return null;
    return switch (c) {
        .int => |v| v,
        .wide => |v| v,
    };
}

fn applyRanges(function: *const ssa.Function, facts: ?*const ssa_phase.Result, ranges: []Range) void {
    for (function.blocks) |block| {
        for (block.ops) |op| {
            switch (op.inst) {
                .const_ => |inst| {
                    if (op.defs.len > 0) ranges[op.defs[0]] = Range.constant(inst.value);
                },
                .const_wide => |inst| {
                    if (op.defs.len > 0) ranges[op.defs[0]] = Range.constant(inst.value);
                },
                .add_int_lit8 => |inst| if (op.defs.len > 0 and op.uses.len > 0) {
                    if (constantFor(facts, op.uses[0])) |base| ranges[op.defs[0]] = Range.constant(base + inst.lit);
                },
                .add_int_lit16 => |inst| if (op.defs.len > 0 and op.uses.len > 0) {
                    if (constantFor(facts, op.uses[0])) |base| ranges[op.defs[0]] = Range.constant(base + inst.lit);
                },
                else => {},
            }
        }
    }
}

pub fn run(
    allocator: std.mem.Allocator,
    function: *const ssa.Function,
    tree: *const dom.Tree,
    facts: ?*const ssa_phase.Result,
) Error!Result {
    function.verify() catch return error.InvalidInput;
    if (tree.graph != function.graph) return error.InvalidInput;
    if (facts) |f| {
        if (f.function != function) return error.InvalidInput;
    }

    var loops_list: std.ArrayList(Loop) = .empty;
    defer loops_list.deinit(allocator);

    const ranges = try allocator.alloc(Range, function.values.len);
    errdefer allocator.free(ranges);
    @memset(ranges, .{});
    applyRanges(function, facts, ranges);

    var stats: Stats = .{};
    for (function.graph.edges) |edge| {
        if (!tree.dominates(edge.to, edge.from)) continue;
        const blocks = try collectLoopBlocks(allocator, function.graph, edge.to, edge.from);
        errdefer allocator.free(blocks);

        var invariants: std.ArrayList(ssa.OpRef) = .empty;
        errdefer invariants.deinit(allocator);
        var inductions: std.ArrayList(ssa.ValueId) = .empty;
        errdefer inductions.deinit(allocator);
        var strengths: std.ArrayList(ssa.OpRef) = .empty;
        errdefer strengths.deinit(allocator);

        for (blocks) |block_id| {
            for (function.blocks[block_id].ops, 0..) |op, i| {
                if (opLoopInvariant(function, blocks, op)) try invariants.append(allocator, .{ .block = block_id, .index = @intCast(i) });
            }
        }
        try findInductions(function, edge.to, blocks, &inductions, allocator);
        for (blocks) |block_id| {
            for (function.blocks[block_id].ops, 0..) |op, i| {
                if (strengthReductionCandidate(function, inductions.items, op)) try strengths.append(allocator, .{ .block = block_id, .index = @intCast(i) });
            }
        }

        const unroll: u8 = if (blocks.len <= 3 and !loopHasSideEffects(function, blocks)) 4 else 1;
        const delete_candidate = !loopHasSideEffects(function, blocks) and inductions.items.len == 0;

        stats.loops += 1;
        stats.invariants += @intCast(invariants.items.len);
        stats.inductions += @intCast(inductions.items.len);
        stats.strength_reductions += @intCast(strengths.items.len);
        if (unroll > 1) stats.unroll_candidates += 1;
        if (delete_candidate) stats.delete_candidates += 1;

        try loops_list.append(allocator, .{
            .header = edge.to,
            .latch = edge.from,
            .blocks = blocks,
            .preheader = findPreheader(function.graph, blocks, edge.to),
            .invariant_ops = try invariants.toOwnedSlice(allocator),
            .induction_values = try inductions.toOwnedSlice(allocator),
            .strength_reduction_ops = try strengths.toOwnedSlice(allocator),
            .unroll_factor = unroll,
            .delete_candidate = delete_candidate,
        });
    }

    return .{
        .allocator = allocator,
        .function = function,
        .tree = tree,
        .ssa_facts = facts,
        .loops = try loops_list.toOwnedSlice(allocator),
        .ranges = ranges,
        .stats = stats,
    };
}

fn buildPipeline(insts: []const Instruction, graph: *cfg.Graph, tree: *dom.Tree, function: *ssa.Function, facts: *ssa_phase.Result) !void {
    graph.* = try cfg.build(std.testing.allocator, insts);
    errdefer graph.deinit();
    tree.* = try dom.build(std.testing.allocator, graph);
    errdefer tree.deinit();
    function.* = try ssa.build(std.testing.allocator, graph, tree);
    errdefer function.deinit();
    facts.* = try ssa_phase.run(std.testing.allocator, function);
}

test "loop_phase detects induction variables and strength reduction" {
    const insts = [_]Instruction{
        .{ .const_ = .{ .dest = 0, .value = 0 } },
        .{ .if_eqz = .{ .src = 0, .offset = 4 } },
        .{ .mul_int_lit8 = .{ .dest = 2, .src = 0, .lit = 4 } },
        .{ .add_int_lit8 = .{ .dest = 0, .src = 0, .lit = 1 } },
        .{ .goto_ = .{ .offset = -3 } },
        .return_void,
    };
    var graph: cfg.Graph = undefined;
    var tree: dom.Tree = undefined;
    var function: ssa.Function = undefined;
    var facts: ssa_phase.Result = undefined;
    try buildPipeline(&insts, &graph, &tree, &function, &facts);
    defer facts.deinit();
    defer function.deinit();
    defer tree.deinit();
    defer graph.deinit();
    var result = try run(std.testing.allocator, &function, &tree, &facts);
    defer result.deinit();

    try std.testing.expectEqual(@as(u32, 1), result.stats.loops);
    try std.testing.expect(result.stats.inductions >= 1);
    try std.testing.expect(result.stats.strength_reductions >= 1);
}

test "loop_phase finds loop invariant operations" {
    const insts = [_]Instruction{
        .{ .const_ = .{ .dest = 0, .value = 1 } },
        .{ .const_ = .{ .dest = 3, .value = 7 } },
        .{ .if_eqz = .{ .src = 0, .offset = 4 } },
        .{ .add_int_lit8 = .{ .dest = 4, .src = 3, .lit = 2 } },
        .{ .add_int_lit8 = .{ .dest = 0, .src = 0, .lit = 1 } },
        .{ .goto_ = .{ .offset = -3 } },
        .return_void,
    };
    var graph: cfg.Graph = undefined;
    var tree: dom.Tree = undefined;
    var function: ssa.Function = undefined;
    var facts: ssa_phase.Result = undefined;
    try buildPipeline(&insts, &graph, &tree, &function, &facts);
    defer facts.deinit();
    defer function.deinit();
    defer tree.deinit();
    defer graph.deinit();
    var result = try run(std.testing.allocator, &function, &tree, &facts);
    defer result.deinit();

    try std.testing.expect(result.stats.invariants >= 1);
}

test "loop_phase marks small pure loops for unrolling or deletion" {
    const insts = [_]Instruction{
        .{ .if_eqz = .{ .src = 0, .offset = 2 } },
        .{ .goto_ = .{ .offset = -1 } },
        .return_void,
    };
    var graph: cfg.Graph = undefined;
    var tree: dom.Tree = undefined;
    var function: ssa.Function = undefined;
    var facts: ssa_phase.Result = undefined;
    try buildPipeline(&insts, &graph, &tree, &function, &facts);
    defer facts.deinit();
    defer function.deinit();
    defer tree.deinit();
    defer graph.deinit();
    var result = try run(std.testing.allocator, &function, &tree, &facts);
    defer result.deinit();

    try std.testing.expect(result.stats.unroll_candidates >= 1);
    try std.testing.expect(result.stats.delete_candidates >= 1);
}

test "loop_phase computes constant ranges" {
    const insts = [_]Instruction{
        .{ .const_ = .{ .dest = 0, .value = 5 } },
        .{ .add_int_lit8 = .{ .dest = 1, .src = 0, .lit = 3 } },
        .return_void,
    };
    var graph: cfg.Graph = undefined;
    var tree: dom.Tree = undefined;
    var function: ssa.Function = undefined;
    var facts: ssa_phase.Result = undefined;
    try buildPipeline(&insts, &graph, &tree, &function, &facts);
    defer facts.deinit();
    defer function.deinit();
    defer tree.deinit();
    defer graph.deinit();
    var result = try run(std.testing.allocator, &function, &tree, &facts);
    defer result.deinit();

    const op = function.blocks[graph.entry].ops[1];
    try std.testing.expectEqual(@as(?i64, 8), result.ranges[op.defs[0]].min);
    try std.testing.expectEqual(@as(?i64, 8), result.ranges[op.defs[0]].max);
}

test "loop_phase print helper emits stable summary" {
    const insts = [_]Instruction{
        .{ .if_eqz = .{ .src = 0, .offset = 2 } },
        .{ .goto_ = .{ .offset = -1 } },
        .return_void,
    };
    var graph: cfg.Graph = undefined;
    var tree: dom.Tree = undefined;
    var function: ssa.Function = undefined;
    var facts: ssa_phase.Result = undefined;
    try buildPipeline(&insts, &graph, &tree, &function, &facts);
    defer facts.deinit();
    defer function.deinit();
    defer tree.deinit();
    defer graph.deinit();
    var result = try run(std.testing.allocator, &function, &tree, &facts);
    defer result.deinit();

    var buf: [512]u8 = undefined;
    var stream = std.Io.Writer.fixed(&buf);
    try result.print(&stream);
    const output = stream.buffered();
    try std.testing.expect(std.mem.indexOf(u8, output, "loop_phase loops=") != null);
    try std.testing.expect(std.mem.indexOf(u8, output, "loop0") != null);
}
