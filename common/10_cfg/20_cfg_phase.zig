//! CFG canonicalization and cleanup analysis.
//!
//! The base CFG builder preserves bytecode block boundaries. This phase records
//! production-ready cleanup decisions for later graph rewriting/lowering:
//! unreachable blocks, jump threading, mergeable fallthrough blocks, redundant
//! branches, loop backedges, and normalized switch successor sets.

const std = @import("std");
const cfg = @import("cfg");
const dom = @import("dominator");
const instmod = @import("instructions");
const Instruction = instmod.Instruction;

pub const Error = error{
    EmptyGraph,
    InvalidGraph,
    OutOfMemory,
};

pub const JumpInfo = struct {
    block: cfg.BlockId,
    target: cfg.BlockId,
    canonical_target: cfg.BlockId,
};

pub const MergeInfo = struct {
    from: cfg.BlockId,
    into: cfg.BlockId,
};

pub const RedundantBranch = struct {
    block: cfg.BlockId,
    target: cfg.BlockId,
};

pub const Backedge = struct {
    from: cfg.BlockId,
    header: cfg.BlockId,
};

pub const SwitchInfo = struct {
    block: cfg.BlockId,
    unique_targets: []cfg.BlockId,
    has_fallthrough: bool,
};

pub const Stats = struct {
    unreachable_blocks: u32 = 0,
    jump_threads: u32 = 0,
    merge_candidates: u32 = 0,
    redundant_branches: u32 = 0,
    loops: u32 = 0,
    switches: u32 = 0,
};

pub const Result = struct {
    allocator: std.mem.Allocator,
    graph: *const cfg.Graph,
    reachable: []bool,
    canonical_successor: []cfg.BlockId,
    jumps: []JumpInfo,
    merges: []MergeInfo,
    redundant_branches: []RedundantBranch,
    backedges: []Backedge,
    switches: []SwitchInfo,
    stats: Stats,

    pub fn deinit(self: *Result) void {
        for (self.switches) |switch_info| self.allocator.free(switch_info.unique_targets);
        self.allocator.free(self.switches);
        self.allocator.free(self.backedges);
        self.allocator.free(self.redundant_branches);
        self.allocator.free(self.merges);
        self.allocator.free(self.jumps);
        self.allocator.free(self.canonical_successor);
        self.allocator.free(self.reachable);
        self.* = undefined;
    }

    pub fn print(self: *const Result, writer: anytype) !void {
        try writer.print(
            "cfg_phase blocks={d} unreachable={d} jumps={d} merges={d} redundant={d} loops={d} switches={d}\n",
            .{
                self.graph.blocks.len,
                self.stats.unreachable_blocks,
                self.stats.jump_threads,
                self.stats.merge_candidates,
                self.stats.redundant_branches,
                self.stats.loops,
                self.stats.switches,
            },
        );

        try writer.print("unreachable:", .{});
        var any_unreachable = false;
        for (self.reachable, 0..) |reachable, i| {
            if (!reachable) {
                any_unreachable = true;
                try writer.print(" b{d}", .{i});
            }
        }
        if (!any_unreachable) try writer.print(" <none>", .{});
        try writer.print("\n", .{});

        for (self.jumps) |jump| {
            try writer.print("jump b{d} -> b{d} canonical=b{d}\n", .{ jump.block, jump.target, jump.canonical_target });
        }
        for (self.merges) |merge| {
            try writer.print("merge b{d} into b{d}\n", .{ merge.from, merge.into });
        }
        for (self.redundant_branches) |branch| {
            try writer.print("redundant_branch b{d} target=b{d}\n", .{ branch.block, branch.target });
        }
        for (self.backedges) |edge| {
            try writer.print("loop backedge b{d} -> b{d}\n", .{ edge.from, edge.header });
        }
        for (self.switches) |switch_info| {
            try writer.print("switch b{d} fallthrough={} targets:", .{ switch_info.block, switch_info.has_fallthrough });
            for (switch_info.unique_targets) |target| try writer.print(" b{d}", .{target});
            if (switch_info.unique_targets.len == 0) try writer.print(" <none>", .{});
            try writer.print("\n", .{});
        }
    }
};

fn isJumpOnly(graph: *const cfg.Graph, block_id: cfg.BlockId) bool {
    const block = graph.blocks[block_id];
    if (block.len() != 1 or block.successors.len != 1) return false;
    if (graph.instructions[block.start] != .goto_) return false;
    for (graph.edges) |edge| {
        if (edge.from == block_id and edge.kind != .branch) return false;
    }
    return true;
}

fn canonicalTarget(graph: *const cfg.Graph, start: cfg.BlockId) cfg.BlockId {
    var target = start;
    var guard: usize = 0;
    while (target < graph.blocks.len and guard < graph.blocks.len) : (guard += 1) {
        if (!isJumpOnly(graph, target)) return target;
        const next = graph.blocks[target].successors[0];
        if (next == target) return target;
        target = next;
    }
    return start;
}

fn markReachable(graph: *const cfg.Graph, id: cfg.BlockId, reachable: []bool) void {
    if (id >= graph.blocks.len or reachable[id]) return;
    reachable[id] = true;
    for (graph.blocks[id].successors) |succ| markReachable(graph, succ, reachable);
}

fn hasEdgeKind(graph: *const cfg.Graph, from: cfg.BlockId, to: cfg.BlockId, kind: cfg.EdgeKind) bool {
    for (graph.edges) |edge| {
        if (edge.from == from and edge.to == to and edge.kind == kind) return true;
    }
    return false;
}

fn isConditional(inst: Instruction) bool {
    return switch (inst) {
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
        => true,
        else => false,
    };
}

fn isSwitch(inst: Instruction) bool {
    return switch (inst) {
        .packed_switch, .sparse_switch => true,
        else => false,
    };
}

fn appendUniqueBlock(list: *std.ArrayList(cfg.BlockId), allocator: std.mem.Allocator, value: cfg.BlockId) !void {
    for (list.items) |existing| {
        if (existing == value) return;
    }
    try list.append(allocator, value);
}

fn blockLess(_: void, a: cfg.BlockId, b: cfg.BlockId) bool {
    return a < b;
}

pub fn run(allocator: std.mem.Allocator, graph: *const cfg.Graph) Error!Result {
    if (graph.blocks.len == 0 or graph.entry >= graph.blocks.len) return error.InvalidGraph;

    const reachable = try allocator.alloc(bool, graph.blocks.len);
    errdefer allocator.free(reachable);
    @memset(reachable, false);
    markReachable(graph, graph.entry, reachable);

    const canonical_successor = try allocator.alloc(cfg.BlockId, graph.blocks.len);
    errdefer allocator.free(canonical_successor);
    for (canonical_successor, 0..) |*slot, i| slot.* = canonicalTarget(graph, @intCast(i));

    var jumps_list: std.ArrayList(JumpInfo) = .empty;
    defer jumps_list.deinit(allocator);
    var merges_list: std.ArrayList(MergeInfo) = .empty;
    defer merges_list.deinit(allocator);
    var redundant_list: std.ArrayList(RedundantBranch) = .empty;
    defer redundant_list.deinit(allocator);
    var backedge_list: std.ArrayList(Backedge) = .empty;
    defer backedge_list.deinit(allocator);
    var switch_list: std.ArrayList(SwitchInfo) = .empty;
    defer switch_list.deinit(allocator);

    var tree = try dom.build(allocator, graph);
    defer tree.deinit();

    var stats: Stats = .{};
    for (reachable) |is_reachable| {
        if (!is_reachable) stats.unreachable_blocks += 1;
    }

    for (graph.blocks) |block| {
        if (isJumpOnly(graph, block.id)) {
            const target = block.successors[0];
            const canonical = canonical_successor[block.id];
            try jumps_list.append(allocator, .{ .block = block.id, .target = target, .canonical_target = canonical });
            if (canonical != target) stats.jump_threads += 1;
        }

        if (block.successors.len == 1) {
            const succ = block.successors[0];
            if (succ < graph.blocks.len and graph.blocks[succ].predecessors.len == 1 and hasEdgeKind(graph, block.id, succ, .fallthrough)) {
                const term = graph.instructions[block.end - 1];
                if (!isConditional(term) and term != .goto_ and term != .return_void and term != .return_ and term != .return_wide and term != .return_object and term != .throw_) {
                    try merges_list.append(allocator, .{ .from = block.id, .into = succ });
                    stats.merge_candidates += 1;
                }
            }
        }

        if (block.len() > 0 and isConditional(graph.instructions[block.end - 1]) and block.successors.len >= 2) {
            const first = canonical_successor[block.successors[0]];
            var all_same = true;
            for (block.successors[1..]) |succ| {
                if (canonical_successor[succ] != first) {
                    all_same = false;
                    break;
                }
            }
            if (all_same) {
                try redundant_list.append(allocator, .{ .block = block.id, .target = first });
                stats.redundant_branches += 1;
            }
        }

        if (block.len() > 0 and isSwitch(graph.instructions[block.end - 1])) {
            var targets: std.ArrayList(cfg.BlockId) = .empty;
            errdefer targets.deinit(allocator);
            var has_fallthrough = false;
            for (graph.edges) |edge| {
                if (edge.from != block.id) continue;
                if (edge.kind == .fallthrough) has_fallthrough = true;
                if (edge.kind == .fallthrough or edge.kind == .switch_case) {
                    try appendUniqueBlock(&targets, allocator, canonical_successor[edge.to]);
                }
            }
            std.mem.sort(cfg.BlockId, targets.items, {}, blockLess);
            try switch_list.append(allocator, .{
                .block = block.id,
                .unique_targets = try targets.toOwnedSlice(allocator),
                .has_fallthrough = has_fallthrough,
            });
            stats.switches += 1;
        }
    }

    for (graph.edges) |edge| {
        if (edge.from < graph.blocks.len and edge.to < graph.blocks.len and tree.dominates(edge.to, edge.from)) {
            try backedge_list.append(allocator, .{ .from = edge.from, .header = edge.to });
            stats.loops += 1;
        }
    }

    return .{
        .allocator = allocator,
        .graph = graph,
        .reachable = reachable,
        .canonical_successor = canonical_successor,
        .jumps = try jumps_list.toOwnedSlice(allocator),
        .merges = try merges_list.toOwnedSlice(allocator),
        .redundant_branches = try redundant_list.toOwnedSlice(allocator),
        .backedges = try backedge_list.toOwnedSlice(allocator),
        .switches = try switch_list.toOwnedSlice(allocator),
        .stats = stats,
    };
}

test "cfg_phase reports unreachable blocks when graph keeps them" {
    const insts = [_]Instruction{
        .{ .goto_ = .{ .offset = 2 } },
        .{ .const_ = .{ .dest = 0, .value = 99 } },
        .return_void,
    };
    var graph = try cfg.buildWithOptions(std.testing.allocator, &insts, &.{}, .{ .prune_unreachable = false, .order = .linear });
    defer graph.deinit();
    var result = try run(std.testing.allocator, &graph);
    defer result.deinit();

    try std.testing.expectEqual(@as(u32, 1), result.stats.unreachable_blocks);
    try std.testing.expect(!result.reachable[graph.blockForPc(1).?.id]);
}

test "cfg_phase finds jump threading targets" {
    const insts = [_]Instruction{
        .{ .goto_ = .{ .offset = 1 } },
        .{ .goto_ = .{ .offset = 1 } },
        .return_void,
    };
    var graph = try cfg.build(std.testing.allocator, &insts);
    defer graph.deinit();
    var result = try run(std.testing.allocator, &graph);
    defer result.deinit();

    try std.testing.expect(result.jumps.len >= 2);
    try std.testing.expectEqual(graph.blockForPc(2).?.id, result.canonical_successor[graph.entry]);
    try std.testing.expectEqual(@as(u32, 1), result.stats.jump_threads);
}

test "cfg_phase records mergeable fallthrough blocks" {
    const handlers = [_]instmod.CatchHandler{.{ .type_idx = 1, .target_pc = 2 }};
    const tries = [_]instmod.TryBlock{.{ .start_pc = 0, .end_pc = 1, .handlers = &handlers }};
    const insts = [_]Instruction{
        .{ .const_ = .{ .dest = 1, .value = 1 } },
        .return_void,
        .return_void,
    };
    var graph = try cfg.buildWithTries(std.testing.allocator, &insts, &tries);
    defer graph.deinit();
    var result = try run(std.testing.allocator, &graph);
    defer result.deinit();

    try std.testing.expect(result.merges.len >= 1);
}

test "cfg_phase detects redundant conditional branches" {
    const insts = [_]Instruction{
        .{ .if_eqz = .{ .src = 0, .offset = 1 } },
        .return_void,
    };
    var graph = try cfg.build(std.testing.allocator, &insts);
    defer graph.deinit();
    var result = try run(std.testing.allocator, &graph);
    defer result.deinit();

    try std.testing.expectEqual(@as(u32, 1), result.stats.redundant_branches);
    try std.testing.expectEqual(graph.blockForPc(1).?.id, result.redundant_branches[0].target);
}

test "cfg_phase normalizes loop backedges" {
    const insts = [_]Instruction{
        .{ .const_ = .{ .dest = 0, .value = 0 } },
        .{ .if_eqz = .{ .src = 0, .offset = 2 } },
        .{ .goto_ = .{ .offset = -1 } },
        .return_void,
    };
    var graph = try cfg.build(std.testing.allocator, &insts);
    defer graph.deinit();
    var result = try run(std.testing.allocator, &graph);
    defer result.deinit();

    try std.testing.expectEqual(@as(u32, 1), result.stats.loops);
    try std.testing.expectEqual(graph.blockForPc(1).?.id, result.backedges[0].header);
}

test "cfg_phase normalizes switch successor targets" {
    const payload = instmod.SwitchPayload{
        .keys = &[_]i32{ 1, 2, 3 },
        .targets = &[_]i32{ 2, 2, 4 },
    };
    const insts = [_]Instruction{
        .{ .packed_switch = .{ .src = 0, .payload_offset = 0, .payload = @constCast(&payload) } },
        .{ .const_ = .{ .dest = 1, .value = 0 } },
        .{ .const_ = .{ .dest = 1, .value = 1 } },
        .{ .goto_ = .{ .offset = 2 } },
        .{ .const_ = .{ .dest = 1, .value = 2 } },
        .return_void,
    };
    var graph = try cfg.build(std.testing.allocator, &insts);
    defer graph.deinit();
    var result = try run(std.testing.allocator, &graph);
    defer result.deinit();

    try std.testing.expectEqual(@as(u32, 1), result.stats.switches);
    try std.testing.expect(result.switches[0].has_fallthrough);
    try std.testing.expect(result.switches[0].unique_targets.len >= 2);
}

test "cfg_phase print helper emits stable summary" {
    const insts = [_]Instruction{
        .{ .if_eqz = .{ .src = 0, .offset = 1 } },
        .return_void,
    };
    var graph = try cfg.build(std.testing.allocator, &insts);
    defer graph.deinit();
    var result = try run(std.testing.allocator, &graph);
    defer result.deinit();

    var buf: [512]u8 = undefined;
    var stream = std.Io.Writer.fixed(&buf);
    try result.print(&stream);
    const output = stream.buffered();

    try std.testing.expect(std.mem.indexOf(u8, output, "cfg_phase blocks=") != null);
    try std.testing.expect(std.mem.indexOf(u8, output, "redundant_branch") != null);
}
