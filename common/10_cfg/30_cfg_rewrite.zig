//! CFG rewrite/canonicalization pass.
//!
//! `cfg_phase.zig` detects cleanup opportunities. This pass applies the safe
//! structural ones and returns a fresh `cfg.Graph` suitable for rerunning
//! dominators/SSA on a cleaner control-flow shape.

const std = @import("std");
const cfg = @import("cfg");
const cfg_phase = @import("cfg_phase");
const instmod = @import("instructions");
const Instruction = instmod.Instruction;

pub const Error = error{
    EmptyGraph,
    InvalidGraph,
    OutOfMemory,
};

pub const Options = struct {
    remove_unreachable: bool = true,
    thread_jumps: bool = true,
    merge_fallthrough: bool = true,
    dedupe_edges: bool = true,
};

pub const Stats = struct {
    removed_blocks: u32 = 0,
    merged_blocks: u32 = 0,
    threaded_edges: u32 = 0,
    removed_edges: u32 = 0,
};

pub const Rewrite = struct {
    graph: cfg.Graph,
    stats: Stats,

    pub fn deinit(self: *Rewrite) void {
        self.graph.deinit();
        self.* = undefined;
    }

    pub fn print(self: *const Rewrite, writer: anytype) !void {
        try writer.print(
            "cfg_rewrite blocks={d} edges={d} removed_blocks={d} merged_blocks={d} threaded_edges={d} removed_edges={d}\n",
            .{
                self.graph.blocks.len,
                self.graph.edges.len,
                self.stats.removed_blocks,
                self.stats.merged_blocks,
                self.stats.threaded_edges,
                self.stats.removed_edges,
            },
        );
        try self.graph.print(writer);
    }
};

fn find(parent: []cfg.BlockId, id: cfg.BlockId) cfg.BlockId {
    var cur = id;
    while (parent[cur] != cur) cur = parent[cur];
    const root = cur;
    cur = id;
    while (parent[cur] != cur) {
        const next = parent[cur];
        parent[cur] = root;
        cur = next;
    }
    return root;
}

fn unite(parent: []cfg.BlockId, a: cfg.BlockId, b: cfg.BlockId) bool {
    const ra = find(parent, a);
    const rb = find(parent, b);
    if (ra == rb) return false;
    parent[rb] = ra;
    return true;
}

fn isTerminal(inst: Instruction) bool {
    return switch (inst) {
        .goto_, .return_void, .return_, .return_wide, .return_object, .throw_ => true,
        else => false,
    };
}

fn edgeExists(edges: []const cfg.Edge, from: cfg.BlockId, to: cfg.BlockId, kind: cfg.EdgeKind) bool {
    for (edges) |edge| {
        if (edge.from == from and edge.to == to and edge.kind == kind) return true;
    }
    return false;
}

fn equivalentEdgeExists(edges: []const cfg.Edge, from: cfg.BlockId, to: cfg.BlockId, kind: cfg.EdgeKind) bool {
    for (edges) |edge| {
        if (edge.from != from or edge.to != to) continue;
        if (edge.kind == kind) return true;
        if (edge.kind != .exception and kind != .exception) return true;
    }
    return false;
}

fn appendEdge(
    allocator: std.mem.Allocator,
    edges: *std.ArrayList(cfg.Edge),
    from: cfg.BlockId,
    to: cfg.BlockId,
    kind: cfg.EdgeKind,
    dedupe: bool,
) !bool {
    if (from == to and kind == .fallthrough) return false;
    if (dedupe and equivalentEdgeExists(edges.items, from, to, kind)) return false;
    try edges.append(allocator, .{ .from = from, .to = to, .kind = kind });
    return true;
}

fn blockLess(_: void, a: cfg.BlockId, b: cfg.BlockId) bool {
    return a < b;
}

fn dfs(blocks: []const cfg.BasicBlock, id: cfg.BlockId, visited: []bool, post: *std.ArrayList(cfg.BlockId), allocator: std.mem.Allocator) !void {
    if (id >= blocks.len or visited[id]) return;
    visited[id] = true;
    for (blocks[id].successors) |succ| try dfs(blocks, succ, visited, post, allocator);
    try post.append(allocator, id);
}

fn computeRpo(allocator: std.mem.Allocator, blocks: []const cfg.BasicBlock, entry: cfg.BlockId) ![]cfg.BlockId {
    const visited = try allocator.alloc(bool, blocks.len);
    defer allocator.free(visited);
    @memset(visited, false);

    var post: std.ArrayList(cfg.BlockId) = .empty;
    defer post.deinit(allocator);
    try dfs(blocks, entry, visited, &post, allocator);

    const rpo = try allocator.alloc(cfg.BlockId, post.items.len);
    for (post.items, 0..) |id, i| rpo[post.items.len - 1 - i] = id;
    return rpo;
}

fn applyRpoIndexes(blocks: []cfg.BasicBlock, rpo: []const cfg.BlockId) void {
    for (blocks) |*block| block.rpo_index = cfg.INVALID_BLOCK;
    for (rpo, 0..) |id, i| blocks[id].rpo_index = @intCast(i);
}

pub fn rewrite(allocator: std.mem.Allocator, graph: *const cfg.Graph, options: Options) Error!Rewrite {
    if (graph.blocks.len == 0 or graph.entry >= graph.blocks.len) return error.InvalidGraph;

    var phase = try cfg_phase.run(allocator, graph);
    defer phase.deinit();

    const parent = try allocator.alloc(cfg.BlockId, graph.blocks.len);
    defer allocator.free(parent);
    for (parent, 0..) |*slot, i| slot.* = @intCast(i);

    var stats: Stats = .{};
    if (options.merge_fallthrough) {
        for (phase.merges) |merge| {
            if (!phase.reachable[merge.from] or !phase.reachable[merge.into]) continue;
            const a = graph.blocks[merge.from];
            const b = graph.blocks[merge.into];
            if (a.end != b.start) continue;
            if (a.len() == 0 or isTerminal(graph.instructions[a.end - 1])) continue;
            if (unite(parent, merge.from, merge.into)) stats.merged_blocks += 1;
        }
    }

    const keep_rep = try allocator.alloc(bool, graph.blocks.len);
    defer allocator.free(keep_rep);
    @memset(keep_rep, false);
    for (graph.blocks) |block| {
        if (options.remove_unreachable and !phase.reachable[block.id]) {
            stats.removed_blocks += 1;
            continue;
        }
        keep_rep[find(parent, block.id)] = true;
    }

    const old_to_new = try allocator.alloc(cfg.BlockId, graph.blocks.len);
    defer allocator.free(old_to_new);
    @memset(old_to_new, cfg.INVALID_BLOCK);

    var reps: std.ArrayList(cfg.BlockId) = .empty;
    defer reps.deinit(allocator);
    for (graph.blocks) |block| {
        const rep = find(parent, block.id);
        if (!keep_rep[rep] or old_to_new[rep] != cfg.INVALID_BLOCK) continue;
        old_to_new[rep] = @intCast(reps.items.len);
        try reps.append(allocator, rep);
    }

    for (graph.blocks) |block| {
        const rep = find(parent, block.id);
        if (keep_rep[rep]) old_to_new[block.id] = old_to_new[rep];
    }

    const blocks = try allocator.alloc(cfg.BasicBlock, reps.items.len);
    errdefer allocator.free(blocks);

    var min_start = try allocator.alloc(u32, reps.items.len);
    defer allocator.free(min_start);
    var max_end = try allocator.alloc(u32, reps.items.len);
    defer allocator.free(max_end);
    for (min_start) |*v| v.* = std.math.maxInt(u32);
    @memset(max_end, 0);

    for (graph.blocks) |block| {
        const new_id = old_to_new[block.id];
        if (new_id == cfg.INVALID_BLOCK) continue;
        min_start[new_id] = @min(min_start[new_id], block.start);
        max_end[new_id] = @max(max_end[new_id], block.end);
    }

    var edge_list: std.ArrayList(cfg.Edge) = .empty;
    defer edge_list.deinit(allocator);
    for (graph.edges) |edge| {
        if (options.remove_unreachable and (!phase.reachable[edge.from] or !phase.reachable[edge.to])) continue;
        const from = old_to_new[edge.from];
        if (from == cfg.INVALID_BLOCK) continue;

        var target_old = edge.to;
        if (options.thread_jumps and edge.kind != .exception and target_old < phase.canonical_successor.len) {
            const canonical = phase.canonical_successor[target_old];
            if (canonical != target_old) {
                target_old = canonical;
                stats.threaded_edges += 1;
            }
        }
        const to = old_to_new[target_old];
        if (to == cfg.INVALID_BLOCK or from == to) continue;
        const added = try appendEdge(allocator, &edge_list, from, to, edge.kind, options.dedupe_edges);
        if (!added) stats.removed_edges += 1;
    }

    const succ_counts = try allocator.alloc(u32, reps.items.len);
    defer allocator.free(succ_counts);
    const pred_counts = try allocator.alloc(u32, reps.items.len);
    defer allocator.free(pred_counts);
    @memset(succ_counts, 0);
    @memset(pred_counts, 0);
    for (edge_list.items) |edge| {
        succ_counts[edge.from] += 1;
        pred_counts[edge.to] += 1;
    }

    var built_blocks: usize = 0;
    errdefer {
        for (blocks[0..built_blocks]) |block| {
            allocator.free(block.successors);
            allocator.free(block.predecessors);
        }
    }
    for (blocks, 0..) |*block, i| {
        const successors = try allocator.alloc(cfg.BlockId, succ_counts[i]);
        const predecessors = allocator.alloc(cfg.BlockId, pred_counts[i]) catch |err| {
            allocator.free(successors);
            return err;
        };
        block.* = .{
            .id = @intCast(i),
            .start = min_start[i],
            .end = max_end[i],
            .rpo_index = cfg.INVALID_BLOCK,
            .successors = successors,
            .predecessors = predecessors,
        };
        built_blocks += 1;
    }

    const edges = try edge_list.toOwnedSlice(allocator);
    errdefer allocator.free(edges);
    @memset(succ_counts, 0);
    @memset(pred_counts, 0);
    for (edges) |edge| {
        blocks[edge.from].successors[succ_counts[edge.from]] = edge.to;
        blocks[edge.to].predecessors[pred_counts[edge.to]] = edge.from;
        succ_counts[edge.from] += 1;
        pred_counts[edge.to] += 1;
    }

    for (blocks) |block| {
        std.mem.sort(cfg.BlockId, block.successors, {}, blockLess);
        std.mem.sort(cfg.BlockId, block.predecessors, {}, blockLess);
    }

    const inst_to_block = try allocator.alloc(cfg.BlockId, graph.inst_to_block.len);
    errdefer allocator.free(inst_to_block);
    @memset(inst_to_block, cfg.INVALID_BLOCK);
    for (graph.inst_to_block, 0..) |old_id, pc| {
        if (old_id == cfg.INVALID_BLOCK) continue;
        inst_to_block[pc] = old_to_new[old_id];
    }

    const entry = old_to_new[graph.entry];
    if (entry == cfg.INVALID_BLOCK) return error.InvalidGraph;
    const rpo = try computeRpo(allocator, blocks, entry);
    errdefer allocator.free(rpo);
    applyRpoIndexes(blocks, rpo);

    return .{
        .graph = .{
            .allocator = allocator,
            .instructions = graph.instructions,
            .blocks = blocks,
            .edges = edges,
            .inst_to_block = inst_to_block,
            .rpo = rpo,
            .entry = entry,
        },
        .stats = stats,
    };
}

test "cfg_rewrite removes retained unreachable blocks" {
    const insts = [_]Instruction{
        .{ .goto_ = .{ .offset = 2 } },
        .{ .const_ = .{ .dest = 0, .value = 99 } },
        .return_void,
    };
    var graph = try cfg.buildWithOptions(std.testing.allocator, &insts, &.{}, .{ .prune_unreachable = false, .order = .linear });
    defer graph.deinit();
    var rewritten = try rewrite(std.testing.allocator, &graph, .{});
    defer rewritten.deinit();

    try std.testing.expectEqual(@as(u32, 1), rewritten.stats.removed_blocks);
    try std.testing.expectEqual(@as(cfg.BlockId, cfg.INVALID_BLOCK), rewritten.graph.inst_to_block[1]);
}

test "cfg_rewrite threads jump-only chains" {
    const insts = [_]Instruction{
        .{ .goto_ = .{ .offset = 1 } },
        .{ .goto_ = .{ .offset = 1 } },
        .return_void,
    };
    var graph = try cfg.build(std.testing.allocator, &insts);
    defer graph.deinit();
    var rewritten = try rewrite(std.testing.allocator, &graph, .{});
    defer rewritten.deinit();

    try std.testing.expect(rewritten.stats.threaded_edges >= 1);
    try std.testing.expectEqual(rewritten.graph.blockForPc(2).?.id, rewritten.graph.blocks[rewritten.graph.entry].successors[0]);
}

test "cfg_rewrite merges safe fallthrough blocks" {
    const handlers = [_]instmod.CatchHandler{.{ .type_idx = 1, .target_pc = 2 }};
    const tries = [_]instmod.TryBlock{.{ .start_pc = 0, .end_pc = 1, .handlers = &handlers }};
    const insts = [_]Instruction{
        .{ .const_ = .{ .dest = 1, .value = 1 } },
        .return_void,
        .return_void,
    };
    var graph = try cfg.buildWithTries(std.testing.allocator, &insts, &tries);
    defer graph.deinit();
    var rewritten = try rewrite(std.testing.allocator, &graph, .{});
    defer rewritten.deinit();

    try std.testing.expectEqual(@as(u32, 1), rewritten.stats.merged_blocks);
    try std.testing.expect(rewritten.graph.blocks.len < graph.blocks.len);
}

test "cfg_rewrite deduplicates redundant branch edges" {
    const insts = [_]Instruction{
        .{ .if_eqz = .{ .src = 0, .offset = 1 } },
        .return_void,
    };
    var graph = try cfg.build(std.testing.allocator, &insts);
    defer graph.deinit();
    var rewritten = try rewrite(std.testing.allocator, &graph, .{});
    defer rewritten.deinit();

    try std.testing.expect(rewritten.graph.edges.len < graph.edges.len);
    try std.testing.expect(rewritten.stats.removed_edges >= 1);
}

test "cfg_rewrite print helper emits stable summary" {
    const insts = [_]Instruction{
        .{ .goto_ = .{ .offset = 1 } },
        .return_void,
    };
    var graph = try cfg.build(std.testing.allocator, &insts);
    defer graph.deinit();
    var rewritten = try rewrite(std.testing.allocator, &graph, .{});
    defer rewritten.deinit();

    var buf: [512]u8 = undefined;
    var stream = std.Io.Writer.fixed(&buf);
    try rewritten.print(&stream);
    const output = stream.buffered();
    try std.testing.expect(std.mem.indexOf(u8, output, "cfg_rewrite blocks=") != null);
    try std.testing.expect(std.mem.indexOf(u8, output, "cfg blocks=") != null);
}
