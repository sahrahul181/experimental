//! Dominator tree construction over `cfg.Graph`.
//!
//! Uses the Cooper-Harvey-Kennedy iterative algorithm, which is very fast in
//! practice when blocks are visited in reverse post-order. The CFG builder
//! emits RPO by default, but this module also derives ranks from `graph.rpo`
//! so it remains correct for retained/unreachable blocks.

const std = @import("std");
const cfg = @import("cfg");
const instmod = @import("instructions");
const Instruction = instmod.Instruction;

pub const Error = error{
    EmptyGraph,
    InvalidGraph,
    OutOfMemory,
};

pub const Tree = struct {
    allocator: std.mem.Allocator,
    graph: *const cfg.Graph,
    idom: []cfg.BlockId,
    depth: []u32,
    children: [][]cfg.BlockId,
    frontier: [][]cfg.BlockId,
    rpo_rank: []u32,
    entry: cfg.BlockId,

    pub fn deinit(self: *Tree) void {
        for (self.children) |children| self.allocator.free(children);
        for (self.frontier) |frontier| self.allocator.free(frontier);
        self.allocator.free(self.children);
        self.allocator.free(self.frontier);
        self.allocator.free(self.rpo_rank);
        self.allocator.free(self.depth);
        self.allocator.free(self.idom);
        self.* = undefined;
    }

    pub inline fn immediateDominator(self: *const Tree, block: cfg.BlockId) ?cfg.BlockId {
        if (block >= self.idom.len) return null;
        const idom = self.idom[block];
        return if (idom == cfg.INVALID_BLOCK) null else idom;
    }

    pub fn dominates(self: *const Tree, dominator: cfg.BlockId, block: cfg.BlockId) bool {
        if (dominator >= self.idom.len or block >= self.idom.len) return false;
        if (self.idom[dominator] == cfg.INVALID_BLOCK or self.idom[block] == cfg.INVALID_BLOCK) return false;
        if (dominator == block) return true;

        var runner = block;
        while (runner != self.entry and runner != cfg.INVALID_BLOCK) {
            runner = self.idom[runner];
            if (runner == dominator) return true;
        }
        return false;
    }

    pub inline fn strictlyDominates(self: *const Tree, dominator: cfg.BlockId, block: cfg.BlockId) bool {
        return dominator != block and self.dominates(dominator, block);
    }

    pub fn print(self: *const Tree, writer: anytype) !void {
        try writer.print("domtree blocks={d} entry=b{d}\n", .{ self.idom.len, self.entry });
        for (0..self.idom.len) |id_usize| {
            const id: cfg.BlockId = @intCast(id_usize);
            const idom = self.idom[id];
            try writer.print("b{d} idom=", .{id});
            if (idom == cfg.INVALID_BLOCK) {
                try writer.print("<none>", .{});
            } else {
                try writer.print("b{d}", .{idom});
            }
            try writer.print(" depth={d}\n", .{self.depth[id]});

            try writer.print("  children:", .{});
            if (self.children[id].len == 0) {
                try writer.print(" <none>", .{});
            } else {
                for (self.children[id]) |child| try writer.print(" b{d}", .{child});
            }
            try writer.print("\n", .{});

            try writer.print("  frontier:", .{});
            if (self.frontier[id].len == 0) {
                try writer.print(" <none>", .{});
            } else {
                for (self.frontier[id]) |frontier| try writer.print(" b{d}", .{frontier});
            }
            try writer.print("\n", .{});
        }
    }
};

fn intersect(idom: []const cfg.BlockId, rpo_rank: []const u32, a: cfg.BlockId, b: cfg.BlockId) cfg.BlockId {
    var finger1 = a;
    var finger2 = b;
    while (finger1 != finger2) {
        while (rpo_rank[finger1] > rpo_rank[finger2]) finger1 = idom[finger1];
        while (rpo_rank[finger2] > rpo_rank[finger1]) finger2 = idom[finger2];
    }
    return finger1;
}

fn appendUnique(list: *std.ArrayList(cfg.BlockId), allocator: std.mem.Allocator, value: cfg.BlockId) !void {
    for (list.items) |existing| {
        if (existing == value) return;
    }
    try list.append(allocator, value);
}

pub fn build(allocator: std.mem.Allocator, graph: *const cfg.Graph) Error!Tree {
    const n = graph.blocks.len;
    if (n == 0) return error.EmptyGraph;
    if (graph.entry >= n or graph.rpo.len == 0) return error.InvalidGraph;

    const idom = try allocator.alloc(cfg.BlockId, n);
    errdefer allocator.free(idom);
    @memset(idom, cfg.INVALID_BLOCK);

    const depth = try allocator.alloc(u32, n);
    errdefer allocator.free(depth);
    @memset(depth, 0);

    const rpo_rank = try allocator.alloc(u32, n);
    errdefer allocator.free(rpo_rank);
    @memset(rpo_rank, std.math.maxInt(u32));
    const seen_rpo = try allocator.alloc(bool, n);
    defer allocator.free(seen_rpo);
    @memset(seen_rpo, false);
    for (graph.rpo, 0..) |block_id, i| {
        if (block_id >= n or seen_rpo[block_id]) return error.InvalidGraph;
        seen_rpo[block_id] = true;
        rpo_rank[block_id] = @intCast(i);
    }
    if (rpo_rank[graph.entry] == std.math.maxInt(u32)) return error.InvalidGraph;

    const entry = graph.entry;
    idom[entry] = entry;

    var changed = true;
    while (changed) {
        changed = false;
        for (graph.rpo) |block_id| {
            if (block_id == entry) continue;

            var new_idom: cfg.BlockId = cfg.INVALID_BLOCK;
            for (graph.blocks[block_id].predecessors) |pred| {
                if (idom[pred] == cfg.INVALID_BLOCK) continue;
                if (new_idom == cfg.INVALID_BLOCK) {
                    new_idom = pred;
                } else {
                    new_idom = intersect(idom, rpo_rank, pred, new_idom);
                }
            }

            if (new_idom != cfg.INVALID_BLOCK and idom[block_id] != new_idom) {
                idom[block_id] = new_idom;
                changed = true;
            }
        }
    }

    for (graph.rpo) |block_id| {
        if (block_id == entry or idom[block_id] == cfg.INVALID_BLOCK) continue;
        depth[block_id] = depth[idom[block_id]] + 1;
    }

    const child_counts = try allocator.alloc(u32, n);
    defer allocator.free(child_counts);
    @memset(child_counts, 0);
    for (0..n) |block_id| {
        const idom_id = idom[block_id];
        if (idom_id == cfg.INVALID_BLOCK or idom_id == block_id) continue;
        child_counts[idom_id] += 1;
    }

    const children = try allocator.alloc([]cfg.BlockId, n);
    errdefer allocator.free(children);
    var allocated_children: usize = 0;
    errdefer {
        for (children[0..allocated_children]) |slice| allocator.free(slice);
    }
    for (0..n) |block_id| {
        children[block_id] = try allocator.alloc(cfg.BlockId, child_counts[block_id]);
        allocated_children += 1;
    }

    @memset(child_counts, 0);
    for (0..n) |block_id_usize| {
        const block_id: cfg.BlockId = @intCast(block_id_usize);
        const idom_id = idom[block_id];
        if (idom_id == cfg.INVALID_BLOCK or idom_id == block_id) continue;
        children[idom_id][child_counts[idom_id]] = block_id;
        child_counts[idom_id] += 1;
    }

    var frontier_lists = try allocator.alloc(std.ArrayList(cfg.BlockId), n);
    defer allocator.free(frontier_lists);
    for (frontier_lists) |*list| list.* = .empty;
    defer {
        for (frontier_lists) |*list| list.deinit(allocator);
    }

    for (graph.blocks) |block| {
        if (block.predecessors.len < 2) continue;
        const block_id = block.id;
        const stop = idom[block_id];
        for (block.predecessors) |pred| {
            var runner = pred;
            while (runner != cfg.INVALID_BLOCK and runner != stop) {
                try appendUnique(&frontier_lists[runner], allocator, block_id);
                runner = idom[runner];
            }
        }
    }

    const frontier = try allocator.alloc([]cfg.BlockId, n);
    errdefer allocator.free(frontier);
    var allocated_frontier: usize = 0;
    errdefer {
        for (frontier[0..allocated_frontier]) |slice| allocator.free(slice);
    }
    for (frontier_lists, 0..) |*list, i| {
        frontier[i] = try list.toOwnedSlice(allocator);
        allocated_frontier += 1;
    }

    return .{
        .allocator = allocator,
        .graph = graph,
        .idom = idom,
        .depth = depth,
        .children = children,
        .frontier = frontier,
        .rpo_rank = rpo_rank,
        .entry = entry,
    };
}

test "dominator builds linear tree" {
    const insts = [_]Instruction{
        .{ .if_eqz = .{ .src = 0, .offset = 2 } },
        .{ .const_ = .{ .dest = 1, .value = 1 } },
        .return_void,
    };
    var graph = try cfg.build(std.testing.allocator, &insts);
    defer graph.deinit();
    var tree = try build(std.testing.allocator, &graph);
    defer tree.deinit();

    try std.testing.expectEqual(graph.entry, tree.idom[graph.entry]);
    try std.testing.expect(tree.dominates(graph.entry, 1));
    try std.testing.expect(tree.strictlyDominates(graph.entry, 2));
}

test "dominator computes diamond immediate dominators and frontier" {
    const insts = [_]Instruction{
        .{ .if_eqz = .{ .src = 0, .offset = 3 } },
        .{ .const_ = .{ .dest = 1, .value = 1 } },
        .{ .goto_ = .{ .offset = 2 } },
        .{ .const_ = .{ .dest = 1, .value = 2 } },
        .return_void,
    };
    var graph = try cfg.build(std.testing.allocator, &insts);
    defer graph.deinit();
    var tree = try build(std.testing.allocator, &graph);
    defer tree.deinit();

    const entry = graph.entry;
    const join = graph.blockForPc(4).?.id;
    try std.testing.expectEqual(entry, tree.idom[join]);
    try std.testing.expect(tree.dominates(entry, join));

    const left = graph.blockForPc(1).?.id;
    const right = graph.blockForPc(3).?.id;
    try std.testing.expectEqualSlices(cfg.BlockId, &[_]cfg.BlockId{join}, tree.frontier[left]);
    try std.testing.expectEqualSlices(cfg.BlockId, &[_]cfg.BlockId{join}, tree.frontier[right]);
}

test "dominator handles loop header frontier" {
    const insts = [_]Instruction{
        .{ .const_ = .{ .dest = 0, .value = 0 } },
        .{ .if_eqz = .{ .src = 0, .offset = 2 } },
        .{ .goto_ = .{ .offset = -1 } },
        .return_void,
    };
    var graph = try cfg.build(std.testing.allocator, &insts);
    defer graph.deinit();
    var tree = try build(std.testing.allocator, &graph);
    defer tree.deinit();

    const entry = graph.entry;
    const header = graph.blockForPc(1).?.id;
    const latch = graph.blockForPc(2).?.id;
    try std.testing.expect(tree.dominates(entry, header));
    try std.testing.expect(tree.dominates(header, latch));
    try std.testing.expectEqualSlices(cfg.BlockId, &[_]cfg.BlockId{header}, tree.frontier[latch]);
}

test "dominator print helper emits stable summary" {
    const insts = [_]Instruction{
        .{ .if_eqz = .{ .src = 0, .offset = 2 } },
        .{ .const_ = .{ .dest = 1, .value = 1 } },
        .return_void,
    };
    var graph = try cfg.build(std.testing.allocator, &insts);
    defer graph.deinit();
    var tree = try build(std.testing.allocator, &graph);
    defer tree.deinit();

    var buf: [512]u8 = undefined;
    var stream = std.Io.Writer.fixed(&buf);
    try tree.print(&stream);
    const output = stream.buffered();

    try std.testing.expect(std.mem.indexOf(u8, output, "domtree blocks=3 entry=b0") != null);
    try std.testing.expect(std.mem.indexOf(u8, output, "children:") != null);
    try std.testing.expect(std.mem.indexOf(u8, output, "frontier:") != null);
}
