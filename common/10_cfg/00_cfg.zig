//! Production control-flow graph construction for decoded Dalvik instructions.
//!
//! The parser resolves Dalvik bytecode offsets into instruction-index offsets,
//! so this CFG uses instruction indices as semantic PCs.

const std = @import("std");
const instmod = @import("instructions");
const Instruction = instmod.Instruction;
const TryBlock = instmod.TryBlock;

pub const BlockId = u32;
pub const INVALID_BLOCK: BlockId = std.math.maxInt(BlockId);

pub const Error = error{
    BadBranchTarget,
    EmptyInstructionStream,
    OutOfMemory,
};

pub const EdgeKind = enum(u8) {
    fallthrough,
    branch,
    switch_case,
    exception,
};

pub const Order = enum(u8) {
    linear,
    rpo,
};

pub const Options = struct {
    prune_unreachable: bool = true,
    order: Order = .rpo,
};

pub const Edge = struct {
    from: BlockId,
    to: BlockId,
    kind: EdgeKind,
};

pub const BasicBlock = struct {
    id: BlockId,
    start: u32,
    end: u32,
    rpo_index: u32,
    successors: []BlockId,
    predecessors: []BlockId,

    pub inline fn len(self: BasicBlock) u32 {
        return self.end - self.start;
    }

    pub inline fn contains(self: BasicBlock, pc: u32) bool {
        return pc >= self.start and pc < self.end;
    }
};

pub const Graph = struct {
    allocator: std.mem.Allocator,
    instructions: []const Instruction,
    blocks: []BasicBlock,
    edges: []Edge,
    inst_to_block: []BlockId,
    rpo: []BlockId,
    entry: BlockId,

    pub fn deinit(self: *Graph) void {
        self.allocator.free(self.rpo);
        self.allocator.free(self.inst_to_block);
        self.allocator.free(self.edges);
        for (self.blocks) |block| {
            self.allocator.free(block.successors);
            self.allocator.free(block.predecessors);
        }
        self.allocator.free(self.blocks);
        self.* = undefined;
    }

    pub inline fn blockForPc(self: *const Graph, pc: u32) ?*const BasicBlock {
        if (pc >= self.inst_to_block.len) return null;
        const id = self.inst_to_block[pc];
        if (id == INVALID_BLOCK) return null;
        return &self.blocks[id];
    }

    pub inline fn entryBlock(self: *const Graph) *const BasicBlock {
        return &self.blocks[self.entry];
    }

    pub fn print(self: *const Graph, writer: anytype) !void {
        try writer.print("cfg blocks={d} edges={d} entry=b{d}\n", .{ self.blocks.len, self.edges.len, self.entry });
        try writer.print("rpo:", .{});
        for (self.rpo) |id| try writer.print(" b{d}", .{id});
        try writer.print("\n", .{});

        for (self.blocks) |block| {
            try writer.print(
                "b{d} pc=[{d},{d}) len={d} rpo={d}\n",
                .{ block.id, block.start, block.end, block.len(), block.rpo_index },
            );

            try writer.print("  succ:", .{});
            if (block.successors.len == 0) {
                try writer.print(" <none>", .{});
            } else {
                for (block.successors) |succ| try writer.print(" b{d}", .{succ});
            }
            try writer.print("\n", .{});

            try writer.print("  pred:", .{});
            if (block.predecessors.len == 0) {
                try writer.print(" <none>", .{});
            } else {
                for (block.predecessors) |pred| try writer.print(" b{d}", .{pred});
            }
            try writer.print("\n", .{});
        }

        try writer.print("edges:\n", .{});
        for (self.edges) |edge| {
            try writer.print("  b{d} -> b{d} {s}\n", .{ edge.from, edge.to, @tagName(edge.kind) });
        }
    }
};

const Succ = struct {
    target_pc: u32,
    kind: EdgeKind,
};

const RawBlock = struct {
    start: u32,
    end: u32,
};

fn hasFallthrough(inst: Instruction) bool {
    return switch (inst) {
        .goto_, .return_void, .return_, .return_wide, .return_object, .throw_ => false,
        else => true,
    };
}

fn successorIsLeader(inst: Instruction, kind: EdgeKind) bool {
    return switch (kind) {
        .branch, .switch_case, .exception => true,
        .fallthrough => switch (inst) {
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
            => true,
            else => false,
        },
    };
}

fn canThrow(inst: Instruction) bool {
    return switch (inst) {
        .nop,
        .move,
        .move_wide,
        .move_object,
        .move_result,
        .move_result_wide,
        .move_result_object,
        .move_exception,
        .return_void,
        .return_,
        .return_wide,
        .return_object,
        .const_,
        .const_wide,
        .goto_,
        .packed_switch,
        .sparse_switch,
        .cmpl_float,
        .cmpg_float,
        .cmpl_double,
        .cmpg_double,
        .cmp_long,
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
        => false,
        .div_int,
        .rem_int,
        .div_long,
        .rem_long,
        .div_int_lit16,
        .rem_int_lit16,
        .div_int_lit8,
        .rem_int_lit8,
        => true,
        else => true,
    };
}

fn checkedPc(pc: u32, len: usize) Error!void {
    if (pc >= len) return error.BadBranchTarget;
}

fn checkedTarget(pc: usize, offset: i32, len: usize) Error!u32 {
    const target = @as(i64, @intCast(pc)) + @as(i64, offset);
    if (target < 0 or target >= @as(i64, @intCast(len))) return error.BadBranchTarget;
    return @intCast(target);
}

fn appendSucc(buf: []Succ, count: *usize, target_pc: u32, kind: EdgeKind) void {
    for (buf[0..count.*]) |succ| {
        if (succ.target_pc == target_pc and succ.kind == kind) return;
    }
    buf[count.*] = .{ .target_pc = target_pc, .kind = kind };
    count.* += 1;
}

fn normalSuccessors(allocator: std.mem.Allocator, insts: []const Instruction, pc: usize, stack_buf: *[2]Succ) Error![]Succ {
    const inst = insts[pc];
    switch (inst) {
        .goto_ => |op| {
            stack_buf[0] = .{ .target_pc = try checkedTarget(pc, op.offset, insts.len), .kind = .branch };
            return stack_buf[0..1];
        },
        .if_eq, .if_ne, .if_lt, .if_ge, .if_gt, .if_le => |op| {
            stack_buf[0] = .{ .target_pc = @intCast(pc + 1), .kind = .fallthrough };
            stack_buf[1] = .{ .target_pc = try checkedTarget(pc, op.offset, insts.len), .kind = .branch };
            return stack_buf[0..2];
        },
        .if_eqz, .if_nez, .if_ltz, .if_gez, .if_gtz, .if_lez => |op| {
            stack_buf[0] = .{ .target_pc = @intCast(pc + 1), .kind = .fallthrough };
            stack_buf[1] = .{ .target_pc = try checkedTarget(pc, op.offset, insts.len), .kind = .branch };
            return stack_buf[0..2];
        },
        .packed_switch, .sparse_switch => |op| {
            const payload = op.payload orelse {
                stack_buf[0] = .{ .target_pc = @intCast(pc + 1), .kind = .fallthrough };
                return stack_buf[0..1];
            };
            const out = try allocator.alloc(Succ, payload.targets.len + 1);
            errdefer allocator.free(out);
            var count: usize = 0;
            appendSucc(out, &count, @intCast(pc + 1), .fallthrough);
            for (payload.targets) |offset| {
                appendSucc(out, &count, try checkedTarget(pc, offset, insts.len), .switch_case);
            }
            return try allocator.realloc(out, count);
        },
        .return_void, .return_, .return_wide, .return_object, .throw_ => return stack_buf[0..0],
        else => {
            if (pc + 1 >= insts.len) return stack_buf[0..0];
            stack_buf[0] = .{ .target_pc = @intCast(pc + 1), .kind = .fallthrough };
            return stack_buf[0..1];
        },
    }
}

fn freeDynamicSuccs(allocator: std.mem.Allocator, inst: Instruction, succs: []Succ) void {
    switch (inst) {
        .packed_switch, .sparse_switch => |op| if (op.payload != null) allocator.free(succs),
        else => {},
    }
}

fn appendExceptionSuccs(allocator: std.mem.Allocator, out: *std.ArrayList(Succ), insts: []const Instruction, tries: []const TryBlock, pc: usize) Error!void {
    if (!canThrow(insts[pc])) return;
    for (tries) |try_block| {
        if (pc < try_block.start_pc or pc >= try_block.end_pc) continue;
        for (try_block.handlers) |handler| {
            try checkedPc(handler.target_pc, insts.len);
            for (out.items) |succ| {
                if (succ.target_pc == handler.target_pc and succ.kind == .exception) break;
            } else {
                try out.append(allocator, .{ .target_pc = handler.target_pc, .kind = .exception });
            }
        }
    }
}

fn blockSuccessors(allocator: std.mem.Allocator, insts: []const Instruction, tries: []const TryBlock, block: RawBlock) Error![]Succ {
    var list = std.ArrayList(Succ).empty;
    errdefer list.deinit(allocator);

    const term_pc: usize = block.end - 1;
    var stack_buf: [2]Succ = undefined;
    const term_succs = try normalSuccessors(allocator, insts, term_pc, &stack_buf);
    defer freeDynamicSuccs(allocator, insts[term_pc], term_succs);
    for (term_succs) |succ| try list.append(allocator, succ);

    var pc: usize = block.start;
    while (pc < block.end) : (pc += 1) {
        try appendExceptionSuccs(allocator, &list, insts, tries, pc);
    }

    return try list.toOwnedSlice(allocator);
}

fn markLeader(leaders: []bool, count: *usize, pc: u32) void {
    if (!leaders[pc]) {
        leaders[pc] = true;
        count.* += 1;
    }
}

fn dfs(blocks: []const BasicBlock, id: BlockId, visited: []bool, postorder: *std.ArrayList(BlockId), allocator: std.mem.Allocator) Error!void {
    if (visited[id]) return;
    visited[id] = true;
    for (blocks[id].successors) |succ| {
        try dfs(blocks, succ, visited, postorder, allocator);
    }
    try postorder.append(allocator, id);
}

fn finishGraph(
    allocator: std.mem.Allocator,
    insts: []const Instruction,
    old_blocks: []BasicBlock,
    old_edges: []Edge,
    old_inst_to_block: []BlockId,
    options: Options,
) Error!Graph {
    const visited = try allocator.alloc(bool, old_blocks.len);
    defer allocator.free(visited);
    @memset(visited, false);

    var postorder = std.ArrayList(BlockId).empty;
    defer postorder.deinit(allocator);
    try dfs(old_blocks, 0, visited, &postorder, allocator);

    const keep_count = if (options.prune_unreachable) postorder.items.len else old_blocks.len;
    const order = try allocator.alloc(BlockId, keep_count);
    errdefer allocator.free(order);
    defer allocator.free(order);

    if (options.order == .rpo) {
        if (options.prune_unreachable) {
            for (postorder.items, 0..) |old_id, i| order[keep_count - 1 - i] = old_id;
        } else {
            var n: usize = 0;
            for (postorder.items, 0..) |old_id, i| {
                _ = i;
                order[postorder.items.len - 1 - n] = old_id;
                n += 1;
            }
            for (old_blocks, 0..) |_, old_id| {
                if (!visited[old_id]) {
                    order[n] = @intCast(old_id);
                    n += 1;
                }
            }
        }
    } else {
        var n: usize = 0;
        for (old_blocks, 0..) |_, old_id| {
            if (!options.prune_unreachable or visited[old_id]) {
                order[n] = @intCast(old_id);
                n += 1;
            }
        }
    }

    const old_to_new = try allocator.alloc(BlockId, old_blocks.len);
    defer allocator.free(old_to_new);
    @memset(old_to_new, INVALID_BLOCK);
    for (order, 0..) |old_id, new_id| old_to_new[old_id] = @intCast(new_id);

    const blocks = try allocator.alloc(BasicBlock, keep_count);
    errdefer allocator.free(blocks);

    const succ_counts = try allocator.alloc(u32, keep_count);
    defer allocator.free(succ_counts);
    const pred_counts = try allocator.alloc(u32, keep_count);
    defer allocator.free(pred_counts);
    @memset(succ_counts, 0);
    @memset(pred_counts, 0);

    var edge_count: usize = 0;
    for (old_edges) |edge| {
        const from = old_to_new[edge.from];
        const to = old_to_new[edge.to];
        if (from == INVALID_BLOCK or to == INVALID_BLOCK) continue;
        succ_counts[from] += 1;
        pred_counts[to] += 1;
        edge_count += 1;
    }

    var allocated_blocks: usize = 0;
    errdefer {
        for (blocks[0..allocated_blocks]) |block| {
            allocator.free(block.successors);
            allocator.free(block.predecessors);
        }
    }
    for (order, 0..) |old_id, new_id| {
        blocks[new_id] = .{
            .id = @intCast(new_id),
            .start = old_blocks[old_id].start,
            .end = old_blocks[old_id].end,
            .rpo_index = if (options.order == .rpo) @intCast(new_id) else INVALID_BLOCK,
            .successors = try allocator.alloc(BlockId, succ_counts[new_id]),
            .predecessors = try allocator.alloc(BlockId, pred_counts[new_id]),
        };
        allocated_blocks += 1;
    }

    const edges = try allocator.alloc(Edge, edge_count);
    errdefer allocator.free(edges);

    @memset(succ_counts, 0);
    @memset(pred_counts, 0);
    var edge_idx: usize = 0;
    for (old_edges) |edge| {
        const from = old_to_new[edge.from];
        const to = old_to_new[edge.to];
        if (from == INVALID_BLOCK or to == INVALID_BLOCK) continue;

        blocks[from].successors[succ_counts[from]] = to;
        blocks[to].predecessors[pred_counts[to]] = from;
        succ_counts[from] += 1;
        pred_counts[to] += 1;
        edges[edge_idx] = .{ .from = from, .to = to, .kind = edge.kind };
        edge_idx += 1;
    }

    const inst_to_block = try allocator.alloc(BlockId, insts.len);
    errdefer allocator.free(inst_to_block);
    @memset(inst_to_block, INVALID_BLOCK);
    for (old_inst_to_block, 0..) |old_id, pc| {
        if (old_id == INVALID_BLOCK) continue;
        const new_id = old_to_new[old_id];
        if (new_id != INVALID_BLOCK) inst_to_block[pc] = new_id;
    }

    const rpo = try allocator.alloc(BlockId, keep_count);
    errdefer allocator.free(rpo);
    if (options.order == .rpo) {
        for (rpo, 0..) |*slot, i| slot.* = @intCast(i);
    } else {
        for (blocks, 0..) |block, i| {
            _ = block;
            rpo[i] = @intCast(i);
        }
    }

    return .{
        .allocator = allocator,
        .instructions = insts,
        .blocks = blocks,
        .edges = edges,
        .inst_to_block = inst_to_block,
        .rpo = rpo,
        .entry = old_to_new[0],
    };
}

pub fn build(allocator: std.mem.Allocator, insts: []const Instruction) Error!Graph {
    return buildWithOptions(allocator, insts, &.{}, .{});
}

pub fn buildWithTries(allocator: std.mem.Allocator, insts: []const Instruction, tries: []const TryBlock) Error!Graph {
    return buildWithOptions(allocator, insts, tries, .{});
}

pub fn buildWithOptions(allocator: std.mem.Allocator, insts: []const Instruction, tries: []const TryBlock, options: Options) Error!Graph {
    if (insts.len == 0) return error.EmptyInstructionStream;
    if (insts.len > std.math.maxInt(u32)) return error.BadBranchTarget;

    const leaders = try allocator.alloc(bool, insts.len);
    defer allocator.free(leaders);
    @memset(leaders, false);
    leaders[0] = true;
    var block_count: usize = 1;

    for (tries) |try_block| {
        if (try_block.start_pc >= insts.len or try_block.end_pc > insts.len or try_block.start_pc > try_block.end_pc) return error.BadBranchTarget;
        if (try_block.start_pc < insts.len) markLeader(leaders, &block_count, try_block.start_pc);
        if (try_block.end_pc < insts.len) markLeader(leaders, &block_count, try_block.end_pc);
        for (try_block.handlers) |handler| {
            try checkedPc(handler.target_pc, insts.len);
            markLeader(leaders, &block_count, handler.target_pc);
        }
    }

    for (insts, 0..) |inst, pc| {
        var stack_buf: [2]Succ = undefined;
        const succs = try normalSuccessors(allocator, insts, pc, &stack_buf);
        defer freeDynamicSuccs(allocator, inst, succs);

        for (succs) |succ| {
            if (successorIsLeader(inst, succ.kind)) markLeader(leaders, &block_count, succ.target_pc);
        }
        if (!hasFallthrough(inst) and pc + 1 < insts.len) markLeader(leaders, &block_count, @intCast(pc + 1));
    }

    const raw_blocks = try allocator.alloc(RawBlock, block_count);
    defer allocator.free(raw_blocks);
    const old_inst_to_block = try allocator.alloc(BlockId, insts.len);
    defer allocator.free(old_inst_to_block);
    @memset(old_inst_to_block, INVALID_BLOCK);

    var raw_count: usize = 0;
    var start: usize = 0;
    while (start < insts.len) {
        while (start < insts.len and !leaders[start]) start += 1;
        if (start >= insts.len) break;
        var end = start + 1;
        while (end < insts.len and !leaders[end]) end += 1;
        raw_blocks[raw_count] = .{ .start = @intCast(start), .end = @intCast(end) };
        for (start..end) |pc| old_inst_to_block[pc] = @intCast(raw_count);
        raw_count += 1;
        start = end;
    }

    const old_blocks = try allocator.alloc(BasicBlock, raw_count);
    defer {
        for (old_blocks) |block| {
            allocator.free(block.successors);
            allocator.free(block.predecessors);
        }
        allocator.free(old_blocks);
    }

    const succ_counts = try allocator.alloc(u32, raw_count);
    defer allocator.free(succ_counts);
    const pred_counts = try allocator.alloc(u32, raw_count);
    defer allocator.free(pred_counts);
    @memset(succ_counts, 0);
    @memset(pred_counts, 0);

    var edge_count: usize = 0;
    for (raw_blocks[0..raw_count], 0..) |block, from| {
        const succs = try blockSuccessors(allocator, insts, tries, block);
        defer allocator.free(succs);
        for (succs) |succ| {
            const to = old_inst_to_block[succ.target_pc];
            if (to == INVALID_BLOCK) return error.BadBranchTarget;
            succ_counts[from] += 1;
            pred_counts[to] += 1;
            edge_count += 1;
        }
    }

    var allocated_blocks: usize = 0;
    errdefer {
        for (old_blocks[0..allocated_blocks]) |block| {
            allocator.free(block.successors);
            allocator.free(block.predecessors);
        }
    }
    for (raw_blocks[0..raw_count], 0..) |raw, id| {
        old_blocks[id] = .{
            .id = @intCast(id),
            .start = raw.start,
            .end = raw.end,
            .rpo_index = INVALID_BLOCK,
            .successors = try allocator.alloc(BlockId, succ_counts[id]),
            .predecessors = try allocator.alloc(BlockId, pred_counts[id]),
        };
        allocated_blocks += 1;
    }

    const old_edges = try allocator.alloc(Edge, edge_count);
    defer allocator.free(old_edges);
    @memset(succ_counts, 0);
    @memset(pred_counts, 0);
    var edge_idx: usize = 0;
    for (raw_blocks[0..raw_count], 0..) |block, from| {
        const succs = try blockSuccessors(allocator, insts, tries, block);
        defer allocator.free(succs);
        for (succs) |succ| {
            const to = old_inst_to_block[succ.target_pc];
            old_blocks[from].successors[succ_counts[from]] = to;
            old_blocks[to].predecessors[pred_counts[to]] = @intCast(from);
            succ_counts[from] += 1;
            pred_counts[to] += 1;
            old_edges[edge_idx] = .{ .from = @intCast(from), .to = to, .kind = succ.kind };
            edge_idx += 1;
        }
    }

    return finishGraph(allocator, insts, old_blocks, old_edges, old_inst_to_block, options);
}

test "cfg builds single straight-line block" {
    const insts = [_]Instruction{
        .nop,
        .{ .const_ = .{ .dest = 0, .value = 1 } },
        .{ .return_ = .{ .src = 0 } },
    };

    var graph = try build(std.testing.allocator, &insts);
    defer graph.deinit();

    try std.testing.expectEqual(@as(usize, 1), graph.blocks.len);
    try std.testing.expectEqual(@as(u32, 0), graph.entryBlock().start);
    try std.testing.expectEqual(@as(u32, 3), graph.entryBlock().end);
    try std.testing.expectEqual(@as(usize, 0), graph.edges.len);
    try std.testing.expectEqual(@as(BlockId, 0), graph.inst_to_block[2]);
}

test "cfg splits conditional branch and stores blocks in rpo" {
    const insts = [_]Instruction{
        .{ .if_eqz = .{ .src = 0, .offset = 2 } },
        .{ .const_ = .{ .dest = 1, .value = 1 } },
        .{ .const_ = .{ .dest = 1, .value = 2 } },
        .{ .return_ = .{ .src = 1 } },
    };

    var graph = try build(std.testing.allocator, &insts);
    defer graph.deinit();

    try std.testing.expectEqual(@as(usize, 3), graph.blocks.len);
    try std.testing.expectEqual(@as(u32, 0), graph.blocks[0].start);
    try std.testing.expectEqualSlices(BlockId, &[_]BlockId{ 1, 2 }, graph.blocks[0].successors);
    try std.testing.expectEqual(@as(u32, 0), graph.blocks[0].rpo_index);
    try std.testing.expectEqual(@as(u32, 1), graph.blocks[1].rpo_index);
    try std.testing.expectEqual(@as(u32, 2), graph.blocks[2].rpo_index);
    try std.testing.expectEqual(EdgeKind.fallthrough, graph.edges[0].kind);
    try std.testing.expectEqual(EdgeKind.branch, graph.edges[1].kind);
}

test "cfg prunes unreachable block after goto by default" {
    const insts = [_]Instruction{
        .{ .goto_ = .{ .offset = 2 } },
        .{ .const_ = .{ .dest = 0, .value = 99 } },
        .return_void,
    };

    var graph = try build(std.testing.allocator, &insts);
    defer graph.deinit();

    try std.testing.expectEqual(@as(usize, 2), graph.blocks.len);
    try std.testing.expectEqual(@as(BlockId, INVALID_BLOCK), graph.inst_to_block[1]);
    try std.testing.expectEqualSlices(BlockId, &[_]BlockId{1}, graph.blocks[0].successors);
}

test "cfg can retain unreachable blocks when requested" {
    const insts = [_]Instruction{
        .{ .goto_ = .{ .offset = 2 } },
        .{ .const_ = .{ .dest = 0, .value = 99 } },
        .return_void,
    };

    var graph = try buildWithOptions(std.testing.allocator, &insts, &.{}, .{ .prune_unreachable = false, .order = .linear });
    defer graph.deinit();

    try std.testing.expectEqual(@as(usize, 3), graph.blocks.len);
    try std.testing.expectEqual(@as(BlockId, 1), graph.inst_to_block[1]);
    try std.testing.expectEqual(@as(usize, 0), graph.blocks[1].predecessors.len);
}

test "cfg deduplicates switch targets and preserves fallthrough" {
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
        .{ .return_ = .{ .src = 1 } },
    };

    var graph = try build(std.testing.allocator, &insts);
    defer graph.deinit();

    try std.testing.expectEqual(@as(usize, 5), graph.blocks.len);
    try std.testing.expectEqual(@as(usize, 3), graph.blocks[0].successors.len);
    try std.testing.expectEqual(EdgeKind.fallthrough, graph.edges[0].kind);
    try std.testing.expectEqual(EdgeKind.switch_case, graph.edges[1].kind);
    try std.testing.expectEqual(EdgeKind.switch_case, graph.edges[2].kind);
}

test "cfg switch without payload falls through" {
    const insts = [_]Instruction{
        .{ .sparse_switch = .{ .src = 0, .payload_offset = 0, .payload = null } },
        .return_void,
    };

    var graph = try build(std.testing.allocator, &insts);
    defer graph.deinit();

    try std.testing.expectEqual(@as(usize, 2), graph.blocks.len);
    try std.testing.expectEqualSlices(BlockId, &[_]BlockId{1}, graph.blocks[0].successors);
}

test "cfg wires try catch exception edges" {
    const handlers = [_]instmod.CatchHandler{
        .{ .type_idx = 1, .target_pc = 3 },
        .{ .type_idx = null, .target_pc = 4 },
    };
    const tries = [_]TryBlock{
        .{ .start_pc = 0, .end_pc = 2, .handlers = &handlers },
    };
    const insts = [_]Instruction{
        .{ .const_ = .{ .dest = 0, .value = 10 } },
        .{ .div_int = .{ .dest = 1, .src1 = 0, .src2 = 2 } },
        .return_void,
        .{ .const_ = .{ .dest = 3, .value = 1 } },
        .return_void,
    };

    var graph = try buildWithTries(std.testing.allocator, &insts, &tries);
    defer graph.deinit();

    const throwing_block = graph.blockForPc(0).?;
    try std.testing.expectEqual(@as(usize, 3), throwing_block.successors.len);
    var exception_edges: usize = 0;
    for (graph.edges) |edge| {
        if (edge.from == throwing_block.id and edge.kind == .exception) exception_edges += 1;
    }
    try std.testing.expectEqual(@as(usize, 2), exception_edges);
    try std.testing.expect(graph.blockForPc(3) != null);
    try std.testing.expect(graph.blockForPc(4) != null);
}

test "cfg ignores non throwing instructions inside try for exception edges" {
    const handlers = [_]instmod.CatchHandler{.{ .type_idx = 1, .target_pc = 2 }};
    const tries = [_]TryBlock{.{ .start_pc = 0, .end_pc = 1, .handlers = &handlers }};
    const insts = [_]Instruction{
        .{ .const_ = .{ .dest = 0, .value = 10 } },
        .return_void,
        .return_void,
    };

    var graph = try buildWithTries(std.testing.allocator, &insts, &tries);
    defer graph.deinit();

    for (graph.edges) |edge| {
        try std.testing.expect(edge.kind != .exception);
    }
}

test "cfg rejects bad targets and empty instruction streams" {
    try std.testing.expectError(error.EmptyInstructionStream, build(std.testing.allocator, &.{}));

    const bad_negative = [_]Instruction{.{ .goto_ = .{ .offset = -1 } }};
    try std.testing.expectError(error.BadBranchTarget, build(std.testing.allocator, &bad_negative));

    const bad_positive = [_]Instruction{
        .{ .if_nez = .{ .src = 0, .offset = 4 } },
        .return_void,
    };
    try std.testing.expectError(error.BadBranchTarget, build(std.testing.allocator, &bad_positive));

    const handlers = [_]instmod.CatchHandler{.{ .type_idx = 1, .target_pc = 9 }};
    const tries = [_]TryBlock{.{ .start_pc = 0, .end_pc = 1, .handlers = &handlers }};
    const insts = [_]Instruction{ .{ .div_int = .{ .dest = 0, .src1 = 0, .src2 = 1 } }, .return_void };
    try std.testing.expectError(error.BadBranchTarget, buildWithTries(std.testing.allocator, &insts, &tries));
}

test "cfg print helper emits stable summary" {
    const insts = [_]Instruction{
        .{ .if_eqz = .{ .src = 0, .offset = 2 } },
        .{ .const_ = .{ .dest = 1, .value = 1 } },
        .return_void,
    };

    var graph = try build(std.testing.allocator, &insts);
    defer graph.deinit();

    var buf: [512]u8 = undefined;
    var stream = std.Io.Writer.fixed(&buf);
    try graph.print(&stream);
    const output = stream.buffered();

    try std.testing.expect(std.mem.indexOf(u8, output, "cfg blocks=3 edges=3 entry=b0") != null);
    try std.testing.expect(std.mem.indexOf(u8, output, "b0 pc=[0,1)") != null);
    try std.testing.expect(std.mem.indexOf(u8, output, "edges:") != null);
    try std.testing.expect(std.mem.indexOf(u8, output, "fallthrough") != null);
    try std.testing.expect(std.mem.indexOf(u8, output, "branch") != null);
}
