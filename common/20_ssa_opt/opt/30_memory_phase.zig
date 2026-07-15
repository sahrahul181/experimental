//! Memory/object optimization phase metadata.
//!
//! This pass is conservative and does not rewrite SSA. It records alias classes,
//! redundant loads, dead stores, non-escaping allocations, scalar replacement
//! candidates, and field forwarding opportunities for lowering/codegen.

const std = @import("std");
const cfg = @import("cfg");
const dom = @import("dominator");
const ssa = @import("ssa");
const typedir = @import("typedir");
const instmod = @import("instructions");
const Instruction = instmod.Instruction;

pub const Error = error{
    InvalidInput,
    OutOfMemory,
};

pub const AliasKind = enum(u8) {
    unknown,
    object_alloc,
    array_alloc,
    static_memory,
};

pub const ValueInfo = struct {
    alias: AliasKind = .unknown,
    allocation_pc: ?u32 = null,
    escapes: bool = false,
    scalar_replace: bool = false,
};

pub const OpInfo = struct {
    redundant_load: bool = false,
    dead_store: bool = false,
    forwarded_value: ?ssa.ValueId = null,
    alias_barrier: bool = false,
};

pub const Stats = struct {
    allocations: u32 = 0,
    non_escaping: u32 = 0,
    scalar_replacements: u32 = 0,
    redundant_loads: u32 = 0,
    dead_stores: u32 = 0,
    forwarded_fields: u32 = 0,
};

const MemKind = enum(u8) {
    instance,
    static,
};

const MemKey = struct {
    kind: MemKind,
    base: ssa.ValueId,
    field: u32,
};

const StoreRef = struct {
    op: ssa.OpRef,
    value: ssa.ValueId,
    live: bool = false,
};

const StoreInfo = struct {
    key: MemKey,
    value: ssa.ValueId,
};

const LoadRef = struct {
    op: ssa.OpRef,
    value: ssa.ValueId,
};

pub const Result = struct {
    allocator: std.mem.Allocator,
    function: *const ssa.Function,
    types: *const typedir.Function,
    values: []ValueInfo,
    ops: [][]OpInfo,
    stats: Stats,

    pub fn deinit(self: *Result) void {
        for (self.ops) |ops| self.allocator.free(ops);
        self.allocator.free(self.ops);
        self.allocator.free(self.values);
        self.* = undefined;
    }

    pub fn print(self: *const Result, writer: anytype) !void {
        try writer.print(
            "memory_phase values={d} allocations={d} non_escaping={d} scalar={d} redundant_loads={d} dead_stores={d} forwarded={d}\n",
            .{
                self.values.len,
                self.stats.allocations,
                self.stats.non_escaping,
                self.stats.scalar_replacements,
                self.stats.redundant_loads,
                self.stats.dead_stores,
                self.stats.forwarded_fields,
            },
        );
        for (self.function.graph.rpo) |block_id| {
            const block = self.function.blocks[block_id];
            try writer.print("b{d}\n", .{block_id});
            for (block.ops, 0..) |op, i| {
                const info = self.ops[block_id][i];
                try writer.print(
                    "  pc{d} {s} redundant_load={} dead_store={} barrier={}",
                    .{ op.pc, @tagName(op.inst), info.redundant_load, info.dead_store, info.alias_barrier },
                );
                if (info.forwarded_value) |value| try writer.print(" forward=v{d}", .{value});
                try writer.print("\n", .{});
            }
        }
        try writer.print("values:\n", .{});
        for (self.values, 0..) |value, i| {
            if (value.alias == .unknown and !value.escapes and !value.scalar_replace) continue;
            try writer.print("  v{d} alias={s} escapes={} scalar={}", .{ i, @tagName(value.alias), value.escapes, value.scalar_replace });
            if (value.allocation_pc) |pc| try writer.print(" alloc_pc={d}", .{pc});
            try writer.print("\n", .{});
        }
    }
};

fn canonicalCopy(function: *const ssa.Function, value: ssa.ValueId) ssa.ValueId {
    if (value >= function.values.len) return value;
    return value;
}

fn isObjectAllocation(inst: Instruction) bool {
    return switch (inst) {
        .new_instance => true,
        else => false,
    };
}

fn isArrayAllocation(inst: Instruction) bool {
    return switch (inst) {
        .new_array, .filled_new_array => true,
        else => false,
    };
}

fn instanceLoad(op: ssa.Operation) ?MemKey {
    return switch (op.inst) {
        .iget,
        .iget_wide,
        .iget_object,
        .iget_boolean,
        .iget_byte,
        .iget_char,
        .iget_short,
        .iget_quick,
        .iget_wide_quick,
        .iget_object_quick,
        => |inst| if (op.uses.len >= 1) .{ .kind = .instance, .base = op.uses[0], .field = inst.field_idx } else null,
        else => null,
    };
}

fn staticLoad(op: ssa.Operation) ?MemKey {
    return switch (op.inst) {
        .sget,
        .sget_wide,
        .sget_object,
        .sget_boolean,
        .sget_byte,
        .sget_char,
        .sget_short,
        => |inst| .{ .kind = .static, .base = 0, .field = inst.field_idx },
        else => null,
    };
}

fn instanceStore(op: ssa.Operation) ?StoreInfo {
    return switch (op.inst) {
        .iput,
        .iput_object,
        .iput_boolean,
        .iput_byte,
        .iput_char,
        .iput_short,
        .iput_quick,
        .iput_object_quick,
        => |inst| if (op.uses.len >= 2) .{
            .key = .{ .kind = .instance, .base = op.uses[op.uses.len - 1], .field = inst.field_idx },
            .value = op.uses[0],
        } else null,
        .iput_wide, .iput_wide_quick => |inst| if (op.uses.len >= 3) .{
            .key = .{ .kind = .instance, .base = op.uses[op.uses.len - 1], .field = inst.field_idx },
            .value = op.uses[0],
        } else null,
        else => null,
    };
}

fn staticStore(op: ssa.Operation) ?StoreInfo {
    return switch (op.inst) {
        .sput,
        .sput_object,
        .sput_boolean,
        .sput_byte,
        .sput_char,
        .sput_short,
        => |inst| if (op.uses.len >= 1) .{ .key = .{ .kind = .static, .base = 0, .field = inst.field_idx }, .value = op.uses[0] } else null,
        .sput_wide => |inst| if (op.uses.len >= 1) .{ .key = .{ .kind = .static, .base = 0, .field = inst.field_idx }, .value = op.uses[0] } else null,
        else => null,
    };
}

fn loadKey(op: ssa.Operation) ?MemKey {
    return instanceLoad(op) orelse staticLoad(op);
}

fn storeInfo(op: ssa.Operation) ?StoreInfo {
    return instanceStore(op) orelse staticStore(op);
}

fn opIsAliasBarrier(op: ssa.Operation) bool {
    return switch (op.inst) {
        .invoke,
        .invoke_virtual_quick,
        .invoke_super_quick,
        .throw_,
        .monitor_enter,
        .monitor_exit,
        .fill_array_data,
        .aput,
        .aput_wide,
        .aput_object,
        .aput_boolean,
        .aput_byte,
        .aput_char,
        .aput_short,
        => true,
        else => false,
    };
}

fn markEscapesFromUse(values: []ValueInfo, op: ssa.Operation) void {
    switch (op.inst) {
        .return_object, .throw_, .sput_object, .aput_object, .invoke, .invoke_virtual_quick, .invoke_super_quick => {
            for (op.uses) |use| {
                if (use < values.len and values[use].alias != .unknown) values[use].escapes = true;
            }
        },
        .iput_object, .iput_object_quick => {
            if (op.uses.len > 0 and op.uses[0] < values.len and values[op.uses[0]].alias != .unknown) values[op.uses[0]].escapes = true;
        },
        else => {},
    }
}

fn allocationHasOnlyScalarUses(function: *const ssa.Function, values: []const ValueInfo, allocation: ssa.ValueId) bool {
    _ = values;
    for (function.blocks) |block| {
        for (block.ops) |op| {
            for (op.uses) |use| {
                if (use != allocation) continue;
                switch (op.inst) {
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
                    .check_cast,
                    => {},
                    else => return false,
                }
            }
        }
    }
    return true;
}

pub fn run(
    allocator: std.mem.Allocator,
    function: *const ssa.Function,
    types: *const typedir.Function,
) Error!Result {
    function.verify() catch |err| switch (err) {
        error.OutOfMemory => return error.OutOfMemory,
        else => return error.InvalidInput,
    };
    types.verify() catch |err| switch (err) {
        error.OutOfMemory => return error.OutOfMemory,
        else => return error.InvalidInput,
    };
    if (types.source != function) return error.InvalidInput;

    const values = try allocator.alloc(ValueInfo, function.values.len);
    errdefer allocator.free(values);
    @memset(values, .{});

    const ops = try allocator.alloc([]OpInfo, function.blocks.len);
    errdefer allocator.free(ops);
    var built_ops: usize = 0;
    errdefer for (ops[0..built_ops]) |slice| allocator.free(slice);
    for (function.blocks, 0..) |block, i| {
        ops[i] = try allocator.alloc(OpInfo, block.ops.len);
        @memset(ops[i], .{});
        built_ops += 1;
    }

    var stats: Stats = .{};
    for (function.blocks) |block| {
        for (block.ops) |op| {
            if (op.defs.len == 0) continue;
            if (isObjectAllocation(op.inst)) {
                values[op.defs[0]] = .{ .alias = .object_alloc, .allocation_pc = op.pc };
                stats.allocations += 1;
            } else if (isArrayAllocation(op.inst)) {
                values[op.defs[0]] = .{ .alias = .array_alloc, .allocation_pc = op.pc };
                stats.allocations += 1;
            }
        }
    }

    for (function.blocks) |block| {
        for (block.ops) |op| markEscapesFromUse(values, op);
    }

    var last_load = std.AutoHashMap(MemKey, LoadRef).init(allocator);
    defer last_load.deinit();
    var last_store = std.AutoHashMap(MemKey, StoreRef).init(allocator);
    defer last_store.deinit();

    for (function.graph.rpo) |block_id| {
        const block = function.blocks[block_id];
        for (block.ops, 0..) |op, i| {
            if (opIsAliasBarrier(op)) {
                ops[block_id][i].alias_barrier = true;
                last_load.clearRetainingCapacity();
                last_store.clearRetainingCapacity();
            }

            if (loadKey(op)) |raw_key| {
                var key = raw_key;
                key.base = canonicalCopy(function, key.base);
                if (last_load.get(key)) |_| {
                    ops[block_id][i].redundant_load = true;
                    stats.redundant_loads += 1;
                }
                if (last_store.getPtr(key)) |store| {
                    store.live = true;
                    ops[block_id][i].forwarded_value = store.value;
                    stats.forwarded_fields += 1;
                }
                if (op.defs.len > 0) try last_load.put(key, .{ .op = .{ .block = block_id, .index = @intCast(i) }, .value = op.defs[0] });
            }

            if (storeInfo(op)) |raw_store| {
                var store = raw_store;
                store.key.base = canonicalCopy(function, store.key.base);
                if (last_store.get(store.key)) |prior| {
                    if (!prior.live) {
                        ops[prior.op.block][prior.op.index].dead_store = true;
                        stats.dead_stores += 1;
                    }
                }
                _ = last_load.remove(store.key);
                try last_store.put(store.key, .{ .op = .{ .block = block_id, .index = @intCast(i) }, .value = store.value });
            }
        }
    }

    for (values, 0..) |*value, i| {
        if (value.alias == .unknown) continue;
        if (!value.escapes) stats.non_escaping += 1;
        if (!value.escapes and value.alias == .object_alloc and allocationHasOnlyScalarUses(function, values, @intCast(i))) {
            value.scalar_replace = true;
            stats.scalar_replacements += 1;
        }
    }

    return .{
        .allocator = allocator,
        .function = function,
        .types = types,
        .values = values,
        .ops = ops,
        .stats = stats,
    };
}

fn buildPipeline(insts: []const Instruction, graph: *cfg.Graph, tree: *dom.Tree, function: *ssa.Function, types: *typedir.Function) !void {
    graph.* = try cfg.build(std.testing.allocator, insts);
    errdefer graph.deinit();
    tree.* = try dom.build(std.testing.allocator, graph);
    errdefer tree.deinit();
    function.* = try ssa.build(std.testing.allocator, graph, tree);
    errdefer function.deinit();
    types.* = try typedir.build(std.testing.allocator, function);
}

test "memory_phase detects redundant instance loads and forwarding" {
    const insts = [_]Instruction{
        .{ .new_instance = .{ .dest = 0, .type_idx = 1 } },
        .{ .const_ = .{ .dest = 1, .value = 7 } },
        .{ .iput = .{ .field_idx = 10, .dest_or_src = 1, .obj = 0 } },
        .{ .iget = .{ .field_idx = 10, .dest_or_src = 2, .obj = 0 } },
        .{ .iget = .{ .field_idx = 10, .dest_or_src = 3, .obj = 0 } },
        .return_void,
    };
    var graph: cfg.Graph = undefined;
    var tree: dom.Tree = undefined;
    var function: ssa.Function = undefined;
    var types: typedir.Function = undefined;
    try buildPipeline(&insts, &graph, &tree, &function, &types);
    defer types.deinit();
    defer function.deinit();
    defer tree.deinit();
    defer graph.deinit();
    var result = try run(std.testing.allocator, &function, &types);
    defer result.deinit();

    try std.testing.expect(result.ops[graph.entry][4].redundant_load);
    try std.testing.expect(result.ops[graph.entry][3].forwarded_value != null);
    try std.testing.expect(result.stats.forwarded_fields >= 1);
}

test "memory_phase detects dead overwritten stores" {
    const insts = [_]Instruction{
        .{ .new_instance = .{ .dest = 0, .type_idx = 1 } },
        .{ .const_ = .{ .dest = 1, .value = 1 } },
        .{ .iput = .{ .field_idx = 3, .dest_or_src = 1, .obj = 0 } },
        .{ .const_ = .{ .dest = 2, .value = 2 } },
        .{ .iput = .{ .field_idx = 3, .dest_or_src = 2, .obj = 0 } },
        .return_void,
    };
    var graph: cfg.Graph = undefined;
    var tree: dom.Tree = undefined;
    var function: ssa.Function = undefined;
    var types: typedir.Function = undefined;
    try buildPipeline(&insts, &graph, &tree, &function, &types);
    defer types.deinit();
    defer function.deinit();
    defer tree.deinit();
    defer graph.deinit();
    var result = try run(std.testing.allocator, &function, &types);
    defer result.deinit();

    try std.testing.expect(result.ops[graph.entry][2].dead_store);
    try std.testing.expectEqual(@as(u32, 1), result.stats.dead_stores);
}

test "memory_phase detects escaping and scalar replaceable allocations" {
    const local_insts = [_]Instruction{
        .{ .new_instance = .{ .dest = 0, .type_idx = 1 } },
        .{ .const_ = .{ .dest = 1, .value = 1 } },
        .{ .iput = .{ .field_idx = 3, .dest_or_src = 1, .obj = 0 } },
        .return_void,
    };
    var graph: cfg.Graph = undefined;
    var tree: dom.Tree = undefined;
    var function: ssa.Function = undefined;
    var types: typedir.Function = undefined;
    try buildPipeline(&local_insts, &graph, &tree, &function, &types);
    defer types.deinit();
    defer function.deinit();
    defer tree.deinit();
    defer graph.deinit();
    var result = try run(std.testing.allocator, &function, &types);
    defer result.deinit();

    const alloc = function.blocks[graph.entry].ops[0].defs[0];
    try std.testing.expect(!result.values[alloc].escapes);
    try std.testing.expect(result.values[alloc].scalar_replace);

    const escape_insts = [_]Instruction{
        .{ .new_instance = .{ .dest = 0, .type_idx = 1 } },
        .{ .return_object = .{ .src = 0 } },
    };
    var graph2: cfg.Graph = undefined;
    var tree2: dom.Tree = undefined;
    var function2: ssa.Function = undefined;
    var types2: typedir.Function = undefined;
    try buildPipeline(&escape_insts, &graph2, &tree2, &function2, &types2);
    defer types2.deinit();
    defer function2.deinit();
    defer tree2.deinit();
    defer graph2.deinit();
    var result2 = try run(std.testing.allocator, &function2, &types2);
    defer result2.deinit();

    const escaped = function2.blocks[graph2.entry].ops[0].defs[0];
    try std.testing.expect(result2.values[escaped].escapes);
    try std.testing.expect(!result2.values[escaped].scalar_replace);
}

test "memory_phase handles static field simplification" {
    const insts = [_]Instruction{
        .{ .const_ = .{ .dest = 0, .value = 11 } },
        .{ .sput = .{ .field_idx = 8, .dest_or_src = 0 } },
        .{ .sget = .{ .field_idx = 8, .dest_or_src = 1 } },
        .{ .sget = .{ .field_idx = 8, .dest_or_src = 2 } },
        .return_void,
    };
    var graph: cfg.Graph = undefined;
    var tree: dom.Tree = undefined;
    var function: ssa.Function = undefined;
    var types: typedir.Function = undefined;
    try buildPipeline(&insts, &graph, &tree, &function, &types);
    defer types.deinit();
    defer function.deinit();
    defer tree.deinit();
    defer graph.deinit();
    var result = try run(std.testing.allocator, &function, &types);
    defer result.deinit();

    try std.testing.expect(result.ops[graph.entry][2].forwarded_value != null);
    try std.testing.expect(result.ops[graph.entry][3].redundant_load);
}

test "memory_phase print helper emits stable summary" {
    const insts = [_]Instruction{
        .{ .new_instance = .{ .dest = 0, .type_idx = 1 } },
        .return_void,
    };
    var graph: cfg.Graph = undefined;
    var tree: dom.Tree = undefined;
    var function: ssa.Function = undefined;
    var types: typedir.Function = undefined;
    try buildPipeline(&insts, &graph, &tree, &function, &types);
    defer types.deinit();
    defer function.deinit();
    defer tree.deinit();
    defer graph.deinit();
    var result = try run(std.testing.allocator, &function, &types);
    defer result.deinit();

    var buf: [512]u8 = undefined;
    var stream = std.Io.Writer.fixed(&buf);
    try result.print(&stream);
    const output = stream.buffered();
    try std.testing.expect(std.mem.indexOf(u8, output, "memory_phase values=") != null);
    try std.testing.expect(std.mem.indexOf(u8, output, "allocations=") != null);
}
