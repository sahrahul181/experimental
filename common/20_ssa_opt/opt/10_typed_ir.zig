//! Typed IR analysis layer.
//!
//! This pass sits after SSA type discovery and early optimizer facts. It does
//! not rewrite SSA; it records production lowering decisions and conservative
//! safety-elision hints that the next lowering/codegen phase can consume.

const std = @import("std");
const cfg = @import("cfg");
const dom = @import("dominator");
const ssa = @import("ssa");
const typedir = @import("typedir");
const optimizer = @import("optimizer");
const instmod = @import("instructions");
const Instruction = instmod.Instruction;

pub const Error = error{
    InvalidInput,
    OutOfMemory,
};

pub const VerifyError = error{
    BadArrayBounds,
    BadDevirtHint,
    BadLowering,
    BadNullElision,
    InvalidInput,
    OutOfMemory,
};

pub const Nullness = enum(u8) {
    unknown,
    null_value,
    non_null,
    conflict,

    pub fn merge(a: Nullness, b: Nullness) Nullness {
        if (a == b) return a;
        if (a == .unknown) return b;
        if (b == .unknown) return a;
        return .conflict;
    }
};

pub const DevirtHint = enum(u8) {
    none,
    direct_exact,
    static_exact,
    quickened_virtual,
    quickened_super,
};

pub const LoweringChoice = enum(u8) {
    none,
    int32,
    int64,
    float32,
    float64,
    reference,
    branch,
    call,
    memory,
    throw_path,
    return_path,
};

pub const ValueInfo = struct {
    ty: typedir.Type,
    nullness: Nullness = .unknown,
    array_length: ?u32 = null,
};

pub const OpInfo = struct {
    type_simplified: bool = false,
    null_check_elided: bool = false,
    bounds_check_elided: bool = false,
    devirt: DevirtHint = .none,
    lowering: LoweringChoice = .none,
};

pub const Stats = struct {
    type_simplifications: u32 = 0,
    null_checks_elided: u32 = 0,
    bounds_checks_elided: u32 = 0,
    devirt_hints: u32 = 0,
};

pub const Function = struct {
    allocator: std.mem.Allocator,
    source: *const ssa.Function,
    types: *const typedir.Function,
    opt: ?*const optimizer.Result,
    values: []ValueInfo,
    ops: [][]OpInfo,
    stats: Stats,

    pub fn deinit(self: *Function) void {
        for (self.ops) |ops| self.allocator.free(ops);
        self.allocator.free(self.ops);
        self.allocator.free(self.values);
        self.* = undefined;
    }

    pub inline fn valueInfo(self: *const Function, value: ssa.ValueId) ?ValueInfo {
        if (value >= self.values.len) return null;
        return self.values[value];
    }

    pub inline fn opInfo(self: *const Function, block: cfg.BlockId, index: usize) ?OpInfo {
        if (block >= self.ops.len or index >= self.ops[block].len) return null;
        return self.ops[block][index];
    }

    pub fn verify(self: *const Function) VerifyError!void {
        self.source.verify() catch |err| switch (err) {
            error.OutOfMemory => return error.OutOfMemory,
            else => return error.InvalidInput,
        };
        self.types.verify() catch |err| switch (err) {
            error.OutOfMemory => return error.OutOfMemory,
            else => return error.InvalidInput,
        };
        if (self.values.len != self.source.values.len or self.ops.len != self.source.blocks.len) return error.InvalidInput;
        if (self.opt) |opt_result| {
            if (opt_result.function != self.source or opt_result.facts.len != self.source.values.len) return error.InvalidInput;
        }

        for (self.values, 0..) |value, i| {
            const expected = self.types.typeOf(@intCast(i)) orelse return error.InvalidInput;
            if (value.ty != expected) return error.InvalidInput;
            if (value.array_length != null and value.nullness != .non_null) return error.BadArrayBounds;
        }

        for (self.source.blocks) |block| {
            if (self.ops[block.id].len != block.ops.len) return error.InvalidInput;
            for (block.ops, 0..) |op, i| {
                const info = self.ops[block.id][i];
                if (info.null_check_elided and !opHasNonNullObjectUse(self, op)) return error.BadNullElision;
                if (info.bounds_check_elided and !opHasProvableBounds(self, op)) return error.BadArrayBounds;
                if (info.devirt != .none and !opCanHaveDevirt(op.inst)) return error.BadDevirtHint;
                if (info.lowering == .none and op.inst != .nop) return error.BadLowering;
            }
        }
    }

    pub fn print(self: *const Function, writer: anytype) !void {
        try writer.print(
            "typed_ir values={d} blocks={d} type_simplified={d} null_elided={d} bounds_elided={d} devirt={d}\n",
            .{
                self.values.len,
                self.source.blocks.len,
                self.stats.type_simplifications,
                self.stats.null_checks_elided,
                self.stats.bounds_checks_elided,
                self.stats.devirt_hints,
            },
        );

        for (self.source.graph.rpo) |block_id| {
            const block = self.source.blocks[block_id];
            try writer.print("b{d}\n", .{block_id});
            for (block.phis) |phi| {
                const info = self.values[phi.dest];
                try writer.print("  v{d}:{s} null={s}", .{ phi.dest, @tagName(info.ty), @tagName(info.nullness) });
                if (info.array_length) |len| try writer.print(" len={d}", .{len});
                try writer.print(" phi\n", .{});
            }
            for (block.ops, 0..) |op, i| {
                const info = self.ops[block_id][i];
                try writer.print(
                    "  pc{d} {s} lower={s} null_elide={} bounds_elide={} devirt={s}",
                    .{ op.pc, @tagName(op.inst), @tagName(info.lowering), info.null_check_elided, info.bounds_check_elided, @tagName(info.devirt) },
                );
                if (info.type_simplified) try writer.print(" type_simplified", .{});
                try writer.print("\n", .{});
            }
        }
    }
};

fn intConstant(opt: ?*const optimizer.Result, value: ssa.ValueId) ?i32 {
    const opt_result = opt orelse return null;
    if (value >= opt_result.facts.len) return null;
    const constant = opt_result.facts[value].constant orelse return null;
    return switch (constant) {
        .int => |v| v,
        else => null,
    };
}

fn sameOptionalLength(a: ?u32, b: ?u32) ?u32 {
    if (a == null or b == null) return null;
    return if (a.? == b.?) a.? else null;
}

fn setValue(values: []ValueInfo, value: ssa.ValueId, nullness: ?Nullness, array_length: ?u32) bool {
    if (value >= values.len) return false;
    var changed = false;
    if (nullness) |n| {
        const merged = Nullness.merge(values[value].nullness, n);
        if (merged != values[value].nullness) {
            values[value].nullness = merged;
            changed = true;
        }
    }
    if (array_length) |len| {
        if (values[value].array_length == null or values[value].array_length.? != len) {
            values[value].array_length = len;
            values[value].nullness = .non_null;
            changed = true;
        }
    }
    return changed;
}

fn applyPhi(values: []ValueInfo, phi: ssa.Phi) bool {
    var nullness: Nullness = .unknown;
    var length: ?u32 = null;
    var saw_length = false;
    for (phi.incoming) |incoming| {
        const source = values[incoming.value];
        nullness = Nullness.merge(nullness, source.nullness);
        if (!saw_length) {
            length = source.array_length;
            saw_length = true;
        } else {
            length = sameOptionalLength(length, source.array_length);
        }
    }

    var changed = false;
    if (values[phi.dest].nullness != nullness) {
        values[phi.dest].nullness = nullness;
        changed = true;
    }
    if (values[phi.dest].array_length != length) {
        values[phi.dest].array_length = length;
        changed = true;
    }
    return changed;
}

fn opCanHaveDevirt(inst: Instruction) bool {
    return switch (inst) {
        .invoke, .invoke_virtual_quick, .invoke_super_quick => true,
        else => false,
    };
}

fn objectUseForNullCheck(op: ssa.Operation) ?ssa.ValueId {
    return switch (op.inst) {
        .monitor_enter, .monitor_exit, .check_cast, .throw_ => if (op.uses.len >= 1) op.uses[0] else null,
        .array_length, .fill_array_data => if (op.uses.len >= 1) op.uses[0] else null,
        .aget, .aget_wide, .aget_object, .aget_boolean, .aget_byte, .aget_char, .aget_short => if (op.uses.len >= 1) op.uses[0] else null,
        .aput, .aput_wide, .aput_object, .aput_boolean, .aput_byte, .aput_char, .aput_short => if (op.uses.len >= 2) op.uses[op.uses.len - 2] else null,
        .iget, .iget_wide, .iget_object, .iget_boolean, .iget_byte, .iget_char, .iget_short, .iget_quick, .iget_wide_quick, .iget_object_quick => if (op.uses.len >= 1) op.uses[0] else null,
        .iput, .iput_wide, .iput_object, .iput_boolean, .iput_byte, .iput_char, .iput_short, .iput_quick, .iput_wide_quick, .iput_object_quick => if (op.uses.len >= 1) op.uses[op.uses.len - 1] else null,
        .invoke, .invoke_virtual_quick, .invoke_super_quick => if (op.uses.len >= 1) op.uses[0] else null,
        else => null,
    };
}

fn boundsOperands(op: ssa.Operation) ?struct { array: ssa.ValueId, index: ssa.ValueId } {
    return switch (op.inst) {
        .aget, .aget_wide, .aget_object, .aget_boolean, .aget_byte, .aget_char, .aget_short => if (op.uses.len >= 2) .{ .array = op.uses[0], .index = op.uses[1] } else null,
        .aput, .aput_wide, .aput_object, .aput_boolean, .aput_byte, .aput_char, .aput_short => if (op.uses.len >= 3) .{ .array = op.uses[op.uses.len - 2], .index = op.uses[op.uses.len - 1] } else null,
        else => null,
    };
}

fn opHasNonNullObjectUse(self: *const Function, op: ssa.Operation) bool {
    const value = objectUseForNullCheck(op) orelse return false;
    return value < self.values.len and self.values[value].nullness == .non_null;
}

fn opHasProvableBounds(self: *const Function, op: ssa.Operation) bool {
    const operands = boundsOperands(op) orelse return false;
    if (operands.array >= self.values.len) return false;
    const length = self.values[operands.array].array_length orelse return false;
    const index = intConstant(self.opt, operands.index) orelse return false;
    return index >= 0 and @as(u32, @intCast(index)) < length;
}

fn devirtHint(inst: Instruction) DevirtHint {
    return switch (inst) {
        .invoke => |invoke| switch (invoke.kind) {
            .direct => .direct_exact,
            .static => .static_exact,
            else => .none,
        },
        .invoke_virtual_quick => .quickened_virtual,
        .invoke_super_quick => .quickened_super,
        else => .none,
    };
}

fn loweringFor(op: ssa.Operation, types: *const typedir.Function) LoweringChoice {
    return switch (op.inst) {
        .nop => .none,
        .goto_,
        .packed_switch,
        .sparse_switch,
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
        => .branch,
        .return_void, .return_, .return_wide, .return_object => .return_path,
        .throw_ => .throw_path,
        .invoke, .invoke_virtual_quick, .invoke_super_quick => .call,
        .aget,
        .aget_wide,
        .aget_object,
        .aget_boolean,
        .aget_byte,
        .aget_char,
        .aget_short,
        .aput,
        .aput_wide,
        .aput_object,
        .aput_boolean,
        .aput_byte,
        .aput_char,
        .aput_short,
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
        .iget_quick,
        .iget_wide_quick,
        .iget_object_quick,
        .iput_quick,
        .iput_wide_quick,
        .iput_object_quick,
        .array_length,
        .new_array,
        .fill_array_data,
        .new_instance,
        .filled_new_array,
        => .memory,
        .monitor_enter, .monitor_exit, .check_cast => .reference,
        else => {
            if (op.defs.len == 0) return .none;
            const ty = types.typeOf(op.defs[0]) orelse .unknown;
            return switch (ty) {
                .int => .int32,
                .long => .int64,
                .float => .float32,
                .double => .float64,
                .object => .reference,
                else => .none,
            };
        },
    };
}

fn typeSimplified(op: ssa.Operation, typed: *const typedir.Function) bool {
    return switch (op.inst) {
        .move, .move_object, .move_wide => blk: {
            if (op.uses.len == 0 or op.defs.len == 0) break :blk false;
            const src = typed.typeOf(op.uses[0]) orelse .unknown;
            const dst = typed.typeOf(op.defs[0]) orelse .unknown;
            break :blk src != .unknown and src == dst;
        },
        .check_cast => if (op.uses.len >= 1) (typed.typeOf(op.uses[0]) orelse .unknown) == .object else false,
        else => false,
    };
}

fn applyValueFacts(function: *Function, op: ssa.Operation) bool {
    var changed = false;
    switch (op.inst) {
        .const_ => |inst| {
            if (inst.value == 0 and op.defs.len >= 1 and function.values[op.defs[0]].ty == .object) {
                changed = setValue(function.values, op.defs[0], .null_value, null) or changed;
            }
        },
        .const_string, .const_class, .new_instance => {
            if (op.defs.len >= 1) changed = setValue(function.values, op.defs[0], .non_null, null) or changed;
        },
        .new_array => {
            if (op.defs.len >= 1 and op.uses.len >= 1) {
                const size = intConstant(function.opt, op.uses[0]);
                if (size) |len| {
                    if (len >= 0) changed = setValue(function.values, op.defs[0], .non_null, @intCast(len)) or changed;
                } else {
                    changed = setValue(function.values, op.defs[0], .non_null, null) or changed;
                }
            }
        },
        .move, .move_object, .move_wide => {
            for (op.defs, 0..) |def, i| {
                if (i >= op.uses.len) break;
                const source = function.values[op.uses[i]];
                changed = setValue(function.values, def, source.nullness, source.array_length) or changed;
            }
        },
        else => {},
    }
    return changed;
}

pub fn build(
    allocator: std.mem.Allocator,
    source: *const ssa.Function,
    typed: *const typedir.Function,
    opt: ?*const optimizer.Result,
) Error!Function {
    source.verify() catch |err| switch (err) {
        error.OutOfMemory => return error.OutOfMemory,
        else => return error.InvalidInput,
    };
    typed.verify() catch |err| switch (err) {
        error.OutOfMemory => return error.OutOfMemory,
        else => return error.InvalidInput,
    };
    if (typed.source != source) return error.InvalidInput;
    if (opt) |opt_result| {
        if (opt_result.function != source) return error.InvalidInput;
    }

    const values = try allocator.alloc(ValueInfo, source.values.len);
    errdefer allocator.free(values);
    for (values, 0..) |*value, i| value.* = .{ .ty = typed.typeOf(@intCast(i)) orelse .unknown };

    const ops = try allocator.alloc([]OpInfo, source.blocks.len);
    errdefer allocator.free(ops);
    var built_ops: usize = 0;
    errdefer {
        for (ops[0..built_ops]) |slice| allocator.free(slice);
    }
    for (source.blocks, 0..) |block, i| {
        ops[i] = try allocator.alloc(OpInfo, block.ops.len);
        @memset(ops[i], .{});
        built_ops += 1;
    }

    var out = Function{
        .allocator = allocator,
        .source = source,
        .types = typed,
        .opt = opt,
        .values = values,
        .ops = ops,
        .stats = .{},
    };

    var changed = true;
    while (changed) {
        changed = false;
        for (source.graph.rpo) |block_id| {
            const block = source.blocks[block_id];
            for (block.phis) |phi| changed = applyPhi(out.values, phi) or changed;
            for (block.ops) |op| changed = applyValueFacts(&out, op) or changed;
        }
    }

    for (source.graph.rpo) |block_id| {
        const block = source.blocks[block_id];
        for (block.ops, 0..) |op, i| {
            var info = OpInfo{
                .type_simplified = typeSimplified(op, typed),
                .null_check_elided = opHasNonNullObjectUse(&out, op),
                .bounds_check_elided = opHasProvableBounds(&out, op),
                .devirt = devirtHint(op.inst),
                .lowering = loweringFor(op, typed),
            };
            if (info.bounds_check_elided) info.null_check_elided = true;
            out.ops[block_id][i] = info;
            if (info.type_simplified) out.stats.type_simplifications += 1;
            if (info.null_check_elided) out.stats.null_checks_elided += 1;
            if (info.bounds_check_elided) out.stats.bounds_checks_elided += 1;
            if (info.devirt != .none) out.stats.devirt_hints += 1;
        }
    }

    return out;
}

fn initPipeline(
    allocator: std.mem.Allocator,
    insts: []const Instruction,
    graph: *cfg.Graph,
    tree: *dom.Tree,
    function: *ssa.Function,
    typed: *typedir.Function,
    opt: *optimizer.Result,
) !Function {
    graph.* = try cfg.build(allocator, insts);
    errdefer graph.deinit();
    tree.* = try dom.build(allocator, graph);
    errdefer tree.deinit();
    function.* = try ssa.build(allocator, graph, tree);
    errdefer function.deinit();
    typed.* = try typedir.build(allocator, function);
    errdefer typed.deinit();
    opt.* = try optimizer.run(allocator, function, .{});
    errdefer opt.deinit();
    return try build(allocator, function, typed, opt);
}

test "typed_ir chooses lowering by discovered type" {
    const insts = [_]Instruction{
        .{ .const_ = .{ .dest = 0, .value = 1 } },
        .{ .const_wide = .{ .dest = 2, .value = 10 } },
        .{ .int_to_float = .{ .dest = 4, .src = 0 } },
        .return_void,
    };
    var graph: cfg.Graph = undefined;
    var tree: dom.Tree = undefined;
    var function: ssa.Function = undefined;
    var typed: typedir.Function = undefined;
    var opt: optimizer.Result = undefined;
    var tir = try initPipeline(std.testing.allocator, &insts, &graph, &tree, &function, &typed, &opt);
    defer tir.deinit();
    defer opt.deinit();
    defer typed.deinit();
    defer function.deinit();
    defer tree.deinit();
    defer graph.deinit();

    const block = function.blocks[graph.entry];
    try std.testing.expectEqual(LoweringChoice.int32, tir.opInfo(graph.entry, 0).?.lowering);
    try std.testing.expectEqual(LoweringChoice.int64, tir.opInfo(graph.entry, 1).?.lowering);
    try std.testing.expectEqual(LoweringChoice.float32, tir.opInfo(graph.entry, 2).?.lowering);
    try std.testing.expectEqual(typedir.Type.long, tir.valueInfo(block.ops[1].defs[0]).?.ty);
    try tir.verify();
}

test "typed_ir elides bounds and null checks for known new-array access" {
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
    var typed: typedir.Function = undefined;
    var opt: optimizer.Result = undefined;
    var tir = try initPipeline(std.testing.allocator, &insts, &graph, &tree, &function, &typed, &opt);
    defer tir.deinit();
    defer opt.deinit();
    defer typed.deinit();
    defer function.deinit();
    defer tree.deinit();
    defer graph.deinit();

    const array_value = function.blocks[graph.entry].ops[1].defs[0];
    try std.testing.expectEqual(Nullness.non_null, tir.valueInfo(array_value).?.nullness);
    try std.testing.expectEqual(@as(?u32, 4), tir.valueInfo(array_value).?.array_length);
    try std.testing.expect(tir.opInfo(graph.entry, 3).?.null_check_elided);
    try std.testing.expect(tir.opInfo(graph.entry, 3).?.bounds_check_elided);
    try tir.verify();
}

test "typed_ir keeps unknown bounds when index is not provable" {
    const insts = [_]Instruction{
        .{ .const_ = .{ .dest = 0, .value = 4 } },
        .{ .new_array = .{ .dest = 1, .size = 0, .type_idx = 1 } },
        .{ .aget = .{ .dest_or_src = 3, .array = 1, .index = 2 } },
        .{ .return_ = .{ .src = 3 } },
    };
    var graph: cfg.Graph = undefined;
    var tree: dom.Tree = undefined;
    var function: ssa.Function = undefined;
    var typed: typedir.Function = undefined;
    var opt: optimizer.Result = undefined;
    var tir = try initPipeline(std.testing.allocator, &insts, &graph, &tree, &function, &typed, &opt);
    defer tir.deinit();
    defer opt.deinit();
    defer typed.deinit();
    defer function.deinit();
    defer tree.deinit();
    defer graph.deinit();

    try std.testing.expect(tir.opInfo(graph.entry, 2).?.null_check_elided);
    try std.testing.expect(!tir.opInfo(graph.entry, 2).?.bounds_check_elided);
    try tir.verify();
}

test "typed_ir records devirtualization hints" {
    var direct = instmod.Invoke{
        .class_name = "LExample;",
        .method_name = "f",
        .signature = "()V",
        .dest = null,
        .kind = .direct,
    };
    var static = instmod.Invoke{
        .class_name = "LExample;",
        .method_name = "g",
        .signature = "()V",
        .dest = null,
        .kind = .static,
    };
    const insts = [_]Instruction{
        .{ .invoke = &direct },
        .{ .invoke = &static },
        .return_void,
    };
    var graph: cfg.Graph = undefined;
    var tree: dom.Tree = undefined;
    var function: ssa.Function = undefined;
    var typed: typedir.Function = undefined;
    var opt: optimizer.Result = undefined;
    var tir = try initPipeline(std.testing.allocator, &insts, &graph, &tree, &function, &typed, &opt);
    defer tir.deinit();
    defer opt.deinit();
    defer typed.deinit();
    defer function.deinit();
    defer tree.deinit();
    defer graph.deinit();

    try std.testing.expectEqual(DevirtHint.direct_exact, tir.opInfo(graph.entry, 0).?.devirt);
    try std.testing.expectEqual(DevirtHint.static_exact, tir.opInfo(graph.entry, 1).?.devirt);
    try tir.verify();
}

test "typed_ir marks redundant typed moves and casts as simplifiable" {
    const insts = [_]Instruction{
        .{ .const_string = .{ .dest = 0, .index = 1 } },
        .{ .move_object = .{ .dest = 1, .src = 0 } },
        .{ .check_cast = .{ .src = 1, .type_idx = 1 } },
        .{ .return_object = .{ .src = 1 } },
    };
    var graph: cfg.Graph = undefined;
    var tree: dom.Tree = undefined;
    var function: ssa.Function = undefined;
    var typed: typedir.Function = undefined;
    var opt: optimizer.Result = undefined;
    var tir = try initPipeline(std.testing.allocator, &insts, &graph, &tree, &function, &typed, &opt);
    defer tir.deinit();
    defer opt.deinit();
    defer typed.deinit();
    defer function.deinit();
    defer tree.deinit();
    defer graph.deinit();

    try std.testing.expect(tir.opInfo(graph.entry, 1).?.type_simplified);
    try std.testing.expect(tir.opInfo(graph.entry, 2).?.type_simplified);
    try std.testing.expect(tir.opInfo(graph.entry, 2).?.null_check_elided);
    try tir.verify();
}

test "typed_ir print helper emits stable summary" {
    const insts = [_]Instruction{
        .{ .const_ = .{ .dest = 0, .value = 1 } },
        .return_void,
    };
    var graph: cfg.Graph = undefined;
    var tree: dom.Tree = undefined;
    var function: ssa.Function = undefined;
    var typed: typedir.Function = undefined;
    var opt: optimizer.Result = undefined;
    var tir = try initPipeline(std.testing.allocator, &insts, &graph, &tree, &function, &typed, &opt);
    defer tir.deinit();
    defer opt.deinit();
    defer typed.deinit();
    defer function.deinit();
    defer tree.deinit();
    defer graph.deinit();

    var buf: [512]u8 = undefined;
    var stream = std.Io.Writer.fixed(&buf);
    try tir.print(&stream);
    const output = stream.buffered();

    try std.testing.expect(std.mem.indexOf(u8, output, "typed_ir values=") != null);
    try std.testing.expect(std.mem.indexOf(u8, output, "lower=int32") != null);
    try std.testing.expect(std.mem.indexOf(u8, output, "null_elided=") != null);
}
