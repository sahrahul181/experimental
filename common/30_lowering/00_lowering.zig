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
};

pub const ValueId = ssa.ValueId;

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
    array_load,
    array_store,
    field_load,
    field_store,
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
    field_idx: ?u32 = null,
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
};

pub const Stats = struct {
    lowered: u32 = 0,
    skipped_dead: u32 = 0,
    constants_materialized: u32 = 0,
    null_checks_elided: u32 = 0,
    bounds_checks_elided: u32 = 0,
    forwarded_loads: u32 = 0,
    direct_calls: u32 = 0,
};

pub const Function = struct {
    allocator: std.mem.Allocator,
    source: *const ssa.Function,
    blocks: []Block,
    value_types: []typedir.Type,
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
        self.* = undefined;
    }

    pub fn verify(self: *const Function) VerifyError!void {
        if (self.blocks.len != self.source.blocks.len or self.value_types.len != self.source.values.len) return error.InvalidInput;
        for (self.blocks, 0..) |block, i| {
            if (block.id != i) return error.BadBlock;
            for (block.insts) |inst| {
                for (inst.defs) |def| if (def >= self.value_types.len) return error.BadValue;
                for (inst.uses) |use| if (use >= self.value_types.len) return error.BadValue;
                switch (inst.kind) {
                    .branch => if (inst.target == null) return error.BadInstruction,
                    .cond_branch => if (inst.target == null or inst.false_target == null or inst.uses.len == 0) return error.BadInstruction,
                    .field_load, .field_store, .static_load, .static_store => if (inst.field_idx == null) return error.BadInstruction,
                    else => {},
                }
            }
        }
    }

    pub fn print(self: *const Function, writer: anytype) !void {
        try writer.print(
            "lowering blocks={d} values={d} lowered={d} skipped_dead={d} consts={d} null_elided={d} bounds_elided={d} forwarded={d} direct_calls={d}\n",
            .{
                self.blocks.len,
                self.value_types.len,
                self.stats.lowered,
                self.stats.skipped_dead,
                self.stats.constants_materialized,
                self.stats.null_checks_elided,
                self.stats.bounds_checks_elided,
                self.stats.forwarded_loads,
                self.stats.direct_calls,
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
    try list.append(allocator, inst);
    stats.lowered += 1;
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

fn fieldIndex(inst: Instruction) ?u32 {
    return switch (inst) {
        .iget, .iget_wide, .iget_object, .iget_boolean, .iget_byte, .iget_char, .iget_short,
        .iput, .iput_wide, .iput_object, .iput_boolean, .iput_byte, .iput_char, .iput_short,
        .iget_quick, .iget_wide_quick, .iget_object_quick,
        .iput_quick, .iput_wide_quick, .iput_object_quick,
        => |op| op.field_idx,
        .sget, .sget_wide, .sget_object, .sget_boolean, .sget_byte, .sget_char, .sget_short,
        .sput, .sput_wide, .sput_object, .sput_boolean, .sput_byte, .sput_char, .sput_short,
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

fn lowerOperation(
    allocator: std.mem.Allocator,
    inputs: Inputs,
    block_id: cfg.BlockId,
    op_index: usize,
    op: ssa.Operation,
    list: *std.ArrayList(Inst),
    stats: *Stats,
) !void {
    if (opDead(inputs, block_id, op_index)) {
        stats.skipped_dead += 1;
        return;
    }

    const tinfo = typedInfo(inputs, block_id, op_index);
    const minfo = memInfo(inputs, block_id, op_index);
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
            try appendInst(list, allocator, .{ .kind = .copy, .pc = op.pc, .defs = try dupeValues(allocator, op.defs), .uses = try dupeValues(allocator, op.uses), .flags = flags }, stats);
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
            if (!flags.bounds_check_elided) try appendInst(list, allocator, .{ .kind = .check_bounds, .pc = op.pc, .uses = try dupeValues(allocator, op.uses[0..2]) }, stats);
            try appendInst(list, allocator, .{ .kind = .array_load, .pc = op.pc, .defs = try dupeValues(allocator, op.defs), .uses = try dupeValues(allocator, op.uses), .flags = flags }, stats);
        },
        .aput, .aput_wide, .aput_object, .aput_boolean, .aput_byte, .aput_char, .aput_short => {
            try appendInst(list, allocator, .{ .kind = .array_store, .pc = op.pc, .uses = try dupeValues(allocator, op.uses), .flags = flags }, stats);
        },
        .iget, .iget_wide, .iget_object, .iget_boolean, .iget_byte, .iget_char, .iget_short, .iget_quick, .iget_wide_quick, .iget_object_quick => {
            if (!flags.null_check_elided) try appendInst(list, allocator, .{ .kind = .check_null, .pc = op.pc, .uses = try dupeValues(allocator, op.uses[0..1]) }, stats);
            try appendInst(list, allocator, .{ .kind = .field_load, .pc = op.pc, .defs = try dupeValues(allocator, op.defs), .uses = try dupeValues(allocator, op.uses), .field_idx = fieldIndex(op.inst), .flags = flags }, stats);
        },
        .iput, .iput_wide, .iput_object, .iput_boolean, .iput_byte, .iput_char, .iput_short, .iput_quick, .iput_wide_quick, .iput_object_quick => {
            try appendInst(list, allocator, .{ .kind = .field_store, .pc = op.pc, .uses = try dupeValues(allocator, op.uses), .field_idx = fieldIndex(op.inst), .flags = flags }, stats);
        },
        .sget, .sget_wide, .sget_object, .sget_boolean, .sget_byte, .sget_char, .sget_short => {
            try appendInst(list, allocator, .{ .kind = .static_load, .pc = op.pc, .defs = try dupeValues(allocator, op.defs), .field_idx = fieldIndex(op.inst), .flags = flags }, stats);
        },
        .sput, .sput_wide, .sput_object, .sput_boolean, .sput_byte, .sput_char, .sput_short => {
            try appendInst(list, allocator, .{ .kind = .static_store, .pc = op.pc, .uses = try dupeValues(allocator, op.uses), .field_idx = fieldIndex(op.inst), .flags = flags }, stats);
        },
        .invoke, .invoke_virtual_quick, .invoke_super_quick => {
            const kind = lowerCallKind(op.inst, tinfo);
            if (kind == .call_direct or kind == .call_static or kind == .call_quick) stats.direct_calls += 1;
            try appendInst(list, allocator, .{ .kind = kind, .pc = op.pc, .defs = try dupeValues(allocator, op.defs), .uses = try dupeValues(allocator, op.uses), .flags = flags }, stats);
        },
        else => {
            const kind = lowerArithmetic(op.inst, tinfo.lowering);
            try appendInst(list, allocator, .{ .kind = kind, .pc = op.pc, .defs = try dupeValues(allocator, op.defs), .uses = try dupeValues(allocator, op.uses), .field_idx = fieldIndex(op.inst), .flags = flags }, stats);
        },
    }
}

pub fn build(allocator: std.mem.Allocator, inputs: Inputs) Error!Function {
    inputs.function.verify() catch return error.InvalidInput;
    inputs.types.verify() catch return error.InvalidInput;
    inputs.typed.verify() catch return error.InvalidInput;
    if (inputs.types.source != inputs.function or inputs.typed.source != inputs.function) return error.InvalidInput;
    if (inputs.ssa_facts) |facts| if (facts.function != inputs.function) return error.InvalidInput;
    if (inputs.memory) |memory| if (memory.function != inputs.function) return error.InvalidInput;

    const value_types = try allocator.alloc(typedir.Type, inputs.function.values.len);
    errdefer allocator.free(value_types);
    for (value_types, 0..) |*slot, i| slot.* = inputs.types.typeOf(@intCast(i)) orelse .unknown;

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

    var stats: Stats = .{};
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

        for (block.ops, 0..) |op, op_i| {
            try lowerOperation(allocator, inputs, @intCast(block_i), op_i, op, &list, &stats);
        }

        blocks[block_i] = .{ .id = @intCast(block_i), .insts = try list.toOwnedSlice(allocator) };
        built_blocks += 1;
    }

    return .{
        .allocator = allocator,
        .source = inputs.function,
        .blocks = blocks,
        .value_types = value_types,
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
