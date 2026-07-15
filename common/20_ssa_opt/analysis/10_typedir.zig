//! Typed SSA directory.
//!
//! This pass annotates SSA values with a compact Dalvik value category. It is
//! deliberately target-independent and conservative: unknown parameters remain
//! unknown until a defining operation or phi merge proves something stronger.

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

pub const VerifyError = error{
    BadValue,
    BadPhiType,
    BadOperationType,
    InvalidSsa,
    OutOfMemory,
};

pub const Type = enum(u8) {
    unknown,
    int,
    long,
    float,
    double,
    object,
    conflict,

    pub fn merge(a: Type, b: Type) Type {
        if (a == b) return a;
        if (a == .unknown) return b;
        if (b == .unknown) return a;
        return .conflict;
    }
};

pub const ValueInfo = struct {
    ty: Type = .unknown,
};

pub const Function = struct {
    allocator: std.mem.Allocator,
    source: *const ssa.Function,
    values: []ValueInfo,

    pub fn deinit(self: *Function) void {
        self.allocator.free(self.values);
        self.* = undefined;
    }

    pub inline fn typeOf(self: *const Function, value: ssa.ValueId) ?Type {
        if (value >= self.values.len) return null;
        return self.values[value].ty;
    }

    pub fn verify(self: *const Function) VerifyError!void {
        self.source.verify() catch |err| switch (err) {
            error.OutOfMemory => return error.OutOfMemory,
            else => return error.InvalidSsa,
        };
        if (self.values.len != self.source.values.len) return error.InvalidSsa;

        for (self.source.values, 0..) |value, i| {
            if (value.id != i) return error.BadValue;
            if (self.values[i].ty == .conflict and value.kind != .phi) return error.BadOperationType;
        }

        for (self.source.blocks) |block| {
            for (block.phis) |phi| {
                var merged: Type = .unknown;
                for (phi.incoming) |incoming| {
                    if (incoming.value >= self.values.len) return error.BadValue;
                    merged = Type.merge(merged, self.values[incoming.value].ty);
                }
                if (self.values[phi.dest].ty != merged) return error.BadPhiType;
            }

            for (block.ops) |op| {
                var expected: [2]Type = .{ .unknown, .unknown };
                const count = expectedDefTypes(self, op, &expected);
                if (count == 0) continue;
                if (op.defs.len < count) return error.BadOperationType;
                for (expected[0..count], 0..) |ty, i| {
                    if (ty == .unknown) continue;
                    if (self.values[op.defs[i]].ty != ty) return error.BadOperationType;
                }
            }
        }
    }

    pub fn print(self: *const Function, writer: anytype) !void {
        try writer.print("typedir values={d} blocks={d}\n", .{ self.values.len, self.source.blocks.len });
        for (self.source.graph.rpo) |block_id| {
            const block = self.source.blocks[block_id];
            try writer.print("b{d}\n", .{block_id});
            for (block.phis) |phi| {
                try writer.print("  v{d}: {s} = phi r{d}\n", .{ phi.dest, @tagName(self.values[phi.dest].ty), phi.reg });
            }
            for (block.ops) |op| {
                try writer.print("  pc{d} {s}", .{ op.pc, @tagName(op.inst) });
                if (op.defs.len == 0) {
                    try writer.print(" defs:<none>", .{});
                } else {
                    try writer.print(" defs:", .{});
                    for (op.defs) |def| try writer.print(" v{d}:{s}", .{ def, @tagName(self.values[def].ty) });
                }
                try writer.print("\n", .{});
            }
        }
    }
};

fn firstUseType(self: *const Function, op: ssa.Operation) Type {
    if (op.uses.len == 0 or op.uses[0] >= self.values.len) return .unknown;
    return self.values[op.uses[0]].ty;
}

fn expectedDefTypes(self: *const Function, op: ssa.Operation, out: *[2]Type) usize {
    out.* = .{ .unknown, .unknown };
    switch (op.inst) {
        .nop,
        .return_void,
        .return_,
        .return_wide,
        .return_object,
        .monitor_enter,
        .monitor_exit,
        .check_cast,
        .fill_array_data,
        .throw_,
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
        .aput,
        .aput_wide,
        .aput_object,
        .aput_boolean,
        .aput_byte,
        .aput_char,
        .aput_short,
        .iput,
        .iput_wide,
        .iput_object,
        .iput_boolean,
        .iput_byte,
        .iput_char,
        .iput_short,
        .sput,
        .sput_wide,
        .sput_object,
        .sput_boolean,
        .sput_byte,
        .sput_char,
        .sput_short,
        .iput_quick,
        .iput_wide_quick,
        .iput_object_quick,
        => return 0,

        .move, .move_wide, .move_object => {
            const ty = firstUseType(self, op);
            out[0] = ty;
            if (op.defs.len > 1) out[1] = ty;
            return @min(op.defs.len, 2);
        },
        .move_result => {
            out[0] = .int;
            return 1;
        },
        .move_result_wide => {
            out.* = .{ .long, .long };
            return 2;
        },
        .move_result_object, .move_exception => {
            out[0] = .object;
            return 1;
        },

        .const_,
        .const_method_handle,
        .const_method_type,
        .array_length,
        .instance_of,
        .cmpl_float,
        .cmpg_float,
        .cmpl_double,
        .cmpg_double,
        .cmp_long,
        .int_to_byte,
        .int_to_char,
        .int_to_short,
        .long_to_int,
        .float_to_int,
        .double_to_int,
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
        .add_int_lit16,
        .rsub_int_lit16,
        .mul_int_lit16,
        .div_int_lit16,
        .rem_int_lit16,
        .and_int_lit16,
        .or_int_lit16,
        .xor_int_lit16,
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
        .neg_int,
        .not_int,
        => {
            out[0] = .int;
            return 1;
        },

        .const_string,
        .const_class,
        .new_instance,
        .new_array,
        .aget_object,
        .sget_object,
        .iget_object,
        .iget_object_quick,
        => {
            out[0] = .object;
            return 1;
        },

        .const_wide,
        .int_to_long,
        .float_to_long,
        .double_to_long,
        .add_long,
        .sub_long,
        .mul_long,
        .div_long,
        .rem_long,
        .and_long,
        .or_long,
        .xor_long,
        .shl_long,
        .shr_long,
        .ushr_long,
        .neg_long,
        .not_long,
        .aget_wide,
        .sget_wide,
        .iget_wide,
        .iget_wide_quick,
        => {
            out.* = .{ .long, .long };
            return 2;
        },

        .int_to_float,
        .long_to_float,
        .double_to_float,
        .add_float,
        .sub_float,
        .mul_float,
        .div_float,
        .rem_float,
        .neg_float,
        => {
            out[0] = .float;
            return 1;
        },

        .int_to_double,
        .long_to_double,
        .float_to_double,
        .add_double,
        .sub_double,
        .mul_double,
        .div_double,
        .rem_double,
        .neg_double,
        => {
            out.* = .{ .double, .double };
            return 2;
        },

        .aget,
        .aget_boolean,
        .aget_byte,
        .aget_char,
        .aget_short,
        .sget,
        .sget_boolean,
        .sget_byte,
        .sget_char,
        .sget_short,
        .iget,
        .iget_boolean,
        .iget_byte,
        .iget_char,
        .iget_short,
        .iget_quick,
        => {
            out[0] = .int;
            return 1;
        },

        .invoke, .invoke_virtual_quick, .invoke_super_quick => {
            out[0] = .unknown;
            return if (op.defs.len == 0) 0 else 1;
        },

        .filled_new_array => return 0,
    }
}

fn applyOpTypes(typed: *Function, op: ssa.Operation) bool {
    var expected: [2]Type = .{ .unknown, .unknown };
    const count = expectedDefTypes(typed, op, &expected);
    var changed = false;
    for (expected[0..count], 0..) |ty, i| {
        if (i >= op.defs.len or ty == .unknown) continue;
        const def = op.defs[i];
        const merged = Type.merge(typed.values[def].ty, ty);
        if (merged != typed.values[def].ty) {
            typed.values[def].ty = merged;
            changed = true;
        }
    }
    return changed;
}

fn applyPhiTypes(typed: *Function, phi: ssa.Phi) bool {
    var merged: Type = .unknown;
    for (phi.incoming) |incoming| merged = Type.merge(merged, typed.values[incoming.value].ty);
    if (typed.values[phi.dest].ty == merged) return false;
    typed.values[phi.dest].ty = merged;
    return true;
}

pub fn build(allocator: std.mem.Allocator, source: *const ssa.Function) Error!Function {
    source.verify() catch |err| switch (err) {
        error.OutOfMemory => return error.OutOfMemory,
        else => return error.InvalidSsa,
    };

    const values = try allocator.alloc(ValueInfo, source.values.len);
    errdefer allocator.free(values);
    @memset(values, .{});

    var typed = Function{
        .allocator = allocator,
        .source = source,
        .values = values,
    };

    var changed = true;
    while (changed) {
        changed = false;
        for (source.graph.rpo) |block_id| {
            const block = source.blocks[block_id];
            for (block.phis) |phi| changed = applyPhiTypes(&typed, phi) or changed;
            for (block.ops) |op| changed = applyOpTypes(&typed, op) or changed;
        }
    }

    return typed;
}

test "typedir infers integer arithmetic types" {
    const insts = [_]Instruction{
        .{ .const_ = .{ .dest = 0, .value = 1 } },
        .{ .const_ = .{ .dest = 1, .value = 2 } },
        .{ .add_int = .{ .dest = 2, .src1 = 0, .src2 = 1 } },
        .{ .return_ = .{ .src = 2 } },
    };
    var graph = try cfg.build(std.testing.allocator, &insts);
    defer graph.deinit();
    var tree = try dom.build(std.testing.allocator, &graph);
    defer tree.deinit();
    var function = try ssa.build(std.testing.allocator, &graph, &tree);
    defer function.deinit();
    var typed = try build(std.testing.allocator, &function);
    defer typed.deinit();

    const op = function.blocks[graph.entry].ops[2];
    try std.testing.expectEqual(Type.int, typed.typeOf(op.defs[0]).?);
    try typed.verify();
}

test "typedir propagates move type from source" {
    const insts = [_]Instruction{
        .{ .const_string = .{ .dest = 0, .index = 9 } },
        .{ .move_object = .{ .dest = 1, .src = 0 } },
        .{ .return_object = .{ .src = 1 } },
    };
    var graph = try cfg.build(std.testing.allocator, &insts);
    defer graph.deinit();
    var tree = try dom.build(std.testing.allocator, &graph);
    defer tree.deinit();
    var function = try ssa.build(std.testing.allocator, &graph, &tree);
    defer function.deinit();
    var typed = try build(std.testing.allocator, &function);
    defer typed.deinit();

    const move_op = function.blocks[graph.entry].ops[1];
    try std.testing.expectEqual(Type.object, typed.typeOf(move_op.defs[0]).?);
    try typed.verify();
}

test "typedir merges matching phi types" {
    const insts = [_]Instruction{
        .{ .if_eqz = .{ .src = 0, .offset = 3 } },
        .{ .const_ = .{ .dest = 1, .value = 1 } },
        .{ .goto_ = .{ .offset = 2 } },
        .{ .const_ = .{ .dest = 1, .value = 2 } },
        .{ .return_ = .{ .src = 1 } },
    };
    var graph = try cfg.build(std.testing.allocator, &insts);
    defer graph.deinit();
    var tree = try dom.build(std.testing.allocator, &graph);
    defer tree.deinit();
    var function = try ssa.build(std.testing.allocator, &graph, &tree);
    defer function.deinit();
    var typed = try build(std.testing.allocator, &function);
    defer typed.deinit();

    const join = graph.blockForPc(4).?.id;
    const phi = function.blocks[join].phis[0];
    try std.testing.expectEqual(Type.int, typed.typeOf(phi.dest).?);
    try typed.verify();
}

test "typedir detects conflicting phi types conservatively" {
    const insts = [_]Instruction{
        .{ .if_eqz = .{ .src = 0, .offset = 3 } },
        .{ .const_ = .{ .dest = 1, .value = 1 } },
        .{ .goto_ = .{ .offset = 2 } },
        .{ .const_string = .{ .dest = 1, .index = 2 } },
        .{ .return_ = .{ .src = 1 } },
    };
    var graph = try cfg.build(std.testing.allocator, &insts);
    defer graph.deinit();
    var tree = try dom.build(std.testing.allocator, &graph);
    defer tree.deinit();
    var function = try ssa.build(std.testing.allocator, &graph, &tree);
    defer function.deinit();
    var typed = try build(std.testing.allocator, &function);
    defer typed.deinit();

    const join = graph.blockForPc(4).?.id;
    const phi = function.blocks[join].phis[0];
    try std.testing.expectEqual(Type.conflict, typed.typeOf(phi.dest).?);
    try typed.verify();
}

test "typedir tracks wide float and double families" {
    const insts = [_]Instruction{
        .{ .const_wide = .{ .dest = 0, .value = 1 } },
        .{ .long_to_double = .{ .dest = 2, .src = 0 } },
        .{ .double_to_float = .{ .dest = 4, .src = 2 } },
        .return_void,
    };
    var graph = try cfg.build(std.testing.allocator, &insts);
    defer graph.deinit();
    var tree = try dom.build(std.testing.allocator, &graph);
    defer tree.deinit();
    var function = try ssa.build(std.testing.allocator, &graph, &tree);
    defer function.deinit();
    var typed = try build(std.testing.allocator, &function);
    defer typed.deinit();

    const block = function.blocks[graph.entry];
    try std.testing.expectEqual(Type.long, typed.typeOf(block.ops[0].defs[0]).?);
    try std.testing.expectEqual(Type.double, typed.typeOf(block.ops[1].defs[0]).?);
    try std.testing.expectEqual(Type.float, typed.typeOf(block.ops[2].defs[0]).?);
    try typed.verify();
}

test "typedir print helper emits stable summary" {
    const insts = [_]Instruction{
        .{ .const_ = .{ .dest = 0, .value = 1 } },
        .{ .return_ = .{ .src = 0 } },
    };
    var graph = try cfg.build(std.testing.allocator, &insts);
    defer graph.deinit();
    var tree = try dom.build(std.testing.allocator, &graph);
    defer tree.deinit();
    var function = try ssa.build(std.testing.allocator, &graph, &tree);
    defer function.deinit();
    var typed = try build(std.testing.allocator, &function);
    defer typed.deinit();

    var buf: [512]u8 = undefined;
    var stream = std.Io.Writer.fixed(&buf);
    try typed.print(&stream);
    const output = stream.buffered();

    try std.testing.expect(std.mem.indexOf(u8, output, "typedir values=") != null);
    try std.testing.expect(std.mem.indexOf(u8, output, "pc0 const_") != null);
    try std.testing.expect(std.mem.indexOf(u8, output, "int") != null);
}
