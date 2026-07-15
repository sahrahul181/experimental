//! Minimal x86-64 native encoder.
//!
//! This backend is intentionally narrow but executable: it lowers constant
//! integer computations and returns into real x86-64 machine code. Unsupported
//! register-machine operations fail explicitly so the JIT never mistakes the
//! debug register bytecode format for native code.

const std = @import("std");
const builtin = @import("builtin");
const cfg = @import("cfg");
const code_buffer = @import("code_buffer");
const jit_memory = @import("jit_memory");
const machine = @import("machine_bridge");
const optimizer = @import("optimizer");
const Instruction = @import("instructions").Instruction;

pub const Error = code_buffer.Error || error{
    InvalidMachine,
    UnsupportedInstruction,
};

pub const Stats = struct {
    bytes: u32 = 0,
    blocks: u32 = 0,
    native_insts: u32 = 0,
    constants: u32 = 0,
    folded_ops: u32 = 0,
    parameter_loads: u32 = 0,
    stack_slots: u32 = 0,
    returns: u32 = 0,
    jumps: u32 = 0,
    conditional_branches: u32 = 0,
    edge_moves: u32 = 0,
};

const ConstValue = union(enum) {
    i32: i32,
    i64: i64,
};

pub const Function = struct {
    allocator: std.mem.Allocator,
    source: *const machine.Function,
    buffer: code_buffer.Buffer,
    block_labels: []code_buffer.LabelId,
    stats: Stats,

    pub fn deinit(self: *Function) void {
        self.allocator.free(self.block_labels);
        self.buffer.deinit();
        self.* = undefined;
    }

    pub fn verify(self: *Function) Error!void {
        self.source.verify() catch return error.InvalidMachine;
        if (self.block_labels.len != self.source.blocks.len) return error.InvalidMachine;
        try self.buffer.verify();
    }

    pub fn finalize(self: *Function) Error![]u8 {
        try self.verify();
        const bytes = try self.buffer.finalize();
        self.stats.bytes = @intCast(bytes.len);
        return bytes;
    }

    pub fn print(self: *const Function, writer: anytype) !void {
        try writer.print(
            "x64_encoder bytes={d} blocks={d} native_insts={d} constants={d} folded_ops={d} params={d} stack_slots={d} returns={d} jumps={d} cond_branches={d} edge_moves={d}\n",
            .{
                self.buffer.len(),
                self.stats.blocks,
                self.stats.native_insts,
                self.stats.constants,
                self.stats.folded_ops,
                self.stats.parameter_loads,
                self.stats.stack_slots,
                self.stats.returns,
                self.stats.jumps,
                self.stats.conditional_branches,
                self.stats.edge_moves,
            },
        );
        try self.buffer.print(writer);
    }
};

fn setConst(constants: []?ConstValue, reg: machine.RegId, value: ConstValue) Error!void {
    if (reg >= constants.len) return error.InvalidMachine;
    constants[reg] = value;
}

fn getI32(constants: []const ?ConstValue, reg: machine.RegId) ?i32 {
    if (reg >= constants.len) return null;
    return switch (constants[reg] orelse return null) {
        .i32 => |value| value,
        .i64 => null,
    };
}

fn getI64(constants: []const ?ConstValue, reg: machine.RegId) ?i64 {
    if (reg >= constants.len) return null;
    return switch (constants[reg] orelse return null) {
        .i32 => |value| value,
        .i64 => |value| value,
    };
}

fn clearDefs(constants: []?ConstValue, defs: []const machine.RegId) void {
    for (defs) |def| {
        if (def < constants.len) constants[def] = null;
    }
}

fn emitMovEaxImm32(buffer: *code_buffer.Buffer, value: i32) Error!void {
    try buffer.emitU8(0xb8);
    try buffer.emitU32(@bitCast(value));
}

fn emitMovRaxImm64(buffer: *code_buffer.Buffer, value: i64) Error!void {
    try buffer.emitU8(0x48);
    try buffer.emitU8(0xb8);
    try buffer.emitU64(@bitCast(value));
}

fn emitEpilogue(buffer: *code_buffer.Buffer) Error!void {
    try buffer.emitU8(0xc9);
    try buffer.emitU8(0xc3);
}

fn emitPrologue(buffer: *code_buffer.Buffer, stack_size: u32) Error!void {
    try buffer.emitU8(0x55);
    try buffer.emitBytes(&[_]u8{ 0x48, 0x89, 0xe5 });
    if (stack_size != 0) {
        try buffer.emitBytes(&[_]u8{ 0x48, 0x81, 0xec });
        try buffer.emitU32(stack_size);
    }
}

fn emitJump(buffer: *code_buffer.Buffer, label: code_buffer.LabelId) Error!void {
    try buffer.emitU8(0xe9);
    _ = try buffer.reloc(label, .rel32, 0);
}

fn emitJcc(buffer: *code_buffer.Buffer, condition: u8, label: code_buffer.LabelId) Error!void {
    try buffer.emitU8(0x0f);
    try buffer.emitU8(condition);
    _ = try buffer.reloc(label, .rel32, 0);
}

fn slotOffset(reg: machine.RegId) u32 {
    return (reg + 1) * 8;
}

fn emitDisp32(buffer: *code_buffer.Buffer, offset: u32) Error!void {
    try buffer.emitU32(@bitCast(-@as(i32, @intCast(offset))));
}

fn emitLoadEaxSlot(buffer: *code_buffer.Buffer, reg: machine.RegId) Error!void {
    try buffer.emitU8(0x8b);
    try buffer.emitU8(0x85);
    try emitDisp32(buffer, slotOffset(reg));
}

fn emitLoadRaxSlot(buffer: *code_buffer.Buffer, reg: machine.RegId) Error!void {
    try buffer.emitBytes(&[_]u8{ 0x48, 0x8b, 0x85 });
    try emitDisp32(buffer, slotOffset(reg));
}

fn emitStoreEaxSlot(buffer: *code_buffer.Buffer, reg: machine.RegId) Error!void {
    try buffer.emitU8(0x89);
    try buffer.emitU8(0x85);
    try emitDisp32(buffer, slotOffset(reg));
}

fn emitStoreRaxSlot(buffer: *code_buffer.Buffer, reg: machine.RegId) Error!void {
    try buffer.emitBytes(&[_]u8{ 0x48, 0x89, 0x85 });
    try emitDisp32(buffer, slotOffset(reg));
}

fn emitStoreImm32Slot(buffer: *code_buffer.Buffer, reg: machine.RegId, value: i32) Error!void {
    try buffer.emitU8(0xc7);
    try buffer.emitU8(0x85);
    try emitDisp32(buffer, slotOffset(reg));
    try buffer.emitU32(@bitCast(value));
}

fn emitStoreParamSlot(buffer: *code_buffer.Buffer, reg: machine.RegId, param_index: u32, ty: anytype) Error!void {
    const abi_regs = switch (builtin.os.tag) {
        .windows => [_]u8{ 1, 2, 8, 9 },
        else => [_]u8{ 7, 6, 2, 1, 8, 9 },
    };
    if (param_index >= abi_regs.len) return error.UnsupportedInstruction;
    const src = abi_regs[param_index];
    const wide = isWideType(ty);
    const rex = 0x40 | (if (wide) @as(u8, 0x08) else 0) | (if (src >= 8) @as(u8, 0x04) else 0);
    if (rex != 0x40) try buffer.emitU8(rex);
    try buffer.emitU8(0x89);
    try buffer.emitU8(0x80 | ((src & 7) << 3) | 0x05);
    try emitDisp32(buffer, slotOffset(reg));
}

fn emitCmpSlotZero(buffer: *code_buffer.Buffer, reg: machine.RegId) Error!void {
    try buffer.emitU8(0x83);
    try buffer.emitU8(0xbd);
    try emitDisp32(buffer, slotOffset(reg));
    try buffer.emitU8(0);
}

fn emitCmpSlots(buffer: *code_buffer.Buffer, lhs: machine.RegId, rhs: machine.RegId) Error!void {
    try emitLoadEaxSlot(buffer, lhs);
    try buffer.emitU8(0x3b);
    try buffer.emitU8(0x85);
    try emitDisp32(buffer, slotOffset(rhs));
}

fn emitBinaryI32Slot(buffer: *code_buffer.Buffer, opcode: machine.Opcode, dest: machine.RegId, lhs: machine.RegId, rhs: machine.RegId) Error!void {
    try emitLoadEaxSlot(buffer, lhs);
    switch (opcode) {
        .add_i32 => try buffer.emitU8(0x03),
        .sub_i32 => try buffer.emitU8(0x2b),
        .and_i32 => try buffer.emitU8(0x23),
        .or_i32 => try buffer.emitU8(0x0b),
        .xor_i32 => try buffer.emitU8(0x33),
        .mul_i32 => {
            try buffer.emitU8(0x0f);
            try buffer.emitU8(0xaf);
        },
        else => return error.UnsupportedInstruction,
    }
    try buffer.emitU8(0x85);
    try emitDisp32(buffer, slotOffset(rhs));
    try emitStoreEaxSlot(buffer, dest);
}

fn isWideType(ty: anytype) bool {
    return switch (ty) {
        .long, .double => true,
        else => false,
    };
}

fn emitCopySlot(buffer: *code_buffer.Buffer, ty: anytype, dst: machine.RegId, src: machine.RegId) Error!void {
    if (dst == src) return;
    if (isWideType(ty)) {
        try emitLoadRaxSlot(buffer, src);
        try emitStoreRaxSlot(buffer, dst);
    } else {
        try emitLoadEaxSlot(buffer, src);
        try emitStoreEaxSlot(buffer, dst);
    }
}

fn maxEdgeMoveCount(source: *const machine.Function) u32 {
    var max: u32 = 0;
    for (source.edges) |edge| max = @max(max, @as(u32, @intCast(edge.moves.len)));
    return max;
}

fn edgeMoves(source: *const machine.Function, from: cfg.BlockId, to: cfg.BlockId) []const machine.Move {
    for (source.edges) |edge| {
        if (edge.from == from and edge.to == to) return edge.moves;
    }
    return &.{};
}

fn emitParallelEdgeMoves(
    buffer: *code_buffer.Buffer,
    source: *const machine.Function,
    moves: []const machine.Move,
    scratch_base: machine.RegId,
    constants: []?ConstValue,
    stats: *Stats,
) Error!void {
    if (moves.len == 0) return;
    for (moves, 0..) |move, i| {
        if (move.src >= source.reg_types.len or move.dst >= source.reg_types.len) return error.InvalidMachine;
        if (move.src == move.dst) continue;
        const scratch = scratch_base + @as(machine.RegId, @intCast(i));
        try emitCopySlot(buffer, move.ty, scratch, move.src);
    }
    for (moves, 0..) |move, i| {
        if (move.src == move.dst) continue;
        const scratch = scratch_base + @as(machine.RegId, @intCast(i));
        try emitCopySlot(buffer, move.ty, move.dst, scratch);
        if (move.dst < constants.len) constants[move.dst] = null;
        stats.edge_moves += 1;
        stats.native_insts += 2;
    }
}

fn align16(value: u32) u32 {
    return (value + 15) & ~@as(u32, 15);
}

fn branchCondition(condition: machine.Condition) u8 {
    return switch (condition) {
        .eq => 0x84,
        .ne => 0x85,
        .lt => 0x8c,
        .ge => 0x8d,
        .gt => 0x8f,
        .le => 0x8e,
    };
}

fn emitCondBranch(
    buffer: *code_buffer.Buffer,
    labels: []const code_buffer.LabelId,
    source: *const machine.Function,
    from: cfg.BlockId,
    inst: machine.Inst,
    scratch_base: machine.RegId,
    constants: []?ConstValue,
    stats: *Stats,
) Error!void {
    const condition = inst.condition orelse return error.InvalidMachine;
    if (inst.uses.len == 1) {
        try emitCmpSlotZero(buffer, inst.uses[0]);
    } else if (inst.uses.len == 2) {
        try emitCmpSlots(buffer, inst.uses[0], inst.uses[1]);
    } else {
        return error.InvalidMachine;
    }
    const true_target = inst.target orelse return error.InvalidMachine;
    const false_target = inst.false_target orelse return error.InvalidMachine;
    const true_moves = edgeMoves(source, from, true_target);
    const false_moves = edgeMoves(source, from, false_target);
    if (true_moves.len == 0 and false_moves.len == 0) {
        try emitJcc(buffer, branchCondition(condition), try checkedTarget(labels, inst.target));
        try emitJump(buffer, try checkedTarget(labels, inst.false_target));
        return;
    }

    const true_edge_label = try buffer.newLabel();
    try emitJcc(buffer, branchCondition(condition), true_edge_label);
    try emitParallelEdgeMoves(buffer, source, false_moves, scratch_base, constants, stats);
    try emitJump(buffer, try checkedTarget(labels, false_target));
    try buffer.bindLabel(true_edge_label);
    try emitParallelEdgeMoves(buffer, source, true_moves, scratch_base, constants, stats);
    try emitJump(buffer, try checkedTarget(labels, true_target));
}

fn checkedTarget(labels: []const code_buffer.LabelId, target: ?cfg.BlockId) Error!code_buffer.LabelId {
    const block = target orelse return error.InvalidMachine;
    if (block >= labels.len) return error.InvalidMachine;
    return labels[block];
}

fn foldI32(constants: []?ConstValue, inst: machine.Inst) ?i32 {
    if (inst.defs.len != 1 or inst.uses.len != 2) return null;
    const a = getI32(constants, inst.uses[0]) orelse return null;
    const b = getI32(constants, inst.uses[1]) orelse return null;
    return switch (inst.opcode) {
        .add_i32 => a +% b,
        .sub_i32 => a -% b,
        .mul_i32 => a *% b,
        .and_i32 => a & b,
        .or_i32 => a | b,
        .xor_i32 => a ^ b,
        else => null,
    };
}

fn isTerminator(inst: machine.Inst) bool {
    return switch (inst.opcode) {
        .jump, .branch, .ret, .throw_, .switch_ => true,
        else => false,
    };
}

fn blockHasTerminator(block: machine.Block) bool {
    if (block.insts.len == 0) return false;
    return isTerminator(block.insts[block.insts.len - 1]);
}

fn encodeInst(
    buffer: *code_buffer.Buffer,
    labels: []const code_buffer.LabelId,
    source: *const machine.Function,
    block_id: cfg.BlockId,
    scratch_base: machine.RegId,
    constants: []?ConstValue,
    inst: machine.Inst,
    stats: *Stats,
) Error!void {
    switch (inst.opcode) {
        .const_i32 => {
            if (inst.defs.len != 1) return error.InvalidMachine;
            try emitStoreImm32Slot(buffer, inst.defs[0], @intCast(inst.imm));
            try setConst(constants, inst.defs[0], .{ .i32 = @intCast(inst.imm) });
            stats.constants += 1;
            stats.native_insts += 1;
        },
        .const_i64 => {
            if (inst.defs.len == 0 or inst.defs.len > 2) return error.InvalidMachine;
            const low = inst.defs[0];
            try emitMovRaxImm64(buffer, inst.imm);
            try emitStoreRaxSlot(buffer, low);
            try setConst(constants, low, .{ .i64 = inst.imm });
            if (inst.defs.len == 2) try setConst(constants, inst.defs[1], .{ .i64 = inst.imm });
            stats.constants += 1;
            stats.native_insts += 1;
        },
        .mov => {
            if (inst.defs.len == 1 and inst.uses.len == 1) {
                if (inst.uses[0] >= constants.len or inst.defs[0] >= constants.len) return error.InvalidMachine;
                constants[inst.defs[0]] = constants[inst.uses[0]];
                try emitLoadEaxSlot(buffer, inst.uses[0]);
                try emitStoreEaxSlot(buffer, inst.defs[0]);
            } else if (inst.defs.len == 2 and inst.uses.len == 2) {
                if (inst.uses[0] >= constants.len or inst.defs[0] >= constants.len) return error.InvalidMachine;
                constants[inst.defs[0]] = constants[inst.uses[0]];
                if (inst.defs[1] < constants.len and inst.uses[1] < constants.len) constants[inst.defs[1]] = constants[inst.uses[1]];
                try emitLoadRaxSlot(buffer, inst.uses[0]);
                try emitStoreRaxSlot(buffer, inst.defs[0]);
            } else {
                return error.InvalidMachine;
            }
            stats.native_insts += 1;
        },
        .add_i32, .sub_i32, .mul_i32, .and_i32, .or_i32, .xor_i32 => {
            if (inst.defs.len != 1 or inst.uses.len != 2) return error.InvalidMachine;
            try emitBinaryI32Slot(buffer, inst.opcode, inst.defs[0], inst.uses[0], inst.uses[1]);
            if (foldI32(constants, inst)) |value| {
                try setConst(constants, inst.defs[0], .{ .i32 = value });
                stats.folded_ops += 1;
            } else {
                clearDefs(constants, inst.defs);
            }
            stats.native_insts += 1;
        },
        .jump => {
            const target = inst.target orelse return error.InvalidMachine;
            try emitParallelEdgeMoves(buffer, source, edgeMoves(source, block_id, target), scratch_base, constants, stats);
            try emitJump(buffer, try checkedTarget(labels, target));
            stats.native_insts += 1;
            stats.jumps += 1;
        },
        .branch => {
            try emitCondBranch(buffer, labels, source, block_id, inst, scratch_base, constants, stats);
            stats.native_insts += 1;
            stats.conditional_branches += 1;
            stats.jumps += 1;
        },
        .ret => {
            if (inst.uses.len == 0) {
                try emitEpilogue(buffer);
            } else if (inst.uses.len == 1 or inst.uses.len == 2) {
                const reg = inst.uses[0];
                if (getI32(constants, reg)) |value| {
                    try emitMovEaxImm32(buffer, value);
                } else if (getI64(constants, reg)) |value| {
                    try emitMovRaxImm64(buffer, value);
                } else {
                    if (reg >= source.reg_types.len) return error.InvalidMachine;
                    switch (source.reg_types[reg]) {
                        .long, .double => try emitLoadRaxSlot(buffer, reg),
                        else => try emitLoadEaxSlot(buffer, reg),
                    }
                }
                try emitEpilogue(buffer);
            } else {
                return error.UnsupportedInstruction;
            }
            stats.native_insts += 1;
            stats.returns += 1;
        },
        else => {
            clearDefs(constants, inst.defs);
            return error.UnsupportedInstruction;
        },
    }
}

pub fn encode(allocator: std.mem.Allocator, source: *const machine.Function) Error!Function {
    source.verify() catch return error.InvalidMachine;

    var buffer = code_buffer.Buffer.init(allocator);
    errdefer buffer.deinit();

    const labels = try allocator.alloc(code_buffer.LabelId, source.blocks.len);
    errdefer allocator.free(labels);
    for (labels) |*label| label.* = try buffer.newLabel();

    const constants = try allocator.alloc(?ConstValue, source.reg_types.len);
    defer allocator.free(constants);
    @memset(constants, null);

    var stats: Stats = .{
        .blocks = @intCast(source.blocks.len),
        .stack_slots = @intCast(source.reg_types.len + maxEdgeMoveCount(source)),
    };

    const scratch_base: machine.RegId = @intCast(source.reg_types.len);
    try emitPrologue(&buffer, align16(@intCast(stats.stack_slots * 8)));
    var param_index: u32 = 0;
    for (source.value_kinds, 0..) |kind, value_id| {
        if (kind != .parameter) continue;
        try emitStoreParamSlot(&buffer, @intCast(value_id), param_index, source.reg_types[value_id]);
        param_index += 1;
        stats.parameter_loads += 1;
        stats.native_insts += 1;
    }
    for (source.blocks) |block| {
        try buffer.alignTo(16, 0x90);
        try buffer.bindLabel(labels[block.id]);
        for (block.insts) |inst| try encodeInst(&buffer, labels, source, block.id, scratch_base, constants, inst, &stats);
        if (!blockHasTerminator(block)) {
            if (block.id >= source.successors.len) return error.InvalidMachine;
            if (source.successors[block.id].len > 1) return error.InvalidMachine;
            if (source.successors[block.id].len == 1) {
                const target = source.successors[block.id][0];
                try emitParallelEdgeMoves(&buffer, source, edgeMoves(source, block.id, target), scratch_base, constants, &stats);
                try emitJump(&buffer, try checkedTarget(labels, target));
                stats.native_insts += 1;
                stats.jumps += 1;
            }
        }
    }
    try buffer.verify();
    stats.bytes = buffer.len();

    return .{
        .allocator = allocator,
        .source = source,
        .buffer = buffer,
        .block_labels = labels,
        .stats = stats,
    };
}

fn optimizedMachine(allocator: std.mem.Allocator, insts: []const Instruction) !*optimizer.OptimizedFunction {
    return try optimizer.optimize(allocator, insts, &.{}, .{});
}

fn expectExecI32(insts: []const Instruction, expected: i32) !void {
    if (builtin.cpu.arch != .x86_64) return error.SkipZigTest;

    var optimized = try optimizedMachine(std.testing.allocator, insts);
    defer optimized.deinit();
    var native = try encode(std.testing.allocator, &optimized.machine);
    defer native.deinit();
    const bytes = try native.finalize();
    defer std.testing.allocator.free(bytes);

    var cache = jit_memory.Cache.init(std.testing.allocator);
    defer cache.deinit();
    const allocation = try cache.addBytes(bytes);
    const Fn = fn () callconv(.c) i32;
    try std.testing.expectEqual(expected, allocation.typedEntry(Fn)());
}

fn expectExecI64(insts: []const Instruction, expected: i64) !void {
    if (builtin.cpu.arch != .x86_64) return error.SkipZigTest;

    var optimized = try optimizedMachine(std.testing.allocator, insts);
    defer optimized.deinit();
    var native = try encode(std.testing.allocator, &optimized.machine);
    defer native.deinit();
    const bytes = try native.finalize();
    defer std.testing.allocator.free(bytes);

    var cache = jit_memory.Cache.init(std.testing.allocator);
    defer cache.deinit();
    const allocation = try cache.addBytes(bytes);
    const Fn = fn () callconv(.c) i64;
    try std.testing.expectEqual(expected, allocation.typedEntry(Fn)());
}

fn expectUnaryI32(insts: []const Instruction, arg: i32, expected: i32) !void {
    if (builtin.cpu.arch != .x86_64) return error.SkipZigTest;

    var optimized = try optimizedMachine(std.testing.allocator, insts);
    defer optimized.deinit();
    var native = try encode(std.testing.allocator, &optimized.machine);
    defer native.deinit();
    const bytes = try native.finalize();
    defer std.testing.allocator.free(bytes);

    var cache = jit_memory.Cache.init(std.testing.allocator);
    defer cache.deinit();
    const allocation = try cache.addBytes(bytes);
    const Fn = fn (i32) callconv(.c) i32;
    try std.testing.expectEqual(expected, allocation.typedEntry(Fn)(arg));
}

fn expectUnaryI64(insts: []const Instruction, arg: i32, expected: i64) !void {
    if (builtin.cpu.arch != .x86_64) return error.SkipZigTest;

    var optimized = try optimizedMachine(std.testing.allocator, insts);
    defer optimized.deinit();
    var native = try encode(std.testing.allocator, &optimized.machine);
    defer native.deinit();
    const bytes = try native.finalize();
    defer std.testing.allocator.free(bytes);

    var cache = jit_memory.Cache.init(std.testing.allocator);
    defer cache.deinit();
    const allocation = try cache.addBytes(bytes);
    const Fn = fn (i32) callconv(.c) i64;
    try std.testing.expectEqual(expected, allocation.typedEntry(Fn)(arg));
}

fn expectBinaryI32(insts: []const Instruction, a: i32, b: i32, expected: i32) !void {
    if (builtin.cpu.arch != .x86_64) return error.SkipZigTest;

    var optimized = try optimizedMachine(std.testing.allocator, insts);
    defer optimized.deinit();
    var native = try encode(std.testing.allocator, &optimized.machine);
    defer native.deinit();
    const bytes = try native.finalize();
    defer std.testing.allocator.free(bytes);

    var cache = jit_memory.Cache.init(std.testing.allocator);
    defer cache.deinit();
    const allocation = try cache.addBytes(bytes);
    const Fn = fn (i32, i32) callconv(.c) i32;
    try std.testing.expectEqual(expected, allocation.typedEntry(Fn)(a, b));
}

test "x64_encoder emits native constant return bytes" {
    const insts = [_]Instruction{
        .{ .const_ = .{ .dest = 0, .value = 42 } },
        .{ .return_ = .{ .src = 0 } },
    };
    var optimized = try optimizedMachine(std.testing.allocator, &insts);
    defer optimized.deinit();

    var native = try encode(std.testing.allocator, &optimized.machine);
    defer native.deinit();

    const bytes = try native.finalize();
    defer std.testing.allocator.free(bytes);
    try std.testing.expect(bytes.len > 6);
    try std.testing.expectEqual(@as(u8, 0x55), bytes[0]);
    try std.testing.expect(std.mem.indexOf(u8, bytes, &[_]u8{ 0xb8, 42, 0, 0, 0 }) != null);
    try std.testing.expectEqual(@as(u8, 0xc3), bytes[bytes.len - 1]);
    try std.testing.expectEqual(@as(u32, 1), native.stats.returns);
}

test "x64_encoder executes constant return through jit memory" {
    if (builtin.cpu.arch != .x86_64) return error.SkipZigTest;

    const insts = [_]Instruction{
        .{ .const_ = .{ .dest = 0, .value = 7 } },
        .{ .return_ = .{ .src = 0 } },
    };
    var optimized = try optimizedMachine(std.testing.allocator, &insts);
    defer optimized.deinit();

    var native = try encode(std.testing.allocator, &optimized.machine);
    defer native.deinit();
    const bytes = try native.finalize();
    defer std.testing.allocator.free(bytes);

    var cache = jit_memory.Cache.init(std.testing.allocator);
    defer cache.deinit();
    const allocation = try cache.addBytes(bytes);
    const Fn = fn () callconv(.c) i32;
    const entry = allocation.typedEntry(Fn);
    try std.testing.expectEqual(@as(i32, 7), entry());
}

test "x64_encoder folds integer arithmetic before return" {
    if (builtin.cpu.arch != .x86_64) return error.SkipZigTest;

    const insts = [_]Instruction{
        .{ .const_ = .{ .dest = 0, .value = 12 } },
        .{ .const_ = .{ .dest = 1, .value = 5 } },
        .{ .add_int = .{ .dest = 2, .src1 = 0, .src2 = 1 } },
        .{ .return_ = .{ .src = 2 } },
    };
    var optimized = try optimizedMachine(std.testing.allocator, &insts);
    defer optimized.deinit();

    var native = try encode(std.testing.allocator, &optimized.machine);
    defer native.deinit();
    const bytes = try native.finalize();
    defer std.testing.allocator.free(bytes);

    var cache = jit_memory.Cache.init(std.testing.allocator);
    defer cache.deinit();
    const allocation = try cache.addBytes(bytes);
    const Fn = fn () callconv(.c) i32;
    try std.testing.expectEqual(@as(i32, 17), allocation.typedEntry(Fn)());
}

test "x64_encoder executes wide constant return through jit memory" {
    try expectExecI64(&[_]Instruction{
        .{ .const_wide = .{ .dest = 0, .value = 0x1122334455667788 } },
        .{ .return_wide = .{ .src = 0 } },
    }, 0x1122334455667788);

    try expectExecI64(&[_]Instruction{
        .{ .const_wide = .{ .dest = 0, .value = std.math.minInt(i64) } },
        .{ .return_wide = .{ .src = 0 } },
    }, std.math.minInt(i64));
}

test "x64_encoder executes move-wide through jit memory" {
    try expectExecI64(&[_]Instruction{
        .{ .const_wide = .{ .dest = 0, .value = 0x0102030405060708 } },
        .{ .move_wide = .{ .dest = 2, .src = 0 } },
        .{ .return_wide = .{ .src = 2 } },
    }, 0x0102030405060708);
}

test "x64_encoder executes parameter return through jit memory" {
    if (builtin.cpu.arch != .x86_64) return error.SkipZigTest;

    const insts = [_]Instruction{
        .{ .return_ = .{ .src = 0 } },
    };
    var optimized = try optimizedMachine(std.testing.allocator, &insts);
    defer optimized.deinit();

    var native = try encode(std.testing.allocator, &optimized.machine);
    defer native.deinit();
    const bytes = try native.finalize();
    defer std.testing.allocator.free(bytes);

    var cache = jit_memory.Cache.init(std.testing.allocator);
    defer cache.deinit();
    const allocation = try cache.addBytes(bytes);
    const Fn = fn (i32) callconv(.c) i32;
    try std.testing.expectEqual(@as(i32, 1234), allocation.typedEntry(Fn)(1234));
    try std.testing.expectEqual(@as(u32, 1), native.stats.parameter_loads);
}

test "x64_encoder executes parameter arithmetic through jit memory" {
    if (builtin.cpu.arch != .x86_64) return error.SkipZigTest;

    const insts = [_]Instruction{
        .{ .add_int = .{ .dest = 2, .src1 = 0, .src2 = 1 } },
        .{ .return_ = .{ .src = 2 } },
    };
    var optimized = try optimizedMachine(std.testing.allocator, &insts);
    defer optimized.deinit();

    var native = try encode(std.testing.allocator, &optimized.machine);
    defer native.deinit();
    const bytes = try native.finalize();
    defer std.testing.allocator.free(bytes);

    var cache = jit_memory.Cache.init(std.testing.allocator);
    defer cache.deinit();
    const allocation = try cache.addBytes(bytes);
    const Fn = fn (i32, i32) callconv(.c) i32;
    try std.testing.expectEqual(@as(i32, 42), allocation.typedEntry(Fn)(30, 12));
    try std.testing.expect(native.stats.parameter_loads >= 2);
}

test "x64_encoder executes all supported i32 binary arithmetic through jit memory" {
    try expectBinaryI32(&[_]Instruction{
        .{ .sub_int = .{ .dest = 2, .src1 = 0, .src2 = 1 } },
        .{ .return_ = .{ .src = 2 } },
    }, -10, 7, -17);

    try expectBinaryI32(&[_]Instruction{
        .{ .mul_int = .{ .dest = 2, .src1 = 0, .src2 = 1 } },
        .{ .return_ = .{ .src = 2 } },
    }, -9, 6, -54);

    try expectBinaryI32(&[_]Instruction{
        .{ .mul_int = .{ .dest = 2, .src1 = 0, .src2 = 1 } },
        .{ .return_ = .{ .src = 2 } },
    }, std.math.maxInt(i32), 2, -2);

    try expectBinaryI32(&[_]Instruction{
        .{ .and_int = .{ .dest = 2, .src1 = 0, .src2 = 1 } },
        .{ .return_ = .{ .src = 2 } },
    }, @bitCast(@as(u32, 0xff00_ff00)), @bitCast(@as(u32, 0x0f0f_f0f0)), @bitCast(@as(u32, 0x0f00_f000)));

    try expectBinaryI32(&[_]Instruction{
        .{ .or_int = .{ .dest = 2, .src1 = 0, .src2 = 1 } },
        .{ .return_ = .{ .src = 2 } },
    }, @bitCast(@as(u32, 0xf000_000f)), @bitCast(@as(u32, 0x0f00_00f0)), @bitCast(@as(u32, 0xff00_00ff)));

    try expectBinaryI32(&[_]Instruction{
        .{ .xor_int = .{ .dest = 2, .src1 = 0, .src2 = 1 } },
        .{ .return_ = .{ .src = 2 } },
    }, @bitCast(@as(u32, 0xffff_0000)), @bitCast(@as(u32, 0x0f0f_0f0f)), @bitCast(@as(u32, 0xf0f0_0f0f)));
}

test "x64_encoder preserves wrapping i32 constant arithmetic" {
    try expectExecI32(&[_]Instruction{
        .{ .const_ = .{ .dest = 0, .value = std.math.maxInt(i32) } },
        .{ .const_ = .{ .dest = 1, .value = 1 } },
        .{ .add_int = .{ .dest = 2, .src1 = 0, .src2 = 1 } },
        .{ .return_ = .{ .src = 2 } },
    }, std.math.minInt(i32));

    try expectExecI32(&[_]Instruction{
        .{ .const_ = .{ .dest = 0, .value = std.math.minInt(i32) } },
        .{ .const_ = .{ .dest = 1, .value = 1 } },
        .{ .sub_int = .{ .dest = 2, .src1 = 0, .src2 = 1 } },
        .{ .return_ = .{ .src = 2 } },
    }, std.math.maxInt(i32));
}

test "x64_encoder executes if-eqz branch through jit memory" {
    if (builtin.cpu.arch != .x86_64) return error.SkipZigTest;

    const insts = [_]Instruction{
        .{ .if_eqz = .{ .src = 0, .offset = 3 } },
        .{ .const_ = .{ .dest = 1, .value = 1 } },
        .{ .return_ = .{ .src = 1 } },
        .{ .const_ = .{ .dest = 1, .value = 2 } },
        .{ .return_ = .{ .src = 1 } },
    };
    var optimized = try optimizedMachine(std.testing.allocator, &insts);
    defer optimized.deinit();

    var native = try encode(std.testing.allocator, &optimized.machine);
    defer native.deinit();
    const bytes = try native.finalize();
    defer std.testing.allocator.free(bytes);

    var cache = jit_memory.Cache.init(std.testing.allocator);
    defer cache.deinit();
    const allocation = try cache.addBytes(bytes);
    const Fn = fn (i32) callconv(.c) i32;
    const entry = allocation.typedEntry(Fn);
    try std.testing.expectEqual(@as(i32, 2), entry(0));
    try std.testing.expectEqual(@as(i32, 1), entry(9));
    try std.testing.expectEqual(@as(u32, 1), native.stats.conditional_branches);
}

test "x64_encoder executes if-eq parameter branch through jit memory" {
    if (builtin.cpu.arch != .x86_64) return error.SkipZigTest;

    const insts = [_]Instruction{
        .{ .if_eq = .{ .src1 = 0, .src2 = 1, .offset = 3 } },
        .{ .const_ = .{ .dest = 2, .value = 10 } },
        .{ .return_ = .{ .src = 2 } },
        .{ .const_ = .{ .dest = 2, .value = 20 } },
        .{ .return_ = .{ .src = 2 } },
    };
    var optimized = try optimizedMachine(std.testing.allocator, &insts);
    defer optimized.deinit();

    var native = try encode(std.testing.allocator, &optimized.machine);
    defer native.deinit();
    const bytes = try native.finalize();
    defer std.testing.allocator.free(bytes);

    var cache = jit_memory.Cache.init(std.testing.allocator);
    defer cache.deinit();
    const allocation = try cache.addBytes(bytes);
    const Fn = fn (i32, i32) callconv(.c) i32;
    const entry = allocation.typedEntry(Fn);
    try std.testing.expectEqual(@as(i32, 20), entry(7, 7));
    try std.testing.expectEqual(@as(i32, 10), entry(7, 8));
}

test "x64_encoder executes remaining zero branch conditions through jit memory" {
    try expectUnaryI32(&[_]Instruction{
        .{ .if_nez = .{ .src = 0, .offset = 3 } },
        .{ .const_ = .{ .dest = 1, .value = 1 } },
        .{ .return_ = .{ .src = 1 } },
        .{ .const_ = .{ .dest = 1, .value = 2 } },
        .{ .return_ = .{ .src = 1 } },
    }, 0, 1);
    try expectUnaryI32(&[_]Instruction{
        .{ .if_nez = .{ .src = 0, .offset = 3 } },
        .{ .const_ = .{ .dest = 1, .value = 1 } },
        .{ .return_ = .{ .src = 1 } },
        .{ .const_ = .{ .dest = 1, .value = 2 } },
        .{ .return_ = .{ .src = 1 } },
    }, -1, 2);

    try expectUnaryI32(&[_]Instruction{
        .{ .if_ltz = .{ .src = 0, .offset = 3 } },
        .{ .const_ = .{ .dest = 1, .value = 1 } },
        .{ .return_ = .{ .src = 1 } },
        .{ .const_ = .{ .dest = 1, .value = 2 } },
        .{ .return_ = .{ .src = 1 } },
    }, -1, 2);
    try expectUnaryI32(&[_]Instruction{
        .{ .if_ltz = .{ .src = 0, .offset = 3 } },
        .{ .const_ = .{ .dest = 1, .value = 1 } },
        .{ .return_ = .{ .src = 1 } },
        .{ .const_ = .{ .dest = 1, .value = 2 } },
        .{ .return_ = .{ .src = 1 } },
    }, 0, 1);

    try expectUnaryI32(&[_]Instruction{
        .{ .if_gez = .{ .src = 0, .offset = 3 } },
        .{ .const_ = .{ .dest = 1, .value = 1 } },
        .{ .return_ = .{ .src = 1 } },
        .{ .const_ = .{ .dest = 1, .value = 2 } },
        .{ .return_ = .{ .src = 1 } },
    }, 0, 2);
    try expectUnaryI32(&[_]Instruction{
        .{ .if_gtz = .{ .src = 0, .offset = 3 } },
        .{ .const_ = .{ .dest = 1, .value = 1 } },
        .{ .return_ = .{ .src = 1 } },
        .{ .const_ = .{ .dest = 1, .value = 2 } },
        .{ .return_ = .{ .src = 1 } },
    }, 0, 1);
    try expectUnaryI32(&[_]Instruction{
        .{ .if_gtz = .{ .src = 0, .offset = 3 } },
        .{ .const_ = .{ .dest = 1, .value = 1 } },
        .{ .return_ = .{ .src = 1 } },
        .{ .const_ = .{ .dest = 1, .value = 2 } },
        .{ .return_ = .{ .src = 1 } },
    }, 1, 2);
    try expectUnaryI32(&[_]Instruction{
        .{ .if_lez = .{ .src = 0, .offset = 3 } },
        .{ .const_ = .{ .dest = 1, .value = 1 } },
        .{ .return_ = .{ .src = 1 } },
        .{ .const_ = .{ .dest = 1, .value = 2 } },
        .{ .return_ = .{ .src = 1 } },
    }, 0, 2);
    try expectUnaryI32(&[_]Instruction{
        .{ .if_lez = .{ .src = 0, .offset = 3 } },
        .{ .const_ = .{ .dest = 1, .value = 1 } },
        .{ .return_ = .{ .src = 1 } },
        .{ .const_ = .{ .dest = 1, .value = 2 } },
        .{ .return_ = .{ .src = 1 } },
    }, 1, 1);
}

test "x64_encoder executes remaining parameter branch conditions through jit memory" {
    try expectBinaryI32(&[_]Instruction{
        .{ .if_ne = .{ .src1 = 0, .src2 = 1, .offset = 3 } },
        .{ .const_ = .{ .dest = 2, .value = 1 } },
        .{ .return_ = .{ .src = 2 } },
        .{ .const_ = .{ .dest = 2, .value = 2 } },
        .{ .return_ = .{ .src = 2 } },
    }, 4, 5, 2);
    try expectBinaryI32(&[_]Instruction{
        .{ .if_lt = .{ .src1 = 0, .src2 = 1, .offset = 3 } },
        .{ .const_ = .{ .dest = 2, .value = 1 } },
        .{ .return_ = .{ .src = 2 } },
        .{ .const_ = .{ .dest = 2, .value = 2 } },
        .{ .return_ = .{ .src = 2 } },
    }, -3, 2, 2);
    try expectBinaryI32(&[_]Instruction{
        .{ .if_ge = .{ .src1 = 0, .src2 = 1, .offset = 3 } },
        .{ .const_ = .{ .dest = 2, .value = 1 } },
        .{ .return_ = .{ .src = 2 } },
        .{ .const_ = .{ .dest = 2, .value = 2 } },
        .{ .return_ = .{ .src = 2 } },
    }, 2, 2, 2);
    try expectBinaryI32(&[_]Instruction{
        .{ .if_gt = .{ .src1 = 0, .src2 = 1, .offset = 3 } },
        .{ .const_ = .{ .dest = 2, .value = 1 } },
        .{ .return_ = .{ .src = 2 } },
        .{ .const_ = .{ .dest = 2, .value = 2 } },
        .{ .return_ = .{ .src = 2 } },
    }, 3, 2, 2);
    try expectBinaryI32(&[_]Instruction{
        .{ .if_le = .{ .src1 = 0, .src2 = 1, .offset = 3 } },
        .{ .const_ = .{ .dest = 2, .value = 1 } },
        .{ .return_ = .{ .src = 2 } },
        .{ .const_ = .{ .dest = 2, .value = 2 } },
        .{ .return_ = .{ .src = 2 } },
    }, 2, 3, 2);
}

test "x64_encoder executes phi edge moves at diamond join" {
    const insts = [_]Instruction{
        .{ .if_eqz = .{ .src = 0, .offset = 3 } },
        .{ .const_ = .{ .dest = 1, .value = 10 } },
        .{ .goto_ = .{ .offset = 2 } },
        .{ .const_ = .{ .dest = 1, .value = 20 } },
        .{ .return_ = .{ .src = 1 } },
    };
    try expectUnaryI32(&insts, 0, 20);
    try expectUnaryI32(&insts, 5, 10);

    var optimized = try optimizedMachine(std.testing.allocator, &insts);
    defer optimized.deinit();
    var native = try encode(std.testing.allocator, &optimized.machine);
    defer native.deinit();
    try std.testing.expect(native.stats.edge_moves >= 2);
}

test "x64_encoder executes wide phi edge moves at diamond join" {
    const insts = [_]Instruction{
        .{ .if_eqz = .{ .src = 0, .offset = 3 } },
        .{ .const_wide = .{ .dest = 2, .value = 0x1111222233334444 } },
        .{ .goto_ = .{ .offset = 2 } },
        .{ .const_wide = .{ .dest = 2, .value = 0x5555666677770001 } },
        .{ .return_wide = .{ .src = 2 } },
    };
    try expectUnaryI64(&insts, 0, 0x5555666677770001);
    try expectUnaryI64(&insts, 7, 0x1111222233334444);
}

test "x64_encoder emits direct jump relocation" {
    const insts = [_]Instruction{
        .{ .goto_ = .{ .offset = 2 } },
        .{ .const_ = .{ .dest = 0, .value = 99 } },
        .return_void,
    };
    var optimized = try optimizer.optimize(std.testing.allocator, &insts, &.{}, .{
        .cfg_options = .{ .prune_unreachable = false, .order = .linear },
    });
    defer optimized.deinit();

    var native = try encode(std.testing.allocator, &optimized.machine);
    defer native.deinit();
    const bytes = try native.finalize();
    defer std.testing.allocator.free(bytes);

    try std.testing.expect(bytes.len >= 6);
    try std.testing.expect(std.mem.indexOfScalar(u8, bytes, 0xe9) != null);
    try std.testing.expectEqual(@as(u32, 1), native.stats.jumps);
}

test "x64_encoder rejects unsupported operations explicitly" {
    const insts = [_]Instruction{
        .{ .new_instance = .{ .dest = 0, .type_idx = 1 } },
        .{ .return_object = .{ .src = 0 } },
    };
    var optimized = try optimizedMachine(std.testing.allocator, &insts);
    defer optimized.deinit();

    try std.testing.expectError(error.UnsupportedInstruction, encode(std.testing.allocator, &optimized.machine));
}

test "x64_encoder print helper emits stable summary" {
    const insts = [_]Instruction{
        .return_void,
    };
    var optimized = try optimizedMachine(std.testing.allocator, &insts);
    defer optimized.deinit();

    var native = try encode(std.testing.allocator, &optimized.machine);
    defer native.deinit();

    var storage: [512]u8 = undefined;
    var stream = std.Io.Writer.fixed(&storage);
    try native.print(&stream);
    const output = stream.buffered();
    try std.testing.expect(std.mem.indexOf(u8, output, "x64_encoder bytes=") != null);
    try std.testing.expect(std.mem.indexOf(u8, output, "code_buffer bytes=") != null);
}
