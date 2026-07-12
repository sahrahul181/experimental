//! Register-machine binary encoder.
//!
//! This is the first concrete emission layer after `machine_bridge.zig`. It
//! serializes register-machine instructions into a compact, relocatable byte
//! stream backed by `code_buffer.zig`. A native backend can either replace this
//! opcode mapping or use the same structure while writing real target bytes.

const std = @import("std");
const cfg = @import("cfg");
const code_buffer = @import("code_buffer");
const machine = @import("machine_bridge");
const optimizer = @import("optimizer");
const typedir = @import("typedir");

pub const Error = code_buffer.Error || error{
    InvalidMachine,
    TooManyOperands,
};

const MAGIC = [_]u8{ 'R', 'M', 'C', '1' };
const NO_PC: u32 = std.math.maxInt(u32);
const NO_FIELD: u32 = std.math.maxInt(u32);

const Record = enum(u8) {
    header = 0x01,
    edge_moves = 0x02,
    block = 0x03,
    inst = 0x04,
    target_rel32 = 0x05,
};

pub const Stats = struct {
    bytes: u32 = 0,
    blocks: u32 = 0,
    instructions: u32 = 0,
    edge_move_records: u32 = 0,
    edge_moves: u32 = 0,
    branch_relocs: u32 = 0,
};

pub const EncodedFunction = struct {
    allocator: std.mem.Allocator,
    source: *const machine.Function,
    buffer: code_buffer.Buffer,
    block_labels: []code_buffer.LabelId,
    stats: Stats,

    pub fn deinit(self: *EncodedFunction) void {
        self.allocator.free(self.block_labels);
        self.buffer.deinit();
        self.* = undefined;
    }

    pub fn verify(self: *EncodedFunction) Error!void {
        self.source.verify() catch return error.InvalidMachine;
        if (self.block_labels.len != self.source.blocks.len) return error.InvalidMachine;
        try self.buffer.verify();
    }

    pub fn finalize(self: *EncodedFunction) Error![]u8 {
        try self.verify();
        const bytes = try self.buffer.finalize();
        self.stats.bytes = @intCast(bytes.len);
        return bytes;
    }

    pub fn print(self: *const EncodedFunction, writer: anytype) !void {
        try writer.print(
            "register_encoder bytes={d} blocks={d} insts={d} edge_records={d} edge_moves={d} branch_relocs={d}\n",
            .{
                self.buffer.len(),
                self.stats.blocks,
                self.stats.instructions,
                self.stats.edge_move_records,
                self.stats.edge_moves,
                self.stats.branch_relocs,
            },
        );
        try self.buffer.print(writer);
    }
};

fn opcodeByte(opcode: machine.Opcode) u8 {
    return @intFromEnum(opcode);
}

fn typeByte(ty: typedir.Type) u8 {
    return @intFromEnum(ty);
}

fn flagByte(flags: machine.Flags) u8 {
    var out: u8 = 0;
    if (flags.null_check_elided) out |= 1 << 0;
    if (flags.bounds_check_elided) out |= 1 << 1;
    if (flags.forwarded) out |= 1 << 2;
    if (flags.cse) out |= 1 << 3;
    return out;
}

fn emitI64(buffer: *code_buffer.Buffer, value: i64) Error!void {
    try buffer.emitU64(@bitCast(value));
}

fn emitRegList(buffer: *code_buffer.Buffer, regs: []const machine.RegId) Error!void {
    if (regs.len > std.math.maxInt(u8)) return error.TooManyOperands;
    try buffer.emitU8(@intCast(regs.len));
    for (regs) |reg| try buffer.emitU32(reg);
}

fn emitHeader(buffer: *code_buffer.Buffer, function: *const machine.Function) Error!void {
    try buffer.emitU8(@intFromEnum(Record.header));
    try buffer.emitBytes(&MAGIC);
    try buffer.emitU16(1);
    try buffer.emitU32(@intCast(function.reg_types.len));
    try buffer.emitU32(@intCast(function.blocks.len));
}

fn emitEdgeMoves(buffer: *code_buffer.Buffer, edge: machine.EdgeMoves, stats: *Stats) Error!void {
    if (edge.moves.len == 0) return;
    if (edge.moves.len > std.math.maxInt(u16)) return error.TooManyOperands;
    try buffer.emitU8(@intFromEnum(Record.edge_moves));
    try buffer.emitU32(edge.from);
    try buffer.emitU32(edge.to);
    try buffer.emitU16(@intCast(edge.moves.len));
    for (edge.moves) |move| {
        try buffer.emitU32(move.dst);
        try buffer.emitU32(move.src);
        try buffer.emitU8(typeByte(move.ty));
    }
    stats.edge_move_records += 1;
    stats.edge_moves += @intCast(edge.moves.len);
}

fn emitTarget(buffer: *code_buffer.Buffer, label: code_buffer.LabelId, stats: *Stats) Error!void {
    try buffer.emitU8(@intFromEnum(Record.target_rel32));
    _ = try buffer.reloc(label, .rel32, 0);
    stats.branch_relocs += 1;
}

fn checkedLabel(labels: []const code_buffer.LabelId, target: ?cfg.BlockId) Error!code_buffer.LabelId {
    const id = target orelse return error.InvalidMachine;
    if (id >= labels.len) return error.InvalidMachine;
    return labels[id];
}

fn emitInst(buffer: *code_buffer.Buffer, labels: []const code_buffer.LabelId, inst: machine.Inst, stats: *Stats) Error!void {
    try buffer.emitU8(@intFromEnum(Record.inst));
    try buffer.emitU8(opcodeByte(inst.opcode));
    try buffer.emitU8(flagByte(inst.flags));
    try buffer.emitU32(inst.pc orelse NO_PC);
    try emitRegList(buffer, inst.defs);
    try emitRegList(buffer, inst.uses);
    try emitI64(buffer, inst.imm);
    try buffer.emitU32(inst.field_idx orelse NO_FIELD);

    switch (inst.opcode) {
        .jump => try emitTarget(buffer, try checkedLabel(labels, inst.target), stats),
        .branch => {
            try emitTarget(buffer, try checkedLabel(labels, inst.target), stats);
            try emitTarget(buffer, try checkedLabel(labels, inst.false_target), stats);
        },
        else => {},
    }
    stats.instructions += 1;
}

pub fn encode(allocator: std.mem.Allocator, function: *const machine.Function) Error!EncodedFunction {
    function.verify() catch return error.InvalidMachine;

    var buffer = code_buffer.Buffer.init(allocator);
    errdefer buffer.deinit();

    const labels = try allocator.alloc(code_buffer.LabelId, function.blocks.len);
    errdefer allocator.free(labels);
    for (labels) |*label| label.* = try buffer.newLabel();

    var stats: Stats = .{ .blocks = @intCast(function.blocks.len) };
    try emitHeader(&buffer, function);

    for (function.edges) |edge| try emitEdgeMoves(&buffer, edge, &stats);

    for (function.blocks) |block| {
        try buffer.alignTo(4, 0);
        try buffer.bindLabel(labels[block.id]);
        try buffer.emitU8(@intFromEnum(Record.block));
        try buffer.emitU32(block.id);
        try buffer.emitU32(@intCast(block.insts.len));
        for (block.insts) |inst| try emitInst(&buffer, labels, inst, &stats);
    }

    try buffer.verify();
    stats.bytes = buffer.len();

    return .{
        .allocator = allocator,
        .source = function,
        .buffer = buffer,
        .block_labels = labels,
        .stats = stats,
    };
}

test "register_encoder emits optimized function into relocatable bytes" {
    const insts = [_]@import("instructions").Instruction{
        .{ .const_ = .{ .dest = 0, .value = 1 } },
        .{ .const_ = .{ .dest = 1, .value = 2 } },
        .{ .add_int = .{ .dest = 2, .src1 = 0, .src2 = 1 } },
        .{ .return_ = .{ .src = 2 } },
    };
    var optimized = try optimizer.optimize(std.testing.allocator, &insts, &.{}, .{});
    defer optimized.deinit();

    var encoded = try encode(std.testing.allocator, &optimized.machine);
    defer encoded.deinit();
    try encoded.verify();

    const bytes = try encoded.finalize();
    defer std.testing.allocator.free(bytes);
    try std.testing.expect(bytes.len > MAGIC.len);
    try std.testing.expect(std.mem.indexOf(u8, bytes, &MAGIC) != null);
    try std.testing.expect(encoded.stats.instructions > 0);
}

test "register_encoder emits branch relocations" {
    const insts = [_]@import("instructions").Instruction{
        .{ .const_ = .{ .dest = 0, .value = 1 } },
        .{ .if_eqz = .{ .src = 0, .offset = 2 } },
        .return_void,
        .{ .return_ = .{ .src = 0 } },
    };
    var optimized = try optimizer.optimize(std.testing.allocator, &insts, &.{}, .{});
    defer optimized.deinit();

    var encoded = try encode(std.testing.allocator, &optimized.machine);
    defer encoded.deinit();

    try std.testing.expect(encoded.stats.branch_relocs >= 2);
    const bytes = try encoded.finalize();
    defer std.testing.allocator.free(bytes);
    try std.testing.expect(encoded.buffer.patched);
}

test "register_encoder emits phi edge move records" {
    const insts = [_]@import("instructions").Instruction{
        .{ .if_eqz = .{ .src = 0, .offset = 3 } },
        .{ .const_ = .{ .dest = 1, .value = 10 } },
        .{ .goto_ = .{ .offset = 2 } },
        .{ .const_ = .{ .dest = 1, .value = 20 } },
        .{ .return_ = .{ .src = 1 } },
    };
    var optimized = try optimizer.optimize(std.testing.allocator, &insts, &.{}, .{});
    defer optimized.deinit();

    var encoded = try encode(std.testing.allocator, &optimized.machine);
    defer encoded.deinit();

    try std.testing.expect(encoded.stats.edge_move_records >= 1);
    try std.testing.expect(encoded.stats.edge_moves >= 2);
}

test "register_encoder print helper emits stable summary" {
    const insts = [_]@import("instructions").Instruction{
        .{ .const_ = .{ .dest = 0, .value = 1 } },
        .return_void,
    };
    var optimized = try optimizer.optimize(std.testing.allocator, &insts, &.{}, .{});
    defer optimized.deinit();

    var encoded = try encode(std.testing.allocator, &optimized.machine);
    defer encoded.deinit();

    var storage: [1024]u8 = undefined;
    var stream = std.Io.Writer.fixed(&storage);
    try encoded.print(&stream);
    const output = stream.buffered();
    try std.testing.expect(std.mem.indexOf(u8, output, "register_encoder bytes=") != null);
    try std.testing.expect(std.mem.indexOf(u8, output, "code_buffer bytes=") != null);
}
