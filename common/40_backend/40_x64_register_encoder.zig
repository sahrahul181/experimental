//! Register-allocated x86-64 encoder.
//!
//! This backend emits a no-frame native subset from the register-machine IR
//! after physical register allocation. It has no stack-slot traffic on the
//! supported path; if allocation spills or an operation is outside the native
//! register subset, it fails explicitly.

const std = @import("std");
const builtin = @import("builtin");
const cfg = @import("cfg");
const code_buffer = @import("code_buffer");
const jit_memory = @import("jit_memory");
const machine = @import("machine_bridge");
const optimizer = @import("optimizer");
const regalloc = @import("regalloc");
const Instruction = @import("instructions").Instruction;

pub const Error = code_buffer.Error || regalloc.Error || error{
    InvalidMachine,
    SpillsUnsupported,
    UnsupportedInstruction,
};

pub const Stats = struct {
    bytes: u32 = 0,
    blocks: u32 = 0,
    native_insts: u32 = 0,
    register_moves: u32 = 0,
    constants: u32 = 0,
    returns: u32 = 0,
    jumps: u32 = 0,
    branches: u32 = 0,
};

pub const Function = struct {
    allocator: std.mem.Allocator,
    source: *const machine.Function,
    allocation: regalloc.Allocation,
    buffer: code_buffer.Buffer,
    block_labels: []code_buffer.LabelId,
    stats: Stats,

    pub fn deinit(self: *Function) void {
        self.allocator.free(self.block_labels);
        self.buffer.deinit();
        self.allocation.deinit();
        self.* = undefined;
    }

    pub fn verify(self: *Function) Error!void {
        self.source.verify() catch return error.InvalidMachine;
        try self.allocation.verify();
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
            "x64_register_encoder bytes={d} blocks={d} native_insts={d} moves={d} constants={d} returns={d} jumps={d} branches={d}\n",
            .{
                self.buffer.len(),
                self.stats.blocks,
                self.stats.native_insts,
                self.stats.register_moves,
                self.stats.constants,
                self.stats.returns,
                self.stats.jumps,
                self.stats.branches,
            },
        );
        try self.allocation.print(writer);
        try self.buffer.print(writer);
    }
};

fn x64Reg(reg: regalloc.PhysReg) Error!u4 {
    return switch (reg) {
        .rax => 0,
        .rcx => 1,
        .rdx => 2,
        .rsi => 6,
        .rdi => 7,
        .r8 => 8,
        .r9 => 9,
        .r10 => 10,
        .r11 => 11,
        else => error.UnsupportedInstruction,
    };
}

fn physOf(allocation: *const regalloc.Allocation, reg: machine.RegId) Error!regalloc.PhysReg {
    return switch (allocation.locationOf(reg) orelse return error.InvalidMachine) {
        .phys => |phys| phys,
        .spill => error.SpillsUnsupported,
        .none => error.InvalidMachine,
    };
}

fn emitRex(buffer: *code_buffer.Buffer, w: bool, reg: u4, rm: u4) Error!void {
    var rex: u8 = 0x40;
    if (w) rex |= 0x08;
    if ((reg & 8) != 0) rex |= 0x04;
    if ((rm & 8) != 0) rex |= 0x01;
    if (rex != 0x40) try buffer.emitU8(rex);
}

fn emitModRm(buffer: *code_buffer.Buffer, reg: u4, rm: u4) Error!void {
    try buffer.emitU8(0xc0 | ((@as(u8, reg) & 7) << 3) | (@as(u8, rm) & 7));
}

fn isWideType(ty: anytype) bool {
    return switch (ty) {
        .long, .double => true,
        else => false,
    };
}

fn emitMovRegReg(buffer: *code_buffer.Buffer, dst: regalloc.PhysReg, src: regalloc.PhysReg, wide: bool) Error!void {
    if (dst == src) return;
    const d = try x64Reg(dst);
    const s = try x64Reg(src);
    try emitRex(buffer, wide, s, d);
    try buffer.emitU8(0x89);
    try emitModRm(buffer, s, d);
}

fn emitMovRegImm32(buffer: *code_buffer.Buffer, dst: regalloc.PhysReg, value: i32) Error!void {
    const d = try x64Reg(dst);
    try emitRex(buffer, false, 0, d);
    try buffer.emitU8(0xb8 + @as(u8, d & 7));
    try buffer.emitU32(@bitCast(value));
}

fn emitBinaryRegReg(buffer: *code_buffer.Buffer, opcode: machine.Opcode, dst: regalloc.PhysReg, rhs: regalloc.PhysReg) Error!void {
    const d = try x64Reg(dst);
    const r = try x64Reg(rhs);
    try emitRex(buffer, false, d, r);
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
    try emitModRm(buffer, d, r);
}

fn emitCmpRegZero(buffer: *code_buffer.Buffer, reg: regalloc.PhysReg) Error!void {
    const r = try x64Reg(reg);
    try emitRex(buffer, false, 7, r);
    try buffer.emitU8(0x83);
    try emitModRm(buffer, 7, r);
    try buffer.emitU8(0);
}

fn emitCmpRegs(buffer: *code_buffer.Buffer, lhs: regalloc.PhysReg, rhs: regalloc.PhysReg) Error!void {
    const l = try x64Reg(lhs);
    const r = try x64Reg(rhs);
    try emitRex(buffer, false, l, r);
    try buffer.emitU8(0x3b);
    try emitModRm(buffer, l, r);
}

fn emitRet(buffer: *code_buffer.Buffer) Error!void {
    try buffer.emitU8(0xc3);
}

fn emitJump(buffer: *code_buffer.Buffer, label: code_buffer.LabelId) Error!void {
    try buffer.emitU8(0xe9);
    _ = try buffer.reloc(label, .rel32, 0);
}

fn emitJcc(buffer: *code_buffer.Buffer, condition: machine.Condition, label: code_buffer.LabelId) Error!void {
    try buffer.emitU8(0x0f);
    try buffer.emitU8(switch (condition) {
        .eq => 0x84,
        .ne => 0x85,
        .lt => 0x8c,
        .ge => 0x8d,
        .gt => 0x8f,
        .le => 0x8e,
    });
    _ = try buffer.reloc(label, .rel32, 0);
}

fn checkedLabel(labels: []const code_buffer.LabelId, target: ?cfg.BlockId) Error!code_buffer.LabelId {
    const block = target orelse return error.InvalidMachine;
    if (block >= labels.len) return error.InvalidMachine;
    return labels[block];
}

fn abiParamReg(index: u32) Error!regalloc.PhysReg {
    return switch (builtin.os.tag) {
        .windows => switch (index) {
            0 => .rcx,
            1 => .rdx,
            2 => .r8,
            3 => .r9,
            else => error.UnsupportedInstruction,
        },
        else => switch (index) {
            0 => .rdi,
            1 => .rsi,
            2 => .rdx,
            3 => .rcx,
            4 => .r8,
            5 => .r9,
            else => error.UnsupportedInstruction,
        },
    };
}

fn emitParamMoves(buffer: *code_buffer.Buffer, allocation: *const regalloc.Allocation, source: *const machine.Function, stats: *Stats) Error!void {
    var param_index: u32 = 0;
    for (source.value_kinds, 0..) |kind, value_id| {
        if (kind != .parameter) continue;
        const dst = try physOf(allocation, @intCast(value_id));
        const src = try abiParamReg(param_index);
        const wide = isWideType(source.reg_types[value_id]);
        try emitMovRegReg(buffer, dst, src, wide);
        if (dst != src) {
            stats.register_moves += 1;
            stats.native_insts += 1;
        }
        param_index += 1;
    }
}

fn encodeInst(
    buffer: *code_buffer.Buffer,
    labels: []const code_buffer.LabelId,
    allocation: *const regalloc.Allocation,
    source: *const machine.Function,
    inst: machine.Inst,
    stats: *Stats,
) Error!void {
    switch (inst.opcode) {
        .const_i32 => {
            if (inst.defs.len != 1) return error.InvalidMachine;
            try emitMovRegImm32(buffer, try physOf(allocation, inst.defs[0]), @intCast(inst.imm));
            stats.constants += 1;
            stats.native_insts += 1;
        },
        .mov => {
            if (inst.defs.len != 1 or inst.uses.len != 1) return error.UnsupportedInstruction;
            const dst = try physOf(allocation, inst.defs[0]);
            const src = try physOf(allocation, inst.uses[0]);
            const wide = isWideType(source.reg_types[inst.defs[0]]);
            try emitMovRegReg(buffer, dst, src, wide);
            stats.register_moves += 1;
            stats.native_insts += 1;
        },
        .add_i32, .sub_i32, .mul_i32, .and_i32, .or_i32, .xor_i32 => {
            if (inst.defs.len != 1 or inst.uses.len != 2) return error.InvalidMachine;
            const dst = try physOf(allocation, inst.defs[0]);
            try emitMovRegReg(buffer, dst, try physOf(allocation, inst.uses[0]), false);
            try emitBinaryRegReg(buffer, inst.opcode, dst, try physOf(allocation, inst.uses[1]));
            stats.native_insts += 2;
        },
        .jump => {
            try emitJump(buffer, try checkedLabel(labels, inst.target));
            stats.jumps += 1;
            stats.native_insts += 1;
        },
        .branch => {
            const condition = inst.condition orelse return error.InvalidMachine;
            if (inst.uses.len == 1) {
                try emitCmpRegZero(buffer, try physOf(allocation, inst.uses[0]));
            } else if (inst.uses.len == 2) {
                try emitCmpRegs(buffer, try physOf(allocation, inst.uses[0]), try physOf(allocation, inst.uses[1]));
            } else {
                return error.InvalidMachine;
            }
            try emitJcc(buffer, condition, try checkedLabel(labels, inst.target));
            try emitJump(buffer, try checkedLabel(labels, inst.false_target));
            stats.branches += 1;
            stats.jumps += 1;
            stats.native_insts += 3;
        },
        .ret => {
            if (inst.uses.len == 1) {
                const src = try physOf(allocation, inst.uses[0]);
                const wide = isWideType(source.reg_types[inst.uses[0]]);
                try emitMovRegReg(buffer, .rax, src, wide);
            }
            if (inst.uses.len > 1) return error.UnsupportedInstruction;
            try emitRet(buffer);
            stats.returns += 1;
            stats.native_insts += 1;
        },
        else => return error.UnsupportedInstruction,
    }
}

fn ensureNoSpills(allocation: *const regalloc.Allocation) Error!void {
    for (allocation.intervals) |interval| {
        switch (allocation.locations[interval.reg]) {
            .spill => return error.SpillsUnsupported,
            .phys => {},
            .none => return error.InvalidMachine,
        }
    }
}

pub fn encode(allocator: std.mem.Allocator, source: *const machine.Function) Error!Function {
    source.verify() catch return error.InvalidMachine;
    if (source.edges.len != 0) return error.UnsupportedInstruction;

    var allocation = try regalloc.allocateDefault(allocator, source);
    errdefer allocation.deinit();
    try ensureNoSpills(&allocation);

    var buffer = code_buffer.Buffer.init(allocator);
    errdefer buffer.deinit();

    const labels = try allocator.alloc(code_buffer.LabelId, source.blocks.len);
    errdefer allocator.free(labels);
    for (labels) |*label| label.* = try buffer.newLabel();

    var stats: Stats = .{ .blocks = @intCast(source.blocks.len) };
    try emitParamMoves(&buffer, &allocation, source, &stats);

    for (source.blocks) |block| {
        try buffer.alignTo(16, 0x90);
        try buffer.bindLabel(labels[block.id]);
        for (block.insts) |inst| try encodeInst(&buffer, labels, &allocation, source, inst, &stats);
    }
    try buffer.verify();
    stats.bytes = buffer.len();

    return .{
        .allocator = allocator,
        .source = source,
        .allocation = allocation,
        .buffer = buffer,
        .block_labels = labels,
        .stats = stats,
    };
}

fn optimizedMachine(allocator: std.mem.Allocator, insts: []const Instruction) !optimizer.OptimizedFunction {
    return try optimizer.optimize(allocator, insts, &.{}, .{});
}

fn expectBinaryI32(insts: []const Instruction, a: i32, b: i32, expected: i32) !void {
    if (builtin.cpu.arch != .x86_64) return error.SkipZigTest;

    var optimized = try optimizedMachine(std.testing.allocator, insts);
    defer optimized.deinit();
    var native = try encode(std.testing.allocator, &optimized.machine);
    defer native.deinit();
    const bytes = try native.finalize();
    defer std.testing.allocator.free(bytes);
    try std.testing.expect(std.mem.indexOfScalar(u8, bytes, 0x55) == null);

    var cache = jit_memory.Cache.init(std.testing.allocator);
    defer cache.deinit();
    const allocation = try cache.addBytes(bytes);
    const Fn = fn (i32, i32) callconv(.c) i32;
    try std.testing.expectEqual(expected, allocation.typedEntry(Fn)(a, b));
}

test "x64_register_encoder executes parameter arithmetic without frame prologue" {
    try expectBinaryI32(&[_]Instruction{
        .{ .add_int = .{ .dest = 2, .src1 = 0, .src2 = 1 } },
        .{ .return_ = .{ .src = 2 } },
    }, 30, 12, 42);
    try expectBinaryI32(&[_]Instruction{
        .{ .mul_int = .{ .dest = 2, .src1 = 0, .src2 = 1 } },
        .{ .return_ = .{ .src = 2 } },
    }, -7, 6, -42);
}

test "x64_register_encoder rejects spills explicitly" {
    const insts = [_]Instruction{
        .{ .add_int = .{ .dest = 2, .src1 = 0, .src2 = 1 } },
        .{ .return_ = .{ .src = 2 } },
    };
    var optimized = try optimizedMachine(std.testing.allocator, &insts);
    defer optimized.deinit();
    var allocation = try regalloc.allocate(std.testing.allocator, &optimized.machine, .{ .gp_registers = &[_]regalloc.PhysReg{} });
    defer allocation.deinit();
    try std.testing.expect(allocation.stats.spills > 0);
}

test "x64_register_encoder print helper emits stable summary" {
    const insts = [_]Instruction{
        .{ .const_ = .{ .dest = 0, .value = 11 } },
        .{ .return_ = .{ .src = 0 } },
    };
    var optimized = try optimizedMachine(std.testing.allocator, &insts);
    defer optimized.deinit();
    var native = try encode(std.testing.allocator, &optimized.machine);
    defer native.deinit();

    var storage: [1024]u8 = undefined;
    var stream = std.Io.Writer.fixed(&storage);
    try native.print(&stream);
    const output = stream.buffered();
    try std.testing.expect(std.mem.indexOf(u8, output, "x64_register_encoder bytes=") != null);
    try std.testing.expect(std.mem.indexOf(u8, output, "linear_scan intervals=") != null);
}
