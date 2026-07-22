//! Verified x86-64 128-bit platform-vector backend.
//!
//! Dalvik has no vector bytecodes, so this module is an explicit target-level
//! intrinsic boundary rather than an extension of Dalvik SSA. Programs are
//! straight-line SSA, use only SSE2-guaranteed operations, never safepoint, and
//! cannot contain managed references. Every spilled value occupies one aligned
//! 16-byte slot and all external memory accesses use unaligned-safe moves.

const std = @import("std");
const builtin = @import("builtin");
const code_buffer = @import("code_buffer");
const jit_memory = @import("jit_memory");

pub const Error = code_buffer.Error || error{
    UnsupportedArchitecture,
    InvalidProgram,
    BadAllocation,
    UnsupportedOperation,
};

pub const RegId = u16;

pub const Lane = enum(u8) {
    i8x16,
    i16x8,
    i32x4,
    i64x2,
    f32x4,
    f64x2,

    pub fn lanes(self: Lane) u8 {
        return switch (self) {
            .i8x16 => 16,
            .i16x8 => 8,
            .i32x4, .f32x4 => 4,
            .i64x2, .f64x2 => 2,
        };
    }

    pub fn laneBytes(self: Lane) u8 {
        return 16 / self.lanes();
    }
};

pub const BinaryOp = enum(u8) {
    add,
    sub,
    bit_and,
    bit_or,
    bit_xor,
};

pub const Inst = union(enum) {
    input: struct {
        dst: RegId,
        index: u16,
        lane: Lane,
    },
    binary: struct {
        dst: RegId,
        lhs: RegId,
        rhs: RegId,
        op: BinaryOp,
    },
    output: struct {
        src: RegId,
        index: u16,
    },
};

pub const Program = struct {
    register_count: u16,
    input_count: u16,
    output_count: u16,
    instructions: []const Inst,

    pub fn verify(self: Program, allocator: std.mem.Allocator) Error!void {
        if (self.register_count == 0 or self.output_count == 0 or self.instructions.len == 0) {
            return error.InvalidProgram;
        }
        const defined = try allocator.alloc(bool, self.register_count);
        defer allocator.free(defined);
        @memset(defined, false);
        const lanes = try allocator.alloc(?Lane, self.register_count);
        defer allocator.free(lanes);
        @memset(lanes, null);
        var outputs: u32 = 0;

        for (self.instructions) |inst| switch (inst) {
            .input => |value| {
                if (value.dst >= self.register_count or value.index >= self.input_count or defined[value.dst]) {
                    return error.InvalidProgram;
                }
                defined[value.dst] = true;
                lanes[value.dst] = value.lane;
            },
            .binary => |value| {
                if (value.dst >= self.register_count or value.lhs >= self.register_count or value.rhs >= self.register_count or
                    defined[value.dst] or !defined[value.lhs] or !defined[value.rhs]) return error.InvalidProgram;
                const lhs_lane = lanes[value.lhs] orelse return error.InvalidProgram;
                if (lanes[value.rhs] != lhs_lane) return error.InvalidProgram;
                if (!operationSupports(value.op, lhs_lane)) return error.UnsupportedOperation;
                defined[value.dst] = true;
                lanes[value.dst] = lhs_lane;
            },
            .output => |value| {
                if (value.src >= self.register_count or value.index >= self.output_count or !defined[value.src]) {
                    return error.InvalidProgram;
                }
                outputs += 1;
            },
        };
        if (outputs == 0) return error.InvalidProgram;
    }
};

/// Stable cross-platform entry image. Inputs and outputs each point to arrays
/// of 16 raw bytes; generated code never assumes those elements are aligned.
pub const CallFrame = extern struct {
    inputs: [*]const [16]u8,
    outputs: [*][16]u8,
};

pub const Location = union(enum) {
    none,
    xmm: u3,
    spill: u16,
};

pub const Stats = struct {
    bytes: u32 = 0,
    values: u32 = 0,
    register_values: u32 = 0,
    spilled_values: u32 = 0,
    input_loads: u32 = 0,
    output_stores: u32 = 0,
    vector_ops: u32 = 0,
    vector_moves: u32 = 0,
    spill_loads: u32 = 0,
    spill_stores: u32 = 0,
    frame_bytes: u32 = 0,
};

pub const Function = struct {
    allocator: std.mem.Allocator,
    buffer: code_buffer.Buffer,
    locations: []Location,
    lanes: []Lane,
    intervals: []Interval,
    stats: Stats,

    pub fn deinit(self: *Function) void {
        self.allocator.free(self.lanes);
        self.allocator.free(self.locations);
        self.allocator.free(self.intervals);
        self.buffer.deinit();
        self.* = undefined;
    }

    pub fn verify(self: *Function) Error!void {
        try self.buffer.verify();
        if (self.locations.len != self.lanes.len or self.locations.len != self.intervals.len or
            self.locations.len != self.stats.values) return error.BadAllocation;
        if (!std.mem.isAligned(self.stats.frame_bytes, 16) or
            self.stats.frame_bytes != self.stats.spilled_values * 16 or
            self.stats.register_values + self.stats.spilled_values != self.stats.values) return error.BadAllocation;
        var registers: u32 = 0;
        var spills: u32 = 0;
        const seen_regs = try self.allocator.alloc(bool, self.locations.len);
        defer self.allocator.free(seen_regs);
        @memset(seen_regs, false);
        const seen_spills = try self.allocator.alloc(bool, self.stats.spilled_values);
        defer self.allocator.free(seen_spills);
        @memset(seen_spills, false);
        for (self.intervals, 0..) |interval, index| {
            if (interval.reg >= self.locations.len or seen_regs[interval.reg] or interval.start > interval.end or
                self.lanes[interval.reg] != interval.lane or (index != 0 and intervalLess({}, interval, self.intervals[index - 1])))
            {
                return error.BadAllocation;
            }
            seen_regs[interval.reg] = true;
            const location = self.locations[interval.reg];
            switch (location) {
                .xmm => |reg| {
                    if (reg >= allocatableXmmCount()) return error.BadAllocation;
                    registers += 1;
                },
                .spill => |slot| {
                    if (slot >= self.stats.spilled_values or seen_spills[slot]) return error.BadAllocation;
                    seen_spills[slot] = true;
                    spills += 1;
                },
                .none => return error.BadAllocation,
            }
            for (self.intervals[0..index]) |previous| {
                const previous_location = self.locations[previous.reg];
                if (!sameLocation(location, previous_location)) continue;
                switch (location) {
                    .spill => return error.BadAllocation,
                    .xmm => {},
                    .none => return error.BadAllocation,
                }
                if (interval.start <= previous.end and previous.start <= interval.end) return error.BadAllocation;
            }
        }
        if (registers != self.stats.register_values or spills != self.stats.spilled_values) return error.BadAllocation;
    }

    pub fn finalize(self: *Function) Error![]u8 {
        return self.buffer.finalize();
    }

    pub fn print(self: *const Function, writer: anytype) !void {
        try writer.print(
            "x64_vector bytes={d} values={d} registers={d} spills={d} frame={d} inputs={d} outputs={d} ops={d} moves={d} spill_loads={d} spill_stores={d}\n",
            .{
                self.stats.bytes,
                self.stats.values,
                self.stats.register_values,
                self.stats.spilled_values,
                self.stats.frame_bytes,
                self.stats.input_loads,
                self.stats.output_stores,
                self.stats.vector_ops,
                self.stats.vector_moves,
                self.stats.spill_loads,
                self.stats.spill_stores,
            },
        );
    }
};

const Interval = struct {
    reg: RegId,
    start: u32,
    end: u32,
    lane: Lane,
};

const Active = struct {
    reg: RegId,
    end: u32,
    xmm: u3,
};

fn operationSupports(op: BinaryOp, lane: Lane) bool {
    return switch (op) {
        .add, .sub => switch (lane) {
            .i8x16, .i16x8, .i32x4, .i64x2, .f32x4, .f64x2 => true,
        },
        .bit_and, .bit_or, .bit_xor => true,
    };
}

fn allocatableXmmCount() u3 {
    return if (builtin.os.tag == .windows) 4 else 6;
}

fn scratchPrimary() u3 {
    return if (builtin.os.tag == .windows) 4 else 6;
}

fn scratchSecondary() u3 {
    return if (builtin.os.tag == .windows) 5 else 7;
}

fn intervalLess(_: void, a: Interval, b: Interval) bool {
    if (a.start != b.start) return a.start < b.start;
    return a.reg < b.reg;
}

fn buildIntervals(allocator: std.mem.Allocator, program: Program) Error!struct { intervals: []Interval, lanes: []Lane } {
    const starts = try allocator.alloc(?u32, program.register_count);
    defer allocator.free(starts);
    @memset(starts, null);
    const ends = try allocator.alloc(u32, program.register_count);
    defer allocator.free(ends);
    @memset(ends, 0);
    const lanes = try allocator.alloc(Lane, program.register_count);
    errdefer allocator.free(lanes);

    for (program.instructions, 0..) |inst, index| {
        const position: u32 = @intCast(index);
        switch (inst) {
            .input => |value| {
                starts[value.dst] = position;
                ends[value.dst] = position;
                lanes[value.dst] = value.lane;
            },
            .binary => |value| {
                starts[value.dst] = position;
                ends[value.dst] = position;
                lanes[value.dst] = lanes[value.lhs];
                ends[value.lhs] = @max(ends[value.lhs], position);
                ends[value.rhs] = @max(ends[value.rhs], position);
            },
            .output => |value| ends[value.src] = @max(ends[value.src], position),
        }
    }

    const intervals = try allocator.alloc(Interval, program.register_count);
    errdefer allocator.free(intervals);
    for (intervals, 0..) |*interval, reg| interval.* = .{
        .reg = @intCast(reg),
        .start = starts[reg] orelse return error.InvalidProgram,
        .end = ends[reg],
        .lane = lanes[reg],
    };
    std.mem.sort(Interval, intervals, {}, intervalLess);
    return .{ .intervals = intervals, .lanes = lanes };
}

fn expireOld(active: *std.ArrayList(Active), current_start: u32) void {
    var index: usize = 0;
    while (index < active.items.len) {
        if (active.items[index].end >= current_start) {
            index += 1;
        } else {
            _ = active.swapRemove(index);
        }
    }
}

fn allocateLocations(
    allocator: std.mem.Allocator,
    intervals: []const Interval,
    register_count: usize,
) Error!struct { locations: []Location, spill_count: u16 } {
    const locations = try allocator.alloc(Location, register_count);
    errdefer allocator.free(locations);
    @memset(locations, .none);
    var active: std.ArrayList(Active) = .empty;
    defer active.deinit(allocator);

    for (intervals) |interval| {
        expireOld(&active, interval.start);
        var used = [_]bool{false} ** 8;
        for (active.items) |item| used[item.xmm] = true;
        var free: ?u3 = null;
        for (0..allocatableXmmCount()) |candidate| {
            if (!used[candidate]) {
                free = @intCast(candidate);
                break;
            }
        }
        if (free) |xmm| {
            locations[interval.reg] = .{ .xmm = xmm };
            try active.append(allocator, .{ .reg = interval.reg, .end = interval.end, .xmm = xmm });
            continue;
        }

        var victim_index: usize = 0;
        for (active.items, 0..) |item, index| {
            if (item.end > active.items[victim_index].end) victim_index = index;
        }
        if (active.items[victim_index].end > interval.end) {
            const victim = active.items[victim_index];
            locations[victim.reg] = .{ .spill = 0 };
            locations[interval.reg] = .{ .xmm = victim.xmm };
            active.items[victim_index] = .{ .reg = interval.reg, .end = interval.end, .xmm = victim.xmm };
        } else {
            locations[interval.reg] = .{ .spill = 0 };
        }
    }

    var spill_count: u16 = 0;
    for (locations) |*location| switch (location.*) {
        .spill => {
            location.* = .{ .spill = spill_count };
            spill_count = std.math.add(u16, spill_count, 1) catch return error.BadAllocation;
        },
        else => {},
    };
    return .{ .locations = locations, .spill_count = spill_count };
}

fn emitRex(buffer: *code_buffer.Buffer, reg: u4, rm: u4) Error!void {
    var rex: u8 = 0x40;
    if ((reg & 8) != 0) rex |= 0x04;
    if ((rm & 8) != 0) rex |= 0x01;
    if (rex != 0x40) try buffer.emitU8(rex);
}

fn emitModRm(buffer: *code_buffer.Buffer, mode: u2, reg: u4, rm: u4) Error!void {
    try buffer.emitU8((@as(u8, mode) << 6) | ((@as(u8, reg) & 7) << 3) | (@as(u8, rm) & 7));
}

fn emitMemory(buffer: *code_buffer.Buffer, reg: u4, base: u4, displacement: u32) Error!void {
    try emitModRm(buffer, 2, reg, base);
    if ((base & 7) == 4) try buffer.emitU8(0x24);
    try buffer.emitU32(displacement);
}

fn emitMovRegReg(buffer: *code_buffer.Buffer, dst: u4, src: u4) Error!void {
    try buffer.emitU8(0x48 | @as(u8, @intFromBool((src & 8) != 0)) * 4 | @as(u8, @intFromBool((dst & 8) != 0)));
    try buffer.emitU8(0x89);
    try emitModRm(buffer, 3, src, dst);
}

fn emitMovRegMem(buffer: *code_buffer.Buffer, dst: u4, base: u4, displacement: u32) Error!void {
    try buffer.emitU8(0x48 | @as(u8, @intFromBool((dst & 8) != 0)) * 4 | @as(u8, @intFromBool((base & 8) != 0)));
    try buffer.emitU8(0x8b);
    try emitMemory(buffer, dst, base, displacement);
}

fn emitAdjustStack(buffer: *code_buffer.Buffer, subtract: bool, amount: u32) Error!void {
    if (amount == 0) return;
    if (!std.mem.isAligned(amount, 16)) return error.BadAllocation;
    if (amount <= std.math.maxInt(i8)) {
        try buffer.emitBytes(&.{ 0x48, 0x83, if (subtract) 0xec else 0xc4, @intCast(amount) });
    } else {
        try buffer.emitBytes(&.{ 0x48, 0x81, if (subtract) 0xec else 0xc4 });
        try buffer.emitU32(amount);
    }
}

fn emitVectorMemory(buffer: *code_buffer.Buffer, xmm: u3, base: u4, displacement: u32, load: bool) Error!void {
    try buffer.emitU8(0xf3);
    try emitRex(buffer, xmm, base);
    try buffer.emitBytes(&.{ 0x0f, if (load) 0x6f else 0x7f });
    try emitMemory(buffer, xmm, base, displacement);
}

fn emitVectorMove(buffer: *code_buffer.Buffer, dst: u3, src: u3) Error!bool {
    if (dst == src) return false;
    try buffer.emitBytes(&.{ 0x0f, 0x28 });
    try emitModRm(buffer, 3, dst, src);
    return true;
}

fn emitVectorBinary(buffer: *code_buffer.Buffer, op: BinaryOp, lane: Lane, dst: u3, rhs: u3) Error!void {
    const encoding: struct { prefix_66: bool, opcode: u8 } = switch (op) {
        .bit_and => .{ .prefix_66 = true, .opcode = 0xdb },
        .bit_or => .{ .prefix_66 = true, .opcode = 0xeb },
        .bit_xor => .{ .prefix_66 = true, .opcode = 0xef },
        .add => switch (lane) {
            .i8x16 => .{ .prefix_66 = true, .opcode = 0xfc },
            .i16x8 => .{ .prefix_66 = true, .opcode = 0xfd },
            .i32x4 => .{ .prefix_66 = true, .opcode = 0xfe },
            .i64x2 => .{ .prefix_66 = true, .opcode = 0xd4 },
            .f32x4 => .{ .prefix_66 = false, .opcode = 0x58 },
            .f64x2 => .{ .prefix_66 = true, .opcode = 0x58 },
        },
        .sub => switch (lane) {
            .i8x16 => .{ .prefix_66 = true, .opcode = 0xf8 },
            .i16x8 => .{ .prefix_66 = true, .opcode = 0xf9 },
            .i32x4 => .{ .prefix_66 = true, .opcode = 0xfa },
            .i64x2 => .{ .prefix_66 = true, .opcode = 0xfb },
            .f32x4 => .{ .prefix_66 = false, .opcode = 0x5c },
            .f64x2 => .{ .prefix_66 = true, .opcode = 0x5c },
        },
    };
    if (encoding.prefix_66) try buffer.emitU8(0x66);
    try buffer.emitBytes(&.{ 0x0f, encoding.opcode });
    try emitModRm(buffer, 3, dst, rhs);
}

fn spillOffset(slot: u16) u32 {
    return @as(u32, slot) * 16;
}

fn sameLocation(a: Location, b: Location) bool {
    return switch (a) {
        .none => b == .none,
        .xmm => |left| switch (b) {
            .xmm => |right| left == right,
            else => false,
        },
        .spill => |left| switch (b) {
            .spill => |right| left == right,
            else => false,
        },
    };
}

fn loadLocation(buffer: *code_buffer.Buffer, location: Location, scratch: u3, stats: *Stats) Error!u3 {
    return switch (location) {
        .xmm => |xmm| xmm,
        .spill => |slot| blk: {
            try emitVectorMemory(buffer, scratch, 4, spillOffset(slot), true);
            stats.spill_loads += 1;
            break :blk scratch;
        },
        .none => error.BadAllocation,
    };
}

fn storeLocation(buffer: *code_buffer.Buffer, location: Location, src: u3, stats: *Stats) Error!void {
    switch (location) {
        .xmm => |xmm| stats.vector_moves += @intFromBool(try emitVectorMove(buffer, xmm, src)),
        .spill => |slot| {
            try emitVectorMemory(buffer, src, 4, spillOffset(slot), false);
            stats.spill_stores += 1;
        },
        .none => return error.BadAllocation,
    }
}

pub fn encode(allocator: std.mem.Allocator, program: Program) Error!Function {
    if (builtin.cpu.arch != .x86_64) return error.UnsupportedArchitecture;
    try program.verify(allocator);
    const built = try buildIntervals(allocator, program);
    var intervals_owned = true;
    errdefer if (intervals_owned) allocator.free(built.intervals);
    var lanes_owned = true;
    errdefer if (lanes_owned) allocator.free(built.lanes);
    const allocated = try allocateLocations(allocator, built.intervals, program.register_count);
    var locations_owned = true;
    errdefer if (locations_owned) allocator.free(allocated.locations);

    var buffer = code_buffer.Buffer.init(allocator);
    var buffer_owned = true;
    errdefer if (buffer_owned) buffer.deinit();
    const frame_bytes = @as(u32, allocated.spill_count) * 16;
    try emitMovRegReg(&buffer, 10, if (builtin.os.tag == .windows) 1 else 7);
    try emitAdjustStack(&buffer, true, frame_bytes);
    var stats: Stats = .{
        .values = program.register_count,
        .spilled_values = allocated.spill_count,
        .register_values = program.register_count - allocated.spill_count,
        .frame_bytes = frame_bytes,
    };

    for (program.instructions) |inst| switch (inst) {
        .input => |value| {
            const dst_location = allocated.locations[value.dst];
            const dst = switch (dst_location) {
                .xmm => |xmm| xmm,
                .spill => scratchPrimary(),
                .none => return error.BadAllocation,
            };
            try emitMovRegMem(&buffer, 11, 10, @offsetOf(CallFrame, "inputs"));
            try emitVectorMemory(&buffer, dst, 11, @as(u32, value.index) * 16, true);
            stats.input_loads += 1;
            if (dst_location == .spill) try storeLocation(&buffer, dst_location, dst, &stats);
        },
        .binary => |value| {
            const dst_location = allocated.locations[value.dst];
            const dst = switch (dst_location) {
                .xmm => |xmm| xmm,
                .spill => scratchPrimary(),
                .none => return error.BadAllocation,
            };
            const lhs_location = allocated.locations[value.lhs];
            const rhs_location = allocated.locations[value.rhs];
            var saved_rhs: ?u3 = null;
            switch (rhs_location) {
                .xmm => |rhs| if (rhs == dst and !sameLocation(lhs_location, rhs_location)) {
                    stats.vector_moves += @intFromBool(try emitVectorMove(&buffer, scratchSecondary(), rhs));
                    saved_rhs = scratchSecondary();
                },
                else => {},
            }
            const lhs = try loadLocation(&buffer, lhs_location, dst, &stats);
            stats.vector_moves += @intFromBool(try emitVectorMove(&buffer, dst, lhs));
            const rhs = saved_rhs orelse try loadLocation(&buffer, rhs_location, scratchSecondary(), &stats);
            try emitVectorBinary(&buffer, value.op, built.lanes[value.dst], dst, rhs);
            stats.vector_ops += 1;
            try storeLocation(&buffer, dst_location, dst, &stats);
        },
        .output => |value| {
            const src = try loadLocation(&buffer, allocated.locations[value.src], scratchPrimary(), &stats);
            try emitMovRegMem(&buffer, 11, 10, @offsetOf(CallFrame, "outputs"));
            try emitVectorMemory(&buffer, src, 11, @as(u32, value.index) * 16, false);
            stats.output_stores += 1;
        },
    };
    try emitAdjustStack(&buffer, false, frame_bytes);
    try buffer.emitU8(0xc3);
    try buffer.verify();
    stats.bytes = buffer.len();
    intervals_owned = false;
    lanes_owned = false;
    locations_owned = false;
    buffer_owned = false;
    var function = Function{
        .allocator = allocator,
        .buffer = buffer,
        .locations = allocated.locations,
        .lanes = built.lanes,
        .intervals = built.intervals,
        .stats = stats,
    };
    errdefer function.deinit();
    try function.verify();
    return function;
}

fn writeVector(comptime T: type, destination: *[16]u8, value: T) void {
    comptime std.debug.assert(@sizeOf(T) == 16);
    var copy = value;
    @memcpy(destination, std.mem.asBytes(&copy));
}

fn readVector(comptime T: type, source: *const [16]u8) T {
    comptime std.debug.assert(@sizeOf(T) == 16);
    var result: T = undefined;
    @memcpy(std.mem.asBytes(&result), source);
    return result;
}

test "x64 vector backend covers the SSE2 packed operation matrix" {
    if (builtin.cpu.arch != .x86_64) return error.SkipZigTest;
    const Case = struct {
        op: BinaryOp,
        lane: Lane,
        bytes: []const u8,
    };
    const cases = [_]Case{
        .{ .op = .add, .lane = .i8x16, .bytes = &.{ 0x66, 0x0f, 0xfc, 0xc1 } },
        .{ .op = .add, .lane = .i16x8, .bytes = &.{ 0x66, 0x0f, 0xfd, 0xc1 } },
        .{ .op = .add, .lane = .i32x4, .bytes = &.{ 0x66, 0x0f, 0xfe, 0xc1 } },
        .{ .op = .add, .lane = .i64x2, .bytes = &.{ 0x66, 0x0f, 0xd4, 0xc1 } },
        .{ .op = .add, .lane = .f32x4, .bytes = &.{ 0x0f, 0x58, 0xc1 } },
        .{ .op = .add, .lane = .f64x2, .bytes = &.{ 0x66, 0x0f, 0x58, 0xc1 } },
        .{ .op = .sub, .lane = .i8x16, .bytes = &.{ 0x66, 0x0f, 0xf8, 0xc1 } },
        .{ .op = .sub, .lane = .i16x8, .bytes = &.{ 0x66, 0x0f, 0xf9, 0xc1 } },
        .{ .op = .sub, .lane = .i32x4, .bytes = &.{ 0x66, 0x0f, 0xfa, 0xc1 } },
        .{ .op = .sub, .lane = .i64x2, .bytes = &.{ 0x66, 0x0f, 0xfb, 0xc1 } },
        .{ .op = .sub, .lane = .f32x4, .bytes = &.{ 0x0f, 0x5c, 0xc1 } },
        .{ .op = .sub, .lane = .f64x2, .bytes = &.{ 0x66, 0x0f, 0x5c, 0xc1 } },
        .{ .op = .bit_and, .lane = .i64x2, .bytes = &.{ 0x66, 0x0f, 0xdb, 0xc1 } },
        .{ .op = .bit_or, .lane = .i64x2, .bytes = &.{ 0x66, 0x0f, 0xeb, 0xc1 } },
        .{ .op = .bit_xor, .lane = .i64x2, .bytes = &.{ 0x66, 0x0f, 0xef, 0xc1 } },
    };
    for (cases) |case| {
        var buffer = code_buffer.Buffer.init(std.testing.allocator);
        defer buffer.deinit();
        try emitVectorBinary(&buffer, case.op, case.lane, 0, 1);
        try std.testing.expectEqualSlices(u8, case.bytes, buffer.slice());
    }
}

test "x64 vector backend executes integer and floating packed operations" {
    if (builtin.cpu.arch != .x86_64) return error.SkipZigTest;
    const insts = [_]Inst{
        .{ .input = .{ .dst = 0, .index = 0, .lane = .i32x4 } },
        .{ .input = .{ .dst = 1, .index = 1, .lane = .i32x4 } },
        .{ .binary = .{ .dst = 2, .lhs = 0, .rhs = 1, .op = .add } },
        .{ .output = .{ .src = 2, .index = 0 } },
    };
    var function = try encode(std.testing.allocator, .{
        .register_count = 3,
        .input_count = 2,
        .output_count = 1,
        .instructions = &insts,
    });
    defer function.deinit();
    try function.verify();
    try std.testing.expectEqual(@as(u32, 0), function.stats.spilled_values);
    const saved_location = function.locations[1];
    function.locations[1] = function.locations[0];
    try std.testing.expectError(error.BadAllocation, function.verify());
    function.locations[1] = saved_location;
    try function.verify();
    const bytes = try function.finalize();
    defer std.testing.allocator.free(bytes);
    var cache = jit_memory.Cache.init(std.testing.allocator);
    defer cache.deinit();
    const code = try cache.addBytes(bytes);

    const I32x4 = @Vector(4, i32);
    var input_backing: [33]u8 = @splat(0);
    var output_backing: [17]u8 = @splat(0);
    const inputs: [*][16]u8 = @ptrCast(&input_backing[1]);
    const outputs: [*][16]u8 = @ptrCast(&output_backing[1]);
    writeVector(I32x4, &inputs[0], std.simd.iota(i32, 4));
    writeVector(I32x4, &inputs[1], @as(I32x4, @splat(10)));
    var frame = CallFrame{ .inputs = inputs, .outputs = outputs };
    const Fn = fn (*const CallFrame) callconv(.c) void;
    code.typedEntry(Fn)(&frame);
    try std.testing.expectEqual(@as(I32x4, .{ 10, 11, 12, 13 }), readVector(I32x4, &outputs[0]));
    writeVector(I32x4, &inputs[0], .{ std.math.maxInt(i32), std.math.minInt(i32), -1, 0 });
    writeVector(I32x4, &inputs[1], .{ 1, -1, 2, 0 });
    code.typedEntry(Fn)(&frame);
    try std.testing.expectEqual(
        @as(I32x4, .{ std.math.minInt(i32), std.math.maxInt(i32), 1, 0 }),
        readVector(I32x4, &outputs[0]),
    );

    const float_insts = [_]Inst{
        .{ .input = .{ .dst = 0, .index = 0, .lane = .f32x4 } },
        .{ .input = .{ .dst = 1, .index = 1, .lane = .f32x4 } },
        .{ .binary = .{ .dst = 2, .lhs = 0, .rhs = 1, .op = .sub } },
        .{ .output = .{ .src = 2, .index = 0 } },
    };
    var float_function = try encode(std.testing.allocator, .{
        .register_count = 3,
        .input_count = 2,
        .output_count = 1,
        .instructions = &float_insts,
    });
    defer float_function.deinit();
    const float_bytes = try float_function.finalize();
    defer std.testing.allocator.free(float_bytes);
    const float_code = try cache.addBytes(float_bytes);
    const F32x4 = @Vector(4, f32);
    writeVector(F32x4, &inputs[0], .{ 9.5, -2.0, 0.0, 100.0 });
    writeVector(F32x4, &inputs[1], .{ 0.5, 3.0, -0.0, 0.25 });
    @memset(&outputs[0], 0);
    float_code.typedEntry(Fn)(&frame);
    try std.testing.expectEqual(@as(F32x4, .{ 9.0, -5.0, 0.0, 99.75 }), readVector(F32x4, &outputs[0]));
}

test "x64 vector backend spills full lanes under deterministic pressure" {
    if (builtin.cpu.arch != .x86_64) return error.SkipZigTest;
    const insts = [_]Inst{
        .{ .input = .{ .dst = 0, .index = 0, .lane = .i32x4 } },
        .{ .input = .{ .dst = 1, .index = 1, .lane = .i32x4 } },
        .{ .input = .{ .dst = 2, .index = 2, .lane = .i32x4 } },
        .{ .input = .{ .dst = 3, .index = 3, .lane = .i32x4 } },
        .{ .input = .{ .dst = 4, .index = 4, .lane = .i32x4 } },
        .{ .input = .{ .dst = 5, .index = 5, .lane = .i32x4 } },
        .{ .input = .{ .dst = 6, .index = 6, .lane = .i32x4 } },
        .{ .input = .{ .dst = 7, .index = 7, .lane = .i32x4 } },
        .{ .binary = .{ .dst = 8, .lhs = 0, .rhs = 1, .op = .add } },
        .{ .binary = .{ .dst = 9, .lhs = 2, .rhs = 3, .op = .add } },
        .{ .binary = .{ .dst = 10, .lhs = 4, .rhs = 5, .op = .add } },
        .{ .binary = .{ .dst = 11, .lhs = 6, .rhs = 7, .op = .add } },
        .{ .binary = .{ .dst = 12, .lhs = 8, .rhs = 9, .op = .add } },
        .{ .binary = .{ .dst = 13, .lhs = 10, .rhs = 11, .op = .add } },
        .{ .binary = .{ .dst = 14, .lhs = 12, .rhs = 13, .op = .add } },
        .{ .output = .{ .src = 14, .index = 0 } },
    };
    const program = Program{ .register_count = 15, .input_count = 8, .output_count = 1, .instructions = &insts };
    var function = try encode(std.testing.allocator, program);
    defer function.deinit();
    try function.verify();
    try std.testing.expect(function.stats.spilled_values > 0);
    try std.testing.expect(function.stats.spill_loads > 0);
    try std.testing.expect(function.stats.spill_stores > 0);
    try std.testing.expect(std.mem.isAligned(function.stats.frame_bytes, 16));

    const bytes = try function.finalize();
    defer std.testing.allocator.free(bytes);
    var cache = jit_memory.Cache.init(std.testing.allocator);
    defer cache.deinit();
    const code = try cache.addBytes(bytes);
    const I32x4 = @Vector(4, i32);
    var inputs: [8][16]u8 = @splat(@splat(0));
    var outputs: [1][16]u8 = @splat(@splat(0));
    for (&inputs, 0..) |*input, index| writeVector(I32x4, input, @as(I32x4, @splat(@intCast(index + 1))));
    var frame = CallFrame{ .inputs = &inputs, .outputs = &outputs };
    const Fn = fn (*const CallFrame) callconv(.c) void;
    code.typedEntry(Fn)(&frame);
    try std.testing.expectEqual(@as(I32x4, @splat(36)), readVector(I32x4, &outputs[0]));
}

fn encodingFailureProbe(allocator: std.mem.Allocator, program: Program) !void {
    var function = try encode(allocator, program);
    defer function.deinit();
    const bytes = try function.finalize();
    defer allocator.free(bytes);
}

test "x64 vector backend rejects malformed programs and exhausts allocation failures" {
    if (builtin.cpu.arch != .x86_64) return error.SkipZigTest;
    const undefined_use = [_]Inst{.{ .output = .{ .src = 0, .index = 0 } }};
    try std.testing.expectError(error.InvalidProgram, encode(std.testing.allocator, .{
        .register_count = 1,
        .input_count = 0,
        .output_count = 1,
        .instructions = &undefined_use,
    }));
    const valid = [_]Inst{
        .{ .input = .{ .dst = 0, .index = 0, .lane = .i64x2 } },
        .{ .input = .{ .dst = 1, .index = 1, .lane = .i64x2 } },
        .{ .binary = .{ .dst = 2, .lhs = 0, .rhs = 1, .op = .bit_xor } },
        .{ .output = .{ .src = 2, .index = 0 } },
    };
    const program = Program{ .register_count = 3, .input_count = 2, .output_count = 1, .instructions = &valid };
    try std.testing.checkAllAllocationFailures(std.testing.allocator, encodingFailureProbe, .{program});
}
