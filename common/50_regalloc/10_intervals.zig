//! Register-machine live interval construction.
//!
//! This phase converts virtual-register uses/defs into compact linear
//! intervals. It is intentionally backend-neutral: target-specific register
//! choices happen in `20_linear_scan.zig`.

const std = @import("std");
const machine = @import("machine_bridge");
const typedir = @import("typedir");

pub const Error = error{
    InvalidMachine,
    OutOfMemory,
};

pub const Position = u32;
pub const INVALID_POS: Position = std.math.maxInt(Position);

pub const Interval = struct {
    reg: machine.RegId,
    start: Position,
    end: Position,
    ty: typedir.Type,
    uses: u32 = 0,
    defs: u32 = 0,

    pub inline fn covers(self: Interval, pos: Position) bool {
        return self.start <= pos and pos <= self.end;
    }
};

pub const Stats = struct {
    regs: u32 = 0,
    intervals: u32 = 0,
    uses: u32 = 0,
    defs: u32 = 0,
    edge_moves: u32 = 0,
    positions: u32 = 0,
};

pub const Result = struct {
    allocator: std.mem.Allocator,
    source: *const machine.Function,
    intervals: []Interval,
    stats: Stats,

    pub fn deinit(self: *Result) void {
        self.allocator.free(self.intervals);
        self.* = undefined;
    }

    pub fn verify(self: *const Result) Error!void {
        self.source.verify() catch return error.InvalidMachine;
        for (self.intervals) |interval| {
            if (interval.reg >= self.source.reg_types.len) return error.InvalidMachine;
            if (interval.start > interval.end) return error.InvalidMachine;
            if (interval.uses == 0 and interval.defs == 0) return error.InvalidMachine;
        }
    }

    pub fn print(self: *const Result, writer: anytype) !void {
        try writer.print(
            "intervals regs={d} intervals={d} uses={d} defs={d} edge_moves={d} positions={d}\n",
            .{
                self.stats.regs,
                self.stats.intervals,
                self.stats.uses,
                self.stats.defs,
                self.stats.edge_moves,
                self.stats.positions,
            },
        );
        for (self.intervals) |interval| {
            try writer.print(
                "  r{d}:{s} [{d},{d}] uses={d} defs={d}\n",
                .{ interval.reg, @tagName(interval.ty), interval.start, interval.end, interval.uses, interval.defs },
            );
        }
    }
};

fn touch(
    starts: []Position,
    ends: []Position,
    counts: []u32,
    reg: machine.RegId,
    pos: Position,
) Error!void {
    if (reg >= starts.len) return error.InvalidMachine;
    starts[reg] = @min(starts[reg], pos);
    ends[reg] = @max(ends[reg], pos);
    counts[reg] += 1;
}

fn isParameter(function: *const machine.Function, reg: machine.RegId) bool {
    return reg < function.value_kinds.len and function.value_kinds[reg] == .parameter;
}

pub fn build(allocator: std.mem.Allocator, function: *const machine.Function) Error!Result {
    function.verify() catch return error.InvalidMachine;

    const reg_count = function.reg_types.len;
    const starts = try allocator.alloc(Position, reg_count);
    defer allocator.free(starts);
    const ends = try allocator.alloc(Position, reg_count);
    defer allocator.free(ends);
    const uses = try allocator.alloc(u32, reg_count);
    defer allocator.free(uses);
    const defs = try allocator.alloc(u32, reg_count);
    defer allocator.free(defs);

    @memset(starts, INVALID_POS);
    @memset(ends, 0);
    @memset(uses, 0);
    @memset(defs, 0);

    var stats: Stats = .{ .regs = @intCast(reg_count) };
    for (0..reg_count) |reg| {
        if (isParameter(function, @intCast(reg))) {
            try touch(starts, ends, defs, @intCast(reg), 0);
            stats.defs += 1;
        }
    }

    var pos: Position = 2;
    for (function.blocks) |block| {
        for (block.insts) |inst| {
            for (inst.uses) |reg| {
                try touch(starts, ends, uses, reg, pos);
                stats.uses += 1;
            }
            for (inst.defs) |reg| {
                try touch(starts, ends, defs, reg, pos);
                stats.defs += 1;
            }
            pos += 2;
        }
    }

    for (function.edges) |edge| {
        for (edge.moves) |move| {
            try touch(starts, ends, uses, move.src, pos);
            try touch(starts, ends, defs, move.dst, pos);
            stats.uses += 1;
            stats.defs += 1;
            stats.edge_moves += 1;
            pos += 2;
        }
    }
    stats.positions = pos;

    var list: std.ArrayList(Interval) = .empty;
    errdefer list.deinit(allocator);
    for (0..reg_count) |reg| {
        if (starts[reg] == INVALID_POS) continue;
        try list.append(allocator, .{
            .reg = @intCast(reg),
            .start = starts[reg],
            .end = ends[reg],
            .ty = function.reg_types[reg],
            .uses = uses[reg],
            .defs = defs[reg],
        });
    }

    const intervals = try list.toOwnedSlice(allocator);
    stats.intervals = @intCast(intervals.len);

    return .{
        .allocator = allocator,
        .source = function,
        .intervals = intervals,
        .stats = stats,
    };
}

test "intervals builds intervals for arithmetic function" {
    const optimizer = @import("optimizer");
    const Instruction = @import("instructions").Instruction;
    const insts = [_]Instruction{
        .{ .add_int = .{ .dest = 2, .src1 = 0, .src2 = 1 } },
        .{ .return_ = .{ .src = 2 } },
    };
    var optimized = try optimizer.optimize(std.testing.allocator, &insts, &.{}, .{});
    defer optimized.deinit();

    var live = try build(std.testing.allocator, &optimized.machine);
    defer live.deinit();
    try live.verify();
    try std.testing.expect(live.stats.intervals >= 3);
    try std.testing.expect(live.stats.uses >= 3);
}

test "intervals includes phi edge moves" {
    const optimizer = @import("optimizer");
    const Instruction = @import("instructions").Instruction;
    const insts = [_]Instruction{
        .{ .if_eqz = .{ .src = 0, .offset = 3 } },
        .{ .const_ = .{ .dest = 1, .value = 10 } },
        .{ .goto_ = .{ .offset = 2 } },
        .{ .const_ = .{ .dest = 1, .value = 20 } },
        .{ .return_ = .{ .src = 1 } },
    };
    var optimized = try optimizer.optimize(std.testing.allocator, &insts, &.{}, .{});
    defer optimized.deinit();

    var live = try build(std.testing.allocator, &optimized.machine);
    defer live.deinit();
    try live.verify();
    try std.testing.expect(live.stats.edge_moves >= 2);
}
