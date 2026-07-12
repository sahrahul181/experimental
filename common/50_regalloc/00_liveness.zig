//! Register liveness public facade.
//!
//! The concrete live-interval builder lives in `10_intervals.zig`. This module
//! keeps the short `liveness` import stable for older callers.

pub const intervals = @import("intervals");

pub const Error = intervals.Error;
pub const Position = intervals.Position;
pub const INVALID_POS = intervals.INVALID_POS;
pub const Interval = intervals.Interval;
pub const Stats = intervals.Stats;
pub const Result = intervals.Result;
pub const build = intervals.build;

test "liveness facade exposes interval builder" {
    const optimizer = @import("optimizer");
    const Instruction = @import("instructions").Instruction;
    const insts = [_]Instruction{
        .{ .add_int = .{ .dest = 2, .src1 = 0, .src2 = 1 } },
        .{ .return_ = .{ .src = 2 } },
    };
    var optimized = try optimizer.optimize(@import("std").testing.allocator, &insts, &.{}, .{});
    defer optimized.deinit();

    var live = try build(@import("std").testing.allocator, &optimized.machine);
    defer live.deinit();
    try live.verify();
}
