//! Register allocation orchestrator.
//!
//! Public entry point for the register-allocation phase. It runs linear scan,
//! builds spill rewrite metadata, and exposes a stable API to native backends.

const std = @import("std");
const linear_scan = @import("linear_scan");
const spill_rewrite = @import("spill_rewrite");
const machine = @import("machine_bridge");

pub const Error = linear_scan.Error || spill_rewrite.Error;
pub const RegClass = linear_scan.RegClass;
pub const PhysReg = linear_scan.PhysReg;
pub const Location = linear_scan.Location;
pub const Options = linear_scan.Options;
pub const Stats = linear_scan.Stats;
pub const Allocation = linear_scan.Allocation;
pub const SpillPlan = spill_rewrite.Plan;
pub const SpillSlot = spill_rewrite.SpillSlot;

pub const Result = struct {
    allocation: Allocation,
    spill_plan: SpillPlan,

    pub fn deinit(self: *Result) void {
        self.spill_plan.deinit();
        self.allocation.deinit();
        self.* = undefined;
    }

    pub fn verify(self: *const Result) Error!void {
        try self.allocation.verify();
        try self.spill_plan.verify();
    }

    pub fn print(self: *const Result, writer: anytype) !void {
        try writer.print("regalloc_pipeline\n", .{});
        try self.allocation.print(writer);
        try self.spill_plan.print(writer);
    }
};

pub fn allocate(allocator: std.mem.Allocator, function: *const machine.Function, options: Options) Error!Allocation {
    return linear_scan.allocate(allocator, function, options);
}

pub fn allocateDefault(allocator: std.mem.Allocator, function: *const machine.Function) Error!Allocation {
    return linear_scan.allocateDefault(allocator, function);
}

pub fn build(allocator: std.mem.Allocator, function: *const machine.Function, options: Options) Error!Result {
    var allocation = try linear_scan.allocate(allocator, function, options);
    errdefer allocation.deinit();
    var spill_plan = try spill_rewrite.build(allocator, &allocation);
    errdefer spill_plan.deinit();

    var result = Result{
        .allocation = allocation,
        .spill_plan = spill_plan,
    };
    try result.verify();
    return result;
}

pub fn buildDefault(allocator: std.mem.Allocator, function: *const machine.Function) Error!Result {
    return build(allocator, function, .{});
}

test "regalloc orchestrator builds allocation and spill plan" {
    const optimizer = @import("optimizer");
    const Instruction = @import("instructions").Instruction;
    const insts = [_]Instruction{
        .{ .add_int = .{ .dest = 2, .src1 = 0, .src2 = 1 } },
        .{ .return_ = .{ .src = 2 } },
    };
    var optimized = try optimizer.optimize(std.testing.allocator, &insts, &.{}, .{});
    defer optimized.deinit();

    var result = try buildDefault(std.testing.allocator, &optimized.machine);
    defer result.deinit();
    try result.verify();
    try std.testing.expect(result.allocation.stats.intervals >= 3);
}

test "regalloc orchestrator print helper emits nested summaries" {
    const optimizer = @import("optimizer");
    const Instruction = @import("instructions").Instruction;
    const insts = [_]Instruction{
        .{ .const_ = .{ .dest = 0, .value = 1 } },
        .{ .return_ = .{ .src = 0 } },
    };
    var optimized = try optimizer.optimize(std.testing.allocator, &insts, &.{}, .{});
    defer optimized.deinit();

    var result = try buildDefault(std.testing.allocator, &optimized.machine);
    defer result.deinit();

    var storage: [1024]u8 = undefined;
    var stream = std.Io.Writer.fixed(&storage);
    try result.print(&stream);
    const output = stream.buffered();
    try std.testing.expect(std.mem.indexOf(u8, output, "regalloc_pipeline") != null);
    try std.testing.expect(std.mem.indexOf(u8, output, "linear_scan intervals=") != null);
    try std.testing.expect(std.mem.indexOf(u8, output, "spill_rewrite slots=") != null);
}
