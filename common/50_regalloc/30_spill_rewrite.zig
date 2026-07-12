//! Spill rewrite planning.
//!
//! This pass turns spill decisions from linear scan into a concrete stack-slot
//! plan. Backends can use the plan to insert reloads before uses and stores
//! after defs without rediscovering spill metadata.

const std = @import("std");
const linear_scan = @import("linear_scan");
const machine = @import("machine_bridge");
const typedir = @import("typedir");

pub const Error = linear_scan.Error || error{
    BadSpillPlan,
};

pub const SpillSlot = struct {
    reg: machine.RegId,
    slot: u32,
    ty: typedir.Type,
    size: u8,
};

pub const Stats = struct {
    slots: u32 = 0,
    reloads: u32 = 0,
    stores: u32 = 0,
    edge_reloads: u32 = 0,
    edge_stores: u32 = 0,
    stack_bytes: u32 = 0,
};

pub const Plan = struct {
    allocator: std.mem.Allocator,
    source: *const machine.Function,
    location_count: usize,
    slots: []SpillSlot,
    stats: Stats,

    pub fn deinit(self: *Plan) void {
        self.allocator.free(self.slots);
        self.* = undefined;
    }

    pub fn verify(self: *const Plan) Error!void {
        self.source.verify() catch return error.InvalidMachine;
        var seen = try self.allocator.alloc(bool, self.location_count);
        defer self.allocator.free(seen);
        @memset(seen, false);

        for (self.slots) |slot| {
            if (slot.reg >= self.location_count) return error.BadSpillPlan;
            if (seen[slot.reg]) return error.BadSpillPlan;
            seen[slot.reg] = true;
        }
    }

    pub fn print(self: *const Plan, writer: anytype) !void {
        try writer.print(
            "spill_rewrite slots={d} reloads={d} stores={d} edge_reloads={d} edge_stores={d} stack_bytes={d}\n",
            .{
                self.stats.slots,
                self.stats.reloads,
                self.stats.stores,
                self.stats.edge_reloads,
                self.stats.edge_stores,
                self.stats.stack_bytes,
            },
        );
        for (self.slots) |slot| {
            try writer.print("  spill[{d}] r{d}:{s} size={d}\n", .{ slot.slot, slot.reg, @tagName(slot.ty), slot.size });
        }
    }
};

fn typeSize(ty: typedir.Type) u8 {
    return switch (ty) {
        .long, .double => 8,
        else => 4,
    };
}

fn isSpilled(allocation: *const linear_scan.Allocation, reg: machine.RegId) bool {
    if (reg >= allocation.locations.len) return false;
    return switch (allocation.locations[reg]) {
        .spill => true,
        else => false,
    };
}

fn stackBytes(slots: []const SpillSlot) u32 {
    var bytes: u32 = 0;
    for (slots) |slot| bytes += slot.size;
    return (bytes + 15) & ~@as(u32, 15);
}

pub fn build(allocator: std.mem.Allocator, allocation: *const linear_scan.Allocation) Error!Plan {
    try allocation.verify();

    var slots_list: std.ArrayList(SpillSlot) = .empty;
    errdefer slots_list.deinit(allocator);
    for (allocation.intervals) |interval| {
        switch (allocation.locations[interval.reg]) {
            .spill => |slot| try slots_list.append(allocator, .{
                .reg = interval.reg,
                .slot = slot,
                .ty = interval.ty,
                .size = typeSize(interval.ty),
            }),
            else => {},
        }
    }

    var stats: Stats = .{};
    for (allocation.source.blocks) |block| {
        for (block.insts) |inst| {
            for (inst.uses) |reg| {
                if (isSpilled(allocation, reg)) stats.reloads += 1;
            }
            for (inst.defs) |reg| {
                if (isSpilled(allocation, reg)) stats.stores += 1;
            }
        }
    }
    for (allocation.source.edges) |edge| {
        for (edge.moves) |move| {
            if (isSpilled(allocation, move.src)) stats.edge_reloads += 1;
            if (isSpilled(allocation, move.dst)) stats.edge_stores += 1;
        }
    }

    const slots = try slots_list.toOwnedSlice(allocator);
    stats.slots = @intCast(slots.len);
    stats.stack_bytes = stackBytes(slots);

    var plan = Plan{
        .allocator = allocator,
        .source = allocation.source,
        .location_count = allocation.locations.len,
        .slots = slots,
        .stats = stats,
    };
    try plan.verify();
    return plan;
}

test "spill_rewrite builds empty plan for register-only allocation" {
    const optimizer = @import("optimizer");
    const Instruction = @import("instructions").Instruction;
    const insts = [_]Instruction{
        .{ .add_int = .{ .dest = 2, .src1 = 0, .src2 = 1 } },
        .{ .return_ = .{ .src = 2 } },
    };
    var optimized = try optimizer.optimize(std.testing.allocator, &insts, &.{}, .{});
    defer optimized.deinit();
    var allocation = try linear_scan.allocateDefault(std.testing.allocator, &optimized.machine);
    defer allocation.deinit();

    var plan = try build(std.testing.allocator, &allocation);
    defer plan.deinit();
    try plan.verify();
    try std.testing.expectEqual(@as(u32, 0), plan.stats.slots);
}

test "spill_rewrite counts reloads and stores for constrained allocation" {
    const optimizer = @import("optimizer");
    const Instruction = @import("instructions").Instruction;
    const insts = [_]Instruction{
        .{ .add_int = .{ .dest = 2, .src1 = 0, .src2 = 1 } },
        .{ .add_int = .{ .dest = 3, .src1 = 2, .src2 = 1 } },
        .{ .return_ = .{ .src = 3 } },
    };
    var optimized = try optimizer.optimize(std.testing.allocator, &insts, &.{}, .{});
    defer optimized.deinit();

    const one_gp = [_]linear_scan.PhysReg{.r10};
    var allocation = try linear_scan.allocate(std.testing.allocator, &optimized.machine, .{ .gp_registers = &one_gp });
    defer allocation.deinit();

    var plan = try build(std.testing.allocator, &allocation);
    defer plan.deinit();
    try std.testing.expect(plan.stats.slots > 0);
    try std.testing.expect(plan.stats.reloads + plan.stats.stores > 0);
}
