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
    /// Byte offset from the managed frame's captured rsp.
    byte_offset: u32,
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
        const seen_slots = try self.allocator.alloc(bool, self.slots.len);
        defer self.allocator.free(seen_slots);
        @memset(seen_slots, false);

        var cursor: u32 = 0;
        for (self.slots, 0..) |slot, index| {
            if (slot.reg >= self.location_count) return error.BadSpillPlan;
            if (seen[slot.reg]) return error.BadSpillPlan;
            seen[slot.reg] = true;
            if (slot.slot >= seen_slots.len or seen_slots[slot.slot] or slot.slot != index) return error.BadSpillPlan;
            seen_slots[slot.slot] = true;
            if (slot.size != 4 and slot.size != 8) return error.BadSpillPlan;
            cursor = alignForward(cursor, slot.size);
            if (slot.byte_offset != cursor) return error.BadSpillPlan;
            cursor = std.math.add(u32, cursor, slot.size) catch return error.BadSpillPlan;
        }
        if (self.stats.stack_bytes != alignForward(cursor, 16)) return error.BadSpillPlan;
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
            try writer.print("  spill[{d}] r{d}:{s} size={d} offset={d}\n", .{ slot.slot, slot.reg, @tagName(slot.ty), slot.size, slot.byte_offset });
        }
    }
};

fn typeSize(ty: typedir.Type, gc_root: bool) u8 {
    // Canonical Handles are the runtime's 64-bit GC identity.
    if (gc_root) return @sizeOf(u64);
    return switch (ty) {
        .long, .double, .object => 8,
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

fn alignForward(value: u32, alignment: u8) u32 {
    const mask = @as(u32, alignment) - 1;
    return (value + mask) & ~mask;
}

fn slotLess(_: void, a: SpillSlot, b: SpillSlot) bool {
    return a.slot < b.slot;
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
                .size = typeSize(interval.ty, allocation.source.isGcRoot(interval.reg)),
                .byte_offset = 0,
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
            if (inst.address) |reg| {
                if (isSpilled(allocation, reg)) stats.reloads += 1;
            }
            if (inst.state_handle) |reg| {
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

    std.mem.sort(SpillSlot, slots_list.items, {}, slotLess);
    var stack_cursor: u32 = 0;
    for (slots_list.items) |*slot| {
        stack_cursor = alignForward(stack_cursor, slot.size);
        slot.byte_offset = stack_cursor;
        stack_cursor = std.math.add(u32, stack_cursor, slot.size) catch return error.BadSpillPlan;
    }
    const slots = try slots_list.toOwnedSlice(allocator);
    stats.slots = @intCast(slots.len);
    stats.stack_bytes = alignForward(stack_cursor, 16);

    var plan = Plan{
        .allocator = allocator,
        .source = allocation.source,
        .location_count = allocation.locations.len,
        .slots = slots,
        .stats = stats,
    };
    errdefer plan.deinit();
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

fn allocationFailureProbe(allocator: std.mem.Allocator, allocation: *const linear_scan.Allocation) !void {
    var plan = try build(allocator, allocation);
    defer plan.deinit();
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
    try std.testing.expect(std.mem.isAligned(plan.stats.stack_bytes, 16));
    var previous_end: u32 = 0;
    for (plan.slots, 0..) |slot, index| {
        try std.testing.expectEqual(@as(u32, @intCast(index)), slot.slot);
        try std.testing.expect(std.mem.isAligned(slot.byte_offset, slot.size));
        try std.testing.expect(slot.byte_offset >= previous_end);
        previous_end = slot.byte_offset + slot.size;
    }
    try std.testing.checkAllAllocationFailures(std.testing.allocator, allocationFailureProbe, .{&allocation});
}
