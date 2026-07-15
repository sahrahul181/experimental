//! Post-register-allocation safety proof for derived heap addresses.
//!
//! The current policy is intentionally strict: derived pointers are
//! rematerializable compiler temporaries and must remain in physical registers.
//! They cannot occupy spill slots, GC-root locations, phi moves, or the same
//! physical register as their simultaneously-live canonical handle.

const std = @import("std");
const machine = @import("machine_bridge");
const machine_verify = @import("derived_verify");
const linear_scan = @import("linear_scan");
const spill_rewrite = @import("spill_rewrite");

pub const Error = machine_verify.Error || error{
    BadSpillLocation,
    CanonicalHandleUnavailable,
    DerivedPointerSpilled,
    InvalidAllocation,
    LocationConflict,
};

pub const Stats = struct {
    derived_intervals: u32 = 0,
    physical_derived: u32 = 0,
    canonical_handle_uses: u32 = 0,
    root_spills_checked: u32 = 0,
    overlap_pairs_checked: u32 = 0,
};

fn intervalsOverlap(a: @import("intervals").Interval, b: @import("intervals").Interval) bool {
    return a.start <= b.end and b.start <= a.end;
}

fn spillSlotFor(plan: *const spill_rewrite.Plan, reg: machine.RegId) ?spill_rewrite.SpillSlot {
    for (plan.slots) |slot| if (slot.reg == reg) return slot;
    return null;
}

fn verifySpills(
    allocation: *const linear_scan.Allocation,
    plan: ?*const spill_rewrite.Plan,
    stats: *Stats,
) Error!void {
    if (plan) |spill_plan| {
        spill_plan.verify() catch |err| switch (err) {
            error.OutOfMemory => return error.OutOfMemory,
            else => return error.InvalidAllocation,
        };
        if (spill_plan.source != allocation.source or spill_plan.location_count != allocation.locations.len) return error.InvalidAllocation;
    }

    for (allocation.intervals) |interval| {
        const location = allocation.locations[interval.reg];
        switch (location) {
            .spill => |slot_index| {
                switch (allocation.source.runtime_values[interval.reg]) {
                    .derived_ptr => return error.DerivedPointerSpilled,
                    .dalvik => |value| {
                        if (value.gc_root) stats.root_spills_checked += 1;
                    },
                }
                if (plan) |spill_plan| {
                    const slot = spillSlotFor(spill_plan, interval.reg) orelse return error.BadSpillLocation;
                    if (slot.slot != slot_index) return error.BadSpillLocation;
                }
            },
            .phys => {
                switch (allocation.source.runtime_values[interval.reg]) {
                    .derived_ptr => {
                        stats.derived_intervals += 1;
                        stats.physical_derived += 1;
                    },
                    .dalvik => {},
                }
            },
            .none => return error.InvalidAllocation,
        }
    }
}

fn verifyPhysicalConflicts(allocation: *const linear_scan.Allocation, stats: *Stats) Error!void {
    for (allocation.intervals, 0..) |a, a_index| {
        const a_phys = switch (allocation.locations[a.reg]) {
            .phys => |phys| phys,
            else => continue,
        };
        for (allocation.intervals[a_index + 1 ..]) |b| {
            const b_phys = switch (allocation.locations[b.reg]) {
                .phys => |phys| phys,
                else => continue,
            };
            stats.overlap_pairs_checked += 1;
            if (a_phys == b_phys and intervalsOverlap(a, b)) return error.LocationConflict;
        }
    }
}

fn verifyAddressLocations(allocation: *const linear_scan.Allocation, stats: *Stats) Error!void {
    for (allocation.source.blocks) |block| {
        for (block.insts) |inst| {
            const address = inst.address orelse continue;
            const handle = inst.state_handle orelse return error.CanonicalHandleUnavailable;
            const address_location = allocation.locationOf(address) orelse return error.InvalidAllocation;
            const handle_location = allocation.locationOf(handle) orelse return error.InvalidAllocation;
            const address_phys = switch (address_location) {
                .phys => |phys| phys,
                .spill => return error.DerivedPointerSpilled,
                .none => return error.InvalidAllocation,
            };
            switch (handle_location) {
                .phys => |handle_phys| if (handle_phys == address_phys) return error.LocationConflict,
                .spill => {},
                .none => return error.CanonicalHandleUnavailable,
            }
            if (!allocation.source.isGcRoot(handle)) return error.CanonicalHandleUnavailable;
            stats.canonical_handle_uses += 1;
        }
    }
}

pub fn verify(
    allocator: std.mem.Allocator,
    allocation: *const linear_scan.Allocation,
    spill_plan: ?*const spill_rewrite.Plan,
) Error!Stats {
    allocation.verify() catch |err| switch (err) {
        error.OutOfMemory => return error.OutOfMemory,
        else => return error.InvalidAllocation,
    };

    var machine_proof = try machine_verify.run(allocator, allocation.source);
    defer machine_proof.deinit();
    try machine_proof.verify();

    var stats: Stats = .{};
    try verifySpills(allocation, spill_plan, &stats);
    try verifyPhysicalConflicts(allocation, &stats);
    try verifyAddressLocations(allocation, &stats);
    return stats;
}

test "post allocation verifier rejects derived pointer spills" {
    const optimizer = @import("optimizer");
    const Instruction = @import("instructions").Instruction;
    const insts = [_]Instruction{
        .{ .const_string = .{ .dest = 0, .index = 1 } },
        .{ .iget = .{ .field_idx = 1, .dest_or_src = 1, .obj = 0 } },
        .return_void,
    };
    var optimized = try optimizer.optimize(std.testing.allocator, &insts, &.{}, .{});
    defer optimized.deinit();

    var allocation = try linear_scan.allocate(std.testing.allocator, &optimized.machine, .{ .gp_registers = &.{} });
    defer allocation.deinit();
    try std.testing.expectError(error.DerivedPointerSpilled, verify(std.testing.allocator, &allocation, null));
}

fn allocationFailureProbe(allocator: std.mem.Allocator, allocation: *const linear_scan.Allocation) !void {
    _ = try verify(allocator, allocation, null);
}

test "post allocation verifier accepts physical derived pointers and is allocation-failure safe" {
    const optimizer = @import("optimizer");
    const Instruction = @import("instructions").Instruction;
    const insts = [_]Instruction{
        .{ .const_string = .{ .dest = 0, .index = 1 } },
        .{ .iget = .{ .field_idx = 1, .dest_or_src = 1, .obj = 0 } },
        .return_void,
    };
    var optimized = try optimizer.optimize(std.testing.allocator, &insts, &.{}, .{});
    defer optimized.deinit();
    var allocation = try linear_scan.allocateDefault(std.testing.allocator, &optimized.machine);
    defer allocation.deinit();

    const stats = try verify(std.testing.allocator, &allocation, null);
    try std.testing.expectEqual(@as(u32, 1), stats.derived_intervals);
    try std.testing.expectEqual(@as(u32, 1), stats.physical_derived);
    try std.testing.expect(stats.canonical_handle_uses >= 1);
    try std.testing.checkAllAllocationFailures(std.testing.allocator, allocationFailureProbe, .{&allocation});
}
