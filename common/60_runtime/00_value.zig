//! Stable managed handles and atomic object-location publication.
//!
//! A handle-table entry is one atomic u64. Generation, region, offset, and
//! lifecycle state therefore change at one linearization point; readers can
//! never combine fields from different publications.

const std = @import("std");

pub const object_alignment: usize = 8;
pub const null_index: u32 = std.math.maxInt(u32);

pub const EntryState = enum(u4) {
    free,
    constructing,
    live,
    evacuating,
    retired,
};

/// Region allocation/evacuation lifecycle. Only `.active` accepts new object
/// publications. `.evacuating` is sealed but remains readable; `.retired`
/// remains readable until an external epoch proof permits reuse.
pub const RegionState = enum(u8) {
    active,
    evacuating,
    retired,
    reclaiming,
};

pub const Handle = packed struct(u64) {
    index: u32,
    generation: u16,
    kind: u8 = 0,
    flags: u8 = 0,

    pub const none: Handle = .{
        .index = null_index,
        .generation = 0,
    };

    pub inline fn isNull(self: Handle) bool {
        return self.index == null_index;
    }
};

pub const LocationDescriptor = packed struct(u64) {
    /// Offset from the immutable region base in eight-byte units.
    offset_units: u36,
    region_id: u8,
    generation: u16,
    state: EntryState,

    pub inline fn bits(self: LocationDescriptor) u64 {
        return @bitCast(self);
    }

    pub inline fn fromBits(value: u64) LocationDescriptor {
        return @bitCast(value);
    }
};

pub const Region = struct {
    /// Region bytes are externally owned, immutable in base/length, and must
    /// outlive every handle table that references them.
    base: usize,
    len: usize,

    pub fn fromSlice(bytes: []u8) Error!Region {
        const base = @intFromPtr(bytes.ptr);
        if (!std.mem.isAligned(base, object_alignment)) return error.UnalignedRegion;
        if (bytes.len > std.math.maxInt(usize) - base) return error.InvalidRegion;
        return .{ .base = base, .len = bytes.len };
    }
};

pub const Error = error{
    AddressOutsideRegion,
    GenerationExhausted,
    InvalidHandle,
    InvalidRegion,
    InvalidState,
    RegionReferenced,
    RegionUnavailable,
    Retired,
    StaleHandle,
    TableFull,
    UnalignedAddress,
    UnalignedRegion,
};

const Entry = struct {
    descriptor: std.atomic.Value(u64),
};

const RegionLifecycle = struct {
    state: std.atomic.Value(RegionState),
};

pub const RelocationTicket = struct {
    handle: Handle,
    observed_descriptor: u64,
};

pub const HandleTable = struct {
    allocator: std.mem.Allocator,
    entries: []Entry,
    regions: []Region,
    region_lifecycles: []RegionLifecycle,
    next_candidate: std.atomic.Value(usize),

    pub fn init(
        allocator: std.mem.Allocator,
        capacity: usize,
        immutable_regions: []const Region,
    ) (Error || std.mem.Allocator.Error)!HandleTable {
        if (capacity == 0 or capacity > std.math.maxInt(u32)) return error.InvalidHandle;
        if (immutable_regions.len == 0 or immutable_regions.len > 256) return error.InvalidRegion;

        const entries = try allocator.alloc(Entry, capacity);
        errdefer allocator.free(entries);

        const regions = try allocator.dupe(Region, immutable_regions);
        errdefer allocator.free(regions);

        const region_lifecycles = try allocator.alloc(RegionLifecycle, immutable_regions.len);
        errdefer allocator.free(region_lifecycles);

        for (regions) |region| {
            if (!std.mem.isAligned(region.base, object_alignment)) return error.UnalignedRegion;
            if (region.len > std.math.maxInt(usize) - region.base) return error.InvalidRegion;
        }

        const initial = (LocationDescriptor{
            .offset_units = 0,
            .region_id = 0,
            .generation = 1,
            .state = .free,
        }).bits();
        for (entries) |*entry| {
            entry.* = .{ .descriptor = std.atomic.Value(u64).init(initial) };
        }
        for (region_lifecycles) |*lifecycle| {
            lifecycle.* = .{ .state = std.atomic.Value(RegionState).init(.active) };
        }

        return .{
            .allocator = allocator,
            .entries = entries,
            .regions = regions,
            .region_lifecycles = region_lifecycles,
            .next_candidate = std.atomic.Value(usize).init(0),
        };
    }

    pub fn deinit(self: *HandleTable) void {
        self.allocator.free(self.region_lifecycles);
        self.allocator.free(self.regions);
        self.allocator.free(self.entries);
        self.* = undefined;
    }

    /// Reserves one entry. Only the returned handle may publish or cancel it.
    pub fn reserve(self: *HandleTable, kind: u8, flags: u8) Error!Handle {
        const start = self.next_candidate.fetchAdd(1, .monotonic);
        for (0..self.entries.len) |attempt| {
            const index = (start +% attempt) % self.entries.len;
            const entry = &self.entries[index];
            const old_bits = entry.descriptor.load(.acquire);
            var old = LocationDescriptor.fromBits(old_bits);
            if (old.state != .free) continue;

            old.state = .constructing;
            if (entry.descriptor.cmpxchgStrong(
                old_bits,
                old.bits(),
                .acq_rel,
                .acquire,
            ) == null) {
                return .{
                    .index = @intCast(index),
                    .generation = old.generation,
                    .kind = kind,
                    .flags = flags,
                };
            }
        }
        return error.TableFull;
    }

    /// Publishes a fully initialized object. This release operation is the
    /// object-construction linearization point.
    pub fn publish(
        self: *HandleTable,
        handle: Handle,
        region_id: u8,
        address: *anyopaque,
    ) Error!void {
        try self.requireActiveRegion(region_id);
        const new_location = try self.location(region_id, address, handle.generation, .live);
        const entry = try self.entryFor(handle);
        const old_bits = entry.descriptor.load(.acquire);
        const old = LocationDescriptor.fromBits(old_bits);
        try validateGeneration(handle, old);
        if (old.state != .constructing) return error.InvalidState;

        if (entry.descriptor.cmpxchgStrong(
            old_bits,
            new_location.bits(),
            .release,
            .acquire,
        ) != null) return error.InvalidState;
    }

    pub fn cancelReservation(self: *HandleTable, handle: Handle) Error!void {
        const entry = try self.entryFor(handle);
        const old_bits = entry.descriptor.load(.acquire);
        var old = LocationDescriptor.fromBits(old_bits);
        try validateGeneration(handle, old);
        if (old.state != .constructing) return error.InvalidState;
        old.state = .free;
        if (entry.descriptor.cmpxchgStrong(old_bits, old.bits(), .release, .acquire) != null) {
            return error.InvalidState;
        }
    }

    /// Acquires the object publication and returns its current address.
    pub fn resolve(self: *const HandleTable, handle: Handle) Error!*anyopaque {
        if (handle.isNull()) return error.InvalidHandle;
        const entry = try self.entryForConst(handle);
        const descriptor = LocationDescriptor.fromBits(entry.descriptor.load(.acquire));
        try validateGeneration(handle, descriptor);
        switch (descriptor.state) {
            .live, .evacuating => {},
            .retired => return error.Retired,
            else => return error.InvalidState,
        }
        return try self.addressOf(descriptor);
    }

    /// Atomically changes a live entry to evacuating and captures that exact
    /// descriptor. Mutators route this uncommon state through the epoch-safe
    /// slow path. Multiple collectors may copy concurrently, but only a ticket
    /// matching the still-current descriptor can win publication.
    pub fn beginRelocation(self: *HandleTable, handle: Handle) Error!RelocationTicket {
        const entry = try self.entryFor(handle);
        while (true) {
            const observed = entry.descriptor.load(.acquire);
            var descriptor = LocationDescriptor.fromBits(observed);
            try validateGeneration(handle, descriptor);
            if (descriptor.state != .live) return error.InvalidState;
            descriptor.state = .evacuating;
            const evacuating = descriptor.bits();
            if (entry.descriptor.cmpxchgStrong(observed, evacuating, .acq_rel, .acquire) == null) {
                return .{ .handle = handle, .observed_descriptor = evacuating };
            }
        }
    }

    /// Returns true only for the collector that atomically publishes the copy.
    pub fn commitRelocation(
        self: *HandleTable,
        ticket: RelocationTicket,
        region_id: u8,
        new_address: *anyopaque,
    ) Error!bool {
        try self.requireActiveRegion(region_id);
        const old = LocationDescriptor.fromBits(ticket.observed_descriptor);
        try validateGeneration(ticket.handle, old);
        if (old.state != .evacuating) return error.InvalidState;
        const new_location = try self.location(region_id, new_address, old.generation, .live);
        const entry = try self.entryFor(ticket.handle);
        return entry.descriptor.cmpxchgStrong(
            ticket.observed_descriptor,
            new_location.bits(),
            .acq_rel,
            .acquire,
        ) == null;
    }

    /// Makes the handle unreachable to new resolvers. Physical memory remains
    /// owned by the retiring epoch until every reader is quiescent.
    pub fn retire(self: *HandleTable, handle: Handle) Error!bool {
        const entry = try self.entryFor(handle);
        while (true) {
            const old_bits = entry.descriptor.load(.acquire);
            var old = LocationDescriptor.fromBits(old_bits);
            try validateGeneration(handle, old);
            switch (old.state) {
                .retired => return false,
                .live => {},
                else => return error.InvalidState,
            }
            old.state = .retired;
            if (entry.descriptor.cmpxchgStrong(old_bits, old.bits(), .acq_rel, .acquire) == null) {
                return true;
            }
        }
    }

    /// Recycles an entry only after the caller has proven epoch quiescence.
    pub fn recycleAfterQuiescence(self: *HandleTable, handle: Handle) Error!void {
        const entry = try self.entryFor(handle);
        while (true) {
            const old_bits = entry.descriptor.load(.acquire);
            const old = LocationDescriptor.fromBits(old_bits);
            try validateGeneration(handle, old);
            if (old.state != .retired) return error.InvalidState;
            if (old.generation == std.math.maxInt(u16)) return error.GenerationExhausted;

            const free = LocationDescriptor{
                .offset_units = 0,
                .region_id = 0,
                .generation = old.generation + 1,
                .state = .free,
            };
            if (entry.descriptor.cmpxchgStrong(old_bits, free.bits(), .acq_rel, .acquire) == null) {
                return;
            }
        }
    }

    pub fn inspect(self: *const HandleTable, handle: Handle) Error!LocationDescriptor {
        const entry = try self.entryForConst(handle);
        const descriptor = LocationDescriptor.fromBits(entry.descriptor.load(.acquire));
        try validateGeneration(handle, descriptor);
        return descriptor;
    }

    /// Stable JIT metadata. Entries and regions never resize after init and
    /// remain alive until `deinit`, so managed entry trampolines may cache
    /// these addresses for the duration of a runtime lease.
    pub fn entryCapacity(self: *const HandleTable) u32 {
        return @intCast(self.entries.len);
    }

    pub fn descriptorBaseAddress(self: *const HandleTable) usize {
        return @intFromPtr(self.entries.ptr);
    }

    pub fn descriptorStride(_: *const HandleTable) u8 {
        return @sizeOf(Entry);
    }

    pub fn regionCount(self: *const HandleTable) u16 {
        return @intCast(self.regions.len);
    }

    pub fn regionAt(self: *const HandleTable, index: u16) Error!Region {
        if (index >= self.regions.len) return error.InvalidRegion;
        return self.regions[index];
    }

    pub fn regionState(self: *const HandleTable, region_id: u8) Error!RegionState {
        const index: usize = region_id;
        if (index >= self.region_lifecycles.len) return error.InvalidRegion;
        return self.region_lifecycles[index].state.load(.acquire);
    }

    /// Seals a region against new publications. Publications already racing
    /// this CAS are closed by the coordinator's later mutator handshake and
    /// final reference scan before retirement.
    pub fn beginRegionEvacuation(self: *HandleTable, region_id: u8) Error!void {
        const lifecycle = try self.regionLifecycle(region_id);
        if (lifecycle.state.cmpxchgStrong(.active, .evacuating, .acq_rel, .acquire) != null) {
            return error.RegionUnavailable;
        }
    }

    /// Cancels a seal before a retirement handshake has begun. Relocations
    /// already published elsewhere remain valid; the source simply resumes
    /// accepting allocations for any objects that remain there.
    pub fn cancelRegionEvacuation(self: *HandleTable, region_id: u8) Error!void {
        const lifecycle = try self.regionLifecycle(region_id);
        if (lifecycle.state.cmpxchgStrong(.evacuating, .active, .acq_rel, .acquire) != null) {
            return error.InvalidState;
        }
    }

    /// Transitions an empty sealed region to epoch-retired. Memory remains
    /// readable and must not be reused until the allocator reset handoff.
    pub fn retireEvacuatedRegion(self: *HandleTable, region_id: u8) Error!void {
        if (try self.regionReferenceCount(region_id) != 0) return error.RegionReferenced;
        const lifecycle = try self.regionLifecycle(region_id);
        if (lifecycle.state.cmpxchgStrong(.evacuating, .retired, .acq_rel, .acquire) != null) {
            return error.InvalidState;
        }
    }

    /// Caller must hold a completed quiescence proof for the retirement epoch.
    /// The resulting state is still non-allocatable while heap metadata and
    /// physical bytes are reset.
    pub fn claimRegionForReuseAfterQuiescence(self: *HandleTable, region_id: u8) Error!void {
        const lifecycle = try self.regionLifecycle(region_id);
        if (lifecycle.state.cmpxchgStrong(.retired, .reclaiming, .acq_rel, .acquire) != null) {
            return error.InvalidState;
        }
    }

    /// Publishes a fully reset region to mutator allocation. The allocator
    /// calls this only after clearing bump/object metadata and reusable bytes.
    pub fn activateRegionAfterReset(self: *HandleTable, region_id: u8) Error!void {
        const lifecycle = try self.regionLifecycle(region_id);
        if (lifecycle.state.cmpxchgStrong(.reclaiming, .active, .release, .acquire) != null) {
            return error.InvalidState;
        }
    }

    /// Allocation-free collector scan. Only states that still name physical
    /// object storage count; free/constructing entries have no published body.
    pub fn regionReferenceCount(self: *const HandleTable, region_id: u8) Error!usize {
        const index: usize = region_id;
        if (index >= self.regions.len) return error.InvalidRegion;
        var count: usize = 0;
        for (self.entries) |*entry| {
            const descriptor = LocationDescriptor.fromBits(entry.descriptor.load(.acquire));
            if (descriptor.region_id != region_id) continue;
            switch (descriptor.state) {
                .live, .evacuating, .retired => count += 1,
                .free, .constructing => {},
            }
        }
        return count;
    }

    fn location(
        self: *const HandleTable,
        region_id: u8,
        address: *anyopaque,
        generation: u16,
        state: EntryState,
    ) Error!LocationDescriptor {
        const region_index: usize = region_id;
        if (region_index >= self.regions.len) return error.InvalidRegion;
        const region = self.regions[region_index];
        const raw = @intFromPtr(address);
        if (!std.mem.isAligned(raw, object_alignment)) return error.UnalignedAddress;
        if (raw < region.base) return error.AddressOutsideRegion;
        const offset = raw - region.base;
        if (offset >= region.len) return error.AddressOutsideRegion;
        if (!std.mem.isAligned(offset, object_alignment)) return error.UnalignedAddress;
        const units = offset / object_alignment;
        if (units > std.math.maxInt(u36)) return error.AddressOutsideRegion;
        return .{
            .offset_units = @intCast(units),
            .region_id = region_id,
            .generation = generation,
            .state = state,
        };
    }

    fn regionLifecycle(self: *HandleTable, region_id: u8) Error!*RegionLifecycle {
        const index: usize = region_id;
        if (index >= self.region_lifecycles.len) return error.InvalidRegion;
        return &self.region_lifecycles[index];
    }

    fn requireActiveRegion(self: *const HandleTable, region_id: u8) Error!void {
        if (try self.regionState(region_id) != .active) return error.RegionUnavailable;
    }

    fn addressOf(self: *const HandleTable, descriptor: LocationDescriptor) Error!*anyopaque {
        const region_index: usize = descriptor.region_id;
        if (region_index >= self.regions.len) return error.InvalidRegion;
        const region = self.regions[region_index];
        const offset = @as(usize, descriptor.offset_units) * object_alignment;
        if (offset >= region.len) return error.AddressOutsideRegion;
        return @ptrFromInt(region.base + offset);
    }

    fn entryFor(self: *HandleTable, handle: Handle) Error!*Entry {
        if (handle.isNull() or handle.index >= self.entries.len) return error.InvalidHandle;
        return &self.entries[handle.index];
    }

    fn entryForConst(self: *const HandleTable, handle: Handle) Error!*const Entry {
        if (handle.isNull() or handle.index >= self.entries.len) return error.InvalidHandle;
        return &self.entries[handle.index];
    }
};

fn validateGeneration(handle: Handle, descriptor: LocationDescriptor) Error!void {
    if (handle.generation != descriptor.generation) return error.StaleHandle;
}

test "descriptor is exactly one atomic word" {
    try std.testing.expectEqual(@as(usize, @sizeOf(u64)), @sizeOf(LocationDescriptor));
    try std.testing.expectEqual(@as(usize, @sizeOf(u64)), @sizeOf(Entry));
}

test "handle publication relocation retirement and stale rejection" {
    var region_a: [128]u8 align(object_alignment) = undefined;
    var region_b: [128]u8 align(object_alignment) = undefined;
    const regions = [_]Region{
        try Region.fromSlice(&region_a),
        try Region.fromSlice(&region_b),
    };

    var table = try HandleTable.init(std.testing.allocator, 4, &regions);
    defer table.deinit();

    const handle = try table.reserve(3, 7);
    const original: *anyopaque = @ptrCast(&region_a[16]);
    try table.publish(handle, 0, original);
    try std.testing.expectEqual(@intFromPtr(original), @intFromPtr(try table.resolve(handle)));

    const ticket = try table.beginRelocation(handle);
    const moved: *anyopaque = @ptrCast(&region_b[32]);
    try std.testing.expect(try table.commitRelocation(ticket, 1, moved));
    try std.testing.expectEqual(@intFromPtr(moved), @intFromPtr(try table.resolve(handle)));
    try std.testing.expect(!(try table.commitRelocation(ticket, 1, @ptrCast(&region_b[48]))));

    try std.testing.expect(try table.retire(handle));
    try std.testing.expectError(error.Retired, table.resolve(handle));
    try table.recycleAfterQuiescence(handle);
    try std.testing.expectError(error.StaleHandle, table.resolve(handle));
}

test "reservation cancellation and capacity are deterministic" {
    var storage: [64]u8 align(object_alignment) = undefined;
    const regions = [_]Region{try Region.fromSlice(&storage)};
    var table = try HandleTable.init(std.testing.allocator, 1, &regions);
    defer table.deinit();

    const first = try table.reserve(0, 0);
    try std.testing.expectError(error.TableFull, table.reserve(0, 0));
    try table.cancelReservation(first);
    const second = try table.reserve(0, 0);
    try std.testing.expectEqual(first.index, second.index);
    try std.testing.expectEqual(first.generation, second.generation);
}

test "racing collectors publish exactly one relocation" {
    var region_a: [128]u8 align(object_alignment) = undefined;
    var region_b: [128]u8 align(object_alignment) = undefined;
    const regions = [_]Region{
        try Region.fromSlice(&region_a),
        try Region.fromSlice(&region_b),
    };
    var table = try HandleTable.init(std.testing.allocator, 1, &regions);
    defer table.deinit();

    const handle = try table.reserve(0, 0);
    try table.publish(handle, 0, @ptrCast(&region_a[8]));
    const ticket = try table.beginRelocation(handle);
    var wins = std.atomic.Value(u32).init(0);

    const Context = struct {
        table: *HandleTable,
        ticket: RelocationTicket,
        address: *anyopaque,
        wins: *std.atomic.Value(u32),

        fn run(ctx: *@This()) void {
            if (ctx.table.commitRelocation(ctx.ticket, 1, ctx.address) catch false) {
                _ = ctx.wins.fetchAdd(1, .monotonic);
            }
        }
    };

    var a = Context{ .table = &table, .ticket = ticket, .address = @ptrCast(&region_b[8]), .wins = &wins };
    var b = Context{ .table = &table, .ticket = ticket, .address = @ptrCast(&region_b[24]), .wins = &wins };
    const thread_a = try std.Thread.spawn(.{}, Context.run, .{&a});
    const thread_b = try std.Thread.spawn(.{}, Context.run, .{&b});
    thread_a.join();
    thread_b.join();

    try std.testing.expectEqual(@as(u32, 1), wins.load(.acquire));
    const resolved = @intFromPtr(try table.resolve(handle));
    try std.testing.expect(resolved == @intFromPtr(a.address) or resolved == @intFromPtr(b.address));
}

test "region lifecycle seals publication and requires an empty quiescent reopen" {
    var region_a: [128]u8 align(object_alignment) = undefined;
    var region_b: [128]u8 align(object_alignment) = undefined;
    const regions = [_]Region{
        try Region.fromSlice(&region_a),
        try Region.fromSlice(&region_b),
    };
    var table = try HandleTable.init(std.testing.allocator, 2, &regions);
    defer table.deinit();

    const live = try table.reserve(0, 0);
    try table.publish(live, 0, @ptrCast(&region_a[8]));
    try table.beginRegionEvacuation(0);
    try std.testing.expectEqual(RegionState.evacuating, try table.regionState(0));
    try std.testing.expectEqual(@as(usize, 1), try table.regionReferenceCount(0));
    try std.testing.expectError(error.RegionReferenced, table.retireEvacuatedRegion(0));

    const rejected = try table.reserve(0, 0);
    try std.testing.expectError(error.RegionUnavailable, table.publish(rejected, 0, @ptrCast(&region_a[24])));
    try table.cancelReservation(rejected);

    const ticket = try table.beginRelocation(live);
    try std.testing.expect(try table.commitRelocation(ticket, 1, @ptrCast(&region_b[16])));
    try std.testing.expectEqual(@as(usize, 0), try table.regionReferenceCount(0));
    try table.retireEvacuatedRegion(0);
    try std.testing.expectEqual(RegionState.retired, try table.regionState(0));
    const retired_rejected = try table.reserve(0, 0);
    try std.testing.expectError(error.RegionUnavailable, table.publish(retired_rejected, 0, @ptrCast(&region_a[32])));
    try table.cancelReservation(retired_rejected);

    try table.claimRegionForReuseAfterQuiescence(0);
    try std.testing.expectEqual(RegionState.reclaiming, try table.regionState(0));
    try table.activateRegionAfterReset(0);
    try std.testing.expectEqual(RegionState.active, try table.regionState(0));
}

fn allocationFailureInit(allocator: std.mem.Allocator) !void {
    var storage: [64]u8 align(object_alignment) = undefined;
    const regions = [_]Region{try Region.fromSlice(&storage)};
    var table = try HandleTable.init(allocator, 8, &regions);
    defer table.deinit();
}

test "handle table initialization is leak-free at every allocation failure" {
    try std.testing.checkAllAllocationFailures(
        std.testing.allocator,
        allocationFailureInit,
        .{},
    );
}
