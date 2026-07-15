//! Concurrent evacuation and epoch-gated region reuse.
//!
//! Object copying and handle publication happen while mutators run. A final
//! asynchronous registry handshake invalidates every pre-retirement derived
//! pointer. Region memory becomes reusable only after all captured mutators
//! acknowledge and are released; this coordinator never waits for them.

const std = @import("std");
const runtime_value = @import("runtime_value");
const thread_registry = @import("runtime_thread_registry");

const Handle = runtime_value.Handle;
const HandleTable = runtime_value.HandleTable;
const Registry = thread_registry.Registry;
const ThreadContext = thread_registry.ThreadContext;

pub const Error = runtime_value.Error || thread_registry.Error || std.mem.Allocator.Error || error{
    ActiveCycle,
    InvalidAlignment,
    InvalidObject,
    InvalidTlabSize,
    NotQuiescent,
    ObjectAlreadyPublished,
    OutOfRegionMemory,
    StaleReservation,
    WrongThread,
    WrongPhase,
    WrongSourceRegion,
};

pub const Phase = enum(u8) {
    evacuating,
    retired,
    handshaking,
    quiescent,
    reclaimed,
    cancelled,
};

pub const Stats = struct {
    cycles_started: u64,
    relocation_wins: u64,
    relocation_losses: u64,
    handshakes_started: u64,
    mutators_released: u64,
    regions_reclaimed: u64,
    last_retirement_epoch: u64,
};

const AtomicStats = struct {
    cycles_started: std.atomic.Value(u64) = std.atomic.Value(u64).init(0),
    relocation_wins: std.atomic.Value(u64) = std.atomic.Value(u64).init(0),
    relocation_losses: std.atomic.Value(u64) = std.atomic.Value(u64).init(0),
    handshakes_started: std.atomic.Value(u64) = std.atomic.Value(u64).init(0),
    mutators_released: std.atomic.Value(u64) = std.atomic.Value(u64).init(0),
    regions_reclaimed: std.atomic.Value(u64) = std.atomic.Value(u64).init(0),
    last_retirement_epoch: std.atomic.Value(u64) = std.atomic.Value(u64).init(0),
};

pub const ReclaimedRegion = struct {
    region_id: u8,
    retirement_epoch: u64,
};

const AtomicU64 = std.atomic.Value(u64);
const ObjectHeader = extern struct {
    layout: AtomicU64,
    size: AtomicU64,
};

const object_header_size: usize = @sizeOf(ObjectHeader);
const object_header_magic: u32 = 0x4f42_4a31;
const object_layout_magic: u32 = 0x4c41_5931;

const AllocationRegion = struct {
    bytes: runtime_value.Region,
    top: std.atomic.Value(usize),
    generation: std.atomic.Value(u64),
    start_words: []AtomicU64,
};

pub const AllocationStats = struct {
    tlab_refills: u64,
    reserved_bytes: u64,
    objects_published: u64,
    allocation_failures: u64,
    regions_reset: u64,
};

const AtomicAllocationStats = struct {
    tlab_refills: AtomicU64 = AtomicU64.init(0),
    reserved_bytes: AtomicU64 = AtomicU64.init(0),
    objects_published: AtomicU64 = AtomicU64.init(0),
    allocation_failures: AtomicU64 = AtomicU64.init(0),
    regions_reset: AtomicU64 = AtomicU64.init(0),
};

pub const Reservation = struct {
    heap: *ManagedHeap,
    region_id: u8,
    offset: usize,
    allocated_size: u32,
    region_generation: u64,

    pub fn address(self: Reservation) *anyopaque {
        const region = &self.heap.regions[self.region_id];
        return @ptrFromInt(region.bytes.base + self.offset);
    }
};

/// Side metadata and atomic region extents. Initialization allocates every
/// bitmap/size table up front; mutator allocation and publication never call
/// an allocator or take a mutex.
pub const ManagedHeap = struct {
    allocator: std.mem.Allocator,
    handles: *HandleTable,
    regions: []AllocationRegion,
    start_storage: []AtomicU64,
    tlab_bytes: usize,
    next_region: std.atomic.Value(usize),
    counters: AtomicAllocationStats = .{},

    pub fn init(
        allocator: std.mem.Allocator,
        handles: *HandleTable,
        tlab_bytes: usize,
    ) Error!ManagedHeap {
        if (tlab_bytes < runtime_value.object_alignment or
            !std.mem.isAligned(tlab_bytes, runtime_value.object_alignment)) return error.InvalidTlabSize;

        const region_count: usize = handles.regionCount();
        const regions = try allocator.alloc(AllocationRegion, region_count);
        errdefer allocator.free(regions);

        var total_words: usize = 0;
        for (0..region_count) |index| {
            const bytes = try handles.regionAt(@intCast(index));
            if (try handles.regionState(@intCast(index)) != .active or
                try handles.regionReferenceCount(@intCast(index)) != 0) return error.InvalidState;
            const slots = bytes.len / runtime_value.object_alignment;
            const words = slots / 64 + @intFromBool(slots % 64 != 0);
            if (words > std.math.maxInt(usize) - total_words) {
                return error.InvalidRegion;
            }
            total_words += words;
        }

        const start_storage = try allocator.alloc(AtomicU64, total_words);
        errdefer allocator.free(start_storage);
        for (start_storage) |*word| word.* = AtomicU64.init(0);

        var word_cursor: usize = 0;
        for (regions, 0..) |*region, index| {
            const bytes = try handles.regionAt(@intCast(index));
            const region_bytes: [*]u8 = @ptrFromInt(bytes.base);
            @memset(region_bytes[0..bytes.len], 0);
            const slots = bytes.len / runtime_value.object_alignment;
            const words = slots / 64 + @intFromBool(slots % 64 != 0);
            region.* = .{
                .bytes = bytes,
                .top = std.atomic.Value(usize).init(0),
                .generation = AtomicU64.init(1),
                .start_words = start_storage[word_cursor .. word_cursor + words],
            };
            word_cursor += words;
        }

        return .{
            .allocator = allocator,
            .handles = handles,
            .regions = regions,
            .start_storage = start_storage,
            .tlab_bytes = tlab_bytes,
            .next_region = std.atomic.Value(usize).init(0),
        };
    }

    pub fn deinit(self: *ManagedHeap) void {
        self.allocator.free(self.start_storage);
        self.allocator.free(self.regions);
        self.* = undefined;
    }

    pub fn threadAllocator(self: *ManagedHeap) ThreadAllocator {
        return .{ .heap = self, .owner = std.Thread.getCurrentId() };
    }

    pub fn publishObject(self: *ManagedHeap, reservation: Reservation, handle: Handle) Error!void {
        return self.publishObjectWithLayout(reservation, handle, 0);
    }

    /// Publishes immutable GC layout identity before making the handle live.
    /// Layout zero is reserved for explicitly opaque leaf objects.
    pub fn publishObjectWithLayout(
        self: *ManagedHeap,
        reservation: Reservation,
        handle: Handle,
        layout_id: u32,
    ) Error!void {
        const region = try self.validateReservation(reservation);
        const slot = reservation.offset / runtime_value.object_alignment;
        const header = try objectHeader(region, reservation.offset);
        if (header.size.cmpxchgStrong(0, packObjectSize(reservation.allocated_size), .acq_rel, .acquire) != null) {
            return error.ObjectAlreadyPublished;
        }
        header.layout.store(packObjectLayout(layout_id), .monotonic);
        const word = &region.start_words[slot / 64];
        const bit: u6 = @intCast(slot % 64);
        if (word.bitSet(bit, .release) != 0) {
            header.layout.store(0, .monotonic);
            header.size.store(0, .release);
            return error.ObjectAlreadyPublished;
        }

        self.handles.publish(handle, reservation.region_id, reservation.address()) catch |err| {
            header.layout.store(0, .monotonic);
            header.size.store(0, .release);
            _ = word.bitReset(bit, .release);
            return err;
        };
        _ = self.counters.objects_published.fetchAdd(1, .monotonic);
    }

    pub fn isObjectStart(self: *const ManagedHeap, region_id: u8, offset: usize) Error!bool {
        const region = try self.allocationRegionConst(region_id);
        if (!std.mem.isAligned(offset, runtime_value.object_alignment) or offset >= region.bytes.len) {
            return error.InvalidObject;
        }
        const slot = offset / runtime_value.object_alignment;
        const word = region.start_words[slot / 64].load(.acquire);
        const bit: u6 = @intCast(slot % 64);
        return (word & (@as(u64, 1) << bit)) != 0;
    }

    pub fn objectSize(self: *const ManagedHeap, region_id: u8, offset: usize) Error!u32 {
        if (!try self.isObjectStart(region_id, offset)) return error.InvalidObject;
        const region = try self.allocationRegionConst(region_id);
        const header = try objectHeaderConst(region, offset);
        const bits = header.size.load(.acquire);
        if (@as(u32, @truncate(bits >> 32)) != object_header_magic) return error.InvalidObject;
        const size: u32 = @truncate(bits);
        if (size == 0) return error.InvalidObject;
        return size;
    }

    pub fn objectLayoutId(self: *const ManagedHeap, region_id: u8, offset: usize) Error!u32 {
        if (!try self.isObjectStart(region_id, offset)) return error.InvalidObject;
        const region = try self.allocationRegionConst(region_id);
        const header = try objectHeaderConst(region, offset);
        const bits = header.layout.load(.acquire);
        if (@as(u32, @truncate(bits >> 32)) != object_layout_magic) return error.InvalidObject;
        return @truncate(bits);
    }

    /// Clears physical bytes and all side metadata while the handle table keeps
    /// the region non-allocatable in `.reclaiming`, then release-publishes the
    /// new allocator generation and activates the region.
    pub fn resetReclaimedRegion(self: *ManagedHeap, reclaimed: ReclaimedRegion) Error!void {
        if (try self.handles.regionState(reclaimed.region_id) != .reclaiming) return error.InvalidState;
        const region = try self.allocationRegion(reclaimed.region_id);
        const generation = region.generation.load(.acquire);
        if (generation == std.math.maxInt(u64)) return error.GenerationExhausted;

        const bytes: [*]u8 = @ptrFromInt(region.bytes.base);
        @memset(bytes[0..region.bytes.len], 0);
        for (region.start_words) |*word| word.store(0, .monotonic);
        region.top.store(0, .monotonic);
        region.generation.store(generation + 1, .release);
        try self.handles.activateRegionAfterReset(reclaimed.region_id);
        _ = self.counters.regions_reset.fetchAdd(1, .monotonic);
    }

    pub fn stats(self: *const ManagedHeap) AllocationStats {
        return .{
            .tlab_refills = self.counters.tlab_refills.load(.acquire),
            .reserved_bytes = self.counters.reserved_bytes.load(.acquire),
            .objects_published = self.counters.objects_published.load(.acquire),
            .allocation_failures = self.counters.allocation_failures.load(.acquire),
            .regions_reset = self.counters.regions_reset.load(.acquire),
        };
    }

    fn validateReservation(self: *ManagedHeap, reservation: Reservation) Error!*AllocationRegion {
        if (reservation.heap != self) return error.StaleReservation;
        const region = try self.allocationRegion(reservation.region_id);
        if (region.generation.load(.acquire) != reservation.region_generation) return error.StaleReservation;
        const size: usize = reservation.allocated_size;
        if (size == 0 or !std.mem.isAligned(size, runtime_value.object_alignment) or
            reservation.offset < object_header_size or
            !std.mem.isAligned(reservation.offset, runtime_value.object_alignment) or
            reservation.offset > region.bytes.len or size > region.bytes.len - reservation.offset or
            reservation.offset + size > region.top.load(.acquire))
        {
            return error.InvalidObject;
        }
        return region;
    }

    fn allocationRegion(self: *ManagedHeap, region_id: u8) Error!*AllocationRegion {
        const index: usize = region_id;
        if (index >= self.regions.len) return error.InvalidRegion;
        return &self.regions[index];
    }

    fn allocationRegionConst(self: *const ManagedHeap, region_id: u8) Error!*const AllocationRegion {
        const index: usize = region_id;
        if (index >= self.regions.len) return error.InvalidRegion;
        return &self.regions[index];
    }
};

pub const ThreadAllocator = struct {
    heap: *ManagedHeap,
    owner: std.Thread.Id,
    region_id: ?u8 = null,
    cursor: usize = 0,
    limit: usize = 0,
    region_generation: u64 = 0,

    pub fn allocate(self: *ThreadAllocator, size: usize, requested_alignment: usize) Error!Reservation {
        if (std.Thread.getCurrentId() != self.owner) return error.WrongThread;
        const alignment = @max(requested_alignment, runtime_value.object_alignment);
        if (!std.math.isPowerOfTwo(alignment)) return error.InvalidAlignment;
        const allocated_size = try roundedObjectSize(size);

        if (self.region_id) |region_id| {
            const region = &self.heap.regions[region_id];
            if (try self.heap.handles.regionState(region_id) == .active and
                region.generation.load(.acquire) == self.region_generation)
            {
                const start = payloadOffset(region.bytes, self.cursor, alignment) catch self.limit;
                if (start <= self.limit and allocated_size <= self.limit - start) {
                    self.cursor = start + allocated_size;
                    return .{
                        .heap = self.heap,
                        .region_id = region_id,
                        .offset = start,
                        .allocated_size = @intCast(allocated_size),
                        .region_generation = self.region_generation,
                    };
                }
            }
        }
        return self.refill(allocated_size, alignment);
    }

    fn refill(self: *ThreadAllocator, allocated_size: usize, alignment: usize) Error!Reservation {
        const region_count = self.heap.regions.len;
        const start_index = self.heap.next_region.fetchAdd(1, .monotonic);
        for (0..region_count) |attempt| {
            const index = (start_index +% attempt) % region_count;
            const region_id: u8 = @intCast(index);
            if (try self.heap.handles.regionState(region_id) != .active) continue;
            const region = &self.heap.regions[index];
            const generation = region.generation.load(.acquire);
            if (allocated_size > std.math.maxInt(usize) - object_header_size) continue;
            const extent_size = @max(self.heap.tlab_bytes, allocated_size + object_header_size);
            const extent = reserveExtent(region, extent_size, alignment) catch continue;
            if (try self.heap.handles.regionState(region_id) != .active or
                region.generation.load(.acquire) != generation) continue;

            self.region_id = region_id;
            self.cursor = extent.payload + allocated_size;
            self.limit = extent.end;
            self.region_generation = generation;
            _ = self.heap.counters.tlab_refills.fetchAdd(1, .monotonic);
            _ = self.heap.counters.reserved_bytes.fetchAdd(extent.end - extent.start, .monotonic);
            return .{
                .heap = self.heap,
                .region_id = region_id,
                .offset = extent.payload,
                .allocated_size = @intCast(allocated_size),
                .region_generation = generation,
            };
        }
        _ = self.heap.counters.allocation_failures.fetchAdd(1, .monotonic);
        return error.OutOfRegionMemory;
    }
};

const Extent = struct { start: usize, payload: usize, end: usize };

fn reserveExtent(region: *AllocationRegion, extent_size: usize, alignment: usize) Error!Extent {
    var observed = region.top.load(.acquire);
    while (true) {
        if (observed > std.math.maxInt(usize) - object_header_size) return error.OutOfRegionMemory;
        const payload = try payloadOffset(region.bytes, observed, alignment);
        const start = payload - object_header_size;
        if (start > region.bytes.len or extent_size > region.bytes.len - start) return error.OutOfRegionMemory;
        const end = start + extent_size;
        if (region.top.cmpxchgWeak(observed, end, .acq_rel, .acquire)) |actual| {
            observed = actual;
            continue;
        }
        return .{ .start = start, .payload = payload, .end = end };
    }
}

fn alignForward(value: usize, alignment: usize) Error!usize {
    const mask = alignment - 1;
    if (value > std.math.maxInt(usize) - mask) return error.OutOfRegionMemory;
    return (value + mask) & ~mask;
}

fn alignRegionOffset(region: runtime_value.Region, offset: usize, alignment: usize) Error!usize {
    if (offset > region.len or offset > std.math.maxInt(usize) - region.base) return error.OutOfRegionMemory;
    const address = try alignForward(region.base + offset, alignment);
    if (address < region.base) return error.OutOfRegionMemory;
    return address - region.base;
}

fn payloadOffset(region: runtime_value.Region, cursor: usize, alignment: usize) Error!usize {
    if (cursor > std.math.maxInt(usize) - object_header_size) return error.OutOfRegionMemory;
    const payload = try alignRegionOffset(region, cursor + object_header_size, alignment);
    if (payload < object_header_size) return error.OutOfRegionMemory;
    return payload;
}

fn packObjectSize(size: u32) u64 {
    return (@as(u64, object_header_magic) << 32) | size;
}

fn packObjectLayout(layout_id: u32) u64 {
    return (@as(u64, object_layout_magic) << 32) | layout_id;
}

fn objectHeader(region: *AllocationRegion, payload_offset: usize) Error!*ObjectHeader {
    if (payload_offset < object_header_size) return error.InvalidObject;
    return @ptrFromInt(region.bytes.base + payload_offset - object_header_size);
}

fn objectHeaderConst(region: *const AllocationRegion, payload_offset: usize) Error!*const ObjectHeader {
    if (payload_offset < object_header_size) return error.InvalidObject;
    return @ptrFromInt(region.bytes.base + payload_offset - object_header_size);
}

fn roundedObjectSize(size: usize) Error!usize {
    if (size == 0) return error.InvalidObject;
    const rounded = try alignForward(size, runtime_value.object_alignment);
    if (rounded > std.math.maxInt(u32)) return error.InvalidObject;
    return rounded;
}

/// Runtime-owned singleton. The atomic active bit rejects overlapping
/// collectors without a heap lock. Per-cycle mutation remains owner-thread
/// confined, just like a registry `Handshake`.
pub const EvacuationCoordinator = struct {
    allocator: std.mem.Allocator,
    handles: *HandleTable,
    registry: *Registry,
    member_storage: []*ThreadContext,
    released_storage: []bool,
    active: std.atomic.Value(bool),
    counters: AtomicStats = .{},

    pub fn init(
        allocator: std.mem.Allocator,
        handles: *HandleTable,
        registry: *Registry,
        max_threads: usize,
    ) Error!EvacuationCoordinator {
        if (max_threads < registry.memberCapacity()) return error.MemberBufferTooSmall;
        const members = try allocator.alloc(*ThreadContext, max_threads);
        errdefer allocator.free(members);
        const released = try allocator.alloc(bool, max_threads);
        errdefer allocator.free(released);
        @memset(released, false);
        return .{
            .allocator = allocator,
            .handles = handles,
            .registry = registry,
            .member_storage = members,
            .released_storage = released,
            .active = std.atomic.Value(bool).init(false),
        };
    }

    pub fn deinit(self: *EvacuationCoordinator) Error!void {
        if (self.active.load(.acquire)) return error.ActiveCycle;
        self.allocator.free(self.released_storage);
        self.allocator.free(self.member_storage);
        self.* = undefined;
    }

    pub fn begin(self: *EvacuationCoordinator, source_region: u8) Error!EvacuationCycle {
        if (self.active.cmpxchgStrong(false, true, .acq_rel, .acquire) != null) {
            return error.ActiveCycle;
        }
        errdefer self.active.store(false, .release);
        try self.handles.beginRegionEvacuation(source_region);
        @memset(self.released_storage, false);
        _ = self.counters.cycles_started.fetchAdd(1, .monotonic);
        return .{
            .coordinator = self,
            .source_region = source_region,
        };
    }

    pub fn stats(self: *const EvacuationCoordinator) Stats {
        return .{
            .cycles_started = self.counters.cycles_started.load(.acquire),
            .relocation_wins = self.counters.relocation_wins.load(.acquire),
            .relocation_losses = self.counters.relocation_losses.load(.acquire),
            .handshakes_started = self.counters.handshakes_started.load(.acquire),
            .mutators_released = self.counters.mutators_released.load(.acquire),
            .regions_reclaimed = self.counters.regions_reclaimed.load(.acquire),
            .last_retirement_epoch = self.counters.last_retirement_epoch.load(.acquire),
        };
    }
};

pub const EvacuationCycle = struct {
    coordinator: *EvacuationCoordinator,
    source_region: u8,
    phase: Phase = .evacuating,
    handshake: ?thread_registry.Handshake = null,
    retirement_epoch: u64 = 0,
    released_count: usize = 0,

    /// Publishes one already-copied object. Exactly one contender can win the
    /// handle-table CAS; losing copies remain caller-owned and are discarded
    /// by the copying policy.
    pub fn commitRelocation(
        self: *EvacuationCycle,
        ticket: runtime_value.RelocationTicket,
        destination_region: u8,
        destination: *anyopaque,
    ) Error!bool {
        if (self.phase != .evacuating and self.phase != .retired and self.phase != .handshaking) {
            return error.WrongPhase;
        }
        const observed = runtime_value.LocationDescriptor.fromBits(ticket.observed_descriptor);
        if (observed.region_id != self.source_region) return error.WrongSourceRegion;
        const won = try self.coordinator.handles.commitRelocation(ticket, destination_region, destination);
        const counter = if (won)
            &self.coordinator.counters.relocation_wins
        else
            &self.coordinator.counters.relocation_losses;
        _ = counter.fetchAdd(1, .monotonic);
        return won;
    }

    /// Seals the empty source as retired and requests a registry epoch. If a
    /// different handshake is active, retrying this method is safe: the region
    /// remains retired and therefore cannot be reused or allocated into.
    pub fn requestRetirement(self: *EvacuationCycle) Error!u64 {
        if (self.phase == .evacuating) {
            try self.coordinator.handles.retireEvacuatedRegion(self.source_region);
            self.phase = .retired;
        }
        if (self.phase != .retired) return error.WrongPhase;

        const handshake = try self.coordinator.registry.beginHandshake(self.coordinator.member_storage);
        self.retirement_epoch = handshake.epoch;
        @memset(self.coordinator.released_storage[0..handshake.members.len], false);
        self.released_count = 0;
        self.handshake = handshake;
        self.phase = .handshaking;
        _ = self.coordinator.counters.handshakes_started.fetchAdd(1, .monotonic);
        self.coordinator.counters.last_retirement_epoch.store(handshake.epoch, .release);
        return handshake.epoch;
    }

    pub fn isReady(self: *const EvacuationCycle, context: *const ThreadContext) bool {
        if (self.phase != .handshaking) return false;
        const handshake = self.handshake orelse return false;
        return handshake.isReady(context);
    }

    pub fn snapshot(self: *const EvacuationCycle, context: *const ThreadContext) Error![]const Handle {
        if (self.phase != .handshaking) return error.WrongPhase;
        const handshake = self.handshake orelse return error.WrongPhase;
        return handshake.snapshot(context);
    }

    /// Releases every mutator that has independently acknowledged. Returns
    /// true once the retirement epoch is fully quiescent. It never spins or
    /// waits for a mutator that has not reached a poll.
    pub fn advance(self: *EvacuationCycle) Error!bool {
        if (self.phase == .quiescent) return true;
        if (self.phase != .handshaking) return error.WrongPhase;
        const handshake = if (self.handshake) |*value| value else return error.WrongPhase;

        for (handshake.members, 0..) |member, index| {
            if (self.coordinator.released_storage[index]) continue;
            if (!handshake.isReady(member)) continue;
            try handshake.release(member);
            self.coordinator.released_storage[index] = true;
            self.released_count += 1;
            _ = self.coordinator.counters.mutators_released.fetchAdd(1, .monotonic);
        }
        if (self.released_count != handshake.members.len) return false;

        // A publisher may have observed `.active` immediately before the
        // source was sealed and linearized its descriptor afterward. Every
        // captured mutator has now crossed the retirement epoch, so this final
        // acquire scan closes that race. Keep the epoch open and fail closed;
        // the collector may relocate late entries and call `advance` again.
        if (try self.coordinator.handles.regionReferenceCount(self.source_region) != 0) {
            return error.RegionReferenced;
        }

        try handshake.finish();
        self.phase = .quiescent;
        return true;
    }

    /// Grants physical reuse only after `advance` completed the epoch. The
    /// returned token is the allocator's authorization to clear/reuse bytes.
    pub fn reclaim(self: *EvacuationCycle) Error!ReclaimedRegion {
        if (self.phase != .quiescent) return error.NotQuiescent;
        try self.coordinator.handles.claimRegionForReuseAfterQuiescence(self.source_region);
        self.phase = .reclaimed;
        self.coordinator.active.store(false, .release);
        _ = self.coordinator.counters.regions_reclaimed.fetchAdd(1, .monotonic);
        return .{ .region_id = self.source_region, .retirement_epoch = self.retirement_epoch };
    }

    /// Safe only before retirement starts. This is used for transactional
    /// cancellation after copy/allocation failures.
    pub fn cancel(self: *EvacuationCycle) Error!void {
        if (self.phase != .evacuating) return error.WrongPhase;
        try self.coordinator.handles.cancelRegionEvacuation(self.source_region);
        self.phase = .cancelled;
        self.coordinator.active.store(false, .release);
    }
};

fn waitReady(cycle: *const EvacuationCycle, context: *const ThreadContext) !void {
    for (0..1_000_000) |_| {
        if (cycle.isReady(context)) return;
        std.atomic.spinLoopHint();
        std.Thread.yield() catch {};
    }
    return error.Timeout;
}

test "evacuation refuses retirement while a source handle remains" {
    var source: [128]u8 align(runtime_value.object_alignment) = undefined;
    var destination: [128]u8 align(runtime_value.object_alignment) = undefined;
    const regions = [_]runtime_value.Region{
        try runtime_value.Region.fromSlice(&source),
        try runtime_value.Region.fromSlice(&destination),
    };
    var handles = try HandleTable.init(std.testing.allocator, 1, &regions);
    defer handles.deinit();
    const handle = try handles.reserve(0, 0);
    try handles.publish(handle, 0, @ptrCast(&source[8]));
    var registry = try Registry.init(std.testing.allocator, std.testing.io, 0);
    defer registry.deinit() catch unreachable;
    var coordinator = try EvacuationCoordinator.init(std.testing.allocator, &handles, &registry, 0);
    defer coordinator.deinit() catch unreachable;

    var cycle = try coordinator.begin(0);
    try std.testing.expectError(error.ActiveCycle, coordinator.begin(1));
    try std.testing.expectError(error.ActiveCycle, coordinator.deinit());
    try std.testing.expectError(error.RegionReferenced, cycle.requestRetirement());
    try cycle.cancel();
    try std.testing.expectEqual(runtime_value.RegionState.active, try handles.regionState(0));
}

test "region reuse waits for a concurrent mutator retirement acknowledgement" {
    var source: [128]u8 align(runtime_value.object_alignment) = undefined;
    var destination: [128]u8 align(runtime_value.object_alignment) = undefined;
    const regions = [_]runtime_value.Region{
        try runtime_value.Region.fromSlice(&source),
        try runtime_value.Region.fromSlice(&destination),
    };
    var handles = try HandleTable.init(std.testing.allocator, 1, &regions);
    defer handles.deinit();
    const handle = try handles.reserve(0, 0);
    try handles.publish(handle, 0, @ptrCast(&source[16]));

    var registry = try Registry.init(std.testing.allocator, std.testing.io, 1);
    defer registry.deinit() catch unreachable;
    var context = try ThreadContext.init(std.testing.allocator, 1);
    defer context.deinit();
    try context.addRoot(&handle);
    try registry.register(&context);
    defer registry.unregister(&context) catch unreachable;
    var coordinator = try EvacuationCoordinator.init(std.testing.allocator, &handles, &registry, 1);
    defer coordinator.deinit() catch unreachable;

    const Worker = struct {
        registry: *Registry,
        context: *ThreadContext,
        initial_epoch: u64,
        completed: *std.atomic.Value(bool),

        fn run(self: *@This()) void {
            while (self.registry.requestEpoch() == self.initial_epoch) std.atomic.spinLoopHint();
            _ = self.registry.poll(self.context) catch return;
            self.completed.store(true, .release);
        }
    };

    var completed = std.atomic.Value(bool).init(false);
    var worker = Worker{
        .registry = &registry,
        .context = &context,
        .initial_epoch = registry.requestEpoch(),
        .completed = &completed,
    };
    const thread = try std.Thread.spawn(.{}, Worker.run, .{&worker});

    var cycle = try coordinator.begin(0);
    const ticket = try handles.beginRelocation(handle);
    try std.testing.expect(try cycle.commitRelocation(ticket, 1, @ptrCast(&destination[24])));
    try std.testing.expect(!(try cycle.commitRelocation(ticket, 1, @ptrCast(&destination[40]))));
    const epoch = try cycle.requestRetirement();
    try std.testing.expectError(error.NotQuiescent, cycle.reclaim());
    try std.testing.expectEqual(runtime_value.RegionState.retired, try handles.regionState(0));

    try waitReady(&cycle, &context);
    try std.testing.expectEqualSlices(Handle, &.{handle}, try cycle.snapshot(&context));
    try std.testing.expect(try cycle.advance());
    thread.join();
    try std.testing.expect(completed.load(.acquire));

    const reclaimed = try cycle.reclaim();
    try std.testing.expectEqual(@as(u8, 0), reclaimed.region_id);
    try std.testing.expectEqual(epoch, reclaimed.retirement_epoch);
    try std.testing.expectEqual(runtime_value.RegionState.reclaiming, try handles.regionState(0));
    try handles.activateRegionAfterReset(0);
    try std.testing.expectEqual(runtime_value.RegionState.active, try handles.regionState(0));
    try std.testing.expectEqual(@intFromPtr(&destination[24]), @intFromPtr(try handles.resolve(handle)));
    const stats = coordinator.stats();
    try std.testing.expectEqual(@as(u64, 1), stats.cycles_started);
    try std.testing.expectEqual(@as(u64, 1), stats.relocation_wins);
    try std.testing.expectEqual(@as(u64, 1), stats.relocation_losses);
    try std.testing.expectEqual(@as(u64, 1), stats.handshakes_started);
    try std.testing.expectEqual(@as(u64, 1), stats.mutators_released);
    try std.testing.expectEqual(@as(u64, 1), stats.regions_reclaimed);
}

test "empty registry retires without a pause" {
    var source: [64]u8 align(runtime_value.object_alignment) = undefined;
    var destination: [64]u8 align(runtime_value.object_alignment) = undefined;
    const regions = [_]runtime_value.Region{
        try runtime_value.Region.fromSlice(&source),
        try runtime_value.Region.fromSlice(&destination),
    };
    var handles = try HandleTable.init(std.testing.allocator, 1, &regions);
    defer handles.deinit();
    var registry = try Registry.init(std.testing.allocator, std.testing.io, 0);
    defer registry.deinit() catch unreachable;
    var coordinator = try EvacuationCoordinator.init(std.testing.allocator, &handles, &registry, 0);
    defer coordinator.deinit() catch unreachable;

    var cycle = try coordinator.begin(0);
    var no_members: [0]*ThreadContext = .{};
    var competing = try registry.beginHandshake(&no_members);
    try std.testing.expectError(error.HandshakeInProgress, cycle.requestRetirement());
    try std.testing.expectEqual(runtime_value.RegionState.retired, try handles.regionState(0));
    try competing.finish();
    _ = try cycle.requestRetirement();
    try std.testing.expect(try cycle.advance());
    _ = try cycle.reclaim();
    try handles.activateRegionAfterReset(0);
}

test "coordinator rejects handshake storage below registry capacity" {
    var storage: [64]u8 align(runtime_value.object_alignment) = undefined;
    const regions = [_]runtime_value.Region{try runtime_value.Region.fromSlice(&storage)};
    var handles = try HandleTable.init(std.testing.allocator, 1, &regions);
    defer handles.deinit();
    var registry = try Registry.init(std.testing.allocator, std.testing.io, 2);
    defer registry.deinit() catch unreachable;
    try std.testing.expectError(
        error.MemberBufferTooSmall,
        EvacuationCoordinator.init(std.testing.allocator, &handles, &registry, 1),
    );
}

test "tlab publication exposes precise object-start and size metadata" {
    var storage: [512]u8 align(runtime_value.object_alignment) = @splat(0);
    const regions = [_]runtime_value.Region{try runtime_value.Region.fromSlice(&storage)};
    var handles = try HandleTable.init(std.testing.allocator, 4, &regions);
    defer handles.deinit();
    var heap = try ManagedHeap.init(std.testing.allocator, &handles, 64);
    defer heap.deinit();
    var tlab = heap.threadAllocator();

    try std.testing.expectError(error.InvalidObject, tlab.allocate(0, 8));
    try std.testing.expectError(error.InvalidAlignment, tlab.allocate(8, 12));
    const reservation = try tlab.allocate(21, 16);
    try std.testing.expect(std.mem.isAligned(@intFromPtr(reservation.address()), 16));
    try std.testing.expectEqual(@as(u32, 24), reservation.allocated_size);
    const body: [*]u8 = @ptrCast(reservation.address());
    @memset(body[0..reservation.allocated_size], 0x5a);

    const handle = try handles.reserve(0, 0);
    try heap.publishObject(reservation, handle);
    try std.testing.expectEqual(@intFromPtr(reservation.address()), @intFromPtr(try handles.resolve(handle)));
    try std.testing.expect(try heap.isObjectStart(reservation.region_id, reservation.offset));
    try std.testing.expectEqual(@as(u32, 24), try heap.objectSize(reservation.region_id, reservation.offset));
    try std.testing.expectEqual(@as(u32, 0), try heap.objectLayoutId(reservation.region_id, reservation.offset));
    try std.testing.expect(!(try heap.isObjectStart(reservation.region_id, reservation.offset + 8)));

    const duplicate = try handles.reserve(0, 0);
    try std.testing.expectError(error.ObjectAlreadyPublished, heap.publishObject(reservation, duplicate));
    try handles.cancelReservation(duplicate);
    const typed_reservation = try tlab.allocate(16, 8);
    const typed_handle = try handles.reserve(0, 0);
    try heap.publishObjectWithLayout(typed_reservation, typed_handle, 42);
    try std.testing.expectEqual(
        @as(u32, 42),
        try heap.objectLayoutId(typed_reservation.region_id, typed_reservation.offset),
    );
    const stats = heap.stats();
    try std.testing.expectEqual(@as(u64, 2), stats.objects_published);
    try std.testing.expectEqual(@as(u64, 2), stats.tlab_refills);
}

fn addressLess(_: void, lhs: usize, rhs: usize) bool {
    return lhs < rhs;
}

test "concurrent tlab refills reserve unique aligned extents" {
    const thread_count = 4;
    const allocations_per_thread = 100;
    var storage: [32 * 1024]u8 align(runtime_value.object_alignment) = undefined;
    const regions = [_]runtime_value.Region{try runtime_value.Region.fromSlice(&storage)};
    var handles = try HandleTable.init(std.testing.allocator, 1, &regions);
    defer handles.deinit();
    var heap = try ManagedHeap.init(std.testing.allocator, &handles, 256);
    defer heap.deinit();
    var addresses: [thread_count * allocations_per_thread]usize = undefined;

    const Worker = struct {
        heap: *ManagedHeap,
        output: []usize,

        fn run(self: *@This()) void {
            var tlab = self.heap.threadAllocator();
            for (self.output) |*address| {
                const reservation = tlab.allocate(24, 16) catch {
                    address.* = 0;
                    continue;
                };
                address.* = @intFromPtr(reservation.address());
            }
        }
    };

    var workers: [thread_count]Worker = undefined;
    var threads: [thread_count]std.Thread = undefined;
    for (0..thread_count) |index| {
        const start = index * allocations_per_thread;
        workers[index] = .{
            .heap = &heap,
            .output = addresses[start .. start + allocations_per_thread],
        };
        threads[index] = try std.Thread.spawn(.{}, Worker.run, .{&workers[index]});
    }
    for (threads) |thread| thread.join();

    std.mem.sort(usize, &addresses, {}, addressLess);
    for (addresses, 0..) |address, index| {
        try std.testing.expect(address >= @intFromPtr(&storage));
        try std.testing.expect(address + 24 <= @intFromPtr(&storage) + storage.len);
        try std.testing.expect(std.mem.isAligned(address, 16));
        if (index != 0) try std.testing.expect(addresses[index - 1] != address);
    }
    try std.testing.expect(heap.stats().tlab_refills >= thread_count);
}

test "thread allocator rejects cross-thread use before touching local cursor" {
    var storage: [128]u8 align(runtime_value.object_alignment) = undefined;
    const regions = [_]runtime_value.Region{try runtime_value.Region.fromSlice(&storage)};
    var handles = try HandleTable.init(std.testing.allocator, 1, &regions);
    defer handles.deinit();
    var heap = try ManagedHeap.init(std.testing.allocator, &handles, 64);
    defer heap.deinit();
    var tlab = heap.threadAllocator();
    var rejected = std.atomic.Value(bool).init(false);

    const Worker = struct {
        tlab: *ThreadAllocator,
        rejected: *std.atomic.Value(bool),

        fn run(self: *@This()) void {
            _ = self.tlab.allocate(8, 8) catch |err| {
                self.rejected.store(err == error.WrongThread, .release);
                return;
            };
        }
    };
    var worker = Worker{ .tlab = &tlab, .rejected = &rejected };
    const thread = try std.Thread.spawn(.{}, Worker.run, .{&worker});
    thread.join();
    try std.testing.expect(rejected.load(.acquire));
}

test "reclamation reset invalidates stale tlabs and clears side metadata" {
    var source: [256]u8 align(runtime_value.object_alignment) = @splat(0);
    var destination: [256]u8 align(runtime_value.object_alignment) = @splat(0);
    const regions = [_]runtime_value.Region{
        try runtime_value.Region.fromSlice(&source),
        try runtime_value.Region.fromSlice(&destination),
    };
    var handles = try HandleTable.init(std.testing.allocator, 4, &regions);
    defer handles.deinit();
    var heap = try ManagedHeap.init(std.testing.allocator, &handles, 64);
    defer heap.deinit();
    var tlab = heap.threadAllocator();
    const old_reservation = try tlab.allocate(24, 8);
    const old_body: [*]u8 = @ptrCast(old_reservation.address());
    @memset(old_body[0..old_reservation.allocated_size], 0x7b);
    const handle = try handles.reserve(0, 0);
    try heap.publishObject(old_reservation, handle);

    var registry = try Registry.init(std.testing.allocator, std.testing.io, 0);
    defer registry.deinit() catch unreachable;
    var coordinator = try EvacuationCoordinator.init(std.testing.allocator, &handles, &registry, 0);
    defer coordinator.deinit() catch unreachable;
    var cycle = try coordinator.begin(0);
    const ticket = try handles.beginRelocation(handle);
    try std.testing.expect(try cycle.commitRelocation(ticket, 1, @ptrCast(&destination[8])));
    _ = try cycle.requestRetirement();
    try std.testing.expect(try cycle.advance());
    const reclaimed = try cycle.reclaim();
    try std.testing.expectEqual(runtime_value.RegionState.reclaiming, try handles.regionState(0));

    // The old local extent is unusable while resetting; allocation refills in
    // another active region instead of touching reclaimable bytes.
    const during_reset = try tlab.allocate(8, 8);
    try std.testing.expectEqual(@as(u8, 1), during_reset.region_id);
    try heap.resetReclaimedRegion(reclaimed);
    try std.testing.expectEqual(runtime_value.RegionState.active, try handles.regionState(0));
    try std.testing.expect(!(try heap.isObjectStart(0, old_reservation.offset)));
    for (source) |byte| try std.testing.expectEqual(@as(u8, 0), byte);

    const stale_handle = try handles.reserve(0, 0);
    try std.testing.expectError(error.StaleReservation, heap.publishObject(old_reservation, stale_handle));
    try handles.cancelReservation(stale_handle);
    var fresh_tlab = heap.threadAllocator();
    const fresh = try fresh_tlab.allocate(16, 8);
    try std.testing.expectEqual(@as(u8, 0), fresh.region_id);
    try std.testing.expectEqual(object_header_size, fresh.offset);
    try std.testing.expectEqual(@as(u64, 1), heap.stats().regions_reset);
}

fn allocationFailureProbe(allocator: std.mem.Allocator) !void {
    var storage: [64]u8 align(runtime_value.object_alignment) = undefined;
    const regions = [_]runtime_value.Region{try runtime_value.Region.fromSlice(&storage)};
    var handles = try HandleTable.init(std.testing.allocator, 1, &regions);
    defer handles.deinit();
    var registry = try Registry.init(std.testing.allocator, std.testing.io, 1);
    defer registry.deinit() catch unreachable;
    var coordinator = try EvacuationCoordinator.init(allocator, &handles, &registry, 1);
    defer coordinator.deinit() catch unreachable;
}

fn heapAllocationFailureProbe(allocator: std.mem.Allocator) !void {
    var storage: [256]u8 align(runtime_value.object_alignment) = undefined;
    const regions = [_]runtime_value.Region{try runtime_value.Region.fromSlice(&storage)};
    var handles = try HandleTable.init(std.testing.allocator, 1, &regions);
    defer handles.deinit();
    var heap = try ManagedHeap.init(allocator, &handles, 64);
    defer heap.deinit();
}

test "evacuation coordinator initialization is leak-free at every allocation failure" {
    try std.testing.checkAllAllocationFailures(std.testing.allocator, allocationFailureProbe, .{});
    try std.testing.checkAllAllocationFailures(std.testing.allocator, heapAllocationFailureProbe, .{});
}
