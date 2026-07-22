//! Bounded object monitors keyed directly by stable Handle-table index.
//!
//! Association is lock-free and collision-free. Per-object contention uses an
//! uncancelable FIFO ticket wait under one condition variable. A contended
//! mutator publishes roots and enters the registry's blocked state before it
//! waits, allowing collector handshakes to snapshot it passively.

const std = @import("std");
const runtime_gc = @import("runtime_gc");
const thread_registry = @import("runtime_thread_registry");
const runtime_value = @import("runtime_value");

const Handle = runtime_value.Handle;
const HandleTable = runtime_value.HandleTable;

pub const Error = runtime_gc.Error || error{
    ActiveMonitors,
    IllegalMonitorState,
    MonitorBusy,
    RecursionOverflow,
    TicketOverflow,
};

pub const Stats = struct {
    enters: u64,
    reentrant_enters: u64,
    contended_enters: u64,
    exits: u64,
    associations: u64,
    disassociations: u64,
    wait_wakeups: u64,
};

const AtomicStats = struct {
    enters: std.atomic.Value(u64) = std.atomic.Value(u64).init(0),
    reentrant_enters: std.atomic.Value(u64) = std.atomic.Value(u64).init(0),
    contended_enters: std.atomic.Value(u64) = std.atomic.Value(u64).init(0),
    exits: std.atomic.Value(u64) = std.atomic.Value(u64).init(0),
    associations: std.atomic.Value(u64) = std.atomic.Value(u64).init(0),
    disassociations: std.atomic.Value(u64) = std.atomic.Value(u64).init(0),
    wait_wakeups: std.atomic.Value(u64) = std.atomic.Value(u64).init(0),
};

const Slot = struct {
    key: std.atomic.Value(u64),
    /// A matching key is not consumable until its static-root insertion
    /// barrier has completed. This closes the publication window between the
    /// key CAS and an active marker discovering the new root.
    ready: std.atomic.Value(bool),
    users: std.atomic.Value(usize),
    mutex: std.Io.Mutex,
    condition: std.Io.Condition,
    owner: ?std.Thread.Id,
    recursion: u32,
    waiters: u32,
    next_ticket: u64,
    serving_ticket: u64,
};

const Lease = struct {
    table: *MonitorTable,
    slot: *Slot,
    bits: u64,
    collector: *runtime_gc.ConcurrentCollector,
    satb: *runtime_gc.SatbBuffer,

    fn release(self: *Lease) void {
        const table = self.table;
        const slot = self.slot;
        slot.mutex.lockUncancelable(table.io);
        if (slot.owner == null and slot.waiters == 0 and slot.users.load(.acquire) == 1) {
            const was_ready = slot.ready.swap(false, .acq_rel);
            // A racing prospective lessee may have observed `ready` before
            // the gate closed. Its user credit makes removal ineligible; its
            // validation below will either join this association after the
            // gate reopens or discard that credit.
            if (slot.users.load(.acquire) == 1) {
                if (table.preClear(self.collector, self.satb, @intFromPtr(&slot.key))) {
                    if (slot.key.cmpxchgStrong(self.bits, @bitCast(Handle.none), .release, .acquire) == null) {
                        _ = table.counters.disassociations.fetchAdd(1, .monotonic);
                    } else {
                        slot.ready.store(was_ready, .release);
                    }
                } else {
                    slot.ready.store(was_ready, .release);
                }
            } else {
                slot.ready.store(was_ready, .release);
            }
        }
        slot.mutex.unlock(table.io);
        const previous = slot.users.fetchSub(1, .release);
        std.debug.assert(previous != 0);
    }
};

pub const MonitorTable = struct {
    allocator: std.mem.Allocator,
    io: std.Io,
    handles: *HandleTable,
    slots: []Slot,
    root_slots: []usize,
    counters: AtomicStats = .{},

    pub fn init(allocator: std.mem.Allocator, io: std.Io, handles: *HandleTable) !MonitorTable {
        const count: usize = handles.entryCapacity();
        const slots = try allocator.alloc(Slot, count);
        errdefer allocator.free(slots);
        const root_slots = try allocator.alloc(usize, count);
        errdefer allocator.free(root_slots);
        for (slots, root_slots) |*slot, *root_address| {
            slot.* = .{
                .key = std.atomic.Value(u64).init(@bitCast(Handle.none)),
                .ready = std.atomic.Value(bool).init(false),
                .users = std.atomic.Value(usize).init(0),
                .mutex = .init,
                .condition = .init,
                .owner = null,
                .recursion = 0,
                .waiters = 0,
                .next_ticket = 0,
                .serving_ticket = 0,
            };
            root_address.* = @intFromPtr(&slot.key);
        }
        return .{
            .allocator = allocator,
            .io = io,
            .handles = handles,
            .slots = slots,
            .root_slots = root_slots,
        };
    }

    pub fn deinit(self: *MonitorTable) Error!void {
        for (self.slots) |*slot| {
            if (slot.key.load(.acquire) != @as(u64, @bitCast(Handle.none)) or
                slot.ready.load(.acquire) or
                slot.users.load(.acquire) != 0 or slot.owner != null or slot.waiters != 0)
            {
                return error.ActiveMonitors;
            }
        }
        self.allocator.free(self.root_slots);
        self.allocator.free(self.slots);
        self.* = undefined;
    }

    pub fn rootSlotAddresses(self: *const MonitorTable) []const usize {
        return self.root_slots;
    }

    pub fn handleTable(self: *const MonitorTable) *HandleTable {
        return self.handles;
    }

    fn preClear(
        self: *MonitorTable,
        collector: *runtime_gc.ConcurrentCollector,
        satb: *runtime_gc.SatbBuffer,
        slot_address: usize,
    ) bool {
        _ = self;
        for (0..1024) |_| {
            collector.referenceStorePreWrite(satb, slot_address, false) catch |err| switch (err) {
                error.SatbQueueFull => {
                    _ = collector.drainSatb(1) catch {
                        std.atomic.spinLoopHint();
                        continue;
                    };
                    continue;
                },
                error.RetryBarrier => {
                    std.atomic.spinLoopHint();
                    continue;
                },
                else => return false,
            };
            return true;
        }
        return false;
    }

    fn postAssociate(
        self: *MonitorTable,
        collector: *runtime_gc.ConcurrentCollector,
        slot_address: usize,
        handle: Handle,
    ) bool {
        _ = self;
        for (0..1024) |_| {
            collector.referenceStaticStorePostWrite(slot_address, handle) catch |err| switch (err) {
                error.RetryBarrier => {
                    std.atomic.spinLoopHint();
                    continue;
                },
                else => return false,
            };
            return true;
        }
        return false;
    }

    fn lease(
        self: *MonitorTable,
        handle: Handle,
        collector: *runtime_gc.ConcurrentCollector,
        satb: *runtime_gc.SatbBuffer,
    ) Error!Lease {
        if (handle.isNull() or handle.index >= self.slots.len) return error.InvalidHandle;
        const location = try self.handles.inspect(handle);
        switch (location.state) {
            .live, .evacuating => {},
            else => return error.InvalidHandle,
        }
        const slot = &self.slots[handle.index];
        const bits: u64 = @bitCast(handle);
        for (0..1024) |_| {
            const key = slot.key.load(.acquire);
            if (key == @as(u64, @bitCast(Handle.none))) {
                if (slot.key.cmpxchgStrong(key, bits, .acq_rel, .acquire) != null) continue;
                slot.ready.store(false, .monotonic);
                _ = slot.users.fetchAdd(1, .acquire);
                _ = self.counters.associations.fetchAdd(1, .monotonic);
                if (!self.postAssociate(collector, @intFromPtr(&slot.key), handle)) {
                    var failed = Lease{
                        .table = self,
                        .slot = slot,
                        .bits = bits,
                        .collector = collector,
                        .satb = satb,
                    };
                    failed.release();
                    return error.RetryBarrier;
                }
                slot.ready.store(true, .release);
                return .{
                    .table = self,
                    .slot = slot,
                    .bits = bits,
                    .collector = collector,
                    .satb = satb,
                };
            } else if (key != bits or !slot.ready.load(.acquire)) {
                std.atomic.spinLoopHint();
                continue;
            }
            _ = slot.users.fetchAdd(1, .acquire);
            if (slot.key.load(.acquire) == bits and slot.ready.load(.acquire)) return .{
                .table = self,
                .slot = slot,
                .bits = bits,
                .collector = collector,
                .satb = satb,
            };
            _ = slot.users.fetchSub(1, .release);
        }
        return error.MonitorBusy;
    }

    pub fn enter(
        self: *MonitorTable,
        handle: Handle,
        collector: *runtime_gc.ConcurrentCollector,
        registry: *thread_registry.Registry,
        context: *thread_registry.ThreadContext,
        satb: *runtime_gc.SatbBuffer,
    ) Error!void {
        var association = try self.lease(handle, collector, satb);
        defer association.release();
        const slot = association.slot;
        const thread_id = std.Thread.getCurrentId();
        slot.mutex.lockUncancelable(self.io);
        defer slot.mutex.unlock(self.io);

        if (slot.owner == thread_id) {
            if (slot.recursion == std.math.maxInt(u32)) return error.RecursionOverflow;
            slot.recursion += 1;
            _ = self.counters.reentrant_enters.fetchAdd(1, .monotonic);
            _ = self.counters.enters.fetchAdd(1, .monotonic);
            return;
        }
        if (slot.owner == null and slot.waiters == 0) {
            slot.owner = thread_id;
            slot.recursion = 1;
            _ = self.counters.enters.fetchAdd(1, .monotonic);
            return;
        }
        if (slot.next_ticket == std.math.maxInt(u64)) return error.TicketOverflow;

        // Roots were refreshed by the interpreter immediately before this
        // callback. Publish them as passive before any condition wait.
        try collector.enterBlockedForMark(registry, context, satb);
        const ticket = slot.next_ticket;
        slot.next_ticket += 1;
        slot.waiters += 1;
        _ = self.counters.contended_enters.fetchAdd(1, .monotonic);
        while (slot.owner != null or ticket != slot.serving_ticket) {
            slot.condition.waitUncancelable(self.io, &slot.mutex);
            _ = self.counters.wait_wakeups.fetchAdd(1, .monotonic);
        }
        slot.waiters -= 1;
        slot.serving_ticket += 1;
        slot.owner = thread_id;
        slot.recursion = 1;
        registry.leaveBlocked(context) catch |err| {
            slot.owner = null;
            slot.recursion = 0;
            slot.condition.broadcast(self.io);
            return err;
        };
        _ = self.counters.enters.fetchAdd(1, .monotonic);
    }

    pub fn exit(
        self: *MonitorTable,
        handle: Handle,
        collector: *runtime_gc.ConcurrentCollector,
        satb: *runtime_gc.SatbBuffer,
    ) Error!void {
        var association = try self.lease(handle, collector, satb);
        defer association.release();
        const slot = association.slot;
        slot.mutex.lockUncancelable(self.io);
        defer slot.mutex.unlock(self.io);
        if (slot.owner == null or slot.owner.? != std.Thread.getCurrentId() or slot.recursion == 0) {
            return error.IllegalMonitorState;
        }
        slot.recursion -= 1;
        _ = self.counters.exits.fetchAdd(1, .monotonic);
        if (slot.recursion != 0) return;
        slot.owner = null;
        slot.condition.broadcast(self.io);
    }

    pub fn stats(self: *const MonitorTable) Stats {
        return .{
            .enters = self.counters.enters.load(.acquire),
            .reentrant_enters = self.counters.reentrant_enters.load(.acquire),
            .contended_enters = self.counters.contended_enters.load(.acquire),
            .exits = self.counters.exits.load(.acquire),
            .associations = self.counters.associations.load(.acquire),
            .disassociations = self.counters.disassociations.load(.acquire),
            .wait_wakeups = self.counters.wait_wakeups.load(.acquire),
        };
    }
};

fn allocationFailureProbe(allocator: std.mem.Allocator) !void {
    var storage: [64]u8 align(runtime_value.object_alignment) = @splat(0);
    const regions = [_]runtime_value.Region{try runtime_value.Region.fromSlice(&storage)};
    var handles = try HandleTable.init(allocator, 4, &regions);
    defer handles.deinit();
    var table = try MonitorTable.init(allocator, std.testing.io, &handles);
    try table.deinit();
}

test "monitor root slots are stable sorted and initialization is failure atomic" {
    var storage: [64]u8 align(runtime_value.object_alignment) = @splat(0);
    const regions = [_]runtime_value.Region{try runtime_value.Region.fromSlice(&storage)};
    var handles = try HandleTable.init(std.testing.allocator, 4, &regions);
    defer handles.deinit();
    var table = try MonitorTable.init(std.testing.allocator, std.testing.io, &handles);
    for (table.rootSlotAddresses(), 0..) |address, index| {
        try std.testing.expect(std.mem.isAligned(address, @alignOf(std.atomic.Value(u64))));
        if (index != 0) try std.testing.expect(table.rootSlotAddresses()[index - 1] < address);
    }
    table.slots[0].key.store(0, .release);
    table.slots[0].ready.store(true, .release);
    try std.testing.expectError(error.ActiveMonitors, table.deinit());
    table.slots[0].ready.store(false, .release);
    table.slots[0].key.store(@bitCast(Handle.none), .release);
    try table.deinit();

    try std.testing.checkAllAllocationFailures(std.testing.allocator, allocationFailureProbe, .{});
}
