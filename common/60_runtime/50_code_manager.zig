//! Lock-free managed-code dispatch with quiescent epoch reclamation.
//!
//! Compiler threads prepare immutable W^X candidates off-path. Publication and
//! invalidation hold one short non-nested spin guard only to order the method
//! entry swap with the global code epoch. Managed readers never acquire it:
//! they publish an owner-confined reader lease, verify epoch stability, and
//! load the immutable current version. Reclamation scans fixed reader metadata
//! and never waits for a thread.

const std = @import("std");
const builtin = @import("builtin");
const jit_memory = @import("jit_memory");

pub const Error = jit_memory.Error || std.mem.Allocator.Error || error{
    ActiveCandidate,
    ActiveLease,
    CorruptState,
    EpochExhausted,
    InvalidCandidate,
    InvalidCapacity,
    InvalidMethod,
    NoCode,
    NoFreeReader,
    NoFreeVersion,
    ReadersRegistered,
    StaleReader,
};

const ReaderState = enum(u8) {
    free,
    initializing,
    registered,
};

const VersionState = enum(u8) {
    free,
    constructing,
    candidate,
    live,
    retired,
    reclaiming,
};

const ReaderSlot = struct {
    state: std.atomic.Value(ReaderState) = std.atomic.Value(ReaderState).init(.free),
    generation: std.atomic.Value(u32) = std.atomic.Value(u32).init(0),
    active: std.atomic.Value(bool) = std.atomic.Value(bool).init(false),
    observed_epoch: std.atomic.Value(u64) = std.atomic.Value(u64).init(0),
};

const MethodSlot = struct {
    current: std.atomic.Value(usize) = std.atomic.Value(usize).init(0),
};

const VersionSlot = struct {
    state: std.atomic.Value(VersionState) = std.atomic.Value(VersionState).init(.free),
    generation: std.atomic.Value(u32) = std.atomic.Value(u32).init(0),
    allocation: ?*jit_memory.Allocation = null,
    entry_address: usize = 0,
    publish_epoch: u64 = 0,
    retire_epoch: u64 = 0,
    metadata: ?Metadata = null,
};

/// Immutable metadata retained for exactly the lifetime of one code version.
/// Callbacks must be allocation-free, non-blocking, and thread-safe. Pointer
/// fields are opaque here so the code manager remains independent of compiler
/// and runtime module types.
pub const Metadata = struct {
    context: *anyopaque,
    stack_maps: usize = 0,
    deopt_table: usize = 0,
    osr_entries: usize = 0,
    osr_entry_count: u32 = 0,
    retain: *const fn (*anyopaque) void,
    release: *const fn (*anyopaque) void,

    fn acquire(self: Metadata) void {
        self.retain(self.context);
    }

    fn drop(self: Metadata) void {
        self.release(self.context);
    }
};

pub const Stats = struct {
    epoch: u64,
    prepared: u64,
    published: u64,
    invalidated: u64,
    retired: u64,
    reclaimed: u64,
    active_leases: u64,
    executable_versions: u32,
    metadata_versions: u32,
};

pub const MethodToken = struct {
    raw: usize,
};

const Counters = struct {
    prepared: std.atomic.Value(u64) = std.atomic.Value(u64).init(0),
    published: std.atomic.Value(u64) = std.atomic.Value(u64).init(0),
    invalidated: std.atomic.Value(u64) = std.atomic.Value(u64).init(0),
    retired: std.atomic.Value(u64) = std.atomic.Value(u64).init(0),
    reclaimed: std.atomic.Value(u64) = std.atomic.Value(u64).init(0),
    active_leases: std.atomic.Value(u64) = std.atomic.Value(u64).init(0),
    metadata_versions: std.atomic.Value(u32) = std.atomic.Value(u32).init(0),
};

pub const Manager = struct {
    allocator: std.mem.Allocator,
    methods: []MethodSlot,
    readers: []ReaderSlot,
    versions: []VersionSlot,
    code_epoch: std.atomic.Value(u64) = std.atomic.Value(u64).init(1),
    publication_guard: std.atomic.Value(bool) = std.atomic.Value(bool).init(false),
    cache_guard: std.atomic.Value(bool) = std.atomic.Value(bool).init(false),
    cache: jit_memory.Cache,
    counters: Counters = .{},

    pub fn init(
        allocator: std.mem.Allocator,
        method_capacity: usize,
        reader_capacity: usize,
        version_capacity: usize,
    ) Error!Manager {
        if (method_capacity == 0 or reader_capacity == 0 or version_capacity == 0 or
            method_capacity > std.math.maxInt(u32) or reader_capacity > std.math.maxInt(u32) or
            version_capacity > std.math.maxInt(u32)) return error.InvalidCapacity;

        const methods = try allocator.alloc(MethodSlot, method_capacity);
        errdefer allocator.free(methods);
        const readers = try allocator.alloc(ReaderSlot, reader_capacity);
        errdefer allocator.free(readers);
        const versions = try allocator.alloc(VersionSlot, version_capacity);
        errdefer allocator.free(versions);
        for (methods) |*slot| slot.* = .{};
        for (readers) |*slot| slot.* = .{};
        for (versions) |*slot| slot.* = .{};
        return .{
            .allocator = allocator,
            .methods = methods,
            .readers = readers,
            .versions = versions,
            .cache = jit_memory.Cache.init(allocator),
        };
    }

    pub fn deinit(self: *Manager) Error!void {
        for (self.readers) |*reader| {
            if (reader.state.load(.acquire) != .free) return error.ReadersRegistered;
            if (reader.active.load(.acquire)) return error.ActiveLease;
        }
        for (self.versions) |*version| {
            const state = version.state.load(.acquire);
            if (state == .constructing or state == .candidate) return error.ActiveCandidate;
        }
        for (self.versions) |*version| {
            if (version.metadata) |metadata| {
                metadata.drop();
                _ = self.counters.metadata_versions.fetchSub(1, .release);
            }
            version.metadata = null;
        }
        self.cache.deinit();
        self.allocator.free(self.versions);
        self.allocator.free(self.readers);
        self.allocator.free(self.methods);
        self.* = undefined;
    }

    pub fn prepare(self: *Manager, code: []const u8) Error!Candidate {
        return self.prepareWithMetadata(code, null);
    }

    pub fn prepareWithMetadata(self: *Manager, code: []const u8, metadata: ?Metadata) Error!Candidate {
        const claimed = self.claimVersion() orelse return error.NoFreeVersion;
        const slot = &self.versions[claimed.index];
        const allocation = blk: {
            lock(&self.cache_guard);
            defer unlock(&self.cache_guard);
            break :blk self.cache.addBytes(code) catch |err| {
                slot.state.store(.free, .release);
                return err;
            };
        };
        slot.allocation = allocation;
        slot.entry_address = allocation.entryAddress();
        slot.publish_epoch = 0;
        slot.retire_epoch = 0;
        if (metadata) |value| {
            value.acquire();
            _ = self.counters.metadata_versions.fetchAdd(1, .monotonic);
        }
        slot.metadata = metadata;
        slot.state.store(.candidate, .release);
        _ = self.counters.prepared.fetchAdd(1, .monotonic);
        return .{
            .manager = self,
            .index = claimed.index,
            .generation = claimed.generation,
        };
    }

    pub fn publish(self: *Manager, method: u32, candidate: *Candidate) Error!void {
        if (method >= self.methods.len) return error.InvalidMethod;
        const version = try self.candidateSlot(candidate);
        lock(&self.publication_guard);
        defer unlock(&self.publication_guard);

        _ = try self.publishLocked(method, candidate, version);
    }

    pub fn snapshot(self: *const Manager, method: u32) Error!MethodToken {
        if (method >= self.methods.len) return error.InvalidMethod;
        return .{ .raw = self.methods[method].current.load(.acquire) };
    }

    /// Publish only if the method still names the version observed before
    /// compilation. A losing compiler retains candidate ownership and can
    /// cancel it without ever exposing its executable page to dispatch.
    pub fn publishIfCurrent(self: *Manager, method: u32, expected: MethodToken, candidate: *Candidate) Error!bool {
        if (method >= self.methods.len) return error.InvalidMethod;
        const version = try self.candidateSlot(candidate);
        lock(&self.publication_guard);
        defer unlock(&self.publication_guard);
        if (self.methods[method].current.load(.acquire) != expected.raw) return false;
        _ = try self.publishLocked(method, candidate, version);
        return true;
    }

    fn publishLocked(self: *Manager, method: u32, candidate: *Candidate, version: *VersionSlot) Error!bool {
        const old_address = self.methods[method].current.load(.acquire);
        const epoch = self.code_epoch.load(.acquire);
        if (old_address != 0 and epoch == std.math.maxInt(u64)) return error.EpochExhausted;

        version.publish_epoch = epoch;
        version.state.store(.live, .release);
        const old = self.methods[method].current.swap(@intFromPtr(version), .acq_rel);
        candidate.active = false;
        _ = self.counters.published.fetchAdd(1, .monotonic);
        if (old != 0) self.retireSwapped(@ptrFromInt(old), epoch + 1);
        return true;
    }

    pub fn invalidate(self: *Manager, method: u32) Error!bool {
        if (method >= self.methods.len) return error.InvalidMethod;
        lock(&self.publication_guard);
        defer unlock(&self.publication_guard);
        const old_address = self.methods[method].current.load(.acquire);
        if (old_address == 0) return false;
        const epoch = self.code_epoch.load(.acquire);
        if (epoch == std.math.maxInt(u64)) return error.EpochExhausted;
        const old = self.methods[method].current.swap(0, .acq_rel);
        if (old == 0) return false;
        self.retireSwapped(@ptrFromInt(old), epoch + 1);
        _ = self.counters.invalidated.fetchAdd(1, .monotonic);
        return true;
    }

    pub fn registerReader(self: *Manager) Error!Reader {
        for (self.readers, 0..) |*slot, index| {
            if (slot.state.cmpxchgStrong(.free, .initializing, .acq_rel, .acquire) != null) continue;
            const generation = slot.generation.load(.monotonic);
            if (generation == std.math.maxInt(u32)) {
                slot.state.store(.free, .release);
                continue;
            }
            const next = generation + 1;
            slot.generation.store(next, .monotonic);
            slot.active.store(false, .monotonic);
            slot.observed_epoch.store(self.code_epoch.load(.acquire), .monotonic);
            slot.state.store(.registered, .release);
            return .{ .manager = self, .index = @intCast(index), .generation = next };
        }
        return error.NoFreeReader;
    }

    pub fn unregisterReader(self: *Manager, reader: *Reader) Error!void {
        const slot = try self.readerSlot(reader);
        if (slot.active.load(.acquire)) return error.ActiveLease;
        if (slot.state.cmpxchgStrong(.registered, .initializing, .acq_rel, .acquire) != null) return error.StaleReader;
        slot.state.store(.free, .release);
        reader.active = false;
    }

    pub fn enter(self: *Manager, reader: *Reader, method: u32) Error!Lease {
        if (method >= self.methods.len) return error.InvalidMethod;
        const slot = try self.readerSlot(reader);
        if (slot.active.cmpxchgStrong(false, true, .acq_rel, .acquire) != null) return error.ActiveLease;
        while (true) {
            const epoch = self.code_epoch.load(.acquire);
            slot.observed_epoch.store(epoch, .release);
            if (self.code_epoch.load(.acquire) != epoch) continue;
            const address = self.methods[method].current.load(.acquire);
            if (address == 0) {
                slot.active.store(false, .release);
                return error.NoCode;
            }
            const version: *VersionSlot = @ptrFromInt(address);
            const state = version.state.load(.acquire);
            if (state != .live and state != .retired) {
                slot.active.store(false, .release);
                return error.NoCode;
            }
            _ = self.counters.active_leases.fetchAdd(1, .monotonic);
            return .{
                .manager = self,
                .reader_index = reader.index,
                .reader_generation = reader.generation,
                .version = version,
            };
        }
    }

    /// Reclaim every currently quiescent retired version. This operation never
    /// waits for a reader; unsafe versions remain RX for a later scan.
    pub fn reclaim(self: *Manager) Error!u32 {
        var reclaimed: u32 = 0;
        for (self.versions) |*version| {
            if (version.state.load(.acquire) != .retired) continue;
            if (!self.epochIsQuiescent(version.retire_epoch)) continue;
            if (version.state.cmpxchgStrong(.retired, .reclaiming, .acq_rel, .acquire) != null) continue;
            const allocation = version.allocation orelse return error.InvalidCandidate;
            lock(&self.cache_guard);
            self.cache.release(allocation) catch |err| {
                unlock(&self.cache_guard);
                return err;
            };
            unlock(&self.cache_guard);
            version.allocation = null;
            version.entry_address = 0;
            version.publish_epoch = 0;
            version.retire_epoch = 0;
            if (version.metadata) |metadata| {
                metadata.drop();
                _ = self.counters.metadata_versions.fetchSub(1, .release);
            }
            version.metadata = null;
            version.state.store(.free, .release);
            reclaimed += 1;
            _ = self.counters.reclaimed.fetchAdd(1, .monotonic);
        }
        return reclaimed;
    }

    pub fn stats(self: *const Manager) Stats {
        return .{
            .epoch = self.code_epoch.load(.acquire),
            .prepared = self.counters.prepared.load(.acquire),
            .published = self.counters.published.load(.acquire),
            .invalidated = self.counters.invalidated.load(.acquire),
            .retired = self.counters.retired.load(.acquire),
            .reclaimed = self.counters.reclaimed.load(.acquire),
            .active_leases = self.counters.active_leases.load(.acquire),
            .executable_versions = self.cache.stats.functions,
            .metadata_versions = self.counters.metadata_versions.load(.acquire),
        };
    }

    /// Quiescent diagnostic verification. Callers must not mutate manager
    /// metadata concurrently with this scan.
    pub fn verify(self: *const Manager) Error!void {
        var allocation_count: u32 = 0;
        for (self.versions) |*version| {
            const state = version.state.load(.acquire);
            const owns_allocation = version.allocation != null;
            switch (state) {
                .free => if (owns_allocation or version.entry_address != 0 or version.retire_epoch != 0 or version.metadata != null) return error.CorruptState,
                .constructing => {},
                .candidate, .live, .retired, .reclaiming => if (!owns_allocation or version.entry_address == 0) return error.CorruptState,
            }
            if (state == .retired and version.retire_epoch == 0) return error.CorruptState;
            if (owns_allocation) allocation_count += 1;
        }
        for (self.methods) |*method| {
            const current = method.current.load(.acquire);
            if (current == 0) continue;
            var found = false;
            for (self.versions) |*version| {
                if (@intFromPtr(version) != current) continue;
                if (version.state.load(.acquire) != .live) return error.CorruptState;
                found = true;
                break;
            }
            if (!found) return error.CorruptState;
        }
        var active_readers: u64 = 0;
        for (self.readers) |*reader| {
            if (reader.active.load(.acquire)) active_readers += 1;
        }
        if (active_readers != self.counters.active_leases.load(.acquire) or
            allocation_count != self.cache.stats.functions) return error.CorruptState;
    }

    fn claimVersion(self: *Manager) ?struct { index: u32, generation: u32 } {
        for (self.versions, 0..) |*slot, index| {
            if (slot.state.cmpxchgStrong(.free, .constructing, .acq_rel, .acquire) != null) continue;
            const generation = slot.generation.load(.monotonic);
            if (generation == std.math.maxInt(u32)) {
                slot.state.store(.free, .release);
                continue;
            }
            const next = generation + 1;
            slot.generation.store(next, .monotonic);
            return .{ .index = @intCast(index), .generation = next };
        }
        return null;
    }

    fn candidateSlot(self: *Manager, candidate: *Candidate) Error!*VersionSlot {
        if (!candidate.active or candidate.manager != self or candidate.index >= self.versions.len) return error.InvalidCandidate;
        const slot = &self.versions[candidate.index];
        if (slot.generation.load(.acquire) != candidate.generation or slot.state.load(.acquire) != .candidate) return error.InvalidCandidate;
        return slot;
    }

    fn cancelCandidate(self: *Manager, candidate: *Candidate) Error!void {
        const slot = try self.candidateSlot(candidate);
        if (slot.state.cmpxchgStrong(.candidate, .reclaiming, .acq_rel, .acquire) != null) return error.InvalidCandidate;
        const allocation = slot.allocation orelse return error.InvalidCandidate;
        lock(&self.cache_guard);
        self.cache.release(allocation) catch |err| {
            unlock(&self.cache_guard);
            return err;
        };
        unlock(&self.cache_guard);
        slot.allocation = null;
        slot.entry_address = 0;
        if (slot.metadata) |metadata| {
            metadata.drop();
            _ = self.counters.metadata_versions.fetchSub(1, .release);
        }
        slot.metadata = null;
        slot.state.store(.free, .release);
        candidate.active = false;
    }

    fn readerSlot(self: *Manager, reader: *const Reader) Error!*ReaderSlot {
        if (!reader.active or reader.manager != self or reader.index >= self.readers.len) return error.StaleReader;
        const slot = &self.readers[reader.index];
        if (slot.state.load(.acquire) != .registered or slot.generation.load(.acquire) != reader.generation) return error.StaleReader;
        return slot;
    }

    fn retireSwapped(self: *Manager, old: *VersionSlot, retire_epoch: u64) void {
        self.code_epoch.store(retire_epoch, .release);
        old.retire_epoch = retire_epoch;
        old.state.store(.retired, .release);
        _ = self.counters.retired.fetchAdd(1, .monotonic);
    }

    fn epochIsQuiescent(self: *const Manager, retire_epoch: u64) bool {
        for (self.readers) |*reader| {
            if (reader.state.load(.acquire) != .registered) continue;
            if (!reader.active.load(.acquire)) continue;
            if (reader.observed_epoch.load(.acquire) < retire_epoch) return false;
        }
        return true;
    }
};

pub const Candidate = struct {
    manager: *Manager,
    index: u32,
    generation: u32,
    active: bool = true,

    pub fn entryAddress(self: *const Candidate) Error!usize {
        const slot = try self.manager.candidateSlot(@constCast(self));
        return slot.entry_address;
    }

    pub fn cancel(self: *Candidate) Error!void {
        if (!self.active) return;
        try self.manager.cancelCandidate(self);
    }

    pub fn deinit(self: *Candidate) void {
        self.cancel() catch unreachable;
    }
};

pub const Reader = struct {
    manager: *Manager,
    index: u32,
    generation: u32,
    active: bool = true,

    pub fn deinit(self: *Reader) void {
        if (!self.active) return;
        self.manager.unregisterReader(self) catch unreachable;
    }
};

pub const Lease = struct {
    manager: *Manager,
    reader_index: u32,
    reader_generation: u32,
    version: *VersionSlot,
    active: bool = true,

    pub fn entryAddress(self: *const Lease) usize {
        return self.version.entry_address;
    }

    pub fn typedEntry(self: *const Lease, comptime Fn: type) *const Fn {
        return @ptrFromInt(self.entryAddress());
    }

    pub fn codeSize(self: *const Lease) u32 {
        const allocation = self.version.allocation orelse return 0;
        return @intCast(allocation.bytes().len);
    }

    pub fn metadata(self: *const Lease) ?Metadata {
        return self.version.metadata;
    }

    pub fn deinit(self: *Lease) void {
        if (!self.active) return;
        const reader = &self.manager.readers[self.reader_index];
        std.debug.assert(reader.generation.load(.acquire) == self.reader_generation);
        reader.active.store(false, .release);
        _ = self.manager.counters.active_leases.fetchSub(1, .release);
        self.active = false;
    }
};

fn lock(guard: *std.atomic.Value(bool)) void {
    var attempts: u32 = 0;
    while (guard.cmpxchgWeak(false, true, .acquire, .monotonic) != null) : (attempts +%= 1) {
        std.atomic.spinLoopHint();
        if ((attempts & 0xff) == 0) std.Thread.yield() catch {};
    }
}

fn unlock(guard: *std.atomic.Value(bool)) void {
    guard.store(false, .release);
}

fn returnI32(value: u32) [6]u8 {
    return .{ 0xb8, @truncate(value), @truncate(value >> 8), @truncate(value >> 16), @truncate(value >> 24), 0xc3 };
}

test "code manager publishes invalidates and reclaims only after quiescence" {
    if (builtin.cpu.arch != .x86_64) return error.SkipZigTest;
    var manager = try Manager.init(std.testing.allocator, 2, 2, 4);
    defer manager.deinit() catch unreachable;
    var reader = try manager.registerReader();
    defer reader.deinit();
    var observer = try manager.registerReader();
    defer observer.deinit();

    const first_code = returnI32(11);
    var first = try manager.prepare(&first_code);
    defer first.deinit();
    try manager.publish(0, &first);
    var old_lease = try manager.enter(&reader, 0);
    const Fn = fn () callconv(.c) u32;
    try std.testing.expectEqual(@as(u32, 11), old_lease.typedEntry(Fn)());

    const expected = try manager.snapshot(0);
    const second_code = returnI32(22);
    var second = try manager.prepare(&second_code);
    defer second.deinit();
    const loser_code = returnI32(33);
    var loser = try manager.prepare(&loser_code);
    defer loser.deinit();
    try std.testing.expect(try manager.publishIfCurrent(0, expected, &second));
    try std.testing.expect(!(try manager.publishIfCurrent(0, expected, &loser)));
    try loser.cancel();
    try std.testing.expectEqual(@as(u32, 0), try manager.reclaim());
    old_lease.deinit();
    try std.testing.expectEqual(@as(u32, 1), try manager.reclaim());

    var new_lease = try manager.enter(&reader, 0);
    try std.testing.expectEqual(@as(u32, 22), new_lease.typedEntry(Fn)());
    try std.testing.expect(try manager.invalidate(0));
    try std.testing.expectError(error.NoCode, manager.enter(&observer, 0));
    try std.testing.expectEqual(@as(u32, 0), try manager.reclaim());
    new_lease.deinit();
    try std.testing.expectEqual(@as(u32, 1), try manager.reclaim());
    try std.testing.expectEqual(@as(u32, 0), manager.stats().executable_versions);
    try std.testing.expectEqual(@as(u64, 2), manager.stats().retired);
    try std.testing.expectEqual(@as(u64, 2), manager.stats().reclaimed);
    try manager.verify();
}

test "code manager rejects active reader teardown and cancels candidates" {
    var manager = try Manager.init(std.testing.allocator, 1, 1, 2);
    defer manager.deinit() catch unreachable;
    var reader = try manager.registerReader();
    const code = returnI32(7);
    var candidate = try manager.prepare(&code);
    try std.testing.expect(try candidate.entryAddress() != 0);
    candidate.deinit();
    try std.testing.expectEqual(@as(u32, 0), manager.stats().executable_versions);
    try std.testing.expectError(error.ReadersRegistered, manager.deinit());
    reader.deinit();
}

test "code manager retains metadata through lease quiescence" {
    const Owner = struct {
        references: std.atomic.Value(u32) = std.atomic.Value(u32).init(1),

        fn retain(raw: *anyopaque) void {
            const self: *@This() = @ptrCast(@alignCast(raw));
            _ = self.references.fetchAdd(1, .acq_rel);
        }

        fn release(raw: *anyopaque) void {
            const self: *@This() = @ptrCast(@alignCast(raw));
            const previous = self.references.fetchSub(1, .acq_rel);
            std.debug.assert(previous > 1);
        }
    };
    var owner = Owner{};
    const metadata = Metadata{
        .context = &owner,
        .stack_maps = 0x1111,
        .deopt_table = 0x2222,
        .retain = Owner.retain,
        .release = Owner.release,
    };
    var manager = try Manager.init(std.testing.allocator, 1, 1, 2);
    defer manager.deinit() catch unreachable;
    var reader = try manager.registerReader();
    defer reader.deinit();
    const code = returnI32(9);
    var candidate = try manager.prepareWithMetadata(&code, metadata);
    defer candidate.deinit();
    try std.testing.expectEqual(@as(u32, 2), owner.references.load(.acquire));
    try manager.publish(0, &candidate);
    var lease = try manager.enter(&reader, 0);
    try std.testing.expectEqual(@as(usize, 0x1111), lease.metadata().?.stack_maps);
    try std.testing.expect(try manager.invalidate(0));
    try std.testing.expectEqual(@as(u32, 0), try manager.reclaim());
    try std.testing.expectEqual(@as(u32, 2), owner.references.load(.acquire));
    lease.deinit();
    try std.testing.expectEqual(@as(u32, 1), try manager.reclaim());
    try std.testing.expectEqual(@as(u32, 1), owner.references.load(.acquire));
    try std.testing.expectEqual(@as(u32, 0), manager.stats().metadata_versions);
}

test "code manager races lock-free dispatch against publication invalidation and reclamation" {
    if (builtin.cpu.arch != .x86_64) return error.SkipZigTest;
    const rounds: u32 = if (builtin.mode == .Debug) 64 else 256;
    var manager = try Manager.init(std.testing.allocator, 1, 1, rounds + 4);
    defer manager.deinit() catch unreachable;
    var reader = try manager.registerReader();

    const first_code = returnI32(0x1111);
    var first = try manager.prepare(&first_code);
    defer first.deinit();
    try manager.publish(0, &first);

    const Worker = struct {
        manager: *Manager,
        reader: *Reader,
        stop: *std.atomic.Value(bool),
        ready: *std.atomic.Value(bool),
        failed: *std.atomic.Value(bool),
        calls: *std.atomic.Value(u32),

        fn run(self: *@This()) void {
            const Fn = fn () callconv(.c) u32;
            self.ready.store(true, .release);
            while (!self.stop.load(.acquire)) {
                var lease = self.manager.enter(self.reader, 0) catch |err| switch (err) {
                    error.NoCode => {
                        std.atomic.spinLoopHint();
                        continue;
                    },
                    else => {
                        self.failed.store(true, .release);
                        return;
                    },
                };
                const value = lease.typedEntry(Fn)();
                if (value != 0x1111 and value != 0x2222) self.failed.store(true, .release);
                lease.deinit();
                _ = self.calls.fetchAdd(1, .release);
            }
        }
    };

    var stop = std.atomic.Value(bool).init(false);
    var ready = std.atomic.Value(bool).init(false);
    var failed = std.atomic.Value(bool).init(false);
    var calls = std.atomic.Value(u32).init(0);
    var worker = Worker{
        .manager = &manager,
        .reader = &reader,
        .stop = &stop,
        .ready = &ready,
        .failed = &failed,
        .calls = &calls,
    };
    const thread = try std.Thread.spawn(.{}, Worker.run, .{&worker});
    while (!ready.load(.acquire)) std.atomic.spinLoopHint();

    var round: u32 = 0;
    while (round < rounds) : (round += 1) {
        const code = returnI32(if ((round & 1) == 0) 0x2222 else 0x1111);
        var candidate = try manager.prepare(&code);
        defer candidate.deinit();
        const before = calls.load(.acquire);
        try manager.publish(0, &candidate);
        for (0..1_000_000) |attempt| {
            if (calls.load(.acquire) != before or failed.load(.acquire)) break;
            std.atomic.spinLoopHint();
            if ((attempt & 0xff) == 0) std.Thread.yield() catch {};
        } else return error.Timeout;
        if ((round % 7) == 6) _ = try manager.invalidate(0);
        _ = try manager.reclaim();
    }

    stop.store(true, .release);
    thread.join();
    try std.testing.expect(!failed.load(.acquire));
    try std.testing.expect(calls.load(.acquire) >= rounds);
    _ = try manager.invalidate(0);
    reader.deinit();
    while (try manager.reclaim() != 0) {}
    try std.testing.expectEqual(@as(u32, 0), manager.stats().executable_versions);
    try std.testing.expectEqual(@as(u64, 0), manager.stats().active_leases);
    try manager.verify();
}

fn allocationFailureProbe(allocator: std.mem.Allocator) !void {
    var manager = try Manager.init(allocator, 2, 2, 3);
    defer manager.deinit() catch unreachable;
    const code = returnI32(1);
    var candidate = try manager.prepare(&code);
    defer candidate.deinit();
}

const MetadataProbeOwner = struct {
    references: std.atomic.Value(u32) = std.atomic.Value(u32).init(1),

    fn retain(raw: *anyopaque) void {
        const self: *@This() = @ptrCast(@alignCast(raw));
        _ = self.references.fetchAdd(1, .acq_rel);
    }

    fn release(raw: *anyopaque) void {
        const self: *@This() = @ptrCast(@alignCast(raw));
        const previous = self.references.fetchSub(1, .acq_rel);
        std.debug.assert(previous > 1);
    }
};

fn metadataAllocationFailureProbe(allocator: std.mem.Allocator, owner: *MetadataProbeOwner) !void {
    var manager = try Manager.init(allocator, 1, 1, 1);
    defer manager.deinit() catch unreachable;
    const code = returnI32(3);
    var candidate = try manager.prepareWithMetadata(&code, .{
        .context = owner,
        .retain = MetadataProbeOwner.retain,
        .release = MetadataProbeOwner.release,
    });
    defer candidate.deinit();
    if (owner.references.load(.acquire) != 2) return error.TestUnexpectedResult;
}

test "code manager initialization and W^X preparation are allocation-failure safe" {
    try std.testing.checkAllAllocationFailures(std.testing.allocator, allocationFailureProbe, .{});
    var owner = MetadataProbeOwner{};
    try std.testing.checkAllAllocationFailures(std.testing.allocator, metadataAllocationFailureProbe, .{&owner});
    try std.testing.expectEqual(@as(u32, 1), owner.references.load(.acquire));
}
