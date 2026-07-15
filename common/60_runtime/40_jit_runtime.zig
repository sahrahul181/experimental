//! Managed JIT-entry leases and cooperative handle resolution.
//!
//! This module is the runtime side of the x86-64 reserved-register contract.
//! It owns the immutable region-base vector used by generated code, exposes a
//! register image without leaking private handle-table layout, and makes slow
//! resolution participate in thread root handshakes without allocation.

const std = @import("std");
const runtime_value = @import("runtime_value");
const runtime_stack_map = @import("runtime_stack_map");
const thread_registry = @import("runtime_thread_registry");
const runtime_gc = @import("runtime_gc");

const Handle = runtime_value.Handle;
const HandleTable = runtime_value.HandleTable;
const Registry = thread_registry.Registry;
const ThreadContext = thread_registry.ThreadContext;

pub const Error = runtime_value.Error || runtime_stack_map.Error || thread_registry.Error || runtime_gc.Error || std.mem.Allocator.Error || error{
    ActiveEntries,
    InactiveEntry,
    InvalidTableLayout,
    MissingCollector,
    ThreadNotRunning,
};

/// Values loaded into reserved registers by the architecture trampoline.
pub const RegisterImage = extern struct {
    r12_acknowledged_epoch: u64,
    r13_region_bases: usize,
    r14_descriptor_base: usize,
    r15_thread_state: usize,
    handle_capacity: u32,
    region_count: u16,
    descriptor_stride: u8,
    reserved: u8 = 0,
};

pub const SlowResolveStatus = enum(u32) {
    ok = 0,
    inactive_entry,
    thread_not_running,
    invalid_handle,
    stale_handle,
    retired,
    invalid_state,
    root_capacity,
    shutdown,
    runtime_error,
    missing_root_map,
    missing_canonical_root,
    missing_collector,
    gc_barrier_failure,
};

/// Pinned for the duration of a native call. r15 points here, allowing the
/// architecture adapter to find the correct runtime without global state.
pub const NativeThreadState = extern struct {
    runtime: *Runtime,
    context: *ThreadContext,
    acknowledged_epoch: u64,
    last_site_key: u64 = 0,
    last_error: SlowResolveStatus = .ok,
    active: u32 = 1,
    root_map_table: usize = 0,
    collector: usize = 0,
    satb_buffer: usize = 0,
    last_card_destination: u64 = 0,
    captured_gp: [16]u64 = @splat(0),
};

pub const Stats = struct {
    managed_entries: u64,
    active_entries: u64,
    slow_resolves: u64,
    handshake_polls: u64,
    resolve_failures: u64,
    last_slow_site: u64,
    pre_write_barriers: u64,
    post_write_barriers: u64,
    barrier_failures: u64,
};

const Counters = struct {
    managed_entries: std.atomic.Value(u64) = std.atomic.Value(u64).init(0),
    active_entries: std.atomic.Value(u64) = std.atomic.Value(u64).init(0),
    slow_resolves: std.atomic.Value(u64) = std.atomic.Value(u64).init(0),
    handshake_polls: std.atomic.Value(u64) = std.atomic.Value(u64).init(0),
    resolve_failures: std.atomic.Value(u64) = std.atomic.Value(u64).init(0),
    last_slow_site: std.atomic.Value(u64) = std.atomic.Value(u64).init(0),
    pre_write_barriers: std.atomic.Value(u64) = std.atomic.Value(u64).init(0),
    post_write_barriers: std.atomic.Value(u64) = std.atomic.Value(u64).init(0),
    barrier_failures: std.atomic.Value(u64) = std.atomic.Value(u64).init(0),
};

pub const Runtime = struct {
    allocator: std.mem.Allocator,
    handles: *HandleTable,
    registry: *Registry,
    region_bases: []usize,
    collector: ?*runtime_gc.ConcurrentCollector = null,
    counters: Counters = .{},

    pub fn init(
        allocator: std.mem.Allocator,
        handles: *HandleTable,
        registry: *Registry,
    ) Error!Runtime {
        if (handles.descriptorStride() != @sizeOf(u64)) return error.InvalidTableLayout;
        if (!std.mem.isAligned(handles.descriptorBaseAddress(), @alignOf(u64))) return error.InvalidTableLayout;

        const region_count = handles.regionCount();
        const region_bases = try allocator.alloc(usize, region_count);
        errdefer allocator.free(region_bases);
        for (region_bases, 0..) |*base, index| {
            base.* = (try handles.regionAt(@intCast(index))).base;
        }

        return .{
            .allocator = allocator,
            .handles = handles,
            .registry = registry,
            .region_bases = region_bases,
        };
    }

    pub fn deinit(self: *Runtime) Error!void {
        if (self.counters.active_entries.load(.acquire) != 0) return error.ActiveEntries;
        self.allocator.free(self.region_bases);
        self.* = undefined;
    }

    pub fn installCollector(self: *Runtime, collector: *runtime_gc.ConcurrentCollector) Error!void {
        if (self.counters.active_entries.load(.acquire) != 0) return error.ActiveEntries;
        if (collector.handleTable() != self.handles) return error.InvalidTableLayout;
        self.collector = collector;
    }

    /// Enters managed code after acknowledging any pending root handshake.
    /// The returned epoch is installed in r12 and remains the reclamation lease
    /// for derived pointers until the next compiler safepoint/poll.
    pub fn enter(self: *Runtime, context: *ThreadContext) Error!ManagedEntry {
        if (!context.isRunning()) return error.ThreadNotRunning;
        if (try self.registry.poll(context)) {
            _ = self.counters.handshake_polls.fetchAdd(1, .monotonic);
        }
        const collector_address: usize = if (self.collector) |collector| @intFromPtr(collector) else 0;
        const buffer_address: usize = if (self.collector) |collector|
            if (collector.bufferForThread(context)) |buffer| @intFromPtr(buffer) else return error.MissingSatbBuffer
        else
            0;
        // Validate every dependency before publishing an active native entry;
        // failed entry attempts must not strand Runtime.deinit forever.
        _ = self.counters.managed_entries.fetchAdd(1, .monotonic);
        _ = self.counters.active_entries.fetchAdd(1, .acq_rel);
        return .{
            .runtime = self,
            .context = context,
            .acknowledged_epoch = context.observedEpoch(),
            .native_state = .{
                .runtime = self,
                .context = context,
                .acknowledged_epoch = context.observedEpoch(),
                .collector = collector_address,
                .satb_buffer = buffer_address,
            },
        };
    }

    pub fn stats(self: *const Runtime) Stats {
        return .{
            .managed_entries = self.counters.managed_entries.load(.acquire),
            .active_entries = self.counters.active_entries.load(.acquire),
            .slow_resolves = self.counters.slow_resolves.load(.acquire),
            .handshake_polls = self.counters.handshake_polls.load(.acquire),
            .resolve_failures = self.counters.resolve_failures.load(.acquire),
            .last_slow_site = self.counters.last_slow_site.load(.acquire),
            .pre_write_barriers = self.counters.pre_write_barriers.load(.acquire),
            .post_write_barriers = self.counters.post_write_barriers.load(.acquire),
            .barrier_failures = self.counters.barrier_failures.load(.acquire),
        };
    }
};

pub const ManagedEntry = struct {
    runtime: *Runtime,
    context: *ThreadContext,
    acknowledged_epoch: u64,
    native_state: NativeThreadState,
    active: bool = true,

    pub fn deinit(self: *ManagedEntry) void {
        if (!self.active) return;
        _ = self.runtime.counters.active_entries.fetchSub(1, .release);
        self.native_state.active = 0;
        self.active = false;
    }

    pub fn refresh(self: *ManagedEntry) Error!bool {
        if (!self.active) return error.InactiveEntry;
        if (!self.context.isRunning()) return error.ThreadNotRunning;
        const participated = try self.runtime.registry.poll(self.context);
        if (participated) _ = self.runtime.counters.handshake_polls.fetchAdd(1, .monotonic);
        self.acknowledged_epoch = self.context.observedEpoch();
        self.native_state.acknowledged_epoch = self.acknowledged_epoch;
        return participated;
    }

    pub fn registerImage(self: *ManagedEntry) Error!RegisterImage {
        if (!self.active) return error.InactiveEntry;
        if (!self.context.isRunning()) return error.ThreadNotRunning;
        // A preserve-all slow helper may have acknowledged a handshake without
        // returning through `ManagedEntry.refresh`. Re-read the authoritative
        // context epoch so a subsequent native entry can never reinstall an
        // older from-space lease in r12.
        self.acknowledged_epoch = self.context.observedEpoch();
        self.native_state.runtime = self.runtime;
        self.native_state.context = self.context;
        self.native_state.acknowledged_epoch = self.acknowledged_epoch;
        self.native_state.active = 1;
        self.native_state.collector = if (self.runtime.collector) |collector| @intFromPtr(collector) else 0;
        self.native_state.satb_buffer = if (self.runtime.collector) |collector|
            if (collector.bufferForThread(self.context)) |buffer| @intFromPtr(buffer) else return error.MissingSatbBuffer
        else
            0;
        return .{
            .r12_acknowledged_epoch = self.acknowledged_epoch,
            .r13_region_bases = @intFromPtr(self.runtime.region_bases.ptr),
            .r14_descriptor_base = self.runtime.handles.descriptorBaseAddress(),
            .r15_thread_state = @intFromPtr(&self.native_state),
            .handle_capacity = self.runtime.handles.entryCapacity(),
            .region_count = self.runtime.handles.regionCount(),
            .descriptor_stride = self.runtime.handles.descriptorStride(),
        };
    }

    pub fn installRootMaps(self: *ManagedEntry, table: *const runtime_stack_map.Table) Error!void {
        if (!self.active) return error.InactiveEntry;
        // The compiled function owns this immutable table and must outlive the
        // managed native call. Publication happens before entry, so the helper
        // only performs lock-free reads.
        self.native_state.root_map_table = @intFromPtr(table);
    }

    /// Runtime target behind the architecture-specific preserve-all shim.
    /// `site_key` is `(dex_pc << 32) | resolve_id` and is retained for failure
    /// diagnostics. This operation allocates no memory and holds no mutex while
    /// resolving the descriptor.
    pub fn resolveSlow(self: *ManagedEntry, handle: Handle, site_key: u64) Error!*anyopaque {
        if (!self.active) return error.InactiveEntry;
        if (!self.context.isRunning()) return error.ThreadNotRunning;

        var rooted_handle = handle;
        var roots = try self.context.beginRootScope();
        defer roots.deinit();
        try roots.add(&rooted_handle);

        _ = self.runtime.counters.slow_resolves.fetchAdd(1, .monotonic);
        self.runtime.counters.last_slow_site.store(site_key, .release);
        _ = self.refresh() catch |err| {
            _ = self.runtime.counters.resolve_failures.fetchAdd(1, .monotonic);
            return err;
        };
        return self.runtime.handles.resolve(rooted_handle) catch |err| {
            _ = self.runtime.counters.resolve_failures.fetchAdd(1, .monotonic);
            return err;
        };
    }

    pub fn resolveSlowBits(self: *ManagedEntry, handle_bits: u64, site_key: u64) Error!*anyopaque {
        return self.resolveSlow(@bitCast(handle_bits), site_key);
    }
};

fn statusFor(err: Error) SlowResolveStatus {
    return switch (err) {
        error.InactiveEntry => .inactive_entry,
        error.ThreadNotRunning => .thread_not_running,
        error.InvalidHandle => .invalid_handle,
        error.StaleHandle => .stale_handle,
        error.Retired => .retired,
        error.InvalidState => .invalid_state,
        error.RootCapacityExceeded => .root_capacity,
        error.Shutdown => .shutdown,
        error.MissingSafepoint => .missing_root_map,
        error.MissingCollector, error.MissingSatbBuffer => .missing_collector,
        else => .runtime_error,
    };
}

fn addMappedRoots(
    state: *NativeThreadState,
    handle_bits: u64,
    site_key: u64,
    roots: *thread_registry.RootScope,
) Error!void {
    if (state.root_map_table == 0) return error.MissingSafepoint;
    const table: *const runtime_stack_map.Table = @ptrFromInt(state.root_map_table);
    const resolve_id: u32 = @truncate(site_key);
    const record = try table.find(resolve_id);
    var found_canonical = false;
    for (table.rootsFor(record)) |location| {
        if (location.kind != .native_register or location.payload >= state.captured_gp.len) return error.InvalidLocation;
        const slot = &state.captured_gp[location.payload];
        try roots.add(@ptrCast(slot));
        if (slot.* == handle_bits) found_canonical = true;
    }
    if (!found_canonical) return error.InvalidLocation;
}

/// Normal platform-ABI target called by the preserve-all machine-code adapter.
/// Zero is an error sentinel; managed heap regions can never contain address 0.
pub fn slowResolveBridge(state: *NativeThreadState, handle_bits: u64, site_key: u64) callconv(.c) usize {
    state.last_site_key = site_key;
    state.last_error = .ok;
    if (state.active == 0) {
        state.last_error = .inactive_entry;
        return 0;
    }
    if (!state.context.isRunning()) {
        state.last_error = .thread_not_running;
        return 0;
    }

    var roots = state.context.beginRootScope() catch |err| {
        state.last_error = statusFor(err);
        _ = state.runtime.counters.resolve_failures.fetchAdd(1, .monotonic);
        return 0;
    };
    defer roots.deinit();
    addMappedRoots(state, handle_bits, site_key, &roots) catch |err| {
        state.last_error = if (err == error.InvalidLocation) .missing_canonical_root else statusFor(err);
        _ = state.runtime.counters.resolve_failures.fetchAdd(1, .monotonic);
        return 0;
    };

    _ = state.runtime.counters.slow_resolves.fetchAdd(1, .monotonic);
    state.runtime.counters.last_slow_site.store(site_key, .release);
    const participated = state.runtime.registry.poll(state.context) catch |err| {
        state.last_error = statusFor(err);
        _ = state.runtime.counters.resolve_failures.fetchAdd(1, .monotonic);
        return 0;
    };
    if (participated) _ = state.runtime.counters.handshake_polls.fetchAdd(1, .monotonic);
    state.acknowledged_epoch = state.context.observedEpoch();

    const handle: Handle = @bitCast(handle_bits);
    const address = state.runtime.handles.resolve(handle) catch |err| {
        state.last_error = statusFor(err);
        _ = state.runtime.counters.resolve_failures.fetchAdd(1, .monotonic);
        return 0;
    };
    return @intFromPtr(address);
}

/// Explicit no-safepoint fallback. All managed registers remain saved by the
/// adapter, and the acknowledged epoch keeps from-space alive. Normal compiled
/// code uses `slowResolveBridge`; this target exists for bring-up and diagnosis.
pub fn slowResolveLeafBridge(state: *NativeThreadState, handle_bits: u64, site_key: u64) callconv(.c) usize {
    state.last_site_key = site_key;
    state.last_error = .ok;
    if (state.active == 0) {
        state.last_error = .inactive_entry;
        return 0;
    }
    if (!state.context.isRunning()) {
        state.last_error = .thread_not_running;
        return 0;
    }

    _ = state.runtime.counters.slow_resolves.fetchAdd(1, .monotonic);
    state.runtime.counters.last_slow_site.store(site_key, .release);
    const handle: Handle = @bitCast(handle_bits);
    const address = state.runtime.handles.resolve(handle) catch |err| {
        state.last_error = statusFor(err);
        _ = state.runtime.counters.resolve_failures.fetchAdd(1, .monotonic);
        return 0;
    };
    return @intFromPtr(address);
}

fn barrierFailure(state: *NativeThreadState, status: SlowResolveStatus) usize {
    state.last_error = status;
    _ = state.runtime.counters.barrier_failures.fetchAdd(1, .monotonic);
    return 0;
}

/// Platform-ABI target behind the preserve-all SATB adapter. The generated
/// store has not executed yet. Bounded mutator assistance drains one SATB item
/// under queue pressure; a transient termination election is retried before
/// the helper returns success.
pub fn referencePreWriteBridge(
    state: *NativeThreadState,
    slot_address: usize,
    repeat_proven_bits: u64,
) callconv(.c) usize {
    state.last_error = .ok;
    if (state.active == 0) return barrierFailure(state, .inactive_entry);
    if (state.collector == 0 or state.satb_buffer == 0) return barrierFailure(state, .missing_collector);
    const collector: *runtime_gc.ConcurrentCollector = @ptrFromInt(state.collector);
    const buffer: *runtime_gc.SatbBuffer = @ptrFromInt(state.satb_buffer);
    for (0..1024) |_| {
        collector.referenceStorePreWrite(buffer, slot_address, repeat_proven_bits != 0) catch |err| switch (err) {
            error.SatbQueueFull => {
                _ = collector.drainSatb(1) catch |drain_err| switch (drain_err) {
                    error.RetryBarrier => {
                        std.atomic.spinLoopHint();
                        continue;
                    },
                    else => return barrierFailure(state, .gc_barrier_failure),
                };
                continue;
            },
            error.RetryBarrier => {
                std.atomic.spinLoopHint();
                continue;
            },
            else => return barrierFailure(state, .gc_barrier_failure),
        };
        _ = state.runtime.counters.pre_write_barriers.fetchAdd(1, .monotonic);
        return 1;
    }
    return barrierFailure(state, .gc_barrier_failure);
}

/// Platform-ABI target behind the preserve-all card adapter. The reference
/// store is already globally visible; any failure is fatal and the generated
/// adapter traps rather than continuing with an unremembered edge.
pub fn referencePostWriteBridge(
    state: *NativeThreadState,
    destination_bits: u64,
    stored_bits: u64,
) callconv(.c) usize {
    return referencePostWriteBridgeImpl(state, destination_bits, stored_bits, false);
}

pub fn referencePostWriteRepeatBridge(
    state: *NativeThreadState,
    destination_bits: u64,
    stored_bits: u64,
) callconv(.c) usize {
    return referencePostWriteBridgeImpl(state, destination_bits, stored_bits, true);
}

/// Platform-ABI target for a pinned static root slot. The generated store is
/// already visible. Retry only spans the collector's short phase-election
/// window; an unregistered slot or invalid handle fails closed in the shim.
pub fn referenceStaticPostWriteBridge(
    state: *NativeThreadState,
    slot_address: usize,
    stored_bits: u64,
) callconv(.c) usize {
    state.last_error = .ok;
    if (state.active == 0) return barrierFailure(state, .inactive_entry);
    if (state.collector == 0) return barrierFailure(state, .missing_collector);
    const collector: *runtime_gc.ConcurrentCollector = @ptrFromInt(state.collector);
    for (0..1024) |_| {
        collector.referenceStaticStorePostWrite(slot_address, @bitCast(stored_bits)) catch |err| switch (err) {
            error.RetryBarrier => {
                std.atomic.spinLoopHint();
                continue;
            },
            else => return barrierFailure(state, .gc_barrier_failure),
        };
        _ = state.runtime.counters.post_write_barriers.fetchAdd(1, .monotonic);
        return 1;
    }
    return barrierFailure(state, .gc_barrier_failure);
}

fn referencePostWriteBridgeImpl(
    state: *NativeThreadState,
    destination_bits: u64,
    stored_bits: u64,
    repeat_proven: bool,
) usize {
    state.last_error = .ok;
    if (state.active == 0) return barrierFailure(state, .inactive_entry);
    if (state.collector == 0) return barrierFailure(state, .missing_collector);
    const collector: *runtime_gc.ConcurrentCollector = @ptrFromInt(state.collector);
    collector.referenceStorePostWrite(
        @bitCast(destination_bits),
        @bitCast(stored_bits),
        repeat_proven,
        &state.last_card_destination,
    ) catch {
        return barrierFailure(state, .gc_barrier_failure);
    };
    _ = state.runtime.counters.post_write_barriers.fetchAdd(1, .monotonic);
    return 1;
}

fn waitForValue(value: *const std.atomic.Value(bool), expected: bool) !void {
    for (0..1_000_000) |_| {
        if (value.load(.acquire) == expected) return;
        std.atomic.spinLoopHint();
        std.Thread.yield() catch {};
    }
    return error.Timeout;
}

fn waitHandshake(handshake: *const thread_registry.Handshake, context: *const ThreadContext) !void {
    for (0..1_000_000) |_| {
        if (handshake.isReady(context)) return;
        std.atomic.spinLoopHint();
        std.Thread.yield() catch {};
    }
    return error.Timeout;
}

test "managed entry exposes stable register image and follows relocation" {
    var from_space: [128]u8 align(runtime_value.object_alignment) = undefined;
    var to_space: [128]u8 align(runtime_value.object_alignment) = undefined;
    const regions = [_]runtime_value.Region{
        try runtime_value.Region.fromSlice(&from_space),
        try runtime_value.Region.fromSlice(&to_space),
    };
    var handles = try HandleTable.init(std.testing.allocator, 8, &regions);
    defer handles.deinit();
    var registry = try Registry.init(std.testing.allocator, std.testing.io, 1);
    defer registry.deinit() catch unreachable;
    var context = try ThreadContext.init(std.testing.allocator, 4);
    defer context.deinit();
    try registry.register(&context);
    defer registry.unregister(&context) catch unreachable;
    var runtime = try Runtime.init(std.testing.allocator, &handles, &registry);
    defer runtime.deinit() catch unreachable;

    const handle = try handles.reserve(0, 0);
    const original: *anyopaque = @ptrCast(&from_space[16]);
    try handles.publish(handle, 0, original);

    var entry = try runtime.enter(&context);
    defer entry.deinit();
    const image = try entry.registerImage();
    try std.testing.expectEqual(handles.descriptorBaseAddress(), image.r14_descriptor_base);
    try std.testing.expectEqual(@intFromPtr(runtime.region_bases.ptr), image.r13_region_bases);
    try std.testing.expectEqual(@intFromPtr(&entry.native_state), image.r15_thread_state);
    try std.testing.expectEqual(@as(u8, 8), image.descriptor_stride);
    try std.testing.expectEqual(@intFromPtr(original), @intFromPtr(try entry.resolveSlow(handle, 0x12_0000_0003)));

    const ticket = try handles.beginRelocation(handle);
    const moved: *anyopaque = @ptrCast(&to_space[24]);
    try std.testing.expect(try handles.commitRelocation(ticket, 1, moved));
    try std.testing.expectEqual(@intFromPtr(moved), @intFromPtr(try entry.resolveSlow(handle, 0x13_0000_0004)));
    try std.testing.expectEqual(@as(u64, 2), runtime.stats().slow_resolves);
    try std.testing.expectEqual(@as(u64, 0x13_0000_0004), runtime.stats().last_slow_site);
}

test "polling bridge fails closed until its precise root map matches the receiver" {
    var storage: [64]u8 align(runtime_value.object_alignment) = undefined;
    const regions = [_]runtime_value.Region{try runtime_value.Region.fromSlice(&storage)};
    var handles = try HandleTable.init(std.testing.allocator, 1, &regions);
    defer handles.deinit();
    const handle = try handles.reserve(0, 0);
    const object: *anyopaque = @ptrCast(&storage[16]);
    try handles.publish(handle, 0, object);

    var registry = try Registry.init(std.testing.allocator, std.testing.io, 1);
    defer registry.deinit() catch unreachable;
    var context = try ThreadContext.init(std.testing.allocator, 1);
    defer context.deinit();
    try registry.register(&context);
    defer registry.unregister(&context) catch unreachable;
    var runtime = try Runtime.init(std.testing.allocator, &handles, &registry);
    defer runtime.deinit() catch unreachable;
    var entry = try runtime.enter(&context);
    defer entry.deinit();

    const handle_bits: u64 = @bitCast(handle);
    try std.testing.expectEqual(@as(usize, 0), slowResolveBridge(&entry.native_state, handle_bits, 0x55_0000_0007));
    try std.testing.expectEqual(SlowResolveStatus.missing_root_map, entry.native_state.last_error);
    try std.testing.expectEqual(@as(usize, 0), context.rootCount());

    const locations = [_]runtime_stack_map.RootLocation{runtime_stack_map.RootLocation.nativeRegister(2)};
    const specs = [_]runtime_stack_map.MapSpec{.{ .pc_offset = 7, .roots = &locations }};
    var maps = try runtime_stack_map.Table.init(std.testing.allocator, &specs, .{
        .native_register_count = 16,
        .interpreter_register_count = 0,
        .max_frame_depth = 0,
        .max_shadow_roots = 0,
    });
    defer maps.deinit();
    try entry.installRootMaps(&maps);

    entry.native_state.captured_gp[2] = 0;
    try std.testing.expectEqual(@as(usize, 0), slowResolveBridge(&entry.native_state, handle_bits, 0x55_0000_0007));
    try std.testing.expectEqual(SlowResolveStatus.missing_canonical_root, entry.native_state.last_error);
    try std.testing.expectEqual(@as(usize, 0), context.rootCount());

    entry.native_state.captured_gp[2] = handle_bits;
    try std.testing.expectEqual(@intFromPtr(object), slowResolveBridge(&entry.native_state, handle_bits, 0x55_0000_0007));
    try std.testing.expectEqual(SlowResolveStatus.ok, entry.native_state.last_error);
    try std.testing.expectEqual(@as(usize, 0), context.rootCount());
    try std.testing.expectEqual(@as(u64, 1), runtime.stats().slow_resolves);
    try std.testing.expectEqual(@as(u64, 2), runtime.stats().resolve_failures);
}

test "slow resolver publishes temporary root during concurrent handshake" {
    var storage: [128]u8 align(runtime_value.object_alignment) = undefined;
    const regions = [_]runtime_value.Region{try runtime_value.Region.fromSlice(&storage)};
    var handles = try HandleTable.init(std.testing.allocator, 2, &regions);
    defer handles.deinit();
    const handle = try handles.reserve(0, 0);
    try handles.publish(handle, 0, @ptrCast(&storage[16]));

    var registry = try Registry.init(std.testing.allocator, std.testing.io, 1);
    defer registry.deinit() catch unreachable;
    var context = try ThreadContext.init(std.testing.allocator, 2);
    defer context.deinit();
    try registry.register(&context);
    defer registry.unregister(&context) catch unreachable;
    var runtime = try Runtime.init(std.testing.allocator, &handles, &registry);
    defer runtime.deinit() catch unreachable;

    const Worker = struct {
        runtime: *Runtime,
        context: *ThreadContext,
        handle: Handle,
        entered: *std.atomic.Value(bool),
        completed: *std.atomic.Value(bool),

        fn run(self: *@This()) void {
            var entry = self.runtime.enter(self.context) catch return;
            defer entry.deinit();
            self.entered.store(true, .release);
            while (self.runtime.registry.requestEpoch() == entry.acknowledged_epoch) std.atomic.spinLoopHint();
            _ = entry.resolveSlow(self.handle, 0x20_0000_0001) catch return;
            self.completed.store(true, .release);
        }
    };

    var entered = std.atomic.Value(bool).init(false);
    var completed = std.atomic.Value(bool).init(false);
    var worker = Worker{
        .runtime = &runtime,
        .context = &context,
        .handle = handle,
        .entered = &entered,
        .completed = &completed,
    };
    const thread = try std.Thread.spawn(.{}, Worker.run, .{&worker});
    try waitForValue(&entered, true);

    var members: [1]*ThreadContext = undefined;
    var handshake = try registry.beginHandshake(&members);
    try waitHandshake(&handshake, &context);
    try std.testing.expectEqualSlices(Handle, &.{handle}, try handshake.snapshot(&context));
    try handshake.release(&context);
    try handshake.finish();
    thread.join();
    try std.testing.expect(completed.load(.acquire));
    try std.testing.expectEqual(@as(usize, 0), context.rootCount());
    try std.testing.expectEqual(@as(u64, 1), runtime.stats().handshake_polls);
}

test "managed entry lifecycle fails closed for stale handles and active shutdown" {
    var storage: [64]u8 align(runtime_value.object_alignment) = undefined;
    const regions = [_]runtime_value.Region{try runtime_value.Region.fromSlice(&storage)};
    var handles = try HandleTable.init(std.testing.allocator, 1, &regions);
    defer handles.deinit();
    const stale = try handles.reserve(0, 0);
    try handles.publish(stale, 0, @ptrCast(&storage[8]));
    try std.testing.expect(try handles.retire(stale));
    try handles.recycleAfterQuiescence(stale);

    var registry = try Registry.init(std.testing.allocator, std.testing.io, 1);
    defer registry.deinit() catch unreachable;
    var context = try ThreadContext.init(std.testing.allocator, 1);
    defer context.deinit();
    var runtime = try Runtime.init(std.testing.allocator, &handles, &registry);

    try std.testing.expectError(error.ThreadNotRunning, runtime.enter(&context));
    try registry.register(&context);
    var entry = try runtime.enter(&context);
    try std.testing.expectError(error.ActiveEntries, runtime.deinit());
    try std.testing.expectError(error.StaleHandle, entry.resolveSlow(stale, 0x30_0000_0001));
    try std.testing.expectEqual(@as(usize, 0), context.rootCount());
    try std.testing.expectEqual(@as(u64, 1), runtime.stats().resolve_failures);
    entry.deinit();
    try std.testing.expectError(error.InactiveEntry, entry.registerImage());
    try registry.unregister(&context);
    try runtime.deinit();
}

test "concurrent slow resolvers observe one of two atomically published locations" {
    var from_space: [128]u8 align(runtime_value.object_alignment) = undefined;
    var to_space: [128]u8 align(runtime_value.object_alignment) = undefined;
    const regions = [_]runtime_value.Region{
        try runtime_value.Region.fromSlice(&from_space),
        try runtime_value.Region.fromSlice(&to_space),
    };
    var handles = try HandleTable.init(std.testing.allocator, 1, &regions);
    defer handles.deinit();
    const handle = try handles.reserve(0, 0);
    const original: *anyopaque = @ptrCast(&from_space[8]);
    const moved: *anyopaque = @ptrCast(&to_space[24]);
    try handles.publish(handle, 0, original);
    const ticket = try handles.beginRelocation(handle);

    var registry = try Registry.init(std.testing.allocator, std.testing.io, 2);
    defer registry.deinit() catch unreachable;
    var first_context = try ThreadContext.init(std.testing.allocator, 1);
    defer first_context.deinit();
    var second_context = try ThreadContext.init(std.testing.allocator, 1);
    defer second_context.deinit();
    try registry.register(&first_context);
    try registry.register(&second_context);
    defer registry.unregister(&second_context) catch unreachable;
    defer registry.unregister(&first_context) catch unreachable;
    var runtime = try Runtime.init(std.testing.allocator, &handles, &registry);
    defer runtime.deinit() catch unreachable;

    const Worker = struct {
        runtime: *Runtime,
        context: *ThreadContext,
        handle: Handle,
        original: usize,
        moved: usize,
        start: *std.atomic.Value(bool),
        failures: *std.atomic.Value(u32),

        fn run(self: *@This()) void {
            var entry = self.runtime.enter(self.context) catch {
                _ = self.failures.fetchAdd(1, .monotonic);
                return;
            };
            defer entry.deinit();
            while (!self.start.load(.acquire)) std.atomic.spinLoopHint();
            for (0..2_000) |iteration| {
                const address = entry.resolveSlow(self.handle, 0x40_0000_0000 | iteration) catch {
                    _ = self.failures.fetchAdd(1, .monotonic);
                    return;
                };
                const raw = @intFromPtr(address);
                if (raw != self.original and raw != self.moved) {
                    _ = self.failures.fetchAdd(1, .monotonic);
                    return;
                }
            }
        }
    };

    var start = std.atomic.Value(bool).init(false);
    var failures = std.atomic.Value(u32).init(0);
    var first = Worker{
        .runtime = &runtime,
        .context = &first_context,
        .handle = handle,
        .original = @intFromPtr(original),
        .moved = @intFromPtr(moved),
        .start = &start,
        .failures = &failures,
    };
    var second = first;
    second.context = &second_context;
    const first_thread = try std.Thread.spawn(.{}, Worker.run, .{&first});
    const second_thread = try std.Thread.spawn(.{}, Worker.run, .{&second});
    start.store(true, .release);
    try std.testing.expect(try handles.commitRelocation(ticket, 1, moved));
    first_thread.join();
    second_thread.join();

    try std.testing.expectEqual(@as(u32, 0), failures.load(.acquire));
    try std.testing.expectEqual(@intFromPtr(moved), @intFromPtr(try handles.resolve(handle)));
    try std.testing.expectEqual(@as(u64, 4_000), runtime.stats().slow_resolves);
    try std.testing.expectEqual(@as(u64, 0), runtime.stats().active_entries);
}

fn allocationFailureProbe(allocator: std.mem.Allocator) !void {
    var storage: [64]u8 align(runtime_value.object_alignment) = undefined;
    const regions = [_]runtime_value.Region{try runtime_value.Region.fromSlice(&storage)};
    var handles = try HandleTable.init(std.testing.allocator, 1, &regions);
    defer handles.deinit();
    var registry = try Registry.init(std.testing.allocator, std.testing.io, 1);
    defer registry.deinit() catch unreachable;
    var runtime = try Runtime.init(allocator, &handles, &registry);
    defer runtime.deinit() catch unreachable;
}

test "jit runtime initialization is leak-free at every allocation failure" {
    try std.testing.checkAllAllocationFailures(std.testing.allocator, allocationFailureProbe, .{});
}
