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
const runtime_monitor = @import("runtime_monitor");
const runtime_code_manager = @import("runtime_code_manager");
const runtime_deopt = @import("runtime_deopt");

const Handle = runtime_value.Handle;
const HandleTable = runtime_value.HandleTable;
const Registry = thread_registry.Registry;
const ThreadContext = thread_registry.ThreadContext;

pub const Error = runtime_value.Error || runtime_stack_map.Error || thread_registry.Error || runtime_gc.Error || runtime_monitor.Error || runtime_code_manager.Error || runtime_deopt.Error || std.mem.Allocator.Error || error{
    ActiveEntries,
    ActiveCodeLease,
    InactiveEntry,
    InvalidTableLayout,
    MissingCollector,
    MissingMonitorTable,
    PendingException,
    ThreadNotRunning,
};

pub const unmanaged_method_id = std.math.maxInt(u32);

pub const CodeDispatchStatus = enum(u32) {
    ok = 0,
    fallback_no_code,
    fallback_runtime_error,
    inactive_entry,
    active_lease,
    deoptimized,
    deopt_failed,
    fallback_osr_metadata,
};

pub const OsrEntry = extern struct {
    point_id: u32,
    code_offset: u32,
};

pub const ManagedExceptionKind = enum(u32) {
    none = 0,
    array_index_out_of_bounds = 1,
};

pub const max_native_monitors = 16;

/// Allocation-free exception payload produced by generated code. Object
/// materialization and handler lookup happen only after leaving the native
/// frame, so the uncommon edge never invokes platform stack unwinding.
pub const ManagedException = extern struct {
    kind: ManagedExceptionKind = .none,
    dex_pc: u32 = 0,
    index: i32 = 0,
    length: u32 = 0,
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
    missing_monitor,
    illegal_monitor_state,
    monitor_capacity,
};

/// Pinned for the duration of a native call. r15 points here, allowing the
/// architecture adapter to find the correct runtime without global state.
pub const NativeThreadState = extern struct {
    runtime: *Runtime,
    context: *ThreadContext,
    acknowledged_epoch: u64,
    request_epoch_address: usize,
    last_site_key: u64 = 0,
    last_error: SlowResolveStatus = .ok,
    active: u32 = 1,
    root_map_table: usize = 0,
    collector: usize = 0,
    satb_buffer: usize = 0,
    monitor_table: usize = 0,
    held_monitors: [max_native_monitors]u64 = @splat(@bitCast(Handle.none)),
    held_monitor_count: u32 = 0,
    last_card_destination: u64 = 0,
    pending_exception: ManagedException = .{},
    code_manager: usize = 0,
    code_reader: usize = 0,
    code_lease_slot: usize = 0,
    deopt_request: usize = 0,
    last_code_dispatch: CodeDispatchStatus = .ok,
    captured_gp: [16]u64 = @splat(0),
    captured_xmm: [8][16]u8 = @splat(@splat(0)),
    /// Managed rsp before the private helper call. Spill offsets are relative
    /// to this base and are bounded by immutable deoptimization metadata.
    captured_stack_base: usize = 0,
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
    exception_transfers: u64,
    exception_failures: u64,
    code_entries: u64,
    code_fallbacks: u64,
    code_failures: u64,
    deoptimizations: u64,
    deopt_failures: u64,
    monitor_enters: u64,
    monitor_exits: u64,
    monitor_unwind_exits: u64,
    monitor_failures: u64,
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
    exception_transfers: std.atomic.Value(u64) = std.atomic.Value(u64).init(0),
    exception_failures: std.atomic.Value(u64) = std.atomic.Value(u64).init(0),
    code_entries: std.atomic.Value(u64) = std.atomic.Value(u64).init(0),
    code_fallbacks: std.atomic.Value(u64) = std.atomic.Value(u64).init(0),
    code_failures: std.atomic.Value(u64) = std.atomic.Value(u64).init(0),
    deoptimizations: std.atomic.Value(u64) = std.atomic.Value(u64).init(0),
    deopt_failures: std.atomic.Value(u64) = std.atomic.Value(u64).init(0),
    monitor_enters: std.atomic.Value(u64) = std.atomic.Value(u64).init(0),
    monitor_exits: std.atomic.Value(u64) = std.atomic.Value(u64).init(0),
    monitor_unwind_exits: std.atomic.Value(u64) = std.atomic.Value(u64).init(0),
    monitor_failures: std.atomic.Value(u64) = std.atomic.Value(u64).init(0),
};

pub const Runtime = struct {
    allocator: std.mem.Allocator,
    handles: *HandleTable,
    registry: *Registry,
    region_bases: []usize,
    collector: ?*runtime_gc.ConcurrentCollector = null,
    monitor_table: ?*runtime_monitor.MonitorTable = null,
    code_manager: ?*runtime_code_manager.Manager = null,
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

    pub fn installMonitorTable(self: *Runtime, monitors: *runtime_monitor.MonitorTable) Error!void {
        if (self.counters.active_entries.load(.acquire) != 0) return error.ActiveEntries;
        if (monitors.handleTable() != self.handles) return error.InvalidTableLayout;
        const collector = self.collector orelse return error.MissingCollector;
        for (monitors.rootSlotAddresses()) |address| {
            if (!collector.isStaticRootSlot(address)) return error.InvalidTableLayout;
        }
        self.monitor_table = monitors;
    }

    /// Installs the immutable-code publication domain used by architecture
    /// entry trampolines. Each subsequently-created managed entry owns one
    /// reader slot; no manager lock is acquired while executing compiled code.
    pub fn installCodeManager(self: *Runtime, manager: *runtime_code_manager.Manager) Error!void {
        if (self.counters.active_entries.load(.acquire) != 0) return error.ActiveEntries;
        self.code_manager = manager;
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
        var code_reader: ?runtime_code_manager.Reader = null;
        if (self.code_manager) |manager| code_reader = try manager.registerReader();
        errdefer if (code_reader) |*reader| reader.deinit();
        // Validate every dependency before publishing an active native entry;
        // failed entry attempts must not strand Runtime.deinit forever.
        _ = self.counters.managed_entries.fetchAdd(1, .monotonic);
        _ = self.counters.active_entries.fetchAdd(1, .acq_rel);
        return .{
            .runtime = self,
            .context = context,
            .acknowledged_epoch = context.observedEpoch(),
            .code_reader = code_reader,
            .native_state = .{
                .runtime = self,
                .context = context,
                .acknowledged_epoch = context.observedEpoch(),
                .request_epoch_address = self.registry.requestEpochAddress(),
                .collector = collector_address,
                .satb_buffer = buffer_address,
                .monitor_table = if (self.monitor_table) |monitors| @intFromPtr(monitors) else 0,
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
            .exception_transfers = self.counters.exception_transfers.load(.acquire),
            .exception_failures = self.counters.exception_failures.load(.acquire),
            .code_entries = self.counters.code_entries.load(.acquire),
            .code_fallbacks = self.counters.code_fallbacks.load(.acquire),
            .code_failures = self.counters.code_failures.load(.acquire),
            .deoptimizations = self.counters.deoptimizations.load(.acquire),
            .deopt_failures = self.counters.deopt_failures.load(.acquire),
            .monitor_enters = self.counters.monitor_enters.load(.acquire),
            .monitor_exits = self.counters.monitor_exits.load(.acquire),
            .monitor_unwind_exits = self.counters.monitor_unwind_exits.load(.acquire),
            .monitor_failures = self.counters.monitor_failures.load(.acquire),
        };
    }
};

pub const ManagedEntry = struct {
    runtime: *Runtime,
    context: *ThreadContext,
    acknowledged_epoch: u64,
    native_state: NativeThreadState,
    code_reader: ?runtime_code_manager.Reader = null,
    code_lease: ?runtime_code_manager.Lease = null,
    active: bool = true,

    pub fn deinit(self: *ManagedEntry) void {
        if (!self.active) return;
        releaseHeldMonitors(&self.native_state);
        if (self.code_lease) |*lease| lease.deinit();
        self.code_lease = null;
        if (self.code_reader) |*reader| reader.deinit();
        self.code_reader = null;
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
        self.native_state.request_epoch_address = self.runtime.registry.requestEpochAddress();
        return participated;
    }

    pub fn registerImage(self: *ManagedEntry) Error!RegisterImage {
        if (!self.active) return error.InactiveEntry;
        if (!self.context.isRunning()) return error.ThreadNotRunning;
        if (self.native_state.pending_exception.kind != .none) return error.PendingException;
        if (self.code_lease != null) return error.ActiveCodeLease;
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
        self.native_state.monitor_table = if (self.runtime.monitor_table) |monitors| @intFromPtr(monitors) else 0;
        self.native_state.code_manager = if (self.runtime.code_manager) |manager| @intFromPtr(manager) else 0;
        self.native_state.code_reader = if (self.code_reader) |*reader| @intFromPtr(reader) else 0;
        self.native_state.code_lease_slot = @intFromPtr(&self.code_lease);
        self.native_state.deopt_request = 0;
        self.native_state.last_code_dispatch = .ok;
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

    pub fn pendingException(self: *const ManagedEntry) ?ManagedException {
        if (self.native_state.pending_exception.kind == .none) return null;
        return self.native_state.pending_exception;
    }

    /// Consume the owner-confined lazy exception record before the next native
    /// entry. The returned payload is sufficient for handler lookup and later
    /// managed exception-object materialization.
    pub fn takeException(self: *ManagedEntry) ?ManagedException {
        const exception = self.pendingException() orelse return null;
        self.native_state.pending_exception = .{};
        return exception;
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

/// Stable, non-reclaimable target used when dispatch has neither compiled code
/// nor an installed interpreter/deoptimization fallback.
pub fn unavailableCodeTarget(
    _: usize,
    _: usize,
    _: usize,
    _: usize,
    _: usize,
    _: usize,
) callconv(.c) usize {
    return 0;
}

fn codeFallback(state: *NativeThreadState, status: CodeDispatchStatus, fallback_target: usize) usize {
    state.last_code_dispatch = status;
    _ = state.runtime.counters.code_fallbacks.fetchAdd(1, .monotonic);
    if (status != .fallback_no_code) _ = state.runtime.counters.code_failures.fetchAdd(1, .monotonic);
    if (fallback_target != 0) return fallback_target;
    if (status == .fallback_no_code and state.deopt_request != 0) return @intFromPtr(&deoptResumeBridge);
    return @intFromPtr(&unavailableCodeTarget);
}

/// Runtime half of the architecture entry trampoline. The reader is bound to
/// this ManagedEntry and therefore owner-confined; publication and reclamation
/// only observe its atomic reader slot. A successful return owns exactly one
/// lease in `code_lease_slot` until `codeLeaseExitBridge` runs.
pub fn codeLeaseEnterBridge(
    state: *NativeThreadState,
    fallback_target: usize,
    method: u32,
    deopt_request: usize,
) callconv(.c) usize {
    state.deopt_request = deopt_request;
    if (state.active == 0) return codeFallback(state, .inactive_entry, fallback_target);
    if (state.code_manager == 0 or state.code_reader == 0 or state.code_lease_slot == 0) {
        return codeFallback(state, .fallback_runtime_error, fallback_target);
    }

    const lease_slot: *?runtime_code_manager.Lease = @ptrFromInt(state.code_lease_slot);
    if (lease_slot.* != null) return codeFallback(state, .active_lease, fallback_target);

    var entry_roots = state.context.beginRootScope() catch {
        return codeFallback(state, .fallback_runtime_error, fallback_target);
    };
    defer entry_roots.deinit();
    if (deopt_request != 0) {
        const request: *runtime_deopt.Request = @ptrFromInt(deopt_request);
        const RootVisitor = struct {
            fn add(scope: *thread_registry.RootScope, slot: *const Handle) thread_registry.Error!void {
                try scope.add(slot);
            }
        };
        request.table.visitReferenceSlots(request.point_id, .{
            .native_registers = &state.captured_gp,
            .xmm_registers = request.xmm_registers,
            .stack_base = request.stack_base,
            .stack_min_offset = request.stack_min_offset,
            .stack_max_offset = request.stack_max_offset,
        }, &entry_roots, RootVisitor.add) catch {
            return codeFallback(state, .fallback_runtime_error, fallback_target);
        };
    }

    // Entry is a safepoint. Poll before loading the code epoch so generated
    // r12 receives the post-handshake heap epoch on return from this bridge.
    const participated = state.runtime.registry.poll(state.context) catch {
        return codeFallback(state, .fallback_runtime_error, fallback_target);
    };
    if (participated) _ = state.runtime.counters.handshake_polls.fetchAdd(1, .monotonic);
    state.acknowledged_epoch = state.context.observedEpoch();
    state.request_epoch_address = state.runtime.registry.requestEpochAddress();

    const manager: *runtime_code_manager.Manager = @ptrFromInt(state.code_manager);
    const reader: *runtime_code_manager.Reader = @ptrFromInt(state.code_reader);
    lease_slot.* = manager.enter(reader, method) catch |err| switch (err) {
        error.NoCode => return codeFallback(state, .fallback_no_code, fallback_target),
        else => return codeFallback(state, .fallback_runtime_error, fallback_target),
    };
    state.last_code_dispatch = .ok;
    _ = state.runtime.counters.code_entries.fetchAdd(1, .monotonic);
    return lease_slot.*.?.entryAddress();
}

/// Acquires the same code-version lease as normal managed entry, then resolves
/// a compiler-verified no-prologue OSR label owned by that exact version.
/// Missing, malformed, or out-of-range metadata closes the lease and falls
/// back without exposing an interior executable address.
pub fn codeOsrEnterBridge(
    state: *NativeThreadState,
    fallback_target: usize,
    method: u32,
    point_id: u32,
) callconv(.c) usize {
    const base = codeLeaseEnterBridge(state, fallback_target, method, 0);
    if (state.code_lease_slot == 0) return base;
    const lease_slot: *?runtime_code_manager.Lease = @ptrFromInt(state.code_lease_slot);
    const lease = if (lease_slot.*) |*value| value else return base;
    const metadata = lease.metadata() orelse {
        _ = codeLeaseExitBridge(state, 0);
        return codeFallback(state, .fallback_osr_metadata, fallback_target);
    };
    if (metadata.osr_entries == 0 or metadata.osr_entry_count == 0) {
        _ = codeLeaseExitBridge(state, 0);
        return codeFallback(state, .fallback_osr_metadata, fallback_target);
    }
    const entries: [*]const OsrEntry = @ptrFromInt(metadata.osr_entries);
    var previous: ?u32 = null;
    for (entries[0..metadata.osr_entry_count]) |entry| {
        if (previous) |id| if (entry.point_id <= id) break;
        previous = entry.point_id;
        if (entry.point_id < point_id) continue;
        if (entry.point_id != point_id or entry.code_offset == 0 or entry.code_offset >= lease.codeSize() or
            !std.mem.isAligned(entry.code_offset, 16) or entry.code_offset > std.math.maxInt(usize) - base)
        {
            break;
        }
        return base + entry.code_offset;
    }
    _ = codeLeaseExitBridge(state, 0);
    return codeFallback(state, .fallback_osr_metadata, fallback_target);
}

/// Stable fallback ABI selected by the entry trampoline after invalidation.
/// The trampoline passes NativeThreadState as argument zero and has already
/// captured the original managed GP arguments into `captured_gp`.
pub fn deoptResumeBridge(
    state: *NativeThreadState,
    _: usize,
    _: usize,
    _: usize,
    _: usize,
    _: usize,
) callconv(.c) usize {
    if (state.active == 0 or state.deopt_request == 0) {
        return deoptFailure(state, null);
    }
    const request: *runtime_deopt.Request = @ptrFromInt(state.deopt_request);
    const result = resumeDeoptPoint(state, request, request.table, request.point_id, .{
        .native_registers = &state.captured_gp,
        .xmm_registers = request.xmm_registers,
        .stack_base = request.stack_base,
        .stack_min_offset = request.stack_min_offset,
        .stack_max_offset = request.stack_max_offset,
    });
    state.deopt_request = 0;
    return result;
}

fn deoptFailure(state: *NativeThreadState, frame: ?*runtime_deopt.Frame) usize {
    if (frame) |value| value.active = false;
    state.last_code_dispatch = .deopt_failed;
    _ = state.runtime.counters.deopt_failures.fetchAdd(1, .monotonic);
    _ = state.runtime.counters.code_failures.fetchAdd(1, .monotonic);
    return 0;
}

fn deoptFailureDestination(state: *NativeThreadState, destination: runtime_deopt.Destination) usize {
    destination.frame.active = false;
    for (destination.inline_frames) |*frame| frame.active = false;
    return deoptFailure(state, null);
}

fn resumeDeoptPoint(
    state: *NativeThreadState,
    request: *runtime_deopt.Request,
    table: *const runtime_deopt.Table,
    point_id: u32,
    capture: runtime_deopt.Capture,
) usize {
    const exception = if (state.pending_exception.kind == .none)
        request.exception
    else
        runtime_deopt.ExceptionState{
            .kind = @intFromEnum(state.pending_exception.kind),
            .dex_pc = state.pending_exception.dex_pc,
            .payload0 = @bitCast(@as(i64, state.pending_exception.index)),
            .payload1 = state.pending_exception.length,
        };
    const frame = table.reconstruct(
        point_id,
        capture,
        request.destination,
        request.reason,
        exception,
    ) catch return deoptFailure(state, null);
    state.pending_exception = .{};
    var resumed_roots = state.context.beginRootScope() catch return deoptFailureDestination(state, request.destination);
    defer resumed_roots.deinit();
    const RootVisitor = struct {
        fn add(scope: *thread_registry.RootScope, slot: *const Handle) thread_registry.Error!void {
            try scope.add(slot);
        }
    };
    table.visitFrameReferenceSlots(point_id, frame, &resumed_roots, RootVisitor.add) catch {
        return deoptFailureDestination(state, request.destination);
    };
    state.last_code_dispatch = .deoptimized;
    _ = state.runtime.counters.deoptimizations.fetchAdd(1, .monotonic);
    return request.resume_fn(request.resume_context, frame);
}

/// Preserve-all target used by a generated dependency-epoch cold edge. The
/// active code lease pins both RX bytes and immutable metadata until the outer
/// entry trampoline closes it after this call returns from compiled code.
pub fn midFunctionDeoptBridge(state: *NativeThreadState, site_id: u32) callconv(.c) usize {
    if (state.active == 0 or state.deopt_request == 0 or state.code_lease_slot == 0) {
        return deoptFailure(state, null);
    }
    // Until monitor ownership is represented in immutable deoptimization
    // metadata, never transfer a frame while this native activation owns a
    // monitor. The entry remains responsible for fail-safe reverse unwind.
    if (state.held_monitor_count != 0) return deoptFailure(state, null);
    const lease_slot: *?runtime_code_manager.Lease = @ptrFromInt(state.code_lease_slot);
    const lease = if (lease_slot.*) |*value| value else return deoptFailure(state, null);
    const metadata = lease.metadata() orelse return deoptFailure(state, null);
    if (metadata.stack_maps == 0 or metadata.deopt_table == 0) return deoptFailure(state, null);
    const stack_maps: *const runtime_stack_map.Table = @ptrFromInt(metadata.stack_maps);
    const deopt_table: *const runtime_deopt.Table = @ptrFromInt(metadata.deopt_table);
    const stack_record = stack_maps.find(site_id) catch return deoptFailure(state, null);
    if (stack_record.deopt_id == runtime_stack_map.no_deopt) return deoptFailure(state, null);
    const request: *runtime_deopt.Request = @ptrFromInt(state.deopt_request);
    if (request.table != deopt_table) return deoptFailure(state, null);
    if (state.captured_stack_base == 0) return deoptFailure(state, null);
    const stack_bounds = deopt_table.stackBounds(stack_record.deopt_id) catch return deoptFailure(state, null);
    return resumeDeoptPoint(state, request, deopt_table, stack_record.deopt_id, .{
        .native_registers = &state.captured_gp,
        .xmm_registers = &state.captured_xmm,
        .stack_base = @ptrFromInt(state.captured_stack_base),
        .stack_min_offset = stack_bounds.min_offset,
        .stack_max_offset = stack_bounds.max_offset,
    });
}

/// Normal-return edge for both compiled and fallback targets. Returning the
/// result unchanged lets the generated trampoline close the lease without a
/// spill or a second result ABI.
pub fn codeLeaseExitBridge(state: *NativeThreadState, result: usize) callconv(.c) usize {
    if (state.code_lease_slot == 0) return result;
    const lease_slot: *?runtime_code_manager.Lease = @ptrFromInt(state.code_lease_slot);
    if (lease_slot.*) |*lease| lease.deinit();
    lease_slot.* = null;
    state.deopt_request = 0;
    return result;
}

test "code OSR entry resolves a version-owned interior label under one lease" {
    var storage: [64]u8 align(runtime_value.object_alignment) = @splat(0);
    const regions = [_]runtime_value.Region{try runtime_value.Region.fromSlice(&storage)};
    var handles = try runtime_value.HandleTable.init(std.testing.allocator, 2, &regions);
    defer handles.deinit();
    var registry = try thread_registry.Registry.init(std.testing.allocator, std.testing.io, 1);
    defer registry.deinit() catch unreachable;
    var context = try thread_registry.ThreadContext.init(std.testing.allocator, 2);
    defer context.deinit();
    try registry.register(&context);
    defer registry.unregister(&context) catch unreachable;

    var manager = try runtime_code_manager.Manager.init(std.testing.allocator, 1, 1, 1);
    defer manager.deinit() catch unreachable;
    const Owner = struct {
        references: std.atomic.Value(u32) = std.atomic.Value(u32).init(1),

        fn retain(raw: *anyopaque) void {
            const self: *@This() = @ptrCast(@alignCast(raw));
            _ = self.references.fetchAdd(1, .acq_rel);
        }

        fn release(raw: *anyopaque) void {
            const self: *@This() = @ptrCast(@alignCast(raw));
            _ = self.references.fetchSub(1, .acq_rel);
        }
    };
    var owner: Owner = .{};
    const entries = [_]OsrEntry{.{ .point_id = 9, .code_offset = 16 }};
    const code = [_]u8{0x90} ** 31 ++ .{0xc3};
    var candidate = try manager.prepareWithMetadata(&code, .{
        .context = @ptrCast(&owner),
        .osr_entries = @intFromPtr(&entries),
        .osr_entry_count = entries.len,
        .retain = Owner.retain,
        .release = Owner.release,
    });
    try manager.publish(0, &candidate);

    var runtime = try Runtime.init(std.testing.allocator, &handles, &registry);
    defer runtime.deinit() catch unreachable;
    try runtime.installCodeManager(&manager);
    {
        var managed = try runtime.enter(&context);
        defer managed.deinit();
        _ = try managed.registerImage();
        const target = codeOsrEnterBridge(&managed.native_state, 0x1234, 0, 9);
        const lease = if (managed.code_lease) |*value| value else return error.TestUnexpectedResult;
        try std.testing.expectEqual(lease.entryAddress() + 16, target);
        try std.testing.expectEqual(CodeDispatchStatus.ok, managed.native_state.last_code_dispatch);
        try std.testing.expectEqual(@as(u64, 1), manager.stats().active_leases);
        try std.testing.expectEqual(@as(usize, 77), codeLeaseExitBridge(&managed.native_state, 77));
        try std.testing.expectEqual(@as(u64, 0), manager.stats().active_leases);

        try std.testing.expectEqual(@as(usize, 0x1234), codeOsrEnterBridge(&managed.native_state, 0x1234, 0, 10));
        try std.testing.expectEqual(CodeDispatchStatus.fallback_osr_metadata, managed.native_state.last_code_dispatch);
        try std.testing.expectEqual(@as(u64, 0), manager.stats().active_leases);
    }
    try std.testing.expect(try manager.invalidate(0));
    try std.testing.expectEqual(@as(u32, 1), try manager.reclaim());
    try std.testing.expectEqual(@as(u32, 1), owner.references.load(.acquire));
}

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
        error.MissingMonitorTable => .missing_monitor,
        error.IllegalMonitorState => .illegal_monitor_state,
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
        const slot: *const u64 = switch (location.kind) {
            .native_register => blk: {
                if (location.payload >= state.captured_gp.len) return error.InvalidLocation;
                break :blk &state.captured_gp[location.payload];
            },
            .stack_slot => blk: {
                if (state.captured_stack_base == 0) return error.InvalidLocation;
                const offset = location.stackOffset();
                const address = if (offset >= 0) add: {
                    const magnitude: usize = @intCast(offset);
                    if (magnitude > std.math.maxInt(usize) - state.captured_stack_base) return error.InvalidLocation;
                    break :add state.captured_stack_base + magnitude;
                } else sub: {
                    const magnitude: usize = @intCast(-@as(i64, offset));
                    if (magnitude > state.captured_stack_base) return error.InvalidLocation;
                    break :sub state.captured_stack_base - magnitude;
                };
                if (!std.mem.isAligned(address, @alignOf(Handle))) return error.InvalidLocation;
                break :blk @ptrFromInt(address);
            },
            else => return error.InvalidLocation,
        };
        try roots.add(@ptrCast(slot));
        if (slot.* == handle_bits) found_canonical = true;
    }
    if (!found_canonical) return error.InvalidLocation;
}

fn monitorFailure(state: *NativeThreadState, status: SlowResolveStatus) usize {
    state.last_error = status;
    _ = state.runtime.counters.monitor_failures.fetchAdd(1, .monotonic);
    return 0;
}

fn prepareMonitorSafepoint(
    state: *NativeThreadState,
    handle_bits: u64,
    site_key: u64,
    roots: *thread_registry.RootScope,
) Error!void {
    if (state.active == 0) return error.InactiveEntry;
    if (!state.context.isRunning()) return error.ThreadNotRunning;
    try addMappedRoots(state, handle_bits, site_key, roots);
    const participated = try state.runtime.registry.poll(state.context);
    if (participated) _ = state.runtime.counters.handshake_polls.fetchAdd(1, .monotonic);
    state.acknowledged_epoch = state.context.observedEpoch();
}

/// Safepointing monitor-enter target. The preserve-all adapter has captured
/// the exact machine-site roots before this function is reached. Acquisition
/// itself is allocation-free; only the contended path transitions the thread
/// to passive blocked state and waits.
pub fn monitorEnterBridge(state: *NativeThreadState, handle_bits: u64, site_key: u64) callconv(.c) usize {
    state.last_site_key = site_key;
    state.last_error = .ok;
    if (state.held_monitor_count >= max_native_monitors) return monitorFailure(state, .monitor_capacity);
    if (state.monitor_table == 0) return monitorFailure(state, .missing_monitor);
    if (state.collector == 0 or state.satb_buffer == 0) return monitorFailure(state, .missing_collector);

    var roots = state.context.beginRootScope() catch |err| return monitorFailure(state, statusFor(err));
    defer roots.deinit();
    prepareMonitorSafepoint(state, handle_bits, site_key, &roots) catch |err| {
        return monitorFailure(state, if (err == error.InvalidLocation) .missing_canonical_root else statusFor(err));
    };

    const monitors: *runtime_monitor.MonitorTable = @ptrFromInt(state.monitor_table);
    const collector: *runtime_gc.ConcurrentCollector = @ptrFromInt(state.collector);
    const satb: *runtime_gc.SatbBuffer = @ptrFromInt(state.satb_buffer);
    const handle: Handle = @bitCast(handle_bits);
    monitors.enter(handle, collector, state.runtime.registry, state.context, satb) catch |err| {
        return monitorFailure(state, statusFor(err));
    };
    state.held_monitors[state.held_monitor_count] = handle_bits;
    state.held_monitor_count += 1;
    _ = state.runtime.counters.monitor_enters.fetchAdd(1, .monotonic);
    return 1;
}

/// Safepointing monitor-exit target. Ownership is checked against both the
/// native frame's acquisition stack and the shared monitor table before the
/// stack entry is removed.
pub fn monitorExitBridge(state: *NativeThreadState, handle_bits: u64, site_key: u64) callconv(.c) usize {
    state.last_site_key = site_key;
    state.last_error = .ok;
    if (state.held_monitor_count > max_native_monitors) return monitorFailure(state, .runtime_error);
    var found: ?usize = null;
    var cursor: usize = state.held_monitor_count;
    while (cursor != 0) {
        cursor -= 1;
        if (state.held_monitors[cursor] == handle_bits) {
            found = cursor;
            break;
        }
    }
    const index = found orelse return monitorFailure(state, .illegal_monitor_state);
    if (state.monitor_table == 0) return monitorFailure(state, .missing_monitor);
    if (state.collector == 0 or state.satb_buffer == 0) return monitorFailure(state, .missing_collector);

    var roots = state.context.beginRootScope() catch |err| return monitorFailure(state, statusFor(err));
    defer roots.deinit();
    prepareMonitorSafepoint(state, handle_bits, site_key, &roots) catch |err| {
        return monitorFailure(state, if (err == error.InvalidLocation) .missing_canonical_root else statusFor(err));
    };

    const monitors: *runtime_monitor.MonitorTable = @ptrFromInt(state.monitor_table);
    const collector: *runtime_gc.ConcurrentCollector = @ptrFromInt(state.collector);
    const satb: *runtime_gc.SatbBuffer = @ptrFromInt(state.satb_buffer);
    monitors.exit(@bitCast(handle_bits), collector, satb) catch |err| {
        return monitorFailure(state, statusFor(err));
    };
    const count: usize = state.held_monitor_count;
    std.mem.copyForwards(u64, state.held_monitors[index .. count - 1], state.held_monitors[index + 1 .. count]);
    state.held_monitor_count -= 1;
    state.held_monitors[state.held_monitor_count] = @bitCast(Handle.none);
    _ = state.runtime.counters.monitor_exits.fetchAdd(1, .monotonic);
    return 1;
}

fn releaseHeldMonitors(state: *NativeThreadState) void {
    if (state.held_monitor_count > max_native_monitors or state.monitor_table == 0 or
        state.collector == 0 or state.satb_buffer == 0)
    {
        if (state.held_monitor_count != 0) _ = monitorFailure(state, .runtime_error);
        return;
    }
    const monitors: *runtime_monitor.MonitorTable = @ptrFromInt(state.monitor_table);
    const collector: *runtime_gc.ConcurrentCollector = @ptrFromInt(state.collector);
    const satb: *runtime_gc.SatbBuffer = @ptrFromInt(state.satb_buffer);
    while (state.held_monitor_count != 0) {
        const index = state.held_monitor_count - 1;
        const handle: Handle = @bitCast(state.held_monitors[index]);
        monitors.exit(handle, collector, satb) catch |err| {
            _ = monitorFailure(state, statusFor(err));
            return;
        };
        state.held_monitors[index] = @bitCast(Handle.none);
        state.held_monitor_count = index;
        _ = state.runtime.counters.monitor_unwind_exits.fetchAdd(1, .monotonic);
    }
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

fn exceptionFailure(state: *NativeThreadState, status: SlowResolveStatus) usize {
    state.last_error = status;
    _ = state.runtime.counters.exception_failures.fetchAdd(1, .monotonic);
    return 0;
}

/// Platform-ABI target for the mapped array-bounds cold edge. Generated code
/// preloads `pending_exception.index/length`, places the canonical array Handle
/// in r10, and places `(dex_pc << 32) | exception_site_id` in r11. The
/// preserve-all adapter captures every mapped root before this function polls.
/// Success returns a non-zero sentinel; the generated cold edge then returns
/// from the managed native frame with the lazy exception record installed.
pub fn boundsExceptionBridge(state: *NativeThreadState, handle_bits: u64, site_key: u64) callconv(.c) usize {
    state.last_site_key = site_key;
    state.last_error = .ok;
    if (state.active == 0) return exceptionFailure(state, .inactive_entry);
    if (!state.context.isRunning()) return exceptionFailure(state, .thread_not_running);
    if (state.pending_exception.kind != .none) return exceptionFailure(state, .runtime_error);

    const index = state.pending_exception.index;
    const length = state.pending_exception.length;
    if (!(index < 0 or @as(u32, @intCast(index)) >= length)) return exceptionFailure(state, .runtime_error);

    var roots = state.context.beginRootScope() catch |err| {
        return exceptionFailure(state, statusFor(err));
    };
    defer roots.deinit();
    addMappedRoots(state, handle_bits, site_key, &roots) catch |err| {
        return exceptionFailure(state, if (err == error.InvalidLocation) .missing_canonical_root else statusFor(err));
    };

    const participated = state.runtime.registry.poll(state.context) catch |err| {
        return exceptionFailure(state, statusFor(err));
    };
    if (participated) _ = state.runtime.counters.handshake_polls.fetchAdd(1, .monotonic);
    state.acknowledged_epoch = state.context.observedEpoch();
    state.pending_exception.dex_pc = @truncate(site_key >> 32);
    state.pending_exception.kind = .array_index_out_of_bounds;
    _ = state.runtime.counters.exception_transfers.fetchAdd(1, .monotonic);
    return 1;
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

    var spilled_handle = handle;
    const stack_locations = [_]runtime_stack_map.RootLocation{runtime_stack_map.RootLocation.stackSlot(0)};
    const stack_specs = [_]runtime_stack_map.MapSpec{.{ .pc_offset = 8, .roots = &stack_locations }};
    var stack_maps = try runtime_stack_map.Table.init(std.testing.allocator, &stack_specs, .{
        .native_register_count = 16,
        .interpreter_register_count = 0,
        .max_frame_depth = 0,
        .max_shadow_roots = 0,
    });
    defer stack_maps.deinit();
    try entry.installRootMaps(&stack_maps);
    entry.native_state.captured_stack_base = @intFromPtr(&spilled_handle);
    try std.testing.expectEqual(@intFromPtr(object), slowResolveBridge(&entry.native_state, handle_bits, 0x55_0000_0008));
    try std.testing.expectEqual(SlowResolveStatus.ok, entry.native_state.last_error);
    try std.testing.expectEqual(@as(usize, 0), context.rootCount());

    try std.testing.expectEqual(@as(u64, 2), runtime.stats().slow_resolves);
    try std.testing.expectEqual(@as(u64, 2), runtime.stats().resolve_failures);
}

test "slow resolver publishes a spilled root during concurrent handshake" {
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

    const locations = [_]runtime_stack_map.RootLocation{runtime_stack_map.RootLocation.stackSlot(0)};
    var maps = try runtime_stack_map.Table.init(std.testing.allocator, &.{.{
        .pc_offset = 1,
        .roots = &locations,
    }}, .{
        .native_register_count = 16,
        .interpreter_register_count = 0,
        .max_frame_depth = 0,
        .max_shadow_roots = 0,
    });
    defer maps.deinit();

    const Worker = struct {
        runtime: *Runtime,
        context: *ThreadContext,
        handle: Handle,
        maps: *const runtime_stack_map.Table,
        entered: *std.atomic.Value(bool),
        completed: *std.atomic.Value(bool),

        fn run(self: *@This()) void {
            var entry = self.runtime.enter(self.context) catch return;
            defer entry.deinit();
            entry.installRootMaps(self.maps) catch return;
            var spilled_handle = self.handle;
            entry.native_state.captured_stack_base = @intFromPtr(&spilled_handle);
            self.entered.store(true, .release);
            while (self.runtime.registry.requestEpoch() == entry.acknowledged_epoch) std.atomic.spinLoopHint();
            if (slowResolveBridge(&entry.native_state, @bitCast(self.handle), 0x20_0000_0001) == 0) return;
            self.completed.store(true, .release);
        }
    };

    var entered = std.atomic.Value(bool).init(false);
    var completed = std.atomic.Value(bool).init(false);
    var worker = Worker{
        .runtime = &runtime,
        .context = &context,
        .handle = handle,
        .maps = &maps,
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
